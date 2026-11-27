//! The ISP's unfused chain on each backend, from a full 20 MP frame and from a 2 × 2 binned
//! one, to a 1920 × 1280 preview: each stage's time, and whether the CPU and CUDA agree. Then
//! the fused kernel (P1.06b), the whole chain in one pass on the GPU, against the chain.
//!
//! The frame is the test pattern through a synthetic sensor with an offset, vignetting and
//! unequal channels. Its master dark and flat field come from the same sensor, and white
//! balance evens the channels out, so the preview should look like the pattern. Each preview
//! is written as a JPEG.
//!
//! Run it with `cargo run --release -p tetera-imaging --example isp --features cuda` for both
//! backends, or without `--features cuda` for the CPU alone. `-- --smoke` takes a 640 × 480
//! frame to 320 × 240, once; `-- --out DIR` writes the JPEGs to DIR instead of
//! `target/isp`.

use std::path::PathBuf;
use std::time::Instant;
use tetera_compute::Compute;
use tetera_core::{Error, Result};
use tetera_imaging::bayer::{BinMode, bin_cpu};
use tetera_imaging::colour::colour_cpu;
use tetera_imaging::correct::{apply_flat_cpu, subtract_dark_cpu};
use tetera_imaging::demosaic::demosaic_cpu;
use tetera_imaging::pattern::{Sensor, test_pattern};
use tetera_imaging::resize::resize_cpu;
use tetera_imaging::{
    ColourCorrection, Demosaic, FlatField, Image, Isp, IspParams, JpegEncoder, Mosaic, Reduce,
};

const STAGES: [&str; 6] = ["Reduce", "Dark", "Flat", "Demosaic", "Resize", "Colour"];

/// The best of `passes` timings of `work`, in milliseconds, and its last result.
fn best<T>(passes: u32, mut work: impl FnMut() -> Result<T>) -> Result<(f64, T)> {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..passes {
        let start = Instant::now();
        last = Some(work()?);
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    Ok((best, last.expect("at least one pass")))
}

fn sensor() -> Sensor {
    Sensor {
        offset: 128,
        vignetting: 0.3,
        response: [0.7, 1.0, 0.5],
        ..Sensor::default()
    }
}

/// The ISP for `raw` reduced by `reduce`, its dark and flat field made by the sensor at the
/// reduced layout.
fn prepare(raw: &Mosaic, reduce: Reduce, output: (u32, u32)) -> Result<Isp> {
    let base = IspParams {
        reduce,
        colour: ColourCorrection {
            white_balance: [1.0 / 0.7, 1.0, 1.0 / 0.5],
            ..ColourCorrection::IDENTITY
        },
        ..IspParams::preview(raw.layout().max(), output.0, output.1)
    };
    let reduced = base.reduced(raw.layout())?;
    let cpu = tetera_compute::CpuBackend::new(0)?;
    let full = |w: u32, h: u32| -> Result<(Mosaic, Mosaic)> {
        let s = sensor();
        let even = Image::new(w, h, 3, [3_000u16; 3].repeat(w as usize * h as usize))?;
        Ok((s.dark(w, h)?, s.expose(&even)?))
    };
    // Darks and flats are taken in the mode the chain works at: binned frames, binned masters.
    let (dark, even) = match reduce {
        Reduce::Bin(factor) => {
            let (dark, even) = full(raw.layout().width, raw.layout().height)?;
            (
                bin_cpu(&cpu, &dark, factor, BinMode::Mean)?,
                bin_cpu(&cpu, &even, factor, BinMode::Mean)?,
            )
        }
        _ => full(reduced.width, reduced.height)?,
    };
    let flat = FlatField::new(&subtract_dark_cpu(&cpu, &even, &dark)?)?;
    Isp::new(
        IspParams {
            dark: Some(dark),
            flat: Some(flat),
            ..base
        },
        *raw.layout(),
    )
}

/// Each stage on the CPU, timed: the preview and the times.
fn stages_cpu(
    compute: &Compute,
    isp: &Isp,
    raw: &Mosaic,
    passes: u32,
) -> Result<(Image<u8>, [f64; 6])> {
    let cpu = compute.cpu();
    let params = isp.params();
    // A full frame has nothing to reduce: the stage shows as "-".
    let (t0, reduced) = match params.reduce {
        Reduce::Full => (f64::NAN, raw.clone()),
        _ => best(passes, || isp.reduce_cpu(cpu, raw))?,
    };
    let dark = params.dark.as_ref().expect("prepared with one");
    let (t1, darkened) = best(passes, || subtract_dark_cpu(cpu, &reduced, dark))?;
    let flat = params.flat.as_ref().expect("prepared with one");
    let (t2, flattened) = best(passes, || apply_flat_cpu(cpu, &darkened, flat))?;
    let (t3, rgb) = best(passes, || {
        demosaic_cpu::<u16>(cpu, &flattened, Demosaic::MalvarHeCutler)
    })?;
    let (t4, small) = best(passes, || resize_cpu(cpu, &rgb, isp.resize()))?;
    let (t5, shown) = best(passes, || colour_cpu(cpu, &small, isp.colour()))?;
    Ok((shown, [t0, t1, t2, t3, t4, t5]))
}

/// Each stage on the device, timed by events with its output's allocation; the first pass
/// loads the modules and isn't counted. The preview, downloaded, and the times.
#[cfg(feature = "cuda")]
fn stages_gpu(
    gpu: &tetera_compute::CudaBackend,
    isp: &Isp,
    raw: &Mosaic,
    passes: u32,
) -> Result<(Image<u8>, [f64; 6])> {
    use tetera_imaging::bayer::bin_device;
    use tetera_imaging::colour::{DeviceColour, colour_device};
    use tetera_imaging::correct::{apply_flat_device, subtract_dark_device};
    use tetera_imaging::demosaic::demosaic_device;
    use tetera_imaging::resize::{DeviceResize, resize_device};
    let params = isp.params();
    let input = match params.reduce {
        Reduce::Crop(roi) => raw.crop(roi)?,
        _ => raw.clone(),
    };
    let on_device = tetera_imaging::DeviceMosaic::upload(gpu, &input)?;
    let dark = gpu.upload(params.dark.as_ref().expect("prepared with one").sites())?;
    let gains = gpu.upload(params.flat.as_ref().expect("prepared with one").gains())?;
    let (tables, colour) = (
        DeviceResize::upload(gpu, isp.resize())?,
        DeviceColour::upload(gpu, isp.colour())?,
    );
    let timed = |work: &mut dyn FnMut() -> Result<()>| gpu.time_ms(work);
    let mut times = [f64::INFINITY; 6];
    let mut last = None;
    for pass in 0..=passes {
        let mut t = [f32::NAN; 6];
        let binned;
        let reduced = match params.reduce {
            Reduce::Bin(factor) => {
                let mut out = None;
                t[0] = timed(&mut || {
                    out = Some(bin_device(gpu, &on_device, factor, BinMode::Mean)?);
                    Ok(())
                })?;
                binned = out.expect("it ran");
                &binned
            }
            _ => &on_device,
        };
        let mut darkened = None;
        t[1] = timed(&mut || {
            darkened = Some(subtract_dark_device(gpu, reduced, &dark)?);
            Ok(())
        })?;
        let darkened = darkened.expect("it ran");
        let mut flattened = None;
        t[2] = timed(&mut || {
            flattened = Some(apply_flat_device(gpu, &darkened, &gains)?);
            Ok(())
        })?;
        let flattened = flattened.expect("it ran");
        let mut rgb = None;
        t[3] = timed(&mut || {
            rgb = Some(demosaic_device::<u16>(
                gpu,
                &flattened,
                Demosaic::MalvarHeCutler,
            )?);
            Ok(())
        })?;
        let rgb = rgb.expect("it ran");
        let mut small = None;
        t[4] = timed(&mut || {
            small = Some(resize_device(gpu, &rgb, &tables)?);
            Ok(())
        })?;
        let small = small.expect("it ran");
        let mut shown = None;
        t[5] = timed(&mut || {
            shown = Some(colour_device(gpu, &small, &colour)?);
            Ok(())
        })?;
        if pass > 0 {
            for (best, t) in times.iter_mut().zip(t) {
                *best = if t.is_nan() {
                    f64::NAN
                } else {
                    best.min(f64::from(t))
                };
            }
        }
        last = shown;
    }
    Ok((last.expect("it ran").download(gpu)?, times))
}

/// The fused kernel's time, the best of `passes` after one that loads its module, its output's
/// allocation included; checked against the chain's preview.
#[cfg(feature = "cuda")]
fn time_fused(
    gpu: &tetera_compute::CudaBackend,
    isp: &Isp,
    raw: &Mosaic,
    passes: u32,
    preview: &Image<u8>,
) -> Result<f64> {
    let on_device = tetera_imaging::DeviceMosaic::upload(gpu, raw)?;
    let prepared = tetera_imaging::isp::DeviceIsp::upload(gpu, isp)?;
    let mut best = f64::INFINITY;
    let mut last = None;
    for pass in 0..=passes {
        let mut out = None;
        let ms = gpu.time_ms(|| {
            out = Some(prepared.run_fused(gpu, isp, &on_device)?);
            Ok(())
        })?;
        if pass > 0 {
            best = best.min(f64::from(ms));
        }
        last = out;
    }
    let fused = last.expect("it ran").download(gpu)?;
    let d = tetera_imaging::compare_previews(&fused, preview)?;
    if !d.meets_tolerance() {
        return Err(Error::internal(format!(
            "the fused kernel misses its tolerance: {d:?}"
        )));
    }
    if d.max > 0 {
        println!("Fused:      within the tolerance: {d:?}");
    }
    Ok(best)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let smoke = args.iter().any(|arg| arg == "--smoke");
    let out_dir = args
        .iter()
        .position(|arg| arg == "--out")
        .and_then(|i| args.get(i + 1))
        .map_or_else(|| PathBuf::from("target/isp"), PathBuf::from);
    let (width, height, output, passes) = if smoke {
        (640, 480, (320, 240), 1)
    } else {
        (5_472, 3_648, (1_920, 1_280), 5)
    };

    let compute = Compute::detect()?;
    println!("Backend: {}: {}", compute.kind(), compute.why());
    let raw = sensor().expose(&test_pattern(width, height, 12))?;
    println!(
        "Frame:   {width} × {height} from a synthetic sensor, to {} × {}, best of {passes}",
        output.0, output.1
    );
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| Error::internal(format!("couldn't make {}: {e}", out_dir.display())))?;
    let mut encoder = JpegEncoder::new(85)?;

    for (name, reduce) in [("full", Reduce::Full), ("binned", Reduce::Bin(2))] {
        let isp = prepare(&raw, reduce, output)?;
        let reduced = isp.reduced();
        println!(
            "\n{name}: the chain works at {} × {}",
            reduced.width, reduced.height
        );
        println!("{:<12}{:>12}{:>12}", "", "CPU", "CUDA");
        let (preview, cpu) = stages_cpu(&compute, &isp, &raw, passes)?;
        #[allow(unused_mut)]
        let mut gpu_times: Option<[f64; 6]> = None;
        #[allow(unused_mut)]
        let mut fused: Option<f64> = None;
        #[cfg(feature = "cuda")]
        if let Some(gpu) = compute.cuda() {
            let (on_gpu, times) = stages_gpu(gpu, &isp, &raw, passes)?;
            if on_gpu != preview {
                return Err(Error::internal(format!(
                    "the {name} chain's twins disagree"
                )));
            }
            gpu_times = Some(times);
            fused = Some(time_fused(gpu, &isp, &raw, passes, &preview)?);
        }
        let show = |ms: f64| {
            if ms.is_nan() {
                String::from("-")
            } else {
                format!("{ms:.2} ms")
            }
        };
        let total = |times: &[f64; 6]| times.iter().filter(|t| !t.is_nan()).sum::<f64>();
        for (i, stage) in STAGES.iter().enumerate() {
            let on_gpu = gpu_times.map_or_else(|| String::from("-"), |t| show(t[i]));
            println!("{stage:<12}{:>12}{on_gpu:>12}", show(cpu[i]));
        }
        let on_gpu = gpu_times.map_or_else(|| String::from("-"), |t| show(total(&t)));
        println!("{:<12}{:>12}{on_gpu:>12}", "In all", show(total(&cpu)));
        if let Some(ms) = fused {
            println!("{:<12}{:>12}{:>12}", "Fused", "-", format!("{ms:.2} ms"));
        }
        if gpu_times.is_some() {
            println!("Twins:      the previews identical, fused and unfused");
        }
        let path = out_dir.join(format!("isp-{name}.jpg"));
        std::fs::write(&path, encoder.encode(&preview)?)
            .map_err(|e| Error::internal(format!("couldn't write {}: {e}", path.display())))?;
    }
    println!("\nWrote {}/isp-{{full,binned}}.jpg", out_dir.display());
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
