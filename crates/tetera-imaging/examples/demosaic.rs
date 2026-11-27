//! A known RGB test pattern, mosaicked as the MIChrome 20 sees it, then taken back to RGB on
//! each backend: the superpixel, bilinear and Malvar-He-Cutler. For each, the time it takes,
//! how close it comes to the pattern (PSNR), and whether the CPU and CUDA agree. The pattern
//! and both demosaicked images are written as 16-bit PPM files, cropped to their centre.
//!
//! Then the live view's chain (P1.05b): binned 2 × 2, demosaicked, resized to 1920 × 1280,
//! through the display transform to 8 bits, and encoded as a JPEG at quality 80, each step
//! timed on each backend. The JPEG is written as `preview.jpg`.
//!
//! Run it with `cargo run --release -p tetera-imaging --example demosaic --features cuda` for
//! both backends, or without `--features cuda` for the CPU alone. `-- --smoke` takes a
//! 640 × 480 frame once instead of a full frame five times; `-- --out DIR` writes the files
//! to DIR instead of `target/demosaic`.

use std::path::{Path, PathBuf};
use std::time::Instant;
use tetera_compute::Compute;
use tetera_core::Result;
use tetera_core::frame::format::BayerPattern;
use tetera_imaging::bayer::{bin_cpu, mosaic, superpixel_cpu};
use tetera_imaging::demosaic::demosaic_cpu;
use tetera_imaging::display::display_cpu;
use tetera_imaging::pattern::{psnr, test_pattern};
use tetera_imaging::resize::resize_cpu;
use tetera_imaging::{
    BinMode, Curve, Demosaic, DisplayTransform, Image, JpegEncoder, Levels, Mosaic, Resize,
};

const BITS: u8 = 12;

/// The best of `passes` timings of `work`, in milliseconds.
fn best(passes: u32, mut work: impl FnMut() -> Result<()>) -> Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..passes {
        let start = Instant::now();
        work()?;
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    Ok(best)
}

/// `image`'s centre, at most `width × height`, as a 16-bit PPM of values up to `max`.
fn write_ppm(path: &Path, image: &Image<u16>, width: u32, height: u32, max: u16) -> Result<()> {
    let (w, h) = (width.min(image.width()), height.min(image.height()));
    let (x0, y0) = ((image.width() - w) / 2, (image.height() - h) / 2);
    let mut bytes = format!("P6\n{w} {h}\n{max}\n").into_bytes();
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            for &value in image.pixel(x, y) {
                bytes.extend_from_slice(&value.to_be_bytes());
            }
        }
    }
    std::fs::write(path, bytes).map_err(|e| {
        tetera_core::Error::internal(format!("couldn't write {}: {e}", path.display()))
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let smoke = args.iter().any(|arg| arg == "--smoke");
    let out_dir = args
        .iter()
        .position(|arg| arg == "--out")
        .and_then(|i| args.get(i + 1))
        .map_or_else(|| PathBuf::from("target/demosaic"), PathBuf::from);
    let (width, height, passes) = if smoke {
        (640, 480, 1)
    } else {
        (5_472, 3_648, 5)
    };

    let compute = Compute::detect()?;
    println!("Backend: {}: {}", compute.kind(), compute.why());
    println!("Frame:   {width} × {height}, {BITS} bits, RGGB, best of {passes}\n");
    let rgb = test_pattern(width, height, BITS);
    let raw = mosaic(&rgb, BayerPattern::Rggb, BITS)?;
    let max = raw.layout().max();

    #[cfg(feature = "cuda")]
    let on_device = match compute.cuda() {
        Some(gpu) => Some((gpu, tetera_imaging::DeviceMosaic::upload(gpu, &raw)?)),
        None => None,
    };

    println!("{:<18}{:>12}{:>12}{:>11}", "", "CPU", "CUDA", "PSNR");
    let mut superpixel = None;
    let cpu_ms = best(passes, || {
        superpixel = Some(superpixel_cpu(compute.cpu(), &raw));
        Ok(())
    })?;
    #[allow(unused_mut)]
    let mut cuda = String::from("-");
    #[cfg(feature = "cuda")]
    if let Some((gpu, mosaic)) = &on_device {
        let on_gpu = tetera_imaging::bayer::superpixel_device(gpu, mosaic)?; // loads the module
        let mut out = on_gpu;
        let mut kernel = f32::INFINITY;
        for _ in 0..passes {
            kernel = kernel.min(gpu.time_ms(|| {
                gpu.bayer_superpixel(mosaic.buffer(), mosaic.layout(), out.buffer_mut())
            })?);
        }
        if out.download(gpu)? != superpixel.clone().expect("it ran") {
            return Err(tetera_core::Error::internal(
                "the superpixel's twins disagree",
            ));
        }
        cuda = format!("{kernel:.2} ms");
    }
    println!("{:<18}{:>9.2} ms{cuda:>12}", "Superpixel", cpu_ms);

    let mut images = vec![("pattern", rgb.clone())];
    for (method, name) in [
        (Demosaic::Bilinear, "bilinear"),
        (Demosaic::MalvarHeCutler, "malvar-he-cutler"),
    ] {
        let mut on_cpu = None;
        let cpu_ms = best(passes, || {
            on_cpu = Some(demosaic_cpu::<u16>(compute.cpu(), &raw, method)?);
            Ok(())
        })?;
        let on_cpu = on_cpu.expect("it ran");
        let db = psnr(&rgb, &on_cpu, max)?;
        #[allow(unused_mut)]
        let (mut cuda, mut twins) = (String::from("-"), String::new());
        #[cfg(feature = "cuda")]
        if let Some((gpu, mosaic)) = &on_device {
            use tetera_compute::twin::{Tolerance, compare};
            let mut out = tetera_imaging::demosaic::demosaic_device::<u16>(gpu, mosaic, method)?; // loads the module
            let mut kernel = f32::INFINITY;
            for _ in 0..passes {
                kernel = kernel.min(gpu.time_ms(|| {
                    gpu.demosaic(method, mosaic.buffer(), mosaic.layout(), out.buffer_mut())
                })?);
            }
            let comparison = compare(on_cpu.data(), out.download(gpu)?.data(), Tolerance::EXACT)?;
            if !comparison.passes() {
                return Err(tetera_core::Error::internal(format!(
                    "the {name} twins disagree: {comparison}"
                )));
            }
            cuda = format!("{kernel:.2} ms");
            twins = String::from("   twins identical");
        }
        println!(
            "{:<18}{cpu_ms:>9.2} ms{cuda:>12}{db:>8.2} dB{twins}",
            format!("{method:?}")
        );
        images.push((name, on_cpu));
    }

    let jpeg = preview(&compute, &raw, passes, smoke)?;

    std::fs::create_dir_all(&out_dir).map_err(|e| {
        tetera_core::Error::internal(format!("couldn't make {}: {e}", out_dir.display()))
    })?;
    let path = out_dir.join("preview.jpg");
    std::fs::write(&path, &jpeg).map_err(|e| {
        tetera_core::Error::internal(format!("couldn't write {}: {e}", path.display()))
    })?;
    let names: Vec<&str> = images.iter().map(|(name, _)| *name).collect();
    for (name, image) in &images {
        write_ppm(&out_dir.join(format!("{name}.ppm")), image, 960, 640, max)?;
    }
    println!(
        "\nWrote {}/{{{}}}.ppm: each image's centre, at most 960 × 640, in 16 bits; and preview.jpg",
        out_dir.display(),
        names.join(",")
    );
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// The live view's chain on `raw`, each step timed on each backend; the JPEG it makes.
fn preview(compute: &Compute, raw: &Mosaic, passes: u32, smoke: bool) -> Result<Vec<u8>> {
    let (width, height) = if smoke { (320, 240) } else { (1_920, 1_280) };
    let level = Levels {
        black: 0,
        white: raw.layout().max(),
    };
    let transform = DisplayTransform::uniform(3, level, Curve::Srgb)?;
    println!(
        "\nPreview: binned 2 × 2, Malvar-He-Cutler, resized to {width} × {height}, sRGB, JPEG at 80"
    );
    println!("{:<18}{:>12}{:>12}", "", "CPU", "CUDA");

    let mut binned = None;
    let bin_ms = best(passes, || {
        binned = Some(bin_cpu(compute.cpu(), raw, 2, BinMode::Mean)?);
        Ok(())
    })?;
    let binned = binned.expect("it ran");
    let mut rgb = None;
    let demosaic_ms = best(passes, || {
        rgb = Some(demosaic_cpu::<u16>(
            compute.cpu(),
            &binned,
            Demosaic::MalvarHeCutler,
        )?);
        Ok(())
    })?;
    let rgb = rgb.expect("it ran");
    let plan = Resize::new(rgb.width(), rgb.height(), width, height)?;
    let mut small = None;
    let resize_ms = best(passes, || {
        small = Some(resize_cpu(compute.cpu(), &rgb, &plan)?);
        Ok(())
    })?;
    let small = small.expect("it ran");
    let mut shown = None;
    let display_ms = best(passes, || {
        shown = Some(display_cpu(compute.cpu(), &small, &transform)?);
        Ok(())
    })?;
    let shown = shown.expect("it ran");
    let mut encoder = JpegEncoder::new(80)?;
    let mut jpeg = Vec::new();
    let jpeg_ms = best(passes, || {
        jpeg = encoder.encode(&shown)?;
        Ok(())
    })?;
    let cpu = [bin_ms, demosaic_ms, resize_ms, display_ms];

    #[allow(unused_mut)]
    let mut cuda: Option<[f32; 4]> = None;
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        use tetera_imaging::bayer::bin_device;
        use tetera_imaging::demosaic::demosaic_device;
        use tetera_imaging::display::{DeviceDisplay, display_device};
        use tetera_imaging::resize::{DeviceResize, resize_device};
        let mosaic = tetera_imaging::DeviceMosaic::upload(gpu, raw)?;
        let (tables, lookup) = (
            DeviceResize::upload(gpu, &plan)?,
            DeviceDisplay::upload(gpu, &transform)?,
        );
        // Each step's time on the device, its output's allocation included. The first pass
        // loads the modules, and isn't counted.
        let mut times = [f32::INFINITY; 4];
        let mut last = None;
        for pass in 0..=passes {
            let mut binned = None;
            let t0 = gpu.time_ms(|| {
                binned = Some(bin_device(gpu, &mosaic, 2, BinMode::Mean)?);
                Ok(())
            })?;
            let binned = binned.expect("it ran");
            let mut rgb = None;
            let t1 = gpu.time_ms(|| {
                rgb = Some(demosaic_device::<u16>(
                    gpu,
                    &binned,
                    Demosaic::MalvarHeCutler,
                )?);
                Ok(())
            })?;
            let rgb = rgb.expect("it ran");
            let mut small = None;
            let t2 = gpu.time_ms(|| {
                small = Some(resize_device(gpu, &rgb, &tables)?);
                Ok(())
            })?;
            let small = small.expect("it ran");
            let mut shown = None;
            let t3 = gpu.time_ms(|| {
                shown = Some(display_device(gpu, &small, &lookup)?);
                Ok(())
            })?;
            if pass > 0 {
                for (best, t) in times.iter_mut().zip([t0, t1, t2, t3]) {
                    *best = best.min(t);
                }
            }
            last = shown;
        }
        if last.expect("it ran").download(gpu)? != shown {
            return Err(tetera_core::Error::internal("the preview's twins disagree"));
        }
        cuda = Some(times);
    }

    let names = ["Bin 2 × 2", "Demosaic", "Resize", "Display"];
    for (i, name) in names.iter().enumerate() {
        let on_gpu = cuda.map_or_else(|| String::from("-"), |t| format!("{:.2} ms", t[i]));
        println!("{name:<18}{:>9.2} ms{on_gpu:>12}", cpu[i]);
    }
    println!(
        "{:<18}{jpeg_ms:>9.2} ms{:>12}   {} KiB",
        "JPEG",
        "-",
        jpeg.len() / 1_024
    );
    let total_cpu: f64 = cpu.iter().sum::<f64>() + jpeg_ms;
    let total_gpu = cuda.map_or_else(
        || String::from("-"),
        |t| format!("{:.2} ms", f64::from(t.iter().sum::<f32>()) + jpeg_ms),
    );
    println!(
        "{:<18}{total_cpu:>9.2} ms{total_gpu:>12}   the CUDA column's JPEG on the CPU",
        "In all"
    );
    if cuda.is_some() {
        println!("Twins:   the 8-bit previews identical");
    }
    Ok(jpeg)
}
