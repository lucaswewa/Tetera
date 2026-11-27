//! A known RGB test pattern, mosaicked as the MIChrome 20 sees it, then taken back to RGB on
//! each backend: the superpixel, bilinear and Malvar-He-Cutler. For each, the time it takes,
//! how close it comes to the pattern (PSNR), and whether the CPU and CUDA agree. The pattern
//! and both demosaicked images are written as 16-bit PPM files, cropped to their centre.
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
use tetera_imaging::bayer::{mosaic, superpixel_cpu};
use tetera_imaging::demosaic::demosaic_cpu;
use tetera_imaging::pattern::{psnr, test_pattern};
use tetera_imaging::{Demosaic, Image};

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

    std::fs::create_dir_all(&out_dir).map_err(|e| {
        tetera_core::Error::internal(format!("couldn't make {}: {e}", out_dir.display()))
    })?;
    let names: Vec<&str> = images.iter().map(|(name, _)| *name).collect();
    for (name, image) in &images {
        write_ppm(&out_dir.join(format!("{name}.ppm")), image, 960, 640, max)?;
    }
    println!(
        "\nWrote {}/{{{}}}.ppm: each image's centre, at most 960 × 640, in 16 bits",
        out_dir.display(),
        names.join(",")
    );
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
