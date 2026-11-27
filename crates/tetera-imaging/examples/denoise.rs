//! Denoising a noisy synthetic image (P1.07a), on each backend, with the PSNR before and after
//! and each method's time:
//! - **temporal averaging** of 8 raw frames of one still scene, each with its own noise;
//! - **the bilateral filter** on the demosaicked RGB of one noisy frame.
//!
//! Run it with `cargo run --release -p tetera-imaging --example denoise --features cuda` for
//! both backends, or without `--features cuda` for the CPU alone. `-- --smoke` takes a
//! 320 × 240 frame once instead of a full frame three times.

use std::time::Instant;
use tetera_compute::Compute;
use tetera_core::Result;
use tetera_core::frame::format::BayerPattern;
use tetera_imaging::bayer::mosaic;
use tetera_imaging::demosaic::demosaic_cpu;
use tetera_imaging::denoise::bilateral_cpu;
use tetera_imaging::pattern::{noisy, psnr, test_pattern};
use tetera_imaging::{Bilateral, Demosaic, Image, Mosaic, TemporalMean};

const SIGMA: f64 = 80.0;
const FRAMES: u32 = 8;

/// A mosaic's sites as a one-channel image, for PSNR.
fn sites(m: &Mosaic) -> Result<Image<u16>> {
    Image::new(m.layout().width, m.layout().height, 1, m.sites().to_vec())
}

fn main() -> Result<()> {
    let smoke = std::env::args().any(|arg| arg == "--smoke");
    let (width, height, passes) = if smoke {
        (320, 240, 1)
    } else {
        (5_472, 3_648, 3)
    };
    let compute = Compute::detect()?;
    let cpu = compute.cpu();
    println!("Backend: {}: {}", compute.kind(), compute.why());
    println!("Frame:   {width} × {height}, 12 bits, noise σ {SIGMA}, best of {passes}\n");

    let clean = mosaic(&test_pattern(width, height, 12), BayerPattern::Rggb, 12)?;
    let frames: Vec<Mosaic> = (0..u64::from(FRAMES))
        .map(|seed| Mosaic::new(*clean.layout(), noisy(clean.sites(), SIGMA, 4_095, seed)))
        .collect::<Result<_>>()?;
    let reference = sites(&clean)?;
    println!(
        "{:<26}{:>11}{:>11}{:>12}{:>12}",
        "", "before", "after", "CPU", "CUDA"
    );

    // Temporal averaging: the window's mean after its 8th frame.
    let mut best = f64::INFINITY;
    let mut averaged = None;
    for _ in 0..passes {
        let mut window = TemporalMean::new(FRAMES)?;
        let start = Instant::now();
        for frame in &frames {
            averaged = Some(window.push_cpu(cpu, frame, false)?);
        }
        best = best.min(start.elapsed().as_secs_f64() * 1e3 / f64::from(FRAMES));
    }
    let averaged = averaged.expect("it ran");
    #[allow(unused_mut)]
    let mut cuda = String::from("-");
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let on_device: Vec<_> = frames
            .iter()
            .map(|f| tetera_imaging::DeviceMosaic::upload(gpu, f))
            .collect::<Result<_>>()?;
        let mut window = tetera_imaging::denoise::DeviceTemporalMean::new(FRAMES)?;
        window.push(gpu, &on_device[0], false)?; // loads the module
        let mut fastest = f32::INFINITY;
        let mut last = None;
        for _ in 0..passes {
            window = tetera_imaging::denoise::DeviceTemporalMean::new(FRAMES)?;
            let ms = gpu.time_ms(|| {
                for frame in &on_device {
                    last = Some(window.push(gpu, frame, false)?);
                }
                Ok(())
            })?;
            fastest = fastest.min(ms / FRAMES as f32);
        }
        if last.expect("it ran").download(gpu)? != averaged {
            return Err(tetera_core::Error::internal(
                "the temporal mean's twins disagree",
            ));
        }
        cuda = format!("{fastest:.2} ms");
    }
    println!(
        "{:<26}{:>8.2} dB{:>8.2} dB{:>9.2} ms{cuda:>12}",
        format!("Temporal mean of {FRAMES}"),
        psnr(&reference, &sites(&frames[0])?, 4_095)?,
        psnr(&reference, &sites(&averaged)?, 4_095)?,
        best
    );

    // The bilateral filter on one noisy frame, demosaicked.
    let clean_rgb = demosaic_cpu::<u16>(cpu, &clean, Demosaic::MalvarHeCutler)?;
    let noisy_rgb = demosaic_cpu::<u16>(cpu, &frames[0], Demosaic::MalvarHeCutler)?;
    let filter = Bilateral::new(1.0, 2.0 * SIGMA)?;
    let mut best = f64::INFINITY;
    let mut filtered = None;
    for _ in 0..passes {
        let start = Instant::now();
        filtered = Some(bilateral_cpu(cpu, &noisy_rgb, &filter)?);
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    let filtered = filtered.expect("it ran");
    #[allow(unused_mut)]
    let mut cuda = String::from("-");
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        use tetera_imaging::denoise::{DeviceBilateral, bilateral_device};
        let on_device = tetera_imaging::DeviceImage::upload(gpu, &noisy_rgb)?;
        let tables = DeviceBilateral::upload(gpu, &filter)?;
        bilateral_device(gpu, &on_device, &tables)?; // loads the module
        let mut fastest = f32::INFINITY;
        let mut last = None;
        for _ in 0..passes {
            fastest = fastest.min(gpu.time_ms(|| {
                last = Some(bilateral_device(gpu, &on_device, &tables)?);
                Ok(())
            })?);
        }
        if last.expect("it ran").download(gpu)? != filtered {
            return Err(tetera_core::Error::internal(
                "the bilateral filter's twins disagree",
            ));
        }
        cuda = format!("{fastest:.2} ms");
    }
    println!(
        "{:<26}{:>8.2} dB{:>8.2} dB{:>9.2} ms{cuda:>12}",
        format!("Bilateral, σ_s 1, σ_r {}", 2.0 * SIGMA),
        psnr(&clean_rgb, &noisy_rgb, 4_095)?,
        psnr(&clean_rgb, &filtered, 4_095)?,
        best
    );
    println!(
        "\nThe temporal mean's times are per frame pushed. On CUDA, each result is checked to equal the CPU's."
    );
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
