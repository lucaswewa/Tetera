//! Richardson–Lucy (P1.07b, P1.07c): a synthetic specimen of beads and discs, blurred by a
//! widefield objective's PSF (NA 0.5, 550 nm, 0.12 µm a pixel), with noise. It prints the RMS
//! error against the specimen as the iterations go, plain and with total variation, where the
//! error is least, and each iteration's time on the CPU; then the same for a 3D stack. With
//! CUDA, it runs each on the GPU too, checks the two agree within the behaviour spec's
//! tolerance, and times 20 iterations on a full 20 MP frame.
//!
//! Run it with `cargo run --release -p tetera-imaging --example deconvolve --features cuda`,
//! or without the feature for the CPU alone. It writes the specimen, the image and the best
//! estimate to `target/deconvolve` as JPEGs; `-- --out DIR` writes them elsewhere, and
//! `-- --smoke` takes small images and few iterations.

use std::path::{Path, PathBuf};
use std::time::Instant;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::{Error, Result};
use tetera_imaging::deconvolve::{Deconvolution, convolve_cpu};
#[cfg(feature = "cuda")]
use tetera_imaging::deconvolve::{DeviceDeconvolution, RichardsonLucy, richardson_lucy_cpu};
use tetera_imaging::pattern::noisy;
use tetera_imaging::{Image, Optics, Psf, Volume};

const OPTICS: Optics = Optics {
    na: 0.5,
    wavelength_um: 0.55,
    pixel_um: 0.12,
    refractive_index: 1.0,
};
const SIGMA: f64 = 15.0;

/// A specimen `w × h × d`: beads of one pixel, and discs (balls in 3D), on a dim background.
fn specimen(w: u32, h: u32, d: u32) -> Volume {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move |n: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % u64::from(n)) as u32
    };
    let mut values = vec![60.0f32; (w * h * d) as usize];
    let at = |x: u32, y: u32, z: u32| ((z * h + y) * w + x) as usize;
    for _ in 0..(w * h / 2_000).max(4) {
        let (x, y, z) = (next(w), next(h), next(d));
        values[at(x, y, z)] = 1_500.0 + next(2_000) as f32;
    }
    for _ in 0..(w * h / 20_000).max(2) {
        let (cx, cy, cz, r) = (next(w), next(h), next(d), 3 + next(10));
        let level = 400.0 + next(800) as f32;
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let squared = [(x, cx), (y, cy), (z, cz)].map(|(a, b)| a.abs_diff(b).pow(2));
                    if squared.iter().sum::<u32>() <= r * r {
                        values[at(x, y, z)] = level;
                    }
                }
            }
        }
    }
    Volume::new(w, h, d, values).expect("a specimen is finite")
}

/// `volume` blurred by `psf`, with noise of σ 15, in 12 bits.
fn image_of(cpu: &CpuBackend, volume: &Volume, psf: &Psf) -> Result<Volume> {
    let blurred = convolve_cpu(cpu, volume, psf)?;
    let rounded: Vec<u16> = blurred
        .values()
        .iter()
        .map(|v| v.round().min(4_095.0) as u16)
        .collect();
    let (w, h, d) = volume.size();
    Volume::new(
        w,
        h,
        d,
        noisy(&rounded, SIGMA, 4_095, 1)
            .iter()
            .map(|&v| f32::from(v))
            .collect(),
    )
}

fn rms(a: &Volume, b: &Volume) -> f64 {
    let squares: f64 = a
        .values()
        .iter()
        .zip(b.values())
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum();
    (squares / a.values().len() as f64).sqrt()
}

/// Runs `iterations` of Richardson–Lucy, printing the error at `report`'s iterations; returns
/// the least error, its iteration and estimate, and the mean time an iteration.
fn run(
    cpu: &CpuBackend,
    image: &Volume,
    truth: &Volume,
    psf: &Psf,
    lambda: f32,
    iterations: u32,
    report: &[u32],
) -> Result<(f64, u32, Volume, f64)> {
    let mut rl = Deconvolution::new(cpu, image, psf, lambda)?;
    let mut best = (rms(image, truth), 0, image.clone());
    let mut seconds = 0.0;
    let mut row = String::new();
    for _ in 0..iterations {
        let start = Instant::now();
        rl.step();
        seconds += start.elapsed().as_secs_f64();
        let estimate = rl.estimate();
        let error = rms(&estimate, truth);
        if error < best.0 {
            best = (error, rl.iterations(), estimate);
        }
        if report.contains(&rl.iterations()) {
            row += &format!("{error:>8.1}");
        }
    }
    let label = if lambda > 0.0 {
        format!("TV, λ {lambda}")
    } else {
        "plain".to_string()
    };
    if !report.is_empty() {
        println!("{label:<14}{row}");
    }
    Ok((
        best.0,
        best.1,
        best.2,
        seconds * 1e3 / f64::from(iterations),
    ))
}

/// Each case deconvolved for 20 iterations on both backends, timed, the two checked to agree
/// within the behaviour spec's tolerance; then 20 iterations on a frame of `full` on the GPU.
#[cfg(feature = "cuda")]
fn on_gpu(
    gpu: &tetera_compute::CudaBackend,
    cpu: &CpuBackend,
    cases: &[(&str, &Volume, &Psf)],
    full: (u32, u32),
) -> Result<()> {
    const ITERATIONS: u32 = 20;
    println!("\nCUDA, {ITERATIONS} iterations, against the CPU:");
    println!(
        "{:<8}{:>16}{:>16}{:>16}{:>14}",
        "", "CPU, an iter", "CUDA, an iter", "relative RMS", "worst"
    );
    for &(label, image, psf) in cases {
        let start = Instant::now();
        let on_cpu = richardson_lucy_cpu(cpu, image, psf, &RichardsonLucy::new(ITERATIONS))?;
        let cpu_ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(ITERATIONS);
        DeviceDeconvolution::new(gpu, image, psf, 0.0)?.step()?; // loads the module
        let mut run = DeviceDeconvolution::new(gpu, image, psf, 0.0)?;
        let ms = gpu.time_ms(|| {
            for _ in 0..ITERATIONS {
                run.step()?;
            }
            Ok(())
        })?;
        let on_gpu = run.estimate()?;
        let (mut squares, mut norm, mut worst) = (0.0, 0.0, 0.0f64);
        for (&a, &b) in on_gpu.values().iter().zip(on_cpu.values()) {
            let d = f64::from(a) - f64::from(b);
            (squares, norm, worst) = (
                squares + d * d,
                norm + f64::from(b).powi(2),
                worst.max(d.abs()),
            );
        }
        let relative = (squares / norm).sqrt();
        let largest = image.values().iter().fold(0.0f32, |a, &v| a.max(v));
        if relative >= 1e-3 || worst >= 0.01 * f64::from(largest) {
            return Err(Error::internal(format!(
                "the backends disagree beyond the spec's tolerance: relative RMS {relative:e}, worst {worst}"
            )));
        }
        println!(
            "{label:<8}{cpu_ms:>13.1} ms{:>13.2} ms{relative:>16.1e}{worst:>14.4}",
            ms / ITERATIONS as f32
        );
    }

    let (w, h) = full;
    let psf = Psf::widefield_2d(&OPTICS)?;
    let image = image_of(cpu, &specimen(w, h, 1), &psf)?;
    let (pw, ph, _) = psf.size();
    let padding = tetera_compute::deconvolve::Padding::new(
        [w as usize, h as usize, 1],
        [pw as usize, ph as usize, 1],
    )?;
    let start = Instant::now();
    let mut run = DeviceDeconvolution::new(gpu, &image, &psf, 0.0)?;
    gpu.synchronize()?;
    let setup = start.elapsed().as_secs_f64() * 1e3;
    let ms = gpu.time_ms(|| {
        for _ in 0..ITERATIONS {
            run.step()?;
        }
        Ok(())
    })?;
    println!(
        "\n{w} × {h}, padded to {} × {}: {ITERATIONS} iterations in {ms:.0} ms on CUDA, {:.1} ms each, after {setup:.0} ms to plan, upload and transform the PSF",
        padding.padded[0],
        padding.padded[1],
        ms / ITERATIONS as f32
    );
    Ok(())
}

/// A volume of depth 1 as a grey JPEG, its values up to 4,095 scaled to 8 bits.
fn write_jpeg(path: &Path, volume: &Volume) -> Result<()> {
    let (w, h, _) = volume.size();
    let grey = volume
        .values()
        .iter()
        .map(|v| (v / 4_095.0 * 255.0).round().clamp(0.0, 255.0) as u8);
    let jpeg = tetera_imaging::jpeg::encode(&Image::new(w, h, 1, grey.collect())?, 90)?;
    std::fs::write(path, jpeg)
        .map_err(|e| Error::internal(format!("couldn't write {}: {e}", path.display())))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let smoke = args.iter().any(|arg| arg == "--smoke");
    let out_dir = args
        .iter()
        .position(|arg| arg == "--out")
        .and_then(|i| args.get(i + 1))
        .map_or_else(|| PathBuf::from("target/deconvolve"), PathBuf::from);
    let (side, iterations, (side_3d, depth_3d)) = if smoke {
        (128, 10, (48, 16))
    } else {
        (1_024, 60, (192, 48))
    };
    let report: Vec<u32> = [1, 2, 5, 10, 20, 40, 60]
        .into_iter()
        .filter(|&i| i <= iterations)
        .collect();
    let compute = Compute::detect()?;
    let cpu = compute.cpu();
    println!("CPU:   {} threads", cpu.threads());

    let psf = Psf::widefield_2d(&OPTICS)?;
    let truth = specimen(side, side, 1);
    let image = image_of(cpu, &truth, &psf)?;
    println!(
        "Image: {side} × {side}, a {} × {} PSF, noise σ {SIGMA}; RMS error {:.1} against the specimen\n",
        psf.size().0,
        psf.size().1,
        rms(&image, &truth)
    );
    println!(
        "{:<14}{}",
        "Iteration",
        report.iter().map(|i| format!("{i:>8}")).collect::<String>()
    );
    let plain = run(cpu, &image, &truth, &psf, 0.0, iterations, &report)?;
    let tv = run(cpu, &image, &truth, &psf, 0.002, iterations, &report)?;
    println!();
    for (label, (error, at, _, ms)) in [("Plain", &plain), ("With TV", &tv)] {
        println!("{label:<9} least error {error:.1}, at iteration {at}; {ms:.1} ms an iteration");
    }

    let psf_3d = Psf::widefield_3d(&OPTICS, 0.3, 15)?;
    let truth_3d = specimen(side_3d, side_3d, depth_3d);
    let image_3d = image_of(cpu, &truth_3d, &psf_3d)?;
    let (w, h, d) = psf_3d.size();
    println!(
        "\n3D:    {side_3d} × {side_3d} × {depth_3d}, a {w} × {h} × {d} PSF 0.3 µm a plane; RMS error {:.1}",
        rms(&image_3d, &truth_3d)
    );
    let (error, at, _, ms) = run(
        cpu,
        &image_3d,
        &truth_3d,
        &psf_3d,
        0.0,
        iterations.min(20),
        &[],
    )?;
    println!("3D        least error {error:.1}, at iteration {at}; {ms:.1} ms an iteration");

    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let full = if smoke { (640, 480) } else { (5_472, 3_648) };
        on_gpu(
            gpu,
            cpu,
            &[("2D", &image, &psf), ("3D", &image_3d, &psf_3d)],
            full,
        )?;
    }
    #[cfg(feature = "cuda")]
    if compute.cuda().is_none() {
        println!("\nCUDA: none here ({})", compute.why());
    }
    #[cfg(not(feature = "cuda"))]
    println!("\nCUDA: not in this build");

    std::fs::create_dir_all(&out_dir)
        .map_err(|e| Error::internal(format!("couldn't make {}: {e}", out_dir.display())))?;
    write_jpeg(&out_dir.join("specimen.jpg"), &truth)?;
    write_jpeg(&out_dir.join("image.jpg"), &image)?;
    write_jpeg(&out_dir.join("deconvolved.jpg"), &plain.2)?;
    println!(
        "\nWrote the specimen, the image and the plain best estimate to {}",
        out_dir.display()
    );
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
