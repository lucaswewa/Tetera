//! Richardson–Lucy on the CPU (P1.07b): a synthetic specimen of beads and discs, blurred by a
//! widefield objective's PSF (NA 0.5, 550 nm, 0.12 µm a pixel), with noise. It prints the RMS
//! error against the specimen as the iterations go, plain and with total variation, where the
//! error is least, and each iteration's time; then the same for a 3D stack.
//!
//! Run it with `cargo run --release -p tetera-imaging --example deconvolve`. It writes the
//! specimen, the image and the best estimate to `target/deconvolve` as JPEGs; `-- --out DIR`
//! writes them elsewhere, and `-- --smoke` takes a small image and few iterations.

use std::path::{Path, PathBuf};
use std::time::Instant;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::{Error, Result};
use tetera_imaging::deconvolve::{Deconvolution, convolve_cpu};
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
