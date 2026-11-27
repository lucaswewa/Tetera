//! The H&E sample on the CPU (P1.11a): regions rendered at the sampling of the MIChrome 20's
//! 2.4 µm pixels through the 5×, 20× and 50× objectives, depth slices through a nucleus at
//! 50×, and a fold, written as PNGs to `target/he-sample/` with their timings.
//!
//! Run it with `cargo run --release -p tetera-sim --example he-sample [-- seed]`; `-- --smoke`
//! renders small regions. With `--features cuda` and a GPU, each view is rendered on the GPU
//! too, timed and compared with the CPU twin, and a grid of tiles is fetched through their
//! cache (P1.11b).

#[path = "common/png.rs"]
mod png;

use std::path::Path;
use std::time::Instant;
use tetera_core::{Error, Result};
use tetera_sim::{Depth, HeSample, Region, Rendered, SectionConfig, render};

const PIXEL: f64 = 2.4;

/// The GPU, when the `cuda` feature is on and one starts.
#[cfg(feature = "cuda")]
type Gpu = Option<tetera_compute::CudaBackend>;
#[cfg(not(feature = "cuda"))]
type Gpu = Option<()>;

fn gpu() -> Gpu {
    #[cfg(feature = "cuda")]
    {
        tetera_compute::CudaBackend::new(0).ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let smoke = args.iter().any(|a| a == "--smoke");
    let seed: u64 = args.iter().find_map(|a| a.parse().ok()).unwrap_or(7);
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/he-sample");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out)
        .map_err(|e| Error::internal(format!("{}: {e}", out.display())))?;
    let scale = if smoke { 8 } else { 1 };
    let sample = HeSample::new(seed, SectionConfig::default())?;
    let gpu = gpu();
    println!(
        "The H&E sample from seed {seed}, a section 4 µm thick, {}:",
        if gpu.is_some() {
            "on the CPU and the GPU"
        } else {
            "on the CPU"
        }
    );

    // The 5× field's middle, and a nucleus near it for 50×.
    let middle = (2_000.0, 2_000.0);
    let nucleus = sample
        .nuclei(
            middle.0 - 150.0,
            middle.1 - 150.0,
            middle.0 + 150.0,
            middle.1 + 150.0,
        )
        .into_iter()
        .find(|n| (1.0..3.0).contains(&n.depth))
        .ok_or_else(|| Error::not_found("no nucleus near the middle"))?;
    let views = [
        ("overview", PIXEL / 2.0, middle, 1_024, 1_024),
        ("5x", 5.0, middle, 1_536, 1_024),
        ("20x", 20.0, middle, 1_024, 768),
        ("50x", 50.0, nucleus.centre, 768, 768),
    ];
    for (name, magnification, centre, width, height) in views {
        let (width, height) = (width / scale, height / scale);
        let pixel = PIXEL / magnification;
        let region = centred(centre, width, height, pixel, Depth::Whole);
        let r = timed(&sample, &region, &format!("{name}, {pixel:.3} µm a pixel"))?;
        png(&out.join(format!("he-{name}.png")), &r, 1.0)?;
        on_gpu(
            &gpu,
            &sample,
            &region,
            &r,
            &out.join(format!("he-{name}-gpu.png")),
        )?;
    }

    // Depth slices through the nucleus at 50×, each 1 µm, shown as dark as a whole section.
    println!(
        "\nSlices through the nucleus at ({:.1}, {:.1}) µm, its centre {:.2} µm deep:",
        nucleus.centre.0, nucleus.centre.1, nucleus.depth
    );
    for from in [0.0f32, 1.5, 3.0] {
        let region = centred(
            nucleus.centre,
            512 / scale,
            512 / scale,
            PIXEL / 50.0,
            Depth::Range(from, from + 1.0),
        );
        let r = timed(
            &sample,
            &region,
            &format!("{from:.1} to {:.1} µm", from + 1.0),
        )?;
        let mid = &r.stains[(r.height / 2 * r.width + r.width / 2) as usize];
        println!(
            "      its middle pixel's haematoxylin: {:.3} OD",
            mid.haematoxylin
        );
        png(&out.join(format!("he-50x-depth-{from:.1}.png")), &r, 4.0)?;
    }

    // A fold, at 5×.
    let folded = HeSample::new(
        seed,
        SectionConfig {
            folds: true,
            ..SectionConfig::default()
        },
    )?;
    if let Some(x) = (0..20_000)
        .map(|i| f64::from(i) * 5.0)
        .find(|&x| folded.folded(x, middle.1) && folded.column(x, middle.1).stains.eosin > 0.0)
    {
        println!("\nA fold at x = {x:.0} µm:");
        let region = centred(
            (x, middle.1),
            768 / scale,
            512 / scale,
            PIXEL / 5.0,
            Depth::Whole,
        );
        let r = timed(&folded, &region, "5x, folds on")?;
        png(&out.join("he-5x-fold.png"), &r, 1.0)?;
    }
    tiles(&gpu, &sample)?;
    println!("\nWrote the PNGs to {}", out.display());
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

fn centred(centre: (f64, f64), width: u32, height: u32, pixel: f64, depth: Depth) -> Region {
    Region {
        origin: (
            centre.0 - f64::from(width) * pixel / 2.0,
            centre.1 - f64::from(height) * pixel / 2.0,
        ),
        width,
        height,
        pixel,
        depth,
    }
}

fn timed(sample: &HeSample, region: &Region, what: &str) -> Result<Rendered> {
    let start = Instant::now();
    let r = render(sample, region)?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let s = region.points();
    println!(
        "  {what}: {} × {} pixels of {}×{} points, {:.0} × {:.0} µm, in {ms:.0} ms ({:.2} µs a point)",
        region.width,
        region.height,
        s,
        s,
        f64::from(region.width) * region.pixel,
        f64::from(region.height) * region.pixel,
        ms * 1e3 / f64::from(region.width * region.height * s * s)
    );
    Ok(r)
}

/// Writes `r` as an 8-bit RGB PNG, its stains times `gain`.
fn png(path: &Path, r: &Rendered, gain: f32) -> Result<()> {
    let shown = Rendered {
        stains: r
            .stains
            .iter()
            .map(|s| tetera_sim::Stains {
                haematoxylin: s.haematoxylin * gain,
                eosin: s.eosin * gain,
            })
            .collect(),
        ..r.clone()
    };
    png::write(path, r.width, r.height, &shown.srgb8())
}

/// `region` on the GPU, if there is one: its time, how far it is from the CPU's, and its PNG.
#[cfg(feature = "cuda")]
fn on_gpu(
    gpu: &Gpu,
    sample: &HeSample,
    region: &Region,
    cpu: &Rendered,
    path: &Path,
) -> Result<()> {
    let Some(gpu) = gpu else { return Ok(()) };
    sample.render_on(gpu, region)?;
    let ms = gpu.time_ms(|| sample.render_on(gpu, region).map(drop))?;
    let r = sample.render_gpu(gpu, region)?;
    let worst = cpu
        .stains
        .iter()
        .zip(&r.stains)
        .map(|(a, b)| {
            (a.haematoxylin - b.haematoxylin)
                .abs()
                .max((a.eosin - b.eosin).abs())
        })
        .fold(0.0, f32::max);
    println!("    on the GPU in {ms:.2} ms, within {worst:.1e} OD of the CPU's");
    png(path, &r, 1.0)
}

#[cfg(not(feature = "cuda"))]
fn on_gpu(_: &Gpu, _: &HeSample, _: &Region, _: &Rendered, _: &Path) -> Result<()> {
    Ok(())
}

/// A field of 3 × 3 tiles at 5×, seen twice, then moved a column at a time, through a cache
/// that holds 12.
#[cfg(feature = "cuda")]
fn tiles(gpu: &Gpu, sample: &HeSample) -> Result<()> {
    use tetera_sim::gpu::{HeTiles, TILE_BYTES};
    use tetera_sim::{TileDepth, TileKey};
    let Some(gpu) = gpu else { return Ok(()) };
    let gpu = std::sync::Arc::new(gpu.clone());
    let budgets = tetera_compute::budget::VramBudgets::new(64 * TILE_BYTES);
    let tiles = HeTiles::new(
        gpu.clone(),
        *sample,
        budgets.reserve("he-sample", 12 * TILE_BYTES)?,
    )?;
    let level = TileKey::level_for(PIXEL / 5.0);
    println!("\nA field of 3 × 3 tiles at level {level}, through a cache of 12:");
    for (pass, shift) in [0, 0, 1, 2].into_iter().enumerate() {
        let start = Instant::now();
        for row in 30..33 {
            for column in 30 + shift..33 + shift {
                tiles.tile(TileKey {
                    level,
                    column,
                    row,
                    depth: TileDepth::Whole,
                })?;
            }
        }
        gpu.synchronize()?;
        let s = tiles.stats();
        println!(
            "  {}, moved {shift}: {} made, {} found, {} let go so far; {} held, {} of {} KiB, {:.1} ms",
            pass + 1,
            s.misses,
            s.hits,
            s.evictions,
            s.tiles,
            s.bytes / 1024,
            tiles.budget().limit() / 1024,
            start.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn tiles(_: &Gpu, _: &HeSample) -> Result<()> {
    Ok(())
}
