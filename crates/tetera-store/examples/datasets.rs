//! The dataset store (P1.15b, ADR-0021): a project, a dataset of synthetic raw frames written
//! as OME-Zarr and read back, its tree on disk, and a second dataset abandoned half way,
//! which leaves nothing behind.
//!
//! Run it with `cargo run --release -p tetera-store --example datasets`; `-- --smoke` writes
//! small frames.

use serde_json::json;
use std::path::Path;
use std::time::{Duration, Instant};
use tetera_core::frame::format::BayerPattern;
use tetera_core::frame::pool::HostAllocator;
use tetera_core::{Error, FrameLayout, FramePool, PixelFormat, Result};
use tetera_store::{DatasetSpec, Samples, Store};

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let (width, height, n) = if smoke {
        (256, 192, 5)
    } else {
        (5_472, 3_648, 20)
    };
    let root = std::env::temp_dir().join(format!("tetera-datasets-{}", std::process::id()));
    let store = Store::open(&root)?;
    store.create_project("Example")?;
    let format = PixelFormat::bayer16(BayerPattern::Rggb, 12)?;
    let spec = DatasetSpec {
        name: "a burst".into(),
        kind: "capture".into(),
        format,
        width,
        height,
        frames: n,
        pixel_size_um: 0.12,
        interval_s: 0.138,
        instrument: json!({"camera": "virtual MIChrome 20", "objective": "20x", "exposure_ms": 5.0}),
    };
    // Frames like a camera's: a smooth field with some noise, 12 bits in 16.
    let pool = FramePool::new(FrameLayout::new(format, width, height)?, 2, &HostAllocator)?;
    let mut writer = store.create_dataset("Example", spec.clone())?;
    let started = Instant::now();
    for t in 0..n {
        let mut f = pool.take(Duration::from_secs(5))?;
        let stride = f.layout().stride;
        for (y, row) in f.data_mut().chunks_mut(stride).enumerate() {
            for x in 0..width as usize {
                let v = 1_500
                    + ((x / 40 + y / 40 + t as usize) % 7) as u16 * 300
                    + ((x * 7 + y * 13) % 31) as u16;
                row[2 * x..2 * x + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        f.meta_mut().seq = t;
        writer.append(&f.freeze())?;
    }
    let info = writer.finish()?;
    let seconds = started.elapsed().as_secs_f64();
    let raw = n * u64::from(width) * u64::from(height) * 2;
    println!(
        "Wrote {} frames of {width} × {height}, 12 bits in 16, in {seconds:.2} s: {:.0} MB/s of raw frames; {:.1} MB on disk, {:.1} : 1.",
        info.frames,
        raw as f64 / seconds / 1e6,
        info.bytes as f64 / 1e6,
        raw as f64 / info.bytes as f64
    );
    // Read back, exactly.
    let reader = store.read(&info.id)?;
    let Samples::U16(first) = reader.read(0)? else {
        return Err(Error::internal("not 16 bits"));
    };
    let expected = |x: usize, y: usize| {
        1_500 + ((x / 40 + y / 40) % 7) as u16 * 300 + ((x * 7 + y * 13) % 31) as u16
    };
    let exact = first
        .iter()
        .enumerate()
        .all(|(i, v)| *v == expected(i % width as usize, i / width as usize));
    println!(
        "Frame 0 read back: {}.",
        if exact { "exactly" } else { "NOT exactly" }
    );
    println!("\nThe dataset's tree:");
    tree(&root.join(&info.path), 1, 2)?;
    // Abandoned half way: nothing left.
    let mut half = store.create_dataset("Example", spec)?;
    let dir = root.join("projects").join("Example").join(half.id());
    let mut f = pool.take(Duration::from_secs(5))?;
    f.data_mut().fill(0);
    half.append(&f.freeze())?;
    drop(half);
    println!(
        "\nA second dataset, dropped after one frame: its folder {}, and the project lists {} dataset.",
        if dir.exists() { "remains" } else { "is gone" },
        store.datasets("Example")?.len()
    );
    let _ = std::fs::remove_dir_all(&root);
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// Prints `dir`'s tree to `depth`, files with their sizes.
fn tree(dir: &Path, indent: usize, depth: usize) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| Error::internal(format!("{}: {e}", dir.display())))?
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().to_string();
        let pad = "  ".repeat(indent);
        if e.path().is_dir() {
            let files = std::fs::read_dir(e.path()).map(|r| r.count()).unwrap_or(0);
            println!("{pad}{name}/ ({files} entries)");
            if depth > 0 {
                tree(&e.path(), indent + 1, depth - 1)?;
            }
        } else {
            let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
            println!("{pad}{name} ({bytes} bytes)");
        }
    }
    Ok(())
}
