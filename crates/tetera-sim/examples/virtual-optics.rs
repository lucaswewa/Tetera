//! The field the sensor sees (P1.12a): the H&E sample through the 5×, 20× and 50×
//! objectives, through focus at 50×, and under a warm and a cool light, written as PNGs to
//! `target/virtual-optics/`. On the CPU, a scaled-down sensor; with `--features cuda` and a
//! GPU, the full 5472 × 3648 field too, timed cold, still and moving, against the 40 ms
//! budget.
//!
//! Run it with `cargo run --release -p tetera-sim --example virtual-optics`; `-- --smoke`
//! makes fewer and smaller fields.

use std::path::Path;
use std::time::Instant;
use tetera_core::{Error, Result};
use tetera_sim::optics::{Sensor, depth_groups, field};
use tetera_sim::{Field, FieldOfView, HeSample, Light, Optics, SectionConfig};

/// The objectives in `configs/objectives.toml`: magnification and numerical aperture.
const OBJECTIVES: [(&str, f64, f64); 3] =
    [("5x", 5.0, 0.14), ("20x", 20.0, 0.42), ("50x", 50.0, 0.55)];

/// The camera's mounting angle in these examples: half a degree off the stage's axes.
const MOUNTING: f64 = 0.5 * std::f64::consts::PI / 180.0;

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/virtual-optics");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out)
        .map_err(|e| Error::internal(format!("{}: {e}", out.display())))?;
    let sample = HeSample::new(7, SectionConfig::default())?;
    let small = if smoke {
        Sensor {
            width: 192,
            height: 128,
            pixel: 2.4,
        }
    } else {
        Sensor {
            width: 912,
            height: 608,
            pixel: 2.4,
        }
    };
    let centre = (1_852.0, 1_852.0);
    let warm = Light::default();
    println!(
        "The field the sensor sees, on the CPU, a sensor of {} × {} pixels:",
        small.width, small.height
    );
    for (name, m, na) in OBJECTIVES {
        let optics = Optics {
            sensor: small,
            ..Optics::new(m, na, MOUNTING)?
        };
        let groups = depth_groups(&optics, 2.0, 4.0);
        println!(
            "  {name}: {:.3} µm a pixel, depth of field {:.1} µm, {} depth group(s), blur σ {}",
            optics.pixel(),
            optics.depth_of_field(),
            groups.len(),
            groups
                .iter()
                .map(|g| format!("{:.1}", g.sigma))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let f = timed(
            &sample,
            &FieldOfView { centre, optics },
            &warm,
            2.0,
            "focused at 2 µm",
        )?;
        png(
            &out.join(format!("field-{name}.png")),
            &f,
            warm.intensity * 1.45,
        )?;
    }
    println!("\nThrough focus at 50×:");
    let fifty = Optics {
        sensor: small,
        ..Optics::new(50.0, 0.55, MOUNTING)?
    };
    let focuses: &[f64] = if smoke {
        &[2.0, 6.0]
    } else {
        &[0.0, 2.0, 4.0, 6.0, 10.0]
    };
    for &focus in focuses {
        let f = timed(
            &sample,
            &FieldOfView {
                centre,
                optics: fifty,
            },
            &warm,
            focus,
            &format!("focused at {focus} µm"),
        )?;
        png(
            &out.join(format!("field-50x-focus-{focus}.png")),
            &f,
            warm.intensity * 1.45,
        )?;
    }
    println!("\nUnder a cool light, 6,500 K, at 20×, unevenly lit by 15 %:");
    let cool = Light {
        colour_temperature: 6_500.0,
        unevenness: 0.15,
        ..warm
    };
    let twenty = Optics {
        sensor: small,
        ..Optics::new(20.0, 0.42, MOUNTING)?
    };
    let f = timed(
        &sample,
        &FieldOfView {
            centre,
            optics: twenty,
        },
        &cool,
        2.0,
        "focused at 2 µm",
    )?;
    png(&out.join("field-20x-cool.png"), &f, cool.intensity * 1.2)?;
    gpu(&sample, &out, smoke)?;
    println!("\nWrote the PNGs to {}", out.display());
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

fn timed(
    sample: &HeSample,
    fov: &FieldOfView,
    light: &Light,
    focus: f64,
    what: &str,
) -> Result<Field> {
    let start = Instant::now();
    let f = field(sample, fov, light, focus)?;
    let mean = f.rgb.iter().map(|p| f64::from(p[1])).sum::<f64>() / f.rgb.len() as f64;
    println!(
        "    {what}: {:.0} ms, green {mean:.0} e⁻/ms on average",
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(f)
}

/// The full field on the GPU, if there is one: cold, still and moving, at each objective.
#[cfg(feature = "cuda")]
fn gpu(sample: &HeSample, out: &Path, smoke: bool) -> Result<()> {
    use std::sync::Arc;
    use tetera_sim::gpu::{HeTiles, TILE_BYTES};
    let Ok(gpu) = tetera_compute::CudaBackend::new(0) else {
        return Ok(());
    };
    let gpu = Arc::new(gpu);
    let budgets = tetera_compute::budget::VramBudgets::new(8 * 1024 * TILE_BYTES);
    let tiles = HeTiles::new(
        gpu.clone(),
        *sample,
        budgets.reserve("virtual-optics", 5_000 * TILE_BYTES)?,
    )?;
    let light = Light::default();
    println!("\nThe full 5472 × 3648 field on the GPU, against the 40 ms budget:");
    for (name, m, na) in if smoke {
        &OBJECTIVES[..1]
    } else {
        &OBJECTIVES[..]
    } {
        let optics = Optics::new(*m, *na, MOUNTING)?;
        let mut times = Vec::new();
        for step in 0..4 {
            // Cold, still, then moving 100 pixels a frame.
            let shift = f64::from(step.max(1) - 1) * 100.0 * optics.pixel();
            let fov = FieldOfView {
                centre: (2_000.0 + shift, 2_000.0),
                optics,
            };
            let start = Instant::now();
            tiles.field_with(&fov, &light, 2.0, |_| gpu.synchronize())?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "  {name}: {:.0} ms cold, {:.1} ms still, {:.1} and {:.1} ms moving 100 pixels a frame",
            times[0], times[1], times[2], times[3]
        );
        if *name == "20x" {
            let f = tiles.field(
                &FieldOfView {
                    centre: (2_000.0, 2_000.0),
                    optics,
                },
                &light,
                2.0,
            )?;
            png(
                &out.join("field-20x-full-gpu.png"),
                &f,
                light.intensity * 1.45,
            )?;
        }
    }
    let s = tiles.stats();
    println!(
        "  the cache: {} tiles made, {} found, {} MiB held",
        s.misses,
        s.hits,
        s.bytes / 1_048_576
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn gpu(_: &HeSample, _: &Path, _: bool) -> Result<()> {
    Ok(())
}

/// Writes `f` as an 8-bit RGB PNG, `white` photoelectrons a ms the brightest.
fn png(path: &Path, f: &Field, white: f32) -> Result<()> {
    let rgb = f.srgb8(white);
    let row = f.width as usize * 3;
    let mut raw = Vec::with_capacity((row + 1) * f.height as usize);
    for line in rgb.chunks(row) {
        raw.push(0);
        raw.extend_from_slice(line);
    }
    let mut z = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (k, block) in blocks.iter().enumerate() {
        z.push(u8::from(k + 1 == blocks.len()));
        let len = block.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + u32::from(x)) % 65_521;
        b = (b + a) % 65_521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = f.width.to_be_bytes().to_vec();
    header.extend_from_slice(&f.height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    for (kind, data) in [
        (b"IHDR", &header[..]),
        (b"IDAT", &z[..]),
        (b"IEND", &[][..]),
    ] {
        png.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend_from_slice(kind);
        png.extend_from_slice(data);
        let mut crc = !0u32;
        for &byte in &png[start..] {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xedb8_8320
                } else {
                    crc >> 1
                };
            }
        }
        png.extend_from_slice(&(!crc).to_be_bytes());
    }
    std::fs::write(path, png).map_err(|e| Error::internal(format!("{}: {e}", path.display())))
}
