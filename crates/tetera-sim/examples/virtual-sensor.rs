//! Raw frames from the sensor (P1.12b): the H&E field at 20× exposed through the sensor model,
//! still, moving during its exposure, skewed by the rolling shutter, and dim at a high gain,
//! written as PNGs to `target/virtual-sensor/`, each 2 × 2 of the mosaic shown as one pixel.
//! Then the noise on an even field against the model, at each level of light. With
//! `--features cuda` and a GPU, the full 5472 × 3648 raw frame, timed still and moving, and
//! the GPU's sensor against the CPU's.
//!
//! Run it with `cargo run --release -p tetera-sim --example virtual-sensor`; `-- --smoke`
//! makes fewer and smaller frames.

use std::path::Path;
use std::time::{Duration, Instant};
use tetera_core::{Error, Result};
use tetera_sim::optics::{Sensor, blackbody, field_moving};
use tetera_sim::sensor::{bayer, expose};
use tetera_sim::{
    Exposure, FieldOfView, HeSample, Light, Motion, Optics, SectionConfig, SensorModel,
};

/// The camera's mounting angle in these examples: half a degree off the stage's axes.
const MOUNTING: f64 = 0.5 * std::f64::consts::PI / 180.0;

/// The rolling shutter's line time at full resolution: 3,648 rows in 69 ms, 14.49 frames a
/// second at 12 bits.
const LINE_TIME: f64 = 1.0 / (14.49 * 3_648.0);

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/virtual-sensor");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out)
        .map_err(|e| Error::internal(format!("{}: {e}", out.display())))?;
    let sample = HeSample::new(7, SectionConfig::default())?;
    let sensor = if smoke {
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
    let optics = Optics {
        sensor,
        ..Optics::new(20.0, 0.42, MOUNTING)?
    };
    let fov = FieldOfView {
        centre: (1_852.0, 1_852.0),
        optics,
    };
    let model = SensorModel::default();
    let light = Light::default();
    println!(
        "Raw frames at 20×, a sensor of {} × {} pixels, {:.3} µm a pixel:",
        sensor.width,
        sensor.height,
        optics.pixel()
    );
    let ms = |n: u64| Duration::from_millis(n);
    let still = Motion::STILL;
    let drifting = Motion {
        velocity: (200.0, 0.0),
        exposure: 0.01,
        line_time: LINE_TIME,
    };
    let skewing = Motion {
        velocity: (2_000.0, 0.0),
        exposure: 0.0005,
        line_time: LINE_TIME,
    };
    let dim = Light {
        intensity: 20.0,
        ..light
    };
    for (name, what, motion, light, exposure) in [
        (
            "still",
            "still, 10 ms at a gain of 1",
            still,
            light,
            shot(ms(10), 1.0, 12),
        ),
        (
            "8-bit",
            "still, in 8 bits",
            still,
            light,
            shot(ms(10), 1.0, 8),
        ),
        (
            "moving",
            "drifting 200 µm/s along x during 10 ms",
            drifting,
            light,
            shot(ms(10), 1.0, 12),
        ),
        (
            "skewed",
            "moving 2 mm/s, 0.5 ms a row, rows a line time apart",
            skewing,
            light,
            Exposure {
                time: Duration::from_micros(500),
                ..shot(ms(1), 8.0, 12)
            },
        ),
        (
            "dim",
            "a fiftieth of the light at a gain of 16",
            still,
            dim,
            shot(ms(10), 16.0, 12),
        ),
    ] {
        let start = Instant::now();
        let f = field_moving(&sample, &fov, &motion, &light, 2.0)?;
        let raw = expose(
            &f.rgb,
            sensor.width,
            sensor.height,
            &model,
            &exposure,
            11,
            0,
        )?;
        let mean = colour_mean(&raw, sensor.width, 1);
        println!(
            "  {what}: {:.0} ms, green {mean:.0} counts on average{}",
            start.elapsed().as_secs_f64() * 1e3,
            if motion.line_time > 0.0 && motion.velocity.0 > 1_000.0 {
                format!(
                    ", skewed {:.0} pixels top to bottom",
                    motion.velocity.0 * f64::from(sensor.height) * motion.line_time
                        / optics.pixel()
                )
            } else {
                String::new()
            }
        );
        png(
            &out.join(format!("raw-{name}.png")),
            &raw,
            sensor.width,
            sensor.height,
            &model,
            &light,
            &exposure,
        )?;
    }

    println!("\nThe noise on an even field, green, 10 ms at a gain of 1, against the model:");
    let n = if smoke { 128 } else { 256 };
    for irradiance in [0.0f32, 1.0, 10.0, 100.0, 1_000.0, 2_200.0] {
        let e = shot(ms(10), 1.0, 12);
        let even = vec![[irradiance; 3]; (n * n) as usize];
        let raw = expose(&even, n, n, &model, &e, 5, 0)?;
        let (mean, var) = statistics(&raw, n, 1);
        let electrons = f64::from(irradiance * model.quantum_efficiency[1] * 10.0)
            + f64::from(model.dark_current) * 0.01;
        let k = 1.0 / f64::from(model.electrons_per_count);
        let well = f64::from(model.full_well);
        let (model_mean, model_var) = if electrons >= well {
            (f64::from(model.black_level) + well * k, 0.0)
        } else {
            (
                f64::from(model.black_level) + electrons * k,
                (electrons + f64::from(model.read_noise).powi(2)) * k * k + 1.0 / 12.0,
            )
        };
        println!(
            "  {irradiance:>6} e⁻/ms: mean {mean:>7.1} against {model_mean:>7.1}, variance {var:>7.2} against {model_var:>7.2}"
        );
    }
    gpu(&sample, &model, smoke)?;
    println!("\nWrote the PNGs to {}", out.display());
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

fn shot(time: Duration, gain: f32, bits: u8) -> Exposure {
    Exposure {
        time,
        gain,
        bits,
        binning: 1,
    }
}

/// The middle of `times`.
#[cfg(feature = "cuda")]
fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// The mean count of `colour` in a raw frame `width` wide.
fn colour_mean(raw: &[u16], width: u32, colour: usize) -> f64 {
    statistics(raw, width, colour).0
}

/// The mean and variance of `colour`'s counts in a raw frame `width` wide.
fn statistics(raw: &[u16], width: u32, colour: usize) -> (f64, f64) {
    let v: Vec<f64> = (0..raw.len() as u32)
        .filter(|k| bayer(k % width, k / width) == colour)
        .map(|k| f64::from(raw[k as usize]))
        .collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    (mean, var)
}

/// The full raw frame on the GPU, if there is one: still and moving, and its sensor against
/// the CPU's.
#[cfg(feature = "cuda")]
fn gpu(sample: &HeSample, model: &SensorModel, smoke: bool) -> Result<()> {
    use std::sync::Arc;
    use tetera_sim::gpu::{HeTiles, Shot, TILE_BYTES, expose_on};
    let Ok(gpu) = tetera_compute::CudaBackend::new(0) else {
        return Ok(());
    };
    let gpu = Arc::new(gpu);
    let budgets = tetera_compute::budget::VramBudgets::new(8 * 1024 * TILE_BYTES);
    let tiles = HeTiles::new(
        gpu.clone(),
        *sample,
        budgets.reserve("virtual-sensor", 5_000 * TILE_BYTES)?,
    )?;
    let light = Light::default();
    let optics = Optics::new(20.0, 0.42, MOUNTING)?;
    let mut shot = Shot {
        model: *model,
        exposure: Exposure {
            time: Duration::from_millis(10),
            gain: 1.0,
            bits: 12,
            binning: 1,
        },
        seed: 11,
        frame: 0,
    };
    println!("\nThe full 5472 × 3648 raw frame on the GPU at 20×, field and sensor, brought back:");
    let moving = Motion {
        velocity: (1_000.0, 0.0),
        exposure: 0.01,
        line_time: LINE_TIME,
    };
    let fov = FieldOfView {
        centre: (2_000.0, 2_000.0),
        optics,
    };
    // The tiles made first, for the still field and the moving one's wider margin.
    let start = Instant::now();
    tiles.frame(&fov, &moving, &light, 2.0, &shot)?;
    println!(
        "  cold, its tiles made: {:.0} ms",
        start.elapsed().as_secs_f64() * 1e3
    );
    for (what, motion) in [
        ("still", Motion::STILL),
        ("moving 1 mm/s, 16 samples a pixel", moving),
    ] {
        let mut field = Vec::new();
        let mut whole = Vec::new();
        let mut green = 0.0;
        for _ in 0..5 {
            let start = Instant::now();
            tiles.field_moving_with(&fov, &motion, &light, 2.0, |_| gpu.synchronize())?;
            field.push(start.elapsed().as_secs_f64() * 1e3);
            let start = Instant::now();
            let raw = tiles.frame(&fov, &motion, &light, 2.0, &shot)?;
            whole.push(start.elapsed().as_secs_f64() * 1e3);
            green = colour_mean(&raw, 5_472, 1);
            shot.frame += 1;
        }
        println!(
            "  {what}: the field {:.1} ms, the raw frame brought back {:.1} ms (medians of 5), green {green:.0} counts on average",
            median(&mut field),
            median(&mut whole)
        );
    }
    // The sensor alone, on a field already on the GPU.
    let (w, h) = if smoke {
        (512u32, 384u32)
    } else {
        (5_472, 3_648)
    };
    let field: Vec<[f32; 3]> = (0..w * h)
        .map(|k| {
            let t = (k % w) as f32 / w as f32;
            [600.0 * t, 900.0 * t, 400.0 * (1.0 - t) + 5.0]
        })
        .collect();
    let rgb = gpu.upload(&field.iter().flatten().copied().collect::<Vec<f32>>())?;
    let mut raw = gpu.zeros::<u16>((w * h) as usize)?;
    expose_on(&gpu, &rgb, w, h, model, &shot.exposure, 11, 0, &mut raw)?;
    gpu.synchronize()?;
    let start = Instant::now();
    expose_on(&gpu, &rgb, w, h, model, &shot.exposure, 11, 0, &mut raw)?;
    gpu.synchronize()?;
    let took = start.elapsed().as_secs_f64() * 1e3;
    let on_gpu = gpu.download(&raw)?;
    let cpu = expose(&field, w, h, model, &shot.exposure, 11, 0)?;
    let differ = cpu.iter().zip(&on_gpu).filter(|(a, b)| a != b).count();
    println!(
        "  the sensor alone on {w} × {h}: {took:.2} ms; {differ} of {} counts differ from the CPU's",
        cpu.len()
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn gpu(_: &HeSample, _: &SensorModel, _: bool) -> Result<()> {
    Ok(())
}

/// Writes a raw frame as an 8-bit RGB PNG, each 2 × 2 of the mosaic one pixel, balanced so
/// bare glass under `light` would be a little below white.
fn png(
    path: &Path,
    raw: &[u16],
    width: u32,
    height: u32,
    model: &SensorModel,
    light: &Light,
    exposure: &Exposure,
) -> Result<()> {
    let (w, h) = (width / 2, height / 2);
    let warmth = blackbody(light.colour_temperature);
    let ms = exposure.time.as_secs_f32() * 1e3;
    let scale = if exposure.bits == 8 { 16.0 } else { 1.0 };
    let white: [f32; 3] = std::array::from_fn(|c| {
        1.2 * light.intensity * warmth[c] * model.quantum_efficiency[c] * ms * exposure.gain
            / model.electrons_per_count
    });
    let black = f32::from(model.black_level);
    let at = |col: u32, row: u32| f32::from(raw[(row * width + col) as usize]) * scale - black;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for j in 0..h {
        for i in 0..w {
            let (c, r) = (2 * i, 2 * j);
            let v = [
                at(c, r),
                0.5 * (at(c + 1, r) + at(c, r + 1)),
                at(c + 1, r + 1),
            ];
            rgb.extend_from_slice(&tetera_sim::he::srgb8(std::array::from_fn(|k| {
                v[k] / white[k]
            })));
        }
    }
    write_png(path, w, h, &rgb)
}

/// Writes 8-bit RGB as a PNG, its deflate stored, not compressed.
fn write_png(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<()> {
    let row = width as usize * 3;
    let mut raw = Vec::with_capacity((row + 1) * height as usize);
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
    let mut header = width.to_be_bytes().to_vec();
    header.extend_from_slice(&height.to_be_bytes());
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
