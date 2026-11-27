//! A JSON recipe checked and run (P1.08a–d): the registry's nodes, a broken recipe refused
//! with each of its problems, and a good one put in order and run on a synthetic full frame.
//! Then the live chain, node by node, from a synthetic sensor's raw frame to a JPEG, on the CPU
//! and, with the `cuda` feature, on the GPU, node by node and fused, each checked against the
//! ISP's own preview. Then live: a synthetic camera's frames at 14.5 fps, the recipe running
//! each as it can, a frame that comes while one waits taking its place. Last, a batch over a
//! folder of frames, written to `target/recipe/frames` and read back, denoised whole and in
//! tiles, the two checked equal.
//!
//! Run it with `cargo run --release -p tetera-pipeline --example recipe --features cuda`. It
//! writes the preview to `target/recipe/preview.jpg`. With a path,
//! `-- PATH.json` checks that recipe instead, printing its problems or its order;
//! `-- --schema` prints the recipe schema, `-- --catalogue` every node's descriptor, and
//! `-- --smoke` takes a 640 × 480 frame.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;
use tetera_compute::Compute;
use tetera_core::{Error, Result};
use tetera_imaging::correct::subtract_dark;
use tetera_imaging::pattern::noisy;
use tetera_imaging::pattern::{Sensor, test_pattern};
use tetera_imaging::{ColourCorrection, FlatField, Image, Isp, IspParams};
use tetera_pipeline::{Context, Live, Plan, Recipe, Registry, Value};

const GOOD: &str = include_str!("../recipes/crop-and-measure.json");
const BROKEN: &str = include_str!("../recipes/broken.json");
const CHAIN: &str = include_str!("../recipes/live-chain.json");
const DENOISE: &str = include_str!("../recipes/denoise.json");

/// `image`, RGB of 16 bits, as a binary PPM file.
fn write_ppm(path: &std::path::Path, image: &Image<u16>) -> Result<()> {
    let mut bytes = format!("P6\n{} {}\n65535\n", image.width(), image.height()).into_bytes();
    for v in image.data() {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    std::fs::write(path, bytes)
        .map_err(|e| Error::internal(format!("couldn't write {}: {e}", path.display())))
}

/// A binary PPM file of 16 bits back as an RGB image.
fn read_ppm(path: &std::path::Path) -> Result<Image<u16>> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::not_found(format!("{}: {e}", path.display())))?;
    let bad = || Error::invalid_argument(format!("{} isn't a 16-bit PPM", path.display()));
    // The header is four words: P6, the width, the height and 65535, each after white space.
    let mut words = Vec::new();
    let mut at = 0;
    while words.len() < 4 {
        while bytes.get(at).is_some_and(u8::is_ascii_whitespace) {
            at += 1;
        }
        let start = at;
        while bytes.get(at).is_some_and(|b| !b.is_ascii_whitespace()) {
            at += 1;
        }
        words.push(
            std::str::from_utf8(&bytes[start..at])
                .map_err(|_| bad())?
                .to_string(),
        );
    }
    let [magic, w, h, max] = &words[..] else {
        return Err(bad());
    };
    let (w, h): (u32, u32) = (w.parse().map_err(|_| bad())?, h.parse().map_err(|_| bad())?);
    if magic != "P6" || max != "65535" {
        return Err(bad());
    }
    let data = bytes[at + 1..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_be_bytes(b))
        .collect();
    Image::new(w, h, 3, data)
}

/// A batch over a folder of `count` frames of `width × height`, written then read back, run by
/// the denoise recipe whole and in tiles of `tile`.
fn folder(registry: &Registry, width: u32, height: u32, count: u64, tile: u32) -> Result<()> {
    let dir = std::path::Path::new("target/recipe/frames");
    // A fresh folder, so frames of another run don't join the batch.
    if dir.exists() {
        std::fs::remove_dir_all(dir)
            .map_err(|e| Error::internal(format!("couldn't empty {}: {e}", dir.display())))?;
    }
    std::fs::create_dir_all(dir)
        .map_err(|e| Error::internal(format!("couldn't make {}: {e}", dir.display())))?;
    let clean = test_pattern(width, height, 12);
    for seed in 0..count {
        let frame = Image::new(width, height, 3, noisy(clean.data(), 80.0, 4_095, seed))?;
        write_ppm(&dir.join(format!("frame-{seed}.ppm")), &frame)?;
    }
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| Error::internal(format!("couldn't read {}: {e}", dir.display())))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "ppm"))
        .collect();
    paths.sort();
    let frames: Vec<BTreeMap<String, Value>> = paths
        .iter()
        .map(|p| {
            Ok(BTreeMap::from([(
                "image".to_string(),
                Value::U16(read_ppm(p)?),
            )]))
        })
        .collect::<Result<_>>()?;
    let recipe = Recipe::from_json(DENOISE)?;
    let mut plan = Plan::new(&recipe, registry)?;
    let reach = plan.reach()?;
    let compute = Compute::detect()?;
    let context = Context { compute: &compute };
    println!(
        "\nA batch over {} frames read back from {}, `{}`, reaching {reach} pixels:",
        frames.len(),
        dir.display(),
        plan.name()
    );
    let start = Instant::now();
    let whole = plan.run_batch(&context, frames.clone())?;
    let whole_ms = start.elapsed().as_secs_f64() * 1e3 / frames.len() as f64;
    let start = Instant::now();
    let tiled: Vec<BTreeMap<String, Value>> = frames
        .into_iter()
        .map(|f| plan.run_tiled(&context, f, (tile, tile)))
        .collect::<Result<_>>()?;
    let tiled_ms = start.elapsed().as_secs_f64() * 1e3 / whole.len() as f64;
    if tiled != whole {
        return Err(Error::internal(
            "the tiled batch differs from the whole one",
        ));
    }
    println!(
        "  whole, {whole_ms:.0} ms a frame; in tiles of {tile} × {tile}, {tiled_ms:.0} ms a frame; the two the same, exactly"
    );
    Ok(())
}

/// The sensor the ISP's example uses: an offset, vignetting, and unequal channels.
fn sensor() -> Sensor {
    Sensor {
        offset: 128,
        vignetting: 0.3,
        response: [0.7, 1.0, 0.5],
        ..Sensor::default()
    }
}

/// The live chain's recipe run on a synthetic raw frame of `width × height`, on each backend,
/// against the ISP's own preview.
fn live_chain(registry: &Registry, width: u32, height: u32) -> Result<()> {
    let recipe = Recipe::from_json(CHAIN)?;
    let plan = Plan::new(&recipe, registry)?;
    println!("\n`{}`: {}", plan.name(), plan.order().join(" → "));
    println!(
        "  on the GPU, the fused kernel takes the place of {}",
        plan.fused().join(", ")
    );
    let s = sensor();
    let raw = s.expose(&test_pattern(width, height, 12))?;
    let even = Image::new(
        width,
        height,
        3,
        [3_000u16; 3].repeat((width * height) as usize),
    )?;
    let (dark, flat) = (s.dark(width, height)?, s.expose(&even)?);
    let frame = BTreeMap::from([("raw".to_string(), Value::Raw(raw.clone()))]);
    let cpu = Compute::cpu_only()?;
    let isp = Isp::new(
        IspParams {
            dark: Some(dark.clone()),
            flat: Some(FlatField::new(&subtract_dark(&cpu, &flat, &dark)?)?),
            colour: ColourCorrection {
                white_balance: [1.0 / 0.7, 1.0, 2.0],
                ..ColourCorrection::IDENTITY
            },
            ..IspParams::preview(4_095, 1_920, 1_280)
        },
        *raw.layout(),
    )?;
    let expected = Value::U8(isp.run(&cpu, &raw)?);
    let detected = Compute::detect()?;
    let gpu = detected.kind() != tetera_compute::BackendKind::Cpu;
    let mut runs = vec![("CPU, node by node", &cpu, false)];
    if gpu {
        runs.push(("CUDA, node by node", &detected, false));
        runs.push(("CUDA, fused", &detected, true));
    }
    println!(
        "  the dark and the flat held, so each run takes a frame; the best of 3, after a first:"
    );
    let mut jpeg = Vec::new();
    for (label, compute, fuse) in runs {
        let mut plan = Plan::new(&recipe, registry)?;
        plan.set_fusion(fuse);
        plan.hold("dark", Value::Raw(dark.clone()))?;
        plan.hold("flat", Value::Raw(flat.clone()))?;
        let mut best = f64::INFINITY;
        let mut outputs = BTreeMap::new();
        for pass in 0..4 {
            let start = Instant::now();
            outputs = plan.run(&Context { compute }, frame.clone())?;
            if pass > 0 {
                best = best.min(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        if outputs["preview"] != expected {
            return Err(Error::internal(format!(
                "{label}: the recipe's preview isn't the ISP's"
            )));
        }
        if let Some(Value::Encoded(_, bytes)) = outputs.get("jpeg") {
            jpeg.clone_from(bytes);
        }
        let c = plan.copies();
        println!(
            "  {label:<19} {best:>7.1} ms; {} up and {} down, {:.1} MB; the preview the ISP's own, exactly",
            c.uploads,
            c.downloads,
            c.bytes as f64 / 1e6
        );
    }
    live(
        registry,
        &recipe,
        &raw,
        &dark,
        &flat,
        &detected,
        smoke_seconds(width),
    )?;
    let dir = std::path::Path::new("target/recipe");
    std::fs::create_dir_all(dir)
        .map_err(|e| Error::internal(format!("couldn't make {}: {e}", dir.display())))?;
    let path = dir.join("preview.jpg");
    std::fs::write(&path, &jpeg)
        .map_err(|e| Error::internal(format!("couldn't write {}: {e}", path.display())))?;
    println!("  wrote {} ({} KB)", path.display(), jpeg.len() / 1_024);
    Ok(())
}

/// How long the live run takes: 2 s, or half a second for the smoke run's small frame.
fn smoke_seconds(width: u32) -> f64 {
    if width < 1_000 { 0.5 } else { 2.0 }
}

/// The live chain run live, on each backend: frames from a synthetic camera at 14.5 fps for
/// `seconds`, each with its own noise.
fn live(
    registry: &Registry,
    recipe: &Recipe,
    raw: &tetera_imaging::Mosaic,
    dark: &tetera_imaging::Mosaic,
    flat: &tetera_imaging::Mosaic,
    detected: &Compute,
    seconds: f64,
) -> Result<()> {
    const PERIOD: Duration = Duration::from_micros(68_966);
    let frames: Vec<Value> = (0..4)
        .map(|seed| {
            let sites = noisy(raw.sites(), 20.0, 4_095, seed);
            tetera_imaging::Mosaic::new(*raw.layout(), sites).map(Value::Raw)
        })
        .collect::<Result<_>>()?;
    let count = (seconds / PERIOD.as_secs_f64()).round() as u32;
    println!(
        "\nLive, {count} frames from a camera at 14.5 fps, each run as the recipe can, after one to start:"
    );
    let gpu = detected.kind() != tetera_compute::BackendKind::Cpu;
    let mut runs = vec![("CPU, node by node", false, false)];
    if gpu {
        runs.push(("CUDA, node by node", true, false));
        runs.push(("CUDA, fused", true, true));
    }
    for (label, on_gpu, fuse) in runs {
        let compute = Arc::new(if on_gpu {
            Compute::detect()?
        } else {
            Compute::cpu_only()?
        });
        let mut plan = Plan::new(recipe, registry)?;
        plan.set_fusion(fuse);
        plan.hold("dark", Value::Raw(dark.clone()))?;
        plan.hold("flat", Value::Raw(flat.clone()))?;
        // One frame first, so the live run starts with its tables made and masters uploaded.
        let first = BTreeMap::from([("raw".to_string(), frames[0].clone())]);
        plan.run(&Context { compute: &compute }, first)?;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let seen = errors.clone();
        let live = Live::start(plan, compute, "raw", move |out| {
            if let Err(e) = out {
                seen.lock().expect("the list of errors").push(e.to_string());
            }
        })?;
        let start = Instant::now();
        for i in 0..count {
            live.offer(frames[i as usize % frames.len()].clone())?;
            let next = PERIOD * (i + 1);
            if let Some(wait) = next.checked_sub(start.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        while !live.is_idle() {
            std::thread::sleep(Duration::from_millis(1));
        }
        let elapsed = start.elapsed().as_secs_f64();
        let (_, stats) = live.stop()?;
        if let Some(e) = errors.lock().expect("the list of errors").first() {
            return Err(Error::internal(format!("{label}: a live run failed: {e}")));
        }
        println!(
            "  {label:<19} {} of {} run, {} gave way to a later one; {:.1} fps out",
            stats.ran,
            stats.offered,
            stats.replaced,
            stats.ran as f64 / elapsed
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let smoke = args.iter().any(|a| a == "--smoke");
    let registry = Registry::stock();
    if args.iter().any(|a| a == "--schema") {
        println!(
            "{}",
            serde_json::to_string_pretty(&Recipe::schema()).expect("JSON")
        );
        return Ok(());
    }
    if args.iter().any(|a| a == "--catalogue") {
        println!(
            "{}",
            serde_json::to_string_pretty(&registry.catalogue()).expect("JSON")
        );
        return Ok(());
    }
    if let Some(path) = args.iter().find(|a| !a.starts_with("--")) {
        let text =
            std::fs::read_to_string(path).map_err(|e| Error::not_found(format!("{path}: {e}")))?;
        let recipe = Recipe::from_json(&text)?;
        return match Plan::check(&recipe, &registry) {
            Ok(plan) => {
                println!(
                    "`{}` is valid; its nodes run in this order: {}",
                    plan.name(),
                    plan.order().join(", ")
                );
                Ok(())
            }
            Err(problems) => {
                for p in &problems {
                    println!("{p}");
                }
                Err(Error::invalid_argument(format!(
                    "{} problems",
                    problems.len()
                )))
            }
        };
    }

    println!("The registry:");
    for d in registry.descriptors() {
        println!("  {} {}: {}", d.id, d.version, d.summary);
    }

    let broken = Recipe::from_json(BROKEN)?;
    println!("\n`{}` is refused:", broken.name);
    match Plan::check(&broken, &registry) {
        Ok(_) => return Err(Error::internal("the broken recipe passed")),
        Err(problems) => {
            for p in problems {
                println!("  {p}");
            }
        }
    }

    let recipe = Recipe::from_json(GOOD)?;
    let mut plan = Plan::new(&recipe, &registry)?;
    println!(
        "\n`{}` is valid; its nodes run in this order: {}",
        plan.name(),
        plan.order().join(", ")
    );
    let (w, h) = if smoke { (640, 480) } else { (5_472, 3_648) };
    let frame = test_pattern(w, h, 12);
    let compute = Compute::detect()?;
    let start = Instant::now();
    let outputs = plan.run(
        &Context { compute: &compute },
        BTreeMap::from([("frame".to_string(), Value::U16(frame))]),
    )?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!("Run on a {w} × {h} test pattern in {ms:.2} ms:");
    if let Some(Value::U16(cropped)) = outputs.get("cropped") {
        println!(
            "  cropped: {} × {} × {}",
            cropped.width(),
            cropped.height(),
            cropped.channels()
        );
    }
    if let Some(Value::Table(table)) = outputs.get("statistics") {
        for c in 0..3 {
            println!(
                "  channel {c}: mean {:.1}, std {:.1}, from {} to {}",
                table[&format!("mean.{c}")],
                table[&format!("std.{c}")],
                table[&format!("min.{c}")],
                table[&format!("max.{c}")]
            );
        }
    }
    let (width, height) = if smoke { (640, 480) } else { (5_472, 3_648) };
    live_chain(&registry, width, height)?;
    if smoke {
        folder(&registry, 320, 240, 2, 128)?;
    } else {
        folder(&registry, width, height, 4, 1_024)?;
    }
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
