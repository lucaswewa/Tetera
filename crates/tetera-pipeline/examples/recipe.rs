//! A JSON recipe checked and run (P1.08a, P1.08b): the registry's nodes, a broken recipe
//! refused with each of its problems, and a good one put in order and run on a synthetic full
//! frame. Then the live chain, node by node, from a synthetic sensor's raw frame to a JPEG, on
//! the CPU and, with the `cuda` feature, on the GPU, each checked against the ISP's own preview.
//!
//! Run it with `cargo run --release -p tetera-pipeline --example recipe --features cuda`. It
//! writes the preview to `target/recipe/preview.jpg`. With a path,
//! `-- PATH.json` checks that recipe instead, printing its problems or its order;
//! `-- --schema` prints the recipe schema, `-- --catalogue` every node's descriptor, and
//! `-- --smoke` takes a 640 × 480 frame.

use std::collections::BTreeMap;
use std::time::Instant;
use tetera_compute::Compute;
use tetera_core::{Error, Result};
use tetera_imaging::correct::subtract_dark;
use tetera_imaging::pattern::{Sensor, test_pattern};
use tetera_imaging::{ColourCorrection, FlatField, Image, Isp, IspParams};
use tetera_pipeline::{Context, Plan, Recipe, Registry, Value};

const GOOD: &str = include_str!("../recipes/crop-and-measure.json");
const BROKEN: &str = include_str!("../recipes/broken.json");
const CHAIN: &str = include_str!("../recipes/live-chain.json");

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
    let mut plan = Plan::new(&recipe, registry)?;
    println!("\n`{}`: {}", plan.name(), plan.order().join(" → "));
    let s = sensor();
    let raw = s.expose(&test_pattern(width, height, 12))?;
    let even = Image::new(
        width,
        height,
        3,
        [3_000u16; 3].repeat((width * height) as usize),
    )?;
    let (dark, flat) = (s.dark(width, height)?, s.expose(&even)?);
    let inputs = BTreeMap::from([
        ("raw".to_string(), Value::Raw(raw.clone())),
        ("dark".to_string(), Value::Raw(dark.clone())),
        ("flat".to_string(), Value::Raw(flat.clone())),
    ]);
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
    let mut backends = vec![("CPU", cpu)];
    let detected = Compute::detect()?;
    if detected.kind() != tetera_compute::BackendKind::Cpu {
        backends.push(("CUDA", detected));
    }
    let mut jpeg = Vec::new();
    for (label, compute) in &backends {
        let mut best = f64::INFINITY;
        let mut outputs = BTreeMap::new();
        for _ in 0..3 {
            let start = Instant::now();
            outputs = plan.run(&Context { compute }, inputs.clone())?;
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
        if outputs["preview"] != expected {
            return Err(Error::internal(format!(
                "on the {label}, the recipe's preview isn't the ISP's"
            )));
        }
        if let Some(Value::Encoded(_, bytes)) = outputs.get("jpeg") {
            jpeg.clone_from(bytes);
        }
        println!("  {label:<5} {best:>8.1} ms, the best of 3; the preview the ISP's own, exactly");
    }
    let dir = std::path::Path::new("target/recipe");
    std::fs::create_dir_all(dir)
        .map_err(|e| Error::internal(format!("couldn't make {}: {e}", dir.display())))?;
    let path = dir.join("preview.jpg");
    std::fs::write(&path, &jpeg)
        .map_err(|e| Error::internal(format!("couldn't write {}: {e}", path.display())))?;
    println!("  wrote {} ({} KB)", path.display(), jpeg.len() / 1_024);
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
        println!("\nsmoke: ok");
    }
    Ok(())
}
