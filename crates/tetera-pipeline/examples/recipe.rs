//! A JSON recipe checked and run (P1.08a): the registry's nodes, a broken recipe refused with
//! each of its problems, and a good one put in order and run on a synthetic full frame.
//!
//! Run it with `cargo run --release -p tetera-pipeline --example recipe`. With a path,
//! `-- PATH.json` checks that recipe instead, printing its problems or its order;
//! `-- --schema` prints the recipe schema, `-- --catalogue` every node's descriptor, and
//! `-- --smoke` takes a 640 × 480 frame.

use std::collections::BTreeMap;
use std::time::Instant;
use tetera_compute::Compute;
use tetera_core::{Error, Result};
use tetera_imaging::pattern::test_pattern;
use tetera_pipeline::{Context, Plan, Recipe, Registry, Value};

const GOOD: &str = include_str!("../recipes/crop-and-measure.json");
const BROKEN: &str = include_str!("../recipes/broken.json");

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
    let (width, height) = if smoke { (640, 480) } else { (5_472, 3_648) };
    let frame = test_pattern(width, height, 12);
    let compute = Compute::detect()?;
    let start = Instant::now();
    let outputs = plan.run(
        &Context { compute: &compute },
        BTreeMap::from([("frame".to_string(), Value::U16(frame))]),
    )?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!("Run on a {width} × {height} test pattern in {ms:.2} ms:");
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
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
