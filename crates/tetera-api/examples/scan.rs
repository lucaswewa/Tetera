//! A scan (P1.17a, ADR-0022): a 5 × 5 snake at 10× on the virtual instrument, stop and go,
//! each tile at full resolution. It prints the plan and its estimate, then the scan's time
//! against it, each tile's recorded position against the plan and against the simulator's
//! truth, and the dataset.
//!
//! Run it with `cargo run --release -p tetera-api --features cuda --example scan`; `-- --smoke`
//! takes a 2 × 2 snake of a tiny sensor on the CPU.

use std::sync::Arc;
use tetera_api::instrument::{Instrument, ObjectivesConfig, SensorSize, SimulatorConfig};
use tetera_hal::Stage;
use tetera_hal::objective::ObjectiveMount;
use tetera_store::Store;
use tetera_workflows::scan::{
    AutofocusAt, Pattern, ScanControl, ScanRequest, ScanRig, preview, scan,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let simulator = SimulatorConfig {
        gpu: !smoke,
        sensor: if smoke {
            SensorSize::Tiny
        } else {
            SensorSize::Auto
        },
        ..SimulatorConfig::default()
    };
    let instrument = Arc::new(Instrument::new(&simulator, &ObjectivesConfig::default())?);
    instrument.objectives.fit(Some("10x"))?;
    let root = std::env::temp_dir().join(format!("tetera-scan-{}", std::process::id()));
    let store = Store::open(&root)?;
    store.create_project("Example")?;
    let compute = instrument.compute();
    println!(
        "The virtual instrument: the camera on the {} ({} × {}), at 10×.",
        compute.backend, compute.sensor[0], compute.sensor[1]
    );
    let rig = ScanRig {
        camera: instrument.camera.as_ref(),
        stage: instrument.stage.as_ref(),
        bus: &instrument.bus,
        store: &store,
        live: None,
        objective: instrument.objectives.fitted(),
        sensor_pixel_um: instrument.sensor_pixel_um(),
        instrument: instrument.describe(),
    };
    // The field at 10×, and the rectangle 5 × 5 tiles at 10 % overlap cover, around the start.
    let here = instrument.stage.reading()?.position;
    let pixel = instrument.sensor_pixel_um() / 10.0;
    let (fw, fh) = (
        f64::from(compute.sensor[0]) * pixel,
        f64::from(compute.sensor[1]) * pixel,
    );
    let n = if smoke { 2.0 } else { 5.0 };
    let (w, h) = (fw + (n - 1.0) * 0.9 * fw, fh + (n - 1.0) * 0.9 * fh);
    let (cx, cy) = if smoke {
        (here.x.um(), here.y.um())
    } else {
        (8_000.0, 6_000.0)
    };
    let request = ScanRequest {
        project: "Example".into(),
        name: "a snake".into(),
        pattern: Pattern::Snake {
            x0: cx - w / 2.0,
            y0: cy - h / 2.0,
            x1: cx + w / 2.0,
            y1: cy + h / 2.0,
            overlap: 0.1,
        },
        autofocus: AutofocusAt::Never,
        settle_ms: None,
        exposure_ms: None,
        write_mb_s: None,
    };
    let plan = preview(&rig, &request)?;
    println!(
        "The plan: {} × {} tiles of {:.0} × {:.0} µm, from ({:.0}, {:.0}) µm; estimated at {:.2} s.",
        plan.grid[0],
        plan.grid[1],
        plan.field[0],
        plan.field[1],
        plan.positions[0].x.um(),
        plan.positions[0].y.um(),
        plan.estimate_s
    );
    let r = scan(&rig, &request, &ScanControl::default(), &mut |_, _| Ok(()))?;
    let off = (r.seconds - plan.estimate_s) / r.seconds * 100.0;
    println!(
        "The scan took {:.2} s: the estimate {:+.1} % off it.",
        r.seconds, -off
    );
    let worst = |f: &dyn Fn(&tetera_workflows::scan::Tile) -> f64| {
        r.tiles.iter().map(f).fold(0.0, f64::max)
    };
    let from_plan = worst(&|t| {
        (t.recorded.x.um() - t.planned.x.um())
            .abs()
            .max((t.recorded.y.um() - t.planned.y.um()).abs())
    });
    let from_truth = worst(&|t| {
        t.truth.map_or(f64::INFINITY, |s| {
            (t.recorded.x.um() - s.x.um())
                .abs()
                .max((t.recorded.y.um() - s.y.um()).abs())
        })
    });
    println!(
        "Each of the {} tiles' recorded positions: within {from_plan:.4} µm of the plan, the stage's steps of 0.3125 µm; within {from_truth:.4} µm of the simulator's truth.",
        r.tiles.len()
    );
    println!(
        "The dataset: {} frames of {} × {}, {:.0} MB on disk.",
        r.dataset.frames,
        compute.sensor[0],
        compute.sensor[1],
        r.dataset.bytes as f64 / 1e6
    );
    let _ = std::fs::remove_dir_all(&root);
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
