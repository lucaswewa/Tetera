//! Scans (P1.17a and P1.17b, ADR-0022) on the virtual instrument at 10×, stop and go, each
//! tile at full resolution:
//!
//! - **a 5 × 5 snake:** the plan and its estimate, the time taken against it, and each tile's
//!   recorded position against the plan and against the simulator's truth;
//! - **a 21-plane z-scan** around the current z, read back as one 3D array, each plane's
//!   recorded z against the plan;
//! - **a 3 × 3 snake through the `scan` Thing,** over HTTP on a loopback port: paused at its
//!   fourth tile for a second, resumed, and followed to its end.
//!
//! Run it with `cargo run --release -p tetera-api --features cuda --example scan`; `-- --smoke`
//! does the same, smaller, with a tiny sensor on the CPU.

use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetera_api::instrument::{Instrument, ObjectivesConfig, SensorSize, SimulatorConfig};
use tetera_api::server::{Roots, build};
use tetera_hal::Stage;
use tetera_hal::objective::ObjectiveMount;
use tetera_workflows::scan::{
    AutofocusAt, Pattern, ScanControl, ScanRequest, ScanRig, Tile, preview, scan,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
    let service = build(
        instrument.clone(),
        Roots::make(root.join("data"), root.join("scratch"))?,
    )?;
    let store = service.store().clone();
    store.create_project("Example")?;
    let compute = instrument.compute().clone();
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
    // The field at 10×, and the rectangle n × n tiles at 10 % overlap cover around a centre.
    let here = instrument.stage.reading()?.position;
    let pixel = instrument.sensor_pixel_um() / 10.0;
    let (fw, fh) = (
        f64::from(compute.sensor[0]) * pixel,
        f64::from(compute.sensor[1]) * pixel,
    );
    let square = |n: f64, cx: f64, cy: f64| {
        let (w, h) = (fw + (n - 1.0) * 0.9 * fw, fh + (n - 1.0) * 0.9 * fh);
        Pattern::Snake {
            x0: cx - w / 2.0,
            y0: cy - h / 2.0,
            x1: cx + w / 2.0,
            y1: cy + h / 2.0,
            overlap: 0.1,
        }
    };
    let request = |name: &str, pattern: Pattern| ScanRequest {
        project: "Example".into(),
        name: name.into(),
        pattern,
        autofocus: AutofocusAt::Never,
        settle_ms: None,
        exposure_ms: None,
        write_mb_s: None,
    };

    // A snake, its time against the plan's, its positions against the plan and the truth.
    let (cx, cy) = if smoke {
        (here.x.um(), here.y.um())
    } else {
        (8_000.0, 6_000.0)
    };
    let snake = request("a snake", square(if smoke { 2.0 } else { 5.0 }, cx, cy));
    let plan = preview(&rig, &snake)?;
    println!(
        "\nThe plan: {} × {} tiles of {:.0} × {:.0} µm, from ({:.0}, {:.0}) µm; estimated at {:.2} s.",
        plan.grid[0],
        plan.grid[1],
        plan.field[0],
        plan.field[1],
        plan.positions[0].x.um(),
        plan.positions[0].y.um(),
        plan.estimate_s
    );
    let r = scan(&rig, &snake, &ScanControl::default(), &mut |_, _| Ok(()))?;
    let off = (r.seconds - plan.estimate_s) / r.seconds * 100.0;
    println!(
        "The scan took {:.2} s: the estimate {:+.1} % off it.",
        r.seconds, -off
    );
    let worst = |tiles: &[Tile], f: &dyn Fn(&Tile) -> f64| tiles.iter().map(f).fold(0.0, f64::max);
    let from_plan = worst(&r.tiles, &|t| {
        (t.recorded.x.um() - t.planned.x.um())
            .abs()
            .max((t.recorded.y.um() - t.planned.y.um()).abs())
    });
    let from_truth = worst(&r.tiles, &|t| {
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

    // A z-scan, read back as one 3D array.
    let planes = if smoke { 5 } else { 21 };
    let stack = request("a z-stack", Pattern::ZAround { planes, step: 1.0 });
    let plan = preview(&rig, &stack)?;
    let r = scan(&rig, &stack, &ScanControl::default(), &mut |_, _| Ok(()))?;
    println!(
        "\nA z-scan of {planes} planes 1 µm apart, {:.2} to {:.2} µm: {:.2} s, against {:.2} s estimated.",
        plan.positions[0].z.um(),
        plan.positions[planes as usize - 1].z.um(),
        r.seconds,
        plan.estimate_s
    );
    let read = store.read(&r.dataset.id)?.read_stack()?;
    let from_plan = worst(&r.tiles, &|t| (t.recorded.z.um() - t.planned.z.um()).abs());
    println!(
        "Read back as one 3D array, z by y by x: {} × {} × {}, its planes {} µm apart; each plane's recorded z within {from_plan:.4} µm of the plan.",
        read.shape[0], read.shape[1], read.shape[2], read.z_step_um
    );
    drop(read);

    // A snake through the `scan` Thing, paused and resumed.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(service.serve(listener, async move {
        let _ = stopped.await;
    }));
    let n = if smoke { 2.0 } else { 3.0 };
    let pause_at = if smoke { "tile 2 of 4" } else { "tile 4 of 9" };
    let pattern = serde_json::to_value(square(n, cx, cy))?;
    let plan = invoke(addr, "/scan/preview", json!({ "pattern": pattern })).await?;
    let began = Instant::now();
    let job = invoke(
        addr,
        "/scan/scan",
        json!({"project": "Example", "name": "steered", "pattern": pattern}),
    )
    .await?;
    let id = job["id"].as_str().unwrap_or_default().to_string();
    println!(
        "\nThrough the `scan` Thing at {addr}: a {n} × {n} snake, estimated at {:.2} s, job {id}.",
        plan["estimate_s"].as_f64().unwrap_or(0.0)
    );
    until(addr, &id, |job| job["message"] == pause_at).await?;
    invoke(addr, "/scan/pause", json!({})).await?;
    let held = loop {
        let running: Value =
            serde_json::from_slice(&http(addr, "GET", "/scan/running", None).await?)?;
        if running["message"] == "paused" {
            break running;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    let paused = Instant::now();
    tokio::time::sleep(Duration::from_secs(1)).await;
    let after = invoke(addr, "/jobs/get", json!({ "id": id })).await?;
    println!(
        "  paused after {pause_at}, {:.0} % done; a second on, still {:.0} %, \"{}\"",
        held["done"].as_f64().unwrap_or(0.0) * 100.0,
        after["progress"].as_f64().unwrap_or(0.0) * 100.0,
        after["message"].as_str().unwrap_or_default()
    );
    invoke(addr, "/scan/resume", json!({})).await?;
    let held_s = paused.elapsed().as_secs_f64();
    let job = until(addr, &id, |job| {
        matches!(
            job["state"].as_str(),
            Some("succeeded" | "failed" | "cancelled")
        )
    })
    .await?;
    anyhow::ensure!(job["state"] == "succeeded", "the scan ended {job}");
    println!(
        "  resumed after {held_s:.2} s, it ended in {:.2} s: {} tiles, {:.2} s scanning, against the {:.2} s estimated.",
        began.elapsed().as_secs_f64(),
        job["result"]["tiles"].as_array().map_or(0, Vec::len),
        job["result"]["seconds"].as_f64().unwrap_or(0.0) - held_s,
        plan["estimate_s"].as_f64().unwrap_or(0.0)
    );
    let _ = stop.send(());
    serving.await??;
    let _ = std::fs::remove_dir_all(&root);
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// Job `id`, polled until `done` says so.
async fn until(addr: SocketAddr, id: &str, done: impl Fn(&Value) -> bool) -> anyhow::Result<Value> {
    loop {
        let job = invoke(addr, "/jobs/get", json!({ "id": id })).await?;
        if done(&job) {
            return Ok(job);
        }
        anyhow::ensure!(
            !matches!(
                job["state"].as_str(),
                Some("succeeded" | "failed" | "cancelled")
            ),
            "the scan ended {job}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Invokes the action at `path` with `body`, and follows its invocation to its end: its output.
async fn invoke(addr: SocketAddr, path: &str, body: Value) -> anyhow::Result<Value> {
    let started: Value = serde_json::from_slice(&http(addr, "POST", path, Some(&body)).await?)?;
    let href = started["href"]
        .as_str()
        .unwrap_or_default()
        .replace(&format!("http://{addr}"), "");
    loop {
        let now: Value = serde_json::from_slice(&http(addr, "GET", &href, None).await?)?;
        match now["status"].as_str() {
            Some("pending" | "running") => tokio::time::sleep(Duration::from_millis(10)).await,
            Some("completed") => return Ok(now["output"].clone()),
            _ => anyhow::bail!("{path}: {now}"),
        }
    }
}

/// `method path`, with a JSON `body` if given, over a plain socket: the answer's body.
async fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> anyhow::Result<Vec<u8>> {
    let body = body.map(Value::to_string).unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    tokio::task::spawn_blocking(move || {
        let mut s = std::net::TcpStream::connect(addr)?;
        s.write_all(request.as_bytes())?;
        let mut answer = Vec::new();
        s.read_to_end(&mut answer)?;
        let end = answer
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map_or(answer.len(), |at| at + 4);
        let head = String::from_utf8_lossy(&answer[..end]).to_string();
        anyhow::ensure!(
            head.starts_with("HTTP/1.1 2"),
            "{head}{}",
            String::from_utf8_lossy(&answer[end..])
        );
        Ok(answer[end..].to_vec())
    })
    .await?
}
