//! Scans over HTTP: a plan previewed, nothing moved; a z-scan as a job into a z-stack, read
//! back whole, its every step an event; a snake paused and resumed, then one cancelled through
//! the `jobs` Thing while paused, keeping its tiles; and refusals.

use crate::instrument::{Instrument, ObjectivesConfig, SensorSize, SimulatorConfig};
use crate::tests::{Scratch, ended, invoke, serve};
use serde_json::{Value, json};
use std::time::Duration;
use teta_wot::testing::{TestClient, TestEventStream};

/// The server for a tiny instrument on the CPU, at 20×, on a real-time clock.
async fn client(scratch: &Scratch) -> TestClient {
    let instrument = Instrument::new(
        &SimulatorConfig {
            gpu: false,
            sensor: SensorSize::Tiny,
            ..SimulatorConfig::default()
        },
        &ObjectivesConfig::default(),
    )
    .unwrap();
    let (client, _) = serve(scratch, instrument).await;
    assert_eq!(
        invoke(&client, "/datasets/create_project", json!({"name": "p"})).await["status"],
        "completed"
    );
    client
}

/// A 3 × 2 snake at 20× from where the stage starts: the tiny sensor's field is 11.52 × 8.64 µm.
fn snake() -> Value {
    json!({"kind": "snake", "x0": 2_000.0, "y0": 2_000.0, "x1": 2_030.0, "y1": 2_012.0, "overlap": 0.1})
}

/// The next `progress` event, within 30 s.
async fn next(events: &mut TestEventStream) -> Value {
    events
        .next_within(Duration::from_secs(30))
        .await
        .expect("a progress event")
}

/// The `progress` events up to one saying `message`: their messages.
async fn until(events: &mut TestEventStream, message: &str) -> Vec<String> {
    let mut seen = Vec::new();
    loop {
        let e = next(events).await;
        seen.push(e["message"].as_str().unwrap().to_string());
        if e["message"] == message {
            return seen;
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plan_is_previewed_unmoved_and_a_z_scan_is_a_job_into_a_z_stack_read_back_whole() {
    let _cpu = crate::tests::CPU.read().await;
    let scratch = Scratch::new();
    let client = client(&scratch).await;
    let here = client.get("/stage/position").await.json();
    let previewed = invoke(&client, "/scan/preview", json!({ "pattern": snake() })).await;
    assert_eq!(previewed["status"], "completed", "{previewed}");
    let plan = &previewed["output"];
    assert_eq!(plan["grid"], json!([3, 2]), "{plan}");
    assert_eq!(plan["positions"].as_array().unwrap().len(), 6);
    assert!(plan["estimate_s"].as_f64().unwrap() > 0.0);
    assert_eq!(plan["timing"]["settle_ms"], 50.0);
    assert_eq!(
        client.get("/stage/position").await.json(),
        here,
        "nothing moved"
    );
    // A z-scan of five planes, its steps followed as events.
    let mut events = client.events("/scan/progress").await;
    let started = invoke(
        &client,
        "/scan/scan",
        json!({"project": "p", "name": "a stack", "pattern": {"kind": "z_around", "planes": 5, "step": 0.5}}),
    )
    .await;
    let job = &started["output"];
    assert_eq!(
        (job["name"].as_str(), job["kind"].as_str()),
        (Some("scan"), Some("hardware")),
        "{started}"
    );
    let id = job["id"].as_str().unwrap().to_string();
    let mut done = Vec::new();
    let mut messages = Vec::new();
    loop {
        let e = next(&mut events).await;
        assert_eq!(
            (e["job"].as_str(), e["paused"].as_bool()),
            (Some(id.as_str()), Some(false))
        );
        done.push(e["done"].as_f64().unwrap());
        messages.push(e["message"].as_str().unwrap().to_string());
        if e["message"] == "written" {
            break;
        }
    }
    assert_eq!(
        messages,
        [
            "tile 1 of 5",
            "tile 2 of 5",
            "tile 3 of 5",
            "tile 4 of 5",
            "tile 5 of 5",
            "written"
        ]
    );
    assert_eq!(done, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    let job = ended(&client, &id).await;
    assert_eq!(job["state"], "succeeded", "{job}");
    // The planes rising, each recorded where it was planned.
    let tiles = job["result"]["tiles"].as_array().unwrap();
    let zs: Vec<f64> = tiles
        .iter()
        .map(|t| t["recorded"]["z"].as_f64().unwrap())
        .collect();
    assert!(
        zs.windows(2).all(|w| (w[1] - w[0] - 0.5).abs() < 0.32),
        "{zs:?}"
    );
    // One 3D array: c, z, y, x, the planes' step in its scale, read back whole.
    let info = &job["result"]["dataset"];
    let dir = scratch.0.join("data").join(info["path"].as_str().unwrap());
    let read = tetera_store::DatasetReader::open(&dir).unwrap();
    let stack = read.read_stack().unwrap();
    assert_eq!((stack.shape, stack.z_step_um), ([5, 72, 96], 0.5));
    let zarr: Value = serde_json::from_slice(
        &std::fs::read(dir.join(tetera_store::dataset::IMAGE).join("0/zarr.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(zarr["dimension_names"], json!(["c", "z", "y", "x"]));
    assert_eq!(&client.get("/scan/last").await.json()["dataset"], info);
    assert_eq!(client.get("/scan/running").await.json(), Value::Null);
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_scan_is_paused_resumed_and_cancelled_over_http_saying_how_far_it_has_got() {
    let _cpu = crate::tests::CPU.read().await;
    let scratch = Scratch::new();
    let client = client(&scratch).await;
    let mut events = client.events("/scan/progress").await;
    let started = invoke(
        &client,
        "/scan/scan",
        json!({"project": "p", "name": "a snake", "pattern": snake()}),
    )
    .await;
    let id = started["output"]["id"].as_str().unwrap().to_string();
    // Paused at its second tile: it holds at its third.
    until(&mut events, "tile 2 of 6").await;
    assert_eq!(
        invoke(&client, "/scan/pause", json!({})).await["status"],
        "completed"
    );
    let held = until(&mut events, "paused").await;
    assert!(held.len() <= 3, "{held:?}");
    let running = client.get("/scan/running").await.json();
    assert_eq!(
        (
            running["job"].as_str(),
            running["paused"].as_bool(),
            running["message"].as_str()
        ),
        (Some(id.as_str()), Some(true), Some("paused")),
        "{running}"
    );
    let job = invoke(&client, "/jobs/get", json!({ "id": id })).await["output"].clone();
    assert_eq!(
        (job["state"].as_str(), job["message"].as_str()),
        (Some("running"), Some("paused"))
    );
    let at = job["progress"].as_f64().unwrap();
    assert!(at > 0.0 && at < 1.0, "{at}");
    // A while on, no further.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let still = invoke(&client, "/jobs/get", json!({ "id": id })).await["output"].clone();
    assert_eq!(still["progress"], job["progress"]);
    // Resumed: on to the end.
    assert_eq!(
        invoke(&client, "/scan/resume", json!({})).await["status"],
        "completed"
    );
    let rest = until(&mut events, "written").await;
    assert!(rest.iter().any(|m| m == "tile 6 of 6"), "{rest:?}");
    let job = ended(&client, &id).await;
    assert_eq!(job["state"], "succeeded", "{job}");
    assert_eq!(job["result"]["tiles"].as_array().unwrap().len(), 6);
    // Another, cancelled through the `jobs` Thing while paused: its tiles kept.
    let started = invoke(
        &client,
        "/scan/scan",
        json!({"project": "p", "name": "cut short", "pattern": snake()}),
    )
    .await;
    let id = started["output"]["id"].as_str().unwrap().to_string();
    until(&mut events, "tile 3 of 6").await;
    invoke(&client, "/scan/pause", json!({})).await;
    until(&mut events, "paused").await;
    let asked = invoke(&client, "/jobs/cancel", json!({ "id": id })).await;
    assert_eq!(asked["output"]["cancel_requested"], true, "{asked}");
    assert_eq!(ended(&client, &id).await["state"], "cancelled");
    let listed = invoke(&client, "/datasets/list", json!({"project": "p"})).await["output"].clone();
    let cut = listed
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["name"] == "cut short")
        .expect("the partial dataset");
    let frames = cut["frames"].as_u64().unwrap();
    assert!((3..=4).contains(&frames), "{cut}");
    assert_eq!(client.get("/scan/running").await.json(), Value::Null);
    // None runs now: nothing to pause or resume.
    for path in ["/scan/pause", "/scan/resume"] {
        let refused = invoke(&client, path, json!({})).await;
        assert_eq!(refused["status"], "error", "{path}: {refused}");
    }
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_scan_is_refused_at_once_without_an_objective_a_project_or_a_request_it_can_take() {
    let _cpu = crate::tests::CPU.read().await;
    let scratch = Scratch::new();
    let client = client(&scratch).await;
    for bad in [
        json!({"project": "none", "name": "s", "pattern": snake()}),
        json!({"project": "p", "name": "s", "pattern": snake(), "settle_ms": -1.0}),
        json!({"project": "p", "name": "s", "pattern": {"kind": "z_range", "from": 0.0, "to": 1.0, "step": 0.0}}),
        json!({"project": "p", "name": "s", "pattern": {"kind": "z_around", "planes": 3, "step": 1.0}, "autofocus": {"at": "first"}}),
        json!({"project": "p", "name": "s", "pattern": {"kind": "snake", "x0": 99_990.0, "y0": 0.0, "x1": 100_020.0, "y1": 10.0, "overlap": 0.0}}),
    ] {
        let refused = invoke(&client, "/scan/scan", bad.clone()).await;
        assert_eq!(refused["status"], "error", "{bad}: {refused}");
    }
    assert_eq!(
        invoke(&client, "/objective/remove", json!({})).await["status"],
        "completed"
    );
    for (path, body) in [
        ("/scan/preview", json!({"pattern": snake()})),
        (
            "/scan/scan",
            json!({"project": "p", "name": "s", "pattern": snake()}),
        ),
    ] {
        let none = invoke(&client, path, body).await;
        assert_eq!(none["status"], "error", "{none}");
        assert!(none["error"].to_string().contains("objective"), "{none}");
    }
    assert_eq!(
        invoke(&client, "/jobs/list", json!({})).await["output"],
        json!([]),
        "no job made"
    );
    client.stop().await;
}
