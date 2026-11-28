//! The Things over HTTP, through teta-wot's in-process test client.

use crate::instrument::Instrument;
use crate::instrument::tests::instrument;
use crate::server::{Roots, THINGS, build};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use teta_wot::testing::TestClient;
use tetera_sim::GrblState;

/// A folder of the test's own, removed when it drops.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!("tetera-api-test-{}-{n}", std::process::id())))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The server for a test instrument, in-process, its roots in `scratch`.
async fn client(scratch: &Scratch) -> (TestClient, Arc<Instrument>) {
    let roots = Roots::make(scratch.0.join("data"), scratch.0.join("scratch")).unwrap();
    let instrument = Arc::new(instrument());
    let service = build(instrument.clone(), roots).unwrap();
    let client = TestClient::start(service.into_inner()).await.unwrap();
    (client, instrument)
}

/// Invokes `path` with `body`, and waits for the invocation to end: its last state.
async fn invoke(client: &TestClient, path: &str, body: Value) -> Value {
    let started = client.post_json(path, Some(&body)).await;
    assert_eq!(started.status, 201, "{path}: {}", started.json());
    let href = started.json()["href"]
        .as_str()
        .expect("an invocation's href")
        .replace("http://testserver", "");
    for _ in 0..1_000 {
        let now = client.get(&href).await.json();
        if !matches!(now["status"].as_str(), Some("pending" | "running")) {
            return now;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{path} never finished");
}

#[tokio::test]
async fn system_says_what_tetera_runs_on() {
    let scratch = Scratch::new();
    let (client, _) = client(&scratch).await;
    let td = client.get("/system/").await;
    assert_eq!(td.status, 200);
    let td = td.json();
    for property in ["version", "platform", "compute", "memory", "disks"] {
        assert!(td["properties"][property].is_object(), "{property}: {td}");
    }
    assert_eq!(
        client.get("/system/version").await.json(),
        json!(env!("CARGO_PKG_VERSION"))
    );
    let platform = client.get("/system/platform").await.json();
    assert_eq!(platform["os"], std::env::consts::OS);
    let compute = client.get("/system/compute").await.json();
    assert_eq!(compute["backend"], "cpu");
    assert_eq!(compute["sensor"], json!([96, 72]));
    assert!(
        compute["reason"]
            .as_str()
            .is_some_and(|r| r.contains("CPU"))
    );
    let memory = client.get("/system/memory").await.json();
    assert!(memory["total"].as_u64().is_some_and(|t| t > 0), "{memory}");
    let disks = client.get("/system/disks").await.json();
    let roles: Vec<&str> = disks
        .as_array()
        .expect("a list")
        .iter()
        .filter_map(|d| d["role"].as_str())
        .collect();
    assert_eq!(roles, ["data", "scratch"], "{disks}");
    assert_eq!(
        client
            .put_json("/system/version", &json!("9.9"))
            .await
            .status,
        405,
        "read-only"
    );
    client.stop().await;
}

#[tokio::test]
async fn the_openapi_document_and_the_docs_cover_every_thing() {
    let scratch = Scratch::new();
    let (client, _) = client(&scratch).await;
    let doc = client.get("/openapi.json").await;
    assert_eq!(doc.status, 200);
    let doc = doc.json();
    assert_eq!(doc["info"]["title"], "Tetera");
    let paths = doc["paths"].as_object().expect("paths");
    for path in [
        "/system/",
        "/system/version",
        "/system/platform",
        "/system/compute",
        "/system/memory",
        "/system/disks",
        "/simulation/",
        "/simulation/sample",
        "/simulation/load",
        "/simulation/unload",
        "/simulation/inject",
    ] {
        assert!(
            paths.contains_key(path),
            "{path} missing from {:?}",
            paths.keys()
        );
    }
    for thing in THINGS {
        assert_eq!(
            client.get(&format!("/{thing}/")).await.status,
            200,
            "{thing}"
        );
    }
    assert_eq!(client.get("/docs").await.status, 200);
    client.stop().await;
}

#[tokio::test]
async fn simulation_loads_and_unloads_a_sample() {
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    assert_eq!(
        client.get("/simulation/sample").await.json(),
        json!({"seed": 7, "thickness": 4.0, "folds": false})
    );
    let done = invoke(
        &client,
        "/simulation/load",
        json!({"seed": 8, "folds": true}),
    )
    .await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(
        done["output"],
        json!({"seed": 8, "thickness": 4.0, "folds": true})
    );
    assert_eq!(instrument.sample().map(|s| s.seed), Some(8));
    assert_eq!(
        instrument.camera.sample().map(|s| s.section().folds),
        Some(true)
    );
    assert_eq!(client.get("/simulation/sample").await.json()["seed"], 8);
    // A section the simulator doesn't make: refused, and nothing changes.
    let refused = invoke(
        &client,
        "/simulation/load",
        json!({"seed": 1, "thickness": 9.0}),
    )
    .await;
    assert_eq!(refused["status"], "error", "{refused}");
    assert!(
        refused["error"].to_string().contains("invalid_argument"),
        "{refused}"
    );
    assert_eq!(instrument.sample().map(|s| s.seed), Some(8));
    // Not a seed at all: refused before it runs.
    let invalid = client
        .post_json("/simulation/load", Some(&json!({"seed": "seven"})))
        .await;
    assert_eq!(invalid.status, 422);
    let done = invoke(&client, "/simulation/unload", json!({})).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(client.get("/simulation/sample").await.json(), Value::Null);
    assert_eq!(instrument.camera.sample(), None);
    client.stop().await;
}

#[tokio::test]
async fn simulation_shows_imperfections_on_demand() {
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    let done = invoke(
        &client,
        "/simulation/inject",
        json!({"imperfection": "stage_alarm"}),
    )
    .await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(instrument.stage.grbl_state(), GrblState::Alarm);
    let unknown = client
        .post_json(
            "/simulation/inject",
            Some(&json!({"imperfection": "camera_fire"})),
        )
        .await;
    assert_eq!(unknown.status, 422);
    client.stop().await;
}
