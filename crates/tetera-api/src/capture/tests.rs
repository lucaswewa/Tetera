//! A capture and its dataset over HTTP: taken as a job while the live view runs, listed, read,
//! its thumbnail served, deleted; cancelled through the `jobs` Thing, leaving nothing.

use crate::tests::{Scratch, client, invoke};
use serde_json::{Value, json};
use std::time::Duration;
use teta_wot::testing::TestClient;

/// Job `id`, once it has ended.
async fn ended(client: &TestClient, id: &str) -> Value {
    for _ in 0..3_000 {
        let job = invoke(client, "/jobs/get", json!({ "id": id })).await["output"].clone();
        if matches!(
            job["state"].as_str(),
            Some("succeeded" | "failed" | "cancelled")
        ) {
            return job;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("job {id} never ended");
}

/// The camera's frames delivered, once they have grown past `past`.
async fn delivered_past(client: &TestClient, past: u64) -> u64 {
    for _ in 0..1_000 {
        let n = client.get("/camera/frames").await.json()["delivered"]
            .as_u64()
            .unwrap();
        if n > past {
            return n;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the live view never went on past {past} frames");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_capture_over_http_makes_a_dataset_read_back_and_deleted_while_the_live_view_goes_on() {
    let scratch = Scratch::new();
    let (client, _) = client(&scratch).await;
    let made = invoke(
        &client,
        "/datasets/create_project",
        json!({"name": "Tonsil 3"}),
    )
    .await;
    assert_eq!(made["output"]["name"], "Tonsil 3", "{made}");
    let projects = client.get("/datasets/projects").await.json();
    assert_eq!(projects[0]["name"], "Tonsil 3", "{projects}");
    assert_eq!(
        invoke(&client, "/camera/start_streaming", json!({})).await["status"],
        "completed"
    );
    let live = delivered_past(&client, 0).await;
    // A burst, as a job.
    let started = invoke(
        &client,
        "/capture/capture",
        json!({"project": "Tonsil 3", "name": "a burst", "kind": {"kind": "burst", "count": 5}}),
    )
    .await;
    let job = &started["output"];
    assert_eq!(
        (job["name"].as_str(), job["kind"].as_str()),
        (Some("capture"), Some("hardware")),
        "{started}"
    );
    let job = ended(&client, job["id"].as_str().unwrap()).await;
    assert_eq!(job["state"], "succeeded", "{job}");
    let result = &job["result"];
    assert_eq!(result["dataset"]["frames"], 5, "{job}");
    assert!(
        job["log"][0]["text"]
            .as_str()
            .is_some_and(|t| t.contains("5 frames")),
        "{job}"
    );
    assert_eq!(&client.get("/capture/last").await.json(), result);
    // The live view back in its mode, and counted by the camera Thing as before.
    assert_eq!(client.get("/camera/mode").await.json(), json!(1));
    assert_eq!(client.get("/camera/streaming").await.json(), json!(true));
    let after = client.get("/camera/frames").await.json()["delivered"]
        .as_u64()
        .unwrap();
    assert!(after >= live);
    delivered_past(&client, after).await;
    // Listed, and read back: its files, metadata and thumbnail.
    let listed = invoke(&client, "/datasets/list", json!({"project": "Tonsil 3"})).await;
    let id = result["dataset"]["id"].as_str().unwrap();
    assert_eq!(listed["output"][0]["id"], id, "{listed}");
    let got = invoke(&client, "/datasets/get", json!({ "id": id })).await["output"].clone();
    let kinds: Vec<&str> = got["files"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["kind"].as_str())
        .collect();
    for kind in ["ome-zarr", "metadata", "thumbnail"] {
        assert!(kinds.contains(&kind), "{kind}: {got}");
    }
    let spec = &got["metadata"]["spec"];
    assert_eq!(
        (&spec["width"], &spec["height"], &spec["kind"]),
        (&json!(96), &json!(72), &json!("capture")),
        "the whole tiny sensor: {spec}"
    );
    assert_eq!(spec["instrument"]["instrument"]["simulated"], true);
    // The sensor's 2.4 µm pixels through the 20× objective fitted.
    assert_eq!(spec["instrument"]["instrument"]["objective"]["name"], "20x");
    assert!(
        (spec["pixel_size_um"].as_f64().unwrap() - 0.12).abs() < 1e-12,
        "{spec}"
    );
    assert_eq!(spec["instrument"]["camera"]["mode"], 0);
    assert_eq!(got["metadata"]["frames"].as_array().map(Vec::len), Some(5));
    let path = got["thumbnail"].as_str().unwrap();
    assert_eq!(path, format!("/api/v1/datasets/{id}/thumbnail.jpg"));
    let thumbnail = client.get(path).await;
    assert_eq!(thumbnail.status, 200);
    assert_eq!(thumbnail.header("content-type"), Some("image/jpeg"));
    assert!(thumbnail.body.starts_with(&[0xFF, 0xD8]), "a JPEG");
    // Deleted: gone from the list, its thumbnail with it.
    let deleted = invoke(&client, "/datasets/delete", json!({ "id": id })).await;
    assert_eq!(deleted["status"], "completed", "{deleted}");
    let listed = invoke(&client, "/datasets/list", json!({"project": "Tonsil 3"})).await;
    assert_eq!(listed["output"], json!([]));
    assert_eq!(client.get(path).await.status, 404);
    // Refused: a project that isn't there, at once, and requests that can't be taken.
    for (path, body) in [
        (
            "/capture/capture",
            json!({"project": "none", "name": "n", "kind": {"kind": "single"}}),
        ),
        (
            "/capture/capture",
            json!({"project": "Tonsil 3", "name": "n", "kind": {"kind": "burst", "count": 0}}),
        ),
        (
            "/capture/capture",
            json!({"project": "Tonsil 3", "name": "n", "kind": {"kind": "single"}, "exposure_ms": -1.0}),
        ),
        ("/datasets/create_project", json!({"name": ".hidden"})),
        ("/datasets/list", json!({"project": "none"})),
        ("/datasets/get", json!({"id": id})),
        ("/datasets/delete", json!({"id": id})),
    ] {
        let refused = invoke(&client, path, body.clone()).await;
        assert_eq!(refused["status"], "error", "{path} {body}: {refused}");
    }
    assert_eq!(
        invoke(&client, "/jobs/list", json!({})).await["output"]
            .as_array()
            .map(Vec::len),
        Some(1),
        "the refused made no job"
    );
    let _ = invoke(&client, "/camera/stop_streaming", json!({})).await;
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_capture_cancelled_through_the_jobs_thing_leaves_no_dataset_in_the_index() {
    let scratch = Scratch::new();
    let (client, _) = client(&scratch).await;
    let _ = invoke(&client, "/datasets/create_project", json!({"name": "p"})).await;
    let _ = invoke(&client, "/camera/start_streaming", json!({})).await;
    for kind in [
        json!({"kind": "burst", "count": 100_000}),
        json!({"kind": "series", "count": 1_000, "interval_ms": 50.0}),
    ] {
        let started = invoke(
            &client,
            "/capture/capture",
            json!({"project": "p", "name": "cancelled", "kind": kind}),
        )
        .await;
        let id = started["output"]["id"].as_str().unwrap().to_string();
        // Cancelled once it has frames.
        loop {
            let job = invoke(&client, "/jobs/get", json!({ "id": id })).await["output"].clone();
            if job["progress"].as_f64().is_some_and(|p| p > 0.0) {
                break;
            }
            assert!(
                job["state"] == "queued" || job["state"] == "running",
                "{job}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let asked = invoke(&client, "/jobs/cancel", json!({ "id": id })).await;
        assert_eq!(asked["output"]["cancel_requested"], true, "{asked}");
        assert_eq!(ended(&client, &id).await["state"], "cancelled", "{kind}");
        // Nothing in the index, being written or not, nor on disk.
        let index =
            tetera_store::Index::open(&scratch.0.join("data").join(crate::server::DATABASE))
                .unwrap();
        assert_eq!(index.writing().unwrap(), [], "{kind}");
        assert_eq!(index.datasets("p").unwrap(), [], "{kind}");
        let folders = std::fs::read_dir(scratch.0.join("data/projects/p"))
            .unwrap()
            .count();
        assert_eq!(folders, 0, "{kind}");
        let logs = std::fs::read_dir(scratch.0.join("scratch"))
            .unwrap()
            .count();
        assert_eq!(logs, 0, "no frame log left: {kind}");
        assert_eq!(client.get("/camera/mode").await.json(), json!(1));
        assert_eq!(client.get("/capture/last").await.json(), Value::Null);
    }
    let _ = invoke(&client, "/camera/stop_streaming", json!({})).await;
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_dataset_being_written_has_no_thumbnail_yet() {
    let scratch = Scratch::new();
    let roots =
        crate::server::Roots::make(scratch.0.join("data"), scratch.0.join("scratch")).unwrap();
    let service = crate::server::build(
        std::sync::Arc::new(crate::instrument::tests::instrument()),
        roots,
    )
    .unwrap();
    let store = service.store().clone();
    let client = TestClient::start(service.into_inner()).await.unwrap();
    store.create_project("p").unwrap();
    let format =
        tetera_core::PixelFormat::bayer16(tetera_core::frame::format::BayerPattern::Rggb, 12)
            .unwrap();
    let writer = store
        .create_dataset(
            "p",
            tetera_store::DatasetSpec {
                name: "half".into(),
                kind: "capture".into(),
                format,
                width: 16,
                height: 16,
                frames: 2,
                pixel_size_um: 1.0,
                interval_s: 0.0,
                instrument: Value::Null,
            },
        )
        .unwrap();
    let path = crate::media::thumbnail_path(writer.id());
    let r = client.get(&path).await;
    assert_eq!(r.status, 409, "{}", r.text());
    drop(writer);
    assert_eq!(client.get(&path).await.status, 404, "gone with its writer");
    client.stop().await;
}
