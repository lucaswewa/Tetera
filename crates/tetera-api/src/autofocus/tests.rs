//! Autofocus over HTTP: a job that finds focus and leaves the stage there; a blank field's
//! failure, the stage back; refusals; and a cancel through the `jobs` Thing.

use crate::tests::{Scratch, client, invoke};
use serde_json::{Value, json};
use std::time::Duration;
use teta_wot::testing::TestClient;

/// The test instrument's true focus: the section's bottom at 1,000 µm, its middle 2 µm up.
const FOCUS: f64 = 1_002.0;

/// 20×'s depth of field with 2.4 µm pixels.
const D: f64 = 0.55 / (0.42 * 0.42) + 2.4 / (20.0 * 0.42);

/// Job `id`, once it has ended.
async fn ended(client: &TestClient, id: &str) -> Value {
    for _ in 0..6_000 {
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

async fn z(client: &TestClient) -> f64 {
    client.get("/stage/position").await.json()["z"]
        .as_f64()
        .unwrap()
}

/// Moves the stage to `z`, and waits for it.
async fn to(client: &TestClient, z: f64) {
    let moved = invoke(
        client,
        "/stage/move_to",
        json!({"x": 2_000.0, "y": 2_000.0, "z": z}),
    )
    .await;
    assert_eq!(moved["status"], "completed", "{moved}");
}

#[tokio::test(flavor = "multi_thread")]
async fn autofocus_over_http_finds_focus_as_a_job_and_a_blank_field_fails_cleanly() {
    let scratch = Scratch::new();
    let (client, _) = client(&scratch).await;
    to(&client, FOCUS + 2.5 * D).await;
    let started = invoke(&client, "/autofocus/autofocus", json!({})).await;
    let job = &started["output"];
    assert_eq!(
        (job["name"].as_str(), job["kind"].as_str()),
        (Some("autofocus"), Some("hardware")),
        "{started}"
    );
    let job = ended(&client, job["id"].as_str().unwrap()).await;
    assert_eq!(job["state"], "succeeded", "{job}");
    let found = job["result"]["focus"]["z"].as_f64().unwrap();
    assert!((found - FOCUS).abs() <= D / 4.0, "{found}");
    assert!(
        (z(&client).await - found).abs() < 0.32,
        "the stage at the focus"
    );
    assert_eq!(job["result"]["measure"], "normalised_variance");
    // 20×'s, through the tiny sensor's 2.4 µm pixels.
    let d = job["result"]["focus"]["depth_of_field"].as_f64().unwrap();
    assert!((d - D).abs() < 1e-9, "{d}");
    assert_eq!(job["result"]["objective"], "20x");
    assert_eq!(&client.get("/autofocus/last").await.json(), &job["result"]);
    assert!(
        job["log"].as_array().unwrap().iter().any(|l| l["text"]
            .as_str()
            .is_some_and(|t| t.contains("in focus at"))),
        "{job}"
    );
    assert_eq!(client.get("/camera/streaming").await.json(), json!(false));
    // Another measure, by name.
    let other = invoke(
        &client,
        "/autofocus/autofocus",
        json!({"measure": "tenengrad"}),
    )
    .await;
    let other = ended(&client, other["output"]["id"].as_str().unwrap()).await;
    assert_eq!(other["result"]["measure"], "tenengrad", "{other}");
    // A blank field: no focus, the job failed, the stage back where it was.
    assert_eq!(
        invoke(&client, "/simulation/unload", json!({})).await["status"],
        "completed"
    );
    to(&client, FOCUS + 3.0).await;
    let blank = invoke(&client, "/autofocus/autofocus", json!({})).await;
    let blank = ended(&client, blank["output"]["id"].as_str().unwrap()).await;
    assert_eq!(blank["state"], "failed", "{blank}");
    let why = blank["error"].as_str().unwrap();
    assert!(why.contains("no focus") && why.contains("flat"), "{why}");
    assert!(
        (z(&client).await - (FOCUS + 3.0)).abs() < 0.32,
        "back where it was"
    );
    let last = client.get("/autofocus/last").await.json();
    assert_eq!(last["focus"]["z"], Value::Null, "{last}");
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn autofocus_is_refused_without_an_objective_or_with_bad_options_and_cancels_through_jobs() {
    let scratch = Scratch::new();
    let (client, _) = client(&scratch).await;
    let bad = invoke(
        &client,
        "/autofocus/autofocus",
        json!({"options": {"range": 0.0}}),
    )
    .await;
    assert_eq!(bad["status"], "error", "{bad}");
    assert_eq!(
        invoke(&client, "/objective/remove", json!({})).await["status"],
        "completed"
    );
    let none = invoke(&client, "/autofocus/autofocus", json!({})).await;
    assert_eq!(none["status"], "error", "{none}");
    assert!(none["error"].to_string().contains("objective"), "{none}");
    assert_eq!(
        invoke(&client, "/jobs/list", json!({})).await["output"],
        json!([]),
        "no job made"
    );
    // Cancelled once it has measured a position.
    let fitted = client.put_json("/objective/fitted", &json!("20x")).await;
    assert!(fitted.status.is_success(), "{}", fitted.text());
    let started = invoke(&client, "/autofocus/autofocus", json!({})).await;
    let id = started["output"]["id"].as_str().unwrap().to_string();
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
    assert_eq!(ended(&client, &id).await["state"], "cancelled");
    assert_eq!(
        client.get("/camera/streaming").await.json(),
        json!(false),
        "the stream it started, stopped"
    );
    client.stop().await;
}
