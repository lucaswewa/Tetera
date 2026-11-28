//! The Things over HTTP, through teta-wot's in-process test client.

use crate::instrument::Instrument;
use crate::instrument::tests::{instrument, stepped};
use crate::server::{Roots, THINGS, build};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use teta_wot::testing::TestClient;
use tetera_core::Clock;
use tetera_hal::camera::Camera;
use tetera_sim::GrblState;

/// A folder of the test's own, removed when it drops.
pub(crate) struct Scratch(pub(crate) PathBuf);

impl Scratch {
    pub(crate) fn new() -> Self {
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
    serve(scratch, instrument()).await
}

/// The server for `instrument`, in-process, its roots in `scratch`.
async fn serve(scratch: &Scratch, instrument: Instrument) -> (TestClient, Arc<Instrument>) {
    let roots = Roots::make(scratch.0.join("data"), scratch.0.join("scratch")).unwrap();
    let instrument = Arc::new(instrument);
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
    let mut expected = vec![
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
    ];
    let devices: [(&str, &[&str]); 5] = [
        (
            "camera",
            &[
                "capabilities",
                "config",
                "mode",
                "exposure",
                "gain",
                "bit_depth",
                "pixel_format",
                "binning",
                "roi",
                "white_balance",
                "streaming",
                "frames",
                "last_frame",
                "start_streaming",
                "stop_streaming",
            ],
        ),
        (
            "stage",
            &[
                "capabilities",
                "position",
                "state",
                "moving",
                "speed",
                "move_to",
                "move_by",
                "jog",
                "stop_jog",
                "halt",
                "home",
            ],
        ),
        ("objective", &["catalogue", "fitted", "objective", "remove"]),
        (
            "illumination",
            &["capabilities", "on", "intensity", "colour_temperature"],
        ),
        ("preview", &["defaults", "sessions", "statistics", "stream"]),
    ];
    let named: Vec<String> = devices
        .iter()
        .flat_map(|(thing, affordances)| {
            std::iter::once(format!("/{thing}/"))
                .chain(affordances.iter().map(move |a| format!("/{thing}/{a}")))
        })
        .collect();
    expected.extend(named.iter().map(String::as_str));
    for path in expected {
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

#[test]
fn the_roots_are_made_and_the_settings_kept_in_the_data_root() {
    let scratch = Scratch::new();
    let roots = Roots::make(scratch.0.join("d"), scratch.0.join("s")).unwrap();
    assert!(roots.data.is_dir() && roots.scratch.is_dir());
    assert_eq!(roots.settings(), scratch.0.join("d").join("settings"));
    std::fs::write(scratch.0.join("file"), "").unwrap();
    assert!(Roots::make(scratch.0.join("file").join("d"), scratch.0.join("s")).is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn the_service_serves_on_a_port_until_asked_to_stop() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let scratch = Scratch::new();
    let roots = Roots::make(scratch.0.join("data"), scratch.0.join("scratch")).unwrap();
    let service = build(Arc::new(instrument()), roots).unwrap();
    assert!(format!("{service:?}").contains("Service"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(service.serve(listener, async move {
        let _ = stopped.await;
    }));
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    let request =
        format!("GET /system/version HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut answer = String::new();
    stream.read_to_string(&mut answer).await.unwrap();
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert!(!serving.is_finished(), "serving until asked to stop");
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(10), serving)
        .await
        .expect("it stops within its grace")
        .unwrap()
        .unwrap();
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

/// Steps `instrument`'s clock a millisecond at a time while `work` runs: its result.
async fn stepping<T>(instrument: &Instrument, work: impl std::future::Future<Output = T>) -> T {
    let clock = instrument.clock.clone();
    let (stop, mut stopped) = tokio::sync::oneshot::channel::<()>();
    let stepper = tokio::spawn(async move {
        loop {
            if stopped.try_recv().is_ok() {
                break;
            }
            clock.step(Duration::from_millis(1)).unwrap();
            tokio::time::sleep(Duration::from_micros(300)).await;
        }
    });
    let out = work.await;
    let _ = stop.send(());
    stepper.await.unwrap();
    out
}

/// A position as JSON's x, y and z.
fn xyz(v: &Value) -> [f64; 3] {
    ["x", "y", "z"].map(|k| v[k].as_f64().unwrap_or(f64::NAN))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_move_through_http_takes_the_time_the_stage_says_and_is_watched_as_it_goes() {
    use tetera_core::{StagePos, Um};
    use tetera_sim::Profile;
    let scratch = Scratch::new();
    let (client, instrument) = serve(&scratch, stepped()).await;
    assert_eq!(
        xyz(&client.get("/stage/position").await.json()),
        [2_000.0, 2_000.0, 1_002.0]
    );
    assert_eq!(client.get("/stage/state").await.json(), "idle");
    let mut ws = client.websocket("/stage/ws").await;
    ws.send_json(&json!({"messageType": "addPropertyObservation", "data": {"position": true}}))
        .await;
    // A bad request after it: its answer means the observation is in place.
    ws.send_json(&json!({"messageType": "addPropertyObservation", "data": {"nothing": true}}))
        .await;
    ws.receive_json().await;
    let config = instrument.stage.config();
    let limits = config.axes.map(|a| a.limits);
    let (from, to) = (
        StagePos::new(Um(2_000.0), Um(2_000.0), Um(1_002.0)),
        StagePos::new(Um(2_500.0), Um(2_000.0), Um(1_002.0)),
    );
    let expected = Profile::new(from, to, Some(5_000.0), &limits)
        .unwrap()
        .duration();
    let start = instrument.clock.now();
    let done = stepping(
        &instrument,
        invoke(
            &client,
            "/stage/move_to",
            json!({"x": 2_500.0, "y": 2_000.0, "z": 1_002.0, "speed": 5_000.0}),
        ),
    )
    .await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(
        xyz(&done["output"]["position"]),
        [2_500.0, 2_000.0, 1_002.0]
    );
    // Its reading is the first report after the stage stopped: within a report's period
    // and latency of the move's own time.
    let at = Duration::from_nanos(done["output"]["at"].as_u64().unwrap());
    let took = at.saturating_sub(start.as_duration());
    let late = config.report_period + config.report_latency;
    assert!(
        took >= expected && took <= expected + late + Duration::from_millis(2),
        "took {took:?}, the move {expected:?}"
    );
    // Watched over WebSocket: the position on its way.
    let mut seen = Vec::new();
    while let Some(message) = ws.receive_json_within(Duration::from_millis(200)).await {
        if message["messageType"] == "propertyStatus" {
            seen.push(xyz(&message["data"]["position"])[0]);
        }
    }
    let between = seen
        .iter()
        .filter(|x| **x > 2_000.0 && **x < 2_500.0)
        .count();
    assert!(between >= 3, "positions on the way: {seen:?}");
    assert!(seen.windows(2).all(|w| w[1] >= w[0]), "in order: {seen:?}");
    assert_eq!(seen.last().copied(), Some(2_500.0));
    assert_eq!(client.get("/stage/state").await.json(), "idle");
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stage_alarm_is_a_hardware_error_and_homing_clears_it() {
    let scratch = Scratch::new();
    let (client, instrument) = serve(&scratch, stepped()).await;
    invoke(
        &client,
        "/simulation/inject",
        json!({"imperfection": "stage_alarm"}),
    )
    .await;
    let refused = stepping(
        &instrument,
        invoke(&client, "/stage/move_by", json!({"dx": 100.0})),
    )
    .await;
    assert_eq!(refused["status"], "error", "{refused}");
    let said = refused["error"].to_string();
    assert!(said.contains("hardware"), "{said}");
    let homed = stepping(&instrument, invoke(&client, "/stage/home", json!({}))).await;
    assert_eq!(homed["status"], "completed", "{homed}");
    assert_eq!(xyz(&homed["output"]["position"]), [0.0, 0.0, 0.0]);
    let moved = stepping(
        &instrument,
        invoke(&client, "/stage/move_by", json!({"dx": 100.0})),
    )
    .await;
    assert_eq!(moved["status"], "completed", "{moved}");
    // A target beyond the travel: refused as an invalid argument.
    let beyond = stepping(
        &instrument,
        invoke(
            &client,
            "/stage/move_to",
            json!({"x": 1e9, "y": 0.0, "z": 0.0}),
        ),
    )
    .await;
    assert!(
        beyond["error"].to_string().contains("invalid_argument"),
        "{beyond}"
    );
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_move_stops_the_stage_and_a_jog_ends_on_request() {
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    let started = client
        .post_json(
            "/stage/move_to",
            Some(&json!({"x": 60_000.0, "y": 2_000.0, "z": 1_002.0, "speed": 2_000.0})),
        )
        .await;
    let href = started.json()["href"]
        .as_str()
        .unwrap()
        .replace("http://testserver", "");
    for _ in 0..200 {
        if client.get("/stage/state").await.json() == "moving" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(client.get("/stage/state").await.json(), "moving");
    assert_eq!(
        client.get("/stage/moving").await.json(),
        true,
        "moving as it moves"
    );
    client.delete(&href).await;
    let mut status = Value::Null;
    for _ in 0..500 {
        status = client.get(&href).await.json();
        if !matches!(status["status"].as_str(), Some("pending" | "running")) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(status["status"], "cancelled", "{status}");
    let x = instrument.stage.position().x.um();
    assert!(x > 2_000.0 && x < 60_000.0, "stopped on the way, at {x}");
    assert!(instrument.stage.grbl_state() == GrblState::Idle);
    // A jog, then its end.
    let jog = invoke(
        &client,
        "/stage/jog",
        json!({"axis": "y", "speed": 1_000.0}),
    )
    .await;
    assert_eq!(jog["status"], "completed", "{jog}");
    assert_eq!(instrument.stage.grbl_state(), GrblState::Jog);
    let ended = invoke(&client, "/stage/stop_jog", json!({})).await;
    assert_eq!(ended["status"], "completed", "{ended}");
    assert_eq!(ended["output"]["state"], "idle");
    let halted = invoke(&client, "/stage/halt", json!({})).await;
    assert_eq!(halted["status"], "completed", "{halted}");
    assert_eq!(client.get("/stage/moving").await.json(), false, "still");
    // The speed of moves that name none.
    assert_eq!(client.get("/stage/speed").await.json(), json!(5_000.0));
    assert_eq!(
        client.put_json("/stage/speed", &json!(0.0)).await.status,
        422
    );
    assert_eq!(
        client
            .put_json("/stage/speed", &json!(10_000.0))
            .await
            .status,
        201
    );
    assert_eq!(client.get("/stage/speed").await.json(), json!(10_000.0));
    let caps = client.get("/stage/capabilities").await.json();
    assert_eq!(caps["axes"].as_array().map(Vec::len), Some(3), "{caps}");
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_camera_is_configured_property_by_property_and_streams() {
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    let caps = client.get("/camera/capabilities").await.json();
    assert_eq!(caps["modes"].as_array().map(Vec::len), Some(3), "{caps}");
    assert_eq!(
        client.get("/camera/mode").await.json(),
        1,
        "the binned live mode (D5)"
    );
    assert_eq!(client.get("/camera/bit_depth").await.json(), 8);
    assert_eq!(client.get("/camera/exposure").await.json(), json!(10.0));
    assert_eq!(
        client
            .put_json("/camera/exposure", &json!(20.0))
            .await
            .status,
        201
    );
    assert_eq!(client.get("/camera/exposure").await.json(), json!(20.0));
    assert_eq!(
        client.get("/camera/config").await.json()["exposure"],
        json!(20.0)
    );
    for (path, refused) in [
        ("/camera/exposure", json!(50_000.0)),
        ("/camera/exposure", json!(-1.0)),
        ("/camera/mode", json!(7)),
        ("/camera/gain", json!(100.0)),
        ("/camera/bit_depth", json!(10)),
        ("/camera/binning", json!({"x": 4, "y": 4})),
        (
            "/camera/roi",
            json!({"x": 0, "y": 0, "width": 8, "height": 8}),
        ),
    ] {
        let answer = client.put_json(path, &refused).await;
        assert_eq!(answer.status, 422, "{path} = {refused}: {}", answer.json());
    }
    assert_eq!(
        client.put_json("/camera/bit_depth", &json!(8)).await.status,
        201
    );
    assert_eq!(client.get("/camera/bit_depth").await.json(), 8);
    let format = client.get("/camera/pixel_format").await.json();
    assert!(format.to_string().contains('8'), "{format}");
    assert_eq!(client.put_json("/camera/mode", &json!(2)).await.status, 201);
    assert_eq!(client.get("/camera/mode").await.json(), 2);
    // All at once: the configuration with its exposure changed.
    let mut config = client.get("/camera/config").await.json();
    config["exposure"] = json!(30.0);
    assert_eq!(client.put_json("/camera/config", &config).await.status, 201);
    assert_eq!(client.get("/camera/exposure").await.json(), json!(30.0));
    config["mode"] = json!(9);
    assert_eq!(client.put_json("/camera/config", &config).await.status, 422);
    assert_eq!(
        client.put_json("/camera/gain", &json!(4.0)).await.status,
        201
    );
    assert_eq!(client.get("/camera/gain").await.json(), json!(4.0));
    assert_eq!(
        client.get("/camera/binning").await.json(),
        json!({"x": 1, "y": 1})
    );
    assert_eq!(client.get("/camera/roi").await.json(), Value::Null);
    let start = client.get("/camera/white_balance").await.json();
    let gains = [&start["red"], &start["green"], &start["blue"]].map(|g| g.as_f64().unwrap());
    assert!(
        gains
            .iter()
            .zip(instrument.white_balance())
            .all(|(a, b)| (a - b).abs() < 1e-12),
        "balanced for the light at the start: {gains:?}"
    );
    let balance = json!({"red": 1.5, "green": 1.0, "blue": 0.8});
    assert_eq!(
        client
            .put_json("/camera/white_balance", &balance)
            .await
            .status,
        201
    );
    assert_eq!(client.get("/camera/white_balance").await.json(), balance);
    assert_eq!(
        client
            .put_json("/camera/white_balance", &json!({"red": 1.5}))
            .await
            .status,
        422
    );
    // The stream, into the frame bus.
    assert_eq!(client.get("/camera/streaming").await.json(), false);
    let latest = instrument.bus.latest();
    let started = invoke(&client, "/camera/start_streaming", json!({})).await;
    assert_eq!(started["status"], "completed", "{started}");
    assert_eq!(client.get("/camera/streaming").await.json(), true);
    let mut frames = Value::Null;
    for _ in 0..500 {
        frames = client.get("/camera/frames").await.json();
        if frames["delivered"].as_u64().unwrap_or(0) >= 3 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(frames["delivered"].as_u64().unwrap_or(0) >= 3, "{frames}");
    let on_the_bus = latest
        .recv(Duration::from_secs(5))
        .expect("a frame on the bus");
    assert_eq!(on_the_bus.meta().mode, 2);
    assert!(instrument.bus.published() >= 3);
    drop((on_the_bus, latest));
    let last = client.get("/camera/last_frame").await.json();
    assert_eq!(last["mode"], 2, "{last}");
    assert!(last["position"].is_object(), "{last}");
    let busy = client.put_json("/camera/exposure", &json!(5.0)).await;
    assert_eq!(busy.status, 500, "busy while streaming");
    assert!(busy.json().to_string().contains("busy"), "{}", busy.json());
    let again = invoke(&client, "/camera/start_streaming", json!({})).await;
    assert_eq!(again["status"], "error", "a second start: {again}");
    let stopped = invoke(&client, "/camera/stop_streaming", json!({})).await;
    assert_eq!(stopped["status"], "completed", "{stopped}");
    assert_eq!(client.get("/camera/streaming").await.json(), false);
    let after = client.get("/camera/frames").await.json()["delivered"].clone();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        client.get("/camera/frames").await.json()["delivered"],
        after
    );
    // Streaming when the server stops: the stream stops with it.
    let started = invoke(&client, "/camera/start_streaming", json!({})).await;
    assert_eq!(started["status"], "completed", "{started}");
    client.stop().await;
    assert!(!instrument.camera.is_streaming(), "stopped with the server");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_stages_watcher_ends_with_the_server() {
    use crate::stage::StageThing;
    use tetera_core::{StagePos, Um};
    use tetera_hal::stage::Stage;
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    let thing = client
        .runtime()
        .thing_ref::<StageThing>("stage")
        .expect("the stage Thing");
    let thing = Arc::clone(teta_wot::prelude::ThingRef::thing(&thing));
    let shown = thing.shown_position();
    client.stop().await;
    instrument
        .stage
        .move_to(StagePos::new(Um(2_200.0), Um(2_000.0), Um(1_002.0)), None)
        .unwrap();
    instrument.stage.wait(Duration::from_secs(10)).unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        thing.shown_position(),
        shown,
        "nothing watches once it stops"
    );
}

#[test]
fn a_streams_frames_and_drops_are_counted() {
    use crate::camera::CameraThing;
    use tetera_core::frame::pool::HostAllocator;
    use tetera_core::{FrameLayout, FramePool, PixelFormat};
    let thing = Arc::new(CameraThing::default());
    let sink = thing.sink_for_tests();
    let layout = FrameLayout::new(PixelFormat::Mono8, 4, 4).unwrap();
    let pool = FramePool::new(layout, 2, &HostAllocator).unwrap();
    for (seq, dropped) in [(0, 0), (3, 2), (7, 3)] {
        let mut frame = pool.try_take().expect("a buffer");
        frame.meta_mut().seq = seq;
        frame.meta_mut().dropped_before = dropped;
        teta_wot_sink_deliver(&*sink, frame.freeze());
    }
    let count = thing.frame_count();
    assert_eq!((count.delivered, count.dropped), (3, 5));
    assert_eq!(thing.last_seq(), Some(7));
    assert_eq!(pool.stats().in_use, 0, "the bus keeps no frame");
}

fn teta_wot_sink_deliver(sink: &dyn tetera_hal::camera::FrameSink, frame: tetera_core::Frame) {
    sink.deliver(frame);
}

#[tokio::test]
async fn the_objective_fitted_is_said_by_name() {
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    let names: Vec<String> = client
        .get("/objective/catalogue")
        .await
        .json()
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, ["5x", "10x", "20x", "50x"]);
    assert_eq!(client.get("/objective/fitted").await.json(), "20x");
    assert_eq!(
        client
            .put_json("/objective/fitted", &json!("50x"))
            .await
            .status,
        201
    );
    assert_eq!(
        client.get("/objective/objective").await.json()["magnification"],
        50.0
    );
    assert_eq!(
        instrument.camera.capabilities().model,
        "Tetera virtual MIChrome 20"
    );
    assert_eq!(
        client
            .put_json("/objective/fitted", &json!("7x"))
            .await
            .status,
        422
    );
    assert_eq!(
        client.get("/objective/fitted").await.json(),
        "50x",
        "unchanged"
    );
    // teta-wot takes a `null` body as missing, as FastAPI does: `remove` says none is.
    assert_eq!(
        client
            .put_json("/objective/fitted", &Value::Null)
            .await
            .status,
        422
    );
    let removed = invoke(&client, "/objective/remove", json!({})).await;
    assert_eq!(removed["status"], "completed", "{removed}");
    assert_eq!(client.get("/objective/fitted").await.json(), Value::Null);
    assert_eq!(client.get("/objective/objective").await.json(), Value::Null);
    client.stop().await;
}

#[tokio::test]
async fn the_light_is_turned_dimmed_and_warmed() {
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    assert_eq!(client.get("/illumination/on").await.json(), true);
    assert_eq!(
        client.get("/illumination/intensity").await.json(),
        json!(1.0)
    );
    assert_eq!(
        client
            .put_json("/illumination/intensity", &json!(0.25))
            .await
            .status,
        201
    );
    assert_eq!(
        client.get("/illumination/intensity").await.json(),
        json!(0.25)
    );
    assert!((instrument.light.light().intensity - 250.0).abs() < 1e-3);
    assert_eq!(
        client
            .put_json("/illumination/intensity", &json!(1.5))
            .await
            .status,
        422
    );
    assert_eq!(
        client
            .put_json("/illumination/colour_temperature", &json!(5_000.0))
            .await
            .status,
        201
    );
    assert_eq!(
        client.get("/illumination/colour_temperature").await.json(),
        json!(5_000.0)
    );
    assert_eq!(
        client
            .put_json("/illumination/colour_temperature", &json!(9_000.0))
            .await
            .status,
        422
    );
    assert_eq!(
        client
            .put_json("/illumination/on", &json!(false))
            .await
            .status,
        201
    );
    assert_eq!(instrument.light.light().intensity, 0.0);
    assert_eq!(client.get("/illumination/on").await.json(), false);
    let caps = client.get("/illumination/capabilities").await.json();
    assert_eq!(caps["channels"][0]["name"], "transmitted", "{caps}");
    client.stop().await;
}

/// Publishes `n` frames into `instrument`'s bus, 10 ms apart, each exposed now.
fn publish_frames(instrument: &Instrument, n: u64) {
    use tetera_core::frame::format::BayerPattern;
    use tetera_core::frame::pool::HostAllocator;
    use tetera_core::{FrameLayout, FramePool, PixelFormat};
    let format = PixelFormat::Bayer8 {
        pattern: BayerPattern::Rggb,
    };
    let pool = FramePool::new(
        FrameLayout::new(format, 256, 192).unwrap(),
        8,
        &HostAllocator,
    )
    .unwrap();
    for seq in 1..=n {
        let mut f = pool.take(Duration::from_secs(5)).unwrap();
        f.data_mut().fill(if seq % 2 == 0 { 255 } else { 90 });
        let now = instrument.clock.now();
        let meta = f.meta_mut();
        meta.seq = seq;
        meta.exposure_start = now;
        meta.exposure_end = now;
        instrument.bus.publish(f.freeze());
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_preview_plays_as_mjpeg_and_bad_queries_are_refused() {
    use futures_util::StreamExt;
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    let response = client
        .stream("/api/v1/preview/mjpeg?width=64&height=64&quality=60")
        .await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["content-type"],
        "multipart/x-mixed-replace; boundary=tetera-frame"
    );
    let publisher = {
        let instrument = instrument.clone();
        std::thread::spawn(move || publish_frames(&instrument, 30))
    };
    let mut body = response.into_body().into_data_stream();
    let mut parts = Vec::new();
    while parts.len() < 3 {
        let chunk = tokio::time::timeout(Duration::from_secs(10), body.next())
            .await
            .expect("a part within 10 s")
            .expect("the stream goes on")
            .unwrap();
        parts.push(chunk);
    }
    for part in &parts {
        let text = String::from_utf8_lossy(&part[..part.len().min(80)]).to_string();
        assert!(
            text.starts_with("--tetera-frame\r\nContent-Type: image/jpeg\r\nContent-Length: "),
            "{text}"
        );
        let head_end = part.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let length: usize = text
            ["--tetera-frame\r\nContent-Type: image/jpeg\r\nContent-Length: ".len()..]
            .split("\r\n")
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(part.len(), head_end + length + 2, "the JPEG, then CRLF");
        let jpeg = tetera_imaging::jpeg::decode_rgb(&part[head_end..head_end + length]).unwrap();
        assert_eq!((jpeg.width(), jpeg.height()), (64, 48));
    }
    publisher.join().unwrap();
    for bad in [
        "zoom=2",
        "quality=0",
        "width=8",
        "height=9000",
        "max_rate=0",
    ] {
        let r = client.stream(&format!("/api/v1/preview/mjpeg?{bad}")).await;
        assert_eq!(r.status(), 400, "{bad}");
    }
    // Its end with the server is tested over a real socket, in `preview::tests`: the
    // in-process client has no shutdown.
    drop(body);
    client.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_live_statistics_and_the_page_are_served() {
    let scratch = Scratch::new();
    let (client, instrument) = client(&scratch).await;
    assert_eq!(client.get("/preview/statistics").await.json(), Value::Null);
    let publisher = {
        let instrument = instrument.clone();
        std::thread::spawn(move || publish_frames(&instrument, 30))
    };
    publisher.join().unwrap();
    let mut stats = Value::Null;
    for _ in 0..200 {
        stats = client.get("/preview/statistics").await.json();
        if !stats.is_null() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        stats["histogram"][1].as_array().map(Vec::len),
        Some(256),
        "{stats}"
    );
    assert_eq!(stats["mask"], json!([48, 32]));
    assert!(stats["seq"].as_u64().is_some_and(|s| s >= 1));
    let page = client.get("/live-view").await;
    assert_eq!(page.status, 200);
    assert!(
        page.header("content-type")
            .is_some_and(|t| t.starts_with("text/html"))
    );
    let html = page.text();
    assert!(
        html.contains("/preview/stream") && html.contains("TPV1") && html.contains("/stage/jog")
    );
    // Nothing works them out once the server stops.
    let thing = client
        .runtime()
        .thing_ref::<crate::preview::PreviewThing>("preview")
        .expect("the preview Thing");
    let thing = Arc::clone(teta_wot::prelude::ThingRef::thing(&thing));
    assert!(thing.statistics_running());
    client.stop().await;
    assert!(
        !thing.statistics_running(),
        "the statistics' thread stopped"
    );
    let before = thing.statistics_seq();
    publish_frames(&instrument, 30);
    assert_eq!(thing.statistics_seq(), before, "nothing new");
}
