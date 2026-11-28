//! Capture with the virtual camera on the CPU: a frame, a series and a burst, each in the
//! capture's mode and back; cancelled ones leaving nothing; bad requests refused.

use crate::{CaptureKind, CaptureRequest, Rig, capture};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tetera_acquire::FrameBus;
use tetera_core::frame::format::Binning;
use tetera_core::frame::meta::LightState;
use tetera_core::{Error, ErrorKind, Rect, Result, StagePos, Um};
use tetera_hal::camera::{Camera, CameraConfig};
use tetera_hal::fake::{CameraFault, FakeCamera};
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue, ObjectiveMount};
use tetera_sim::camera::{Backend, Scene, VirtualCamera, VirtualCameraConfig};
use tetera_sim::light::CHANNEL;
use tetera_sim::optics::Sensor;
use tetera_sim::{
    HeSample, Rate, SectionConfig, SimClock, VirtualIllumination, VirtualStage, VirtualStageConfig,
};
use tetera_store::Store;

/// A folder of the test's own, removed when it drops.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("tetera-workflows-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(dir.join("scratch")).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The virtual camera over H&E at 20×, on the CPU, 192 × 128, in real time; in its binned live
/// mode at 8 bits.
fn camera() -> Arc<VirtualCamera> {
    let clock = Arc::new(SimClock::new(Rate::RealTime).unwrap());
    let stage = Arc::new(
        VirtualStage::new(
            clock,
            VirtualStageConfig {
                start: StagePos::new(Um(2_000.0), Um(2_000.0), Um(1_002.0)),
                ..VirtualStageConfig::default()
            },
        )
        .unwrap(),
    );
    let mount = ManualObjectiveMount::new(
        ObjectiveCatalogue::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/objectives.toml"),
        )
        .unwrap(),
    );
    mount.fit(Some("20x")).unwrap();
    let light = VirtualIllumination::new(1_000.0, 0.05);
    light
        .set(
            CHANNEL,
            LightState {
                on: true,
                intensity: 1.0,
                colour_temperature_k: Some(3_200.0),
            },
        )
        .unwrap();
    let camera = VirtualCamera::new(
        Scene {
            stage,
            objectives: Arc::new(mount),
            light: Arc::new(light),
            sample: HeSample::new(7, SectionConfig::default()).unwrap(),
        },
        VirtualCameraConfig {
            sensor: Sensor {
                width: 192,
                height: 128,
                pixel: 2.4,
            },
            section_z: 1_000.0,
            ..VirtualCameraConfig::default()
        },
        Backend::Cpu,
    )
    .unwrap();
    camera
        .configure(&CameraConfig {
            mode: 1,
            bit_depth: 8,
            ..camera.config()
        })
        .unwrap();
    Arc::new(camera)
}

struct Setup {
    scratch: Scratch,
    /// Where bursts' frame logs go.
    logs: PathBuf,
    camera: Arc<VirtualCamera>,
    bus: Arc<FrameBus>,
    store: Store,
}

fn setup() -> Setup {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0.join("data")).unwrap();
    store.create_project("p").unwrap();
    Setup {
        logs: scratch.0.join("scratch"),
        camera: camera(),
        bus: FrameBus::new(),
        store,
        scratch,
    }
}

impl Setup {
    fn rig(&self) -> Rig<'_> {
        self.rig_for(self.camera.as_ref())
    }

    fn rig_for<'a>(&'a self, camera: &'a dyn Camera) -> Rig<'a> {
        Rig {
            camera,
            bus: &self.bus,
            store: &self.store,
            scratch: &self.logs,
            live: None,
            pixel_size_um: 0.24,
            instrument: serde_json::json!({"objective": "20x"}),
        }
    }

    fn scratch_left(&self) -> usize {
        std::fs::read_dir(self.scratch.0.join("scratch"))
            .unwrap()
            .count()
    }
}

fn request(kind: CaptureKind) -> CaptureRequest {
    CaptureRequest {
        project: "p".into(),
        name: "a capture".into(),
        kind,
        mode: None,
        bit_depth: None,
        exposure_ms: None,
    }
}

fn going(_: f64, _: &str) -> Result<()> {
    Ok(())
}

/// A step that fails once `limit` has passed: a wait gone wrong ends the test there, not at
/// its timeout.
fn within(limit: Duration) -> impl FnMut(f64, &str) -> Result<()> {
    let began = Instant::now();
    move |_, _| {
        if began.elapsed() > limit {
            Err(Error::timeout(format!("the capture took over {limit:?}")))
        } else {
            Ok(())
        }
    }
}

/// The fake camera, 640 × 480 at up to 100 frames a second, quick to snap; its live view a
/// region of its small mode, binned 2 × 2, at 8 bits.
fn fake() -> FakeCamera {
    let camera = FakeCamera::new(Arc::new(SimClock::new(Rate::RealTime).unwrap()), 100.0);
    camera
        .configure(&CameraConfig {
            mode: 1,
            bit_depth: 8,
            roi: Some(Rect::new(0, 0, 80, 60)),
            binning: Binning { x: 2, y: 2 },
            ..camera.config()
        })
        .unwrap();
    camera
}

/// The first number in `text`.
fn number(text: &str) -> f64 {
    text.split(|c: char| !c.is_ascii_digit())
        .find(|w| !w.is_empty())
        .and_then(|w| w.parse().ok())
        .unwrap_or_else(|| panic!("no number in {text:?}"))
}

#[test]
fn a_single_frame_is_taken_at_full_resolution_and_the_live_view_comes_back() {
    let s = setup();
    let live = s.camera.config();
    s.camera.start(s.bus.clone()).unwrap();
    let mut steps = Vec::new();
    let began = Instant::now();
    let r = capture(&s.rig(), &request(CaptureKind::Single), &mut |done, msg| {
        steps.push((done, msg.to_string()));
        Ok(())
    })
    .unwrap();
    let took_ms = began.elapsed().as_secs_f64() * 1e3;
    // Back as it was: the binned mode, streaming.
    assert_eq!(s.camera.config(), live);
    assert!(s.camera.is_streaming());
    s.camera.stop().unwrap();
    // The switch snaps a frame of mode 0, 138 ms; the switch back restarts the stream.
    assert!(r.switch_ms > 10.0 && r.back_ms > 0.001, "{r:?}");
    assert!(r.switch_ms + r.back_ms < took_ms, "{r:?} in {took_ms} ms");
    assert_eq!((r.dataset.frames, r.missing), (1, 0));
    let read = s.store.read(&r.dataset.id).unwrap();
    let spec = &read.metadata().spec;
    assert_eq!(
        (spec.width, spec.height),
        (192, 128),
        "full resolution, not 96 × 64"
    );
    assert_eq!(
        (
            spec.format.significant_bits(),
            spec.format.bytes_per_sample()
        ),
        (Some(12), 2),
        "12 bits in 16"
    );
    assert_eq!(spec.pixel_size_um, 0.24);
    assert_eq!(spec.instrument["camera"]["mode"], 0);
    assert_eq!(spec.instrument["instrument"]["objective"], "20x");
    assert_eq!(steps.first().map(|s| s.0), Some(0.0));
    assert_eq!(steps.last().map(|s| s.0), Some(1.0));
}

#[test]
fn a_series_keeps_its_interval_and_a_burst_its_rate_and_count() {
    let s = setup();
    let series = CaptureKind::Series {
        count: 3,
        interval_ms: 150.0,
    };
    let r = capture(
        &s.rig(),
        &request(series),
        &mut within(Duration::from_secs(20)),
    )
    .unwrap();
    let frames = s
        .store
        .read(&r.dataset.id)
        .unwrap()
        .metadata()
        .frames
        .clone();
    assert_eq!(frames.len(), 3);
    // Each frame is due an interval after the one before, counted from the first: never
    // early, though one late on a busy CPU lets the next catch up.
    for (i, f) in frames.iter().enumerate() {
        let since = f.exposure_start.saturating_since(frames[0].exposure_start);
        assert!(
            since >= Duration::from_millis(150 * i as u64 - 5 * i as u64),
            "{i}: {since:?}"
        );
    }
    assert_eq!(
        s.store
            .read(&r.dataset.id)
            .unwrap()
            .metadata()
            .spec
            .interval_s,
        0.15
    );
    // A burst: every frame, in order, at the camera's rate, its frame log removed.
    let r = capture(
        &s.rig(),
        &request(CaptureKind::Burst { count: 5 }),
        &mut going,
    )
    .unwrap();
    assert_eq!(r.dataset.frames, 5);
    let frames = s
        .store
        .read(&r.dataset.id)
        .unwrap()
        .metadata()
        .frames
        .clone();
    // In order; any gap in their numbers the camera's own drop, told in the next frame, as the
    // simulated camera can fall behind on a busy CPU. The recorder lost none.
    assert!(
        frames
            .windows(2)
            .all(|w| w[1].seq - w[0].seq - 1 == w[1].dropped_before),
        "{:?}",
        frames
            .iter()
            .map(|f| (f.seq, f.dropped_before))
            .collect::<Vec<_>>()
    );
    let told: u64 = frames.iter().skip(1).map(|f| f.dropped_before).sum();
    assert_eq!(r.missing, told);
    assert_eq!(s.scratch_left(), 0, "the frame log removed");
    assert!(!s.camera.is_streaming(), "it wasn't streaming before");
    // In the binned mode, at 8 bits and its own exposure, when asked.
    let mut asked = request(CaptureKind::Single);
    asked.mode = Some(1);
    asked.bit_depth = Some(8);
    asked.exposure_ms = Some(2.5);
    let r = capture(&s.rig(), &asked, &mut going).unwrap();
    let read = s.store.read(&r.dataset.id).unwrap();
    let spec = &read.metadata().spec;
    assert_eq!((spec.width, spec.pixel_size_um), (96, 0.48), "binned 2 × 2");
    let format = spec.format;
    // 8 bits in 16, as the camera lays them out.
    assert_eq!(
        (format.significant_bits(), format.bytes_per_sample()),
        (Some(8), 2)
    );
    assert_eq!(read.metadata().spec.instrument["camera"]["exposure"], 2.5);
}

#[test]
fn a_cancelled_capture_leaves_no_dataset_and_the_camera_as_it_was() {
    let s = setup();
    let live = s.camera.config();
    s.camera.start(s.bus.clone()).unwrap();
    let stop_at = |at: f64| {
        let mut limit = within(Duration::from_secs(20));
        move |done: f64, msg: &str| -> Result<()> {
            limit(done, msg)?;
            if done >= at {
                Err(Error::cancelled("cancelled"))
            } else {
                Ok(())
            }
        }
    };
    for (kind, at) in [
        (CaptureKind::Burst { count: 8 }, 0.2),
        (CaptureKind::Burst { count: 8 }, 0.75),
        (
            CaptureKind::Series {
                count: 4,
                interval_ms: 100.0,
            },
            0.5,
        ),
    ] {
        let e = capture(&s.rig(), &request(kind), &mut stop_at(at)).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Cancelled, "{kind:?}");
        assert!(s.store.datasets("p").unwrap().is_empty(), "{kind:?}");
        assert_eq!(s.scratch_left(), 0, "no frame log left: {kind:?}");
        let folders = std::fs::read_dir(s.scratch.0.join("data/projects/p"))
            .unwrap()
            .count();
        assert_eq!(folders, 0, "no dataset's folder left: {kind:?}");
        assert_eq!(s.camera.config(), live);
        assert!(s.camera.is_streaming(), "the live view back: {kind:?}");
    }
    s.camera.stop().unwrap();
}

#[test]
fn bad_requests_are_refused_and_a_failed_switch_puts_the_camera_back() {
    let s = setup();
    let before = s.camera.config();
    for kind in [
        CaptureKind::Burst { count: 0 },
        CaptureKind::Series {
            count: 0,
            interval_ms: 10.0,
        },
        CaptureKind::Series {
            count: 2,
            interval_ms: -1.0,
        },
        CaptureKind::Series {
            count: 2,
            interval_ms: f64::NAN,
        },
    ] {
        let e = capture(&s.rig(), &request(kind), &mut going).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{kind:?}");
    }
    let mut zero = request(CaptureKind::Single);
    zero.exposure_ms = Some(0.0);
    assert!(capture(&s.rig(), &zero, &mut going).is_err());
    for (ms, fine) in [
        (Some(0.0), false),
        (Some(-1.0), false),
        (Some(f64::INFINITY), false),
        (Some(f64::NAN), false),
        (Some(0.001), true),
        (None, true),
    ] {
        let mut r = request(CaptureKind::Single);
        r.exposure_ms = ms;
        assert_eq!(r.check().is_ok(), fine, "{ms:?}");
    }
    let mut r = request(CaptureKind::Series {
        count: 1,
        interval_ms: 0.0,
    });
    assert!(r.check().is_ok(), "an interval of 0");
    r.kind = CaptureKind::Burst { count: 1 };
    assert!(r.check().is_ok());
    // A mode the camera hasn't: refused by the camera, and put back.
    let mut nine = request(CaptureKind::Single);
    nine.mode = Some(9);
    assert!(capture(&s.rig(), &nine, &mut going).is_err());
    assert_eq!(s.camera.config(), before);
    // An unknown project: refused, and put back.
    let published = s.bus.published();
    let mut nowhere = request(CaptureKind::Burst { count: 3 });
    nowhere.project = "none".into();
    assert_eq!(
        capture(&s.rig(), &nowhere, &mut going).unwrap_err().kind(),
        ErrorKind::NotFound
    );
    assert_eq!(s.camera.config(), before);
    assert_eq!(s.bus.published(), published, "nothing recorded");
    assert_eq!(CaptureKind::Single.frames(), 1);
}

#[test]
fn the_capture_mode_is_the_whole_sensor_at_12_bits_whatever_the_live_view_set() {
    let s = setup();
    let camera = fake();
    let live = camera.config();
    let r = capture(
        &s.rig_for(&camera),
        &request(CaptureKind::Single),
        &mut going,
    )
    .unwrap();
    let read = s.store.read(&r.dataset.id).unwrap();
    let spec = &read.metadata().spec;
    assert_eq!(
        (spec.width, spec.height),
        (640, 480),
        "no region, no binning"
    );
    assert_eq!(spec.format.significant_bits(), Some(12));
    assert_eq!(spec.instrument["camera"]["roi"], serde_json::Value::Null);
    assert_eq!(camera.config(), live, "the live view's set back");
    // The camera refuses the switch: put back all the same.
    camera.inject(CameraFault::FailConfigure);
    let e = capture(
        &s.rig_for(&camera),
        &request(CaptureKind::Single),
        &mut going,
    )
    .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Hardware, "{e}");
    assert_eq!(camera.config(), live);
}

#[test]
fn a_series_waits_out_its_interval_and_says_how_far_it_has_got() {
    let s = setup();
    let camera = fake();
    let began = Instant::now();
    let mut steps = Vec::new();
    let series = CaptureKind::Series {
        count: 4,
        interval_ms: 150.0,
    };
    let mut first_step = None;
    let r = capture(&s.rig_for(&camera), &request(series), &mut |done, msg| {
        // A wait gone wrong fails here, not at the test's timeout.
        if began.elapsed() > Duration::from_secs(10) {
            return Err(Error::timeout("the series took over 10 s"));
        }
        first_step.get_or_insert_with(|| began.elapsed());
        steps.push((done, msg.to_string()));
        Ok(())
    })
    .unwrap();
    let took = began.elapsed();
    // The fake camera switches in a snap: within the time to the first frame, and the switch
    // back within the whole.
    let first_ms = first_step.unwrap().as_secs_f64() * 1e3;
    assert!(
        r.switch_ms > 0.0 && r.switch_ms <= first_ms,
        "{r:?}, the first step at {first_ms} ms"
    );
    assert!(
        r.back_ms > 0.0 && r.back_ms <= took.as_secs_f64() * 1e3 - first_ms,
        "{r:?}"
    );
    let frames = s
        .store
        .read(&r.dataset.id)
        .unwrap()
        .metadata()
        .frames
        .clone();
    assert_eq!(frames.len(), 4);
    // Never early, counted from the first; not far behind, on the quick fake camera.
    for (i, f) in frames.iter().enumerate() {
        let since = f.exposure_start.saturating_since(frames[0].exposure_start);
        let due = Duration::from_millis(150 * i as u64);
        assert!(
            since + Duration::from_millis(5) >= due && since <= due + Duration::from_millis(400),
            "{i}: {since:?}"
        );
    }
    // Each frame, and each wait for it, at the frames before it over all.
    for (done, msg) in &steps[..steps.len() - 1] {
        assert!(msg.starts_with("frame "), "{msg}");
        assert_eq!(*done, (number(msg) - 1.0) / 4.0, "{msg}");
    }
    assert!(steps.iter().any(|(_, m)| m.ends_with("due")), "it waited");
    assert_eq!(steps.last(), Some(&(1.0, "written".to_string())));
}

#[test]
fn a_bursts_drops_are_counted_once_and_its_progress_said_as_it_records_and_ingests() {
    let s = setup();
    let camera = fake();
    // The stream's first two frames lost, and said to be.
    camera.inject(CameraFault::DropFrames(2));
    let mut steps = Vec::new();
    let r = capture(
        &s.rig_for(&camera),
        &request(CaptureKind::Burst { count: 6 }),
        &mut |done, msg| {
            steps.push((done, msg.to_string()));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!((r.dataset.frames, r.missing), (6, 2));
    // Recording: the frames published so far, over all, makes half; ingest the other half.
    for (done, msg) in &steps {
        let n = number(msg);
        if msg.starts_with("ingesting frame ") {
            assert_eq!(*done, 0.5 + 0.5 * n / 6.0, "{msg}");
        } else {
            assert!(msg.starts_with("frame "), "{msg}");
            assert_eq!(*done, 0.5 * n / 6.0, "{msg}");
        }
    }
    assert!(steps.iter().any(|(_, m)| m.starts_with("ingesting")));
    assert_eq!(steps.last().map(|s| s.0), Some(1.0));
    assert_eq!(s.scratch_left(), 0);
}
