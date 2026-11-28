//! Capture with the virtual camera on the CPU: a frame, a series and a burst, each in the
//! capture's mode and back; cancelled ones leaving nothing; bad requests refused.

use crate::{CaptureKind, CaptureRequest, Rig, capture};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tetera_acquire::FrameBus;
use tetera_core::frame::meta::LightState;
use tetera_core::{Error, ErrorKind, Result, StagePos, Um};
use tetera_hal::camera::{Camera, CameraConfig};
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
        Rig {
            camera: self.camera.as_ref(),
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

#[test]
fn a_single_frame_is_taken_at_full_resolution_and_the_live_view_comes_back() {
    let s = setup();
    let live = s.camera.config();
    s.camera.start(s.bus.clone()).unwrap();
    let mut steps = Vec::new();
    let r = capture(&s.rig(), &request(CaptureKind::Single), &mut |done, msg| {
        steps.push((done, msg.to_string()));
        Ok(())
    })
    .unwrap();
    // Back as it was: the binned mode, streaming.
    assert_eq!(s.camera.config(), live);
    assert!(s.camera.is_streaming());
    s.camera.stop().unwrap();
    assert!(r.switch_ms > 0.0 && r.back_ms > 0.0, "{r:?}");
    assert_eq!((r.dataset.frames, r.missing), (1, 0));
    let read = s.store.read(&r.dataset.id).unwrap();
    let spec = &read.metadata().spec;
    assert_eq!(
        (spec.width, spec.height),
        (192, 128),
        "full resolution, not 96 × 64"
    );
    assert_eq!(spec.format.bytes_per_sample(), 2, "12 bits in 16");
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
    let r = capture(&s.rig(), &request(series), &mut going).unwrap();
    let frames = s
        .store
        .read(&r.dataset.id)
        .unwrap()
        .metadata()
        .frames
        .clone();
    assert_eq!(frames.len(), 3);
    for pair in frames.windows(2) {
        let apart = pair[1]
            .exposure_start
            .saturating_since(pair[0].exposure_start);
        assert!(apart >= Duration::from_millis(140), "{apart:?}");
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
    // At 8 bits and its own exposure, when asked.
    let mut asked = request(CaptureKind::Single);
    asked.bit_depth = Some(8);
    asked.exposure_ms = Some(2.5);
    let r = capture(&s.rig(), &asked, &mut going).unwrap();
    let read = s.store.read(&r.dataset.id).unwrap();
    let format = read.metadata().spec.format;
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
        move |done: f64, _: &str| -> Result<()> {
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
    // A mode the camera hasn't: refused by the camera, and put back.
    let mut nine = request(CaptureKind::Single);
    nine.mode = Some(9);
    assert!(capture(&s.rig(), &nine, &mut going).is_err());
    assert_eq!(s.camera.config(), before);
    // An unknown project: refused, and put back.
    let mut nowhere = request(CaptureKind::Single);
    nowhere.project = "none".into();
    assert_eq!(
        capture(&s.rig(), &nowhere, &mut going).unwrap_err().kind(),
        ErrorKind::NotFound
    );
    assert_eq!(s.camera.config(), before);
    assert_eq!(CaptureKind::Single.frames(), 1);
}
