//! Scans: the planners' coverage and order, the moves' times against the simulator's own,
//! the estimate; and the engine on a small virtual instrument on the CPU: positions, the
//! settle, pause, cancel at every step, autofocus, and refusals.

use super::*;
use crate::autofocus::depth_of_field;
use crate::tests::Scratch;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tetera_acquire::FrameBus;
use tetera_core::frame::meta::LightState;
use tetera_core::{ErrorKind, Timestamp};
use tetera_hal::camera::{Camera, CameraConfig};
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue, ObjectiveMount};
use tetera_hal::stage::Stage;
use tetera_sim::camera::{Backend, Scene, VirtualCamera, VirtualCameraConfig};
use tetera_sim::light::CHANNEL;
use tetera_sim::optics::Sensor;
use tetera_sim::{
    HeSample, Profile, Rate, SectionConfig, SimClock, VirtualIllumination, VirtualStage,
    VirtualStageConfig,
};
use tetera_store::Store;

fn at(x: f64, y: f64, z: f64) -> StagePos {
    StagePos::new(Um(x), Um(y), Um(z))
}

fn caps() -> StageCapabilities {
    VirtualStageConfig::default().capabilities()
}

fn snake(x0: f64, y0: f64, x1: f64, y1: f64, overlap: f64) -> Pattern {
    Pattern::Snake {
        x0,
        y0,
        x1,
        y1,
        overlap,
    }
}

#[test]
fn a_snake_covers_its_rectangle_centred_in_serpentine_order() {
    // A field of 10 × 8 µm at 20 % overlap: steps of 8 and 6.4.
    let (p, grid) = positions(
        &snake(100.0, 200.0, 126.0, 214.0, 0.2),
        at(0.0, 0.0, 50.0),
        [10.0, 8.0],
        &caps(),
    )
    .unwrap();
    assert_eq!(grid, [3, 2]);
    // Across: 3 tiles cover 10 + 2 × 8 = 26, exactly the span; down: 2 cover 14.4 of 14.
    let xs = [105.0, 113.0, 121.0];
    let y0 = 200.0 + 4.0 - (14.4 - 14.0) / 2.0;
    let expected: Vec<StagePos> = [0, 1, 2, 2, 1, 0]
        .iter()
        .zip([0, 0, 0, 1, 1, 1])
        .map(|(&i, j)| at(xs[i], y0 + f64::from(j) * 6.4, 50.0))
        .collect();
    for (got, want) in p.iter().zip(&expected) {
        assert!(
            (got.x.um() - want.x.um()).abs() < 1e-9 && (got.y.um() - want.y.um()).abs() < 1e-9,
            "{got:?} {want:?}"
        );
        assert_eq!(got.z, want.z);
    }
    // The corners either way round: the same.
    let (q, _) = positions(
        &snake(126.0, 214.0, 100.0, 200.0, 0.2),
        at(0.0, 0.0, 50.0),
        [10.0, 8.0],
        &caps(),
    )
    .unwrap();
    assert_eq!(p, q);
    // Covered to its edges, and centred: as much over on each side.
    let left = p.iter().map(|t| t.x.um() - 5.0).fold(f64::MAX, f64::min);
    let right = p.iter().map(|t| t.x.um() + 5.0).fold(f64::MIN, f64::max);
    let top = p.iter().map(|t| t.y.um() - 4.0).fold(f64::MAX, f64::min);
    let bottom = p.iter().map(|t| t.y.um() + 4.0).fold(f64::MIN, f64::max);
    assert!(left <= 100.0 + 1e-9 && right >= 126.0 - 1e-9);
    assert!(top <= 200.0 && bottom >= 214.0);
    assert!(((200.0 - top) - (bottom - 214.0)).abs() < 1e-9);
    // Smaller than the field: one tile, at its middle.
    let (one, grid) = positions(
        &snake(10.0, 10.0, 14.0, 12.0, 0.1),
        at(0.0, 0.0, 1.0),
        [10.0, 8.0],
        &caps(),
    )
    .unwrap();
    assert_eq!((one, grid), (vec![at(12.0, 11.0, 1.0)], [1, 1]));
    // Just over a whole step: one more tile.
    let (_, grid) = positions(
        &snake(100.0, 200.0, 126.1, 208.0, 0.2),
        at(0.0, 0.0, 0.0),
        [10.0, 8.0],
        &caps(),
    )
    .unwrap();
    assert_eq!(grid, [4, 1]);
}

#[test]
fn a_z_scan_rises_by_range_or_around_the_current_z() {
    let here = at(7.0, 8.0, 10.0);
    let (p, grid) = positions(
        &Pattern::ZRange {
            from: 5.0,
            to: 1.0,
            step: 1.0,
        },
        here,
        [1.0, 1.0],
        &caps(),
    )
    .unwrap();
    let zs: Vec<f64> = p.iter().map(|q| q.z.um()).collect();
    assert_eq!((zs, grid), (vec![1.0, 2.0, 3.0, 4.0, 5.0], [1, 5]));
    assert!(p.iter().all(|q| q.x == here.x && q.y == here.y));
    let (p, _) = positions(
        &Pattern::ZRange {
            from: 0.0,
            to: 2.5,
            step: 1.0,
        },
        here,
        [1.0, 1.0],
        &caps(),
    )
    .unwrap();
    assert_eq!(p.len(), 3, "a step that doesn't divide the range: up to it");
    let (p, _) = positions(
        &Pattern::ZRange {
            from: 0.0,
            to: 2.0,
            step: 0.5,
        },
        here,
        [1.0, 1.0],
        &caps(),
    )
    .unwrap();
    let zs: Vec<f64> = p.iter().map(|q| q.z.um()).collect();
    assert_eq!(zs, [0.0, 0.5, 1.0, 1.5, 2.0]);
    let (p, grid) = positions(
        &Pattern::ZAround {
            planes: 4,
            step: 2.0,
        },
        here,
        [1.0, 1.0],
        &caps(),
    )
    .unwrap();
    let zs: Vec<f64> = p.iter().map(|q| q.z.um()).collect();
    assert_eq!((zs, grid), (vec![7.0, 9.0, 11.0, 13.0], [1, 4]));
}

#[test]
fn bad_patterns_and_positions_beyond_the_travel_are_refused() {
    let here = at(1_000.0, 1_000.0, 1_000.0);
    for bad in [
        snake(0.0, 0.0, 10.0, 10.0, 0.51),
        snake(0.0, 0.0, 10.0, 10.0, -0.1),
        snake(f64::NAN, 0.0, 10.0, 10.0, 0.1),
        Pattern::ZRange {
            from: 0.0,
            to: 1.0,
            step: 0.0,
        },
        Pattern::ZRange {
            from: 0.0,
            to: f64::INFINITY,
            step: 1.0,
        },
        Pattern::ZAround {
            planes: 0,
            step: 1.0,
        },
        Pattern::ZAround {
            planes: 3,
            step: -1.0,
        },
        Pattern::ZAround {
            planes: 3,
            step: 0.0,
        },
    ] {
        let e = positions(&bad, here, [10.0, 8.0], &caps()).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{bad:?}");
    }
    for field in [[0.0, 8.0], [10.0, 0.0]] {
        assert!(positions(&snake(0.0, 0.0, 1.0, 1.0, 0.1), here, field, &caps()).is_err());
    }
    // The stage's travel: x up to 100 mm.
    assert!(
        positions(
            &snake(99_990.0, 0.0, 100_020.0, 10.0, 0.0),
            here,
            [10.0, 8.0],
            &caps()
        )
        .is_err()
    );
    // Both ends of the overlap are fine.
    for overlap in [0.0, 0.5] {
        assert!(
            positions(
                &snake(10.0, 10.0, 40.0, 40.0, overlap),
                here,
                [10.0, 8.0],
                &caps()
            )
            .is_ok()
        );
    }
}

#[test]
fn a_moves_time_is_the_simulated_stages_own() {
    let config = VirtualStageConfig::default();
    let limits = [
        config.axes[0].limits,
        config.axes[1].limits,
        config.axes[2].limits,
    ];
    let from = at(1_000.0, 1_000.0, 1_000.0);
    for to in [
        at(30_000.0, 1_000.0, 1_000.0),
        at(1_100.0, 1_000.0, 1_000.0),
        at(5_000.0, 4_000.0, 1_000.0),
        at(1_000.0, 1_000.0, 1_007.9),
        at(1_000.0, 1_000.0, 3_000.0),
        at(1_200.0, 900.0, 1_010.0),
        at(500.0, 1_000.0, 1_000.0),
        at(400.0, 300.0, 990.0),
    ] {
        let theirs = Profile::new(from, to, None, &limits)
            .unwrap()
            .duration()
            .as_secs_f64();
        let ours = move_time(from, to, &caps()).unwrap();
        assert!(
            (ours - theirs).abs() < 1e-6,
            "{to:?}: {ours} against {theirs}"
        );
    }
    assert_eq!(move_time(from, from, &caps()).unwrap(), 0.0);
}

#[test]
fn the_estimate_adds_each_position_or_its_write_and_the_last_write() {
    let timing = Timing {
        report_ms: 10.0,
        settle_ms: 50.0,
        frame_ms: 100.0,
        write_ms: 120.0,
        switch_ms: 500.0,
        autofocus_ms: 2_000.0,
    };
    let start = at(1_000.0, 1_000.0, 1_000.0);
    let p = [at(1_100.0, 1_000.0, 1_000.0), at(1_100.0, 1_100.0, 1_000.0)];
    let m0 = move_time(start, p[0], &caps()).unwrap() * 1e3;
    let m1 = move_time(p[0], p[1], &caps()).unwrap() * 1e3;
    let tile = |m: f64| (m + 10.0 + 50.0 + 100.0).max(120.0);
    let want = (500.0 + tile(m0) + tile(m1) + 120.0) / 1e3;
    let got = estimate(&p, start, &caps(), &timing, |_| false).unwrap();
    assert!((got - want).abs() < 1e-9, "{got} {want}");
    // An autofocus at the first.
    let focused = estimate(&p, start, &caps(), &timing, |i| i == 0).unwrap();
    assert!((focused - (want + 2.0)).abs() < 1e-9, "{focused}");
    // A slow disk: each tile its write.
    let slow = Timing {
        write_ms: 1_000.0,
        ..timing
    };
    let s = estimate(&p, start, &caps(), &slow, |_| false).unwrap();
    assert!((s - 3.5).abs() < 1e-9, "{s}");
    assert_eq!(
        estimate(&[], start, &caps(), &timing, |_| false).unwrap(),
        0.5
    );
}

#[test]
fn autofocus_is_due_as_its_schedule_says() {
    assert!(!(0..5).any(|i| AutofocusAt::Never.at(i)));
    assert_eq!(
        (0..4).map(|i| AutofocusAt::First.at(i)).collect::<Vec<_>>(),
        [true, false, false, false]
    );
    assert_eq!(
        (0..7)
            .map(|i| AutofocusAt::Every { tiles: 3 }.at(i))
            .collect::<Vec<_>>(),
        [true, false, false, true, false, false, true]
    );
    assert!(!AutofocusAt::Every { tiles: 0 }.at(0));
}

/// The section's bottom; its middle, 2 µm above, is the true focus.
const FOCUS: f64 = 1_002.0;

/// A small virtual instrument at 20× on the CPU, in real time, the camera in its binned live
/// mode, and a store with a project `p`.
struct Bench {
    _scratch: Scratch,
    stage: Arc<VirtualStage>,
    mount: Arc<ManualObjectiveMount>,
    camera: VirtualCamera,
    bus: Arc<FrameBus>,
    store: Store,
}

fn bench(z: f64) -> Bench {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0.join("data")).unwrap();
    store.create_project("p").unwrap();
    let clock = Arc::new(SimClock::new(Rate::RealTime).unwrap());
    let stage = Arc::new(
        VirtualStage::new(
            clock,
            VirtualStageConfig {
                start: at(2_000.0, 2_000.0, z),
                ..VirtualStageConfig::default()
            },
        )
        .unwrap(),
    );
    let mount = Arc::new(ManualObjectiveMount::new(
        ObjectiveCatalogue::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/objectives.toml"),
        )
        .unwrap(),
    ));
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
            stage: stage.clone(),
            objectives: mount.clone(),
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
    Bench {
        _scratch: scratch,
        stage,
        mount,
        camera,
        bus: FrameBus::new(),
        store,
    }
}

impl Bench {
    fn rig(&self) -> ScanRig<'_> {
        ScanRig {
            camera: &self.camera,
            stage: self.stage.as_ref(),
            bus: &self.bus,
            store: &self.store,
            live: None,
            objective: self.mount.fitted(),
            sensor_pixel_um: 2.4,
            instrument: serde_json::json!({"simulated": true}),
        }
    }

    fn here(&self) -> StagePos {
        self.stage.reading().unwrap().position
    }
}

/// A 3 × 2 snake at 20×: its field 23.04 × 15.36 µm, 10 % overlap.
fn request(pattern: Pattern) -> ScanRequest {
    ScanRequest {
        project: "p".into(),
        name: "a scan".into(),
        pattern,
        autofocus: AutofocusAt::Never,
        settle_ms: None,
        exposure_ms: None,
        write_mb_s: None,
    }
}

fn three_by_two() -> Pattern {
    snake(2_000.0, 2_000.0, 2_064.0, 2_029.0, 0.1)
}

fn going(_: f64, _: &str) -> Result<()> {
    Ok(())
}

fn since(later: Timestamp, earlier: Timestamp) -> Duration {
    later.saturating_since(earlier)
}

#[test]
fn a_snake_is_taken_stop_and_go_its_positions_recorded_and_settled() {
    let _cpu = crate::tests::shared();
    let b = bench(FOCUS);
    let live = b.camera.config();
    let plan = preview(&b.rig(), &request(three_by_two())).unwrap();
    assert_eq!(plan.grid, [3, 2]);
    assert!((plan.field[0] - 23.04).abs() < 1e-9 && (plan.field[1] - 15.36).abs() < 1e-9);
    assert_eq!(
        b.here(),
        at(2_000.0, 2_000.0, FOCUS),
        "a preview moves nothing"
    );
    let control = ScanControl::default();
    // A settle near the stream's 138 ms interval: without it, a frame's exposure would start
    // anywhere in the interval after the stage stopped, and six would clear it only by chance.
    let settled = ScanRequest {
        settle_ms: Some(120.0),
        ..request(three_by_two())
    };
    let r = scan(&b.rig(), &settled, &control, &mut going).unwrap();
    assert_eq!((r.dataset.frames, r.tiles.len()), (6, 6));
    assert_eq!(r.plan.positions, plan.positions);
    let step = 0.3125;
    for (t, planned) in r.tiles.iter().zip(&plan.positions) {
        assert_eq!(t.planned, *planned);
        // Where the stage said: the plan, to its step; and the simulator's truth.
        for (got, want) in [
            (t.recorded.x, planned.x),
            (t.recorded.y, planned.y),
            (t.recorded.z, planned.z),
        ] {
            assert!((got.um() - want.um()).abs() <= step / 2.0 + 1e-9, "{t:?}");
        }
        assert_eq!(
            t.truth,
            Some(t.recorded),
            "the stage still through the exposure"
        );
        // The settle between the stage's report and the exposure.
        assert!(
            since(t.exposure_start, t.still_at) >= Duration::from_millis(120),
            "{t:?}"
        );
    }
    // Serpentine: the second row back.
    let xs: Vec<f64> = r.tiles.iter().map(|t| t.recorded.x.um()).collect();
    assert!(
        xs[0] < xs[1] && xs[1] < xs[2] && xs[3] > xs[4] && xs[4] > xs[5],
        "{xs:?}"
    );
    // The dataset: the tiles and their positions, the plan in its metadata.
    let read = b.store.read(&r.dataset.id).unwrap();
    let meta = read.metadata();
    assert_eq!((meta.spec.kind.as_str(), meta.spec.width), ("scan", 192));
    for (f, t) in meta.frames.iter().zip(&r.tiles) {
        assert_eq!(f.position, Some(t.recorded));
    }
    assert!(
        (meta.spec.pixel_size_um - 0.12).abs() < 1e-12,
        "2.4 µm at 20×"
    );
    assert_eq!(
        meta.spec.instrument["scan"]["plan"]["grid"],
        serde_json::json!([3, 2])
    );
    assert_eq!(b.camera.config(), live, "the camera back");
    assert!(r.seconds > 0.0 && plan.estimate_s > 0.0);
}

#[test]
fn a_scan_paused_waits_and_resumed_goes_on() {
    let _cpu = crate::tests::shared();
    let b = bench(FOCUS);
    let control = Arc::new(ScanControl::default());
    let resumed = Arc::new(std::sync::Mutex::new(None::<Timestamp>));
    let mut paused_seen = 0;
    let mut resuming = None;
    let r = scan(
        &b.rig(),
        &request(three_by_two()),
        &control,
        &mut |_, msg| {
            if msg.starts_with("tile 3 of") {
                control.pause();
            }
            // Once it says it is paused: resumed 300 ms on, the moment noted.
            if msg == "paused" {
                paused_seen += 1;
                if resuming.is_none() {
                    let (c, at, clock) =
                        (control.clone(), resumed.clone(), b.stage.clock().clone());
                    resuming = Some(std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(300));
                        *at.lock().unwrap() = Some(tetera_core::Clock::now(clock.as_ref()));
                        c.resume();
                    }));
                }
            }
            Ok(())
        },
    )
    .unwrap();
    resuming.unwrap().join().unwrap();
    assert_eq!(r.tiles.len(), 6);
    assert!(paused_seen >= 2, "{paused_seen}");
    // Asked at the third tile's start, the pause held the fourth until it was resumed.
    let resumed = resumed.lock().unwrap().unwrap();
    assert!(
        r.tiles[2].exposure_start < resumed,
        "the third taken before the pause"
    );
    assert!(
        r.tiles[3].still_at > resumed,
        "the fourth moved to after it"
    );
    assert!(!control.is_paused());
}

#[test]
fn a_scan_cancelled_at_any_tile_keeps_those_taken_as_a_valid_dataset() {
    let _cpu = crate::tests::shared();
    let b = bench(FOCUS);
    let live = b.camera.config();
    for k in [1usize, 3, 6] {
        let before = b.store.datasets("p").unwrap().len();
        let e = scan(
            &b.rig(),
            &request(three_by_two()),
            &ScanControl::default(),
            &mut |_, msg| {
                if msg == format!("tile {k} of 6") {
                    Err(Error::cancelled("cancelled"))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Cancelled);
        let datasets = b.store.datasets("p").unwrap();
        if k == 1 {
            assert_eq!(datasets.len(), before, "none taken, no dataset");
        } else {
            assert_eq!(datasets.len(), before + 1);
            let read = b.store.read(&datasets[0].id).unwrap();
            assert_eq!(read.frames(), k as u64 - 1, "{k}");
            read.read(k as u64 - 2).unwrap();
        }
        assert_eq!(b.camera.config(), live);
    }
    // Cancelled while paused.
    let control = ScanControl::default();
    control.pause();
    let e = scan(
        &b.rig(),
        &request(three_by_two()),
        &control,
        &mut |_, msg| {
            if msg == "paused" {
                Err(Error::cancelled("cancelled"))
            } else {
                Ok(())
            }
        },
    )
    .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Cancelled);
}

#[test]
fn a_z_scan_takes_its_planes_rising() {
    let _cpu = crate::tests::shared();
    let b = bench(FOCUS);
    let mut done = Vec::new();
    let r = scan(
        &b.rig(),
        &request(Pattern::ZAround {
            planes: 5,
            step: 1.25,
        }),
        &ScanControl::default(),
        &mut |d, msg| {
            if msg.starts_with("tile ") {
                done.push(d);
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(done, [0.0, 0.2, 0.4, 0.6, 0.8], "each plane's start, done");
    let zs: Vec<f64> = r.tiles.iter().map(|t| t.recorded.z.um()).collect();
    assert_eq!(
        zs,
        [FOCUS - 2.5, FOCUS - 1.25, FOCUS, FOCUS + 1.25, FOCUS + 2.5]
    );
}

#[test]
fn a_snake_autofocused_at_its_first_tile_takes_the_rest_at_its_focus() {
    let _cpu = crate::tests::alone();
    let b = bench(FOCUS + 5.0);
    let mut asked = request(snake(2_000.0, 2_000.0, 2_040.0, 2_010.0, 0.1));
    asked.autofocus = AutofocusAt::First;
    let r = scan(&b.rig(), &asked, &ScanControl::default(), &mut going).unwrap();
    assert_eq!(r.tiles.len(), 2);
    let d = depth_of_field(20.0, 0.42, 2.4);
    for t in &r.tiles {
        assert!((t.recorded.z.um() - FOCUS).abs() <= d / 4.0, "{t:?}");
        assert_eq!(
            t.planned.z, r.tiles[0].planned.z,
            "at the first tile's focus"
        );
    }
}

#[test]
fn bad_requests_are_refused_before_anything_moves() {
    let _cpu = crate::tests::shared();
    let b = bench(FOCUS);
    let here = b.here();
    let cases = [
        ScanRequest {
            settle_ms: Some(-1.0),
            ..request(three_by_two())
        },
        ScanRequest {
            exposure_ms: Some(0.0),
            ..request(three_by_two())
        },
        ScanRequest {
            write_mb_s: Some(0.0),
            ..request(three_by_two())
        },
        ScanRequest {
            autofocus: AutofocusAt::Every { tiles: 0 },
            ..request(three_by_two())
        },
        ScanRequest {
            autofocus: AutofocusAt::First,
            ..request(Pattern::ZAround {
                planes: 3,
                step: 1.0,
            })
        },
    ];
    for bad in cases {
        let e = scan(&b.rig(), &bad, &ScanControl::default(), &mut going).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{bad:?}");
    }
    let mut nowhere = request(three_by_two());
    nowhere.project = "none".into();
    let e = scan(&b.rig(), &nowhere, &ScanControl::default(), &mut going).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::NotFound);
    b.mount.fit(None).unwrap();
    assert!(
        preview(&b.rig(), &request(three_by_two())).is_err(),
        "no objective"
    );
    assert_eq!(b.here(), here, "nothing moved");
    assert!(request(three_by_two()).check().is_ok());
}

#[test]
fn a_requests_settle_exposure_and_throughput_are_checked_at_their_edges() {
    let with = |settle_ms, exposure_ms, write_mb_s| {
        ScanRequest {
            settle_ms,
            exposure_ms,
            write_mb_s,
            ..request(three_by_two())
        }
        .check()
        .is_ok()
    };
    assert!(with(Some(0.0), Some(5.0), Some(50.0)));
    assert!(with(None, Some(0.1), Some(1_000.0)));
    for bad in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(!with(None, Some(bad), None), "an exposure of {bad}");
        assert!(!with(None, None, Some(bad)), "a throughput of {bad}");
    }
    for bad in [-1.0, f64::INFINITY, f64::NAN] {
        assert!(!with(Some(bad), None, None), "a settle of {bad}");
    }
}

#[test]
fn a_scans_camera_is_the_whole_sensor_at_12_bits_its_exposure_asked_or_the_live_views() {
    let live = CameraConfig {
        mode: 1,
        bit_depth: 8,
        binning: tetera_core::frame::format::Binning { x: 2, y: 2 },
        roi: Some(tetera_core::Rect {
            x: 4,
            y: 2,
            width: 32,
            height: 16,
        }),
        exposure: Duration::from_millis(10),
        gain: 3.0,
        trigger: tetera_hal::camera::Trigger::Software,
    };
    let mut asked = request(three_by_two());
    asked.exposure_ms = Some(2.5);
    let config = engine::full(&asked, &live).unwrap();
    assert_eq!(
        config,
        CameraConfig {
            mode: 0,
            bit_depth: 12,
            binning: Default::default(),
            roi: None,
            exposure: Duration::from_micros(2_500),
            ..live.clone()
        }
    );
    asked.exposure_ms = None;
    assert_eq!(engine::full(&asked, &live).unwrap().exposure, live.exposure);
}

#[test]
fn a_plans_time_is_its_moves_reports_settles_frames_writes_and_autofocus() {
    let _cpu = crate::tests::shared();
    let b = bench(FOCUS);
    let mut asked = request(three_by_two());
    asked.settle_ms = Some(30.0);
    asked.exposure_ms = Some(5.0);
    asked.write_mb_s = Some(2.0);
    asked.autofocus = AutofocusAt::Every { tiles: 2 };
    let plan = preview(&b.rig(), &asked).unwrap();
    // Mode 0 at 12 bits: its rate in 16-bit words.
    let camera = b.camera.capabilities();
    let full = camera.modes.iter().find(|m| m.id == 0).unwrap();
    let interval = 1e3 / full.max_fps_16;
    assert!(interval > 5.0 && full.max_fps_16 != full.max_fps);
    let frame = 1.5 * interval + 5.0;
    let here = b.here();
    let d = depth_of_field(20.0, 0.42, 2.4);
    let up = move_time(here, at(2_000.0, 2_000.0, FOCUS + d), &caps()).unwrap();
    let want = Timing {
        report_ms: 15.0,
        settle_ms: 30.0,
        frame_ms: frame,
        write_ms: 192.0 * 128.0 * 2.0 / 2e6 * 1e3,
        switch_ms: 700.0,
        autofocus_ms: 26.0 * (up * 1e3 + 15.0 + 1.5 * frame),
    };
    for (got, want, what) in [
        (plan.timing.report_ms, want.report_ms, "report"),
        (plan.timing.settle_ms, want.settle_ms, "settle"),
        (plan.timing.frame_ms, want.frame_ms, "frame"),
        (plan.timing.write_ms, want.write_ms, "write"),
        (plan.timing.switch_ms, want.switch_ms, "switch"),
        (plan.timing.autofocus_ms, want.autofocus_ms, "autofocus"),
    ] {
        assert!((got - want).abs() < 1e-9, "{what}: {got} against {want}");
    }
    let estimate_s = estimate(&plan.positions, here, &caps(), &want, |i| i % 2 == 0).unwrap();
    assert!(
        (plan.estimate_s - estimate_s).abs() < 1e-9,
        "{} {estimate_s}",
        plan.estimate_s
    );
}
