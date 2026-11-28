//! Autofocus on a small virtual instrument on the CPU: focus found from below and from
//! above, no frame exposed during a move measured, a blank field, and refusals.

use super::*;
use crate::focus::Sweep;
use std::path::Path;
use tetera_core::frame::meta::LightState;
use tetera_hal::camera::CameraConfig;
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue, ObjectiveMount};
use tetera_sim::camera::{Backend, Scene, VirtualCamera, VirtualCameraConfig};
use tetera_sim::light::CHANNEL;
use tetera_sim::optics::Sensor;
use tetera_sim::{
    HeSample, Rate, SectionConfig, SimClock, VirtualIllumination, VirtualStage, VirtualStageConfig,
};

/// The section's bottom; its middle, 2 µm above, is the true focus.
const SECTION_Z: f64 = 1_000.0;
const FOCUS: f64 = 1_002.0;

/// The virtual instrument at 20×, 192 × 128 on the CPU, in real time, the camera in its
/// binned live mode at 8 bits with a 30 ms exposure.
struct Bench {
    stage: Arc<VirtualStage>,
    mount: Arc<ManualObjectiveMount>,
    camera: VirtualCamera,
    bus: Arc<FrameBus>,
}

fn bench(z: f64) -> Bench {
    let clock = Arc::new(SimClock::new(Rate::RealTime).unwrap());
    let stage = Arc::new(
        VirtualStage::new(
            clock,
            VirtualStageConfig {
                start: StagePos::new(Um(2_000.0), Um(2_000.0), Um(z)),
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
            section_z: SECTION_Z,
            ..VirtualCameraConfig::default()
        },
        Backend::Cpu,
    )
    .unwrap();
    camera
        .configure(&CameraConfig {
            mode: 1,
            bit_depth: 8,
            exposure: Duration::from_millis(30),
            ..camera.config()
        })
        .unwrap();
    Bench {
        stage,
        mount,
        camera,
        bus: FrameBus::new(),
    }
}

impl Bench {
    fn rig(&self) -> FocusRig<'_> {
        FocusRig {
            camera: &self.camera,
            stage: self.stage.as_ref(),
            bus: &self.bus,
            live: None,
            objective: self.mount.fitted(),
            sensor_pixel_um: 2.4,
        }
    }

    fn z(&self) -> f64 {
        self.stage.reading().unwrap().position.z.um()
    }
}

/// The CPU, alone: the virtual camera renders every frame of its stream on it.
fn alone() -> std::sync::RwLockWriteGuard<'static, ()> {
    crate::tests::CPU
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn going(_: f64, _: &str) -> Result<()> {
    Ok(())
}

#[test]
fn the_depth_of_field_is_the_wave_term_and_the_geometric_one() {
    let d = depth_of_field(20.0, 0.42, 2.4);
    assert!((d - (0.55 / 0.1764 + 2.4 / 8.4)).abs() < 1e-12, "{d}");
    assert!((depth_of_field(50.0, 0.55, 2.4) - 1.905).abs() < 1e-3);
}

#[test]
fn autofocus_finds_the_sections_middle_from_below_and_above_and_leaves_the_stream_as_it_was() {
    let _cpu = alone();
    let d = depth_of_field(20.0, 0.42, 2.4);
    for (start, streaming) in [(FOCUS - 2.5 * d, false), (FOCUS + 3.5 * d, true)] {
        let b = bench(start);
        if streaming {
            b.camera.start(b.bus.clone()).unwrap();
        }
        let r = autofocus(&b.rig(), &AutofocusRequest::default(), &mut going).unwrap();
        let z = r.focus.found().unwrap();
        assert!((z - FOCUS).abs() <= d / 4.0, "from {start}: {z}");
        assert!(
            (b.z() - z).abs() < 0.32,
            "the stage at the focus, to a step"
        );
        assert_eq!(b.camera.is_streaming(), streaming, "the stream as it was");
        assert_eq!(
            (r.measure, r.objective.as_str()),
            (FocusMeasure::NormalisedVariance, "20x")
        );
        assert!(r.seconds > 0.0);
        assert_eq!(r.focus.curve.len(), 26);
        b.camera.stop().unwrap();
    }
}

#[test]
fn no_frame_exposed_during_a_move_is_measured() {
    let _cpu = alone();
    let b = bench(FOCUS);
    b.camera.start(b.bus.clone()).unwrap();
    let mut probe = StageProbe::new(b.stage.as_ref(), &b.bus, FocusMeasure::Tenengrad).unwrap();
    let d = depth_of_field(20.0, 0.42, 2.4);
    let options = SearchOptions {
        range: 3.0,
        ..SearchOptions::default()
    };
    let focus = search(&mut probe, d, (0.0, 10_000.0), &options, &mut going).unwrap();
    assert!(focus.z.is_some());
    // Each frame measured started its exposure after the stage reported itself still.
    assert_eq!(probe.taken.len(), focus.curve.len());
    for (still, exposed) in &probe.taken {
        assert!(exposed >= still, "{exposed:?} before {still:?}");
    }
    b.camera.stop().unwrap();
}

#[test]
fn a_blank_field_finds_no_focus_and_leaves_the_stage_where_it_was() {
    let _cpu = alone();
    let start = FOCUS + 3.0;
    let b = bench(start);
    b.camera.set_sample(None).unwrap();
    let r = autofocus(&b.rig(), &AutofocusRequest::default(), &mut going).unwrap();
    assert_eq!(r.focus.z, None);
    assert!(
        r.focus
            .why_not
            .as_deref()
            .is_some_and(|w| w.contains("flat")),
        "{r:?}"
    );
    assert!(
        (b.z() - start).abs() < 0.32,
        "back where it started: {}",
        b.z()
    );
    assert!(r.focus.curve.iter().all(|p| p.sweep == Sweep::Coarse));
    assert!(!b.camera.is_streaming());
}

#[test]
fn autofocus_needs_an_objective_and_options_it_can_take_and_a_failed_search_stops_the_stream() {
    let _cpu = alone();
    let b = bench(FOCUS);
    b.mount.fit(None).unwrap();
    let e = autofocus(&b.rig(), &AutofocusRequest::default(), &mut going).unwrap_err();
    assert_eq!(e.kind(), tetera_core::ErrorKind::InvalidArgument, "{e}");
    b.mount.fit(Some("20x")).unwrap();
    let bad = AutofocusRequest {
        options: SearchOptions {
            range: 0.0,
            ..SearchOptions::default()
        },
        ..AutofocusRequest::default()
    };
    assert!(autofocus(&b.rig(), &bad, &mut going).is_err());
    assert!(!b.camera.is_streaming(), "refused before the stream starts");
    // Stopped by its step: the stream it started is stopped.
    let e = autofocus(&b.rig(), &AutofocusRequest::default(), &mut |done, _| {
        if done > 0.2 {
            Err(Error::cancelled("stopped"))
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert_eq!(e.kind(), tetera_core::ErrorKind::Cancelled);
    assert!(!b.camera.is_streaming());
    // A request in JSON: every field optional.
    let r: AutofocusRequest = serde_json::from_str(r#"{"measure": "tenengrad"}"#).unwrap();
    assert_eq!(r.measure, FocusMeasure::Tenengrad);
    assert_eq!(r.options, SearchOptions::default());
}
