//! The HAL's fakes driven through its traits: the fake camera (P1.09a), with its capabilities, a
//! snap, a stream at its rate, a fault and its events; then the fake stage, the objectives and
//! the light (P1.09b).
//!
//! Run it with `cargo run -p tetera-hal --example hal-fakes`; `-- --smoke` streams and jogs
//! for a quarter of the time.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tetera_core::frame::meta::LightState;
use tetera_core::{Clock, Frame, Result, StagePos, SystemClock, Um};
use tetera_hal::{
    Axis, Camera, CameraConfig, CameraFault, FakeCamera, FakeStage, Illumination,
    IlluminationCapabilities, LightChannel, ManualIllumination, ManualObjectiveMount,
    ObjectiveCatalogue, ObjectiveMount, Pace, Span, Stage, StageFault, Trigger,
};

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let seconds = if smoke { 0.25 } else { 1.0 };
    let clock: Arc<dyn Clock> = Arc::new(SystemClock::new());
    camera(clock.clone(), seconds)?;
    stage(clock, seconds)?;
    objectives_and_light()?;
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

fn camera(clock: Arc<dyn Clock>, seconds: f64) -> Result<()> {
    let camera = FakeCamera::new(clock, 30.0);
    let events = camera.subscribe();
    let caps = camera.capabilities();
    println!("{}: a {:?} sensor", caps.model, caps.pattern);
    for mode in &caps.modes {
        println!(
            "  mode {}: {}, {} × {}, {:?} bits, up to {} fps",
            mode.id, mode.name, mode.width, mode.height, mode.bit_depths, mode.max_fps
        );
    }
    println!(
        "  exposure {} to {} ms, gain {} to {}, triggers {:?}",
        caps.exposure_ms.min, caps.exposure_ms.max, caps.gain.min, caps.gain.max, caps.triggers
    );

    let config = CameraConfig {
        exposure: Duration::from_millis(20),
        gain: 2.0,
        ..camera.config()
    };
    camera.configure(&config)?;
    println!(
        "\nConfigured: {}",
        serde_json::to_string(&camera.config()).expect("JSON")
    );
    let frame = camera.snap(Duration::from_secs(1))?;
    println!(
        "Snapped frame {}: {} × {}, {} bytes, exposed {:?}",
        frame.meta().seq,
        frame.layout().width,
        frame.layout().height,
        frame.data().len(),
        frame.meta().exposure
    );

    let count = Arc::new(AtomicU64::new(0));
    let last = Arc::new(AtomicU64::new(0));
    let (n, l) = (count.clone(), last.clone());
    let start = Instant::now();
    camera.start(Arc::new(move |f: Frame| {
        n.fetch_add(1, Ordering::Relaxed);
        l.store(f.meta().seq, Ordering::Relaxed);
    }))?;
    std::thread::sleep(Duration::from_secs_f64(seconds / 2.0));
    camera.inject(CameraFault::DropFrames(2));
    std::thread::sleep(Duration::from_secs_f64(seconds / 2.0));
    camera.stop()?;
    let elapsed = start.elapsed().as_secs_f64();
    let got = count.load(Ordering::Relaxed);
    println!(
        "Streamed {got} frames in {elapsed:.2} s, {:.1} fps, the last numbered {}, 2 dropped on purpose",
        got as f64 / elapsed,
        last.load(Ordering::Relaxed)
    );

    camera.configure(&CameraConfig {
        trigger: Trigger::Software,
        ..config
    })?;
    let triggered = Arc::new(AtomicU64::new(0));
    let t = triggered.clone();
    camera.start(Arc::new(move |_: Frame| {
        t.fetch_add(1, Ordering::Relaxed);
    }))?;
    for _ in 0..3 {
        camera.trigger()?;
    }
    let wait = Instant::now();
    while triggered.load(Ordering::Relaxed) < 3 && wait.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(1));
    }
    camera.stop()?;
    println!(
        "Triggered 3 times: {} frames",
        triggered.load(Ordering::Relaxed)
    );

    camera.inject(CameraFault::Disconnect);
    println!(
        "Disconnected: a snap fails with \"{}\"",
        camera.snap(Duration::from_secs(1)).unwrap_err()
    );

    println!("\nIts events:");
    for event in events.try_iter() {
        println!("  {}", serde_json::to_string(&event).expect("JSON"));
    }
    Ok(())
}

fn at(pos: &StagePos) -> String {
    format!(
        "({:.0}, {:.0}, {:.0}) µm",
        pos.x.um(),
        pos.y.um(),
        pos.z.um()
    )
}

fn stage(clock: Arc<dyn Clock>, seconds: f64) -> Result<()> {
    let stage = FakeStage::new(clock, Pace::Timed);
    let events = stage.subscribe();
    let caps = stage.capabilities();
    println!(
        "\n{}, homing to {}:",
        caps.model,
        at(&caps.home.unwrap_or_default())
    );
    for a in &caps.axes {
        println!(
            "  {:?}: {} to {} µm, up to {} µm/s",
            a.axis, a.travel.min, a.travel.max, a.max_speed
        );
    }
    let wait = Duration::from_secs(5);

    let target = StagePos::new(Um(2_000.0), Um(1_000.0), Um(300.0));
    let start = Instant::now();
    stage.move_to(target, None)?;
    std::thread::sleep(Duration::from_millis(50));
    let during = stage.reading()?;
    let end = stage.wait(wait)?;
    println!(
        "\nMoved to {} in {:.2} s, as fast as it goes; {:?} at {} on the way",
        at(&end.position),
        start.elapsed().as_secs_f64(),
        during.state,
        at(&during.position)
    );
    let start = Instant::now();
    stage.move_by(StagePos::new(Um(-1_000.0), Um(0.0), Um(0.0)), Some(5_000.0))?;
    let end = stage.wait(wait)?;
    println!(
        "Moved by -1 mm in x at 5 mm/s: at {} in {:.2} s",
        at(&end.position),
        start.elapsed().as_secs_f64()
    );
    stage.jog(Axis::X, 1_000.0)?;
    std::thread::sleep(Duration::from_secs_f64(seconds / 4.0));
    stage.stop_jog()?;
    println!(
        "Jogged in x at 1 mm/s for {:.2} s: at {}",
        seconds / 4.0,
        at(&stage.reading()?.position)
    );
    println!(
        "Sent beyond its travel: refused, \"{}\"",
        stage
            .move_to(StagePos::new(Um(200_000.0), Um(0.0), Um(0.0)), None)
            .unwrap_err()
    );
    stage.inject(StageFault::Stall);
    stage.move_to(StagePos::default(), None)?;
    let stalled = stage.wait(wait).unwrap_err();
    println!(
        "Stalled on the way back: \"{stalled}\", {:?} at {}",
        stage.reading()?.state,
        at(&stage.reading()?.position)
    );
    stage.home()?;
    let homed = stage.wait(wait)?;
    println!("Homed: {:?} at {}", homed.state, at(&homed.position));

    println!("\nIts events:");
    for event in events.try_iter() {
        println!("  {}", serde_json::to_string(&event).expect("JSON"));
    }
    Ok(())
}

fn objectives_and_light() -> Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/objectives.toml");
    let catalogue = ObjectiveCatalogue::load(&path)?;
    println!("\nObjectives, from configs/objectives.toml:");
    for o in &catalogue.objectives {
        println!(
            "  {}: {}, NA {}, working distance {} mm",
            o.name, o.model, o.numerical_aperture, o.working_distance_mm
        );
    }
    let mount = ManualObjectiveMount::new(catalogue);
    mount.fit(Some("20x"))?;
    println!(
        "The user fits the {}; a 40x is refused: \"{}\"",
        mount.fitted().map(|o| o.name).unwrap_or_default(),
        mount.fit(Some("40x")).unwrap_err()
    );

    let light = ManualIllumination::new(IlluminationCapabilities {
        model: "the instrument's lamp, set by hand".into(),
        channels: vec![LightChannel {
            name: "transmitted".into(),
            dimmable: true,
            colour_temperature_k: Some(Span {
                min: 2_700.0,
                max: 6_500.0,
            }),
        }],
    });
    light.set(
        "transmitted",
        LightState {
            on: true,
            intensity: 0.8,
            colour_temperature_k: Some(3_200.0),
        },
    )?;
    println!(
        "\nThe light, as the user says it is set: {}",
        serde_json::to_string(&light.state("transmitted")?).expect("JSON")
    );
    println!(
        "Set past full: refused, \"{}\"",
        light
            .set(
                "transmitted",
                LightState {
                    on: true,
                    intensity: 1.5,
                    colour_temperature_k: None,
                },
            )
            .unwrap_err()
    );
    Ok(())
}
