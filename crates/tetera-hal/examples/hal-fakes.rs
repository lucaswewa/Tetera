//! The fake camera driven through the `Camera` trait (P1.09a): its capabilities, a snap, a
//! stream at its rate, a fault, and its events as they come.
//!
//! Run it with `cargo run -p tetera-hal --example hal-fakes`; `-- --smoke` streams for a
//! quarter of a second instead of one.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tetera_core::{Frame, Result, SystemClock};
use tetera_hal::{Camera, CameraConfig, CameraFault, FakeCamera, Trigger};

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let seconds = if smoke { 0.25 } else { 1.0 };
    let camera = FakeCamera::new(Arc::new(SystemClock::new()), 30.0);
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
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
