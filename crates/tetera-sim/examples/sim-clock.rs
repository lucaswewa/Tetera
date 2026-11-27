//! The simulated clock and a move's profile (P1.10a): the clock at 1×, faster and stepped, a
//! thread woken by steps, then a move sampled in stepped time, and the same move held partway.
//!
//! Run it with `cargo run -p tetera-sim --example sim-clock`; `-- --smoke` waits a quarter as
//! long on the host's clock.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tetera_core::{Clock, Result, StagePos, Timestamp, Um};
use tetera_sim::{AxisLimits, Profile, Rate, SimClock};

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let wait = Duration::from_millis(if smoke { 25 } else { 100 });

    for rate in [Rate::RealTime, Rate::Faster(10.0)] {
        let clock = SimClock::new(rate)?;
        let host = Instant::now();
        std::thread::sleep(wait);
        println!(
            "{rate:?}: {:.1} ms of the host's time, {:.1} ms on the clock",
            ms(host.elapsed()),
            ms(clock.now().as_duration())
        );
    }

    let clock = Arc::new(SimClock::stepped());
    let waiter = clock.clone();
    let woken = std::thread::spawn(move || {
        let t = Timestamp::from_duration(Duration::from_millis(50));
        let reached = waiter.wait_until(t, Duration::from_secs(5));
        (reached, waiter.now())
    });
    for _ in 0..5 {
        clock.step(Duration::from_millis(10))?;
    }
    let (reached, at) = woken.join().expect("the waiting thread");
    println!(
        "Stepped: five steps of 10 ms woke a thread waiting for 50 ms ({reached}), at {:.0} ms",
        ms(at.as_duration())
    );

    // The fake stage's axes: x and y at 10 mm/s and 100 mm/s², z at 2 mm/s and 50 mm/s².
    let xy = AxisLimits {
        max_speed: 10_000.0,
        acceleration: 100_000.0,
    };
    let z = AxisLimits {
        max_speed: 2_000.0,
        acceleration: 50_000.0,
    };
    let target = StagePos::new(Um(2_000.0), Um(1_000.0), Um(300.0));
    let profile = Profile::new(StagePos::default(), target, None, &[xy, xy, z])?;
    println!(
        "\nA move from the origin to (2000, 1000, 300) µm: {:.0} µm, at {:.2} mm/s and {:.1} mm/s² along its line, in {:.1} ms",
        profile.length(),
        profile.peak_speed() / 1e3,
        profile.acceleration() / 1e3,
        ms(profile.duration())
    );
    println!("  t (ms)    x (µm)    y (µm)    z (µm)   speed (mm/s)");
    let start = clock.now();
    loop {
        let t = clock.now().saturating_since(start);
        let p = profile.position(t);
        println!(
            "{:8.0}  {:8.1}  {:8.1}  {:8.1}  {:8.2}",
            ms(t),
            p.x.um(),
            p.y.um(),
            p.z.um(),
            profile.along(t).1 / 1e3
        );
        if t >= profile.duration() {
            break;
        }
        clock.step(Duration::from_millis(20))?;
    }
    let held = profile.hold(Duration::from_millis(100));
    let rest = held.to();
    println!(
        "\nHeld at 100 ms, it slows to rest at ({:.1}, {:.1}, {:.1}) µm, {:.1} ms after it set off",
        rest.x.um(),
        rest.y.um(),
        rest.z.um(),
        ms(held.duration())
    );
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
