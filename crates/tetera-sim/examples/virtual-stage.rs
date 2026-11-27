//! The virtual stage (P1.10b): a move's position against time in stepped time, the stage
//! itself ahead of its late reports; then the same stage at 1× (or 10× with `--smoke`),
//! driven through the `Stage` trait: a move, a jog, homing, a stall and a hard limit, and its
//! events.
//!
//! Run it with `cargo run -p tetera-sim --example virtual-stage`.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tetera_core::{Clock, Result, StagePos, Um};
use tetera_hal::{Axis, Stage};
use tetera_sim::{GrblState, Rate, SimClock, VirtualStage, VirtualStageConfig, VirtualStageFault};

fn at(p: &StagePos) -> String {
    format!("({:7.1}, {:7.1}, {:6.1})", p.x.um(), p.y.um(), p.z.um())
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    stepped()?;
    running(if smoke {
        Rate::Faster(10.0)
    } else {
        Rate::RealTime
    })?;
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

fn stepped() -> Result<()> {
    let stage = VirtualStage::new(Arc::new(SimClock::stepped()), VirtualStageConfig::default())?;
    let config = stage.config();
    println!("The virtual stage, in stepped time:");
    for (axis, a) in Axis::ALL.iter().zip(&config.axes) {
        println!(
            "  {axis:?}: {} to {} µm, up to {} mm/s at {} mm/s²",
            a.travel.min,
            a.travel.max,
            a.limits.max_speed / 1e3,
            a.limits.acceleration / 1e3
        );
    }
    println!(
        "  reports every {:.0} ms, each arriving {:.0} ms late",
        ms(config.report_period),
        ms(config.report_latency)
    );
    let target = StagePos::new(Um(5_000.0), Um(2_000.0), Um(500.0));
    stage.move_to(target, None)?;
    println!("\nA move to (5000, 2000, 500) µm, every 50 ms of the clock:");
    println!("  t (ms)  state  the stage itself (µm)       its latest report (µm), and when");
    let start = stage.clock().now();
    loop {
        let t = stage.clock().now().saturating_since(start);
        let report = *stage.track().last().expect("a report");
        let state = stage.grbl_state();
        println!(
            "{:8.0}  {:5}  {}  {} at {:.0} ms",
            ms(t),
            format!("{state:?}"),
            at(&stage.position()),
            at(&report.position),
            ms(report.at.as_duration())
        );
        if report.state == GrblState::Idle && state == GrblState::Idle && t > Duration::ZERO {
            break;
        }
        stage.clock().step(Duration::from_millis(50))?;
    }
    Ok(())
}

fn running(rate: Rate) -> Result<()> {
    let stage = VirtualStage::new(
        Arc::new(SimClock::new(rate)?),
        VirtualStageConfig::default(),
    )?;
    let events = stage.subscribe();
    let wait = Duration::from_secs(30);
    let clock = stage.clock().clone();
    let timed = |what: &str, f: &dyn Fn() -> Result<()>| -> Result<()> {
        let (host, sim) = (Instant::now(), clock.now());
        f()?;
        println!(
            "{what}: {:.0} ms on the clock, {:.0} ms on the host's; at {}",
            ms(clock.now().saturating_since(sim)),
            ms(host.elapsed()),
            at(&stage.reading()?.position)
        );
        Ok(())
    };
    println!("\nThe virtual stage at {rate:?}, through the Stage trait:");
    timed("A move to (5000, 2000, 500) µm, waited for", &|| {
        stage.move_to(StagePos::new(Um(5_000.0), Um(2_000.0), Um(500.0)), None)?;
        stage.wait(wait).map(drop)
    })?;
    timed(
        "A jog in x at 5 mm/s, stopped after 200 ms of the clock",
        &|| {
            stage.jog(Axis::X, 5_000.0)?;
            let t = stage
                .clock()
                .now()
                .saturating_add(Duration::from_millis(200));
            stage.clock().wait_until(t, wait);
            stage.stop_jog()
        },
    )?;
    timed("Homing: z, then x and y, then locating", &|| {
        stage.home()?;
        stage.wait(wait).map(drop)
    })?;
    stage.inject(VirtualStageFault::Stall);
    stage.move_to(StagePos::new(Um(8_000.0), Um(0.0), Um(0.0)), None)?;
    println!(
        "A move to x = 8 mm, stalled: \"{}\", at {}",
        stage.wait(wait).unwrap_err(),
        at(&stage.reading()?.position)
    );
    stage.home()?;
    stage.wait(wait)?;
    stage.jog(Axis::Y, 2_000.0)?;
    let t = stage
        .clock()
        .now()
        .saturating_add(Duration::from_millis(100));
    stage.clock().wait_until(t, wait);
    stage.inject(VirtualStageFault::LimitHit);
    println!(
        "A hard limit while jogging: \"{}\"",
        stage.wait(wait).unwrap_err()
    );
    stage.home()?;
    stage.wait(wait)?;
    println!(
        "Beyond its soft limits: \"{}\"",
        stage
            .move_to(StagePos::new(Um(0.0), Um(80_000.0), Um(0.0)), None)
            .unwrap_err()
    );
    println!(
        "\nIts events ({} reports in its track):",
        stage.track().len()
    );
    for event in events.try_iter() {
        println!("  {}", serde_json::to_string(&event).expect("JSON"));
    }
    Ok(())
}
