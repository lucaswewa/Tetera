//! Autofocus on the virtual instrument (P1.16b): 100 starts at random distances within ±6
//! depths of field of the simulator's true focus, the section's middle, at 5× and at 50×,
//! each search's error against it; then a search's time at 10×; then a blank field, which
//! finds no focus and leaves the stage where it was. `-- --compare` runs the other three
//! measures too, 25 starts each.
//!
//! Run it with `cargo run --release -p tetera-api --features cuda --example autofocus`;
//! `-- --smoke` takes a few starts at 20× on a tiny sensor on the CPU.

use std::time::Duration;
use tetera_api::instrument::{Instrument, ObjectivesConfig, Sample, SensorSize, SimulatorConfig};
use tetera_core::{StagePos, Um};
use tetera_hal::Stage;
use tetera_hal::camera::Camera;
use tetera_hal::objective::ObjectiveMount;
use tetera_imaging::focus::FocusMeasure;
use tetera_workflows::focus::synthetic::Synthetic;
use tetera_workflows::{AutofocusRequest, FocusRig, autofocus, depth_of_field};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let compare = std::env::args().any(|a| a == "--compare");
    let simulator = SimulatorConfig {
        gpu: !smoke,
        sensor: if smoke {
            SensorSize::Tiny
        } else {
            SensorSize::Auto
        },
        // The smoke run's clock ten times as fast, to finish in seconds.
        clock_rate: if smoke { 10.0 } else { 1.0 },
        ..SimulatorConfig::default()
    };
    let instrument = Instrument::new(&simulator, &ObjectivesConfig::default())?;
    let truth = simulator.section_z + f64::from(Sample::default().thickness) / 2.0;
    let compute = instrument.compute();
    println!(
        "The virtual instrument: the camera on the {} ({} × {}), streaming in its binned live mode; the true focus, the section's middle, at {truth} µm.",
        compute.backend, compute.sensor[0], compute.sensor[1]
    );
    instrument.camera.start(instrument.bus.clone())?;
    let mut dice = Synthetic::new(0.0, 0.0, 2026);
    let mut runs = vec![
        ("5x", FocusMeasure::NormalisedVariance, 100),
        ("50x", FocusMeasure::NormalisedVariance, 100),
    ];
    if compare {
        for measure in [
            FocusMeasure::Brenner,
            FocusMeasure::LaplacianVariance,
            FocusMeasure::Tenengrad,
        ] {
            runs.push(("5x", measure, 25));
            runs.push(("50x", measure, 25));
        }
    }
    if smoke {
        runs = vec![("20x", FocusMeasure::NormalisedVariance, 1)];
    }
    println!(
        "\n  {:<10} {:<20} {:>7} {:>14} {:>12} {:>10} {:>10} {:>10}",
        "objective", "measure", "d, µm", "within d / 4", "median", "95th", "worst", "time"
    );
    for (objective, measure, trials) in runs {
        let Tally {
            within,
            errors,
            seconds,
            d,
        } = trials_at(&instrument, &mut dice, objective, measure, trials, truth)?;
        let at = |q: f64| errors[((errors.len() - 1) as f64 * q).round() as usize];
        println!(
            "  {objective:<10} {:<20} {d:>7.2} {:>14} {:>10.3} d {:>8.3} d {:>8.3} d {:>8.2} s",
            measure.name(),
            format!("{within} of {trials}"),
            at(0.5),
            at(0.95),
            at(1.0),
            seconds
        );
    }
    // A search's time at 10×, from 3 depths of field away.
    let tries = if smoke { 1 } else { 10 };
    instrument.objectives.fit(Some("10x"))?;
    let mut times = Vec::new();
    for _ in 0..tries {
        let d = depth(&instrument)?;
        move_to(&instrument, truth + 3.0 * d)?;
        let r = autofocus(
            &rig(&instrument),
            &AutofocusRequest::default(),
            &mut |_, _| Ok(()),
        )?;
        times.push(r.seconds);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "\nAt 10×, a search from 3 depths of field away: {:.2} s median of {tries}, {:.2} s the longest.",
        times[times.len() / 2],
        times[times.len() - 1]
    );
    // A blank field.
    instrument.unload()?;
    let start = truth + 5.0;
    move_to(&instrument, start)?;
    let r = autofocus(
        &rig(&instrument),
        &AutofocusRequest::default(),
        &mut |_, _| Ok(()),
    )?;
    let at = instrument.stage.reading()?.position.z.um();
    println!(
        "A blank field: {}; the stage at {at:.2} µm, where it started, {start:.2} µm.",
        r.focus
            .found()
            .map_or_else(|e| e.to_string(), |z| format!("focus at {z:.2} µm"))
    );
    instrument.camera.stop()?;
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// What a run of searches found.
struct Tally {
    /// The searches within a quarter of the depth of field.
    within: usize,
    /// Every search's error, in depths of field, sorted; infinite where none was found.
    errors: Vec<f64>,
    /// A search's mean time, s.
    seconds: f64,
    /// The depth of field, µm.
    d: f64,
}

/// `trials` searches at `objective` with `measure`, from starts within ±6 depths of field of
/// `truth`.
fn trials_at(
    instrument: &Instrument,
    dice: &mut Synthetic,
    objective: &str,
    measure: FocusMeasure,
    trials: usize,
    truth: f64,
) -> Result<Tally, Box<dyn std::error::Error>> {
    instrument.objectives.fit(Some(objective))?;
    let d = depth(instrument)?;
    let request = AutofocusRequest {
        measure,
        ..AutofocusRequest::default()
    };
    let (mut errors, mut seconds) = (Vec::new(), 0.0);
    for _ in 0..trials {
        move_to(instrument, truth + (dice.uniform() * 2.0 - 1.0) * 6.0 * d)?;
        let r = autofocus(&rig(instrument), &request, &mut |_, _| Ok(()))?;
        seconds += r.seconds;
        errors.push(match r.focus.z {
            Some(_) => (instrument.stage.reading()?.position.z.um() - truth).abs() / d,
            None => f64::INFINITY,
        });
    }
    errors.sort_by(f64::total_cmp);
    let within = errors.iter().filter(|&&e| e <= 0.25).count();
    Ok(Tally {
        within,
        errors,
        seconds: seconds / trials as f64,
        d,
    })
}

/// The fitted objective's depth of field, µm.
fn depth(instrument: &Instrument) -> Result<f64, Box<dyn std::error::Error>> {
    let o = instrument.objectives.fitted().ok_or("no objective")?;
    Ok(depth_of_field(
        o.magnification,
        o.numerical_aperture,
        instrument.sensor_pixel_um(),
    ))
}

/// The stage to `z`, x and y where they are, and waited for.
fn move_to(instrument: &Instrument, z: f64) -> Result<(), Box<dyn std::error::Error>> {
    let here = instrument.stage.reading()?.position;
    instrument
        .stage
        .move_to(StagePos::new(here.x, here.y, Um(z)), None)?;
    instrument.stage.wait(Duration::from_secs(30))?;
    Ok(())
}

/// Autofocus's rig on `instrument`, the stream already running.
fn rig(instrument: &Instrument) -> FocusRig<'_> {
    FocusRig {
        camera: instrument.camera.as_ref(),
        stage: instrument.stage.as_ref(),
        bus: &instrument.bus,
        live: None,
        objective: instrument.objectives.fitted(),
        sensor_pixel_um: instrument.sensor_pixel_um(),
    }
}
