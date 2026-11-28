//! Autofocus's measures and search (P1.16a). First, each focus measure through a stack of
//! frames from the virtual camera, H&E at 20× on the CPU, against the section's middle, the
//! simulator's true focus. Then the search, 100 times at each objective's depth of field, on
//! noisy known curves through a stage with backlash and steps, with its errors.
//!
//! Run it with `cargo run --release -p tetera-workflows --example focus-search`; `-- --smoke`
//! takes a short stack from a tiny sensor and 20 searches at each.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tetera_core::frame::meta::LightState;
use tetera_core::{StagePos, Um};
use tetera_hal::Stage;
use tetera_hal::camera::{Camera, CameraConfig};
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue, ObjectiveMount};
use tetera_imaging::Mosaic;
use tetera_imaging::focus::{FocusMeasure, GREEN_WIDTH, green};
use tetera_sim::camera::{Backend, Scene, VirtualCamera, VirtualCameraConfig};
use tetera_sim::light::CHANNEL;
use tetera_sim::optics::Sensor;
use tetera_sim::{
    HeSample, Rate, SectionConfig, SimClock, VirtualIllumination, VirtualStage, VirtualStageConfig,
};
use tetera_workflows::focus::synthetic::{Shape, Synthetic};
use tetera_workflows::focus::{SearchOptions, search};

/// The section's bottom, as the camera is told; its middle is the true focus.
const SECTION_Z: f64 = 1_000.0;

/// The depth of field, µm, of an objective of `magnification` and `na` with 2.4 µm pixels.
fn depth_of_field(magnification: f64, na: f64) -> f64 {
    0.55 / (na * na) + 2.4 / (magnification * na)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    stack(smoke)?;
    let trials = if smoke { 20 } else { 100 };
    // Peaks as sharp as the gradient measures' on the simulator, and as broad as normalised
    // variance's, which is smoother.
    searches(
        trials,
        "peaks 0.75 to 1.25 d wide, 1 % noise",
        [0.75, 1.0, 1.25],
        0.01,
    )?;
    searches(
        trials,
        "peaks 4 to 6 d wide, 0.2 % noise",
        [4.0, 5.0, 6.0],
        0.002,
    )?;
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// Each measure through a stack at 20×, ±8 depths of field around the section's middle.
fn stack(smoke: bool) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height, half) = if smoke { (96, 72, 8) } else { (384, 256, 16) };
    let section = SectionConfig::default();
    let middle = SECTION_Z + f64::from(section.thickness) / 2.0;
    let clock = Arc::new(SimClock::new(Rate::RealTime)?);
    let stage = Arc::new(VirtualStage::new(
        clock,
        VirtualStageConfig {
            start: StagePos::new(Um(2_000.0), Um(2_000.0), Um(middle)),
            ..VirtualStageConfig::default()
        },
    )?);
    let mount = ManualObjectiveMount::new(ObjectiveCatalogue::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/objectives.toml"),
    )?);
    mount.fit(Some("20x"))?;
    let objective = mount.fitted().ok_or("no objective")?;
    let d = depth_of_field(objective.magnification, objective.numerical_aperture);
    let light = VirtualIllumination::new(1_000.0, 0.05);
    light.set(
        CHANNEL,
        LightState {
            on: true,
            intensity: 1.0,
            colour_temperature_k: Some(3_200.0),
        },
    )?;
    let camera = VirtualCamera::new(
        Scene {
            stage: stage.clone(),
            objectives: Arc::new(mount),
            light: Arc::new(light),
            sample: HeSample::new(7, section)?,
        },
        VirtualCameraConfig {
            sensor: Sensor {
                width,
                height,
                pixel: 2.4,
            },
            section_z: SECTION_Z,
            ..VirtualCameraConfig::default()
        },
        Backend::Cpu,
    )?;
    camera.configure(&CameraConfig {
        mode: 0,
        bit_depth: 12,
        ..camera.config()
    })?;
    println!(
        "Each measure through a stack: H&E at 20×, {width} × {height} on the CPU, its section {} µm thick, in steps of half the depth of field, {d:.2} µm. Each measure over its highest:\n",
        section.thickness
    );
    print!("{:>12}", "z − middle");
    for m in FocusMeasure::ALL {
        print!("{:>20}", m.name());
    }
    println!();
    let mut rows = Vec::new();
    for i in -half..=half {
        let offset = f64::from(i) * d / 2.0;
        stage.move_to(
            StagePos::new(Um(2_000.0), Um(2_000.0), Um(middle + offset)),
            None,
        )?;
        let at = stage.wait(Duration::from_secs(10))?.position.z.um() - middle;
        let frame = camera.snap(Duration::from_secs(30))?;
        let image = green(&Mosaic::from_frame(&frame)?, GREEN_WIDTH)?;
        rows.push((at, FocusMeasure::ALL.map(|m| m.of(&image))));
    }
    let highest: Vec<f64> = (0..4)
        .map(|k| rows.iter().map(|r| r.1[k]).fold(0.0, f64::max))
        .collect();
    for (at, values) in &rows {
        print!("{:>9.2} µm", at);
        for (k, v) in values.iter().enumerate() {
            let mark = if *v == highest[k] { "*" } else { " " };
            print!("{:>19.3}{mark}", v / highest[k]);
        }
        println!();
    }
    println!();
    for (k, m) in FocusMeasure::ALL.iter().enumerate() {
        let (at, _) = rows
            .iter()
            .find(|r| r.1[k] == highest[k])
            .ok_or("an empty stack")?;
        let far = rows
            .first()
            .map_or(0.0, |r| r.1[k])
            .max(rows.last().map_or(0.0, |r| r.1[k]));
        println!(
            "  {:<20} peaks {:+.2} µm from the middle, {:+.2} depths of field; {:.1} times its value 8 depths away",
            m.name(),
            at,
            at / d,
            highest[k] / far.max(f64::MIN_POSITIVE)
        );
    }
    Ok(())
}

/// The search, `trials` times at each objective's depth of field: noisy curves, Gaussian and
/// Lorentzian, of three `widths` in depths of field, through a stage with backlash and the
/// virtual stage's steps, from starts within ±6 depths of field.
fn searches(
    trials: u64,
    label: &str,
    widths: [f64; 3],
    noise: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\nThe search, {trials} times at each objective: {label}, 2 µm of backlash, steps of 0.3125 µm, starts within ±6 depths of field.\n"
    );
    println!(
        "  {:<10} {:>8} {:>16} {:>16} {:>12} {:>12} {:>10}",
        "objective", "d, µm", "within d / 4", "median error", "95th", "worst", "positions"
    );
    for (name, magnification, na) in [
        ("5x", 5.0, 0.14),
        ("10x", 10.0, 0.28),
        ("20x", 20.0, 0.42),
        ("50x", 50.0, 0.55),
    ] {
        let d = depth_of_field(magnification, na);
        let mut errors = Vec::new();
        let mut positions = 0;
        for trial in 0..trials {
            let mut dice = Synthetic::new(0.0, 0.0, 10_000 + trial);
            let offset = (dice.uniform() * 2.0 - 1.0) * 6.0 * d;
            let mut probe = Synthetic::new(5_000.0, 5_000.0 + offset, trial);
            probe.backlash = 2.0;
            probe.resolution = 0.3125;
            probe.noise = noise;
            probe.width = d * widths[trial as usize % 3];
            probe.shape = if trial % 2 == 0 {
                Shape::Gaussian
            } else {
                Shape::Lorentzian
            };
            let focus = search(
                &mut probe,
                d,
                (0.0, 10_000.0),
                &SearchOptions::default(),
                &mut |_, _| Ok(()),
            )?;
            positions += focus.curve.len();
            match focus.z {
                Some(_) => errors.push((probe.carriage() - probe.focus).abs() / d),
                None => errors.push(f64::INFINITY),
            }
        }
        errors.sort_by(f64::total_cmp);
        let at = |q: f64| errors[((errors.len() - 1) as f64 * q).round() as usize];
        let within = errors.iter().filter(|&&e| e <= 0.25).count();
        println!(
            "  {name:<10} {d:>8.2} {:>16} {:>14.3} d {:>10.3} d {:>10.3} d {:>10}",
            format!("{within} of {trials}"),
            at(0.5),
            at(0.95),
            at(1.0),
            positions / trials as usize
        );
    }
    Ok(())
}
