//! The virtual camera, live (P1.12c, Checkpoint A): the MIChrome 20 over the simulated slide
//! at 20×, its stage jogging at 1 mm/s in real time, written as a strip of JPEGs to
//! `target/virtual-camera/`. Then each mode at each bit depth, its rate against the camera's.
//! With `--features cuda` and a GPU, the full 5472 × 3648 sensor on the GPU; otherwise, or
//! with `--cpu`, a 912 × 608 sensor on the CPU.
//!
//! Run it with `cargo run --release -p tetera-sim --features cuda --example virtual-camera`;
//! `-- --smoke` runs a small sensor on the CPU, briefly.

#[path = "common/png.rs"]
mod png;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tetera_core::frame::meta::LightState;
use tetera_core::{Clock, Error, Frame, FrameLayout, Result, StagePos, Timestamp, Um};
use tetera_hal::camera::{Camera, CameraConfig, CameraEvent, FrameSink};
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue, ObjectiveMount};
use tetera_hal::stage::{Axis, Stage};
use tetera_imaging::Image;
use tetera_sim::camera::MODES;
use tetera_sim::light::CHANNEL;
use tetera_sim::optics::{Sensor, blackbody};
use tetera_sim::sensor::bayer;
use tetera_sim::{
    Backend, HeSample, Rate, Scene, SectionConfig, SensorModel, SimClock, VirtualCamera,
    VirtualCameraConfig, VirtualIllumination, VirtualStage, VirtualStageConfig,
};

/// The light through bare glass at full intensity, photoelectrons a ms.
const LIGHT: f32 = 1_000.0;

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let cpu = smoke || std::env::args().any(|a| a == "--cpu");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/virtual-camera");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out)
        .map_err(|e| Error::internal(format!("{}: {e}", out.display())))?;
    let (backend, sensor, what) = backend(cpu, smoke)?;
    let config = VirtualCameraConfig {
        sensor,
        mounting_angle: 0.5_f64.to_radians(),
        ..VirtualCameraConfig::default()
    };
    let scene = scene(&config)?;
    let stage = scene.stage.clone();
    let camera = VirtualCamera::new(scene, config.clone(), backend)?;
    println!(
        "The virtual MIChrome 20 {what}, {} × {} pixels, at 20× over H&E, in real time.",
        sensor.width, sensor.height
    );

    // Live, mode 0 in 8 bits, the stage jogging along x.
    let seconds = if smoke { 0.6 } else { 3.0 };
    set(&camera, 0, 8, 10.0)?;
    warm(&camera)?;
    println!("\nLive in mode 0, 8 bits, 10 ms, jogging at 1 mm/s for {seconds} s:");
    stage.jog(Axis::X, 1_000.0)?;
    let keep = Duration::from_secs_f64(seconds / 6.0);
    let live = run(&camera, seconds, Some(keep))?;
    stage.stop_jog()?;
    live.report(MODES[0].fast_fps);
    if let (Some(a), Some(b)) = (live.positions.first(), live.positions.last()) {
        println!("  the stage from x = {a:.0} to {b:.0} µm");
    }
    for (k, frame) in live.kept.iter().enumerate() {
        let (w, h, rgb) = shown(frame, &config.model, 10.0, 912)?;
        let path = out.join(format!("live-{k}.jpg"));
        let jpeg = tetera_imaging::jpeg::encode(&Image::new(w, h, 3, rgb.clone())?, 85)?;
        std::fs::write(&path, jpeg)
            .map_err(|e| Error::internal(format!("{}: {e}", path.display())))?;
        if k == 0 {
            png::write(&out.join("live-0.png"), w, h, &rgb)?;
        }
    }
    println!(
        "  wrote a frame every {:.1} s to {}, as JPEGs",
        keep.as_secs_f64(),
        out.display()
    );

    // How fast it can jog with each frame within the budget: each frame's new tiles.
    if !smoke {
        println!("\nJogging along y at each speed, mode 0, 8 bits, 10 ms, 1.5 s each:");
        for speed in [100.0, 250.0, 500.0, 1_000.0] {
            stage.jog(Axis::Y, speed)?;
            let r = run(&camera, 1.5, None)?;
            stage.stop_jog()?;
            let worst = r.made.iter().copied().fold(0.0, f64::max);
            println!(
                "  {:>5.0} µm/s: {:>5.2} fps of 14.49, each made in {:>5.1} ms (median, {worst:.0} at most), {} dropped",
                speed,
                r.rate(),
                r.median_ms(),
                r.dropped
            );
        }
    }

    // Each mode at each bit depth, still.
    println!("\nEach mode, still, 1 ms exposures, against the camera's rates:");
    let each = if smoke { 0.4 } else { 2.0 };
    for m in &MODES {
        for (bits, fps) in [(12u8, m.deep_fps), (8, m.fast_fps)] {
            set(&camera, m.id, bits, 1.0)?;
            warm(&camera)?;
            let r = run(&camera, each, None)?;
            println!(
                "  mode {} ({}), {bits:>2} bits: {:>5.2} fps of {fps:>5.2}, each made in {:>5.1} ms (median), {} dropped",
                m.id,
                m.name,
                r.rate(),
                r.median_ms(),
                r.dropped
            );
        }
    }
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// The backend: the GPU when there is one and the CPU isn't asked for.
fn backend(cpu: bool, smoke: bool) -> Result<(Backend, Sensor, &'static str)> {
    #[cfg(feature = "cuda")]
    if !cpu && let Ok(gpu) = tetera_compute::CudaBackend::new(0) {
        let budgets = tetera_compute::budget::VramBudgets::new(3 << 30);
        return Ok((
            Backend::Gpu {
                gpu: Arc::new(gpu),
                budget: budgets.reserve("virtual-camera", 3 << 30)?,
            },
            tetera_sim::optics::MICHROME_20,
            "on the GPU",
        ));
    }
    let _ = cpu;
    let small = Sensor {
        width: if smoke { 192 } else { 912 },
        height: if smoke { 128 } else { 608 },
        pixel: 2.4,
    };
    Ok((Backend::Cpu, small, "on the CPU"))
}

/// The scene: a real-time clock, the stage over tissue in focus, 20× fitted, the light on.
fn scene(config: &VirtualCameraConfig) -> Result<Scene> {
    let clock = Arc::new(SimClock::new(Rate::RealTime)?);
    let stage = VirtualStage::new(
        clock,
        VirtualStageConfig {
            start: StagePos::new(Um(2_000.0), Um(2_000.0), Um(config.section_z + 2.0)),
            ..VirtualStageConfig::default()
        },
    )?;
    let catalogue = ObjectiveCatalogue::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/objectives.toml"),
    )?;
    let mount = ManualObjectiveMount::new(catalogue);
    mount.fit(Some("20x"))?;
    let light = VirtualIllumination::new(LIGHT, 0.05);
    light.set(
        CHANNEL,
        LightState {
            on: true,
            intensity: 1.0,
            colour_temperature_k: Some(3_200.0),
        },
    )?;
    Ok(Scene {
        stage: Arc::new(stage),
        objectives: Arc::new(mount),
        light: Arc::new(light),
        sample: HeSample::new(7, SectionConfig::default())?,
    })
}

/// One frame, so a mode's buffers are made before it is timed: a mode's first frame after a
/// change makes its planes on the GPU and its pool.
fn warm(camera: &VirtualCamera) -> Result<()> {
    camera.snap(Duration::from_secs(10)).map(drop)
}

fn set(camera: &VirtualCamera, mode: u32, bits: u8, ms: f64) -> Result<()> {
    camera.configure(&CameraConfig {
        mode,
        bit_depth: bits,
        exposure: Duration::from_secs_f64(ms / 1e3),
        ..camera.config()
    })
}

/// A frame copied out of the camera's pool, so the pool's buffer goes back to it.
struct Kept {
    layout: FrameLayout,
    data: Vec<u8>,
    gain: f64,
    black: u16,
}

/// What a stream gave: how many frames; each frame's time from its last row's end to its
/// delivery, ms; the stage's x at each; the frames kept; and how many were dropped.
#[derive(Default)]
struct Run {
    frames: usize,
    starts: Vec<Timestamp>,
    made: Vec<f64>,
    positions: Vec<f64>,
    kept: Vec<Kept>,
    dropped: u64,
    seconds: f64,
}

impl Run {
    /// Frames a second, from the first frame's start to the last's, on the clock.
    fn rate(&self) -> f64 {
        match (self.starts.first(), self.starts.last()) {
            (Some(a), Some(b)) if self.starts.len() > 1 => {
                (self.starts.len() - 1) as f64 / b.saturating_since(*a).as_secs_f64()
            }
            _ => f64::NAN,
        }
    }

    fn median_ms(&self) -> f64 {
        let mut v = self.made.clone();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
    }

    fn report(&self, fps: f64) {
        let worst = self.made.iter().copied().fold(0.0, f64::max);
        println!(
            "  {} frames in {:.1} s, {:.2} fps of {fps}; each made in {:.1} ms (median, {worst:.1} at most) after its last row; {} dropped",
            self.frames,
            self.seconds,
            self.rate(),
            self.median_ms(),
            self.dropped
        );
    }
}

/// Streams for `seconds` of the host's time, keeping a copy of a frame every `keep` of the
/// clock's time, and handing every frame straight back to the pool.
fn run(camera: &VirtualCamera, seconds: f64, keep: Option<Duration>) -> Result<Run> {
    let clock = camera.clock().clone();
    let span = camera.timing()?.span();
    let events = camera.subscribe();
    let run = Arc::new(Mutex::new((Run::default(), None::<Timestamp>)));
    let shared = run.clone();
    let sink: Arc<dyn FrameSink> = Arc::new(move |frame: Frame| {
        let meta = frame.meta();
        let ready = meta.exposure_start.saturating_add(span);
        let ms = clock.now().saturating_since(ready).as_secs_f64() * 1e3;
        let Ok(mut guard) = shared.lock() else { return };
        let (r, next) = &mut *guard;
        r.frames += 1;
        r.starts.push(meta.exposure_start);
        r.made.push(ms);
        r.positions.extend(meta.position.map(|p| p.x.um()));
        if let Some(every) = keep
            && next.is_none_or(|t| meta.exposure_start >= t)
        {
            r.kept.push(Kept {
                layout: *frame.layout(),
                data: frame.data().to_vec(),
                gain: meta.gain,
                black: meta.black_level,
            });
            *next = Some(meta.exposure_start.saturating_add(every));
        }
    });
    let start = Instant::now();
    camera.start(sink)?;
    std::thread::sleep(Duration::from_secs_f64(seconds));
    camera.stop()?;
    let seconds = start.elapsed().as_secs_f64();
    let dropped = events
        .try_iter()
        .map(|e| match e {
            CameraEvent::Dropped { count } => count,
            _ => 0,
        })
        .sum();
    let mut guard = run
        .lock()
        .map_err(|_| Error::internal("the sink panicked"))?;
    let mut r = std::mem::take(&mut guard.0);
    r.seconds = seconds;
    r.dropped = dropped;
    Ok(r)
}

/// `frame` for people: each 2 × 2 of the mosaic one pixel, balanced so bare glass is a little
/// below white, scaled down by whole steps to at most `width` across, in sRGB.
fn shown(frame: &Kept, model: &SensorModel, ms: f32, width: u32) -> Result<(u32, u32, Vec<u8>)> {
    let l = frame.layout;
    let bits = l.format.significant_bits().unwrap_or(16);
    let scale = f32::from(1u16 << (12 - bits.min(12)));
    let warmth = blackbody(3_200.0);
    let white: [f32; 3] = std::array::from_fn(|c| {
        1.2 * LIGHT * warmth[c] * model.quantum_efficiency[c] * ms * frame.gain as f32
            / model.electrons_per_count
    });
    let black = f32::from(frame.black) * scale;
    let at = |col: u32, row: u32| {
        let k = row as usize * l.stride + col as usize * 2;
        f32::from(u16::from_le_bytes([frame.data[k], frame.data[k + 1]])) * scale - black
    };
    let (w, h) = (l.width / 2, l.height / 2);
    let step = w.div_ceil(width).max(1);
    let (sw, sh) = (w / step, h / step);
    let mut rgb = Vec::with_capacity((sw * sh * 3) as usize);
    for j in 0..sh {
        for i in 0..sw {
            let mut sum = [0.0f32; 3];
            for dy in 0..step {
                for dx in 0..step {
                    let (c, r) = (2 * (i * step + dx), 2 * (j * step + dy));
                    for (cc, rr) in [(c, r), (c + 1, r), (c, r + 1), (c + 1, r + 1)] {
                        let colour = bayer(cc, rr);
                        let weight = if colour == 1 { 0.5 } else { 1.0 };
                        sum[colour] += weight * at(cc, rr);
                    }
                }
            }
            let n = (step * step) as f32;
            let v: [f32; 3] = std::array::from_fn(|c| sum[c] / n / white[c]);
            rgb.extend_from_slice(&tetera_sim::he::srgb8(v));
        }
    }
    Ok((sw, sh, rgb))
}
