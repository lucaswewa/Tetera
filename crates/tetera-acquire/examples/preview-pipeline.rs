//! The live view's work, before any network (P1.14a): the virtual camera, in its binned live
//! mode at 8 bits, into the frame bus; two preview sessions of different sizes and the live
//! statistics from it; each session's rate and each stage's time; and a few JPEGs written to
//! `target/preview-pipeline/`. With `--features cuda` and a GPU, the camera and the previews
//! run there; otherwise on the CPU, with a smaller sensor.
//!
//! Run it with `cargo run --release -p tetera-acquire --features cuda --example preview-pipeline`;
//! `-- --smoke` runs a small sensor briefly.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tetera_acquire::{
    FrameBus, LiveStatistics, LiveStats, Preview, PreviewParams, Session, StatsParams,
};
use tetera_compute::Compute;
use tetera_core::frame::meta::LightState;
use tetera_core::{Error, Result, StagePos, Um};
use tetera_hal::camera::{Camera, CameraConfig};
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue, ObjectiveMount};
use tetera_hal::stage::{Axis, Stage};
use tetera_sim::camera::{Backend, Scene, VirtualCamera, VirtualCameraConfig};
use tetera_sim::light::CHANNEL;
use tetera_sim::optics::{Sensor, blackbody};
use tetera_sim::{
    HeSample, Rate, SectionConfig, SimClock, VirtualIllumination, VirtualStage, VirtualStageConfig,
};

/// What one session made: its previews, and a few of their JPEGs.
#[derive(Default)]
struct Made {
    previews: Vec<Preview>,
}

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/preview-pipeline");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out)
        .map_err(|e| Error::internal(format!("{}: {e}", out.display())))?;
    let (camera, stage, what) = camera(smoke)?;
    camera.configure(&CameraConfig {
        mode: 1,
        bit_depth: 8,
        exposure: Duration::from_millis(5),
        ..camera.config()
    })?;
    let mode = &camera.capabilities().modes[1];
    println!(
        "The virtual camera {what}, mode 1 ({} × {}) in 8 bits, its fastest {} fps.",
        mode.width, mode.height, mode.max_fps
    );
    let compute = Arc::new(Compute::detect()?);
    println!("Previews on the {}.", compute.kind());
    // White balance for the light's 3,200 K and the sensor's quantum efficiency.
    let warmth = blackbody(3_200.0);
    let qe = tetera_sim::SensorModel::default().quantum_efficiency;
    let green = f64::from(warmth[1] * qe[1]);
    let white_balance: [f64; 3] = std::array::from_fn(|c| green / f64::from(warmth[c] * qe[c]));

    let bus = FrameBus::new();
    let sizes = if smoke {
        [(320, 213), (160, 107)]
    } else {
        [(1_920, 1_280), (640, 427)]
    };
    let mut sessions = Vec::new();
    let mut made = Vec::new();
    for (k, (width, height)) in sizes.into_iter().enumerate() {
        let kept = Arc::new(Mutex::new(Made::default()));
        let into = kept.clone();
        let session = Session::start(
            &bus,
            compute.clone(),
            PreviewParams {
                width,
                height,
                quality: if k == 0 { 80 } else { 70 },
                white_balance,
                ..PreviewParams::default()
            },
            move |p| {
                if let Ok(mut m) = into.lock() {
                    m.previews.push(p);
                }
            },
        )?;
        sessions.push(session);
        made.push(kept);
    }
    let statistics: Arc<Mutex<Vec<LiveStatistics>>> = Arc::default();
    let into = statistics.clone();
    let stats = LiveStats::start(
        &bus,
        Duration::from_millis(100),
        StatsParams::default(),
        move |s| {
            if let (Ok(s), Ok(mut all)) = (s, into.lock()) {
                all.push(s);
            }
        },
    )?;

    let seconds = if smoke { 1.0 } else { 3.0 };
    println!("\nStreaming for {seconds} s, the stage jogging along x at 200 µm/s:");
    camera.start(bus.clone())?;
    stage.jog(Axis::X, 200.0)?;
    let started = Instant::now();
    std::thread::sleep(Duration::from_secs_f64(seconds));
    let rates: Vec<f64> = sessions.iter().map(|s| s.stats().rate).collect();
    camera.stop()?;
    let elapsed = started.elapsed().as_secs_f64();
    stage.stop_jog()?;
    let published = bus.published();
    println!(
        "  the camera: {published} frames in {elapsed:.1} s, {:.1} fps",
        published as f64 / elapsed
    );
    for (k, ((session, kept), rate)) in sessions.iter().zip(&made).zip(rates).enumerate() {
        let s = session.stats();
        let p = session.params();
        println!(
            "  preview {}: {} × {} viewport, {} previews, {rate:.1} a second; {} frames gave way to newer ones, {} failed",
            k + 1,
            p.width,
            p.height,
            s.previews,
            s.replaced,
            s.failed
        );
        println!(
            "    each stage (mean over the last second): frame to mosaic {:.1} ms, the live recipe {:.1} ms, JPEG {:.1} ms; {:.1} ms in all",
            s.mean.convert, s.mean.isp, s.mean.jpeg, s.mean.total
        );
        let kept = kept
            .lock()
            .map_err(|_| Error::internal("a session's thread panicked"))?;
        let every = (kept.previews.len() / 3).max(1);
        for (n, p) in kept.previews.iter().step_by(every).take(3).enumerate() {
            let path = out.join(format!("preview-{}-{n}.jpg", k + 1));
            std::fs::write(&path, &p.jpeg)
                .map_err(|e| Error::internal(format!("{}: {e}", path.display())))?;
        }
        if let Some(p) = kept.previews.last() {
            println!(
                "    the last: frame {}, {} × {}, {:.0} KiB",
                p.meta.seq,
                p.width,
                p.height,
                p.jpeg.len() as f64 / 1024.0
            );
        }
    }
    let all = statistics
        .lock()
        .map_err(|_| Error::internal("the statistics' thread panicked"))?;
    if let Some(last) = all.last() {
        let total: u64 = last.histogram[1].iter().sum();
        let mean_bin = last.histogram[1]
            .iter()
            .enumerate()
            .map(|(b, n)| b as f64 * *n as f64)
            .sum::<f64>()
            / total.max(1) as f64;
        println!(
            "  statistics: {} in {elapsed:.1} s ({:.1} a second); the last, frame {}: green's mean bin {mean_bin:.0} of 256, {:.3} % saturated, focus {:.2e}, green {:.1} ± {:.1} counts",
            all.len(),
            all.len() as f64 / elapsed,
            last.seq,
            last.saturated_fraction * 100.0,
            last.focus,
            last.roi[1].mean,
            last.roi[1].std
        );
    }
    drop(all);
    stats.stop();
    for session in sessions {
        session.stop();
    }
    println!("\nWrote a few previews to {}", out.display());
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// The virtual camera over the H&E sample at 20×, in real time, the stage in focus over
/// tissue: on the GPU when there is one and it isn't a smoke run.
fn camera(smoke: bool) -> Result<(Arc<VirtualCamera>, Arc<VirtualStage>, &'static str)> {
    let clock = Arc::new(SimClock::new(Rate::RealTime)?);
    let stage = Arc::new(VirtualStage::new(
        clock,
        VirtualStageConfig {
            start: StagePos::new(Um(2_000.0), Um(2_000.0), Um(1_002.0)),
            ..VirtualStageConfig::default()
        },
    )?);
    let mount = ManualObjectiveMount::new(ObjectiveCatalogue::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/objectives.toml"),
    )?);
    mount.fit(Some("20x"))?;
    let light = VirtualIllumination::new(1_000.0, 0.05);
    light.set(
        CHANNEL,
        LightState {
            on: true,
            intensity: 1.0,
            colour_temperature_k: Some(3_200.0),
        },
    )?;
    let scene = Scene {
        stage: stage.clone(),
        objectives: Arc::new(mount),
        light: Arc::new(light),
        sample: HeSample::new(7, SectionConfig::default())?,
    };
    let small = Sensor {
        width: if smoke { 384 } else { 912 },
        height: if smoke { 256 } else { 608 },
        pixel: 2.4,
    };
    #[cfg(feature = "cuda")]
    if !smoke && let Ok(gpu) = tetera_compute::CudaBackend::new(0) {
        let budgets = tetera_compute::budget::VramBudgets::new(3 << 30);
        let camera = VirtualCamera::new(
            scene,
            VirtualCameraConfig {
                section_z: 1_000.0,
                ..VirtualCameraConfig::default()
            },
            Backend::Gpu {
                gpu: Arc::new(gpu),
                budget: budgets.reserve("camera", 3 << 30)?,
            },
        )?;
        return Ok((Arc::new(camera), stage, "on the GPU, 5472 × 3648"));
    }
    let camera = VirtualCamera::new(
        scene,
        VirtualCameraConfig {
            sensor: small,
            section_z: 1_000.0,
            ..VirtualCameraConfig::default()
        },
        Backend::Cpu,
    )?;
    Ok((Arc::new(camera), stage, "on the CPU, a small sensor"))
}
