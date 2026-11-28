//! The recorder and ingest (P1.15c, ADR-0021): the virtual camera at full resolution, 12 bits,
//! at its full rate, into the frame bus; every frame onto the scratch disk through the
//! recorder, a lossless subscriber; then the frame log ingested into a dataset as a compute
//! job, and its first frame read back against the log's.
//!
//! Run it with `cargo run --release -p tetera-acquire --features cuda --example recorder`;
//! `-- --smoke` runs a small sensor on the CPU briefly.

use serde_json::json;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetera_acquire::{FrameBus, Recorder, RecorderOptions};
use tetera_core::frame::meta::LightState;
use tetera_core::{Error, Result, StagePos, Um};
use tetera_hal::camera::{Camera, CameraConfig};
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue, ObjectiveMount};
use tetera_jobs::{JobKind, JobService, JobsConfig};
use tetera_sim::camera::{Backend, Scene, VirtualCamera, VirtualCameraConfig};
use tetera_sim::light::CHANNEL;
use tetera_sim::optics::Sensor;
use tetera_sim::{
    HeSample, Rate, SectionConfig, SimClock, VirtualIllumination, VirtualStage, VirtualStageConfig,
};
use tetera_store::{DatasetSpec, FrameLog, Samples, Store};

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/recorder");
    let _ = std::fs::remove_dir_all(&out);
    let (camera, what) = camera(smoke)?;
    camera.configure(&CameraConfig {
        mode: 0,
        bit_depth: 12,
        exposure: Duration::from_millis(5),
        ..camera.config()
    })?;
    let layout = camera.capabilities().layout(&camera.config())?;
    let rate = 1.0 / camera.timing()?.interval.as_secs_f64();
    // The mode's first frame makes its buffers on the GPU: taken before the recording.
    drop(camera.snap(Duration::from_secs(30))?);
    let frame_mb = layout.row_bytes() as f64 * f64::from(layout.height) / 1e6;
    println!(
        "The virtual camera {what}: {} × {}, 12 bits in 16, {rate:.2} fps, {frame_mb:.1} MB a frame: {:.0} MB/s.",
        layout.width,
        layout.height,
        rate * frame_mb
    );

    // Every frame onto the scratch disk.
    let bus = FrameBus::new();
    let log_dir = out.join("scratch").join("log");
    let recorder = Recorder::start(
        &bus,
        &log_dir,
        layout.format,
        layout.width,
        layout.height,
        RecorderOptions::default(),
    )?;
    let seconds = if smoke { 1.0 } else { 4.0 };
    camera.start(bus.clone())?;
    std::thread::sleep(Duration::from_secs_f64(seconds));
    camera.stop()?;
    let published = bus.published();
    let stats = recorder.stop()?;
    println!(
        "Recorded {} of the {published} frames published in {seconds} s: {:.0} MB/s onto {}; {} missing, the camera held {} times.",
        stats.frames,
        stats.rate() / 1e6,
        log_dir.display(),
        stats.missing,
        stats.waits
    );

    // Ingest, as a compute job.
    let root = out.join("data");
    let store = Arc::new(Store::open(&root)?);
    store.create_project("Recordings")?;
    let jobs = JobService::open(&root.join("tetera.db"), JobsConfig::default(), None)?;
    let into = store.clone();
    let dir = log_dir.clone();
    let id = jobs.submit(JobKind::Compute, "ingest", 0, move |ctx| {
        let log = FrameLog::open(&dir)?;
        let spec = DatasetSpec {
            name: "a recording".into(),
            kind: "recording".into(),
            format: log.header().format,
            width: log.header().width,
            height: log.header().height,
            frames: log.len() as u64,
            pixel_size_um: 0.12,
            interval_s: 1.0 / rate,
            instrument: json!({"camera": "virtual MIChrome 20", "mode": 0, "bit_depth": 12}),
        };
        let info = tetera_store::ingest(&into, "Recordings", &log, spec, |done, of| {
            ctx.check()?;
            ctx.progress(done as f64 / of as f64, format!("frame {done} of {of}"));
            Ok(())
        })?;
        Ok(json!({"dataset": info.id, "frames": info.frames, "bytes": info.bytes}))
    })?;
    let started = Instant::now();
    let job = jobs.wait(&id, Duration::from_secs(600))?;
    let took = started.elapsed().as_secs_f64();
    let result = job
        .result
        .ok_or_else(|| Error::internal(format!("ingest {}: {:?}", job.state.name(), job.error)))?;
    let dataset = result["dataset"].as_str().unwrap_or_default().to_string();
    let bytes = result["bytes"].as_u64().unwrap_or(0);
    println!(
        "Ingested {} frames into dataset {dataset} in {took:.1} s, {:.0} MB/s of raw frames; {:.0} MB on disk.",
        result["frames"],
        stats.bytes as f64 / took / 1e6,
        bytes as f64 / 1e6
    );
    let log = FrameLog::open(&log_dir)?;
    let (_, raw) = log.read(0)?;
    let Samples::U16(first) = store.read(&dataset)?.read(0)? else {
        return Err(Error::internal("not 16 bits"));
    };
    let same = raw
        .as_chunks::<2>()
        .0
        .iter()
        .zip(&first)
        .all(|(b, v)| u16::from_le_bytes(*b) == *v);
    println!(
        "Its first frame against the log's: {}.",
        if same {
            "the same, sample for sample"
        } else {
            "DIFFERENT"
        }
    );
    if smoke {
        let _ = std::fs::remove_dir_all(&out);
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// The virtual camera over the H&E sample at 20×, in real time, still over tissue: on the GPU
/// when there is one and it isn't a smoke run.
fn camera(smoke: bool) -> Result<(Arc<VirtualCamera>, &'static str)> {
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
        stage,
        objectives: Arc::new(mount),
        light: Arc::new(light),
        sample: HeSample::new(7, SectionConfig::default())?,
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
        return Ok((Arc::new(camera), "on the GPU"));
    }
    let camera = VirtualCamera::new(
        scene,
        VirtualCameraConfig {
            sensor: Sensor {
                width: if smoke { 384 } else { 912 },
                height: if smoke { 256 } else { 608 },
                pixel: 2.4,
            },
            section_z: 1_000.0,
            ..VirtualCameraConfig::default()
        },
        Backend::Cpu,
    )?;
    Ok((Arc::new(camera), "on the CPU, a small sensor"))
}
