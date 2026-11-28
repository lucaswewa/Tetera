//! Capture (P1.15d, D5): one frame, N frames at an interval, or a burst at the camera's full
//! rate, into a dataset. The camera is switched from its binned live mode to full resolution,
//! and back, each switch timed. Cancelled or failed, a capture leaves no dataset, and the
//! camera as it was.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetera_acquire::{FrameBus, Recorder, RecorderOptions};
use tetera_core::{Error, Result};
use tetera_hal::camera::{Camera, CameraConfig, FrameSink};
use tetera_store::{DatasetInfo, DatasetSpec, FrameLog, Store};

/// How long a snap may take, the mode's first frame included.
pub(crate) const SNAP: Duration = Duration::from_secs(30);

/// What a capture takes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CaptureKind {
    /// One frame.
    Single,
    /// `count` frames, `interval_ms` apart, each snapped.
    Series {
        /// How many.
        count: u32,
        /// From one frame's start to the next's, ms.
        interval_ms: f64,
    },
    /// `count` frames at the camera's full rate, through the recorder.
    Burst {
        /// How many.
        count: u32,
    },
}

impl CaptureKind {
    /// The frames it takes.
    pub fn frames(&self) -> u32 {
        match *self {
            Self::Single => 1,
            Self::Series { count, .. } | Self::Burst { count } => count,
        }
    }
}

/// A capture's request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CaptureRequest {
    /// The project the dataset goes into.
    pub project: String,
    /// The dataset's name.
    pub name: String,
    /// What to take.
    pub kind: CaptureKind,
    /// The camera's mode for it: 0, full resolution, when not given.
    #[serde(default)]
    pub mode: Option<u32>,
    /// Its bits: 12 when not given.
    #[serde(default)]
    pub bit_depth: Option<u8>,
    /// Its exposure, ms: the live view's when not given.
    #[serde(default)]
    pub exposure_ms: Option<f64>,
}

/// What a capture did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CaptureResult {
    /// The dataset it made.
    pub dataset: DatasetInfo,
    /// The switch to the capture's mode, its first frame made, ms.
    pub switch_ms: f64,
    /// The switch back, the live view restarted if it ran, ms.
    pub back_ms: f64,
    /// Frames the camera dropped during a burst: 0 when none was lost.
    pub missing: u64,
}

/// What a capture works with.
pub struct Rig<'a> {
    /// The camera.
    pub camera: &'a dyn Camera,
    /// The frame bus a burst records from.
    pub bus: &'a Arc<FrameBus>,
    /// Where datasets go.
    pub store: &'a Store,
    /// Where a burst's frame log goes before it is ingested.
    pub scratch: &'a Path,
    /// Where the live view's frames go, to restart it after the capture.
    pub live: Option<Arc<dyn FrameSink>>,
    /// A sensor pixel's size at the sample, µm, unbinned, through the objective fitted: a
    /// binned mode's pixels are larger.
    pub pixel_size_um: f64,
    /// The instrument as it is, for the dataset's metadata.
    pub instrument: serde_json::Value,
}

impl CaptureRequest {
    /// Whether it can be taken.
    pub fn check(&self) -> Result<()> {
        let bad = match self.kind {
            CaptureKind::Single => false,
            CaptureKind::Series { count, interval_ms } => {
                count == 0 || !(interval_ms.is_finite() && interval_ms >= 0.0)
            }
            CaptureKind::Burst { count } => count == 0,
        };
        let exposure = self
            .exposure_ms
            .is_some_and(|ms| !(ms.is_finite() && ms > 0.0));
        if bad || exposure {
            return Err(Error::invalid_argument(format!(
                "a capture of {:?}, its exposure {:?} ms: a frame or more, an interval of 0 or more, an exposure above 0",
                self.kind, self.exposure_ms
            )));
        }
        Ok(())
    }
}

/// Takes `request` with `rig`. `step(done, message)` is called as it goes, `done` from 0 to
/// 1: an error from it cancels the capture.
pub fn capture(
    rig: &Rig<'_>,
    request: &CaptureRequest,
    step: &mut dyn FnMut(f64, &str) -> Result<()>,
) -> Result<CaptureResult> {
    request.check()?;
    // Before the camera is touched: a burst would be recorded only to be refused.
    rig.store.project(&request.project)?;
    let camera = rig.camera;
    let before = camera.config();
    let streaming = camera.is_streaming();
    let switching = Instant::now();
    if streaming {
        camera.stop()?;
    }
    let taken = to_full(
        camera,
        request.mode.unwrap_or(0),
        request.bit_depth.unwrap_or(12),
        request.exposure_ms,
        &before,
    )
    .and_then(|config| {
        let switch_ms = switching.elapsed().as_secs_f64() * 1e3;
        take(rig, request, &config, step).map(|(dataset, missing)| (dataset, missing, switch_ms))
    });
    // Back as it was, whatever happened.
    let returning = Instant::now();
    let back = restore(camera, &before, streaming, rig.live.as_ref(), rig.bus);
    let back_ms = returning.elapsed().as_secs_f64() * 1e3;
    let (dataset, missing, switch_ms) = taken?;
    back?;
    Ok(CaptureResult {
        dataset,
        switch_ms,
        back_ms,
        missing,
    })
}

/// The camera in `mode` at `bit_depth` bits, the whole sensor unbinned, at `exposure_ms` or
/// as `before`, its first frame made, which builds the mode's buffers: a capture's or a
/// scan's (D5).
pub(crate) fn to_full(
    camera: &dyn Camera,
    mode: u32,
    bit_depth: u8,
    exposure_ms: Option<f64>,
    before: &CameraConfig,
) -> Result<CameraConfig> {
    let exposure = match exposure_ms {
        Some(ms) => Duration::try_from_secs_f64(ms / 1e3)
            .map_err(|e| Error::invalid_argument(format!("an exposure of {ms} ms: {e}")))?,
        None => before.exposure,
    };
    let config = CameraConfig {
        mode,
        bit_depth,
        roi: None,
        binning: Default::default(),
        exposure,
        ..before.clone()
    };
    camera.configure(&config)?;
    drop(camera.snap(SNAP)?);
    Ok(config)
}

/// The camera back to `before`, and the live view restarted if it ran: into `live`, or else
/// the frame bus.
pub(crate) fn restore(
    camera: &dyn Camera,
    before: &CameraConfig,
    streaming: bool,
    live: Option<&Arc<dyn FrameSink>>,
    bus: &Arc<FrameBus>,
) -> Result<()> {
    if camera.is_streaming() {
        camera.stop()?;
    }
    camera.configure(before)?;
    if streaming {
        let sink: Arc<dyn FrameSink> = match live {
            Some(live) => live.clone(),
            None => bus.clone(),
        };
        camera.start(sink)?;
    }
    Ok(())
}

/// The dataset's description.
fn spec(rig: &Rig<'_>, request: &CaptureRequest, config: &CameraConfig) -> Result<DatasetSpec> {
    let capabilities = rig.camera.capabilities();
    let layout = capabilities.layout(config)?;
    // The first mode reads the whole sensor; a binned one's pixels cover more of the sample.
    let whole = capabilities.modes.first().map_or(layout.width, |m| m.width);
    let pixel_size_um = rig.pixel_size_um * f64::from(whole) / f64::from(layout.width);
    let interval_s = match request.kind {
        CaptureKind::Series { interval_ms, .. } => interval_ms / 1e3,
        _ => 0.0,
    };
    Ok(DatasetSpec {
        name: request.name.clone(),
        kind: "capture".into(),
        format: layout.format,
        width: layout.width,
        height: layout.height,
        frames: u64::from(request.kind.frames()),
        pixel_size_um,
        interval_s,
        z_step_um: None,
        instrument: serde_json::json!({
            "instrument": rig.instrument,
            "camera": config,
            "capture": request.kind,
        }),
    })
}

/// The frames, into a dataset: the frames the camera dropped, for a burst.
fn take(
    rig: &Rig<'_>,
    request: &CaptureRequest,
    config: &CameraConfig,
    step: &mut dyn FnMut(f64, &str) -> Result<()>,
) -> Result<(DatasetInfo, u64)> {
    let spec = spec(rig, request, config)?;
    let n = request.kind.frames();
    match request.kind {
        CaptureKind::Single | CaptureKind::Series { .. } => {
            let interval = match request.kind {
                CaptureKind::Series { interval_ms, .. } => {
                    Duration::from_secs_f64(interval_ms / 1e3)
                }
                _ => Duration::ZERO,
            };
            let mut writer = rig.store.create_dataset(&request.project, spec)?;
            let mut next = Instant::now();
            for i in 0..n {
                // Waits for the frame's time, saying where it is so a cancel is heard.
                while Instant::now() < next {
                    step(
                        f64::from(i) / f64::from(n),
                        &format!("frame {} of {n} due", i + 1),
                    )?;
                    std::thread::sleep((next - Instant::now()).min(Duration::from_millis(10)));
                }
                step(
                    f64::from(i) / f64::from(n),
                    &format!("frame {} of {n}", i + 1),
                )?;
                writer.append(&rig.camera.snap(SNAP)?)?;
                next += interval;
            }
            step(1.0, "written")?;
            Ok((writer.finish()?, 0))
        }
        CaptureKind::Burst { count } => burst(rig, request, count, spec, step),
    }
}

/// `count` frames at the full rate: the recorder, then ingest; the frame log removed.
fn burst(
    rig: &Rig<'_>,
    request: &CaptureRequest,
    count: u32,
    spec: DatasetSpec,
    step: &mut dyn FnMut(f64, &str) -> Result<()>,
) -> Result<(DatasetInfo, u64)> {
    let dir = rig
        .scratch
        .join(format!("capture-{}", uuid::Uuid::now_v7()));
    let removed = |r: Result<(DatasetInfo, u64)>| {
        let _ = std::fs::remove_dir_all(&dir);
        r
    };
    let recorder = Recorder::start(
        rig.bus,
        &dir,
        spec.format,
        spec.width,
        spec.height,
        RecorderOptions::default(),
    )?;
    let from = rig.bus.published();
    let want = u64::from(count);
    let recorded = rig.camera.start(rig.bus.clone()).and_then(|()| {
        loop {
            let got = rig.bus.published().saturating_sub(from);
            if got >= want {
                break Ok(());
            }
            if let Err(e) = step(
                0.5 * got as f64 / want as f64,
                &format!("frame {got} of {count}"),
            ) {
                break Err(e);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    });
    let stopped = rig.camera.stop();
    let stats = recorder.stop();
    if let Err(e) = recorded.and(stopped) {
        return removed(Err(e));
    }
    let stats = match stats {
        Ok(stats) => stats,
        Err(e) => return removed(Err(e)),
    };
    let ingested = FrameLog::open(&dir).and_then(|log| {
        tetera_store::ingest_first(rig.store, &request.project, &log, want, spec, |done, of| {
            step(
                0.5 + 0.5 * done as f64 / of.max(1) as f64,
                &format!("ingesting frame {done} of {of}"),
            )
        })
    });
    removed(ingested.map(|d| (d, stats.missing)))
}
