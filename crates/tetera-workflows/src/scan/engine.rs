//! The stop-and-go engine (P1.17a, ADR-0022): for each position of a plan, a move, the
//! stage's report that it is still, an autofocus if the schedule says so, and the first frame
//! of the full-resolution stream exposed a settle after, with its position from the stage, the
//! frame written by a thread of its own meanwhile. Pause, resume and cancel are heard between positions and in every wait; a
//! cancelled scan keeps the tiles it took, as a valid, partial dataset.

use super::{Pattern, Plan, Timing, estimate, move_time, positions};
use crate::autofocus::{AutofocusRequest, FocusRig, autofocus, depth_of_field};
use crate::capture::{restore, to_full};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use tetera_acquire::FrameBus;
use tetera_core::frame::meta::FrameMeta;
use tetera_core::{Error, Frame, Result, StagePos, Timestamp, Um};
use tetera_hal::camera::{Camera, CameraConfig, FrameSink};
use tetera_hal::objective::Objective;
use tetera_hal::stage::Stage;
use tetera_store::{DatasetInfo, DatasetSpec, DatasetWriter, Store};

/// How long a move may take to end.
const MOVE: Duration = Duration::from_secs(60);

/// How long a frame exposed after the settle may take to come.
const FRAME: Duration = Duration::from_secs(10);

/// The camera's mode and bits for a scan's frames: the whole sensor, 12 bits.
const MODE: u32 = 0;
const BITS: u8 = 12;

/// From a move's end to the stage's report that it is still, on average, ms: half the
/// virtual stage's report period, and its latency.
const REPORT_MS: f64 = 15.0;

/// The switch to full resolution and back, ms (P1.15d).
const SWITCH_MS: f64 = 700.0;

/// The positions an autofocus measures (P1.16a).
const AUTOFOCUS_POSITIONS: f64 = 26.0;

/// When a scan autofocuses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "at", rename_all = "snake_case", deny_unknown_fields)]
pub enum AutofocusAt {
    /// Never: every tile at the scan's z.
    #[default]
    Never,
    /// At the first tile; the rest at its focus.
    First,
    /// At every `tiles`th tile, the first included; those after at its focus.
    Every {
        /// How many tiles apart.
        tiles: u32,
    },
}

impl AutofocusAt {
    /// Whether tile `i` autofocuses.
    pub fn at(self, i: usize) -> bool {
        match self {
            Self::Never => false,
            Self::First => i == 0,
            Self::Every { tiles } => tiles > 0 && i.is_multiple_of(tiles as usize),
        }
    }
}

/// A scan's request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScanRequest {
    /// The project the dataset goes into.
    pub project: String,
    /// The dataset's name.
    pub name: String,
    /// What to cover.
    pub pattern: Pattern,
    /// When to autofocus: a snake's only.
    #[serde(default)]
    pub autofocus: AutofocusAt,
    /// The settle after each move, ms: 50 when not given.
    #[serde(default)]
    pub settle_ms: Option<f64>,
    /// The frames' exposure, ms: the live view's when not given.
    #[serde(default)]
    pub exposure_ms: Option<f64>,
    /// The disk's throughput the estimate assumes, MB/s: 120 when not given.
    #[serde(default)]
    pub write_mb_s: Option<f64>,
}

impl ScanRequest {
    fn settle(&self) -> f64 {
        self.settle_ms.unwrap_or(50.0)
    }

    /// Whether it can be run.
    pub fn check(&self) -> Result<()> {
        let fine = self.settle().is_finite()
            && self.settle() >= 0.0
            && self.exposure_ms.is_none_or(|ms| ms.is_finite() && ms > 0.0)
            && self.write_mb_s.is_none_or(|v| v.is_finite() && v > 0.0)
            && self.autofocus != AutofocusAt::Every { tiles: 0 };
        let focusing = self.autofocus != AutofocusAt::Never;
        if !fine || (focusing && !matches!(self.pattern, Pattern::Snake { .. })) {
            return Err(Error::invalid_argument(format!(
                "a scan settling {:?} ms, exposed {:?} ms, written at {:?} MB/s, autofocusing {:?}: a settle of 0 or more, an exposure and a throughput above 0, autofocus every tile or more, and a snake's only",
                self.settle_ms, self.exposure_ms, self.write_mb_s, self.autofocus
            )));
        }
        Ok(())
    }
}

/// A tile, or a plane, as taken.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Tile {
    /// Where the plan put it; at the last autofocus's z, after one.
    pub planned: StagePos,
    /// Where the stage said it was, still, through the frame's exposure.
    pub recorded: StagePos,
    /// Where the camera says it was, if it knows: the simulator's truth.
    pub truth: Option<StagePos>,
    /// When the stage reported itself still, by its clock.
    pub still_at: Timestamp,
    /// When the frame's exposure started.
    pub exposure_start: Timestamp,
}

/// What a scan did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScanResult {
    /// The dataset it made.
    pub dataset: DatasetInfo,
    /// Its plan.
    pub plan: Plan,
    /// Each tile, as taken.
    pub tiles: Vec<Tile>,
    /// How long it took, s.
    pub seconds: f64,
}

/// What a scan works with.
pub struct ScanRig<'a> {
    /// The camera.
    pub camera: &'a dyn Camera,
    /// The stage.
    pub stage: &'a dyn Stage,
    /// The frame bus, for autofocus's frames.
    pub bus: &'a Arc<FrameBus>,
    /// Where the dataset goes.
    pub store: &'a Store,
    /// Where the live view's frames go, to restart it after the scan.
    pub live: Option<Arc<dyn FrameSink>>,
    /// The objective fitted.
    pub objective: Option<Objective>,
    /// The sensor's pixel pitch, µm.
    pub sensor_pixel_um: f64,
    /// The instrument as it is, for the dataset's metadata.
    pub instrument: serde_json::Value,
}

/// Pause and resume, from outside the scan.
#[derive(Debug, Default)]
pub struct ScanControl {
    paused: Mutex<bool>,
    changed: Condvar,
}

impl ScanControl {
    fn lock(&self) -> MutexGuard<'_, bool> {
        self.paused.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Pauses the scan at its next position.
    pub fn pause(&self) {
        *self.lock() = true;
        self.changed.notify_all();
    }

    /// Resumes it.
    pub fn resume(&self) {
        *self.lock() = false;
        self.changed.notify_all();
    }

    /// Whether it is paused.
    pub fn is_paused(&self) -> bool {
        *self.lock()
    }

    /// Waits while paused, saying so through `step`, which can cancel it.
    fn hold(&self, done: f64, step: &mut dyn FnMut(f64, &str) -> Result<()>) -> Result<()> {
        while self.is_paused() {
            step(done, "paused")?;
            let paused = self.lock();
            if *paused {
                let _ = self
                    .changed
                    .wait_timeout(paused, Duration::from_millis(100))
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
        Ok(())
    }
}

/// The field of view at full resolution through `objective`, µm.
fn field(rig: &ScanRig<'_>, objective: &Objective, config: &CameraConfig) -> Result<[f64; 2]> {
    let layout = rig.camera.capabilities().layout(config)?;
    let pixel = rig.sensor_pixel_um / objective.magnification;
    Ok([
        f64::from(layout.width) * pixel,
        f64::from(layout.height) * pixel,
    ])
}

/// The scan's camera configuration: the whole sensor at 12 bits, the exposure asked for.
pub(super) fn full(request: &ScanRequest, before: &CameraConfig) -> Result<CameraConfig> {
    let exposure = match request.exposure_ms {
        Some(ms) => Duration::try_from_secs_f64(ms / 1e3)
            .map_err(|e| Error::invalid_argument(format!("an exposure of {ms} ms: {e}")))?,
        None => before.exposure,
    };
    Ok(CameraConfig {
        mode: MODE,
        bit_depth: BITS,
        roi: None,
        binning: Default::default(),
        exposure,
        ..before.clone()
    })
}

/// The plan `request` makes from where the stage is, and its estimated time: nothing moves.
pub fn preview(rig: &ScanRig<'_>, request: &ScanRequest) -> Result<Plan> {
    request.check()?;
    let objective = rig.objective.clone().ok_or_else(|| {
        Error::invalid_argument("a scan needs an objective fitted, for its field of view")
    })?;
    let config = full(request, &rig.camera.config())?;
    let field = field(rig, &objective, &config)?;
    let capabilities = rig.stage.capabilities();
    let here = rig.stage.reading()?.position;
    let (positions, grid) = positions(&request.pattern, here, field, &capabilities)?;
    let camera = rig.camera.capabilities();
    let layout = camera.layout(&config)?;
    let interval_ms = camera
        .modes
        .iter()
        .find(|m| m.id == MODE)
        .map_or(0.0, |m| 1e3 / m.fps(BITS));
    // The stream's next exposure, half an interval on average after the settle, then the
    // exposure and its readout, an interval at full resolution.
    let exposure_ms = config.exposure.as_secs_f64() * 1e3;
    let frame_ms = 1.5 * interval_ms.max(exposure_ms) + exposure_ms;
    let bytes = f64::from(layout.width) * f64::from(layout.height) * 2.0;
    let d = depth_of_field(
        objective.magnification,
        objective.numerical_aperture,
        rig.sensor_pixel_um,
    );
    let step = StagePos::new(here.x, here.y, Um(here.z.um() + d));
    let autofocus_ms = AUTOFOCUS_POSITIONS
        * (move_time(here, step, &capabilities)? * 1e3 + REPORT_MS + 1.5 * frame_ms);
    let timing = Timing {
        report_ms: REPORT_MS,
        settle_ms: request.settle(),
        frame_ms,
        write_ms: bytes / (request.write_mb_s.unwrap_or(120.0) * 1e6) * 1e3,
        switch_ms: SWITCH_MS,
        autofocus_ms,
    };
    let estimate_s = estimate(&positions, here, &capabilities, &timing, |i| {
        request.autofocus.at(i)
    })?;
    Ok(Plan {
        pattern: request.pattern,
        positions,
        grid,
        field,
        estimate_s,
        timing,
    })
}

/// Runs `request`'s scan with `rig`, `control` pausing and resuming it. `step(done, message)`
/// is called as it goes, `done` from 0 to 1: an error from it cancels the scan, and the tiles
/// taken are kept as a partial dataset.
pub fn scan(
    rig: &ScanRig<'_>,
    request: &ScanRequest,
    control: &ScanControl,
    step: &mut dyn FnMut(f64, &str) -> Result<()>,
) -> Result<ScanResult> {
    let began = Instant::now();
    let plan = preview(rig, request)?;
    rig.store.project(&request.project)?;
    let camera = rig.camera;
    let before = camera.config();
    let streaming = camera.is_streaming();
    if streaming {
        camera.stop()?;
    }
    // The full-resolution stream, into the bus: each tile its first frame after the settle.
    let ran = to_full(camera, MODE, BITS, request.exposure_ms, &before)
        .and_then(|config| {
            camera.start(rig.bus.clone())?;
            Ok(config)
        })
        .and_then(|config| run(rig, request, &plan, &config, control, step));
    let back = restore(camera, &before, streaming, rig.live.as_ref(), rig.bus);
    let (dataset, tiles) = ran?;
    back?;
    Ok(ScanResult {
        dataset,
        plan,
        tiles,
        seconds: began.elapsed().as_secs_f64(),
    })
}

/// The scan proper, the camera at full resolution: the tiles, and the dataset finished with
/// those taken, whatever stopped it.
fn run(
    rig: &ScanRig<'_>,
    request: &ScanRequest,
    plan: &Plan,
    config: &CameraConfig,
    control: &ScanControl,
    step: &mut dyn FnMut(f64, &str) -> Result<()>,
) -> Result<(DatasetInfo, Vec<Tile>)> {
    let layout = rig.camera.capabilities().layout(config)?;
    let objective = rig.objective.clone();
    let spec = DatasetSpec {
        name: request.name.clone(),
        kind: "scan".into(),
        format: layout.format,
        width: layout.width,
        height: layout.height,
        frames: plan.positions.len() as u64,
        pixel_size_um: rig.sensor_pixel_um / objective.as_ref().map_or(1.0, |o| o.magnification),
        interval_s: 0.0,
        instrument: serde_json::json!({
            "instrument": rig.instrument,
            "camera": config,
            "scan": {
                "request": request,
                "plan": plan,
            },
        }),
    };
    let writer = rig.store.create_dataset(&request.project, spec)?;
    // The writer's thread: two frames deep, so writing overlaps the next move.
    let (frames, queue) = sync_channel::<(Frame, FrameMeta)>(2);
    let writing = std::thread::Builder::new()
        .name("scan writer".into())
        .spawn(move || -> Result<DatasetWriter> {
            let mut writer = writer;
            for (frame, meta) in queue {
                writer.append_as(&frame, meta)?;
            }
            Ok(writer)
        })
        .map_err(|e| Error::internal(format!("the scan's writer won't start: {e}")))?;
    let mut tiles = Vec::with_capacity(plan.positions.len());
    let taken = take(rig, request, plan, control, step, &frames, &mut tiles);
    drop(frames);
    let written = writing
        .join()
        .map_err(|_| Error::internal("the scan's writer panicked"))?;
    // A writer that failed is the cause; else the tiles taken are kept, whatever stopped
    // the scan. None leave no dataset: the writer refuses to finish it, and clears it.
    let finished = written?.finish();
    taken?;
    let dataset = finished?;
    Ok((dataset, tiles))
}

/// Each position: moved to, settled, focused if due, and its frame sent to the writer.
fn take(
    rig: &ScanRig<'_>,
    request: &ScanRequest,
    plan: &Plan,
    control: &ScanControl,
    step: &mut dyn FnMut(f64, &str) -> Result<()>,
    frames: &std::sync::mpsc::SyncSender<(Frame, FrameMeta)>,
    tiles: &mut Vec<Tile>,
) -> Result<()> {
    let n = plan.positions.len();
    let settle = Duration::from_secs_f64(request.settle() / 1e3);
    let mut z = None;
    for (i, &position) in plan.positions.iter().enumerate() {
        let done = i as f64 / n as f64;
        control.hold(done, step)?;
        step(done, &format!("tile {} of {n}", i + 1))?;
        let mut planned = match z {
            Some(z) => StagePos::new(position.x, position.y, z),
            None => position,
        };
        rig.stage.move_to(planned, None)?;
        let mut still_at = rig.stage.wait(MOVE)?.at;
        if request.autofocus.at(i) {
            let focusing = FocusRig {
                camera: rig.camera,
                stage: rig.stage,
                bus: rig.bus,
                live: None,
                objective: rig.objective.clone(),
                sensor_pixel_um: rig.sensor_pixel_um,
            };
            let focused = autofocus(&focusing, &AutofocusRequest::default(), &mut |_, m| {
                step(done, &format!("tile {} of {n}: autofocus, {m}", i + 1))
            })?;
            if let Some(found) = focused.focus.z {
                z = Some(Um(found));
                planned = StagePos::new(planned.x, planned.y, Um(found));
            }
            still_at = rig.stage.wait(MOVE)?.at;
        }
        let recorded = rig.stage.reading()?.position;
        let frame = rig
            .bus
            .first_after(still_at.saturating_add(settle), FRAME)?;
        let after = rig.stage.reading()?.position;
        if after != recorded {
            return Err(Error::hardware(format!(
                "the stage moved from {recorded:?} to {after:?} during tile {}'s exposure",
                i + 1
            )));
        }
        let mut meta = frame.meta().clone();
        let truth = meta.position;
        meta.position = Some(recorded);
        tiles.push(Tile {
            planned,
            recorded,
            truth,
            still_at,
            exposure_start: meta.exposure_start,
        });
        frames
            .send((frame, meta))
            .map_err(|_| Error::internal("the scan's writer stopped"))?;
    }
    step(1.0, "written")?;
    Ok(())
}
