//! The virtual camera (P1.12c, ADR-0016): a MIChrome 20 that images the simulated slide
//! through the fitted objective, under the virtual light, wherever the virtual stage has it,
//! at the camera's real modes and rates. Frames are timed on the stage's simulated clock: a
//! frame's rows expose one line time apart, the stage moving meanwhile, and the frame is made
//! once its last row has read out, on the GPU or the CPU.

use crate::clock::SimClock;
use crate::he::HeSample;
use crate::light::{CHANNEL, VirtualIllumination};
use crate::optics::{FieldOfView, MICHROME_20, Motion, Optics, Sensor, bare_field, field_moving};
use crate::sensor::{Exposure, SensorModel, bayer, expose};
use crate::stage::VirtualStage;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;
use tetera_core::frame::format::BayerPattern;
use tetera_core::frame::pool::HostAllocator;
use tetera_core::{Clock, Error, ErrorKind, Frame, FrameMut, FramePool, Rect, Result, Timestamp};
use tetera_hal::camera::{
    Camera, CameraCapabilities, CameraConfig, CameraEvent, CameraMode, FrameSink, Span, Trigger,
};
use tetera_hal::event::Events;
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::ObjectiveMount;

/// One of the MIChrome 20's readout modes, at the rates OScope measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeRate {
    /// Its number, as the driver knows it.
    pub id: u32,
    /// Its name.
    pub name: &'static str,
    /// The sensor's pixels it bins each way.
    pub binning: u32,
    /// Frames a second in 16-bit words.
    pub deep_fps: f64,
    /// Frames a second in 8 bits.
    pub fast_fps: f64,
}

impl ModeRate {
    /// Its shortest frame interval at `bits`: the time its rows take to read out.
    pub fn interval(&self, bits: u8) -> Duration {
        let fps = if bits == 8 {
            self.fast_fps
        } else {
            self.deep_fps
        };
        Duration::from_secs_f64(1.0 / fps)
    }
}

/// The MIChrome 20's modes: the whole sensor, then binned 2 × 2 and 3 × 3.
pub const MODES: [ModeRate; 3] = [
    ModeRate {
        id: 0,
        name: "full",
        binning: 1,
        deep_fps: 7.24,
        fast_fps: 14.49,
    },
    ModeRate {
        id: 1,
        name: "binned 2 × 2",
        binning: 2,
        deep_fps: 26.57,
        fast_fps: 53.14,
    },
    ModeRate {
        id: 2,
        name: "binned 3 × 3",
        binning: 3,
        deep_fps: 33.87,
        fast_fps: 67.26,
    },
];

/// The mode numbered `id`.
pub fn mode(id: u32) -> Result<&'static ModeRate> {
    MODES
        .iter()
        .find(|m| m.id == id)
        .ok_or_else(|| Error::invalid_argument(format!("the camera has no mode {id}")))
}

/// What the camera's SDK does to the raw by default, which the real driver turns off (plan
/// R4): white balance and gamma, then a rescaling to about 58,000 levels above a black offset
/// near 2,000, in 16 bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdkProcessing {
    /// The gamma: 1 is linear.
    pub gamma: f32,
    /// Red's, green's and blue's gains.
    pub white_balance: [f32; 3],
    /// The black offset, in 16-bit levels.
    pub offset: u16,
    /// The levels from the black offset to a full 12-bit count.
    pub levels: u16,
}

impl Default for SdkProcessing {
    /// Linear, unbalanced, 58,000 levels above 2,000.
    fn default() -> Self {
        Self {
            gamma: 1.0,
            white_balance: [1.0; 3],
            offset: 2_000,
            levels: 58_000,
        }
    }
}

impl SdkProcessing {
    /// A 12-bit `count` of colour `colour`, its black level `black`, as the SDK hands it on.
    pub fn apply(&self, count: u16, black: u16, colour: usize) -> u16 {
        let x = (f32::from(count) - f32::from(black)) / (4_095.0 - f32::from(black))
            * self.white_balance[colour];
        let y = x.signum() * x.abs().powf(1.0 / self.gamma);
        (f32::from(self.offset) + y * f32::from(self.levels))
            .round()
            .clamp(0.0, 65_535.0) as u16
    }
}

/// How the virtual camera is made.
#[derive(Debug, Clone, PartialEq)]
pub struct VirtualCameraConfig {
    /// The sensor, whole: the MIChrome 20's, or a smaller one for tests.
    pub sensor: Sensor,
    /// Its physics.
    pub model: SensorModel,
    /// Its rows against the stage's x axis, radians.
    pub mounting_angle: f64,
    /// The noise's seed.
    pub seed: u64,
    /// The stage's z, µm, at which the objective focuses on the section's lower face.
    pub section_z: f64,
    /// The most blur, in sensor pixels, a frame is given: further from focus, the defocus
    /// is held where the blur reaches it, so a frame far from focus takes no longer.
    pub max_blur: f64,
    /// The SDK's processing, to mimic the real raw; linear 12 bits when `None`.
    pub sdk: Option<SdkProcessing>,
    /// Frames in its pool.
    pub pool: usize,
}

impl Default for VirtualCameraConfig {
    /// The MIChrome 20, mounted square, its section in focus at a stage z of 1 mm.
    fn default() -> Self {
        Self {
            sensor: MICHROME_20,
            model: SensorModel::default(),
            mounting_angle: 0.0,
            seed: 1,
            section_z: 1_000.0,
            max_blur: 48.0,
            sdk: None,
            pool: 4,
        }
    }
}

impl VirtualCameraConfig {
    /// Whether it makes a camera.
    pub fn check(&self) -> Result<()> {
        let s = self.sensor;
        if s.width < 6
            || s.height < 6
            || !(s.pixel.is_finite() && s.pixel > 0.0)
            || !self.mounting_angle.is_finite()
            || !self.section_z.is_finite()
            || !(self.max_blur.is_finite() && self.max_blur >= 1.0)
            || self.pool == 0
        {
            return Err(Error::invalid_argument(format!(
                "a virtual camera of {} × {} pixels of {} µm, a blur of {} at most, {} frames: \
                 6 × 6 pixels or more, a blur of a pixel or more, a frame or more",
                s.width, s.height, s.pixel, self.max_blur, self.pool
            )));
        }
        Ok(())
    }
}

/// What the camera looks at: the stage that carries the slide, the objectives, the light,
/// and the sample on the slide to begin with. The stage's clock is the camera's.
#[derive(Clone)]
pub struct Scene {
    /// The stage.
    pub stage: Arc<VirtualStage>,
    /// The objective mount.
    pub objectives: Arc<dyn ObjectiveMount>,
    /// The light.
    pub light: Arc<VirtualIllumination>,
    /// The sample.
    pub sample: HeSample,
}

impl std::fmt::Debug for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scene")
            .field("stage", &self.stage)
            .field("objective", &self.objectives.fitted().map(|o| o.name))
            .finish_non_exhaustive()
    }
}

/// Where the camera makes its frames.
pub enum Backend {
    /// On the CPU: `optics::field_moving`, then `sensor::expose`. Slow at full size; tests
    /// use a small sensor.
    Cpu,
    /// On the GPU, the sample's tiles cached within `budget`.
    #[cfg(feature = "cuda")]
    Gpu {
        /// The GPU.
        gpu: Arc<tetera_compute::CudaBackend>,
        /// Its memory for the tiles.
        budget: tetera_compute::budget::Budget,
    },
}

enum Renderer {
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu(Box<crate::gpu::HeTiles>, Arc<tetera_compute::CudaBackend>),
}

/// When a frame's rows expose: each for `exposure`, each `line_time` after the one above,
/// and a frame every `interval` when free-running.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTiming {
    /// Each row's exposure.
    pub exposure: Duration,
    /// From one row's start to the next's: the mode's readout over its rows.
    pub line_time: Duration,
    /// The rows.
    pub rows: u32,
    /// From one frame's start to the next's: the readout, or the exposure if longer.
    pub interval: Duration,
}

impl FrameTiming {
    /// The timing of `config`'s frames in `mode`, `rows` rows.
    pub fn of(mode: &ModeRate, rows: u32, config: &CameraConfig) -> Self {
        let readout = mode.interval(config.bit_depth);
        Self {
            exposure: config.exposure,
            line_time: readout / rows.max(1),
            rows,
            interval: readout.max(config.exposure),
        }
    }

    /// From the first row's start to the last row's end.
    pub fn span(&self) -> Duration {
        self.line_time * self.rows.saturating_sub(1) + self.exposure
    }
}

/// The camera's state, shared with its stream.
struct State {
    config: CameraConfig,
    seq: u64,
    pool: Option<FramePool>,
}

struct Shared {
    capabilities: CameraCapabilities,
    made: VirtualCameraConfig,
    scene: Scene,
    renderer: Renderer,
    /// The sample on the stage now, if any.
    sample: RwLock<Option<HeSample>>,
    /// The camera's own threads, so its frames don't queue behind other work on rayon's
    /// global pool.
    workers: rayon::ThreadPool,
    clock: Arc<SimClock>,
    state: Mutex<State>,
    events: Events<CameraEvent>,
}

struct Stream {
    stop: Arc<AtomicBool>,
    triggers: Sender<()>,
    thread: JoinHandle<()>,
}

/// The virtual camera.
pub struct VirtualCamera {
    shared: Arc<Shared>,
    stream: Mutex<Option<Stream>>,
}

impl std::fmt::Debug for VirtualCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualCamera")
            .field("model", &self.shared.capabilities.model)
            .field("config", &self.config())
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// How long a wait on the clock lasts before the stop flag is looked at again.
const POLL: Duration = Duration::from_millis(20);

impl VirtualCamera {
    /// A camera looking at `scene`, made as `config` says, rendering on `backend`.
    pub fn new(scene: Scene, config: VirtualCameraConfig, backend: Backend) -> Result<Self> {
        config.check()?;
        let renderer = match backend {
            Backend::Cpu => Renderer::Cpu,
            #[cfg(feature = "cuda")]
            Backend::Gpu { gpu, budget } => Renderer::Gpu(
                Box::new(crate::gpu::HeTiles::new(gpu.clone(), scene.sample, budget)?),
                gpu,
            ),
        };
        let deep = if config.sdk.is_some() { 16 } else { 12 };
        let capabilities = CameraCapabilities {
            model: "Tetera virtual MIChrome 20".into(),
            pattern: BayerPattern::Rggb,
            modes: MODES
                .iter()
                .map(|m| {
                    let s = config.sensor.binned(m.binning);
                    CameraMode {
                        id: m.id,
                        name: m.name.into(),
                        width: s.width,
                        height: s.height,
                        bit_depths: vec![deep, 8],
                        max_fps: m.fast_fps,
                    }
                })
                .collect(),
            binnings: Vec::new(),
            roi_step: None,
            exposure_ms: Span {
                min: 0.1,
                max: 10_000.0,
            },
            gain: Span {
                min: 1.0,
                max: 16.0,
            },
            triggers: vec![Trigger::FreeRun, Trigger::Software],
        };
        let state = State {
            config: capabilities.default_config(),
            seq: 0,
            pool: None,
        };
        let workers = rayon::ThreadPoolBuilder::new()
            .thread_name(|k| format!("virtual camera {k}"))
            .build()
            .map_err(|e| Error::internal(format!("the virtual camera's threads: {e}")))?;
        Ok(Self {
            shared: Arc::new(Shared {
                capabilities,
                made: config,
                clock: scene.stage.clock().clone(),
                sample: RwLock::new(Some(scene.sample)),
                scene,
                renderer,
                workers,
                state: Mutex::new(state),
                events: Events::default(),
            }),
            stream: Mutex::new(None),
        })
    }

    /// Its clock: the stage's.
    pub fn clock(&self) -> &Arc<SimClock> {
        &self.shared.clock
    }

    /// The sample on the stage, if there is one.
    pub fn sample(&self) -> Option<HeSample> {
        *self
            .shared
            .sample
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Puts `sample` on the stage, or takes the slide away: frames show bare glass. Takes
    /// effect from the next frame.
    pub fn set_sample(&self, sample: Option<HeSample>) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Renderer::Gpu(tiles, _) = &self.shared.renderer {
            tiles.set_sample(sample)?;
        }
        *self
            .shared
            .sample
            .write()
            .unwrap_or_else(PoisonError::into_inner) = sample;
        Ok(())
    }

    /// The timing of its frames as it is set now.
    pub fn timing(&self) -> Result<FrameTiming> {
        let config = self.config();
        self.shared.timing(&config)
    }
}

impl Shared {
    fn timing(&self, config: &CameraConfig) -> Result<FrameTiming> {
        let m = mode(config.mode)?;
        Ok(FrameTiming::of(
            m,
            self.made.sensor.binned(m.binning).height,
            config,
        ))
    }

    /// The pool for the configuration, made on first use after each configure.
    fn pool(&self) -> Result<FramePool> {
        let mut s = lock(&self.state);
        if s.pool.is_none() {
            let layout = self.capabilities.layout(&s.config)?;
            let pool = match &self.renderer {
                Renderer::Cpu => FramePool::new(layout, self.made.pool, &HostAllocator)?,
                #[cfg(feature = "cuda")]
                Renderer::Gpu(_, gpu) => {
                    let pinned: &dyn tetera_core::frame::pool::Allocator = &gpu.pinned_allocator();
                    FramePool::new(layout, self.made.pool, pinned)?
                }
            };
            s.pool = Some(pool);
        }
        Ok(s.pool.clone().expect("made above"))
    }

    /// The next frame's number.
    fn next_seq(&self) -> u64 {
        let mut s = lock(&self.state);
        let seq = s.seq;
        s.seq += 1;
        seq
    }

    /// Frame `seq`, its first row starting at `start`, made into `frame` on the camera's
    /// own threads.
    fn render(
        &self,
        config: &CameraConfig,
        start: Timestamp,
        seq: u64,
        dropped: u64,
        frame: FrameMut,
    ) -> Result<Frame> {
        self.workers
            .install(|| self.render_here(config, start, seq, dropped, frame))
    }

    fn render_here(
        &self,
        config: &CameraConfig,
        start: Timestamp,
        seq: u64,
        dropped: u64,
        mut frame: FrameMut,
    ) -> Result<Frame> {
        let m = mode(config.mode)?;
        let sensor = self.made.sensor.binned(m.binning);
        let timing = FrameTiming::of(m, sensor.height, config);
        let span = timing.span();
        let stage = &self.scene.stage;
        let first = stage.position_at(start);
        let last = stage.position_at(start.saturating_add(span));
        let middle = stage.position_at(start.saturating_add(span / 2));
        let secs = span.as_secs_f64();
        let velocity = if secs > 0.0 {
            (
                (last.x.um() - first.x.um()) / secs,
                (last.y.um() - first.y.um()) / secs,
            )
        } else {
            (0.0, 0.0)
        };
        // With no objective fitted, nothing reaches the sensor but its own noise.
        let fitted = self.scene.objectives.fitted();
        let (magnification, na) = match &fitted {
            Some(o) => (o.magnification, o.numerical_aperture),
            None => (10.0, 0.28),
        };
        let optics = Optics {
            sensor,
            ..Optics::new(magnification, na, self.made.mounting_angle)?
        };
        let mut light = self.scene.light.light();
        if fitted.is_none() {
            light.intensity = 0.0;
        }
        let sample = *self.sample.read().unwrap_or_else(PoisonError::into_inner);
        let thickness = sample.map_or(0.0, |s| f64::from(s.section().thickness));
        let reach = optics.defocus_for_blur(self.made.max_blur);
        let focus = (middle.z.um() - self.made.section_z).clamp(-reach, thickness + reach);
        let fov = FieldOfView {
            centre: (first.x.um(), first.y.um()),
            optics,
        };
        let motion = Motion {
            velocity,
            exposure: timing.exposure.as_secs_f64(),
            line_time: timing.line_time.as_secs_f64(),
        };
        let sdk = self.made.sdk;
        let exposure = Exposure {
            time: timing.exposure,
            gain: config.gain as f32,
            bits: if sdk.is_none() && config.bit_depth == 8 {
                8
            } else {
                12
            },
            binning: m.binning,
        };
        let model = self.made.model;
        let width = sensor.width as usize;
        let stride = frame.layout().stride;
        match &self.renderer {
            Renderer::Cpu => {
                let field = match &sample {
                    Some(sample) => field_moving(sample, &fov, &motion, &light, focus)?,
                    None => bare_field(&fov, &light),
                };
                let raw = expose(
                    &field.rgb,
                    sensor.width,
                    sensor.height,
                    &model,
                    &exposure,
                    self.made.seed,
                    seq,
                )?;
                frame
                    .data_mut()
                    .par_chunks_mut(stride)
                    .zip(raw.par_chunks(width))
                    .for_each(|(out, line)| {
                        for (pair, v) in out.as_chunks_mut::<2>().0.iter_mut().zip(line) {
                            *pair = v.to_le_bytes();
                        }
                    });
            }
            #[cfg(feature = "cuda")]
            Renderer::Gpu(tiles, _) => tiles.frame_into(
                &fov,
                &motion,
                &light,
                focus,
                &crate::gpu::Shot {
                    model,
                    exposure,
                    seed: self.made.seed,
                    frame: seq,
                },
                frame.data_mut(),
            )?,
        }
        let mut black = model.black_level;
        if let Some(sdk) = sdk {
            let shift = if config.bit_depth == 8 { 8 } else { 0 };
            frame
                .data_mut()
                .par_chunks_mut(stride)
                .enumerate()
                .for_each(|(row, line)| {
                    for (col, pair) in line[..2 * width]
                        .as_chunks_mut::<2>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        let count = u16::from_le_bytes(*pair);
                        let c = sdk.apply(count, model.black_level, bayer(col as u32, row as u32));
                        *pair = (c >> shift).to_le_bytes();
                    }
                });
            black = sdk.offset >> shift;
        } else if config.bit_depth == 8 {
            black >>= 4;
        }
        let meta = frame.meta_mut();
        meta.seq = seq;
        meta.exposure_start = start;
        meta.exposure_end = start.saturating_add(timing.exposure);
        meta.line_time = Some(timing.line_time);
        meta.exposure = config.exposure;
        meta.gain = config.gain;
        meta.mode = config.mode;
        meta.binning = config.binning;
        meta.roi = Rect::new(0, 0, sensor.width, sensor.height);
        meta.black_level = black;
        meta.objective = fitted.map(|o| o.name);
        meta.light = self.scene.light.state(CHANNEL).ok();
        meta.position = Some(middle);
        meta.dropped_before = dropped;
        Ok(frame.freeze())
    }
}

/// The stream: frames on the clock, free-running or each on a trigger, until stopped.
fn stream(
    shared: &Shared,
    config: &CameraConfig,
    first: Timestamp,
    pool: &FramePool,
    stop: &AtomicBool,
    wanted: &Receiver<()>,
    sink: &dyn FrameSink,
) -> Result<()> {
    let timing = shared.timing(config)?;
    let clock = &shared.clock;
    let mut next = first;
    let mut dropped = 0;
    while !stop.load(Ordering::Acquire) {
        let start = if config.trigger == Trigger::Software {
            match wanted.recv_timeout(POLL) {
                Ok(()) => clock.now(),
                Err(_) => continue,
            }
        } else {
            next
        };
        // The frame exists once its last row has read out.
        let done = start.saturating_add(timing.span());
        while !clock.wait_until(done, POLL) {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
        }
        let seq = shared.next_seq();
        match pool.try_take() {
            Some(buffer) => {
                let frame = shared.render(config, start, seq, dropped, buffer)?;
                dropped = 0;
                // Looked at again, so no frame reaches the sink once `stop` returns.
                if stop.load(Ordering::Acquire) {
                    return Ok(());
                }
                sink.deliver(frame);
            }
            None => {
                dropped += 1;
                shared.events.emit(CameraEvent::Dropped { count: 1 });
            }
        }
        if config.trigger != Trigger::Software {
            next = next.saturating_add(timing.interval);
            // Fallen behind by whole frames: those frames are lost, and said to be.
            let late = clock
                .now()
                .saturating_since(next.saturating_add(timing.span()));
            let missed = (late.as_nanos() / timing.interval.as_nanos().max(1)) as u64;
            if missed > 0 {
                next = next.saturating_add(timing.interval * missed as u32);
                dropped += missed;
                lock(&shared.state).seq += missed;
                shared.events.emit(CameraEvent::Dropped { count: missed });
            }
        }
    }
    Ok(())
}

impl Camera for VirtualCamera {
    fn capabilities(&self) -> CameraCapabilities {
        self.shared.capabilities.clone()
    }

    fn config(&self) -> CameraConfig {
        lock(&self.shared.state).config.clone()
    }

    fn configure(&self, config: &CameraConfig) -> Result<()> {
        if self.is_streaming() {
            return Err(Error::busy(
                "the camera is streaming; stop it to configure it",
            ));
        }
        self.shared.capabilities.check(config)?;
        {
            let mut s = lock(&self.shared.state);
            s.config = config.clone();
            s.pool = None;
        }
        self.shared.events.emit(CameraEvent::Configured {
            config: config.clone(),
        });
        Ok(())
    }

    fn start(&self, sink: Arc<dyn FrameSink>) -> Result<()> {
        let mut stream_slot = lock(&self.stream);
        if stream_slot
            .as_ref()
            .is_some_and(|s| !s.thread.is_finished())
        {
            return Err(Error::busy("the camera is streaming already"));
        }
        let pool = self.shared.pool()?;
        let config = self.config();
        // The first frame starts now, not when the thread gets going.
        let first = self.shared.clock.now();
        let stop = Arc::new(AtomicBool::new(false));
        let (triggers, wanted) = channel::<()>();
        let (shared, flag) = (self.shared.clone(), stop.clone());
        let thread = std::thread::Builder::new()
            .name("virtual camera".into())
            .spawn(move || {
                if let Err(e) = stream(&shared, &config, first, &pool, &flag, &wanted, &*sink) {
                    shared.events.emit(CameraEvent::Error {
                        kind: e.kind(),
                        message: e.message().to_string(),
                    });
                }
                shared.events.emit(CameraEvent::Stopped);
            })
            .map_err(|e| Error::internal(format!("couldn't start the virtual camera: {e}")))?;
        *stream_slot = Some(Stream {
            stop,
            triggers,
            thread,
        });
        self.shared.events.emit(CameraEvent::Started);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        let stream = lock(&self.stream).take();
        if let Some(s) = stream {
            s.stop.store(true, Ordering::Release);
            s.thread
                .join()
                .map_err(|_| Error::internal("the virtual camera's thread panicked"))?;
        }
        Ok(())
    }

    fn is_streaming(&self) -> bool {
        lock(&self.stream)
            .as_ref()
            .is_some_and(|s| !s.thread.is_finished())
    }

    fn snap(&self, timeout: Duration) -> Result<Frame> {
        if self.is_streaming() {
            return Err(Error::busy(
                "the camera is streaming; stop it to snap a frame",
            ));
        }
        let config = self.config();
        let pool = self.shared.pool()?;
        let timing = self.shared.timing(&config)?;
        let start = self.shared.clock.now();
        if !self
            .shared
            .clock
            .wait_until(start.saturating_add(timing.span()), timeout)
        {
            return Err(Error::timeout(format!(
                "the frame wasn't exposed within {timeout:?}: the clock is {:?}",
                self.shared.clock.rate()
            )));
        }
        let buffer = pool.take(timeout)?;
        let seq = self.shared.next_seq();
        self.shared.render(&config, start, seq, 0, buffer)
    }

    fn trigger(&self) -> Result<()> {
        let stream = lock(&self.stream);
        match stream.as_ref() {
            Some(s) if !s.thread.is_finished() && self.config().trigger == Trigger::Software => s
                .triggers
                .send(())
                .map_err(|_| Error::new(ErrorKind::Internal, "the camera's stream has ended")),
            _ => Err(Error::invalid_argument(
                "a trigger needs a stream on a software trigger",
            )),
        }
    }

    fn subscribe(&self) -> Receiver<CameraEvent> {
        self.shared.events.subscribe()
    }
}

impl Drop for VirtualCamera {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Rate;
    use crate::he::SectionConfig;
    use crate::stage::VirtualStageConfig;
    use tetera_core::frame::meta::LightState;
    use tetera_core::{StagePos, Um};
    use tetera_hal::contract::camera as contract;
    use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue};
    use tetera_hal::stage::{Axis, Stage};

    const OBJECTIVES: &str = include_str!("../../../configs/objectives.toml");

    /// A small sensor, 96 × 72 pixels: modes of 96 × 72, 48 × 36 and 32 × 24.
    const SMALL: Sensor = Sensor {
        width: 96,
        height: 72,
        pixel: 2.4,
    };

    /// A scene on a clock at `rate`: the stage in focus over tissue, `objective` fitted,
    /// the light on.
    fn scene(rate: Rate, objective: Option<&str>) -> Scene {
        scene_at(rate, objective, 1_002.0)
    }

    /// The same, the stage at a height of `z` µm; the section's lower face is in focus at
    /// 1,000.
    fn scene_at(rate: Rate, objective: Option<&str>, z: f64) -> Scene {
        let clock = Arc::new(SimClock::new(rate).unwrap());
        let stage = VirtualStage::new(
            clock,
            VirtualStageConfig {
                start: StagePos::new(Um(1_000.0), Um(1_000.0), Um(z)),
                ..VirtualStageConfig::default()
            },
        )
        .unwrap();
        let mount = ManualObjectiveMount::new(ObjectiveCatalogue::from_toml(OBJECTIVES).unwrap());
        mount.fit(objective).unwrap();
        let light = VirtualIllumination::new(1_000.0, 0.05);
        light
            .set(
                CHANNEL,
                LightState {
                    on: true,
                    intensity: 1.0,
                    colour_temperature_k: Some(3_200.0),
                },
            )
            .unwrap();
        Scene {
            stage: Arc::new(stage),
            objectives: Arc::new(mount),
            light: Arc::new(light),
            sample: HeSample::new(7, SectionConfig::default()).unwrap(),
        }
    }

    fn camera_linear_eight() -> Frame {
        let camera = camera(scene(Rate::Faster(10.0), Some("20x")), None);
        set(&camera, |c| c.bit_depth = 8);
        camera.snap(Duration::from_secs(5)).unwrap()
    }

    fn camera(scene: Scene, sdk: Option<SdkProcessing>) -> VirtualCamera {
        VirtualCamera::new(
            scene,
            VirtualCameraConfig {
                sensor: SMALL,
                sdk,
                ..VirtualCameraConfig::default()
            },
            Backend::Cpu,
        )
        .unwrap()
    }

    fn set(camera: &VirtualCamera, change: impl FnOnce(&mut CameraConfig)) {
        let mut config = camera.config();
        change(&mut config);
        camera.configure(&config).unwrap();
    }

    /// The frame's samples, as numbers.
    fn samples(frame: &Frame) -> Vec<u16> {
        let l = frame.layout();
        (0..l.height as usize)
            .flat_map(|r| {
                frame.data()[r * l.stride..][..l.width as usize * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u16::from_le_bytes(*p))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn mean(v: &[u16]) -> f64 {
        v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64
    }

    #[test]
    fn the_modes_are_the_michrome_20s_at_its_rates() {
        let camera = camera(scene(Rate::Stepped, Some("20x")), None);
        let caps = camera.capabilities();
        let shapes: Vec<(u32, u32, u32, Vec<u8>, f64)> = caps
            .modes
            .iter()
            .map(|m| (m.id, m.width, m.height, m.bit_depths.clone(), m.max_fps))
            .collect();
        assert_eq!(
            shapes,
            [
                (0, 96, 72, vec![12, 8], 14.49),
                (1, 48, 36, vec![12, 8], 53.14),
                (2, 32, 24, vec![12, 8], 67.26),
            ]
        );
        let full = VirtualCamera::new(
            scene(Rate::Stepped, None),
            VirtualCameraConfig {
                sdk: Some(SdkProcessing::default()),
                ..VirtualCameraConfig::default()
            },
            Backend::Cpu,
        )
        .unwrap();
        let caps = full.capabilities();
        assert_eq!(
            caps.modes
                .iter()
                .map(|m| (m.width, m.height, m.bit_depths.clone()))
                .collect::<Vec<_>>(),
            [
                (5_472, 3_648, vec![16, 8]),
                (2_736, 1_824, vec![16, 8]),
                (1_824, 1_216, vec![16, 8]),
            ]
        );
        for (id, bits, fps) in [
            (0, 12, 7.24),
            (0, 8, 14.49),
            (1, 12, 26.57),
            (1, 8, 53.14),
            (2, 16, 33.87),
            (2, 8, 67.26),
        ] {
            let interval = mode(id).unwrap().interval(bits).as_secs_f64();
            // Within a nanosecond, as a Duration holds it.
            assert!(
                (interval * fps - 1.0).abs() < 1e-7,
                "mode {id} at {bits} bits"
            );
        }
        assert!(mode(3).is_err());
        assert!(
            VirtualCamera::new(
                scene(Rate::Stepped, None),
                VirtualCameraConfig {
                    sensor: Sensor { width: 4, ..SMALL },
                    ..VirtualCameraConfig::default()
                },
                Backend::Cpu
            )
            .is_err()
        );
    }

    #[test]
    fn a_camera_it_cannot_make_is_refused() {
        let good = VirtualCameraConfig {
            sensor: SMALL,
            ..VirtualCameraConfig::default()
        };
        good.check().unwrap();
        VirtualCameraConfig {
            sensor: Sensor {
                width: 6,
                height: 6,
                pixel: 2.4,
            },
            max_blur: 1.0,
            pool: 1,
            ..good.clone()
        }
        .check()
        .unwrap();
        let with = |change: &dyn Fn(&mut VirtualCameraConfig)| {
            let mut c = good.clone();
            change(&mut c);
            c
        };
        for bad in [
            with(&|c| c.sensor.width = 5),
            with(&|c| c.sensor.height = 5),
            with(&|c| c.sensor.pixel = 0.0),
            with(&|c| c.sensor.pixel = f64::NAN),
            with(&|c| c.mounting_angle = f64::NAN),
            with(&|c| c.section_z = f64::INFINITY),
            with(&|c| c.max_blur = 0.99),
            with(&|c| c.max_blur = f64::NAN),
            with(&|c| c.pool = 0),
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
        let camera = camera(scene(Rate::Stepped, Some("20x")), None);
        let shown = format!("{camera:?}");
        assert!(
            shown.contains("Tetera virtual MIChrome 20") && shown.contains("mode: 0"),
            "{shown}"
        );
        assert!(format!("{:?}", scene(Rate::Stepped, Some("5x"))).contains("\"5x\""));
    }

    #[test]
    fn a_frame_comes_once_its_last_row_has_read_out() {
        let camera = camera(scene(Rate::Stepped, Some("20x")), None);
        let timing = camera.timing().unwrap();
        // 71 line times of mode 0's 72 rows at 12 bits, and the 10 ms exposure.
        let expected = 71.0 / 72.0 / 7.24 + 0.01;
        assert!((timing.span().as_secs_f64() - expected).abs() < 1e-6);
        let t0 = camera.clock().now();
        let (sink, frames) = contract::channel_sink();
        camera.start(sink).unwrap();
        camera
            .clock()
            .step(timing.span() - Duration::from_millis(1))
            .unwrap();
        assert!(
            frames.recv_timeout(Duration::from_millis(300)).is_err(),
            "not before its last row"
        );
        camera.clock().step(Duration::from_millis(1)).unwrap();
        let frame = frames.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(frame.meta().exposure_start, t0);
        camera.stop().unwrap();
    }

    #[test]
    fn a_frame_is_the_field_along_the_stages_path_through_the_sensor() {
        for (axis, velocity) in [(Axis::X, (400.0, 0.0)), (Axis::Y, (0.0, -300.0))] {
            let scene = scene(Rate::Stepped, Some("20x"));
            let (stage, sample, light) = (scene.stage.clone(), scene.sample, scene.light.light());
            let camera = camera(scene, None);
            stage.jog(axis, velocity.0 + velocity.1).unwrap();
            // At its speed before the frame starts.
            camera.clock().step(Duration::from_millis(20)).unwrap();
            let start = camera.clock().now();
            let timing = camera.timing().unwrap();
            let frame = stream_stepped(&camera, 1).remove(0);
            let span = timing.span();
            assert_eq!(
                frame.meta().position,
                Some(stage.position_at(start.saturating_add(span / 2))),
                "where the stage was halfway through"
            );
            let first = stage.position_at(start);
            let fov = FieldOfView {
                centre: (first.x.um(), first.y.um()),
                optics: Optics {
                    sensor: SMALL,
                    ..Optics::new(20.0, 0.42, 0.0).unwrap()
                },
            };
            let motion = Motion {
                velocity,
                exposure: timing.exposure.as_secs_f64(),
                line_time: timing.line_time.as_secs_f64(),
            };
            let field = field_moving(&sample, &fov, &motion, &light, 2.0).unwrap();
            let exposure = Exposure {
                time: timing.exposure,
                gain: 1.0,
                bits: 12,
                binning: 1,
            };
            let expected = expose(
                &field.rgb,
                SMALL.width,
                SMALL.height,
                &SensorModel::default(),
                &exposure,
                1,
                0,
            )
            .unwrap();
            let got = samples(&frame);
            let differ = got
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.abs_diff(**b) > 1)
                .count();
            assert!(
                differ * 1_000 < got.len(),
                "{axis:?}: {differ} of {} apart",
                got.len()
            );
        }
    }

    #[test]
    fn far_from_focus_a_frame_is_as_blurred_as_it_gets() {
        // Two cameras of one seed, their first frames: held at the same blur beyond 25 µm
        // at 20×, above the section and below it, and not within it.
        let first = |z: f64| {
            let camera = camera(scene_at(Rate::Faster(10.0), Some("20x"), z), None);
            samples(&camera.snap(Duration::from_secs(5)).unwrap())
        };
        assert_eq!(first(1_060.0), first(1_200.0), "held above");
        assert_eq!(first(940.0), first(800.0), "held below");
        assert_ne!(first(1_002.0), first(1_060.0), "nearer, sharper");
    }

    #[test]
    fn the_virtual_camera_keeps_the_contract() {
        contract::all(&|| Box::new(camera(scene(Rate::Faster(10.0), Some("20x")), None)));
        contract::all(&|| {
            Box::new(camera(
                scene(Rate::Faster(10.0), Some("5x")),
                Some(SdkProcessing::default()),
            ))
        });
    }

    /// Starts `camera` streaming on its stepped clock and takes its clock to each of `n`
    /// frames' ends in turn, waiting for each: the frames.
    fn stream_stepped(camera: &VirtualCamera, n: u32) -> Vec<Frame> {
        let timing = camera.timing().unwrap();
        let clock = camera.clock();
        let t0 = clock.now();
        let (sink, frames) = contract::channel_sink();
        camera.start(sink).unwrap();
        let got = (0..n)
            .map(|k| {
                let done = t0
                    .saturating_add(timing.interval * k)
                    .saturating_add(timing.span());
                clock.step(done.saturating_since(clock.now())).unwrap();
                frames.recv_timeout(Duration::from_secs(10)).unwrap()
            })
            .collect();
        camera.stop().unwrap();
        got
    }

    #[test]
    fn frames_come_at_the_modes_interval_on_the_clock() {
        // The spec's rates, as OScope measured them; the third exposure is longer than the
        // mode's readout, so it sets the interval.
        for (id, bits, exposure_ms, fps) in [
            (0u32, 12u8, 5.0f64, 7.24),
            (0, 8, 5.0, 14.49),
            (1, 8, 2.0, 53.14),
            (2, 12, 100.0, 33.87),
        ] {
            let camera = camera(scene(Rate::Stepped, Some("20x")), None);
            set(&camera, |c| {
                c.mode = id;
                c.bit_depth = bits;
                c.exposure = Duration::from_secs_f64(exposure_ms / 1e3);
            });
            let timing = camera.timing().unwrap();
            let readout: f64 = 1.0 / fps;
            assert!(
                (timing.interval.as_secs_f64() - readout.max(exposure_ms / 1e3)).abs() < 1e-8,
                "mode {id} at {bits} bits: {:?}",
                timing.interval
            );
            let rows = SMALL.height / mode(id).unwrap().binning;
            assert!((timing.line_time.as_secs_f64() * f64::from(rows) - readout).abs() < 1e-6);
            let events = camera.subscribe();
            let got = stream_stepped(&camera, 4);
            assert!(
                !events
                    .try_iter()
                    .any(|e| matches!(e, CameraEvent::Dropped { .. })),
                "nothing dropped, keeping up"
            );
            for pair in got.windows(2) {
                let (a, b) = (pair[0].meta(), pair[1].meta());
                assert_eq!(
                    b.exposure_start.saturating_since(a.exposure_start),
                    timing.interval,
                    "mode {id} at {bits} bits, {exposure_ms} ms"
                );
                assert_eq!(b.seq, a.seq + 1);
            }
            let m = got[0].meta();
            assert_eq!(m.line_time, Some(timing.line_time));
            assert_eq!(
                m.exposure_end.saturating_since(m.exposure_start),
                timing.exposure
            );
        }
    }

    #[test]
    fn frames_repeat_bit_for_bit_from_a_seed_in_stepped_time() {
        let run = || {
            let scene = scene(Rate::Stepped, Some("20x"));
            let stage = scene.stage.clone();
            let camera = camera(scene, None);
            stage.jog(Axis::X, 500.0).unwrap();
            stream_stepped(&camera, 3)
                .iter()
                .map(|f| (samples(f), f.meta().position))
                .collect::<Vec<_>>()
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b, "the same frames again");
        assert_ne!(a[0].0, a[1].0, "each frame its own noise, the stage moving");
        assert_ne!(a[0].1, a[2].1, "the stage moved");
    }

    #[test]
    fn the_objective_the_light_and_the_focus_change_the_frame() {
        let scene = scene(Rate::Faster(10.0), Some("20x"));
        let (stage, mount, light) = (
            scene.stage.clone(),
            scene.objectives.clone(),
            scene.light.clone(),
        );
        let camera = camera(scene, None);
        let snap = || samples(&camera.snap(Duration::from_secs(5)).unwrap());
        let contrast = |v: &[u16]| {
            let w = SMALL.width as usize;
            (0..v.len() - 2)
                .filter(|k| (k + 2) % w > 1)
                .map(|k| (f64::from(v[k + 2]) - f64::from(v[k])).powi(2))
                .sum::<f64>()
        };
        let twenty = snap();
        mount.fit(Some("5x")).unwrap();
        let five = snap();
        assert_ne!(twenty, five, "another objective, another field");
        let frame = camera.snap(Duration::from_secs(5)).unwrap();
        assert_eq!(frame.meta().objective.as_deref(), Some("5x"));
        mount.fit(Some("20x")).unwrap();
        // 50 µm from focus, then further: blurred, then held at the most blur.
        stage
            .move_to(StagePos::new(Um(1_000.0), Um(1_000.0), Um(1_052.0)), None)
            .unwrap();
        stage.wait(Duration::from_secs(10)).unwrap();
        let blurred = snap();
        assert!(
            contrast(&blurred) < 0.5 * contrast(&twenty),
            "{} against {}",
            contrast(&blurred),
            contrast(&twenty)
        );
        // No light, no objective: the sensor's black and its noise.
        light
            .set(
                CHANNEL,
                LightState {
                    on: false,
                    intensity: 1.0,
                    colour_temperature_k: None,
                },
            )
            .unwrap();
        let dark = snap();
        assert!((mean(&dark) - 100.0).abs() < 2.0, "{}", mean(&dark));
        light
            .set(
                CHANNEL,
                LightState {
                    on: true,
                    intensity: 1.0,
                    colour_temperature_k: None,
                },
            )
            .unwrap();
        mount.fit(None).unwrap();
        let frame = camera.snap(Duration::from_secs(5)).unwrap();
        assert!((mean(&samples(&frame)) - 100.0).abs() < 2.0);
        assert_eq!(frame.meta().objective, None);
        assert_eq!(frame.meta().black_level, 100);
    }

    #[test]
    fn a_sample_is_taken_away_and_another_put_on_the_stage() {
        let camera = camera(scene(Rate::Faster(10.0), Some("20x")), None);
        let snap = || samples(&camera.snap(Duration::from_secs(5)).unwrap());
        let loaded = snap();
        camera.set_sample(None).unwrap();
        assert_eq!(camera.sample(), None);
        let bare = snap();
        // Bare glass: the light alone, brighter than tissue, and even but for the noise.
        assert!(
            mean(&bare) > mean(&loaded),
            "{} against {}",
            mean(&bare),
            mean(&loaded)
        );
        let other = HeSample::new(9, SectionConfig::default()).unwrap();
        camera.set_sample(Some(other)).unwrap();
        assert_eq!(camera.sample(), Some(other));
        let again = snap();
        assert!((mean(&again) - mean(&bare)).abs() > 1.0, "tissue again");
    }

    #[test]
    fn far_from_focus_the_blur_is_held() {
        let optics = Optics {
            sensor: SMALL,
            ..Optics::new(50.0, 0.55, 0.0).unwrap()
        };
        for sigma in [5.0, 48.0] {
            let d = optics.defocus_for_blur(sigma);
            assert!(
                (optics.blur(d) / optics.pixel() - sigma).abs() < 1e-9,
                "{sigma}: {d}"
            );
        }
        assert_eq!(optics.defocus_for_blur(0.1), 0.0, "within the Airy disc");
    }

    #[test]
    fn the_sdks_processing_rescales_above_its_offset() {
        let sdk = SdkProcessing::default();
        assert_eq!(sdk.apply(100, 100, 1), 2_000, "black at the offset");
        assert_eq!(sdk.apply(4_095, 100, 1), 60_000, "a full count");
        assert_eq!(
            sdk.apply(90, 100, 1),
            1_855,
            "below black, below the offset"
        );
        let warm = SdkProcessing {
            gamma: 2.2,
            white_balance: [0.5, 1.0, 2.0],
            ..sdk
        };
        let half = 100 + (4_095 - 100) / 2;
        assert!(
            warm.apply(half, 100, 1) > sdk.apply(half, 100, 1),
            "gamma lifts the middle"
        );
        assert!(
            warm.apply(half, 100, 0) < warm.apply(half, 100, 1),
            "red's gain the least"
        );
        assert_eq!(warm.apply(4_095, 100, 2), 65_535, "clipped at 16 bits");
        // Through the camera: 16 bits, the black level the offset.
        let camera = camera(scene(Rate::Faster(10.0), Some("20x")), Some(sdk));
        let frame = camera.snap(Duration::from_secs(5)).unwrap();
        assert_eq!(frame.meta().black_level, 2_000);
        assert!(mean(&samples(&frame)) > 2_000.0 + 20.0 * 14.5);
        set(&camera, |c| c.bit_depth = 8);
        let frame = camera.snap(Duration::from_secs(5)).unwrap();
        assert_eq!(frame.meta().black_level, 7);
        assert!(samples(&frame).iter().all(|&v| v < 256));
        // Linear in 8 bits: the black level's top 8 of 12 bits.
        let linear = camera_linear_eight();
        assert_eq!(linear.meta().black_level, 6);
    }

    /// The camera on the GPU, its tiles within 1 GiB.
    #[cfg(feature = "cuda")]
    fn on_gpu(scene: Scene, gpu: tetera_compute::CudaBackend) -> VirtualCamera {
        let budgets = tetera_compute::budget::VramBudgets::new(1 << 30);
        VirtualCamera::new(
            scene,
            VirtualCameraConfig {
                sensor: SMALL,
                ..VirtualCameraConfig::default()
            },
            Backend::Gpu {
                gpu: Arc::new(gpu),
                budget: budgets.reserve("camera", 1 << 30).unwrap(),
            },
        )
        .unwrap()
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn on_the_gpu_it_keeps_the_contract_and_repeats_its_frames() {
        let _ = tetera_compute::cuda_or_skip!();
        contract::all(&|| {
            let gpu = tetera_compute::CudaBackend::new(0).unwrap();
            Box::new(on_gpu(scene(Rate::Faster(10.0), Some("20x")), gpu))
        });
        let run = |gpu: bool| {
            let scene = scene(Rate::Stepped, Some("20x"));
            let stage = scene.stage.clone();
            let camera = if gpu {
                on_gpu(scene, tetera_compute::CudaBackend::new(0).unwrap())
            } else {
                camera(scene, None)
            };
            stage.jog(Axis::Y, 300.0).unwrap();
            stream_stepped(&camera, 3)
                .iter()
                .map(samples)
                .collect::<Vec<_>>()
        };
        let a = run(true);
        assert_eq!(a, run(true), "the same frames again on the GPU");
        // The same scene on the CPU: the same light, within the twins' tolerance.
        for (g, c) in a.iter().zip(&run(false)) {
            assert!(
                (mean(g) - mean(c)).abs() < 0.01 * mean(c),
                "{} on the GPU, {} on the CPU",
                mean(g),
                mean(c)
            );
        }
    }

    #[test]
    fn with_its_pool_used_up_it_drops_frames_until_one_comes_back() {
        let camera = VirtualCamera::new(
            scene(Rate::Stepped, Some("20x")),
            VirtualCameraConfig {
                sensor: SMALL,
                pool: 2,
                ..VirtualCameraConfig::default()
            },
            Backend::Cpu,
        )
        .unwrap();
        let events = camera.subscribe();
        let timing = camera.timing().unwrap();
        let clock = camera.clock().clone();
        let t0 = clock.now();
        let (sink, frames) = contract::channel_sink();
        camera.start(sink).unwrap();
        let step_to = |k: u32| {
            let done = t0
                .saturating_add(timing.interval * k)
                .saturating_add(timing.span());
            clock.step(done.saturating_since(clock.now())).unwrap();
        };
        // Two frames, both held: the pool is empty.
        let mut held = Vec::new();
        for k in 0..2 {
            step_to(k);
            held.push(frames.recv_timeout(Duration::from_secs(10)).unwrap());
        }
        for k in 2..4 {
            step_to(k);
            loop {
                match events.recv_timeout(Duration::from_secs(10)).unwrap() {
                    CameraEvent::Dropped { count } => {
                        assert_eq!(count, 1);
                        break;
                    }
                    _ => continue,
                }
            }
        }
        drop(held);
        step_to(4);
        let next = frames.recv_timeout(Duration::from_secs(10)).unwrap();
        camera.stop().unwrap();
        assert_eq!(next.meta().dropped_before, 2, "{:?}", next.meta());
        assert_eq!(next.meta().seq, 4);
    }

    #[test]
    fn dropping_the_camera_stops_its_stream() {
        let camera = camera(scene(Rate::Faster(10.0), Some("20x")), None);
        let events = camera.subscribe();
        let (sink, frames) = contract::channel_sink();
        camera.start(sink).unwrap();
        frames.recv_timeout(Duration::from_secs(10)).unwrap();
        drop(camera);
        // Its stream had stopped by the time it was gone.
        assert!(
            events.try_iter().any(|e| e == CameraEvent::Stopped),
            "stopped as it went"
        );
    }

    #[test]
    fn a_frame_falls_behind_and_says_how_many_it_lost() {
        let camera = camera(scene(Rate::Stepped, Some("20x")), None);
        let events = camera.subscribe();
        let interval = camera.timing().unwrap().interval;
        let (sink, frames) = contract::channel_sink();
        camera.start(sink).unwrap();
        // Five intervals at once: the first frame comes, the rest are behind.
        camera.clock().step(interval * 5).unwrap();
        let first = frames.recv_timeout(Duration::from_secs(5)).unwrap();
        camera.clock().step(interval).unwrap();
        let next = frames.recv_timeout(Duration::from_secs(5)).unwrap();
        camera.stop().unwrap();
        assert!(next.meta().dropped_before > 0, "{:?}", next.meta());
        assert_eq!(
            next.meta()
                .exposure_start
                .saturating_since(first.meta().exposure_start),
            interval * (1 + next.meta().dropped_before as u32),
            "whole intervals later"
        );
        assert_eq!(
            next.meta().seq,
            first.meta().seq + 1 + next.meta().dropped_before
        );
        assert!(
            events
                .try_iter()
                .any(|e| matches!(e, CameraEvent::Dropped { count } if count > 0))
        );
    }
}
