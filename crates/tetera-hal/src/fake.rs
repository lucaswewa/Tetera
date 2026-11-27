//! The fakes, which stand in for devices in tests and examples, with faults on demand. Here,
//! the fake camera (P1.09a): a moving test pattern at a set rate, from a pool of its own, on a
//! thread of its own. In [`stage`], the fake stage (P1.09b).

pub mod stage;

use crate::camera::{
    Camera, CameraCapabilities, CameraConfig, CameraEvent, CameraMode, FrameSink, Span, Trigger,
};
use crate::event::Events;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetera_core::frame::format::{BayerPattern, Binning};
use tetera_core::frame::pool::HostAllocator;
use tetera_core::{Clock, Error, ErrorKind, Frame, FrameMut, FramePool, Result};

/// A fault the fake camera shows when asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraFault {
    /// The next `configure` fails, as a hardware error.
    FailConfigure,
    /// The next `count` frames are lost, and said to be.
    DropFrames(u64),
    /// The stream fails: an error event, and it stops.
    FailStream,
    /// Gone: every call fails, as a hardware error, until it comes back.
    Disconnect,
    /// Back again.
    Reconnect,
}

/// The fake camera's state, shared with its streaming thread.
struct State {
    config: CameraConfig,
    seq: u64,
    to_drop: u64,
    fail_configure: bool,
    fail_stream: bool,
    disconnected: bool,
    pool: Option<FramePool>,
}

struct Stream {
    stop: Arc<AtomicBool>,
    triggers: Sender<()>,
    thread: JoinHandle<()>,
}

/// A fake camera: the capabilities of a small 12-bit Bayer sensor, frames of a moving test
/// pattern, timestamped by its clock.
pub struct FakeCamera {
    capabilities: CameraCapabilities,
    clock: Arc<dyn Clock>,
    state: Arc<Mutex<State>>,
    stream: Mutex<Option<Stream>>,
    events: Arc<Events<CameraEvent>>,
}

impl std::fmt::Debug for FakeCamera {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeCamera")
            .field("model", &self.capabilities.model)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Buffers in the fake camera's pool.
const POOL: usize = 4;

impl FakeCamera {
    /// A fake camera timestamping by `clock`, at up to `max_fps` frames a second.
    pub fn new(clock: Arc<dyn Clock>, max_fps: f64) -> Self {
        let capabilities = CameraCapabilities {
            model: "Tetera fake camera".into(),
            pattern: BayerPattern::Rggb,
            modes: vec![
                CameraMode {
                    id: 0,
                    name: "full".into(),
                    width: 640,
                    height: 480,
                    bit_depths: vec![12, 8],
                    max_fps,
                },
                CameraMode {
                    id: 1,
                    name: "small".into(),
                    width: 160,
                    height: 120,
                    bit_depths: vec![8, 12],
                    max_fps: max_fps * 2.0,
                },
            ],
            binnings: vec![Binning { x: 2, y: 2 }],
            roi_step: Some((2, 2)),
            exposure_ms: Span {
                min: 0.01,
                max: 2_000.0,
            },
            gain: Span {
                min: 1.0,
                max: 16.0,
            },
            triggers: vec![Trigger::FreeRun, Trigger::Software],
        };
        let config = capabilities.default_config();
        Self {
            capabilities,
            clock,
            state: Arc::new(Mutex::new(State {
                config,
                seq: 0,
                to_drop: 0,
                fail_configure: false,
                fail_stream: false,
                disconnected: false,
                pool: None,
            })),
            stream: Mutex::new(None),
            events: Arc::new(Events::default()),
        }
    }

    /// Shows `fault`.
    pub fn inject(&self, fault: CameraFault) {
        let mut s = lock(&self.state);
        match fault {
            CameraFault::FailConfigure => s.fail_configure = true,
            CameraFault::DropFrames(n) => s.to_drop += n,
            CameraFault::FailStream => s.fail_stream = true,
            CameraFault::Disconnect => s.disconnected = true,
            CameraFault::Reconnect => s.disconnected = false,
        }
    }

    fn connected(&self) -> Result<()> {
        if lock(&self.state).disconnected {
            Err(Error::hardware("the camera is disconnected"))
        } else {
            Ok(())
        }
    }

    /// The pool for the current configuration, made on first use after each configure.
    fn pool(&self) -> Result<FramePool> {
        let mut s = lock(&self.state);
        if s.pool.is_none() {
            let layout = self.capabilities.layout(&s.config)?;
            s.pool = Some(FramePool::new(layout, POOL, &HostAllocator)?);
        }
        Ok(s.pool.clone().expect("made above"))
    }
}

/// One frame: the next number in the stream, its pattern drawn, its metadata filled.
fn draw(
    mut frame: FrameMut,
    config: &CameraConfig,
    seq: u64,
    dropped: u64,
    clock: &dyn Clock,
) -> Frame {
    let start = clock.now();
    let layout = *frame.layout();
    let bits = layout.format.significant_bits().unwrap_or(16);
    let max = (1u32 << bits) - 1;
    let shift = seq as u32;
    let data = frame.data_mut();
    for y in 0..layout.height {
        let row = &mut data[y as usize * layout.stride..][..layout.width as usize * 2];
        for (x, pair) in row.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let v = ((x as u32 + y + shift) * 16) & max;
            *pair = (v as u16).to_le_bytes();
        }
    }
    let meta = frame.meta_mut();
    meta.seq = seq;
    meta.exposure_start = start;
    meta.exposure_end = start.checked_add(config.exposure).unwrap_or(start);
    meta.exposure = config.exposure;
    meta.gain = config.gain;
    meta.mode = config.mode;
    meta.binning = config.binning;
    meta.roi = config.roi.unwrap_or_default();
    meta.dropped_before = dropped;
    frame.freeze()
}

impl Camera for FakeCamera {
    fn capabilities(&self) -> CameraCapabilities {
        self.capabilities.clone()
    }

    fn config(&self) -> CameraConfig {
        lock(&self.state).config.clone()
    }

    fn configure(&self, config: &CameraConfig) -> Result<()> {
        self.connected()?;
        if self.is_streaming() {
            return Err(Error::busy(
                "the camera is streaming; stop it to configure it",
            ));
        }
        self.capabilities.check(config)?;
        {
            let mut s = lock(&self.state);
            if std::mem::take(&mut s.fail_configure) {
                return Err(Error::hardware("the camera refused its configuration"));
            }
            s.config = config.clone();
            s.pool = None;
        }
        self.events.emit(CameraEvent::Configured {
            config: config.clone(),
        });
        Ok(())
    }

    fn start(&self, sink: Arc<dyn FrameSink>) -> Result<()> {
        self.connected()?;
        let mut stream = lock(&self.stream);
        if stream.as_ref().is_some_and(|s| !s.thread.is_finished()) {
            return Err(Error::busy("the camera is streaming already"));
        }
        let pool = self.pool()?;
        let config = self.config();
        let mode = self.capabilities.mode(config.mode)?.clone();
        let period = Duration::from_secs_f64(1.0 / mode.max_fps).max(config.exposure);
        let stop = Arc::new(AtomicBool::new(false));
        let (triggers, wanted) = channel::<()>();
        let (state, events, clock) = (self.state.clone(), self.events.clone(), self.clock.clone());
        let flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("fake camera".into())
            .spawn(move || {
                let mut next = Instant::now();
                let mut dropped = 0;
                while !flag.load(Ordering::Acquire) {
                    if config.trigger == Trigger::Software {
                        match wanted.recv_timeout(Duration::from_millis(5)) {
                            Ok(()) => {}
                            Err(_) => continue,
                        }
                    } else {
                        let now = Instant::now();
                        if now < next {
                            std::thread::sleep((next - now).min(Duration::from_millis(5)));
                            continue;
                        }
                        next += period;
                    }
                    let (seq, drop_this, fail) = {
                        let mut s = lock(&state);
                        let seq = s.seq;
                        s.seq += 1;
                        let drop_this = s.to_drop > 0;
                        s.to_drop = s.to_drop.saturating_sub(1);
                        (seq, drop_this, std::mem::take(&mut s.fail_stream))
                    };
                    if fail {
                        events.emit(CameraEvent::Error {
                            kind: ErrorKind::Hardware,
                            message: "the camera's stream failed".into(),
                        });
                        break;
                    }
                    let buffer = if drop_this { None } else { pool.try_take() };
                    let Some(buffer) = buffer else {
                        dropped += 1;
                        events.emit(CameraEvent::Dropped { count: 1 });
                        continue;
                    };
                    let frame = draw(buffer, &config, seq, dropped, &*clock);
                    dropped = 0;
                    // The stop is checked again, so no frame is delivered once `stop` returns.
                    if flag.load(Ordering::Acquire) {
                        break;
                    }
                    sink.deliver(frame);
                }
                events.emit(CameraEvent::Stopped);
            })
            .map_err(|e| Error::internal(format!("couldn't start the fake camera: {e}")))?;
        *stream = Some(Stream {
            stop,
            triggers,
            thread,
        });
        self.events.emit(CameraEvent::Started);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        let stream = lock(&self.stream).take();
        if let Some(s) = stream {
            s.stop.store(true, Ordering::Release);
            s.thread
                .join()
                .map_err(|_| Error::internal("the fake camera's thread panicked"))?;
        }
        Ok(())
    }

    fn is_streaming(&self) -> bool {
        lock(&self.stream)
            .as_ref()
            .is_some_and(|s| !s.thread.is_finished())
    }

    fn snap(&self, timeout: Duration) -> Result<Frame> {
        self.connected()?;
        if self.is_streaming() {
            return Err(Error::busy("the camera is streaming; stop it to snap"));
        }
        let pool = self.pool()?;
        let buffer = pool.take(timeout)?;
        let config = self.config();
        let seq = {
            let mut s = lock(&self.state);
            s.seq += 1;
            s.seq - 1
        };
        Ok(draw(buffer, &config, seq, 0, &*self.clock))
    }

    fn trigger(&self) -> Result<()> {
        self.connected()?;
        let stream = lock(&self.stream);
        match stream.as_ref() {
            Some(s) if self.config().trigger == Trigger::Software => s
                .triggers
                .send(())
                .map_err(|_| Error::hardware("the camera's stream has ended")),
            Some(_) => Err(Error::invalid_argument(
                "the camera isn't set to a software trigger",
            )),
            None => Err(Error::invalid_argument("the camera isn't streaming")),
        }
    }

    fn subscribe(&self) -> Receiver<CameraEvent> {
        self.events.subscribe()
    }
}

impl Drop for FakeCamera {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::camera::{self as contract, channel_sink};
    use tetera_core::{ManualClock, SystemClock, UtcTime};

    fn fake() -> FakeCamera {
        FakeCamera::new(Arc::new(SystemClock::new()), 200.0)
    }

    #[test]
    fn the_fake_camera_keeps_the_contract() {
        contract::all(&|| Box::new(fake()));
    }

    #[test]
    fn frames_carry_the_pattern_the_clock_and_their_number() {
        let clock = Arc::new(ManualClock::new(UtcTime::now()));
        clock.advance(Duration::from_millis(250));
        let camera = FakeCamera::new(clock.clone(), 200.0);
        let config = CameraConfig {
            mode: 1,
            exposure: Duration::from_millis(4),
            ..camera.config()
        };
        camera.configure(&config).unwrap();
        let frame = camera.snap(Duration::from_secs(1)).unwrap();
        let meta = frame.meta();
        assert_eq!(
            meta.exposure_start.as_duration(),
            Duration::from_millis(250)
        );
        assert_eq!(meta.exposure_end.as_duration(), Duration::from_millis(254));
        assert_eq!(
            (
                meta.seq,
                meta.mode,
                frame.layout().width,
                frame.layout().height
            ),
            (0, 1, 160, 120)
        );
        // The pattern: (x + y + seq) × 16, in 12 bits, little-endian.
        let sample = |f: &Frame, x: usize, y: usize| {
            let at = y * f.layout().stride + x * 2;
            u16::from_le_bytes([f.data()[at], f.data()[at + 1]])
        };
        assert_eq!(
            (
                sample(&frame, 0, 0),
                sample(&frame, 3, 2),
                sample(&frame, 159, 119)
            ),
            (0, 80, 352),
            "the first pixels, and the last of the last row"
        );
        let next = camera.snap(Duration::from_secs(1)).unwrap();
        assert_eq!(
            (next.meta().seq, sample(&next, 0, 0), sample(&next, 159, 96)),
            (1, 16, 0),
            "moved by one, wrapped at 4096"
        );
        // Binned and cut: the frame's metadata says how.
        let cut = CameraConfig {
            mode: 0,
            roi: Some(tetera_core::Rect::new(2, 4, 100, 50)),
            binning: Binning { x: 2, y: 2 },
            ..config
        };
        camera.configure(&cut).unwrap();
        let frame = camera.snap(Duration::from_secs(1)).unwrap();
        assert_eq!((frame.layout().width, frame.layout().height), (50, 24));
        assert_eq!(
            (frame.meta().roi, frame.meta().binning),
            (cut.roi.unwrap(), cut.binning)
        );
    }

    #[test]
    fn faults_come_when_asked() {
        let camera = fake();
        let events = camera.subscribe();
        camera.inject(CameraFault::FailConfigure);
        assert_eq!(
            camera.configure(&camera.config()).unwrap_err().kind(),
            ErrorKind::Hardware
        );
        camera.configure(&camera.config()).unwrap();
        // Dropped frames: said as events, and on the next frame.
        let (sink, frames) = channel_sink();
        camera.inject(CameraFault::DropFrames(3));
        camera.start(sink).unwrap();
        let first = frames.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!((first.meta().seq, first.meta().dropped_before), (3, 3));
        // A failed stream: an error event, and it stops.
        camera.inject(CameraFault::FailStream);
        let start = Instant::now();
        while camera.is_streaming() {
            assert!(start.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(1));
        }
        camera.stop().unwrap();
        let seen: Vec<CameraEvent> = events.try_iter().collect();
        assert_eq!(
            seen.iter()
                .filter(|e| **e == CameraEvent::Dropped { count: 1 })
                .count(),
            3
        );
        assert!(seen.iter().any(|e| matches!(
            e,
            CameraEvent::Error {
                kind: ErrorKind::Hardware,
                ..
            }
        )));
        assert_eq!(seen.last(), Some(&CameraEvent::Stopped));
        // Disconnected: every call fails, until it comes back.
        camera.inject(CameraFault::Disconnect);
        let (sink, _frames) = channel_sink();
        assert_eq!(
            camera.start(sink.clone()).unwrap_err().kind(),
            ErrorKind::Hardware
        );
        assert_eq!(
            camera.snap(Duration::from_secs(1)).unwrap_err().kind(),
            ErrorKind::Hardware
        );
        assert_eq!(
            camera.configure(&camera.config()).unwrap_err().kind(),
            ErrorKind::Hardware
        );
        assert_eq!(camera.trigger().unwrap_err().kind(), ErrorKind::Hardware);
        camera.inject(CameraFault::Reconnect);
        assert!(camera.snap(Duration::from_secs(1)).is_ok());
        assert!(format!("{camera:?}").contains("Tetera fake camera"));
    }

    #[test]
    fn a_sink_that_keeps_every_frame_makes_the_camera_drop_and_say_so() {
        let camera = fake();
        let events = camera.subscribe();
        let kept = Arc::new(Mutex::new(Vec::new()));
        let keeping = Arc::new(AtomicBool::new(true));
        let (tx, frames) = channel();
        let (keep, still) = (kept.clone(), keeping.clone());
        camera
            .start(Arc::new(move |f: Frame| {
                if still.load(Ordering::Acquire) {
                    lock(&keep).push(f);
                } else {
                    let _ = tx.send(f);
                }
            }))
            .unwrap();
        let start = Instant::now();
        while !events
            .try_iter()
            .any(|e| e == CameraEvent::Dropped { count: 1 })
        {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "the pool's frames all kept"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(lock(&kept).len(), POOL, "the pool's frames, and no more");
        // Let them go: the next frame says how many were lost meanwhile.
        keeping.store(false, Ordering::Release);
        lock(&kept).clear();
        let frame = frames.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(frame.meta().dropped_before > 0);
        camera.stop().unwrap();
    }

    #[test]
    fn a_stream_keeps_to_its_rate() {
        // At 200 frames a second, a 10 ms exposure sets the pace: the 20th frame is due
        // 190 ms after the first, and never comes sooner.
        let camera = fake();
        let (sink, frames) = channel_sink();
        let start = Instant::now();
        camera.start(sink).unwrap();
        for _ in 0..20 {
            frames.recv_timeout(Duration::from_secs(1)).unwrap();
        }
        let elapsed = start.elapsed();
        camera.stop().unwrap();
        assert!(
            elapsed >= Duration::from_millis(190),
            "too soon: {elapsed:?}"
        );
    }

    #[test]
    fn dropping_the_camera_stops_its_stream() {
        let camera = fake();
        let events = camera.subscribe();
        let (sink, _frames) = channel_sink();
        camera.start(sink).unwrap();
        drop(camera);
        assert_eq!(events.try_iter().last(), Some(CameraEvent::Stopped));
    }
}
