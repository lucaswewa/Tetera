//! The scratch recorder (P1.15c, ADR-0008, ADR-0021): every frame of the camera's stream,
//! from a lossless subscriber of the frame bus, into a frame log, on a thread of its own. A
//! full queue holds the camera, rather than losing a frame: the camera then drops frames
//! itself, which the frames' numbers show.

use crate::bus::{FrameBus, Lossless};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetera_core::{Error, PixelFormat, Result};
use tetera_store::FrameLogWriter;

/// How a recorder runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecorderOptions {
    /// The frames its queue holds before it holds the camera.
    pub capacity: usize,
    /// The frames in each of the log's segments.
    pub segment_frames: u64,
}

impl Default for RecorderOptions {
    /// A queue of 8 frames, segments of 32.
    fn default() -> Self {
        Self {
            capacity: 8,
            segment_frames: 32,
        }
    }
}

/// What a recording did.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RecordStats {
    /// Frames written.
    pub frames: u64,
    /// Their bytes.
    pub bytes: u64,
    /// From its start to its last frame written, s.
    pub seconds: f64,
    /// Times its full queue held the camera.
    pub waits: u64,
    /// Frames missing between those written, by their numbers, with the camera's own count
    /// of frames it dropped: 0 when every frame was kept.
    pub missing: u64,
}

impl RecordStats {
    /// Its rate, bytes a second.
    pub fn rate(&self) -> f64 {
        if self.seconds > 0.0 {
            self.bytes as f64 / self.seconds
        } else {
            0.0
        }
    }
}

/// A recording: frames from the bus into a frame log, until it stops.
pub struct Recorder {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Result<RecordStats>>>,
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recorder").finish_non_exhaustive()
    }
}

impl Recorder {
    /// Records `bus`'s frames, `format` and `width` × `height`, into a new frame log in `dir`,
    /// from now.
    pub fn start(
        bus: &FrameBus,
        dir: &Path,
        format: PixelFormat,
        width: u32,
        height: u32,
        options: RecorderOptions,
    ) -> Result<Self> {
        Self::start_with(bus, dir, (format, width, height), options, Duration::ZERO)
    }

    fn start_with(
        bus: &FrameBus,
        dir: &Path,
        (format, width, height): (PixelFormat, u32, u32),
        options: RecorderOptions,
        slow: Duration,
    ) -> Result<Self> {
        let log = FrameLogWriter::create(dir, format, width, height, options.segment_frames)?;
        let queue = bus.lossless(options.capacity)?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let thread = std::thread::Builder::new()
            .name("recorder".into())
            .spawn(move || record(log, &queue, &stopping, slow))
            .map_err(|e| Error::internal(format!("the recorder won't start: {e}")))?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }

    /// Stops: the frames queued are written, the log finished, and what it did returned. Stop
    /// the camera first, or its next frames go unrecorded.
    pub fn stop(mut self) -> Result<RecordStats> {
        self.halt()
    }

    fn halt(&mut self) -> Result<RecordStats> {
        self.stop.store(true, Ordering::Release);
        match self.thread.take() {
            Some(t) => t
                .join()
                .map_err(|_| Error::internal("the recorder's thread panicked"))?,
            None => Ok(RecordStats::default()),
        }
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if let Err(e) = self.halt() {
            tracing::warn!("a recording ended badly: {e}");
        }
    }
}

/// The recorder's thread.
fn record(
    mut log: FrameLogWriter,
    queue: &Lossless,
    stop: &AtomicBool,
    slow: Duration,
) -> Result<RecordStats> {
    let started = Instant::now();
    let mut stats = RecordStats::default();
    let mut last: Option<u64> = None;
    let mut ended = started;
    let mut write = |frame: tetera_core::Frame| -> Result<()> {
        if !slow.is_zero() {
            std::thread::sleep(slow);
        }
        log.append(&frame)?;
        let meta = frame.meta();
        stats.missing += meta.dropped_before;
        if let Some(prev) = last {
            stats.missing += meta.seq.saturating_sub(prev + 1);
        }
        last = Some(meta.seq);
        stats.frames += 1;
        stats.bytes += frame.layout().row_bytes() as u64 * u64::from(frame.layout().height);
        ended = Instant::now();
        Ok(())
    };
    loop {
        if stop.load(Ordering::Acquire) {
            // What is queued now, and no more: the camera may still be streaming.
            for _ in 0..queue.len() {
                if let Some(frame) = queue.recv(Duration::ZERO) {
                    write(frame)?;
                }
            }
            break;
        }
        if let Some(frame) = queue.recv(Duration::from_millis(20)) {
            write(frame)?;
        }
    }
    stats.waits = queue.waits();
    stats.seconds = ended.duration_since(started).as_secs_f64();
    log.finish()?;
    Ok(stats)
}

#[cfg(test)]
impl Recorder {
    /// A recorder that takes `slow` over each frame, as a slow disk would.
    pub(crate) fn start_slowed(
        bus: &FrameBus,
        dir: &Path,
        format: PixelFormat,
        size: (u32, u32),
        options: RecorderOptions,
        slow: Duration,
    ) -> Result<Self> {
        Self::start_with(bus, dir, (format, size.0, size.1), options, slow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;
    use tetera_core::frame::format::BayerPattern;
    use tetera_core::frame::pool::HostAllocator;
    use tetera_core::{Frame, FrameLayout, FramePool};
    use tetera_store::FrameLog;

    /// A folder of the test's own, removed when it drops.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            Self(
                std::env::temp_dir()
                    .join(format!("tetera-recorder-test-{}-{n}", std::process::id())),
            )
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn format() -> PixelFormat {
        PixelFormat::bayer16(BayerPattern::Rggb, 12).unwrap()
    }

    /// Frame `seq` from `pool`, each site a pattern of its place and `seq`, `dropped` before it.
    fn frame(pool: &FramePool, seq: u64, dropped: u64) -> Frame {
        let mut f = pool.take(Duration::from_secs(10)).unwrap();
        let stride = f.layout().stride;
        for (y, row) in f.data_mut().chunks_mut(stride).enumerate() {
            for x in 0..64 {
                let v = ((x * 3 + y * 5 + seq as usize) % 4_096) as u16;
                row[2 * x..2 * x + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        let meta = f.meta_mut();
        meta.seq = seq;
        meta.dropped_before = dropped;
        f.freeze()
    }

    fn pool(n: usize) -> FramePool {
        FramePool::new(
            FrameLayout::new(format(), 64, 48).unwrap(),
            n,
            &HostAllocator,
        )
        .unwrap()
    }

    #[test]
    fn a_slow_recorder_holds_the_camera_and_loses_no_frame() {
        let scratch = Scratch::new();
        let bus = FrameBus::new();
        let options = RecorderOptions {
            capacity: 2,
            segment_frames: 16,
        };
        let recorder = Recorder::start_slowed(
            &bus,
            &scratch.0,
            format(),
            (64, 48),
            options,
            Duration::from_millis(5),
        )
        .unwrap();
        assert!(format!("{recorder:?}").contains("Recorder"));
        let pool = pool(8);
        let started = Instant::now();
        for seq in 0..40 {
            bus.publish(frame(&pool, seq, 0));
        }
        let publishing = started.elapsed();
        let stats = recorder.stop().unwrap();
        assert_eq!(
            (stats.frames, stats.missing),
            (40, 0),
            "no frame lost: {stats:?}"
        );
        assert!(
            publishing >= Duration::from_millis(150),
            "the camera held: {publishing:?}"
        );
        assert!(stats.waits > 0);
        assert_eq!(stats.bytes, 40 * 64 * 48 * 2);
        assert!(stats.seconds > 0.15 && stats.rate() > 0.0);
        let log = FrameLog::open(&scratch.0).unwrap();
        assert_eq!(log.len(), 40);
        for i in 0..40 {
            let (meta, bytes) = log.read(i).unwrap();
            assert_eq!(meta.seq, i as u64);
            let f = frame(&pool, i as u64, 0);
            let l = f.layout();
            let rows: Vec<u8> = f
                .data()
                .chunks(l.stride)
                .take(48)
                .flat_map(|r| r[..l.row_bytes()].to_vec())
                .collect();
            assert_eq!(bytes, rows, "frame {i}");
        }
    }

    #[test]
    fn missing_frames_are_counted_and_a_stop_ends_however_the_camera_streams() {
        let scratch = Scratch::new();
        let bus = FrameBus::new();
        let recorder = Recorder::start(
            &bus,
            &scratch.0,
            format(),
            64,
            48,
            RecorderOptions::default(),
        )
        .unwrap();
        let pool = pool(8);
        // 1 skipped by number, and 2 the camera says it dropped.
        for (seq, dropped) in [(0, 0), (1, 0), (3, 0), (4, 2)] {
            bus.publish(frame(&pool, seq, dropped));
        }
        std::thread::sleep(Duration::from_millis(50));
        // A camera that goes on streaming while the recording stops.
        let streaming = Arc::new(AtomicBool::new(true));
        let camera = {
            let (bus, streaming) = (bus.clone(), streaming.clone());
            std::thread::spawn(move || {
                let pool = super::tests::pool(8);
                let mut seq = 5;
                while streaming.load(Ordering::Relaxed) {
                    bus.publish(frame(&pool, seq, 0));
                    seq += 1;
                }
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        let asked = Instant::now();
        let stats = recorder.stop().unwrap();
        assert!(asked.elapsed() < Duration::from_secs(2), "it stopped");
        streaming.store(false, Ordering::Relaxed);
        camera.join().unwrap();
        assert!(stats.frames >= 4);
        assert_eq!(stats.missing, 3, "{stats:?}");
        assert_eq!(RecordStats::default().rate(), 0.0);
        // A frame of the wrong shape ends the recording in an error.
        let other = scratch.0.join("other");
        let recorder =
            Recorder::start(&bus, &other, format(), 32, 48, RecorderOptions::default()).unwrap();
        bus.publish(frame(&pool, 0, 0));
        std::thread::sleep(Duration::from_millis(50));
        assert!(recorder.stop().is_err());
        // Bad options are refused.
        let refused = scratch.0.join("refused");
        let none = RecorderOptions {
            capacity: 0,
            ..RecorderOptions::default()
        };
        assert!(Recorder::start(&bus, &refused, format(), 64, 48, none).is_err());
    }

    #[test]
    fn a_recorder_dropped_finishes_its_log() {
        let scratch = Scratch::new();
        let bus = FrameBus::new();
        let recorder = Recorder::start(
            &bus,
            &scratch.0,
            format(),
            64,
            48,
            RecorderOptions::default(),
        )
        .unwrap();
        let pool = pool(4);
        bus.publish(frame(&pool, 0, 0));
        bus.publish(frame(&pool, 1, 0));
        std::thread::sleep(Duration::from_millis(50));
        drop(recorder);
        assert_eq!(FrameLog::open(&scratch.0).unwrap().len(), 2);
    }
}
