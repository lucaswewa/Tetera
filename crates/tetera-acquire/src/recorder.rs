//! The scratch recorder (P1.15c, ADR-0008, ADR-0021): every frame of the camera's stream,
//! from a lossless subscriber of the frame bus, into a frame log. One thread copies each
//! frame out at once, so its buffer goes straight back to the camera's pool; another writes
//! the copies to disk, with `buffered` of them in memory between, to ride out the disk's
//! stalls. When that is full too, the bus holds the camera rather than losing a frame: the
//! camera then drops frames itself, which the frames' numbers show.

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
    /// The frames its subscriber's queue holds.
    pub capacity: usize,
    /// The frames copied out and waiting for the disk, in memory: about 2 s at 289 MB/s by
    /// default, 640 MB of full frames.
    pub buffered: usize,
    /// The frames in each of the log's segments.
    pub segment_frames: u64,
}

impl Default for RecorderOptions {
    /// A queue of 2 frames, 16 buffered, segments of 32.
    fn default() -> Self {
        Self {
            capacity: 2,
            buffered: 16,
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
        if options.buffered == 0 {
            return Err(Error::invalid_argument(
                "a recorder holds a frame or more for the disk",
            ));
        }
        let log = FrameLogWriter::create(dir, format, width, height, options.segment_frames)?;
        let queue = bus.lossless(options.capacity)?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let buffered = options.buffered;
        let thread = std::thread::Builder::new()
            .name("recorder".into())
            .spawn(move || record(log, &queue, &stopping, buffered, slow))
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

/// A frame copied out: its metadata and its rows, packed.
type Copied = (tetera_core::frame::meta::FrameMeta, Vec<u8>);

/// The recorder's copying thread: each frame out of the bus's buffer at once, to the writer.
fn record(
    log: FrameLogWriter,
    queue: &Lossless,
    stop: &AtomicBool,
    buffered: usize,
    slow: Duration,
) -> Result<RecordStats> {
    let started = Instant::now();
    let (to_disk, copies) = std::sync::mpsc::sync_channel::<Copied>(buffered);
    let (spare, spares) = std::sync::mpsc::channel::<Vec<u8>>();
    let writer = std::thread::Builder::new()
        .name("recorder's writer".into())
        .spawn(move || write(log, &copies, &spare, slow))
        .map_err(|e| Error::internal(format!("the recorder's writer won't start: {e}")))?;
    let mut stats = RecordStats::default();
    let mut last: Option<u64> = None;
    let mut copy = |frame: tetera_core::Frame| -> bool {
        let layout = frame.layout();
        let row = layout.row_bytes();
        let mut bytes = spares.try_recv().unwrap_or_default();
        bytes.clear();
        for r in frame
            .data()
            .chunks(layout.stride)
            .take(layout.height as usize)
        {
            bytes.extend_from_slice(&r[..row.min(r.len())]);
        }
        let meta = frame.meta().clone();
        // The camera's buffer, back to its pool before the disk is waited for.
        drop(frame);
        stats.missing += meta.dropped_before;
        if let Some(prev) = last {
            stats.missing += meta.seq.saturating_sub(prev + 1);
        }
        last = Some(meta.seq);
        stats.frames += 1;
        stats.bytes += bytes.len() as u64;
        to_disk.send((meta, bytes)).is_ok()
    };
    let mut writing = true;
    while writing {
        if stop.load(Ordering::Acquire) {
            // What is queued now, and no more: the camera may still be streaming.
            for _ in 0..queue.len() {
                if let Some(frame) = queue.recv(Duration::ZERO)
                    && !copy(frame)
                {
                    break;
                }
            }
            break;
        }
        if let Some(frame) = queue.recv(Duration::from_millis(20)) {
            writing = copy(frame);
        }
    }
    drop(to_disk);
    let (log, ended) = writer
        .join()
        .map_err(|_| Error::internal("the recorder's writer panicked"))??;
    stats.waits = queue.waits();
    stats.seconds = ended
        .unwrap_or(started)
        .duration_since(started)
        .as_secs_f64();
    log.finish()?;
    Ok(stats)
}

/// The recorder's writing thread: each copy onto the disk, in order, its buffer back for the
/// next. The log back, and when it wrote its last frame.
fn write(
    mut log: FrameLogWriter,
    copies: &std::sync::mpsc::Receiver<Copied>,
    spare: &std::sync::mpsc::Sender<Vec<u8>>,
    slow: Duration,
) -> Result<(FrameLogWriter, Option<Instant>)> {
    let mut ended = None;
    for (meta, bytes) in copies {
        if !slow.is_zero() {
            std::thread::sleep(slow);
        }
        log.append_packed(&meta, &bytes)?;
        ended = Some(Instant::now());
        let _ = spare.send(bytes);
    }
    Ok((log, ended))
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
            buffered: 2,
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
        assert!(stats.seconds > 0.15);
        assert!((stats.rate() - stats.bytes as f64 / stats.seconds).abs() < 1e-6);
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
        for bad in [
            RecorderOptions {
                capacity: 0,
                ..RecorderOptions::default()
            },
            RecorderOptions {
                buffered: 0,
                ..RecorderOptions::default()
            },
        ] {
            assert!(Recorder::start(&bus, &refused, format(), 64, 48, bad).is_err());
        }
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
