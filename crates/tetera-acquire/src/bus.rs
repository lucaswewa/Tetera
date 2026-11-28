//! The frame bus (ADR-0008): each frame from the camera to whatever needs it, shared, never
//! copied. Two kinds of subscriber:
//!
//! - **lossless,** a bounded queue: a full queue holds the publisher until it has room, so a
//!   slow reader slows the camera rather than losing frames. The recorder is one (P1.15b).
//! - **latest-wins,** a one-frame mailbox: a new frame replaces one not yet taken, so a slow
//!   reader always gets the newest. The preview and the statistics are such.
//!
//! And "the first frame exposed after `t`", for workflows. The bus keeps no frame itself:
//! a frame's buffer goes back to the camera's pool once its readers let it go.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};
use tetera_core::{Error, Frame, Result, Timestamp};
use tetera_hal::camera::FrameSink;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Queue {
    frames: Mutex<VecDeque<Frame>>,
    capacity: usize,
    not_full: Condvar,
    not_empty: Condvar,
    closed: AtomicBool,
    /// Times the publisher waited for room.
    waits: AtomicU64,
}

struct Mailbox {
    frame: Mutex<Option<Frame>>,
    ready: Condvar,
    replaced: AtomicU64,
}

#[derive(Default)]
struct Subscribers {
    lossless: Vec<Weak<Queue>>,
    latest: Vec<Weak<Mailbox>>,
}

/// The frame bus.
#[derive(Default)]
pub struct FrameBus {
    subscribers: Mutex<Subscribers>,
    published: AtomicU64,
}

impl std::fmt::Debug for FrameBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameBus")
            .field("published", &self.published())
            .finish_non_exhaustive()
    }
}

/// A lossless subscriber: every frame, in order, from a queue of bounded length.
pub struct Lossless(Arc<Queue>);

/// A latest-wins subscriber: the newest frame not yet taken.
pub struct Latest(Arc<Mailbox>);

impl FrameBus {
    /// A bus with no subscribers.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// A lossless subscriber, its queue `capacity` frames long, 1 or more.
    pub fn lossless(&self, capacity: usize) -> Result<Lossless> {
        if capacity == 0 {
            return Err(Error::invalid_argument(
                "a lossless queue holds a frame or more",
            ));
        }
        let queue = Arc::new(Queue {
            frames: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
            closed: AtomicBool::new(false),
            waits: AtomicU64::new(0),
        });
        lock(&self.subscribers)
            .lossless
            .push(Arc::downgrade(&queue));
        Ok(Lossless(queue))
    }

    /// A latest-wins subscriber.
    pub fn latest(&self) -> Latest {
        let mailbox = Arc::new(Mailbox {
            frame: Mutex::new(None),
            ready: Condvar::new(),
            replaced: AtomicU64::new(0),
        });
        lock(&self.subscribers)
            .latest
            .push(Arc::downgrade(&mailbox));
        Latest(mailbox)
    }

    /// Frames published so far.
    pub fn published(&self) -> u64 {
        self.published.load(Ordering::Relaxed)
    }

    /// Hands `frame` to every subscriber: each mailbox's frame replaced, each queue's frame
    /// added, waiting for room in a full one.
    pub fn publish(&self, frame: Frame) {
        let (lossless, latest) = {
            let mut s = lock(&self.subscribers);
            s.lossless.retain(|q| q.strong_count() > 0);
            s.latest.retain(|m| m.strong_count() > 0);
            let lossless: Vec<Arc<Queue>> = s.lossless.iter().filter_map(Weak::upgrade).collect();
            let latest: Vec<Arc<Mailbox>> = s.latest.iter().filter_map(Weak::upgrade).collect();
            (lossless, latest)
        };
        for mailbox in latest {
            let mut slot = lock(&mailbox.frame);
            if slot.replace(frame.clone()).is_some() {
                mailbox.replaced.fetch_add(1, Ordering::Relaxed);
            }
            mailbox.ready.notify_one();
        }
        for queue in lossless {
            let mut frames = lock(&queue.frames);
            let mut waited = false;
            while frames.len() >= queue.capacity && !queue.closed.load(Ordering::Acquire) {
                waited = true;
                frames = queue
                    .not_full
                    .wait(frames)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if waited {
                queue.waits.fetch_add(1, Ordering::Relaxed);
            }
            if !queue.closed.load(Ordering::Acquire) {
                frames.push_back(frame.clone());
                queue.not_empty.notify_one();
            }
        }
        self.published.fetch_add(1, Ordering::Relaxed);
    }

    /// The first frame whose exposure starts at `t` or later, waiting up to `timeout` for
    /// it: frames published meanwhile that started earlier are passed over.
    pub fn first_after(&self, t: Timestamp, timeout: Duration) -> Result<Frame> {
        let mailbox = self.latest();
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match mailbox.recv(left) {
                Some(frame) if frame.meta().exposure_start >= t => return Ok(frame),
                Some(_) => continue,
                None => {
                    return Err(Error::timeout(format!(
                        "no frame exposed after {t:?} within {timeout:?}"
                    )));
                }
            }
        }
    }
}

impl FrameSink for FrameBus {
    fn deliver(&self, frame: Frame) {
        self.publish(frame);
    }
}

impl Lossless {
    /// The next frame, waiting up to `timeout`.
    pub fn recv(&self, timeout: Duration) -> Option<Frame> {
        let q = &self.0;
        let mut frames = lock(&q.frames);
        let deadline = Instant::now() + timeout;
        while frames.is_empty() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            frames = q
                .not_empty
                .wait_timeout(frames, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        let frame = frames.pop_front();
        q.not_full.notify_one();
        frame
    }

    /// Frames waiting in its queue.
    pub fn len(&self) -> usize {
        lock(&self.0.frames).len()
    }

    /// Whether none waits.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many times the publisher waited for room in its queue.
    pub fn waits(&self) -> u64 {
        self.0.waits.load(Ordering::Relaxed)
    }
}

impl Drop for Lossless {
    /// Unsubscribes, and frees a publisher waiting for room.
    fn drop(&mut self) {
        self.0.closed.store(true, Ordering::Release);
        let _held = lock(&self.0.frames);
        self.0.not_full.notify_all();
    }
}

impl Latest {
    /// The newest frame not yet taken, waiting up to `timeout` for one.
    pub fn recv(&self, timeout: Duration) -> Option<Frame> {
        let m = &self.0;
        let mut slot = lock(&m.frame);
        let deadline = Instant::now() + timeout;
        while slot.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            slot = m
                .ready
                .wait_timeout(slot, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        slot.take()
    }

    /// Frames replaced before they were taken.
    pub fn replaced(&self) -> u64 {
        self.0.replaced.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetera_core::frame::pool::HostAllocator;
    use tetera_core::{FrameLayout, FramePool, PixelFormat};

    /// A pool of `n` small frames.
    fn pool(n: usize) -> FramePool {
        let layout = FrameLayout::new(PixelFormat::Mono8, 8, 8).unwrap();
        FramePool::new(layout, n, &HostAllocator).unwrap()
    }

    /// Frame `seq` from `pool`, exposed at `seq` ms.
    fn frame(pool: &FramePool, seq: u64) -> Frame {
        let mut f = pool.take(Duration::from_secs(5)).unwrap();
        f.meta_mut().seq = seq;
        f.meta_mut().exposure_start = Timestamp(seq * 1_000_000);
        f.freeze()
    }

    #[test]
    fn a_latest_subscriber_gets_the_newest_and_counts_the_rest() {
        let bus = FrameBus::new();
        let latest = bus.latest();
        let pool = pool(8);
        for seq in 0..5 {
            bus.publish(frame(&pool, seq));
        }
        let got = latest.recv(Duration::from_millis(10)).expect("a frame");
        assert_eq!(got.meta().seq, 4, "the newest");
        assert_eq!(latest.replaced(), 4);
        assert!(latest.recv(Duration::from_millis(10)).is_none(), "taken");
        assert_eq!(bus.published(), 5);
        // Only the newest is held: the rest went back to the pool.
        drop(got);
        assert_eq!(pool.stats().in_use, 0);
        // A camera streams into the bus as its sink.
        FrameSink::deliver(&*bus, frame(&pool, 5));
        assert_eq!(
            latest.recv(Duration::from_millis(10)).map(|f| f.meta().seq),
            Some(5)
        );
        assert_eq!(bus.published(), 6);
    }

    #[test]
    fn a_slow_latest_subscriber_drops_frames_and_never_holds_the_publisher() {
        let bus = FrameBus::new();
        let latest = bus.latest();
        let pool = pool(4);
        let started = Instant::now();
        let reader = std::thread::spawn(move || {
            let mut seen = Vec::new();
            while let Some(f) = latest.recv(Duration::from_millis(200)) {
                seen.push(f.meta().seq);
                std::thread::sleep(Duration::from_millis(20));
            }
            (seen, latest.replaced())
        });
        for seq in 0..50 {
            bus.publish(frame(&pool, seq));
            std::thread::sleep(Duration::from_millis(2));
        }
        let publishing = started.elapsed();
        let (seen, replaced) = reader.join().unwrap();
        assert!(
            publishing < Duration::from_millis(500),
            "not held: {publishing:?}"
        );
        assert!(seen.len() < 20, "{seen:?}");
        assert_eq!(seen.last(), Some(&49), "the last frame reaches it");
        assert!(seen.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(
            seen.len() as u64 + replaced,
            50,
            "each frame read or replaced"
        );
    }

    #[test]
    fn a_slow_lossless_subscriber_slows_its_publisher_and_loses_nothing() {
        let bus = FrameBus::new();
        let lossless = bus.lossless(2).unwrap();
        let pool = pool(8);
        let reader = std::thread::spawn(move || {
            let mut seen = Vec::new();
            while let Some(f) = lossless.recv(Duration::from_millis(300)) {
                seen.push(f.meta().seq);
                std::thread::sleep(Duration::from_millis(10));
            }
            (seen, lossless.waits())
        });
        let started = Instant::now();
        for seq in 0..20 {
            bus.publish(frame(&pool, seq));
        }
        let publishing = started.elapsed();
        let (seen, waits) = reader.join().unwrap();
        assert_eq!(seen, (0..20).collect::<Vec<_>>(), "every frame, in order");
        assert!(
            publishing >= Duration::from_millis(150),
            "held by the reader: {publishing:?}"
        );
        assert!(waits > 0);
    }

    #[test]
    fn a_dropped_subscriber_frees_a_waiting_publisher_and_gets_nothing_more() {
        let bus = FrameBus::new();
        let lossless = bus.lossless(1).unwrap();
        let pool = pool(8);
        bus.publish(frame(&pool, 0));
        let publisher = {
            let bus = bus.clone();
            let next = frame(&pool, 1);
            std::thread::spawn(move || bus.publish(next))
        };
        std::thread::sleep(Duration::from_millis(50));
        assert!(!publisher.is_finished(), "waiting for room");
        assert_eq!(lossless.len(), 1);
        drop(lossless);
        publisher.join().unwrap();
        let latest = bus.latest();
        drop(latest);
        bus.publish(frame(&pool, 2));
        assert_eq!(pool.stats().in_use, 0, "no subscriber holds a frame");
        assert!(bus.lossless(0).is_err());
        let s = lock(&bus.subscribers);
        assert!(
            s.lossless.is_empty() && s.latest.is_empty(),
            "the dropped are let go"
        );
    }

    #[test]
    fn a_lossless_queue_counts_its_frames_and_its_publishers_waits() {
        let bus = FrameBus::new();
        let lossless = bus.lossless(3).unwrap();
        let pool = pool(8);
        assert_eq!(
            (lossless.len(), lossless.is_empty(), lossless.waits()),
            (0, true, 0)
        );
        let started = Instant::now();
        assert!(lossless.recv(Duration::from_millis(30)).is_none());
        assert!(
            started.elapsed() >= Duration::from_millis(30),
            "waited for a frame"
        );
        for seq in 0..2 {
            bus.publish(frame(&pool, seq));
        }
        assert_eq!(
            (lossless.len(), lossless.is_empty(), lossless.waits()),
            (2, false, 0)
        );
        bus.publish(frame(&pool, 2));
        // Full: the publisher waits for room, until a frame is taken.
        let publisher = {
            let bus = bus.clone();
            let next = frame(&pool, 3);
            std::thread::spawn(move || bus.publish(next))
        };
        std::thread::sleep(Duration::from_millis(100));
        assert!(!publisher.is_finished(), "waiting for room");
        let taken = lossless
            .recv(Duration::from_millis(10))
            .map(|f| f.meta().seq);
        assert_eq!(taken, Some(0));
        publisher.join().unwrap();
        assert_eq!((lossless.len(), lossless.waits()), (3, 1));
    }

    #[test]
    fn the_first_frame_exposed_after_a_time_is_waited_for() {
        let bus = FrameBus::new();
        let pool = pool(8);
        let waiter = {
            let bus = bus.clone();
            std::thread::spawn(move || {
                bus.first_after(Timestamp(3_000_000), Duration::from_secs(5))
                    .map(|f| f.meta().seq)
            })
        };
        std::thread::sleep(Duration::from_millis(20));
        for seq in 0..6 {
            bus.publish(frame(&pool, seq));
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(waiter.join().unwrap().unwrap(), 3);
        let late = bus.first_after(Timestamp(u64::MAX), Duration::from_millis(20));
        assert_eq!(late.unwrap_err().kind(), tetera_core::ErrorKind::Timeout);
        assert!(format!("{bus:?}").contains("published"));
    }
}
