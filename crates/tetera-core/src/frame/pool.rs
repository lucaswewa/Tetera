//! The frame pool (ADR-0008): a fixed number of buffers, all allocated when the pool is
//! made, so a running camera allocates nothing.
//!
//! [`FramePool::take`] lends a buffer as a [`FrameMut`], which a camera fills and then
//! [freezes](FrameMut::freeze) into a [`Frame`]. Frames share their buffer by reference
//! count; when the last copy drops, the buffer goes back to the pool. With every buffer
//! lent, `take` waits for one to come back, up to its timeout.

use super::format::FrameLayout;
use super::meta::FrameMeta;
use crate::error::{Error, Result};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

/// Every buffer starts on a multiple of this many bytes: a page, for DMA.
pub const BUFFER_ALIGN: usize = 4096;

/// Memory a frame's pixels live in.
pub trait BufferStorage: Send + Sync {
    /// Its bytes.
    fn bytes(&self) -> &[u8];
    /// Its bytes, to write.
    fn bytes_mut(&mut self) -> &mut [u8];
}

/// Where a pool's buffers come from: plain host memory here, pinned memory for CUDA from
/// P1.04.
pub trait Allocator: Send + Sync {
    /// A zeroed buffer of `len` bytes, starting on a multiple of [`BUFFER_ALIGN`].
    fn allocate(&self, len: usize) -> Result<Box<dyn BufferStorage>>;
}

/// Plain host memory.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostAllocator;

/// A host buffer, aligned by starting inside a slightly larger allocation: no unsafe code.
struct HostBuffer {
    bytes: Vec<u8>,
    start: usize,
    len: usize,
}

impl BufferStorage for HostBuffer {
    fn bytes(&self) -> &[u8] {
        &self.bytes[self.start..self.start + self.len]
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[self.start..self.start + self.len]
    }
}

impl Allocator for HostAllocator {
    fn allocate(&self, len: usize) -> Result<Box<dyn BufferStorage>> {
        Ok(Box::new(host_buffer(len)?))
    }
}

/// A host buffer of `len` bytes: one allocation with less than a page to spare.
fn host_buffer(len: usize) -> Result<HostBuffer> {
    let total = len
        .checked_add(BUFFER_ALIGN - 1)
        .ok_or_else(|| Error::invalid_argument(format!("a buffer of {len} bytes is too large")))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total)
        .map_err(|e| Error::internal(format!("{total} bytes for a frame aren't available: {e}")))?;
    // The reservation holds, so the vector doesn't move as it fills.
    bytes.resize(total, 0);
    let start = (BUFFER_ALIGN - bytes.as_ptr() as usize % BUFFER_ALIGN) % BUFFER_ALIGN;
    Ok(HostBuffer { bytes, start, len })
}

/// A pool's counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// Buffers it holds in all.
    pub capacity: usize,
    /// Buffers lent out now.
    pub in_use: usize,
    /// Buffers allocated since it was made: its capacity, and never more.
    pub allocations: u64,
    /// Buffers lent since it was made.
    pub takes: u64,
    /// Takes that had to wait for a buffer to come back.
    pub waits: u64,
    /// Takes that gave up at their timeout.
    pub misses: u64,
}

struct Shared {
    layout: FrameLayout,
    capacity: usize,
    free: Mutex<Vec<Box<dyn BufferStorage>>>,
    returned: Condvar,
    allocations: AtomicU64,
    takes: AtomicU64,
    waits: AtomicU64,
    misses: AtomicU64,
}

impl Shared {
    /// The free list, even if a thread panicked while holding it.
    fn free(&self) -> MutexGuard<'_, Vec<Box<dyn BufferStorage>>> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Hands a buffer back to its pool, or frees it if the pool is gone.
fn give_back(pool: &Weak<Shared>, buffer: Box<dyn BufferStorage>) {
    if let Some(shared) = pool.upgrade() {
        shared.free().push(buffer);
        shared.returned.notify_one();
    }
}

/// A fixed set of frame buffers of one layout. Cloning it shares the same pool.
#[derive(Clone)]
pub struct FramePool {
    shared: Arc<Shared>,
}

impl FramePool {
    /// A pool of `count` buffers for frames of `layout`, all allocated now.
    pub fn new(layout: FrameLayout, count: usize, allocator: &dyn Allocator) -> Result<Self> {
        if count == 0 {
            return Err(Error::invalid_argument(
                "a frame pool needs at least one buffer",
            ));
        }
        let mut free = Vec::with_capacity(count);
        for _ in 0..count {
            free.push(allocator.allocate(layout.bytes())?);
        }
        Ok(Self {
            shared: Arc::new(Shared {
                layout,
                capacity: count,
                free: Mutex::new(free),
                returned: Condvar::new(),
                allocations: AtomicU64::new(count as u64),
                takes: AtomicU64::new(0),
                waits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
            }),
        })
    }

    /// The layout of its frames.
    pub fn layout(&self) -> &FrameLayout {
        &self.shared.layout
    }

    /// Lends a buffer, waiting up to `timeout` for one to come back if all are lent. Its
    /// bytes are whatever the last frame left in them.
    pub fn take(&self, timeout: Duration) -> Result<FrameMut> {
        let shared = &self.shared;
        let deadline = Instant::now().checked_add(timeout);
        let mut free = shared.free();
        let mut waited = false;
        loop {
            if let Some(buffer) = free.pop() {
                shared.takes.fetch_add(1, Ordering::Relaxed);
                if waited {
                    shared.waits.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(self.lend(buffer));
            }
            let left = deadline.map_or(Duration::MAX, |d| {
                d.saturating_duration_since(Instant::now())
            });
            if left.is_zero() {
                shared.misses.fetch_add(1, Ordering::Relaxed);
                return Err(Error::timeout(format!(
                    "all {} frames of the pool were still in use after {} ms",
                    shared.capacity,
                    timeout.as_millis()
                )));
            }
            waited = true;
            free = shared
                .returned
                .wait_timeout(free, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Lends a buffer if one is free now.
    pub fn try_take(&self) -> Option<FrameMut> {
        let buffer = self.shared.free().pop()?;
        self.shared.takes.fetch_add(1, Ordering::Relaxed);
        Some(self.lend(buffer))
    }

    fn lend(&self, buffer: Box<dyn BufferStorage>) -> FrameMut {
        FrameMut {
            buffer: Some(buffer),
            layout: self.shared.layout,
            meta: FrameMeta::default(),
            pool: Arc::downgrade(&self.shared),
        }
    }

    /// Its counts now.
    pub fn stats(&self) -> PoolStats {
        let s = &self.shared;
        PoolStats {
            capacity: s.capacity,
            in_use: s.capacity - s.free().len(),
            allocations: s.allocations.load(Ordering::Relaxed),
            takes: s.takes.load(Ordering::Relaxed),
            waits: s.waits.load(Ordering::Relaxed),
            misses: s.misses.load(Ordering::Relaxed),
        }
    }
}

impl fmt::Debug for FramePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FramePool")
            .field("layout", self.layout())
            .field("stats", &self.stats())
            .finish()
    }
}

/// A frame being filled: one owner, its pixels writable. [`FrameMut::freeze`] makes it a
/// [`Frame`] to share. Dropped unfrozen, its buffer goes back to the pool.
pub struct FrameMut {
    buffer: Option<Box<dyn BufferStorage>>,
    layout: FrameLayout,
    meta: FrameMeta,
    pool: Weak<Shared>,
}

impl FrameMut {
    /// Its layout.
    pub fn layout(&self) -> &FrameLayout {
        &self.layout
    }

    /// Its metadata.
    pub fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    /// Its metadata, to fill.
    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        &mut self.meta
    }

    /// Its bytes.
    pub fn data(&self) -> &[u8] {
        self.buffer.as_deref().map_or(&[], BufferStorage::bytes)
    }

    /// Its bytes, to write.
    pub fn data_mut(&mut self) -> &mut [u8] {
        match self.buffer.as_deref_mut() {
            Some(buffer) => buffer.bytes_mut(),
            None => &mut [],
        }
    }

    /// Makes it a frame to share, read-only from now on.
    pub fn freeze(mut self) -> Frame {
        Frame(Arc::new(FrozenFrame {
            buffer: self.buffer.take(),
            layout: self.layout,
            meta: std::mem::take(&mut self.meta),
            pool: std::mem::take(&mut self.pool),
        }))
    }
}

impl Drop for FrameMut {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            give_back(&self.pool, buffer);
        }
    }
}

impl fmt::Debug for FrameMut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameMut")
            .field("layout", &self.layout)
            .field("meta", &self.meta)
            .finish_non_exhaustive()
    }
}

struct FrozenFrame {
    buffer: Option<Box<dyn BufferStorage>>,
    layout: FrameLayout,
    meta: FrameMeta,
    pool: Weak<Shared>,
}

impl Drop for FrozenFrame {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            give_back(&self.pool, buffer);
        }
    }
}

/// A frame to share: cloning it is cheap and shares its pixels. When the last clone drops,
/// its buffer goes back to the pool.
#[derive(Clone)]
pub struct Frame(Arc<FrozenFrame>);

impl Frame {
    /// Its layout.
    pub fn layout(&self) -> &FrameLayout {
        &self.0.layout
    }

    /// Its metadata.
    pub fn meta(&self) -> &FrameMeta {
        &self.0.meta
    }

    /// Its bytes.
    pub fn data(&self) -> &[u8] {
        self.0.buffer.as_deref().map_or(&[], BufferStorage::bytes)
    }

    /// How many clones share it now.
    pub fn shares(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("layout", self.layout())
            .field("meta", self.meta())
            .field("shares", &self.shares())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;
    use crate::frame::format::{BayerPattern, PixelFormat};
    use std::sync::mpsc;

    fn pool(count: usize) -> FramePool {
        let layout = FrameLayout::new(
            PixelFormat::bayer16(BayerPattern::Rggb, 12).unwrap(),
            64,
            32,
        )
        .unwrap();
        FramePool::new(layout, count, &HostAllocator).unwrap()
    }

    #[test]
    fn buffers_are_aligned_and_zeroed() {
        for len in [1, 4_095, 4_096, 39_923_712] {
            let buffer = HostAllocator.allocate(len).unwrap();
            assert_eq!(buffer.bytes().len(), len);
            assert_eq!(buffer.bytes().as_ptr() as usize % BUFFER_ALIGN, 0, "{len}");
            assert!(buffer.bytes().iter().all(|&b| b == 0));
        }
        assert!(HostAllocator.allocate(usize::MAX).is_err());
        assert!(
            HostAllocator.allocate(usize::MAX - BUFFER_ALIGN).is_err(),
            "more memory than there is"
        );
    }

    #[test]
    fn a_buffer_spares_less_than_a_page() {
        for len in [1, 4_096, 1_000_000] {
            let buffer = host_buffer(len).unwrap();
            assert_eq!(buffer.bytes.len(), len + BUFFER_ALIGN - 1);
            assert!(buffer.start < BUFFER_ALIGN);
        }
    }

    #[test]
    fn a_frame_being_filled_reads_its_own_metadata() {
        let pool = pool(1);
        let mut frame = pool.take(Duration::ZERO).unwrap();
        frame.meta_mut().seq = 7;
        frame.meta_mut().gain = 2.5;
        assert_eq!((frame.meta().seq, frame.meta().gain), (7, 2.5));
    }

    #[test]
    fn a_warm_pool_allocates_nothing_more() {
        let pool = pool(4);
        for i in 0..1_000u64 {
            let mut frame = pool.take(Duration::ZERO).unwrap();
            frame.data_mut()[0] = i as u8;
            frame.meta_mut().seq = i;
            let shared = frame.freeze();
            let copy = shared.clone();
            assert_eq!((copy.meta().seq, copy.data()[0]), (i, i as u8));
        }
        let stats = pool.stats();
        assert_eq!(
            (stats.allocations, stats.takes, stats.in_use, stats.capacity),
            (4, 1_000, 0, 4)
        );
    }

    #[test]
    fn a_buffer_comes_back_only_when_the_last_share_drops() {
        let pool = pool(2);
        let frame = pool.take(Duration::ZERO).unwrap().freeze();
        let (a, b) = (frame.clone(), frame.clone());
        assert_eq!((frame.shares(), pool.stats().in_use), (3, 1));
        drop(frame);
        drop(a);
        assert_eq!(pool.stats().in_use, 1, "one share is still out");
        drop(b);
        assert_eq!(pool.stats().in_use, 0);
        let unfrozen = pool.take(Duration::ZERO).unwrap();
        assert_eq!(pool.stats().in_use, 1);
        drop(unfrozen);
        assert_eq!(pool.stats().in_use, 0, "an unfrozen frame comes back too");
    }

    #[test]
    fn an_exhausted_pool_waits_then_fails_clearly() {
        let pool = pool(2);
        let held: Vec<FrameMut> = (0..2).map(|_| pool.take(Duration::ZERO).unwrap()).collect();
        assert!(pool.try_take().is_none());
        let start = Instant::now();
        let err = pool.take(Duration::from_millis(50)).unwrap_err();
        assert!(
            start.elapsed() >= Duration::from_millis(50),
            "it waited its timeout"
        );
        assert_eq!(err.kind(), ErrorKind::Timeout);
        assert_eq!(
            err.to_string(),
            "all 2 frames of the pool were still in use after 50 ms"
        );
        assert_eq!(pool.stats().misses, 1);
        drop(held);
        assert!(pool.try_take().is_some());
    }

    #[test]
    fn a_waiting_take_gets_the_first_buffer_back() {
        let pool = pool(1);
        let held = pool.take(Duration::ZERO).unwrap();
        let (started, waiting) = mpsc::channel();
        std::thread::scope(|s| {
            let taker = s.spawn(|| {
                started.send(()).unwrap();
                pool.take(Duration::from_secs(10))
            });
            waiting.recv().unwrap();
            drop(held);
            let frame = taker
                .join()
                .unwrap()
                .expect("the returned buffer is lent again");
            assert_eq!(frame.layout(), pool.layout());
        });
        assert_eq!(pool.stats().takes, 2);
    }

    #[test]
    fn a_frame_outliving_its_pool_frees_its_buffer() {
        let pool = pool(1);
        let frame = pool.take(Duration::ZERO).unwrap().freeze();
        let unfrozen = pool.clone().take(Duration::ZERO);
        assert!(unfrozen.is_err());
        drop(pool);
        assert_eq!(frame.data().len(), 64 * 2 * 32);
        drop(frame);
    }

    #[test]
    fn many_threads_take_share_and_return_without_losing_a_buffer() {
        let pool = pool(4);
        std::thread::scope(|s| {
            for t in 0..8u64 {
                let pool = &pool;
                s.spawn(move || {
                    for i in 0..500u64 {
                        let mut frame = pool.take(Duration::from_secs(10)).unwrap();
                        frame.meta_mut().seq = t * 1_000 + i;
                        let shared = frame.freeze();
                        let copy = shared.clone();
                        assert_eq!(copy.meta().seq, t * 1_000 + i);
                    }
                });
            }
        });
        let stats = pool.stats();
        assert_eq!(
            (stats.in_use, stats.takes, stats.allocations, stats.misses),
            (0, 4_000, 4, 0)
        );
        assert!(stats.waits <= stats.takes);
    }

    #[test]
    fn an_empty_pool_is_refused() {
        let layout = FrameLayout::new(PixelFormat::Mono8, 8, 8).unwrap();
        assert!(FramePool::new(layout, 0, &HostAllocator).is_err());
    }

    #[test]
    fn frames_describe_themselves() {
        let pool = pool(1);
        let frame = pool.take(Duration::ZERO).unwrap();
        assert!(format!("{frame:?}").starts_with("FrameMut"));
        let frame = frame.freeze();
        assert!(format!("{frame:?}").contains("shares: 1"));
        assert!(format!("{pool:?}").contains("in_use: 1"));
    }
}
