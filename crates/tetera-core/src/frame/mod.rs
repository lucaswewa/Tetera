//! The frame model (ADR-0008): one way to hold a frame from any camera, real or virtual,
//! that every later stage shares without copying.
//!
//! - [`format`](mod@format): pixel formats, Bayer patterns and their phase after a crop, layouts with
//!   aligned rows, binning, and scaling between significant bits and the full range;
//! - [`meta`]: what a frame carries besides its pixels;
//! - [`pool`]: the fixed pool of buffers, frames being filled, and frames shared;
//! - [`view`]: safe views of rows, planes, samples and Bayer sites.

pub mod format;
pub mod meta;
pub mod pool;
pub mod view;

pub use format::{
    BayerPattern, Binning, Channel, FrameLayout, PixelFormat, from_full_range, to_full_range,
};
pub use meta::{FrameMeta, LightState};
pub use pool::{Allocator, BufferStorage, Frame, FrameMut, FramePool, HostAllocator, PoolStats};
pub use view::PixelView;
