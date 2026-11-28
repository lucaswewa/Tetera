//! Tetera's acquisition services, L4 (plan §6.2, ADR-0008): every frame from the camera to
//! whatever needs it, and the live view's work, before any network is involved.
//!
//! - [`bus`]: the frame bus, with lossless and latest-wins subscribers, and the first frame
//!   exposed after a time.
//! - [`preview`]: preview sessions: the newest frame to a JPEG of a viewport's size, through
//!   a live recipe fused on the GPU, each stage timed.
//! - [`stats`]: live statistics from the raw frame: histograms, the saturation mask, line
//!   profiles, a region's statistics, a focus value.

pub mod bus;
pub mod preview;
pub mod stats;

pub use bus::{FrameBus, Latest, Lossless};
pub use preview::{Link, Preview, PreviewParams, Session, SessionStats, StageTimes};
pub use stats::{ChannelStats, LiveStatistics, LiveStats, StatsParams, statistics};
