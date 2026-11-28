//! Tetera's acquisition services, L4 (plan §6.2, ADR-0008): every frame from the camera to
//! whatever needs it, and the live view's work, before any network is involved.
//!
//! - [`bus`]: the frame bus, with lossless and latest-wins subscribers, and the first frame
//!   exposed after a time.
//! - [`preview`]: preview sessions: the newest frame to a JPEG of a viewport's size, through
//!   a live recipe fused on the GPU, each stage timed.
//! - [`protocol`]: the preview protocol's messages (ADR-0019): each frame's header, the
//!   client's state, the acknowledgement window and the slow link's control.
//! - [`recorder`]: the scratch recorder: every frame, from a lossless subscriber, into a frame
//!   log on disk.
//! - [`stats`]: live statistics from the raw frame: histograms, the saturation mask, line
//!   profiles, a region's statistics, a focus value.

pub mod bus;
pub mod preview;
pub mod protocol;
pub mod recorder;
pub mod stats;

pub use bus::{FrameBus, Latest, Lossless};
pub use preview::{Link, Preview, PreviewParams, Session, SessionStats, StageTimes};
pub use recorder::{RecordStats, Recorder, RecorderOptions};
pub use stats::{ChannelStats, LiveStatistics, LiveStats, StatsParams, statistics};
