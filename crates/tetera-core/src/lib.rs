//! Tetera's foundation layer, L0: the types every other crate shares (plan §6.2). It does
//! no I/O and has no async code.
//!
//! - [`units`]: lengths in micrometres, times as durations, and conversions at the API's edge;
//! - [`geometry`]: points and vectors typed by their unit, affine maps, pixel rectangles,
//!   stage positions;
//! - [`time`]: the monotonic clock, its mapping to UTC, and the drift check;
//! - [`id`]: typed IDs that sort by creation;
//! - [`error`]: one error type, with a kind and a context chain;
//! - [`log`]: log records and the ring that keeps the newest.
//!
//! P1.03 adds the frame model.

pub mod error;
pub mod geometry;
pub mod id;
pub mod log;
pub mod time;
pub mod units;

pub use error::{Error, ErrorKind, Result, ResultExt};
pub use geometry::{Affine2, PointPx, PointUm, Rect, StagePos, VectorPx, VectorUm};
pub use id::Id;
pub use log::{LogEntry, LogLevel, LogRecord, LogRing};
pub use time::{Clock, ManualClock, SystemClock, Timestamp, UtcTime};
pub use units::Um;

/// Tetera's version, as Cargo knows it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Tetera's name with its version, such as `Tetera 0.1.0`.
pub fn name_and_version() -> String {
    format!("Tetera {VERSION}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_and_version_carries_the_package_version() {
        assert_eq!(
            name_and_version(),
            format!("Tetera {}", env!("CARGO_PKG_VERSION"))
        );
    }
}
