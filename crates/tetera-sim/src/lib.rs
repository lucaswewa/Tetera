//! Tetera's simulated world, L2 (plan §8): what stands in for the instrument when there is no
//! hardware, on a clock that tests can control (ADR-0014).
//!
//! From P1.10a:
//! - [`clock`]: [`SimClock`], real time at 1×, faster at N×, or stepped by hand, which
//!   simulated devices wait on;
//! - [`motion`]: a stage's moves as trapezoidal [`Profile`]s within each axis's limits, and
//!   feed holds.

pub mod clock;
pub mod motion;

pub use clock::{Rate, SimClock};
pub use motion::{AxisLimits, Profile};
