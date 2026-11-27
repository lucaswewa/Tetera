//! Tetera's simulated world, L2 (plan §8): what stands in for the instrument when there is no
//! hardware, on a clock that tests can control (ADR-0014).
//!
//! From P1.10a:
//! - [`clock`]: [`SimClock`], real time at 1×, faster at N×, or stepped by hand, which
//!   simulated devices wait on;
//! - [`motion`]: a stage's moves as trapezoidal [`Profile`]s within each axis's limits, and
//!   feed holds.
//!
//! From P1.10b:
//! - [`stage`]: the [`VirtualStage`], a grblHAL-like stage on the simulated clock, with its
//!   states, soft limits, homing, reports and their track, and faults on demand.
//!
//! From P1.11a:
//! - [`noise`]: integer hashing, improved noise, cellular noise and fBm;
//! - [`he`]: the [`HeSample`], a procedural section of H&E-stained tissue, and colour by
//!   absorbance;
//! - [`render`](mod@render): a region of it rendered at any pixel size and depth, the CPU twin.

pub mod clock;
pub mod he;
pub mod motion;
pub mod noise;
pub mod render;
pub mod stage;

pub use clock::{Rate, SimClock};
pub use he::{HeSample, Nucleus, RedCell, SectionConfig, Stains};
pub use motion::{AxisLimits, Profile};
pub use render::{Depth, Region, Rendered, render};
pub use stage::{
    GrblState, Report, VirtualAxis, VirtualStage, VirtualStageConfig, VirtualStageFault,
};
