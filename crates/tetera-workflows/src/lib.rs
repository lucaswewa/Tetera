//! Tetera's workflows, L4 (plan §6.2 and §6.10): work that drives the instrument from start to
//! end, run as jobs (P1.15a).
//!
//! - [`capture`]: one frame, N at an interval, or a burst at the full rate, into a dataset, the
//!   camera switched from its binned live mode to full resolution and back (D5).

pub mod capture;

pub use capture::{CaptureKind, CaptureRequest, CaptureResult, Rig, capture};

#[cfg(test)]
mod tests;
