//! Tetera's workflows, L4 (plan §6.2 and §6.10): work that drives the instrument from start to
//! end, run as jobs (P1.15a).
//!
//! - [`capture`](mod@capture): one frame, N at an interval, or a burst at the full rate, into a
//!   dataset, the camera switched from its binned live mode to full resolution and back (D5).
//! - [`autofocus`](mod@autofocus): the search on the instrument, each position measured on the
//!   first frame exposed after the stage stops (P1.16b).
//! - [`scan`]: scans, planned before they start, run stop and go (P1.17a, ADR-0022).
//! - [`focus`]: autofocus's search through z, against backlash, over a [`FocusProbe`]
//!   (P1.16a); [`focus::synthetic`], a probe of known curves for tests and examples.

pub mod autofocus;
pub mod capture;
pub mod focus;
pub mod scan;

pub use autofocus::{AutofocusRequest, AutofocusResult, FocusRig, autofocus, depth_of_field};
pub use capture::{CaptureKind, CaptureRequest, CaptureResult, Rig, capture};
pub use focus::{Focus, FocusPoint, FocusProbe, SearchOptions, Sweep};

#[cfg(test)]
mod tests;
