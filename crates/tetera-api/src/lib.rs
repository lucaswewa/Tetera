//! Tetera's HTTP interface, L5 (plan §6.2, [ADR-0017]): the instrument as teta-wot Things,
//! with their Thing Descriptions, the OpenAPI document, observation over WebSocket and
//! server-sent events, and the interactive docs at `/docs`. It is the only crate that
//! depends on `teta-wot` ([ADR-0004]).
//!
//! - [`instrument`]: the virtual instrument, assembled from configuration.
//! - [`system`]: the `system` Thing, what Tetera runs on.
//! - [`simulation`]: the `simulation` Thing, the sample and imperfections on demand.
//! - [`camera`], [`stage`], [`objective`] and [`illumination`]: the device Things.
//! - [`preview`]: the `preview` Thing, the live preview's stream (ADR-0019).
//! - [`media`]: Tetera's own routes: the preview as MJPEG, and the `live-view` page.
//! - [`jobs`]: the `jobs` Thing (ADR-0020), and the job service's options.
//! - [`error`]: Tetera's errors as teta-wot's.
//! - [`server`]: the Things served, and shut down gracefully.
//!
//! [ADR-0004]: https://github.com/lucaswewa/Tetera/blob/main/docs/adr/0004-teta-wot-v0-2-0-extended-upstream-first.md
//! [ADR-0017]: https://github.com/lucaswewa/Tetera/blob/main/docs/adr/0017-control-plane-and-media-plane.md

pub mod camera;
pub mod error;
pub mod host;
pub mod illumination;
pub mod instrument;
pub mod jobs;
pub mod media;
pub mod objective;
pub mod preview;
pub mod server;
pub mod simulation;
pub mod stage;
pub mod system;

pub use instrument::{Instrument, ObjectivesConfig, Sample, SensorSize, SimulatorConfig};
pub use jobs::JobsOptions;
pub use server::{Roots, Service, build, build_with, shutdown_signal};

#[cfg(test)]
mod tests;
