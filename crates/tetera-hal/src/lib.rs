//! Tetera's hardware abstraction layer, L2 (plan §6.8, ADR-0013): one interface for each kind
//! of device, which a real driver, a virtual device, a fake, or a bridge to another process can
//! each implement; fakes; and the contract suites every implementation must pass.
//!
//! From P1.09a:
//! - [`camera`]: the [`Camera`] trait, its [`CameraCapabilities`] and [`CameraConfig`], its
//!   events, and the [`FrameSink`] it streams into;
//! - [`event`]: devices' events, to each subscriber on a channel;
//! - [`fake`]: the fake camera, with faults on demand;
//! - [`contract`]: the camera's contract suite.

pub mod camera;
pub mod contract;
pub mod event;
pub mod fake;

pub use camera::{
    Camera, CameraCapabilities, CameraConfig, CameraEvent, CameraMode, FrameSink, Span, Trigger,
};
pub use event::Events;
pub use fake::{CameraFault, FakeCamera};
