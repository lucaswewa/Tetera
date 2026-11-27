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
//!
//! From P1.09b:
//! - [`stage`]: the [`Stage`] trait, its [`StageCapabilities`], readings and events;
//! - [`objective`]: the [`ObjectiveCatalogue`], and the [`ObjectiveMount`] where the user says
//!   which objective is fitted;
//! - [`illumination`]: the light's channels, and [`ManualIllumination`];
//! - the fake stage, in [`fake::stage`], and the stage's contract suite.

pub mod camera;
pub mod contract;
pub mod event;
pub mod fake;
pub mod illumination;
pub mod objective;
pub mod stage;

pub use camera::{
    Camera, CameraCapabilities, CameraConfig, CameraEvent, CameraMode, FrameSink, Span, Trigger,
};
pub use event::Events;
pub use fake::stage::{FakeStage, Pace, StageFault};
pub use fake::{CameraFault, FakeCamera};
pub use illumination::{
    Illumination, IlluminationCapabilities, IlluminationEvent, LightChannel, ManualIllumination,
};
pub use objective::{
    ManualObjectiveMount, Objective, ObjectiveCatalogue, ObjectiveEvent, ObjectiveMount,
};
pub use stage::{
    Axis, AxisCapabilities, Stage, StageCapabilities, StageEvent, StageReading, StageState,
};
