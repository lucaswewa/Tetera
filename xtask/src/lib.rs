//! Tetera's workspace tasks, run as `cargo xtask <task>`:
//!
//! - `layering` checks the layer rules of ADR-0005;
//! - `gate <quick|phase|checkpoint|milestone>` runs a local gate of plan §8;
//! - `doctor` checks this machine's tools, disk and memory.
//!
//! Everything runs on Linux, in WSL. The local gates are the merge gate: GitHub runs
//! nothing (ADR-0005).

pub mod cli;
pub mod doctor;
pub mod exec;
pub mod gate;
pub mod layering;
pub mod metadata;
pub mod reach;
pub mod runner;
pub mod stress;
pub mod toml_lite;
