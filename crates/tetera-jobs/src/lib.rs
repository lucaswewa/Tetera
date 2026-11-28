//! Tetera's job service, L3 (plan §6.2, D15, ADR-0020): long work that can be followed,
//! cancelled, and looked up afterwards.
//!
//! - [`job`]: a job: its kind, state, progress and message, log, and result.
//! - [`service`]: the service: hardware jobs one at a time from a queue, compute jobs on a pool
//!   that shares the GPU through a VRAM budget, cancel through a token each job checks.
//! - [`history`]: the history in SQLite, kept across restarts and pruned by age.

pub mod history;
pub mod job;
pub mod service;

pub use history::History;
pub use job::{Job, JobKind, JobState, LogLine};
pub use service::{JobContext, JobService, JobsConfig};

#[cfg(test)]
mod tests;
