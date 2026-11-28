//! A job (ADR-0020): what it is, where it stands, and what it left behind.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tetera_core::UtcTime;

/// Which runs a job: the hardware queue, one at a time, or the compute pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// It drives the instrument: one at a time, in the order they came.
    Hardware,
    /// It computes: several at once, sharing the GPU through a VRAM budget.
    Compute,
}

/// Where a job stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    /// Waiting for its turn.
    Queued,
    /// Running.
    Running,
    /// It ended well, with its result.
    Succeeded,
    /// It ended with an error.
    Failed,
    /// It was cancelled, before it ran or while it ran.
    Cancelled,
}

impl JobState {
    /// Whether it has ended.
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    /// Its name, as JSON writes it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// A line of a job's log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LogLine {
    /// When it was written.
    pub time: UtcTime,
    /// What it says.
    pub text: String,
}

/// A job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Job {
    /// Its id: a UUID v7, so ids sort by when their jobs were made.
    pub id: String,
    /// Which runs it.
    pub kind: JobKind,
    /// What it does, such as `capture` or `autofocus`.
    pub name: String,
    /// Where it stands.
    pub state: JobState,
    /// How far it has got, 0 to 1.
    pub progress: f64,
    /// What it is doing, or did last.
    pub message: String,
    /// Its log, the latest lines.
    pub log: Vec<LogLine>,
    /// Its result, once it has succeeded.
    pub result: Option<serde_json::Value>,
    /// Why it failed.
    pub error: Option<String>,
    /// Whether it has been asked to stop.
    pub cancel_requested: bool,
    /// The VRAM it asked for, bytes: compute jobs only.
    pub vram: u64,
    /// When it was submitted.
    pub created: UtcTime,
    /// When it started.
    pub started: Option<UtcTime>,
    /// When it ended.
    pub finished: Option<UtcTime>,
}

impl Job {
    /// A job just submitted.
    pub(crate) fn new(id: String, kind: JobKind, name: String, vram: u64) -> Self {
        Self {
            id,
            kind,
            name,
            state: JobState::Queued,
            progress: 0.0,
            message: String::new(),
            log: Vec::new(),
            result: None,
            error: None,
            cancel_requested: false,
            vram,
            created: UtcTime::now(),
            started: None,
            finished: None,
        }
    }

    /// Ends it in `state`, now.
    pub(crate) fn finish(&mut self, state: JobState) {
        self.state = state;
        self.finished = Some(UtcTime::now());
        if state == JobState::Succeeded {
            self.progress = 1.0;
        }
    }
}
