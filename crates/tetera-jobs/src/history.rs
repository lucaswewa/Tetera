//! The jobs' history (ADR-0020), in SQLite: each job as it last stood, kept across restarts,
//! and pruned by age. A job still queued or running when the service stopped is marked
//! failed, interrupted, when the history next opens.

use crate::job::{Job, JobState};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use tetera_core::{Error, Result, UtcTime};

/// A database error as Tetera's.
fn sql(e: rusqlite::Error) -> Error {
    Error::internal(format!("the jobs' history: {e}"))
}

/// The jobs' history.
#[derive(Debug)]
pub struct History {
    db: Connection,
}

impl History {
    /// The history in the SQLite file at `path`, made if missing.
    pub fn open(path: &Path) -> Result<Self> {
        let db = Connection::open(path).map_err(sql)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS jobs (
                 id TEXT PRIMARY KEY,
                 state TEXT NOT NULL,
                 created INTEGER NOT NULL,
                 finished INTEGER,
                 job TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS jobs_created ON jobs (created);
             CREATE INDEX IF NOT EXISTS jobs_finished ON jobs (finished);",
        )
        .map_err(sql)?;
        Ok(Self { db })
    }

    /// Writes `job` as it stands now.
    pub fn save(&self, job: &Job) -> Result<()> {
        let json = serde_json::to_string(job)
            .map_err(|e| Error::internal(format!("a job as JSON: {e}")))?;
        self.db
            .execute(
                "INSERT OR REPLACE INTO jobs (id, state, created, finished, job)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    job.id,
                    job.state.name(),
                    job.created.unix_nanos(),
                    job.finished.map(UtcTime::unix_nanos),
                    json
                ],
            )
            .map_err(sql)?;
        Ok(())
    }

    /// The job `id`, if the history has it.
    pub fn get(&self, id: &str) -> Result<Option<Job>> {
        let json: Option<String> = self
            .db
            .query_row("SELECT job FROM jobs WHERE id = ?1", [id], |r| r.get(0))
            .optional()
            .map_err(sql)?;
        json.map(|j| parse(&j)).transpose()
    }

    /// The `limit` newest jobs, newest first.
    pub fn recent(&self, limit: usize) -> Result<Vec<Job>> {
        let mut q = self
            .db
            .prepare("SELECT job FROM jobs ORDER BY created DESC, id DESC LIMIT ?1")
            .map_err(sql)?;
        let rows = q
            .query_map([limit as i64], |r| r.get::<_, String>(0))
            .map_err(sql)?;
        rows.map(|r| parse(&r.map_err(sql)?)).collect()
    }

    /// Marks the jobs that were queued or running, when the service last stopped, failed:
    /// how many.
    pub fn interrupted(&self) -> Result<usize> {
        let mut q = self
            .db
            .prepare("SELECT job FROM jobs WHERE state IN ('queued', 'running')")
            .map_err(sql)?;
        let open: Vec<Job> = q
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(sql)?
            .map(|r| parse(&r.map_err(sql)?))
            .collect::<Result<_>>()?;
        for mut job in open.iter().cloned() {
            job.error = Some("interrupted: the service stopped before it ended".into());
            job.finish(JobState::Failed);
            self.save(&job)?;
        }
        Ok(open.len())
    }

    /// Deletes the jobs that ended before `before`: how many.
    pub fn prune(&self, before: UtcTime) -> Result<usize> {
        self.db
            .execute(
                "DELETE FROM jobs WHERE finished IS NOT NULL AND finished < ?1",
                [before.unix_nanos()],
            )
            .map_err(sql)
    }
}

fn parse(json: &str) -> Result<Job> {
    serde_json::from_str(json).map_err(|e| Error::internal(format!("a job in the history: {e}")))
}
