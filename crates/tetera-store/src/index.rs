//! The index (ADR-0021), in SQLite: projects, their datasets, and each dataset's files. A
//! dataset is listed once it is complete; while it is written, it is marked so, and a dataset
//! still marked when the store opens was abandoned.

use rusqlite::{Connection, OptionalExtension, params};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tetera_core::{Error, Result, UtcTime};

fn sql(e: rusqlite::Error) -> Error {
    Error::internal(format!("the datasets' index: {e}"))
}

/// A project: a named folder of datasets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Project {
    /// Its id.
    pub id: String,
    /// Its name, which is also its folder's.
    pub name: String,
    /// When it was made.
    pub created: UtcTime,
}

/// A dataset, as the index has it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DatasetInfo {
    /// Its id.
    pub id: String,
    /// Its project's name.
    pub project: String,
    /// Its name.
    pub name: String,
    /// What made it.
    pub kind: String,
    /// When it was made.
    pub created: UtcTime,
    /// Its frames.
    pub frames: u64,
    /// The bytes its files take.
    pub bytes: u64,
    /// Its folder, in the data root.
    pub path: PathBuf,
    /// Whether it is complete: else it is being written.
    pub complete: bool,
}

/// One of a dataset's files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FileInfo {
    /// Its path, in the dataset's folder.
    pub path: String,
    /// What it is: `ome-zarr`, `metadata` or `thumbnail`.
    pub kind: String,
    /// The bytes it takes, a folder's files in all.
    pub bytes: u64,
}

/// The index.
#[derive(Debug)]
pub struct Index {
    db: Connection,
}

const DATASET: &str =
    "SELECT d.id, p.name, d.name, d.kind, d.created, d.frames, d.bytes, d.path, d.state
                       FROM datasets d JOIN projects p ON p.id = d.project";

fn dataset(r: &rusqlite::Row<'_>) -> rusqlite::Result<DatasetInfo> {
    Ok(DatasetInfo {
        id: r.get(0)?,
        project: r.get(1)?,
        name: r.get(2)?,
        kind: r.get(3)?,
        created: UtcTime::from_unix_nanos(r.get(4)?),
        frames: r.get::<_, i64>(5)? as u64,
        bytes: r.get::<_, i64>(6)? as u64,
        path: PathBuf::from(r.get::<_, String>(7)?),
        complete: r.get::<_, String>(8)? == "complete",
    })
}

impl Index {
    /// The index in the SQLite file at `path`, made if missing.
    pub fn open(path: &Path) -> Result<Self> {
        let db = Connection::open(path).map_err(sql)?;
        db.busy_timeout(Duration::from_secs(5)).map_err(sql)?;
        db.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS projects (
                 id TEXT PRIMARY KEY,
                 name TEXT NOT NULL UNIQUE,
                 created INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS datasets (
                 id TEXT PRIMARY KEY,
                 project TEXT NOT NULL REFERENCES projects (id),
                 name TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 state TEXT NOT NULL,
                 created INTEGER NOT NULL,
                 frames INTEGER NOT NULL,
                 bytes INTEGER NOT NULL,
                 path TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS files (
                 dataset TEXT NOT NULL REFERENCES datasets (id) ON DELETE CASCADE,
                 path TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 bytes INTEGER NOT NULL,
                 PRIMARY KEY (dataset, path)
             );",
        )
        .map_err(sql)?;
        Ok(Self { db })
    }

    /// Adds `project`: refused if one of its name exists.
    pub fn add_project(&self, project: &Project) -> Result<()> {
        self.db
            .execute(
                "INSERT INTO projects (id, name, created) VALUES (?1, ?2, ?3)",
                params![project.id, project.name, project.created.unix_nanos()],
            )
            .map_err(|e| match e.sqlite_error_code() {
                Some(rusqlite::ErrorCode::ConstraintViolation) => {
                    Error::invalid_argument(format!("a project named {:?} exists", project.name))
                }
                _ => sql(e),
            })?;
        Ok(())
    }

    /// Every project, by name.
    pub fn projects(&self) -> Result<Vec<Project>> {
        let mut q = self
            .db
            .prepare("SELECT id, name, created FROM projects ORDER BY name")
            .map_err(sql)?;
        let rows = q
            .query_map([], |r| {
                Ok(Project {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    created: UtcTime::from_unix_nanos(r.get(2)?),
                })
            })
            .map_err(sql)?;
        rows.map(|r| r.map_err(sql)).collect()
    }

    /// The project named `name`.
    pub fn project(&self, name: &str) -> Result<Project> {
        self.projects()?
            .into_iter()
            .find(|p| p.name == name)
            .ok_or_else(|| Error::not_found(format!("no project named {name:?}")))
    }

    /// Adds `dataset`, being written, to project `project_id`.
    pub fn add_dataset(&self, project_id: &str, dataset: &DatasetInfo) -> Result<()> {
        self.db
            .execute(
                "INSERT INTO datasets (id, project, name, kind, state, created, frames, bytes, path)
                 VALUES (?1, ?2, ?3, ?4, 'writing', ?5, 0, 0, ?6)",
                params![
                    dataset.id,
                    project_id,
                    dataset.name,
                    dataset.kind,
                    dataset.created.unix_nanos(),
                    dataset.path.to_string_lossy()
                ],
            )
            .map_err(sql)?;
        Ok(())
    }

    /// Marks dataset `id` complete, with its frames and files.
    pub fn complete(&mut self, id: &str, frames: u64, files: &[FileInfo]) -> Result<()> {
        let t = self.db.transaction().map_err(sql)?;
        let bytes: u64 = files.iter().map(|f| f.bytes).sum();
        let changed = t
            .execute(
                "UPDATE datasets SET state = 'complete', frames = ?2, bytes = ?3 WHERE id = ?1",
                params![id, frames as i64, bytes as i64],
            )
            .map_err(sql)?;
        if changed != 1 {
            return Err(Error::not_found(format!("no dataset {id} being written")));
        }
        for f in files {
            t.execute(
                "INSERT INTO files (dataset, path, kind, bytes) VALUES (?1, ?2, ?3, ?4)",
                params![id, f.path, f.kind, f.bytes as i64],
            )
            .map_err(sql)?;
        }
        t.commit().map_err(sql)
    }

    /// Removes dataset `id` and its files from the index.
    pub fn remove_dataset(&self, id: &str) -> Result<()> {
        self.db
            .execute("DELETE FROM datasets WHERE id = ?1", [id])
            .map_err(sql)?;
        Ok(())
    }

    /// Project `project`'s complete datasets, the newest first.
    pub fn datasets(&self, project: &str) -> Result<Vec<DatasetInfo>> {
        let mut q = self
            .db
            .prepare(&format!(
                "{DATASET} WHERE p.name = ?1 AND d.state = 'complete' ORDER BY d.created DESC, d.id DESC"
            ))
            .map_err(sql)?;
        let rows = q.query_map([project], dataset).map_err(sql)?;
        rows.map(|r| r.map_err(sql)).collect()
    }

    /// Dataset `id`, complete or not.
    pub fn dataset(&self, id: &str) -> Result<DatasetInfo> {
        self.db
            .query_row(&format!("{DATASET} WHERE d.id = ?1"), [id], dataset)
            .optional()
            .map_err(sql)?
            .ok_or_else(|| Error::not_found(format!("no dataset {id}")))
    }

    /// The datasets still being written.
    pub fn writing(&self) -> Result<Vec<DatasetInfo>> {
        let mut q = self
            .db
            .prepare(&format!("{DATASET} WHERE d.state = 'writing'"))
            .map_err(sql)?;
        let rows = q.query_map([], dataset).map_err(sql)?;
        rows.map(|r| r.map_err(sql)).collect()
    }

    /// Dataset `id`'s files.
    pub fn files(&self, id: &str) -> Result<Vec<FileInfo>> {
        let mut q = self
            .db
            .prepare("SELECT path, kind, bytes FROM files WHERE dataset = ?1 ORDER BY path")
            .map_err(sql)?;
        let rows = q
            .query_map([id], |r| {
                Ok(FileInfo {
                    path: r.get(0)?,
                    kind: r.get(1)?,
                    bytes: r.get::<_, i64>(2)? as u64,
                })
            })
            .map_err(sql)?;
        rows.map(|r| r.map_err(sql)).collect()
    }
}
