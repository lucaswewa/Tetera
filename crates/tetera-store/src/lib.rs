//! Tetera's dataset store, L2 (plan §6.2 and §6.9, ADR-0021): projects of datasets under the
//! data root, each dataset's frames in OME-Zarr v3 with OME-NGFF 0.5's metadata, its
//! `metadata.json` and a JPEG thumbnail, and their index in SQLite.
//!
//! - [`store`]: the store: projects, and datasets written through a [`DatasetWriter`], listed
//!   once finished; one abandoned leaves nothing behind.
//! - [`dataset`]: a dataset's folder: its OME-Zarr image, metadata and thumbnail, read back.
//! - [`index`]: the index of projects, datasets and their files.
//! - [`framelog`]: the scratch frame log: raw frames appended to segment files as fast as the
//!   camera makes them.
//! - [`ingest`](mod@ingest): a frame log into a dataset, step by step, stopped when told.

pub mod dataset;
pub mod framelog;
pub mod index;
pub mod ingest;
pub mod store;

pub use dataset::{DatasetReader, DatasetSpec, Metadata, Samples, Stack};
pub use framelog::{FrameLog, FrameLogWriter, LogEntry, LogHeader};
pub use index::{DatasetInfo, FileInfo, Index, Project};
pub use ingest::{ingest, ingest_first};
pub use store::{DatasetWriter, Store};

#[cfg(test)]
mod tests;
