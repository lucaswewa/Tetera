//! Ingest (ADR-0021): a frame log into a dataset, frame by frame, saying how far it has got
//! and stopping when told. Stopped, it leaves no dataset: the half-written one goes with its
//! writer. A job runs it (P1.15a), its context's progress and cancel check as `step`.

use crate::dataset::DatasetSpec;
use crate::framelog::FrameLog;
use crate::index::DatasetInfo;
use crate::store::Store;
use tetera_core::Result;

/// Writes `log`'s frames into a new dataset in `project`. `spec` names and describes it; its
/// format, size and number of frames come from the log. `step(done, of)` is called before each
/// frame and once all are written: an error from it stops the ingest, and the dataset is
/// removed.
pub fn ingest(
    store: &Store,
    project: &str,
    log: &FrameLog,
    spec: DatasetSpec,
    mut step: impl FnMut(u64, u64) -> Result<()>,
) -> Result<DatasetInfo> {
    let header = log.header();
    let of = log.len() as u64;
    let spec = DatasetSpec {
        format: header.format,
        width: header.width,
        height: header.height,
        frames: of,
        ..spec
    };
    let mut writer = store.create_dataset(project, spec)?;
    for i in 0..log.len() {
        step(i as u64, of)?;
        let (meta, packed) = log.read(i)?;
        writer.append_packed(meta, &packed)?;
    }
    step(of, of)?;
    writer.finish()
}
