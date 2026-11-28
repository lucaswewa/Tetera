//! The store (ADR-0021): projects of datasets under the data root, with their index in its
//! `tetera.db`. A dataset is written through a [`DatasetWriter`], and listed once it is
//! finished; one dropped unfinished, or left by a crash, leaves nothing behind.

use crate::dataset::{
    self, DatasetReader, DatasetSpec, IMAGE, METADATA, Metadata, Samples, THUMBNAIL,
};
use crate::index::{DatasetInfo, FileInfo, Index, Project};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use tetera_core::{Error, Frame, Result, UtcTime};
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;

/// The index's file in the data root, shared with the jobs' history.
pub const DATABASE: &str = "tetera.db";
/// The projects' folder in the data root.
pub const PROJECTS: &str = "projects";

struct Shared {
    root: PathBuf,
    index: Mutex<Index>,
}

impl Shared {
    fn index(&self) -> MutexGuard<'_, Index> {
        self.index.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Removes dataset `info`'s folder and its place in the index.
    fn remove(&self, info: &DatasetInfo) -> Result<()> {
        let dir = self.root.join(&info.path);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)
                .map_err(|e| Error::internal(format!("{}: {e}", dir.display())))?;
        }
        self.index().remove_dataset(&info.id)
    }
}

/// The store.
pub struct Store {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("root", &self.shared.root)
            .finish_non_exhaustive()
    }
}

/// Whether `name` makes a project's folder: 1 to 64 letters, digits, spaces, dots, dashes or
/// underscores, not beginning with a dot or a space.
fn check_name(name: &str) -> Result<()> {
    let fine = (1..=64).contains(&name.chars().count())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || " ._-".contains(c))
        && !name.starts_with(['.', ' '])
        && !name.ends_with(' ');
    if fine {
        Ok(())
    } else {
        Err(Error::invalid_argument(format!(
            "a project's name, {name:?}: 1 to 64 letters, digits, spaces, dots, dashes or underscores, beginning with neither a dot nor a space"
        )))
    }
}

impl Store {
    /// The store under the data root `root`: the datasets abandoned while they were written,
    /// by a crash, removed.
    pub fn open(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root.join(PROJECTS))
            .map_err(|e| Error::internal(format!("{}: {e}", root.display())))?;
        let shared = Arc::new(Shared {
            root: root.to_path_buf(),
            index: Mutex::new(Index::open(&root.join(DATABASE))?),
        });
        let abandoned = shared.index().writing()?;
        for info in &abandoned {
            shared.remove(info)?;
        }
        tracing::debug!(
            abandoned = abandoned.len(),
            "datasets left unfinished removed"
        );
        Ok(Self { shared })
    }

    /// The data root.
    pub fn root(&self) -> &Path {
        &self.shared.root
    }

    /// A new project named `name`, and its folder.
    pub fn create_project(&self, name: &str) -> Result<Project> {
        check_name(name)?;
        let project = Project {
            id: uuid::Uuid::now_v7().to_string(),
            name: name.to_string(),
            created: UtcTime::now(),
        };
        self.shared.index().add_project(&project)?;
        let dir = self.shared.root.join(PROJECTS).join(name);
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("{}: {e}", dir.display())))?;
        Ok(project)
    }

    /// Every project, by name.
    pub fn projects(&self) -> Result<Vec<Project>> {
        self.shared.index().projects()
    }

    /// Project `project`'s datasets, the newest first: those complete.
    pub fn datasets(&self, project: &str) -> Result<Vec<DatasetInfo>> {
        let index = self.shared.index();
        index.project(project)?;
        index.datasets(project)
    }

    /// Dataset `id`, complete.
    pub fn dataset(&self, id: &str) -> Result<DatasetInfo> {
        let info = self.shared.index().dataset(id)?;
        if !info.complete {
            return Err(Error::busy(format!("dataset {id} is being written")));
        }
        Ok(info)
    }

    /// Dataset `id`'s files.
    pub fn files(&self, id: &str) -> Result<Vec<FileInfo>> {
        self.dataset(id)?;
        self.shared.index().files(id)
    }

    /// Dataset `id`, read back.
    pub fn read(&self, id: &str) -> Result<DatasetReader> {
        let info = self.dataset(id)?;
        DatasetReader::open(&self.shared.root.join(info.path))
    }

    /// Dataset `id`'s JPEG thumbnail.
    pub fn thumbnail(&self, id: &str) -> Result<Vec<u8>> {
        let info = self.dataset(id)?;
        let path = self.shared.root.join(info.path).join(THUMBNAIL);
        std::fs::read(&path).map_err(|e| Error::not_found(format!("{}: {e}", path.display())))
    }

    /// Deletes dataset `id`: its folder, and its place in the index.
    pub fn delete(&self, id: &str) -> Result<()> {
        let info = self.dataset(id)?;
        self.shared.remove(&info)
    }

    /// A new dataset in project `project`, as `spec` says, to be written frame by frame.
    pub fn create_dataset(&self, project: &str, spec: DatasetSpec) -> Result<DatasetWriter> {
        dataset::check_format(spec.format)?;
        if spec.frames == 0
            || spec.width < 2
            || spec.height < 2
            || spec.name.is_empty()
            || !(spec.pixel_size_um.is_finite() && spec.pixel_size_um > 0.0)
            || !(spec.interval_s.is_finite() && spec.interval_s >= 0.0)
        {
            return Err(Error::invalid_argument(format!(
                "a dataset of {} frames of {} × {}, named {:?}, its pixels {} µm, {} s apart",
                spec.frames,
                spec.width,
                spec.height,
                spec.name,
                spec.pixel_size_um,
                spec.interval_s
            )));
        }
        let project = self.shared.index().project(project)?;
        let id = uuid::Uuid::now_v7().to_string();
        let dir = self
            .shared
            .root
            .join(PROJECTS)
            .join(&project.name)
            .join(&id);
        let info = DatasetInfo {
            id: id.clone(),
            project: project.name.clone(),
            name: spec.name.clone(),
            kind: spec.kind.clone(),
            created: UtcTime::now(),
            frames: 0,
            bytes: 0,
            path: dataset::relative(&dir, &self.shared.root),
            complete: false,
        };
        // Into the index first, marked as being written, so a crash leaves a trace to clear.
        self.shared.index().add_dataset(&project.id, &info)?;
        let mut writer = DatasetWriter {
            shared: self.shared.clone(),
            info,
            dir: dir.clone(),
            array: None,
            spec,
            frames: Vec::new(),
            first: None,
            done: false,
        };
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("{}: {e}", dir.display())))?;
        writer.array = Some(dataset::create_image(&dir, &writer.spec)?);
        Ok(writer)
    }
}

/// A dataset being written. Dropped before [`finish`](Self::finish), it leaves nothing behind.
pub struct DatasetWriter {
    shared: Arc<Shared>,
    info: DatasetInfo,
    dir: PathBuf,
    array: Option<Array<FilesystemStore>>,
    spec: DatasetSpec,
    frames: Vec<tetera_core::frame::meta::FrameMeta>,
    first: Option<Samples>,
    done: bool,
}

impl std::fmt::Debug for DatasetWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatasetWriter")
            .field("id", &self.info.id)
            .field("written", &self.frames.len())
            .finish_non_exhaustive()
    }
}

impl DatasetWriter {
    /// Its id.
    pub fn id(&self) -> &str {
        &self.info.id
    }

    /// The frames written so far.
    pub fn written(&self) -> u64 {
        self.frames.len() as u64
    }

    /// Writes `frame`, the next.
    pub fn append(&mut self, frame: &Frame) -> Result<()> {
        self.room()?;
        let samples = dataset::samples(frame, &self.spec)?;
        self.push(samples, frame.meta().clone())
    }

    /// Writes the next frame from its rows, `packed`, and its metadata, as a frame log keeps
    /// them.
    pub fn append_packed(
        &mut self,
        meta: tetera_core::frame::meta::FrameMeta,
        packed: &[u8],
    ) -> Result<()> {
        self.room()?;
        let samples = dataset::samples_from_packed(packed, &self.spec)?;
        self.push(samples, meta)
    }

    fn room(&self) -> Result<()> {
        if self.written() >= self.spec.frames {
            return Err(Error::invalid_argument(format!(
                "dataset {} holds {} frames",
                self.info.id, self.spec.frames
            )));
        }
        Ok(())
    }

    fn push(&mut self, samples: Samples, meta: tetera_core::frame::meta::FrameMeta) -> Result<()> {
        if self.first.is_none() {
            self.first = Some(samples.clone());
        }
        let array = self.array.as_ref().expect("made with the writer");
        dataset::store_frame(array, self.written(), samples)?;
        self.frames.push(meta);
        Ok(())
    }

    /// Finishes it: the frames written, at least one, its metadata and thumbnail, and its
    /// place in the index, complete.
    pub fn finish(mut self) -> Result<DatasetInfo> {
        let first = self
            .first
            .take()
            .ok_or_else(|| Error::invalid_argument("a dataset of no frames"))?;
        let written = self.written();
        if written != self.spec.frames {
            let array = self.array.as_mut().expect("made with the writer");
            let mut shape = array.shape().to_vec();
            shape[0] = written;
            array
                .set_shape(shape)
                .map_err(|e| Error::internal(format!("OME-Zarr: {e}")))?;
            array
                .store_metadata()
                .map_err(|e| Error::internal(format!("OME-Zarr: {e}")))?;
            self.spec.frames = written;
        }
        let metadata = Metadata {
            id: self.info.id.clone(),
            created: self.info.created,
            spec: self.spec.clone(),
            frames: std::mem::take(&mut self.frames),
        };
        let json = serde_json::to_vec_pretty(&metadata)
            .map_err(|e| Error::internal(format!("a dataset's metadata: {e}")))?;
        let write = |name: &str, bytes: &[u8]| {
            let path = self.dir.join(name);
            std::fs::write(&path, bytes)
                .map_err(|e| Error::internal(format!("{}: {e}", path.display())))
        };
        write(METADATA, &json)?;
        write(THUMBNAIL, &dataset::thumbnail(&first, &self.spec)?)?;
        let files = [
            (IMAGE, "ome-zarr", dataset::size(&self.dir.join(IMAGE))),
            (METADATA, "metadata", json.len() as u64),
            (
                THUMBNAIL,
                "thumbnail",
                dataset::size_of(&self.dir.join(THUMBNAIL)),
            ),
        ]
        .map(|(path, kind, bytes)| FileInfo {
            path: path.into(),
            kind: kind.into(),
            bytes,
        });
        self.shared
            .index()
            .complete(&self.info.id, written, &files)?;
        self.done = true;
        self.shared.index().dataset(&self.info.id)
    }
}

impl Drop for DatasetWriter {
    /// Unfinished: its folder and its place in the index removed.
    fn drop(&mut self) {
        if !self.done {
            self.array = None;
            if let Err(e) = self.shared.remove(&self.info) {
                tracing::warn!("dataset {} not cleared: {e}", self.info.id);
            }
        }
    }
}
