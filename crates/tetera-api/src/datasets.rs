//! The `datasets` Thing (P1.15d, ADR-0021): the projects, their datasets, a dataset's files
//! and metadata, and delete. A dataset's thumbnail is a media route (`media`).

use crate::error;
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use teta_wot::prelude::*;
use tetera_store::{DatasetInfo, FileInfo, Metadata, Project, Store};

/// A dataset, all of it but its frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Dataset {
    /// It, as the index has it.
    pub info: DatasetInfo,
    /// Its files.
    pub files: Vec<FileInfo>,
    /// Its `metadata.json`: what it holds, and each frame's metadata.
    pub metadata: Metadata,
    /// Its thumbnail's path on the server.
    pub thumbnail: String,
}

/// The datasets.
#[derive(Thing)]
pub struct DatasetsThing {
    store: OnceLock<Arc<Store>>,
}

impl DatasetsThing {
    /// `work` with the store, off the async threads: SQLite and the disk block.
    async fn with<T: Send + 'static>(
        &self,
        work: impl FnOnce(&Store) -> tetera_core::Result<T> + Send + 'static,
    ) -> Result<T, ActionError> {
        let store = self
            .store
            .get()
            .cloned()
            .ok_or_else(|| ActionError::handled("the datasets haven't started"))?;
        tokio::task::spawn_blocking(move || work(&store))
            .await
            .map_err(anyhow::Error::from)?
            .map_err(error::action)
    }
}

#[thing_impl]
impl DatasetsThing {
    /// Finds the store.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let store = ctx
            .server()
            .service::<Store>()
            .context("the server has no store")?;
        let _ = self.store.set(store);
        Ok(())
    }

    /// Every project, by name.
    #[property]
    async fn projects(&self) -> Result<Vec<Project>, PropertyError> {
        let store = self
            .store
            .get()
            .cloned()
            .ok_or_else(|| PropertyError::failed(anyhow::anyhow!("not started")))?;
        tokio::task::spawn_blocking(move || store.projects())
            .await
            .map_err(|e| PropertyError::failed(anyhow::Error::from(e)))?
            .map_err(|e| error::property(e, None))
    }

    /// A new project named `name`, and its folder: 1 to 64 letters, digits, spaces, dots,
    /// dashes or underscores.
    #[action(global_lock = false)]
    async fn create_project(&self, name: String) -> Result<Project, ActionError> {
        self.with(move |s| s.create_project(&name)).await
    }

    /// Project `project`'s complete datasets, the newest first.
    #[action(global_lock = false)]
    async fn list(&self, project: String) -> Result<Vec<DatasetInfo>, ActionError> {
        self.with(move |s| s.datasets(&project)).await
    }

    /// Dataset `id`: it, its files, its metadata, and its thumbnail's path. One being
    /// written is busy.
    #[action(global_lock = false)]
    async fn get(&self, id: String) -> Result<Dataset, ActionError> {
        self.with(move |s| {
            Ok(Dataset {
                info: s.dataset(&id)?,
                files: s.files(&id)?,
                metadata: s.read(&id)?.metadata().clone(),
                thumbnail: crate::media::thumbnail_path(&id),
            })
        })
        .await
    }

    /// Deletes dataset `id`: its folder, and its place in the index. One being written is
    /// busy.
    #[action(global_lock = false)]
    async fn delete(&self, id: String) -> Result<(), ActionError> {
        self.with(move |s| s.delete(&id)).await
    }
}
