//! The `system` Thing: what Tetera runs on. Its version; the platform, WSL's version among it;
//! where the camera renders; the host's memory; the disks of the data and scratch roots.

use crate::host::{self, Disk, Memory, Platform};
use crate::instrument::{Compute, Instrument};
use crate::server::Roots;
use anyhow::Context;
use std::sync::{Arc, OnceLock};
use teta_wot::prelude::*;

/// What Tetera runs on.
#[derive(Thing)]
pub struct System {
    instrument: OnceLock<Arc<Instrument>>,
    roots: OnceLock<Arc<Roots>>,
    platform: OnceLock<Platform>,
}

#[thing_impl]
impl System {
    /// Finds the instrument and the roots, and reads the platform once: asking Windows for
    /// WSL's version takes a moment.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let server = ctx.server();
        let instrument = server
            .service::<Instrument>()
            .context("the server has no instrument")?;
        let roots = server
            .service::<Roots>()
            .context("the server has no roots")?;
        let platform = tokio::task::spawn_blocking(host::platform).await?;
        let _ = self.instrument.set(instrument);
        let _ = self.roots.set(roots);
        let _ = self.platform.set(platform);
        Ok(())
    }

    /// Tetera's version.
    #[property]
    async fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// The operating system: the kernel, the distribution, and WSL's version.
    #[property]
    async fn platform(&self) -> Result<Platform, PropertyError> {
        self.platform
            .get()
            .cloned()
            .ok_or_else(|| PropertyError::failed(anyhow::anyhow!("not started")))
    }

    /// Where the camera renders: the GPU, its memory and driver, or the CPU and why.
    #[property]
    async fn compute(&self) -> Result<Compute, PropertyError> {
        self.instrument
            .get()
            .map(|i| i.compute().clone())
            .ok_or_else(|| PropertyError::failed(anyhow::anyhow!("not started")))
    }

    /// The host's memory, bytes.
    #[property(blocking)]
    fn memory(&self) -> Result<Memory, PropertyError> {
        host::memory()
            .ok_or_else(|| PropertyError::failed(anyhow::anyhow!("the host's memory is unknown")))
    }

    /// The disks of the data root and the scratch root: their size and free space, bytes.
    #[property(blocking)]
    fn disks(&self) -> Vec<Disk> {
        let Some(roots) = self.roots.get() else {
            return Vec::new();
        };
        [("data", &roots.data), ("scratch", &roots.scratch)]
            .into_iter()
            .filter_map(|(role, path)| host::disk(role, path))
            .collect()
    }
}
