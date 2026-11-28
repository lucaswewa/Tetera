//! The server: the instrument's Things on teta-wot, with its OpenAPI document and its
//! interactive docs, and Tetera's own routes (`media`), served until a shutdown signal, then
//! shut down gracefully.

use crate::camera::CameraThing;
use crate::illumination::IlluminationThing;
use crate::instrument::Instrument;
use crate::objective::ObjectiveThing;
use crate::preview::PreviewThing;
use crate::simulation::Simulation;
use crate::stage::StageThing;
use crate::system::System;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use teta_wot::server::ThingServer;
use tetera_core::{Error, Result};
use tokio::net::TcpListener;

pub use teta_wot::server::shutdown_signal;

/// Where Tetera keeps its data and its scratch files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Roots {
    /// Datasets and settings: `./.tetera/` in development, `/var/lib/tetera/` for the
    /// service.
    pub data: PathBuf,
    /// Raw frame logs and other scratch.
    pub scratch: PathBuf,
}

impl Roots {
    /// The roots, made if missing.
    pub fn make(data: PathBuf, scratch: PathBuf) -> Result<Self> {
        for dir in [&data, &scratch] {
            std::fs::create_dir_all(dir).map_err(|e| {
                Error::invalid_argument(format!("{} can't be made: {e}", dir.display()))
            })?;
        }
        Ok(Self { data, scratch })
    }

    /// Where the Things' settings are kept.
    pub fn settings(&self) -> PathBuf {
        self.data.join("settings")
    }
}

/// How long the server waits for invocations and requests to finish when it shuts down.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// The instrument's Things, ready to serve.
pub struct Service {
    server: ThingServer,
}

impl std::fmt::Debug for Service {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Service").finish_non_exhaustive()
    }
}

/// The names the Things are served under.
pub const THINGS: [&str; 7] = [
    "system",
    "simulation",
    "camera",
    "stage",
    "objective",
    "illumination",
    "preview",
];

/// The server for `instrument`, its settings under `roots`.
pub fn build(instrument: Arc<Instrument>, roots: Roots) -> Result<Service> {
    let preview = Arc::new(PreviewThing::default());
    let media = crate::media::routes(instrument.clone(), preview.clone());
    let server = ThingServer::builder()
        .api_info("Tetera", env!("CARGO_PKG_VERSION"))
        .settings_folder(roots.settings())
        .shutdown_grace(SHUTDOWN_GRACE)
        .service(instrument)
        .service(Arc::new(roots))
        .thing(THINGS[0], System::default())
        .thing(THINGS[1], Simulation::default())
        .thing(THINGS[2], CameraThing::default())
        .thing(THINGS[3], StageThing::default())
        .thing(THINGS[4], ObjectiveThing::default())
        .thing(THINGS[5], IlluminationThing::default())
        .thing_arc(THINGS[6], preview)
        .merge(media)
        .build()
        .map_err(|e| Error::internal(format!("the server won't build: {e}")))?;
    Ok(Service { server })
}

impl Service {
    /// Starts the Things and serves on `listener` until `shutdown` completes, then shuts down
    /// gracefully: unfinished invocations are cancelled, and the Things stopped.
    pub async fn serve(
        self,
        listener: TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<()> {
        self.server
            .serve_with(listener, shutdown)
            .await
            .map_err(|e| Error::internal(format!("the server failed: {e}")))
    }

    /// The server itself, for teta-wot's in-process test client.
    #[cfg(test)]
    pub(crate) fn into_inner(self) -> ThingServer {
        self.server
    }
}
