//! The server: the instrument's Things on teta-wot, with its OpenAPI document and its
//! interactive docs, and Tetera's own routes (`media`), served until a shutdown signal, then
//! shut down gracefully. Its services: the instrument, the roots, the job service, and the
//! dataset store.

use crate::camera::CameraThing;
use crate::capture::CaptureThing;
use crate::datasets::DatasetsThing;
use crate::illumination::IlluminationThing;
use crate::instrument::Instrument;
use crate::jobs::{JobsOptions, JobsThing};
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
use tetera_jobs::JobService;
use tetera_store::Store;
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
    jobs: Arc<JobService>,
    store: Arc<Store>,
}

impl std::fmt::Debug for Service {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Service").finish_non_exhaustive()
    }
}

/// The camera Thing's name.
pub const CAMERA: &str = "camera";

/// The names the Things are served under.
pub const THINGS: [&str; 10] = [
    "system",
    "simulation",
    CAMERA,
    "stage",
    "objective",
    "illumination",
    "preview",
    "jobs",
    "capture",
    "datasets",
];

/// The server for `instrument`, its settings under `roots`, its jobs as `JobsOptions`' defaults.
pub fn build(instrument: Arc<Instrument>, roots: Roots) -> Result<Service> {
    build_with(instrument, roots, &JobsOptions::default())
}

/// The jobs' history and the datasets' index, in the data root (ADR-0021).
pub const DATABASE: &str = tetera_store::store::DATABASE;

/// The server for `instrument`, its settings under `roots`, its job service as `jobs` says:
/// its history in the data root's `tetera.db`, its compute jobs sharing the GPU's budgets;
/// its datasets in the data root, those a crash left half-written removed.
pub fn build_with(
    instrument: Arc<Instrument>,
    roots: Roots,
    jobs: &JobsOptions,
) -> Result<Service> {
    let budget = instrument.budgets.as_ref().and_then(|b| {
        b.reserve("compute jobs", jobs.vram_mib << 20)
            .inspect_err(|e| tracing::warn!("compute jobs get no VRAM budget: {e}"))
            .ok()
    });
    let service = Arc::new(JobService::open(
        &roots.data.join(DATABASE),
        jobs.config(),
        budget,
    )?);
    let store = Arc::new(Store::open(&roots.data)?);
    let preview = Arc::new(PreviewThing::default());
    let media = crate::media::routes(instrument.clone(), preview.clone(), store.clone());
    let server = ThingServer::builder()
        .api_info("Tetera", env!("CARGO_PKG_VERSION"))
        .settings_folder(roots.settings())
        .shutdown_grace(SHUTDOWN_GRACE)
        .service(instrument)
        .service(Arc::new(roots))
        .service(service.clone())
        .service(store.clone())
        .thing(THINGS[0], System::default())
        .thing(THINGS[1], Simulation::default())
        .thing(THINGS[2], CameraThing::default())
        .thing(THINGS[3], StageThing::default())
        .thing(THINGS[4], ObjectiveThing::default())
        .thing(THINGS[5], IlluminationThing::default())
        .thing_arc(THINGS[6], preview)
        .thing(THINGS[7], JobsThing::default())
        .thing(THINGS[8], CaptureThing::default())
        .thing(THINGS[9], DatasetsThing::default())
        .merge(media)
        .build()
        .map_err(|e| Error::internal(format!("the server won't build: {e}")))?;
    Ok(Service {
        server,
        jobs: service,
        store,
    })
}

impl Service {
    /// The job service, for the work that submits jobs.
    pub fn jobs(&self) -> &Arc<JobService> {
        &self.jobs
    }

    /// The dataset store.
    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

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
