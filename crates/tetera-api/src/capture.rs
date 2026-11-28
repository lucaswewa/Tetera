//! The `capture` Thing (P1.15d, D5): one frame, N at an interval, or a burst at the full rate,
//! into a dataset, as a hardware job the `jobs` Thing follows and cancels; and the last
//! capture's result.

use crate::camera::CameraThing;
use crate::error;
use crate::instrument::Instrument;
use crate::server::Roots;
use anyhow::Context;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, Weak};
use teta_wot::prelude::*;
use tetera_core::Error;
use tetera_jobs::{Job, JobKind, JobService};
use tetera_store::Store;
use tetera_workflows::{CaptureKind, CaptureRequest, CaptureResult, Rig};

/// The capture.
#[derive(Thing)]
pub struct CaptureThing {
    /// The last capture's result: its dataset, the mode switch's times there and back, and
    /// the frames a burst lost.
    #[property(readonly, default = None)]
    last: Prop<Option<CaptureResult>>,
    parts: OnceLock<Parts>,
    /// Itself, for a job to set `last` from its worker.
    me: OnceLock<Weak<CaptureThing>>,
}

/// What a capture works with, found when the Thing starts.
#[derive(Clone)]
struct Parts {
    instrument: Arc<Instrument>,
    store: Arc<Store>,
    jobs: Arc<JobService>,
    scratch: PathBuf,
    camera: Weak<CameraThing>,
}

#[thing_impl]
impl CaptureThing {
    /// Finds the instrument, the store, the job service, the scratch root and the camera.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let server = ctx.server();
        let camera = server
            .thing_ref::<CameraThing>(crate::server::CAMERA)
            .context("the camera Thing isn't on the server")?;
        let me = server
            .thing_ref::<CaptureThing>(ctx.name())
            .context("the capture Thing isn't on the server")?;
        let parts = Parts {
            instrument: server
                .service::<Instrument>()
                .context("the server has no instrument")?,
            store: server
                .service::<Store>()
                .context("the server has no store")?,
            jobs: server
                .service::<JobService>()
                .context("the server has no job service")?,
            scratch: server
                .service::<Roots>()
                .context("the server has no roots")?
                .scratch
                .clone(),
            camera: Arc::downgrade(ThingRef::thing(&camera)),
        };
        let _ = self.parts.set(parts);
        let _ = self.me.set(Arc::downgrade(ThingRef::thing(&me)));
        Ok(())
    }

    /// Takes `kind` into a new dataset named `name` in project `project`, as a hardware job
    /// after those before it: the job, which the `jobs` Thing follows and cancels. The camera
    /// is switched from its live mode to `mode` (0, full resolution, when not given) at
    /// `bit_depth` bits (12) and `exposure_ms` (the live view's), and back. A request that
    /// can't be taken, or a project that isn't there, is refused at once.
    #[action(global_lock = false)]
    async fn capture(
        &self,
        project: String,
        name: String,
        kind: CaptureKind,
        #[param(default)] mode: Option<u32>,
        #[param(default)] bit_depth: Option<u8>,
        #[param(default)] exposure_ms: Option<f64>,
    ) -> Result<Job, ActionError> {
        let request = CaptureRequest {
            project,
            name,
            kind,
            mode,
            bit_depth,
            exposure_ms,
        };
        request.check().map_err(error::action)?;
        let parts = self
            .parts
            .get()
            .cloned()
            .ok_or_else(|| ActionError::handled("the capture hasn't started"))?;
        let me = self.me.get().cloned().unwrap_or_default();
        tokio::task::spawn_blocking(move || {
            parts.store.project(&request.project)?;
            let jobs = parts.jobs.clone();
            let id = jobs.submit(JobKind::Hardware, "capture", 0, move |ctx| {
                let result = take(&parts, &request, &mut |done, message| {
                    ctx.progress(done, message);
                    ctx.check()
                })?;
                ctx.log(format!(
                    "dataset {} in {}: {} frames, {} lost; to the capture's mode in {:.0} ms, back in {:.0} ms",
                    result.dataset.id,
                    result.dataset.project,
                    result.dataset.frames,
                    result.missing,
                    result.switch_ms,
                    result.back_ms
                ));
                if let Some(me) = me.upgrade() {
                    let _ = me.last.set(Some(result.clone()));
                }
                serde_json::to_value(&result)
                    .map_err(|e| Error::internal(format!("a capture's result: {e}")))
            })?;
            jobs.get(&id)
        })
        .await
        .map_err(anyhow::Error::from)?
        .map_err(error::action)
    }
}

/// `request`, taken with what `parts` holds: the live view restarted through the camera
/// Thing's sink, so it goes on counting.
fn take(
    parts: &Parts,
    request: &CaptureRequest,
    step: &mut dyn FnMut(f64, &str) -> tetera_core::Result<()>,
) -> tetera_core::Result<CaptureResult> {
    let instrument = &parts.instrument;
    let rig = Rig {
        camera: instrument.camera.as_ref(),
        bus: &instrument.bus,
        store: &parts.store,
        scratch: &parts.scratch,
        live: parts.camera.upgrade().map(|c| c.live_sink()),
        pixel_size_um: instrument.pixel_size_um(),
        instrument: instrument.describe(),
    };
    tetera_workflows::capture(&rig, request, step)
}

#[cfg(test)]
mod tests;
