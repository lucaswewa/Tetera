//! The `autofocus` Thing (P1.16b): autofocus as a hardware job the `jobs` Thing follows and
//! cancels, with the focus curve as its result; and the last search's.

use crate::camera::CameraThing;
use crate::error;
use crate::instrument::Instrument;
use anyhow::Context;
use std::sync::{Arc, OnceLock, Weak};
use teta_wot::prelude::*;
use tetera_core::Error;
use tetera_hal::objective::ObjectiveMount;
use tetera_imaging::focus::FocusMeasure;
use tetera_jobs::{Job, JobKind, JobService};
use tetera_workflows::focus::SearchOptions;
use tetera_workflows::{AutofocusRequest, AutofocusResult, FocusRig};

/// Autofocus.
#[derive(Thing)]
pub struct AutofocusThing {
    /// The last search's result, a focus found or not: where, why not, and its curve.
    #[property(readonly, default = None)]
    last: Prop<Option<AutofocusResult>>,
    parts: OnceLock<Parts>,
    /// Itself, for a job to set `last` from its worker.
    me: OnceLock<Weak<AutofocusThing>>,
}

/// What autofocus works with, found when the Thing starts.
#[derive(Clone)]
struct Parts {
    instrument: Arc<Instrument>,
    jobs: Arc<JobService>,
    camera: Weak<CameraThing>,
}

#[thing_impl]
impl AutofocusThing {
    /// Finds the instrument, the job service and the camera.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let server = ctx.server();
        let camera = server
            .thing_ref::<CameraThing>(crate::server::CAMERA)
            .context("the camera Thing isn't on the server")?;
        let me = server
            .thing_ref::<AutofocusThing>(ctx.name())
            .context("the autofocus Thing isn't on the server")?;
        let parts = Parts {
            instrument: server
                .service::<Instrument>()
                .context("the server has no instrument")?,
            jobs: server
                .service::<JobService>()
                .context("the server has no job service")?,
            camera: Arc::downgrade(ThingRef::thing(&camera)),
        };
        let _ = self.parts.set(parts);
        let _ = self.me.set(Arc::downgrade(ThingRef::thing(&me)));
        Ok(())
    }

    /// Focuses on the sample, as a hardware job after those before it: the job, whose
    /// result is the focus and its curve. `measure` is normalised variance, and `options`
    /// the search's defaults, when not given. The stage is left at the focus or, with none
    /// found, where it started, and the job fails `not_found`. Refused at once without an
    /// objective fitted, or with options the search can't take.
    #[action(global_lock = false)]
    async fn autofocus(
        &self,
        #[param(default)] measure: Option<FocusMeasure>,
        #[param(default)] options: Option<SearchOptions>,
    ) -> Result<Job, ActionError> {
        let request = AutofocusRequest {
            measure: measure.unwrap_or_default(),
            options: options.unwrap_or_default(),
        };
        request.options.check().map_err(error::action)?;
        let parts = self
            .parts
            .get()
            .cloned()
            .ok_or_else(|| ActionError::handled("autofocus hasn't started"))?;
        if parts.instrument.objectives.fitted().is_none() {
            return Err(error::action(Error::invalid_argument(
                "autofocus needs an objective fitted, to know its depth of field",
            )));
        }
        let me = self.me.get().cloned().unwrap_or_default();
        let jobs = parts.jobs.clone();
        let id = jobs
            .submit(JobKind::Hardware, "autofocus", 0, move |ctx| {
                let instrument = &parts.instrument;
                let rig = FocusRig {
                    camera: instrument.camera.as_ref(),
                    stage: instrument.stage.as_ref(),
                    bus: &instrument.bus,
                    live: parts.camera.upgrade().map(|c| c.live_sink()),
                    objective: instrument.objectives.fitted(),
                    sensor_pixel_um: instrument.sensor_pixel_um(),
                };
                let result = tetera_workflows::autofocus(&rig, &request, &mut |done, message| {
                    ctx.progress(done, message);
                    ctx.check()
                })?;
                if let Some(me) = me.upgrade() {
                    let _ = me.last.set(Some(result.clone()));
                }
                let curve: Vec<String> = result
                    .focus
                    .curve
                    .iter()
                    .map(|p| format!("{:.2}:{:.4}", p.z, p.value))
                    .collect();
                ctx.log(format!(
                    "{} at {}, {:.2} µm deep of field, in {:.1} s; the curve, z µm: measure, {}",
                    result.measure.name(),
                    result.objective,
                    result.focus.depth_of_field,
                    result.seconds,
                    curve.join(" ")
                ));
                let z = result.focus.found()?;
                ctx.log(format!("in focus at {z:.2} µm"));
                serde_json::to_value(&result)
                    .map_err(|e| Error::internal(format!("autofocus's result: {e}")))
            })
            .map_err(error::action)?;
        jobs.get(&id).map_err(error::action)
    }
}

#[cfg(test)]
mod tests;
