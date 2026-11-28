//! The `scan` Thing (P1.17b, ADR-0022): a scan's plan previewed, and the scan run stop and go
//! as a hardware job the `jobs` Thing follows and cancels, paused and resumed meanwhile, its
//! every step an event; and the last scan's result.

// The `scan` action takes a scan request's fields, each a parameter as HTTP gives it, and
// teta-wot's macro makes a function of them all, where an attribute on the action can't reach.
#![allow(clippy::too_many_arguments)]

use crate::camera::CameraThing;
use crate::error;
use crate::instrument::Instrument;
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use teta_wot::prelude::*;
use tetera_core::Error;
use tetera_hal::objective::ObjectiveMount;
use tetera_jobs::{Job, JobKind, JobService};
use tetera_store::Store;
use tetera_workflows::scan::{
    AutofocusAt, Pattern, Plan, ScanControl, ScanRequest, ScanResult, ScanRig,
};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// How far the running scan has got.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ScanProgress {
    /// Its job.
    pub job: String,
    /// How much of it is done, 0 to 1.
    pub done: f64,
    /// What it is doing: `tile 3 of 25`, an autofocus at a tile, `paused`, `written`.
    pub message: String,
    /// Whether it is paused, or will be at its next position.
    pub paused: bool,
}

/// Scans.
#[derive(Thing)]
pub struct ScanThing {
    /// The running scan's progress; `null` when none runs.
    #[property(readonly, default = None)]
    running: Prop<Option<ScanProgress>>,
    /// The last scan's result: its dataset, plan, tiles and time.
    #[property(readonly, default = None)]
    last: Prop<Option<ScanResult>>,
    /// The running scan's every step: each tile, each autofocus's, a pause, and the end.
    #[event]
    progress: Event<ScanProgress>,
    parts: OnceLock<Parts>,
    /// Itself, for a job to set `running` and `last` from its worker.
    me: OnceLock<Weak<ScanThing>>,
    /// The running scan's job and its control.
    current: Mutex<Option<(String, Arc<ScanControl>)>>,
}

/// What a scan works with, found when the Thing starts.
#[derive(Clone)]
struct Parts {
    instrument: Arc<Instrument>,
    store: Arc<Store>,
    jobs: Arc<JobService>,
    camera: Weak<CameraThing>,
}

impl Parts {
    /// The instrument as a scan uses it: the live view restarted after through the camera
    /// Thing's sink, so it goes on counting.
    fn rig(&self) -> ScanRig<'_> {
        let instrument = &self.instrument;
        ScanRig {
            camera: instrument.camera.as_ref(),
            stage: instrument.stage.as_ref(),
            bus: &instrument.bus,
            store: &self.store,
            live: self.camera.upgrade().map(|c| c.live_sink()),
            objective: instrument.objectives.fitted(),
            sensor_pixel_um: instrument.sensor_pixel_um(),
            instrument: instrument.describe(),
        }
    }
}

impl ScanThing {
    fn parts(&self) -> Result<Parts, ActionError> {
        self.parts
            .get()
            .cloned()
            .ok_or_else(|| ActionError::handled("the scan hasn't started"))
    }

    /// The running scan's control: refused when none runs.
    fn control(&self) -> Result<Arc<ScanControl>, ActionError> {
        lock(&self.current)
            .as_ref()
            .map(|(_, control)| control.clone())
            .ok_or_else(|| error::action(Error::not_found("no scan is running")))
    }

    /// Says where job `job`'s scan has got.
    fn report(&self, job: &str, done: f64, message: &str, paused: bool) {
        let progress = ScanProgress {
            job: job.into(),
            done,
            message: message.into(),
            paused,
        };
        let _ = self.running.set(Some(progress.clone()));
        self.progress.emit(progress);
    }
}

#[thing_impl]
impl ScanThing {
    /// Finds the instrument, the store, the job service and the camera.
    #[on_start]
    async fn find_parts(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let server = ctx.server();
        let camera = server
            .thing_ref::<CameraThing>(crate::server::CAMERA)
            .context("the camera Thing isn't on the server")?;
        let me = server
            .thing_ref::<ScanThing>(ctx.name())
            .context("the scan Thing isn't on the server")?;
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
            camera: Arc::downgrade(ThingRef::thing(&camera)),
        };
        let _ = self.parts.set(parts);
        let _ = self.me.set(Arc::downgrade(ThingRef::thing(&me)));
        Ok(())
    }

    /// The plan a scan of `pattern` makes from where the stage is: its positions, its grid,
    /// the field of view, and its estimated time and what that is made of. Nothing moves. The
    /// rest as `scan` takes them. Refused without an objective fitted, or with a request a
    /// scan can't take.
    #[action(global_lock = false)]
    async fn preview(
        &self,
        pattern: Pattern,
        #[param(default)] autofocus: Option<AutofocusAt>,
        #[param(default)] settle_ms: Option<f64>,
        #[param(default)] exposure_ms: Option<f64>,
        #[param(default)] write_mb_s: Option<f64>,
    ) -> Result<Plan, ActionError> {
        let request = ScanRequest {
            project: String::new(),
            name: String::new(),
            pattern,
            autofocus: autofocus.unwrap_or_default(),
            settle_ms,
            exposure_ms,
            write_mb_s,
        };
        let parts = self.parts()?;
        tokio::task::spawn_blocking(move || tetera_workflows::scan::preview(&parts.rig(), &request))
            .await
            .map_err(anyhow::Error::from)?
            .map_err(error::action)
    }

    /// Scans `pattern` into a new dataset named `name` in project `project`, stop and go, as
    /// a hardware job after those before it: the job, which the `jobs` Thing follows and
    /// cancels, its result the dataset, the plan, each tile's positions and the time taken.
    /// A z-scan's dataset is a z-stack. `autofocus` is never when not given, `settle_ms` 50,
    /// `exposure_ms` the live view's, and `write_mb_s`, the disk's throughput the estimate
    /// assumes, 120. Refused at once without an objective fitted, with a request a scan can't
    /// take, or with a project that isn't there.
    #[action(global_lock = false)]
    async fn scan(
        &self,
        project: String,
        name: String,
        pattern: Pattern,
        #[param(default)] autofocus: Option<AutofocusAt>,
        #[param(default)] settle_ms: Option<f64>,
        #[param(default)] exposure_ms: Option<f64>,
        #[param(default)] write_mb_s: Option<f64>,
    ) -> Result<Job, ActionError> {
        let request = ScanRequest {
            project,
            name,
            pattern,
            autofocus: autofocus.unwrap_or_default(),
            settle_ms,
            exposure_ms,
            write_mb_s,
        };
        let parts = self.parts()?;
        let me = self.me.get().cloned().unwrap_or_default();
        tokio::task::spawn_blocking(move || {
            // Planned now, so a request the scan would refuse is refused at once.
            tetera_workflows::scan::preview(&parts.rig(), &request)?;
            parts.store.project(&request.project)?;
            let jobs = parts.jobs.clone();
            let id = jobs.submit(JobKind::Hardware, "scan", 0, move |ctx| {
                let me = me.upgrade();
                let job = ctx.id().to_string();
                let control = Arc::new(ScanControl::default());
                if let Some(me) = &me {
                    *lock(&me.current) = Some((job.clone(), control.clone()));
                }
                let scanned = tetera_workflows::scan::scan(
                    &parts.rig(),
                    &request,
                    &control,
                    &mut |done, message| {
                        ctx.progress(done, message);
                        if let Some(me) = &me {
                            me.report(&job, done, message, control.is_paused());
                        }
                        ctx.check()
                    },
                );
                if let Some(me) = &me {
                    *lock(&me.current) = None;
                    let _ = me.running.set(None);
                }
                let result = scanned?;
                ctx.log(format!(
                    "dataset {} in {}: {} tiles in {:.1} s, estimated at {:.1} s",
                    result.dataset.id,
                    result.dataset.project,
                    result.tiles.len(),
                    result.seconds,
                    result.plan.estimate_s
                ));
                if let Some(me) = &me {
                    let _ = me.last.set(Some(result.clone()));
                }
                serde_json::to_value(&result)
                    .map_err(|e| Error::internal(format!("a scan's result: {e}")))
            })?;
            jobs.get(&id)
        })
        .await
        .map_err(anyhow::Error::from)?
        .map_err(error::action)
    }

    /// Pauses the running scan at its next position, the camera kept in its capture mode; a
    /// `progress` event says when it holds. Refused when no scan runs.
    #[action(global_lock = false)]
    async fn pause(&self) -> Result<(), ActionError> {
        self.control()?.pause();
        Ok(())
    }

    /// Resumes the running scan. Refused when none runs.
    #[action(global_lock = false)]
    async fn resume(&self) -> Result<(), ActionError> {
        self.control()?.resume();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
