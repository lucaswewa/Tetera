//! The `stage` Thing: where the stage is and what it is doing, kept current while it moves,
//! so clients can watch it; moves that end when the stage is still, and stop it if cancelled;
//! jogs, a halt, and homing.

use crate::error;
use crate::instrument::Instrument;
use anyhow::Context;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;
use teta_wot::prelude::*;
use tetera_core::{Error, StagePos, Um};
use tetera_hal::stage::{Axis, Stage, StageCapabilities, StageReading, StageState};
use tokio::task::JoinHandle;

/// How often the stage is read to keep its properties current.
pub const WATCH: Duration = Duration::from_millis(10);

/// The longest a move, a jog's end or homing may take before its action gives up.
pub const LONGEST: Duration = Duration::from_secs(300);

/// The motorised stage.
#[derive(Thing)]
pub struct StageThing {
    /// Where the stage is, µm.
    #[property(readonly, default = StagePos::default())]
    position: Prop<StagePos>,
    /// What it is doing: idle, moving, jogging, homing, or in alarm.
    #[property(readonly, default = StageState::Idle)]
    state: Prop<StageState>,
    /// Whether it is moving: a move, a jog or homing.
    #[property(readonly, default = false)]
    moving: Prop<bool>,
    /// The speed of a move that names none, µm/s.
    #[setting(default = 5_000.0, gt = 0, le = 100_000)]
    speed: Prop<f64>,
    stage: OnceLock<Arc<dyn Stage>>,
    watcher: Mutex<Option<JoinHandle<()>>>,
}

impl StageThing {
    fn stage(&self) -> Result<Arc<dyn Stage>, ActionError> {
        self.stage
            .get()
            .cloned()
            .ok_or_else(|| ActionError::handled("the stage hasn't started"))
    }

    /// The properties set from `reading`, if they changed.
    fn show(&self, reading: &StageReading) {
        if self.position.get() != reading.position {
            let _ = self.position.set(reading.position);
        }
        if self.state.get() != reading.state {
            let _ = self.state.set(reading.state);
            let _ = self.moving.set(!reading.state.is_still());
        }
    }

    /// Runs `start` on the stage, then waits until it is still: its reading then. A
    /// cancellation stops the stage.
    async fn until_still(
        &self,
        ctx: &ActionCtx,
        start: impl FnOnce(&dyn Stage) -> tetera_core::Result<()> + Send + 'static,
    ) -> Result<StageReading, ActionError> {
        let stage = self.stage()?;
        let moving = stage.clone();
        let mut done = tokio::task::spawn_blocking(move || {
            start(&*moving)?;
            moving.wait(LONGEST)
        });
        let reading = tokio::select! {
            finished = &mut done => finished.map_err(anyhow::Error::from)?,
            _ = ctx.cancelled() => {
                let stopping = stage.clone();
                tokio::task::spawn_blocking(move || stopping.stop())
                    .await
                    .map_err(anyhow::Error::from)?
                    .map_err(error::action)?;
                let _ = done.await;
                return Err(error::action(Error::cancelled(
                    "the move was cancelled, and the stage stopped",
                )));
            }
        }
        .map_err(error::action)?;
        self.show(&reading);
        Ok(reading)
    }
}

#[thing_impl]
impl StageThing {
    /// Finds the stage, shows where it is, and keeps its properties current.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let server = ctx.server();
        let instrument = server
            .service::<Instrument>()
            .context("the server has no instrument")?;
        let stage: Arc<dyn Stage> = instrument.stage.clone();
        self.show(&stage.reading().map_err(anyhow::Error::new)?);
        let _ = self.stage.set(stage.clone());
        let me = server
            .thing_ref::<StageThing>(ctx.name())
            .context("the stage Thing isn't on the server")?;
        let me = Arc::clone(ThingRef::thing(&me));
        let watcher = tokio::spawn(async move {
            let mut tick = tokio::time::interval(WATCH);
            loop {
                tick.tick().await;
                if let Ok(reading) = stage.reading() {
                    me.show(&reading);
                }
            }
        });
        *self.watcher.lock().unwrap_or_else(PoisonError::into_inner) = Some(watcher);
        Ok(())
    }

    /// Stops keeping its properties current.
    #[on_stop]
    async fn stop_watching(&self) {
        if let Some(watcher) = self
            .watcher
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            watcher.abort();
        }
    }

    /// What the stage can do: its axes, their travel and fastest speeds, its resolution.
    #[property]
    async fn capabilities(&self) -> Result<StageCapabilities, PropertyError> {
        self.stage
            .get()
            .map(|s| s.capabilities())
            .ok_or_else(|| PropertyError::failed(anyhow::anyhow!("not started")))
    }

    /// Moves to (`x`, `y`, `z`) µm at `speed` µm/s, or at the stage's `speed`, and ends when
    /// the stage is still there: its reading then. Cancelling it stops the stage.
    #[action(global_lock = false)]
    async fn move_to(
        &self,
        ctx: ActionCtx,
        x: f64,
        y: f64,
        z: f64,
        #[param(default)] speed: Option<f64>,
    ) -> Result<StageReading, ActionError> {
        let speed = Some(speed.unwrap_or_else(|| self.speed.get()));
        let target = StagePos::new(Um(x), Um(y), Um(z));
        self.until_still(&ctx, move |s| s.move_to(target, speed))
            .await
    }

    /// Moves by (`dx`, `dy`, `dz`) µm from where the stage is, as `move_to` does.
    #[action(global_lock = false)]
    async fn move_by(
        &self,
        ctx: ActionCtx,
        #[param(default = 0.0)] dx: f64,
        #[param(default = 0.0)] dy: f64,
        #[param(default = 0.0)] dz: f64,
        #[param(default)] speed: Option<f64>,
    ) -> Result<StageReading, ActionError> {
        let speed = Some(speed.unwrap_or_else(|| self.speed.get()));
        let delta = StagePos::new(Um(dx), Um(dy), Um(dz));
        self.until_still(&ctx, move |s| s.move_by(delta, speed))
            .await
    }

    /// Starts moving along `axis` at `speed` µm/s, below zero to go back, until `stop_jog`,
    /// `stop`, or the end of the travel. It ends once the jog has started.
    #[action(global_lock = false)]
    async fn jog(&self, axis: Axis, speed: f64) -> Result<(), ActionError> {
        let stage = self.stage()?;
        tokio::task::spawn_blocking(move || stage.jog(axis, speed))
            .await
            .map_err(anyhow::Error::from)?
            .map_err(error::action)
    }

    /// Ends a jog, and ends when the stage is still: its reading then.
    #[action(global_lock = false)]
    async fn stop_jog(&self, ctx: ActionCtx) -> Result<StageReading, ActionError> {
        self.until_still(&ctx, |s| s.stop_jog()).await
    }

    /// Stops whatever the stage is doing: its reading once it is still. (`stop` is the
    /// Thing's own, for the server to stop it.)
    #[action(global_lock = false)]
    async fn halt(&self, ctx: ActionCtx) -> Result<StageReading, ActionError> {
        self.until_still(&ctx, |s| s.stop()).await
    }

    /// Homes the stage, which clears an alarm, and ends when it is home: its reading then.
    #[action(global_lock = false)]
    async fn home(&self, ctx: ActionCtx) -> Result<StageReading, ActionError> {
        self.until_still(&ctx, |s| s.home()).await
    }
}
