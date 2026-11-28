//! The `simulation` Thing: what only a simulated instrument has. The sample on the stage,
//! loaded from a seed or taken away, and the devices' imperfections, shown on demand.

use crate::error;
use crate::instrument::{Instrument, Sample};
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use teta_wot::prelude::*;
use tetera_sim::VirtualStageFault;

/// An imperfection the simulation shows when asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Imperfection {
    /// The stage's next move stalls: it says it is moving, and goes nowhere.
    StageStall,
    /// An alarm stops the stage; homing clears it.
    StageAlarm,
    /// A hard limit stops the stage; homing clears it.
    StageLimit,
}

/// The simulation: the sample on the stage, and imperfections on demand.
#[derive(Thing)]
pub struct Simulation {
    /// The sample on the stage: its seed and section, or `null` with no slide.
    #[property(readonly, default = None)]
    sample: Prop<Option<Sample>>,
    instrument: OnceLock<Arc<Instrument>>,
}

impl Simulation {
    fn instrument(&self) -> Result<&Arc<Instrument>, ActionError> {
        self.instrument
            .get()
            .ok_or_else(|| ActionError::handled("the simulation hasn't started"))
    }
}

#[thing_impl]
impl Simulation {
    /// Finds the instrument, and shows what is on its stage.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let instrument = ctx
            .server()
            .service::<Instrument>()
            .context("the server has no instrument")?;
        self.sample.set(instrument.sample())?;
        let _ = self.instrument.set(instrument);
        Ok(())
    }

    /// Puts an H&E sample on the stage, grown from `seed`: a section `thickness` µm thick,
    /// 3 to 5, folded in places if `folds`. The camera sees it from its next frame.
    #[action(global_lock = false)]
    async fn load(
        &self,
        seed: u64,
        #[param(default = 4.0)] thickness: f32,
        #[param(default = false)] folds: bool,
    ) -> Result<Sample, ActionError> {
        let sample = Sample {
            seed,
            thickness,
            folds,
        };
        let instrument = self.instrument()?.clone();
        tokio::task::spawn_blocking(move || instrument.load(sample))
            .await
            .map_err(anyhow::Error::from)?
            .map_err(error::action)?;
        self.sample.set(Some(sample))?;
        Ok(sample)
    }

    /// Takes the slide off the stage: the camera sees the light alone.
    #[action(global_lock = false)]
    async fn unload(&self) -> Result<(), ActionError> {
        self.instrument()?.unload().map_err(error::action)?;
        self.sample.set(None)?;
        Ok(())
    }

    /// Shows `imperfection` now, as the device would.
    #[action(global_lock = false)]
    async fn inject(&self, imperfection: Imperfection) -> Result<(), ActionError> {
        let fault = match imperfection {
            Imperfection::StageStall => VirtualStageFault::Stall,
            Imperfection::StageAlarm => VirtualStageFault::Alarm,
            Imperfection::StageLimit => VirtualStageFault::LimitHit,
        };
        self.instrument()?.stage.inject(fault);
        Ok(())
    }
}
