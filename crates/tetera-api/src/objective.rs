//! The `objective` Thing: the objectives the instrument knows, and which is fitted. They are
//! swapped by hand: the user says which is fitted, and the camera sees through it.

use crate::error;
use crate::instrument::Instrument;
use anyhow::Context;
use std::sync::{Arc, OnceLock};
use teta_wot::prelude::*;
use tetera_hal::objective::{Objective, ObjectiveMount};

/// The objective mount.
#[derive(Thing)]
pub struct ObjectiveThing {
    mount: OnceLock<Arc<dyn ObjectiveMount>>,
}

impl ObjectiveThing {
    fn mount(&self) -> Result<&Arc<dyn ObjectiveMount>, PropertyError> {
        self.mount
            .get()
            .ok_or_else(|| PropertyError::failed(anyhow::anyhow!("not started")))
    }
}

#[thing_impl]
impl ObjectiveThing {
    /// Finds the objective mount.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let instrument = ctx
            .server()
            .service::<Instrument>()
            .context("the server has no instrument")?;
        let mount: Arc<dyn ObjectiveMount> = instrument.objectives.clone();
        let _ = self.mount.set(mount);
        Ok(())
    }

    /// The objectives it can take: name, model, magnification, NA, working distance.
    #[property]
    async fn catalogue(&self) -> Result<Vec<Objective>, PropertyError> {
        Ok(self.mount()?.catalogue().objectives)
    }

    /// The objective fitted, by name, or `null` for none.
    #[property]
    async fn fitted(&self) -> Result<Option<String>, PropertyError> {
        Ok(self.mount()?.fitted().map(|o| o.name))
    }

    /// Says which objective is fitted, by its name in the catalogue. (teta-wot takes a
    /// `null` body as a missing one, so `remove` says that none is.)
    #[setter(fitted)]
    async fn set_fitted(&self, value: Option<String>) -> Result<(), PropertyError> {
        let json = serde_json::to_value(&value).unwrap_or_default();
        self.mount()?
            .fit(value.as_deref())
            .map_err(|e| error::property(e, Some(&json)))
    }

    /// The objective fitted, in full, or `null` for none.
    #[property]
    async fn objective(&self) -> Result<Option<Objective>, PropertyError> {
        Ok(self.mount()?.fitted())
    }

    /// Says that no objective is fitted: the camera sees nothing but its own noise.
    #[action(global_lock = false)]
    async fn remove(&self) -> Result<(), ActionError> {
        self.mount
            .get()
            .ok_or_else(|| ActionError::handled("the objective mount hasn't started"))?
            .fit(None)
            .map_err(error::action)
    }
}
