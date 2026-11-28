//! The `illumination` Thing: the transmitted light, on or off, its intensity and its colour
//! temperature.

use crate::error;
use crate::instrument::Instrument;
use anyhow::Context;
use std::sync::{Arc, OnceLock};
use teta_wot::prelude::*;
use tetera_core::frame::meta::LightState;
use tetera_hal::illumination::{Illumination, IlluminationCapabilities};
use tetera_sim::light::CHANNEL;

/// The light.
#[derive(Thing)]
pub struct IlluminationThing {
    light: OnceLock<Arc<dyn Illumination>>,
}

impl IlluminationThing {
    fn light(&self) -> Result<&Arc<dyn Illumination>, PropertyError> {
        self.light
            .get()
            .ok_or_else(|| PropertyError::failed(anyhow::anyhow!("not started")))
    }

    fn now(&self) -> Result<LightState, PropertyError> {
        self.light()?
            .state(CHANNEL)
            .map_err(|e| error::property(e, None))
    }

    /// The light's state, changed by `change`, set: a 422 for one it doesn't take.
    fn change(
        &self,
        value: serde_json::Value,
        change: impl FnOnce(&mut LightState),
    ) -> Result<(), PropertyError> {
        let mut state = self.now()?;
        change(&mut state);
        self.light()?
            .set(CHANNEL, state)
            .map_err(|e| error::property(e, Some(&value)))
    }
}

#[thing_impl]
impl IlluminationThing {
    /// Finds the light.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let instrument = ctx
            .server()
            .service::<Instrument>()
            .context("the server has no instrument")?;
        let light: Arc<dyn Illumination> = instrument.light.clone();
        let _ = self.light.set(light);
        Ok(())
    }

    /// What the light can do: its channels, whether each dims, and its colour temperatures.
    #[property]
    async fn capabilities(&self) -> Result<IlluminationCapabilities, PropertyError> {
        Ok(self.light()?.capabilities())
    }

    /// Whether the light is on.
    #[property]
    async fn on(&self) -> Result<bool, PropertyError> {
        Ok(self.now()?.on)
    }

    /// Turns the light on or off.
    #[setter(on)]
    async fn set_on(&self, value: bool) -> Result<(), PropertyError> {
        self.change(value.into(), |s| s.on = value)
    }

    /// Its intensity, from 0 to 1.
    #[property(ge = 0, le = 1)]
    async fn intensity(&self) -> Result<f64, PropertyError> {
        Ok(self.now()?.intensity)
    }

    /// Sets its intensity.
    #[setter(intensity)]
    async fn set_intensity(&self, value: f64) -> Result<(), PropertyError> {
        self.change(value.into(), |s| s.intensity = value)
    }

    /// Its colour temperature, K, or `null` for the light's own.
    #[property]
    async fn colour_temperature(&self) -> Result<Option<f64>, PropertyError> {
        Ok(self.now()?.colour_temperature_k)
    }

    /// Sets its colour temperature.
    #[setter(colour_temperature)]
    async fn set_colour_temperature(&self, value: Option<f64>) -> Result<(), PropertyError> {
        let json = serde_json::to_value(value).unwrap_or_default();
        self.change(json, |s| s.colour_temperature_k = value)
    }
}
