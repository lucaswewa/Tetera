//! The `camera` Thing: its capabilities and configuration, each setting a property of its
//! own; the white balance the preview applies; and a stream into the frame bus, to watch:
//! whether it runs, the frames it has delivered and dropped, and the last frame's metadata.

use crate::error;
use crate::instrument::Instrument;
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock, Weak};
use teta_wot::prelude::*;
use tetera_acquire::FrameBus;
use tetera_core::frame::format::Binning;
use tetera_core::frame::meta::FrameMeta;
use tetera_core::{Frame, PixelFormat, Rect};
use tetera_hal::camera::{Camera, CameraCapabilities, CameraConfig, FrameSink};

/// Red's, green's and blue's gains, as the preview balances a raw frame (P1.14a).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WhiteBalance {
    /// Red's gain.
    pub red: f64,
    /// Green's gain.
    pub green: f64,
    /// Blue's gain.
    pub blue: f64,
}

impl Default for WhiteBalance {
    fn default() -> Self {
        Self {
            red: 1.0,
            green: 1.0,
            blue: 1.0,
        }
    }
}

/// What a stream has done since it started.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FrameCount {
    /// Frames delivered.
    pub delivered: u64,
    /// Frames dropped before them.
    pub dropped: u64,
}

/// The camera.
#[derive(Thing)]
pub struct CameraThing {
    /// The white balance the preview applies: red's, green's and blue's gains.
    #[setting(default = WhiteBalance::default())]
    white_balance: Prop<WhiteBalance>,
    /// Whether the camera is streaming.
    #[property(readonly, default = false)]
    streaming: Prop<bool>,
    /// The frames the stream has delivered, and dropped, since it started.
    #[property(readonly, default = FrameCount::default())]
    frames: Prop<FrameCount>,
    /// The last frame's metadata: its number, times, settings, and where the stage was.
    #[property(readonly, default = None)]
    last_frame: Prop<Option<FrameMeta>>,
    camera: OnceLock<Arc<dyn Camera>>,
    /// The frame bus its stream goes into.
    bus: OnceLock<Arc<FrameBus>>,
    /// Itself, for the stream's sink to update its properties from the camera's thread.
    me: OnceLock<Weak<CameraThing>>,
}

impl CameraThing {
    fn camera(&self) -> Result<&Arc<dyn Camera>, PropertyError> {
        self.camera
            .get()
            .ok_or_else(|| PropertyError::failed(anyhow::anyhow!("not started")))
    }

    /// The white balance's gains, red's, green's and blue's, for the preview.
    pub(crate) fn gains(&self) -> [f64; 3] {
        let w = self.white_balance.get();
        [w.red, w.green, w.blue]
    }

    /// The configuration, changed by `change`, applied: a 422 for one it doesn't take.
    fn reconfigure(
        &self,
        value: serde_json::Value,
        change: impl FnOnce(&mut CameraConfig),
    ) -> Result<(), PropertyError> {
        let camera = self.camera()?;
        let mut config = camera.config();
        change(&mut config);
        camera
            .configure(&config)
            .map_err(|e| error::property(e, Some(&value)))
    }
}

#[thing_impl]
impl CameraThing {
    /// Finds the camera.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let instrument = ctx
            .server()
            .service::<Instrument>()
            .context("the server has no instrument")?;
        let camera: Arc<dyn Camera> = instrument.camera.clone();
        let _ = self.camera.set(camera);
        let _ = self.bus.set(instrument.bus.clone());
        let me = ctx
            .server()
            .thing_ref::<CameraThing>(ctx.name())
            .context("the camera Thing isn't on the server")?;
        let _ = self.me.set(Arc::downgrade(ThingRef::thing(&me)));
        Ok(())
    }

    /// Stops the stream, if it runs.
    #[on_stop]
    async fn stop_camera(&self) {
        if let Some(camera) = self.camera.get().cloned() {
            let _ = tokio::task::spawn_blocking(move || camera.stop()).await;
        }
    }

    /// What the camera can do: its modes and their bit depths and rates, exposures, gains,
    /// binnings, ROIs and triggers.
    #[property]
    async fn capabilities(&self) -> Result<CameraCapabilities, PropertyError> {
        Ok(self.camera()?.capabilities())
    }

    /// How it is set, all at once.
    #[property]
    async fn config(&self) -> Result<CameraConfig, PropertyError> {
        Ok(self.camera()?.config())
    }

    /// Sets it all at once: refused while it streams, or if it doesn't take it.
    #[setter(config)]
    async fn set_config(&self, value: CameraConfig) -> Result<(), PropertyError> {
        let json = serde_json::to_value(&value).unwrap_or_default();
        self.reconfigure(json, |c| *c = value)
    }

    /// Its readout mode, by number.
    #[property]
    async fn mode(&self) -> Result<u32, PropertyError> {
        Ok(self.camera()?.config().mode)
    }

    /// Sets its mode.
    #[setter(mode)]
    async fn set_mode(&self, value: u32) -> Result<(), PropertyError> {
        self.reconfigure(value.into(), |c| c.mode = value)
    }

    /// Its exposure, ms.
    #[property]
    async fn exposure(&self) -> Result<f64, PropertyError> {
        Ok(self.camera()?.config().exposure.as_secs_f64() * 1e3)
    }

    /// Sets its exposure, ms.
    #[setter(exposure)]
    async fn set_exposure(&self, value: f64) -> Result<(), PropertyError> {
        let exposure = std::time::Duration::try_from_secs_f64(value / 1e3).map_err(|_| {
            error::property(
                tetera_core::Error::invalid_argument(format!("an exposure of {value} ms")),
                Some(&value.into()),
            )
        })?;
        self.reconfigure(value.into(), |c| c.exposure = exposure)
    }

    /// Its gain.
    #[property]
    async fn gain(&self) -> Result<f64, PropertyError> {
        Ok(self.camera()?.config().gain)
    }

    /// Sets its gain.
    #[setter(gain)]
    async fn set_gain(&self, value: f64) -> Result<(), PropertyError> {
        self.reconfigure(value.into(), |c| c.gain = value)
    }

    /// The bits each sample carries: 12 or 16 in 16-bit words, or 8.
    #[property]
    async fn bit_depth(&self) -> Result<u8, PropertyError> {
        Ok(self.camera()?.config().bit_depth)
    }

    /// Sets the bits each sample carries.
    #[setter(bit_depth)]
    async fn set_bit_depth(&self, value: u8) -> Result<(), PropertyError> {
        self.reconfigure(value.into(), |c| c.bit_depth = value)
    }

    /// The frames' pixel format, as the configuration makes it.
    #[property]
    async fn pixel_format(&self) -> Result<PixelFormat, PropertyError> {
        let camera = self.camera()?;
        camera
            .capabilities()
            .layout(&camera.config())
            .map(|l| l.format)
            .map_err(|e| error::property(e, None))
    }

    /// Its binning, on top of the mode's.
    #[property]
    async fn binning(&self) -> Result<Binning, PropertyError> {
        Ok(self.camera()?.config().binning)
    }

    /// Sets its binning.
    #[setter(binning)]
    async fn set_binning(&self, value: Binning) -> Result<(), PropertyError> {
        let json = serde_json::to_value(value).unwrap_or_default();
        self.reconfigure(json, |c| c.binning = value)
    }

    /// The region read out, or `null` for the whole sensor.
    #[property]
    async fn roi(&self) -> Result<Option<Rect>, PropertyError> {
        Ok(self.camera()?.config().roi)
    }

    /// Sets the region read out.
    #[setter(roi)]
    async fn set_roi(&self, value: Option<Rect>) -> Result<(), PropertyError> {
        let json = serde_json::to_value(value).unwrap_or_default();
        self.reconfigure(json, |c| c.roi = value)
    }

    /// Starts the stream into the frame bus: `frames` and `last_frame` follow it. Refused if
    /// it runs.
    #[action(global_lock = false)]
    async fn start_streaming(&self) -> Result<(), ActionError> {
        let camera = self
            .camera
            .get()
            .cloned()
            .ok_or_else(|| ActionError::handled("the camera hasn't started"))?;
        let sink = Arc::new(Counter {
            thing: self.me.get().cloned().unwrap_or_default(),
            bus: self.bus.get().cloned().unwrap_or_else(FrameBus::new),
        });
        self.frames.set(FrameCount::default())?;
        camera.start(sink).map_err(error::action)?;
        self.streaming.set(true)?;
        Ok(())
    }

    /// Stops the stream: no frame comes after it ends.
    #[action(global_lock = false)]
    async fn stop_streaming(&self) -> Result<(), ActionError> {
        let camera = self
            .camera
            .get()
            .cloned()
            .ok_or_else(|| ActionError::handled("the camera hasn't started"))?;
        tokio::task::spawn_blocking(move || camera.stop())
            .await
            .map_err(anyhow::Error::from)?
            .map_err(error::action)?;
        self.streaming.set(false)?;
        Ok(())
    }
}

#[cfg(test)]
impl CameraThing {
    /// A sink for this Thing, as `start_streaming` makes.
    pub(crate) fn sink_for_tests(self: &Arc<Self>) -> Arc<dyn FrameSink> {
        Arc::new(Counter {
            thing: Arc::downgrade(self),
            bus: self.bus.get().cloned().unwrap_or_else(FrameBus::new),
        })
    }

    /// The frames counted.
    pub(crate) fn frame_count(&self) -> FrameCount {
        self.frames.get()
    }

    /// The last frame's number.
    pub(crate) fn last_seq(&self) -> Option<u64> {
        self.last_frame.get().map(|m| m.seq)
    }
}

/// Where a stream's frames go: counted, their metadata kept, and into the frame bus, which
/// lets them go once its readers have.
struct Counter {
    thing: Weak<CameraThing>,
    bus: Arc<FrameBus>,
}

impl FrameSink for Counter {
    fn deliver(&self, frame: Frame) {
        let Some(thing) = self.thing.upgrade() else {
            return;
        };
        let meta = frame.meta().clone();
        self.bus.publish(frame);
        let _ = thing.frames.update(|count| {
            count.delivered += 1;
            count.dropped += meta.dropped_before;
        });
        let _ = thing.last_frame.set(Some(meta));
    }
}
