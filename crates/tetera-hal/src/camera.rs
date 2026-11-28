//! The camera: what one can do, how it is set, and the trait every camera implements, a real
//! driver, the virtual camera, a fake, or a bridge to another process (ADR-0013).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;
use tetera_core::frame::format::{BayerPattern, Binning};
use tetera_core::{Error, ErrorKind, Frame, FrameLayout, PixelFormat, Rect, Result};

/// One of a camera's readout modes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CameraMode {
    /// Its number, as the driver knows it.
    pub id: u32,
    /// Its name, for people.
    pub name: String,
    /// The sensor's width read out, in pixels.
    pub width: u32,
    /// Its height.
    pub height: u32,
    /// The bit depths it reads out at.
    pub bit_depths: Vec<u8>,
    /// Its fastest frame rate, whole frame and shortest exposure, in 8 bits.
    pub max_fps: f64,
    /// The same in 16-bit words, 12 or 16 bits a sample: as fast, or slower where the link
    /// carries twice the bytes.
    pub max_fps_16: f64,
}

impl CameraMode {
    /// Its fastest frame rate at `bit_depth` bits: 8, or in 16-bit words above.
    pub fn fps(&self, bit_depth: u8) -> f64 {
        if bit_depth > 8 {
            self.max_fps_16
        } else {
            self.max_fps
        }
    }
}

/// What starts each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    /// The camera's own clock: frame after frame.
    FreeRun,
    /// A software trigger: one frame for each call to `Camera::trigger`.
    Software,
    /// An edge on the camera's trigger input.
    External,
}

/// A range of values a setting takes, both ends included.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Span {
    /// The least.
    pub min: f64,
    /// The most.
    pub max: f64,
}

impl Span {
    /// Whether `v` is finite and within the span.
    pub fn holds(&self, v: f64) -> bool {
        v.is_finite() && self.min <= v && v <= self.max
    }
}

/// What a camera can do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CameraCapabilities {
    /// Its maker and model.
    pub model: String,
    /// Its sensor's Bayer pattern.
    pub pattern: BayerPattern,
    /// Its modes; the first is the default.
    pub modes: Vec<CameraMode>,
    /// The binnings it offers; no binning is always one.
    pub binnings: Vec<Binning>,
    /// The steps an ROI's corner and size come in, across and down, if it takes ROIs.
    pub roi_step: Option<(u32, u32)>,
    /// The exposures it takes, in ms.
    pub exposure_ms: Span,
    /// The gains it takes.
    pub gain: Span,
    /// The triggers it takes; free-running is always one.
    pub triggers: Vec<Trigger>,
}

/// How a camera is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CameraConfig {
    /// The mode, by its number.
    pub mode: u32,
    /// The bit depth, one the mode offers.
    pub bit_depth: u8,
    /// The binning.
    pub binning: Binning,
    /// The region of the sensor read out, before binning; the whole sensor when `None`.
    pub roi: Option<Rect>,
    /// The exposure, in ms.
    #[serde(with = "tetera_core::units::ms")]
    #[schemars(with = "f64")]
    pub exposure: Duration,
    /// The gain.
    pub gain: f64,
    /// What starts each frame.
    pub trigger: Trigger,
}

impl CameraCapabilities {
    /// The configuration a camera starts in: its first mode at its deepest bit depth, the
    /// whole sensor unbinned, the shortest exposure it takes or 10 ms, the least gain,
    /// free-running.
    pub fn default_config(&self) -> CameraConfig {
        let mode = &self.modes[0];
        CameraConfig {
            mode: mode.id,
            bit_depth: mode.bit_depths.iter().copied().max().unwrap_or(8),
            binning: Binning::NONE,
            roi: None,
            exposure: Duration::from_secs_f64(
                self.exposure_ms.min.max(10.0).min(self.exposure_ms.max) / 1e3,
            ),
            gain: self.gain.min,
            trigger: Trigger::FreeRun,
        }
    }

    /// The mode numbered `id`.
    pub fn mode(&self, id: u32) -> Result<&CameraMode> {
        self.modes.iter().find(|m| m.id == id).ok_or_else(|| {
            let ids: Vec<String> = self.modes.iter().map(|m| m.id.to_string()).collect();
            Error::invalid_argument(format!(
                "the camera has no mode {id}; its modes are {}",
                ids.join(", ")
            ))
        })
    }

    /// Whether `config` is one this camera takes, or the first thing wrong with it.
    pub fn check(&self, config: &CameraConfig) -> Result<()> {
        let bad = |why: String| Err(Error::invalid_argument(why));
        let mode = self.mode(config.mode)?;
        if !mode.bit_depths.contains(&config.bit_depth) {
            return bad(format!(
                "mode {} reads out at {:?} bits, not {}",
                mode.id, mode.bit_depths, config.bit_depth
            ));
        }
        if config.binning != Binning::NONE && !self.binnings.contains(&config.binning) {
            return bad(format!(
                "the camera doesn't bin {} × {}",
                config.binning.x, config.binning.y
            ));
        }
        if let Some(roi) = config.roi {
            let Some((sx, sy)) = self.roi_step else {
                return bad("the camera reads out its whole sensor; it takes no ROI".to_string());
            };
            let whole = Rect::new(0, 0, mode.width, mode.height);
            if roi.is_empty() || !whole.contains_rect(&roi) {
                return bad(format!(
                    "the ROI {roi:?} isn't inside the {} × {} sensor",
                    mode.width, mode.height
                ));
            }
            if roi.x % sx != 0 || roi.width % sx != 0 || roi.y % sy != 0 || roi.height % sy != 0 {
                return bad(format!(
                    "an ROI's corner and size come in steps of {sx} across and {sy} down; {roi:?} doesn't"
                ));
            }
        }
        let exposure_ms = config.exposure.as_secs_f64() * 1e3;
        if !self.exposure_ms.holds(exposure_ms) {
            return bad(format!(
                "the camera exposes {} to {} ms, not {exposure_ms}",
                self.exposure_ms.min, self.exposure_ms.max
            ));
        }
        if !self.gain.holds(config.gain) {
            return bad(format!(
                "the camera's gain is {} to {}, not {}",
                self.gain.min, self.gain.max, config.gain
            ));
        }
        if config.trigger != Trigger::FreeRun && !self.triggers.contains(&config.trigger) {
            return bad(format!("the camera takes no {:?} trigger", config.trigger));
        }
        Ok(())
    }

    /// The layout of the frames `config` gives: the mode's sensor, cut to the ROI, binned.
    pub fn layout(&self, config: &CameraConfig) -> Result<FrameLayout> {
        self.check(config)?;
        let mode = self.mode(config.mode)?;
        let format = PixelFormat::bayer16(self.pattern, config.bit_depth)?;
        let whole = FrameLayout::new(format, mode.width, mode.height)?;
        let cut = match config.roi {
            Some(roi) => whole.cropped(roi)?,
            None => whole,
        };
        cut.binned(config.binning)
    }
}

/// What a camera reports as it goes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum CameraEvent {
    /// Set to a new configuration.
    Configured {
        /// The configuration.
        config: CameraConfig,
    },
    /// Streaming started.
    Started,
    /// Frames lost, because no buffer was free or the camera fell behind.
    Dropped {
        /// How many.
        count: u64,
    },
    /// Streaming stopped: no frame comes after this.
    Stopped,
    /// Something went wrong, out of any call.
    Error {
        /// Its kind.
        kind: ErrorKind,
        /// What went wrong.
        message: String,
    },
}

/// Where a streaming camera delivers its frames. Frames come from a pool the driver owns, and
/// a sink may keep them as long as it likes; the driver takes others meanwhile, or drops
/// frames, and says so.
pub trait FrameSink: Send + Sync {
    /// One frame, in order.
    fn deliver(&self, frame: Frame);
}

impl<F: Fn(Frame) + Send + Sync> FrameSink for F {
    fn deliver(&self, frame: Frame) {
        self(frame)
    }
}

/// A camera. Its methods block, so the API layer calls them off its async threads; the trait
/// is `Send + Sync`, so one camera is shared by every thread that uses it.
pub trait Camera: Send + Sync {
    /// What it can do.
    fn capabilities(&self) -> CameraCapabilities;

    /// How it is set now.
    fn config(&self) -> CameraConfig;

    /// Sets it to `config`, checked against its capabilities; `Busy` while streaming.
    fn configure(&self, config: &CameraConfig) -> Result<()>;

    /// Starts streaming frames to `sink`; `Busy` if it is streaming already.
    fn start(&self, sink: Arc<dyn FrameSink>) -> Result<()>;

    /// Stops streaming: once it returns, no more frames reach the sink. Stopping a camera
    /// that isn't streaming does nothing.
    fn stop(&self) -> Result<()>;

    /// Whether it is streaming.
    fn is_streaming(&self) -> bool;

    /// One frame, waiting up to `timeout`; `Busy` while streaming.
    fn snap(&self, timeout: Duration) -> Result<Frame>;

    /// Starts one frame, when streaming on a software trigger.
    fn trigger(&self) -> Result<()>;

    /// Its events, from now on.
    fn subscribe(&self) -> Receiver<CameraEvent>;
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn caps() -> CameraCapabilities {
        CameraCapabilities {
            model: "Test".into(),
            pattern: BayerPattern::Rggb,
            modes: vec![
                CameraMode {
                    id: 0,
                    name: "full".into(),
                    width: 64,
                    height: 48,
                    bit_depths: vec![12, 8],
                    max_fps: 30.0,
                    max_fps_16: 15.0,
                },
                CameraMode {
                    id: 3,
                    name: "half".into(),
                    width: 32,
                    height: 24,
                    bit_depths: vec![8],
                    max_fps: 60.0,
                    max_fps_16: 60.0,
                },
            ],
            binnings: vec![Binning { x: 2, y: 2 }],
            roi_step: Some((4, 2)),
            exposure_ms: Span {
                min: 0.1,
                max: 1_000.0,
            },
            gain: Span {
                min: 1.0,
                max: 16.0,
            },
            triggers: vec![Trigger::FreeRun, Trigger::Software],
        }
    }

    #[test]
    fn a_modes_rate_is_its_8_bit_one_or_in_16_bit_words_above() {
        let c = caps();
        let m = &c.modes[0];
        assert_eq!(
            [m.fps(8), m.fps(9), m.fps(12), m.fps(16)],
            [30.0, 15.0, 15.0, 15.0]
        );
    }

    #[test]
    fn the_default_config_is_valid_and_lays_out_the_whole_sensor() {
        let c = caps();
        let config = c.default_config();
        assert_eq!(
            (config.mode, config.bit_depth, config.exposure),
            (0, 12, Duration::from_millis(10))
        );
        assert_eq!(
            (config.gain, config.trigger, config.roi),
            (1.0, Trigger::FreeRun, None)
        );
        let layout = c.layout(&config).unwrap();
        assert_eq!((layout.width, layout.height), (64, 48));
        assert_eq!(
            layout.format,
            PixelFormat::bayer16(BayerPattern::Rggb, 12).unwrap()
        );
        // A short range's exposure is held inside it.
        let short = CameraCapabilities {
            exposure_ms: Span { min: 0.1, max: 5.0 },
            ..caps()
        };
        assert_eq!(short.default_config().exposure, Duration::from_millis(5));
    }

    #[test]
    fn an_roi_and_binning_shape_the_frames() {
        let c = caps();
        let config = CameraConfig {
            roi: Some(Rect::new(4, 2, 32, 20)),
            binning: Binning { x: 2, y: 2 },
            ..c.default_config()
        };
        let layout = c.layout(&config).unwrap();
        assert_eq!((layout.width, layout.height), (16, 10));
        // An ROI's corner on even columns and rows keeps the Bayer pattern.
        let odd = CameraConfig {
            roi: Some(Rect::new(4, 2, 8, 2)),
            ..c.default_config()
        };
        assert_eq!(
            c.layout(&odd).unwrap().format.bayer_pattern(),
            Some(BayerPattern::Rggb)
        );
    }

    #[test]
    fn each_thing_wrong_with_a_config_is_said() {
        let c = caps();
        let base = c.default_config();
        let err = |config: CameraConfig| c.check(&config).unwrap_err().to_string();
        assert_eq!(
            err(CameraConfig {
                mode: 9,
                ..base.clone()
            }),
            "the camera has no mode 9; its modes are 0, 3"
        );
        assert_eq!(
            err(CameraConfig {
                mode: 3,
                ..base.clone()
            }),
            "mode 3 reads out at [8] bits, not 12"
        );
        assert_eq!(
            err(CameraConfig {
                binning: Binning { x: 4, y: 4 },
                ..base.clone()
            }),
            "the camera doesn't bin 4 × 4"
        );
        assert!(
            err(CameraConfig {
                roi: Some(Rect::new(60, 0, 8, 2)),
                ..base.clone()
            })
            .contains("isn't inside the 64 × 48 sensor")
        );
        assert!(
            err(CameraConfig {
                roi: Some(Rect::new(0, 0, 0, 2)),
                ..base.clone()
            })
            .contains("isn't inside")
        );
        for roi in [
            Rect::new(2, 0, 8, 2),
            Rect::new(0, 1, 8, 2),
            Rect::new(0, 0, 6, 2),
            Rect::new(0, 0, 8, 3),
        ] {
            assert!(
                err(CameraConfig {
                    roi: Some(roi),
                    ..base.clone()
                })
                .contains("steps of 4 across and 2 down"),
                "{roi:?}"
            );
        }
        let whole_only = CameraCapabilities {
            roi_step: None,
            ..caps()
        };
        assert_eq!(
            whole_only
                .check(&CameraConfig {
                    roi: Some(Rect::new(0, 0, 8, 2)),
                    ..base.clone()
                })
                .unwrap_err()
                .to_string(),
            "the camera reads out its whole sensor; it takes no ROI"
        );
        assert_eq!(
            err(CameraConfig {
                exposure: Duration::from_secs(2),
                ..base.clone()
            }),
            "the camera exposes 0.1 to 1000 ms, not 2000"
        );
        assert!(
            err(CameraConfig {
                exposure: Duration::ZERO,
                ..base.clone()
            })
            .contains("not 0")
        );
        assert_eq!(
            err(CameraConfig {
                gain: 0.5,
                ..base.clone()
            }),
            "the camera's gain is 1 to 16, not 0.5"
        );
        assert!(
            err(CameraConfig {
                gain: f64::NAN,
                ..base.clone()
            })
            .contains("not NaN")
        );
        assert_eq!(
            err(CameraConfig {
                trigger: Trigger::External,
                ..base.clone()
            }),
            "the camera takes no External trigger"
        );
        assert!(
            c.check(&CameraConfig {
                trigger: Trigger::Software,
                ..base.clone()
            })
            .is_ok()
        );
        assert!(
            c.check(&CameraConfig {
                exposure: Duration::from_secs(1),
                gain: 16.0,
                ..base.clone()
            })
            .is_ok(),
            "both ends included"
        );
        assert!(c.layout(&CameraConfig { mode: 9, ..base }).is_err());
    }

    #[test]
    fn capabilities_configs_and_events_round_trip_through_json() {
        let c = caps();
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(
            serde_json::from_str::<CameraCapabilities>(&json).unwrap(),
            c
        );
        let config = CameraConfig {
            roi: Some(Rect::new(4, 2, 8, 4)),
            exposure: Duration::from_micros(2_500),
            ..c.default_config()
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["exposure"], 2.5, "in ms");
        assert_eq!(json["trigger"], "free_run");
        assert_eq!(
            serde_json::from_value::<CameraConfig>(json).unwrap(),
            config
        );
        for event in [
            CameraEvent::Configured { config },
            CameraEvent::Started,
            CameraEvent::Dropped { count: 2 },
            CameraEvent::Stopped,
            CameraEvent::Error {
                kind: ErrorKind::Hardware,
                message: "gone".into(),
            },
        ] {
            let json = serde_json::to_string(&event).unwrap();
            assert_eq!(serde_json::from_str::<CameraEvent>(&json).unwrap(), event);
        }
        assert_eq!(
            serde_json::to_value(CameraEvent::Dropped { count: 2 }).unwrap(),
            serde_json::json!({"event": "dropped", "count": 2})
        );
    }
}
