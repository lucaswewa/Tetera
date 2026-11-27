//! The simulated light (P1.12a): a transmitted channel, dimmable, its colour temperature set
//! from 2,700 to 6,500 K, which the field takes as its `Light`. It keeps and checks its state
//! as the manual light does, and turns it into irradiance.

use crate::optics::Light;
use std::sync::mpsc::Receiver;
use tetera_core::Result;
use tetera_core::frame::meta::LightState;
use tetera_hal::{
    Illumination, IlluminationCapabilities, IlluminationEvent, LightChannel, ManualIllumination,
    Span,
};

/// Its one channel.
pub const CHANNEL: &str = "transmitted";

/// The colour temperature it gives when none is set, K.
pub const DEFAULT_KELVIN: f64 = 3_200.0;

/// The simulated light.
#[derive(Debug)]
pub struct VirtualIllumination {
    light: ManualIllumination,
    /// Through bare glass at full intensity, photoelectrons a ms.
    full: f32,
    unevenness: f32,
}

impl VirtualIllumination {
    /// A light giving `full` photoelectrons a ms through bare glass at full intensity, its
    /// field `unevenness` uneven; off, to begin with.
    pub fn new(full: f32, unevenness: f32) -> Self {
        Self {
            light: ManualIllumination::new(IlluminationCapabilities {
                model: "Tetera virtual light".into(),
                channels: vec![LightChannel {
                    name: CHANNEL.into(),
                    dimmable: true,
                    colour_temperature_k: Some(Span {
                        min: 2_700.0,
                        max: 6_500.0,
                    }),
                }],
            }),
            full,
            unevenness,
        }
    }

    /// The light the field takes now: off, it gives nothing.
    pub fn light(&self) -> Light {
        let state = self.light.state(CHANNEL).unwrap_or(LightState {
            on: false,
            intensity: 0.0,
            colour_temperature_k: None,
        });
        Light {
            intensity: if state.on {
                self.full * state.intensity as f32
            } else {
                0.0
            },
            colour_temperature: state.colour_temperature_k.unwrap_or(DEFAULT_KELVIN) as f32,
            unevenness: self.unevenness,
        }
    }
}

impl Illumination for VirtualIllumination {
    fn capabilities(&self) -> IlluminationCapabilities {
        self.light.capabilities()
    }

    fn state(&self, channel: &str) -> Result<LightState> {
        self.light.state(channel)
    }

    fn set(&self, channel: &str, state: LightState) -> Result<()> {
        self.light.set(channel, state)
    }

    fn subscribe(&self) -> Receiver<IlluminationEvent> {
        self.light.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetera_core::ErrorKind;

    #[test]
    fn its_state_becomes_the_fields_light() {
        let lamp = VirtualIllumination::new(2_000.0, 0.04);
        let events = lamp.subscribe();
        assert_eq!(lamp.light().intensity, 0.0, "off, to begin with");
        assert_eq!(lamp.light().colour_temperature, 3_200.0);
        lamp.set(
            CHANNEL,
            LightState {
                on: true,
                intensity: 0.5,
                colour_temperature_k: Some(5_000.0),
            },
        )
        .unwrap();
        assert_eq!(
            lamp.light(),
            Light {
                intensity: 1_000.0,
                colour_temperature: 5_000.0,
                unevenness: 0.04
            }
        );
        assert_eq!(lamp.state(CHANNEL).unwrap().intensity, 0.5);
        assert!(
            lamp.set(
                CHANNEL,
                LightState {
                    on: true,
                    intensity: 0.5,
                    colour_temperature_k: Some(9_000.0)
                }
            )
            .is_err()
        );
        assert_eq!(
            lamp.set(
                "uv",
                LightState {
                    on: true,
                    intensity: 1.0,
                    colour_temperature_k: None
                }
            )
            .unwrap_err()
            .kind(),
            ErrorKind::NotFound
        );
        lamp.set(
            CHANNEL,
            LightState {
                on: false,
                intensity: 0.5,
                colour_temperature_k: None,
            },
        )
        .unwrap();
        assert_eq!(lamp.light().intensity, 0.0, "off again");
        assert_eq!(events.try_iter().count(), 2);
        assert_eq!(lamp.capabilities().channels[0].name, CHANNEL);
    }
}
