//! The light (plan Q25): its channels, and how each is set. The real instrument's light is
//! set by hand, so [`ManualIllumination`] records what the user says; the virtual light
//! (P1.12) is set for real.

use crate::camera::Span;
use crate::event::Events;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, PoisonError};
use tetera_core::frame::meta::LightState;
use tetera_core::{Error, Result};

/// One of a light's channels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LightChannel {
    /// Its name, such as `transmitted`.
    pub name: String,
    /// Whether its intensity is set between 0 and 1; if not, it is 0 or 1.
    pub dimmable: bool,
    /// The colour temperatures it gives, in kelvin, if they can be set.
    pub colour_temperature_k: Option<Span>,
}

/// What a light can do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct IlluminationCapabilities {
    /// Its maker and model.
    pub model: String,
    /// Its channels.
    pub channels: Vec<LightChannel>,
}

impl IlluminationCapabilities {
    /// The channel named `name`.
    pub fn channel(&self, name: &str) -> Result<&LightChannel> {
        self.position(name).map(|i| &self.channels[i])
    }

    fn position(&self, name: &str) -> Result<usize> {
        self.channels
            .iter()
            .position(|c| c.name == name)
            .ok_or_else(|| Error::not_found(format!("the light has no channel named {name:?}")))
    }

    /// Whether `state` is one the channel named `channel` takes, saying what is wrong.
    pub fn check(&self, channel: &str, state: &LightState) -> Result<()> {
        let c = self.channel(channel)?;
        let bad = |what: String| Err(Error::invalid_argument(format!("{channel}: {what}")));
        let intensity = state.intensity;
        if !(0.0..=1.0).contains(&intensity) {
            return bad(format!("an intensity of {intensity}; it is from 0 to 1"));
        }
        if !c.dimmable && intensity != 0.0 && intensity != 1.0 {
            return bad(format!(
                "an intensity of {intensity}; it isn't dimmable, so it is 0 or 1"
            ));
        }
        match (state.colour_temperature_k, c.colour_temperature_k) {
            (Some(k), Some(span)) if !span.holds(k) => bad(format!(
                "{k} K; its colour temperature is from {} to {} K",
                span.min, span.max
            )),
            (Some(k), None) => bad(format!("{k} K; its colour temperature isn't set")),
            _ => Ok(()),
        }
    }
}

/// What a light tells its subscribers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum IlluminationEvent {
    /// A channel was set.
    Changed {
        /// The channel.
        channel: String,
        /// How it is set now.
        state: LightState,
    },
}

/// A light, by channel.
pub trait Illumination: Send + Sync {
    /// What it can do.
    fn capabilities(&self) -> IlluminationCapabilities;

    /// How the channel named `channel` is set.
    fn state(&self, channel: &str) -> Result<LightState>;

    /// Sets the channel named `channel` to `state`; a state it can't take is refused.
    fn set(&self, channel: &str, state: LightState) -> Result<()>;

    /// A channel of its events from now on.
    fn subscribe(&self) -> Receiver<IlluminationEvent>;
}

/// A light set by hand, which records what the user says of it. Every channel is off to
/// begin with.
#[derive(Debug)]
pub struct ManualIllumination {
    capabilities: IlluminationCapabilities,
    states: Mutex<Vec<LightState>>,
    events: Events<IlluminationEvent>,
}

impl ManualIllumination {
    /// A light with `capabilities`.
    pub fn new(capabilities: IlluminationCapabilities) -> Self {
        let off = LightState {
            on: false,
            intensity: 0.0,
            colour_temperature_k: None,
        };
        Self {
            states: Mutex::new(vec![off; capabilities.channels.len()]),
            capabilities,
            events: Events::default(),
        }
    }
}

impl Illumination for ManualIllumination {
    fn capabilities(&self) -> IlluminationCapabilities {
        self.capabilities.clone()
    }

    fn state(&self, channel: &str) -> Result<LightState> {
        let i = self.capabilities.position(channel)?;
        Ok(self.states.lock().unwrap_or_else(PoisonError::into_inner)[i])
    }

    fn set(&self, channel: &str, state: LightState) -> Result<()> {
        self.capabilities.check(channel, &state)?;
        let i = self.capabilities.position(channel)?;
        self.states.lock().unwrap_or_else(PoisonError::into_inner)[i] = state;
        self.events.emit(IlluminationEvent::Changed {
            channel: channel.into(),
            state,
        });
        Ok(())
    }

    fn subscribe(&self) -> Receiver<IlluminationEvent> {
        self.events.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetera_core::ErrorKind;

    fn caps() -> IlluminationCapabilities {
        IlluminationCapabilities {
            model: "a lamp".into(),
            channels: vec![
                LightChannel {
                    name: "transmitted".into(),
                    dimmable: true,
                    colour_temperature_k: Some(Span {
                        min: 2_700.0,
                        max: 6_500.0,
                    }),
                },
                LightChannel {
                    name: "ring".into(),
                    dimmable: false,
                    colour_temperature_k: None,
                },
            ],
        }
    }

    fn lit(intensity: f64, k: Option<f64>) -> LightState {
        LightState {
            on: true,
            intensity,
            colour_temperature_k: k,
        }
    }

    #[test]
    fn each_thing_wrong_with_a_light_state_is_said() {
        let caps = caps();
        for (channel, state) in [
            ("transmitted", lit(0.0, None)),
            ("transmitted", lit(0.5, Some(2_700.0))),
            ("transmitted", lit(1.0, Some(6_500.0))),
            ("ring", lit(0.0, None)),
            ("ring", lit(1.0, None)),
        ] {
            caps.check(channel, &state)
                .unwrap_or_else(|e| panic!("{channel} {state:?}: {e}"));
        }
        for (channel, state, says) in [
            (
                "transmitted",
                lit(1.5, None),
                "transmitted: an intensity of 1.5; it is from 0 to 1",
            ),
            ("transmitted", lit(-0.1, None), "from 0 to 1"),
            ("transmitted", lit(f64::NAN, None), "from 0 to 1"),
            (
                "ring",
                lit(0.5, None),
                "ring: an intensity of 0.5; it isn't dimmable",
            ),
            (
                "transmitted",
                lit(0.5, Some(2_600.0)),
                "2600 K; its colour temperature is from 2700 to 6500 K",
            ),
            (
                "transmitted",
                lit(0.5, Some(6_600.0)),
                "from 2700 to 6500 K",
            ),
            (
                "ring",
                lit(1.0, Some(3_000.0)),
                "ring: 3000 K; its colour temperature isn't set",
            ),
        ] {
            let e = caps.check(channel, &state).unwrap_err();
            assert_eq!(e.kind(), ErrorKind::InvalidArgument);
            assert!(e.message().contains(says), "{e}");
        }
        assert_eq!(
            caps.check("uv", &lit(1.0, None)).unwrap_err().kind(),
            ErrorKind::NotFound
        );
    }

    #[test]
    fn the_manual_light_records_what_the_user_says() {
        let light = ManualIllumination::new(caps());
        let events = light.subscribe();
        assert_eq!(light.capabilities(), caps());
        assert!(!light.state("ring").unwrap().on, "off, to begin with");
        light.set("ring", lit(1.0, None)).unwrap();
        light.set("transmitted", lit(0.8, Some(3_200.0))).unwrap();
        assert_eq!(light.state("ring").unwrap(), lit(1.0, None));
        assert_eq!(light.state("transmitted").unwrap(), lit(0.8, Some(3_200.0)));
        assert!(light.set("ring", lit(0.5, None)).is_err());
        assert_eq!(light.state("ring").unwrap(), lit(1.0, None), "unchanged");
        assert_eq!(light.state("uv").unwrap_err().kind(), ErrorKind::NotFound);
        let seen: Vec<IlluminationEvent> = events.try_iter().collect();
        assert_eq!(seen.len(), 2);
        assert_eq!(
            serde_json::to_string(&seen[1]).unwrap(),
            r#"{"event":"changed","channel":"transmitted","state":{"on":true,"intensity":0.8,"colour_temperature_k":3200.0}}"#
        );
        let json = serde_json::to_string(&caps()).unwrap();
        assert_eq!(
            serde_json::from_str::<IlluminationCapabilities>(&json).unwrap(),
            caps()
        );
    }
}
