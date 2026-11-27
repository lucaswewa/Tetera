//! Nodes: what each declares, its descriptor, and how it runs, prepared once with its
//! parameters and then stepped value by value.

use crate::port::{PortSpec, PortType, Value};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tetera_compute::Compute;
use tetera_core::{Error, Result};

/// The backends a node can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Backends {
    /// The CPU.
    pub cpu: bool,
    /// CUDA.
    pub cuda: bool,
}

/// What a node declares: its id and version, its ports, its parameters' JSON Schema, its
/// backends, and whether the fusion pass may fold it into a fused kernel.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Descriptor {
    /// Its stable id, such as `tetera.crop`.
    pub id: &'static str,
    /// Its version: a recipe may ask for one.
    pub version: u32,
    /// What it does, in a sentence.
    pub summary: &'static str,
    /// Its inputs, in the order its step takes them.
    pub inputs: Vec<PortSpec>,
    /// Its outputs, in the order its step gives them.
    pub outputs: Vec<PortSpec>,
    /// Its parameters' JSON Schema.
    pub parameters: serde_json::Value,
    /// The backends it runs on.
    pub backends: Backends,
    /// Whether the fusion pass may fold it into a fused kernel.
    pub fusable: bool,
}

/// The parameters of a node that takes none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoParameters {}

impl Descriptor {
    /// A node with no ports yet, taking no parameters, on the CPU alone, not fusable.
    pub fn new(id: &'static str, version: u32, summary: &'static str) -> Self {
        Self {
            id,
            version,
            summary,
            inputs: Vec::new(),
            outputs: Vec::new(),
            parameters: schema::<NoParameters>(),
            backends: Backends {
                cpu: true,
                cuda: false,
            },
            fusable: false,
        }
    }

    /// The same, with `port` as its next input.
    pub fn input(mut self, port: PortSpec) -> Self {
        self.inputs.push(port);
        self
    }

    /// The same, with `port` as its next output.
    pub fn output(mut self, port: PortSpec) -> Self {
        self.outputs.push(port);
        self
    }

    /// The same, on CUDA as well as the CPU.
    pub fn on_cuda(mut self) -> Self {
        self.backends.cuda = true;
        self
    }

    /// The same, its parameters those of `P`.
    pub fn parameters<P: JsonSchema>(mut self) -> Self {
        self.parameters = schema::<P>();
        self
    }

    /// Whether it is sound: an id of lower-case words joined by dots; ports named uniquely on
    /// each side; and each output's type either complete or following an input that exists.
    pub fn check(&self) -> Result<()> {
        let bad = |why: String| {
            Err(Error::invalid_argument(format!(
                "node `{}`: {why}",
                self.id
            )))
        };
        let word = |w: &str| {
            !w.is_empty()
                && w.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        };
        if !self.id.split('.').all(word) || !self.id.contains('.') {
            return bad("an id is lower-case words joined by dots, such as `tetera.crop`".into());
        }
        for (side, ports) in [("input", &self.inputs), ("output", &self.outputs)] {
            for (i, port) in ports.iter().enumerate() {
                if !word(port.name) {
                    return bad(format!("{side} `{}` isn't a lower-case word", port.name));
                }
                if ports[..i].iter().any(|p| p.name == port.name) {
                    return bad(format!("two {side}s are named `{}`", port.name));
                }
            }
        }
        for port in &self.outputs {
            match port.follows {
                Some(input) if !self.inputs.iter().any(|p| p.name == input) => {
                    return bad(format!(
                        "output `{}` follows `{input}`, which is no input",
                        port.name
                    ));
                }
                None if !port.ty.is_concrete() => {
                    return bad(format!(
                        "output `{}` leaves its type open, and follows no input",
                        port.name
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The types of its outputs, given the types of its inputs, in order.
    pub fn output_types(&self, inputs: &[PortType]) -> Vec<PortType> {
        self.outputs
            .iter()
            .map(|port| match port.follows {
                Some(name) => {
                    let i = self.inputs.iter().position(|p| p.name == name);
                    i.map_or(port.ty, |i| port.ty.following(&inputs[i]))
                }
                None => port.ty,
            })
            .collect()
    }
}

/// `P`'s JSON Schema, as JSON.
fn schema<P: JsonSchema>() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(P)).expect("a schema is JSON")
}

/// What a step runs with.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    /// The backends chosen.
    pub compute: &'a Compute,
}

/// A node in the registry: its descriptor, and a step made from its parameters.
pub trait Node: Send + Sync {
    /// What it declares.
    fn descriptor(&self) -> &Descriptor;

    /// A step that runs it with `parameters`, or why they won't do, in a phrase.
    fn prepare(&self, parameters: &serde_json::Value) -> Result<Box<dyn Step>>;
}

/// A node prepared with its parameters, run on one set of inputs after another. It may keep
/// state from one to the next, as temporal averaging does.
pub trait Step: Send {
    /// Its outputs, from `inputs`, which are of the types its node declared, in order.
    fn run(&mut self, context: &Context, inputs: &[&Value]) -> Result<Vec<Value>>;
}

/// `parameters` read as `P`, or serde's words for why not, such as ``missing field `width` ``.
pub fn parameters<P: DeserializeOwned>(parameters: &serde_json::Value) -> Result<P> {
    serde_json::from_value(parameters.clone()).map_err(|e| Error::invalid_argument(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::SampleType;

    fn image(name: &'static str) -> PortSpec {
        PortSpec::new(name, PortType::ANY_IMAGE, "an image")
    }

    #[derive(Debug, serde::Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct Gain {
        /// The factor.
        gain: f64,
    }

    #[test]
    fn a_descriptor_gives_its_outputs_types_from_its_inputs() {
        let d = Descriptor::new("test.convert", 1, "converts")
            .input(image("image"))
            .input(PortSpec::new("gain", PortType::Scalar, "a gain"))
            .output(
                PortSpec::new(
                    "image",
                    PortType::Image {
                        channels: None,
                        sample: Some(SampleType::U8),
                    },
                    "it in 8 bits",
                )
                .following("image"),
            )
            .output(PortSpec::new("count", PortType::Scalar, "pixels"));
        d.check().unwrap();
        let given = [PortType::image(3, SampleType::U16), PortType::Scalar];
        assert_eq!(
            d.output_types(&given),
            [PortType::image(3, SampleType::U8), PortType::Scalar]
        );
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(json["outputs"][0]["follows"], "image");
        assert_eq!(json["outputs"][0]["type"]["sample"], "u8");
        assert!(json["outputs"][1].get("follows").is_none());
        assert_eq!(json["backends"]["cpu"], true);
    }

    #[test]
    fn a_descriptor_that_isnt_sound_is_refused() {
        let ok = || Descriptor::new("test.ok", 1, "fine").input(image("image"));
        let message = |d: Descriptor| d.check().unwrap_err().to_string();
        for id in ["crop", "Test.Crop", "test..crop", "test.crop-it", ".crop"] {
            assert!(Descriptor::new(id, 1, "x").check().is_err(), "{id}");
        }
        assert!(Descriptor::new("tetera.crop_2", 1, "x").check().is_ok());
        assert_eq!(
            message(ok().input(image("image"))),
            "node `test.ok`: two inputs are named `image`"
        );
        assert_eq!(
            message(ok().output(image("out"))),
            "node `test.ok`: output `out` leaves its type open, and follows no input"
        );
        assert_eq!(
            message(ok().output(image("out").following("picture"))),
            "node `test.ok`: output `out` follows `picture`, which is no input"
        );
        assert_eq!(
            message(ok().input(image("Big"))),
            "node `test.ok`: input `Big` isn't a lower-case word"
        );
        let two_outputs = ok()
            .output(image("a").following("image"))
            .output(image("a").following("image"));
        assert_eq!(
            message(two_outputs),
            "node `test.ok`: two outputs are named `a`"
        );
        assert!(ok().output(image("out").following("image")).check().is_ok());
    }

    #[test]
    fn parameters_are_read_with_serdes_own_words_and_have_a_schema() {
        let gain: Gain = parameters(&serde_json::json!({"gain": 2.5})).unwrap();
        assert_eq!(gain.gain, 2.5);
        let err = |v| parameters::<Gain>(&v).unwrap_err().to_string();
        assert_eq!(err(serde_json::json!({})), "missing field `gain`");
        assert_eq!(
            err(serde_json::json!({"gain": 1.0, "offset": 2})),
            "unknown field `offset`, expected `gain`"
        );
        assert_eq!(
            err(serde_json::json!({"gain": "high"})),
            "invalid type: string \"high\", expected f64"
        );
        let d = Descriptor::new("test.gain", 2, "gains").parameters::<Gain>();
        assert_eq!(d.parameters["properties"]["gain"]["type"], "number");
        let none = Descriptor::new("test.none", 1, "nothing");
        assert_eq!(none.parameters["type"], "object");
        assert!(parameters::<NoParameters>(&serde_json::json!({})).is_ok());
        assert!(parameters::<NoParameters>(&serde_json::json!({"x": 1})).is_err());
    }
}
