//! Running a plan (P1.08c). Each value is kept where its readers need it: a step on the GPU
//! takes its frames and images there, a step on the host takes them on the host, and a value
//! is copied across once, at the boundary, and the copy counted. Each value is freed after its
//! last reader. Held inputs keep their values from one run to the next, and the steps they
//! alone feed run once and keep their outputs, unless a node is stateful.

use crate::node::Context;
use crate::plan::{Plan, Source};
use crate::port::{PortType, Value};
use std::collections::BTreeMap;
use tetera_core::{Error, Result};

/// The copies a run made between the host and the GPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Copies {
    /// Values copied to the GPU.
    pub uploads: u32,
    /// Values copied to the host.
    pub downloads: u32,
    /// The bytes copied, both ways.
    pub bytes: u64,
}

/// A value, on the host, on the GPU, or both once it has been copied.
#[derive(Debug, Clone, Default)]
pub(crate) struct Slot {
    host: Option<Value>,
    #[cfg(feature = "cuda")]
    device: Option<Value>,
}

impl Slot {
    pub(crate) fn new(value: Value) -> Self {
        #[cfg(feature = "cuda")]
        if value.on_device() {
            return Self {
                host: None,
                device: Some(value),
            };
        }
        Self {
            host: Some(value),
            #[cfg(feature = "cuda")]
            device: None,
        }
    }

    /// Makes it ready where a step needs it: on the GPU, for a step there, if it is a frame or
    /// an image; on the host otherwise. It copies at most once, and counts the copy.
    #[cfg(feature = "cuda")]
    fn prepare(&mut self, on_device: bool, context: &Context, copies: &mut Copies) -> Result<()> {
        let gpu = context.compute.cuda();
        if on_device && self.device.is_none() {
            let gpu = gpu.ok_or_else(|| Error::internal("a step on the GPU without one"))?;
            let host = self.host.as_ref().expect("a value is somewhere");
            if let Some((value, bytes)) = host.upload(gpu)? {
                self.device = Some(value);
                copies.uploads += 1;
                copies.bytes += bytes;
            }
        }
        if !on_device && self.host.is_none() {
            let gpu = gpu.ok_or_else(|| Error::internal("a value on the GPU without one"))?;
            let device = self.device.as_ref().expect("a value is somewhere");
            let (value, bytes) = device
                .download(gpu)?
                .expect("a value on the GPU comes down");
            self.host = Some(value);
            copies.downloads += 1;
            copies.bytes += bytes;
        }
        Ok(())
    }

    /// On the host alone, everything is ready where it is.
    #[cfg(not(feature = "cuda"))]
    fn prepare(&mut self, _: bool, _: &Context, _: &mut Copies) -> Result<()> {
        Ok(())
    }

    /// It where `prepare` made it ready: on the GPU for a step there, when it can be.
    #[cfg(feature = "cuda")]
    fn get(&self, on_device: bool) -> &Value {
        match (&self.host, &self.device) {
            (_, Some(d)) if on_device => d,
            (Some(h), _) => h,
            (None, d) => d.as_ref().expect("a value is somewhere"),
        }
    }

    /// It, on the host.
    #[cfg(not(feature = "cuda"))]
    fn get(&self, _: bool) -> &Value {
        self.host.as_ref().expect("a value is somewhere")
    }
}

/// Every value of a run, by where it comes from.
struct Values<'a> {
    held: &'a mut [Option<Slot>],
    given: Vec<Option<Slot>>,
    constants: &'a mut [Option<Vec<Slot>>],
    slots: Vec<Option<Vec<Slot>>>,
    constant: Vec<bool>,
}

impl Values<'_> {
    fn slot_mut(&mut self, source: Source) -> &mut Slot {
        match source {
            Source::Input(k) => self.held[k]
                .as_mut()
                .or(self.given[k].as_mut())
                .expect("an input is there until its last reader"),
            Source::Output(j, o) => {
                let outputs = if self.constant[j] {
                    &mut self.constants[j]
                } else {
                    &mut self.slots[j]
                };
                &mut outputs
                    .as_mut()
                    .expect("an output is there until its last reader")[o]
            }
        }
    }

    fn slot(&self, source: Source) -> &Slot {
        match source {
            Source::Input(k) => self.held[k]
                .as_ref()
                .or(self.given[k].as_ref())
                .expect("an input is there until its last reader"),
            Source::Output(j, o) => {
                let outputs = if self.constant[j] {
                    &self.constants[j]
                } else {
                    &self.slots[j]
                };
                &outputs
                    .as_ref()
                    .expect("an output is there until its last reader")[o]
            }
        }
    }
}

fn listed_types(types: &[PortType]) -> String {
    if types.is_empty() {
        return "nothing".to_string();
    }
    types
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

impl Plan {
    /// Keeps `value` as input `name` for every run from now on, until held again or let go;
    /// the steps it alone feeds then run once, not every time.
    pub fn hold(&mut self, name: &str, value: Value) -> Result<()> {
        let Some(k) = self.inputs.iter().position(|(n, _)| n == name) else {
            return Err(Error::invalid_argument(format!(
                "recipe `{}` has no input `{name}`",
                self.name
            )));
        };
        let ty = self.inputs[k].1;
        if value.port_type() != ty {
            return Err(Error::invalid_argument(format!(
                "recipe `{}` takes {ty} as input `{name}`, not {}",
                self.name,
                value.port_type()
            )));
        }
        self.held[k] = Some(Slot::new(value));
        self.constants.iter_mut().for_each(|c| *c = None);
        Ok(())
    }

    /// Lets go of a held input: runs must give it again.
    pub fn release(&mut self, name: &str) {
        if let Some(k) = self.inputs.iter().position(|(n, _)| n == name) {
            self.held[k] = None;
            self.constants.iter_mut().for_each(|c| *c = None);
        }
    }

    /// The copies the last run made between the host and the GPU.
    pub fn copies(&self) -> Copies {
        self.copies
    }

    /// The nodes whose steps the last run ran, in order: not those whose outputs were kept
    /// from an earlier run. The nodes the fused kernel stood for are one, their ids joined
    /// by ` + `.
    pub fn ran(&self) -> &[String] {
        &self.ran
    }

    /// The recipe run on `inputs`, one for each of its inputs not held, by name: its outputs,
    /// by name, on the host.
    pub fn run(
        &mut self,
        context: &Context,
        mut inputs: BTreeMap<String, Value>,
    ) -> Result<BTreeMap<String, Value>> {
        let mut given = Vec::with_capacity(self.inputs.len());
        for (k, (name, ty)) in self.inputs.iter().enumerate() {
            if self.held[k].is_some() {
                if inputs.contains_key(name) {
                    return Err(Error::invalid_argument(format!(
                        "recipe `{}` holds input `{name}`; let it go to give it each run",
                        self.name
                    )));
                }
                given.push(None);
                continue;
            }
            let value = inputs.remove(name).ok_or_else(|| {
                Error::invalid_argument(format!(
                    "recipe `{}` takes input `{name}`, which wasn't given",
                    self.name
                ))
            })?;
            if value.port_type() != *ty {
                return Err(Error::invalid_argument(format!(
                    "recipe `{}` takes {ty} as input `{name}`, not {}",
                    self.name,
                    value.port_type()
                )));
            }
            given.push(Some(Slot::new(value)));
        }
        if let Some(name) = inputs.keys().next() {
            return Err(Error::invalid_argument(format!(
                "recipe `{}` has no input `{name}`",
                self.name
            )));
        }

        #[cfg(feature = "cuda")]
        let gpu = context.compute.cuda().is_some();
        #[cfg(not(feature = "cuda"))]
        let gpu = false;
        // What ran on one backend is no use to another: a change of backend works it again.
        if self.last_backend != Some(gpu) {
            self.constants.iter_mut().for_each(|c| *c = None);
            self.last_backend = Some(gpu);
        }
        let fused = if gpu && self.fusion_on {
            self.fusion
                .as_ref()
                .map(|f| (f.chain.clone(), f.sources.clone()))
        } else {
            None
        };
        let n = self.steps.len();
        let last_of_chain = fused
            .as_ref()
            .map(|(chain, _)| *chain.last().expect("a chain"));
        let skipped = |i: usize| {
            fused
                .as_ref()
                .is_some_and(|(chain, _)| chain.contains(&i) && Some(i) != last_of_chain)
        };
        let sources_of = |i: usize| -> &[Source] {
            match &fused {
                Some((_, sources)) if Some(i) == last_of_chain => sources,
                _ => &self.steps[i].sources,
            }
        };

        // Which steps hold their outputs from run to run, and each value's last reader.
        let held_inputs: Vec<bool> = self.held.iter().map(Option::is_some).collect();
        let mut constant = vec![false; n];
        let mut last = vec![0usize; n];
        let mut last_input = vec![0usize; self.inputs.len()];
        for i in (0..n).filter(|&i| !skipped(i)) {
            // The fused step stands for stock nodes, none of them stateful.
            constant[i] = !self.steps[i].descriptor.stateful
                && sources_of(i).iter().all(|s| match *s {
                    Source::Input(k) => held_inputs[k],
                    Source::Output(j, _) => constant[j],
                });
            for s in sources_of(i) {
                match *s {
                    Source::Input(k) => last_input[k] = i,
                    Source::Output(j, _) => last[j] = i,
                }
            }
        }
        for (_, j, _) in &self.outputs {
            last[*j] = usize::MAX;
        }

        let mut copies = Copies::default();
        let names: Vec<String> = self.steps.iter().map(|s| s.id.clone()).collect();
        let mut ran = Vec::new();
        let Plan {
            steps,
            held,
            constants,
            fusion,
            outputs,
            ..
        } = self;
        let mut values = Values {
            held,
            given,
            constants,
            slots: vec![None; n],
            constant,
        };
        for (i, planned) in steps.iter_mut().enumerate() {
            if skipped(i) || (values.constant[i] && values.constants[i].is_some()) {
                continue;
            }
            let fused_here = fused.is_some() && Some(i) == last_of_chain;
            let (step, sources, on_device) = match fusion.as_mut() {
                #[cfg(feature = "cuda")]
                Some(f) if fused_here => (&mut f.step, f.sources.as_slice(), true),
                _ => (
                    &mut planned.step,
                    planned.sources.as_slice(),
                    gpu && planned.descriptor.backends.cuda,
                ),
            };
            ran.push(match &fused {
                Some((chain, _)) if fused_here => chain
                    .iter()
                    .map(|&c| names[c].as_str())
                    .collect::<Vec<_>>()
                    .join(" + "),
                _ => planned.id.clone(),
            });
            for &s in sources {
                values
                    .slot_mut(s)
                    .prepare(on_device, context, &mut copies)?;
            }
            let args: Vec<&Value> = sources
                .iter()
                .map(|&s| values.slot(s).get(on_device))
                .collect();
            let out = step.run(context, &args)?;
            let types: Vec<PortType> = out.iter().map(Value::port_type).collect();
            if types != planned.outputs {
                return Err(Error::internal(format!(
                    "node `{}` ({}) gave {} instead of {}",
                    planned.id,
                    planned.descriptor.id,
                    listed_types(&types),
                    listed_types(&planned.outputs)
                )));
            }
            let slots: Vec<Slot> = out.into_iter().map(Slot::new).collect();
            if values.constant[i] {
                values.constants[i] = Some(slots);
            } else {
                values.slots[i] = Some(slots);
            }
            // Free what this step read last, unless it is held or kept from run to run.
            for &s in sources {
                match s {
                    Source::Input(k) if last_input[k] == i => values.given[k] = None,
                    Source::Output(j, _) if last[j] == i => values.slots[j] = None,
                    _ => {}
                }
            }
        }

        let mut result = BTreeMap::new();
        for (name, j, o) in outputs.iter() {
            let source = Source::Output(*j, *o);
            values
                .slot_mut(source)
                .prepare(false, context, &mut copies)?;
            result.insert(name.clone(), values.slot(source).get(false).clone());
        }
        self.copies = copies;
        self.ran = ran;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Descriptor, Node, Step, parameters};
    use crate::port::PortSpec;
    use crate::recipe::Recipe;
    use crate::registry::Registry;
    use std::sync::{Arc, Mutex};
    use tetera_compute::Compute;

    /// Each run's node and sum.
    type Log = Arc<Mutex<Vec<(String, f64)>>>;

    /// `test.add` adds two scalars, and logs its node's id and the sum; `test.count` is the
    /// same, but stateful.
    struct Add(Descriptor, Log);

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct Name {
        name: String,
    }

    struct AddStep(String, Log);

    impl Node for Add {
        fn descriptor(&self) -> &Descriptor {
            &self.0
        }

        fn prepare(&self, params: &serde_json::Value) -> Result<Box<dyn Step>> {
            let p: Name = parameters(params)?;
            Ok(Box::new(AddStep(p.name, self.1.clone())))
        }
    }

    impl Step for AddStep {
        fn run(&mut self, _: &Context, inputs: &[&Value]) -> Result<Vec<Value>> {
            let [Value::Scalar(a), Value::Scalar(b)] = inputs else {
                return Err(Error::internal("add takes two scalars"));
            };
            self.1.lock().unwrap().push((self.0.clone(), a + b));
            Ok(vec![Value::Scalar(a + b)])
        }
    }

    fn registry() -> (Registry, Log) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = Registry::new();
        for (id, stateful) in [("test.add", false), ("test.count", true)] {
            let scalar = |name| PortSpec::new(name, PortType::Scalar, "a number");
            let mut d = Descriptor::new(id, 1, "adds")
                .input(scalar("a"))
                .input(scalar("b"))
                .output(scalar("sum"))
                .parameters::<Name>();
            if stateful {
                d = d.stateful();
            }
            registry.register(Arc::new(Add(d, log.clone()))).unwrap();
        }
        (registry, log)
    }

    /// `twice` adds x to itself, `held_too` adds `twice` to itself, `count` does the same
    /// statefully, and `sum` adds `held_too` to y.
    const RECIPE: &str = r#"{
        "name": "held",
        "inputs": {"x": {"kind": "scalar"}, "y": {"kind": "scalar"}},
        "nodes": [
            {"id": "twice", "node": "test.add", "params": {"name": "twice"}, "inputs": {"a": "x", "b": "x"}},
            {"id": "held_too", "node": "test.add", "params": {"name": "held_too"}, "inputs": {"a": "twice.sum", "b": "twice.sum"}},
            {"id": "count", "node": "test.count", "params": {"name": "count"}, "inputs": {"a": "twice.sum", "b": "twice.sum"}},
            {"id": "sum", "node": "test.add", "params": {"name": "sum"}, "inputs": {"a": "held_too.sum", "b": "y"}}
        ],
        "outputs": {"sum": "sum.sum", "count": "count.sum"}
    }"#;

    fn names(log: &Mutex<Vec<(String, f64)>>) -> Vec<String> {
        log.lock().unwrap().drain(..).map(|(n, _)| n).collect()
    }

    #[test]
    fn what_held_inputs_alone_feed_runs_once_unless_stateful() {
        let (registry, log) = registry();
        let mut plan = Plan::new(&Recipe::from_json(RECIPE).unwrap(), &registry).unwrap();
        let compute = Compute::cpu_only().unwrap();
        let context = Context { compute: &compute };
        let y = |v: f64| BTreeMap::from([("y".to_string(), Value::Scalar(v))]);
        plan.hold("x", Value::Scalar(1.0)).unwrap();
        let out = plan.run(&context, y(10.0)).unwrap();
        assert_eq!(
            (&out["sum"], &out["count"]),
            (&Value::Scalar(14.0), &Value::Scalar(4.0))
        );
        assert_eq!(names(&log), ["twice", "held_too", "count", "sum"]);
        assert_eq!(plan.ran(), ["twice", "held_too", "count", "sum"]);
        let out = plan.run(&context, y(20.0)).unwrap();
        assert_eq!(out["sum"], Value::Scalar(24.0));
        assert_eq!(
            names(&log),
            ["count", "sum"],
            "the held steps kept; the stateful one ran"
        );
        assert_eq!(plan.ran(), ["count", "sum"]);
        // Held again: worked again.
        plan.hold("x", Value::Scalar(2.0)).unwrap();
        assert_eq!(
            plan.run(&context, y(0.0)).unwrap()["sum"],
            Value::Scalar(8.0)
        );
        assert_eq!(names(&log), ["twice", "held_too", "count", "sum"]);
        assert_eq!(
            plan.copies(),
            Copies::default(),
            "nothing copied on the CPU"
        );
        // Let go: x must be given again, and every step runs each time.
        plan.release("x");
        let both = |x: f64| {
            BTreeMap::from([
                ("x".to_string(), Value::Scalar(x)),
                ("y".to_string(), Value::Scalar(0.0)),
            ])
        };
        plan.run(&context, both(1.0)).unwrap();
        plan.run(&context, both(1.0)).unwrap();
        assert_eq!(names(&log).len(), 8);
        assert!(
            plan.run(&context, y(0.0))
                .unwrap_err()
                .to_string()
                .contains("takes input `x`, which wasn't given")
        );
    }

    #[test]
    fn holding_checks_its_input_and_a_held_input_isnt_given_again() {
        let (registry, _) = registry();
        let mut plan = Plan::new(&Recipe::from_json(RECIPE).unwrap(), &registry).unwrap();
        assert_eq!(
            plan.hold("z", Value::Scalar(1.0)).unwrap_err().to_string(),
            "recipe `held` has no input `z`"
        );
        assert_eq!(
            plan.hold("x", Value::Table(Default::default()))
                .unwrap_err()
                .to_string(),
            "recipe `held` takes a scalar as input `x`, not a table"
        );
        plan.hold("x", Value::Scalar(1.0)).unwrap();
        let compute = Compute::cpu_only().unwrap();
        let both = BTreeMap::from([
            ("x".to_string(), Value::Scalar(1.0)),
            ("y".to_string(), Value::Scalar(0.0)),
        ]);
        assert_eq!(
            plan.run(&Context { compute: &compute }, both)
                .unwrap_err()
                .to_string(),
            "recipe `held` holds input `x`; let it go to give it each run"
        );
        plan.release("nothing");
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use crate::recipe::Recipe;
    use crate::registry::Registry;
    use tetera_compute::Compute;
    use tetera_imaging::Image;
    use tetera_imaging::pattern::{Sensor, test_pattern};

    fn frames(w: u32, h: u32) -> [Value; 3] {
        let sensor = Sensor {
            offset: 128,
            vignetting: 0.3,
            response: [0.7, 1.0, 0.5],
            ..Sensor::default()
        };
        let even = Image::new(w, h, 3, [3_000u16; 3].repeat((w * h) as usize)).unwrap();
        [
            Value::Raw(sensor.expose(&test_pattern(w, h, 12)).unwrap()),
            Value::Raw(sensor.dark(w, h).unwrap()),
            Value::Raw(sensor.expose(&even).unwrap()),
        ]
    }

    /// Another sensor's dark and flat: a higher offset, more vignetting.
    fn frames2(w: u32, h: u32) -> [Value; 3] {
        let sensor = Sensor {
            offset: 200,
            vignetting: 0.5,
            response: [0.7, 1.0, 0.5],
            ..Sensor::default()
        };
        let even = Image::new(w, h, 3, [3_000u16; 3].repeat((w * h) as usize)).unwrap();
        [
            Value::Raw(sensor.expose(&test_pattern(w, h, 12)).unwrap()),
            Value::Raw(sensor.dark(w, h).unwrap()),
            Value::Raw(sensor.expose(&even).unwrap()),
        ]
    }

    fn chain() -> Recipe {
        let mut recipe = Recipe::from_json(include_str!("../recipes/live-chain.json")).unwrap();
        recipe
            .nodes
            .iter_mut()
            .find(|n| n.id == "small")
            .unwrap()
            .params = serde_json::json!({"width": 96, "height": 64});
        recipe
    }

    #[test]
    fn a_chain_on_the_gpu_copies_only_at_its_boundaries_fused_or_not() {
        let _gpu = tetera_compute::cuda_or_skip!();
        let gpu = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = Compute::cpu_only().unwrap();
        let (w, h) = (192, 128);
        let [raw, dark, flat] = frames(w, h);
        let frame_bytes = u64::from(w * h) * 2;
        let preview_bytes = 96 * 64 * 3;
        let mut expected = Plan::new(&chain(), &Registry::stock()).unwrap();
        let inputs = BTreeMap::from([
            ("raw".to_string(), raw.clone()),
            ("dark".to_string(), dark.clone()),
            ("flat".to_string(), flat.clone()),
        ]);
        let on_cpu = expected
            .run(&Context { compute: &cpu }, inputs.clone())
            .unwrap();
        assert_eq!(expected.copies(), Copies::default());
        for fuse in [false, true] {
            let mut plan = Plan::new(&chain(), &Registry::stock()).unwrap();
            plan.set_fusion(fuse);
            // Everything given each run: three frames up, the preview down for the JPEG.
            let out = plan
                .run(&Context { compute: &gpu }, inputs.clone())
                .unwrap();
            assert_eq!(out, on_cpu, "fused: {fuse}");
            let all = Copies {
                uploads: 3,
                downloads: 1,
                bytes: 3 * frame_bytes + preview_bytes,
            };
            assert_eq!(plan.copies(), all, "fused: {fuse}");
            // The dark and the flat held: uploaded once, then only the frame each run.
            plan.hold("dark", dark.clone()).unwrap();
            plan.hold("flat", flat.clone()).unwrap();
            let frame = BTreeMap::from([("raw".to_string(), raw.clone())]);
            assert_eq!(
                plan.run(&Context { compute: &gpu }, frame.clone()).unwrap(),
                on_cpu
            );
            assert_eq!(plan.copies(), all, "the first run uploads the held masters");
            for _ in 0..2 {
                assert_eq!(
                    plan.run(&Context { compute: &gpu }, frame.clone()).unwrap(),
                    on_cpu
                );
                let each = Copies {
                    uploads: 1,
                    downloads: 1,
                    bytes: frame_bytes + preview_bytes,
                };
                assert_eq!(plan.copies(), each, "fused: {fuse}");
            }
            let order = if fuse {
                vec!["less_dark + even + rgb + small + colour", "jpeg"]
            } else {
                vec!["less_dark", "even", "rgb", "small", "colour", "jpeg"]
            };
            assert_eq!(plan.ran(), order, "the flat's own dark subtraction kept");
            // Another master flat, then another dark: the preview the CPU gives with them.
            let [_, dark2, flat2] = frames2(w, h);
            for (name, value) in [("flat", flat2.clone()), ("dark", dark2.clone())] {
                plan.hold(name, value).unwrap();
                let mut inputs = BTreeMap::from([("raw".to_string(), raw.clone())]);
                let d = if name == "dark" {
                    dark2.clone()
                } else {
                    dark.clone()
                };
                inputs.insert("dark".to_string(), d);
                inputs.insert("flat".to_string(), flat2.clone());
                let expected = Plan::new(&chain(), &Registry::stock())
                    .unwrap()
                    .run(&Context { compute: &cpu }, inputs)
                    .unwrap();
                let got = plan.run(&Context { compute: &gpu }, frame.clone()).unwrap();
                assert_eq!(got, expected, "a new {name}, fused: {fuse}");
                assert_ne!(got, on_cpu);
            }
            plan.hold("dark", dark.clone()).unwrap();
            plan.hold("flat", flat.clone()).unwrap();
            // Worked on the GPU, then the same plan on the CPU works its held steps again,
            // there, though nothing was held since.
            assert_eq!(
                plan.run(&Context { compute: &gpu }, frame.clone()).unwrap(),
                on_cpu
            );
            // The same plan on the CPU works its held steps again, there.
            assert_eq!(plan.run(&Context { compute: &cpu }, frame).unwrap(), on_cpu);
            assert_eq!(plan.copies(), Copies::default());
        }
    }

    #[test]
    fn a_step_on_the_host_between_gpu_steps_takes_its_values_down_and_back() {
        let _gpu = tetera_compute::cuda_or_skip!();
        let gpu = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = Compute::cpu_only().unwrap();
        // demosaic (GPU) → crop (host) → resize (GPU) → display (GPU), and the statistics of
        // the crop on the host.
        let recipe = Recipe::from_json(
            r#"{"name": "mixed", "inputs": {"raw": {"kind": "raw"}},
            "nodes": [
                {"id": "rgb", "node": "tetera.demosaic", "inputs": {"raw": "raw"}},
                {"id": "crop", "node": "tetera.crop", "params": {"x": 8, "y": 8, "width": 64, "height": 48}, "inputs": {"image": "rgb.image"}},
                {"id": "stats", "node": "tetera.statistics", "inputs": {"image": "crop.image"}},
                {"id": "small", "node": "tetera.resize", "params": {"width": 32, "height": 24}, "inputs": {"image": "crop.image"}},
                {"id": "show", "node": "tetera.display", "params": {"levels": [{"black": 0, "white": 4095}]}, "inputs": {"image": "small.image"}}
            ],
            "outputs": {"show": "show.image", "stats": "stats.table"}}"#,
        )
        .unwrap();
        let [raw, _, _] = frames(96, 64);
        let inputs = BTreeMap::from([("raw".to_string(), raw)]);
        let mut plan = Plan::new(&recipe, &Registry::stock()).unwrap();
        assert!(plan.fused().is_empty());
        let out = plan
            .run(&Context { compute: &gpu }, inputs.clone())
            .unwrap();
        assert_eq!(
            out,
            plan.run(&Context { compute: &cpu }, inputs.clone())
                .unwrap()
        );
        plan.run(&Context { compute: &gpu }, inputs).unwrap();
        // The frame up; the demosaicked image down for the crop; the crop up again, once, for
        // the resize (the statistics read it on the host); the display's image down.
        let copies = plan.copies();
        assert_eq!((copies.uploads, copies.downloads), (2, 2));
        assert_eq!(
            copies.bytes,
            96 * 64 * 2 + 96 * 64 * 3 * 2 + 64 * 48 * 3 * 2 + 32 * 24 * 3
        );
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod freeing {
    use super::*;
    use crate::node::{Descriptor, NoParameters, Node, Step, parameters};
    use crate::port::PortSpec;
    use crate::recipe::Recipe;
    use crate::registry::Registry;
    use std::sync::{Arc, Mutex};
    use tetera_compute::Compute;
    use tetera_imaging::pattern::{Sensor, test_pattern};

    /// What the taps kept of the values they read on the GPU.
    type Kept = Arc<Mutex<Vec<Value>>>;

    /// `test.tap_raw` and `test.tap_image` keep what they read, and give a scalar;
    /// `test.holders` gives the most holders any kept value has.
    struct Test(Descriptor, Kept);

    struct TestStep(String, Kept);

    impl Node for Test {
        fn descriptor(&self) -> &Descriptor {
            &self.0
        }

        fn prepare(&self, params: &serde_json::Value) -> Result<Box<dyn Step>> {
            let _: NoParameters = parameters(params)?;
            Ok(Box::new(TestStep(self.0.id.to_string(), self.1.clone())))
        }
    }

    fn holders(value: &Value) -> usize {
        match value {
            Value::DeviceRaw(v) => v.holders(),
            Value::DeviceU16(v) => v.holders(),
            _ => 0,
        }
    }

    impl Step for TestStep {
        fn run(&mut self, _: &Context, inputs: &[&Value]) -> Result<Vec<Value>> {
            let mut kept = self.1.lock().unwrap();
            if self.0 == "test.holders" {
                let most = kept.iter().map(holders).max().unwrap_or(0);
                return Ok(vec![Value::Scalar(most as f64)]);
            }
            assert!(inputs[0].on_device(), "a tap on the GPU");
            kept.push(inputs[0].clone());
            Ok(vec![Value::Scalar(0.0)])
        }
    }

    #[test]
    fn a_value_is_let_go_after_its_last_reader() {
        let _gpu = tetera_compute::cuda_or_skip!();
        let gpu = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let kept: Kept = Arc::default();
        let mut registry = Registry::stock();
        let scalar = |name| PortSpec::new(name, PortType::Scalar, "a number");
        for (id, input) in [
            ("test.tap_raw", PortType::Raw),
            ("test.tap_image", PortType::ANY_IMAGE),
        ] {
            let d = Descriptor::new(id, 1, "keeps what it reads")
                .input(PortSpec::new("value", input, "the value"))
                .output(scalar("zero"))
                .on_cuda()
                .stateful();
            registry.register(Arc::new(Test(d, kept.clone()))).unwrap();
        }
        let d = Descriptor::new("test.holders", 1, "counts holders")
            .input(scalar("a"))
            .input(scalar("b"))
            .output(scalar("most"))
            .stateful();
        registry.register(Arc::new(Test(d, kept.clone()))).unwrap();
        // The raw frame's last reader is its tap, the image's its tap; then the count.
        let recipe = Recipe::from_json(
            r#"{"name": "taps", "inputs": {"raw": {"kind": "raw"}},
            "nodes": [
                {"id": "rgb", "node": "tetera.demosaic", "inputs": {"raw": "raw"}},
                {"id": "tap_raw", "node": "test.tap_raw", "inputs": {"value": "raw"}},
                {"id": "tap_rgb", "node": "test.tap_image", "inputs": {"value": "rgb.image"}},
                {"id": "count", "node": "test.holders", "inputs": {"a": "tap_raw.zero", "b": "tap_rgb.zero"}}
            ],
            "outputs": {"most": "count.most"}}"#,
        )
        .unwrap();
        let mut plan = Plan::new(&recipe, &registry).unwrap();
        let raw = Value::Raw(Sensor::default().expose(&test_pattern(32, 16, 12)).unwrap());
        let inputs = BTreeMap::from([("raw".to_string(), raw)]);
        let out = plan.run(&Context { compute: &gpu }, inputs).unwrap();
        assert_eq!(kept.lock().unwrap().len(), 2);
        assert_eq!(
            out["most"],
            Value::Scalar(1.0),
            "only the taps hold what they read"
        );
    }
}
