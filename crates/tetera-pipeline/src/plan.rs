//! A recipe checked against the registry, its nodes prepared and put in the order they run,
//! and that order run on values.

use crate::fuse::Fusion;
use crate::node::{Descriptor, Step};
use crate::port::PortType;
use crate::recipe::Recipe;
use crate::registry::Registry;
use crate::run::{Copies, Slot};
use std::collections::BTreeMap;
use std::fmt;
use tetera_core::{Error, Result};

/// One thing wrong with a recipe: where, and what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// Where in the recipe, such as ``node `stats`, input `image` ``.
    pub at: String,
    /// What is wrong there.
    pub message: String,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.at, self.message)
    }
}

/// Where a value comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    /// The recipe's input of that index, by name.
    Input(usize),
    /// A step's output: the step's index in the order, and the output's.
    Output(usize, usize),
}

/// A node of the recipe, prepared, in the order steps run.
pub(crate) struct Planned {
    pub(crate) id: String,
    pub(crate) descriptor: Descriptor,
    /// Its parameters, as the recipe gave them.
    pub(crate) params: serde_json::Value,
    pub(crate) step: Box<dyn Step>,
    pub(crate) sources: Vec<Source>,
    pub(crate) outputs: Vec<PortType>,
}

/// A recipe checked and ready to run: each node prepared, in an order where every node runs
/// after those that feed it.
pub struct Plan {
    pub(crate) name: String,
    pub(crate) inputs: Vec<(String, PortType)>,
    pub(crate) steps: Vec<Planned>,
    /// Each output's name, and the step and output it comes from.
    pub(crate) outputs: Vec<(String, usize, usize)>,
    /// The inputs held from run to run.
    pub(crate) held: Vec<Option<Slot>>,
    /// The outputs of the steps held inputs alone feed, kept from run to run.
    pub(crate) constants: Vec<Option<Vec<Slot>>>,
    /// The live chain, found for the fused kernel.
    pub(crate) fusion: Option<Fusion>,
    /// Whether the fused kernel runs in the chain's place, on the GPU.
    pub(crate) fusion_on: bool,
    /// Whether the last run had the GPU.
    pub(crate) last_backend: Option<bool>,
    pub(crate) copies: Copies,
}

impl fmt::Debug for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan")
            .field("name", &self.name)
            .field("order", &self.order())
            .finish()
    }
}

/// A reference's two parts: a node's id and its output's name, or a recipe input's name.
fn split(reference: &str) -> (&str, Option<&str>) {
    match reference.split_once('.') {
        Some((node, port)) => (node, Some(port)),
        None => (reference, None),
    }
}

/// A list of names in backticks: `` `a`, `b` and `c` ``.
fn listed<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let names: Vec<String> = names.into_iter().map(|n| format!("`{n}`")).collect();
    match names.as_slice() {
        [] => "none".to_string(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// What a recipe's nodes resolve to as validation goes.
struct Checked {
    descriptor: Descriptor,
    step: Box<dyn Step>,
}

impl Plan {
    /// `recipe` checked against `registry`, or everything wrong with it, as one error.
    pub fn new(recipe: &Recipe, registry: &Registry) -> Result<Self> {
        Self::check(recipe, registry).map_err(|problems| {
            let all: Vec<String> = problems.iter().map(Problem::to_string).collect();
            Error::invalid_argument(format!(
                "recipe `{}` has {} problem{}: {}",
                recipe.name,
                problems.len(),
                if problems.len() == 1 { "" } else { "s" },
                all.join("; ")
            ))
        })
    }

    /// `recipe` checked against `registry`, or each thing wrong with it.
    pub fn check(recipe: &Recipe, registry: &Registry) -> std::result::Result<Self, Vec<Problem>> {
        let mut problems = Vec::new();
        let mut problem = |at: String, message: String| problems.push(Problem { at, message });

        // The recipe's inputs: plain names, of types with nothing left open.
        let inputs: Vec<(String, PortType)> =
            recipe.inputs.iter().map(|(n, t)| (n.clone(), *t)).collect();
        for (name, ty) in &inputs {
            if name.contains('.') || name.is_empty() {
                problem(
                    format!("input `{name}`"),
                    "a name can't be empty or hold a `.`".into(),
                );
            } else if !ty.is_concrete() {
                problem(
                    format!("input `{name}`"),
                    format!("{ty} leaves its channels or sample type open"),
                );
            }
        }
        if recipe.outputs.is_empty() {
            problem("the recipe".into(), "it gives no outputs".into());
        }

        // Each node: its id, the node it names, and its parameters.
        let mut checked: Vec<Option<Checked>> = Vec::with_capacity(recipe.nodes.len());
        for (i, use_) in recipe.nodes.iter().enumerate() {
            let at = format!("node `{}`", use_.id);
            if use_.id.contains('.') || use_.id.is_empty() {
                problem(at.clone(), "an id can't be empty or hold a `.`".into());
            } else if recipe.nodes[..i].iter().any(|u| u.id == use_.id) {
                problem(at.clone(), "another node has this id".into());
            } else if recipe.inputs.contains_key(&use_.id) {
                problem(at.clone(), "the recipe has an input of this name".into());
            }
            let node = match registry.find(&use_.node, use_.version) {
                Ok(node) => node,
                Err(why) => {
                    problem(at, why);
                    checked.push(None);
                    continue;
                }
            };
            let descriptor = node.descriptor().clone();
            for name in use_.inputs.keys() {
                if !descriptor.inputs.iter().any(|p| p.name == name) {
                    problem(
                        format!("{at}, input `{name}`"),
                        format!(
                            "`{}` has no such input; its inputs are {}",
                            descriptor.id,
                            listed(descriptor.inputs.iter().map(|p| p.name))
                        ),
                    );
                }
            }
            match node.prepare(&use_.params) {
                Ok(step) => checked.push(Some(Checked { descriptor, step })),
                Err(e) => {
                    problem(format!("{at}, parameters"), e.message().to_string());
                    checked.push(None);
                }
            }
        }

        // What feeds each node's inputs: its sources, as recipe indices, when they resolve.
        let index_of = |id: &str| recipe.nodes.iter().position(|u| u.id == id);
        let resolve = |reference: &str| -> std::result::Result<Resolved, String> {
            match split(reference) {
                (name, None) => match inputs.iter().position(|(n, _)| n == name) {
                    Some(i) => Ok(Resolved::Input(i)),
                    None => Err(format!(
                        "`{name}` is no input of the recipe; its inputs are {}",
                        listed(inputs.iter().map(|(n, _)| n.as_str()))
                    )),
                },
                (node, Some(port)) => {
                    let Some(n) = index_of(node) else {
                        return Err(format!("`{reference}` names no node of the recipe"));
                    };
                    match &checked[n] {
                        None => Err(String::new()),
                        Some(c) => match c.descriptor.outputs.iter().position(|p| p.name == port) {
                            Some(o) => Ok(Resolved::Output(n, o)),
                            None => Err(format!(
                                "`{node}` has no output `{port}`; its outputs are {}",
                                listed(c.descriptor.outputs.iter().map(|p| p.name))
                            )),
                        },
                    }
                }
            }
        };
        let mut feeds: Vec<Vec<Option<Resolved>>> = Vec::with_capacity(recipe.nodes.len());
        for (use_, c) in recipe.nodes.iter().zip(&checked) {
            let Some(c) = c else {
                feeds.push(Vec::new());
                continue;
            };
            let mut sources = Vec::with_capacity(c.descriptor.inputs.len());
            for port in &c.descriptor.inputs {
                let at = format!("node `{}`, input `{}`", use_.id, port.name);
                sources.push(match use_.inputs.get(port.name) {
                    None => {
                        problem(at, "nothing feeds it".into());
                        None
                    }
                    Some(reference) => match resolve(reference) {
                        Ok(source) => Some(source),
                        Err(why) => {
                            if !why.is_empty() {
                                problem(at, why);
                            }
                            None
                        }
                    },
                });
            }
            feeds.push(sources);
        }

        // The order: each node after those that feed it, ties kept in the recipe's order.
        let n = recipe.nodes.len();
        let depends = |i: usize| -> Vec<usize> {
            feeds[i]
                .iter()
                .filter_map(|s| match s {
                    Some(Resolved::Output(from, _)) => Some(*from),
                    _ => None,
                })
                .collect()
        };
        let mut placed = vec![false; n];
        let mut order = Vec::with_capacity(n);
        while placed.contains(&false) {
            let ready = (0..n).find(|&i| !placed[i] && depends(i).iter().all(|&d| placed[d]));
            if let Some(i) = ready {
                placed[i] = true;
                order.push(i);
                continue;
            }
            // Every node left waits on another: walk back along what each waits on to a node
            // seen before, and report the loop, in the order data flows.
            let start = (0..n).find(|&i| !placed[i]).expect("a node is left");
            let mut path = vec![start];
            let cycle = loop {
                let last = *path.last().expect("the path starts with a node");
                let next = depends(last)
                    .into_iter()
                    .find(|&d| !placed[d])
                    .expect("a node left waits on another left");
                if let Some(at) = path.iter().position(|&p| p == next) {
                    break path[at..].to_vec();
                }
                path.push(next);
            };
            // Data flows against the walk; the loop is told from its first node in the recipe.
            let mut flow: Vec<usize> = cycle.iter().rev().copied().collect();
            let first = (0..flow.len())
                .min_by_key(|&k| flow[k])
                .expect("a loop has a node");
            flow.rotate_left(first);
            flow.push(flow[0]);
            let flow: Vec<&str> = flow.iter().map(|&i| recipe.nodes[i].id.as_str()).collect();
            let arrows: Vec<String> = flow.iter().map(|id| format!("`{id}`")).collect();
            problem(
                "the recipe".into(),
                format!("its nodes {} form a cycle", arrows.join(" → ")),
            );
            // Its nodes are set aside, so what they feed is still ordered, and left unchecked.
            for &i in &cycle {
                placed[i] = true;
            }
        }

        // Each input's type against what feeds it, in order, so every source's type is known
        // first. A node downstream of a problem is left unchecked, to report each problem once.
        let mut types: Vec<Option<Vec<PortType>>> = vec![None; n];
        for &i in &order {
            let Some(c) = &checked[i] else { continue };
            let mut given = Vec::with_capacity(c.descriptor.inputs.len());
            for (port, source) in c.descriptor.inputs.iter().zip(&feeds[i]) {
                let ty = match source {
                    Some(Resolved::Input(k)) => Some(inputs[*k].1),
                    Some(Resolved::Output(from, o)) => types[*from].as_ref().map(|t| t[*o]),
                    None => None,
                };
                if let Some(ty) = ty
                    && !port.ty.accepts(&ty)
                {
                    problem(
                        format!("node `{}`, input `{}`", recipe.nodes[i].id, port.name),
                        format!(
                            "it takes {}, but `{}` gives {ty}",
                            port.ty, recipe.nodes[i].inputs[port.name]
                        ),
                    );
                }
                given.push(ty.filter(|t| port.ty.accepts(t)));
            }
            if given.iter().all(Option::is_some) {
                let given: Vec<PortType> = given.into_iter().flatten().collect();
                types[i] = Some(c.descriptor.output_types(&given));
            }
        }

        // The recipe's outputs.
        let mut outputs = Vec::with_capacity(recipe.outputs.len());
        for (name, reference) in &recipe.outputs {
            let at = format!("output `{name}`");
            match resolve(reference) {
                Ok(Resolved::Output(from, o)) => outputs.push((name.clone(), (from, o))),
                Ok(Resolved::Input(_)) => {
                    problem(
                        at,
                        format!(
                            "`{reference}` is an input; an output comes from a node, `node.port`"
                        ),
                    );
                }
                Err(why) if !why.is_empty() => problem(at, why),
                Err(_) => {}
            }
        }

        if !problems.is_empty() {
            return Err(problems);
        }
        let position: BTreeMap<usize, usize> =
            order.iter().enumerate().map(|(p, &i)| (i, p)).collect();
        let mut checked: Vec<Option<Checked>> = checked;
        let steps: Vec<Planned> = order
            .iter()
            .map(|&i| {
                let c = checked[i].take().expect("every node checked");
                let sources = feeds[i]
                    .iter()
                    .map(|s| match s.expect("every input fed") {
                        Resolved::Input(k) => Source::Input(k),
                        Resolved::Output(from, o) => Source::Output(position[&from], o),
                    })
                    .collect();
                Planned {
                    id: recipe.nodes[i].id.clone(),
                    descriptor: c.descriptor,
                    params: recipe.nodes[i].params.clone(),
                    step: c.step,
                    sources,
                    outputs: types[i].clone().expect("every type known"),
                }
            })
            .collect();
        let fusion = crate::fuse::find(&steps, &outputs_planned(&outputs, &position));
        Ok(Self {
            name: recipe.name.clone(),
            held: vec![None; inputs.len()],
            constants: (0..steps.len()).map(|_| None).collect(),
            fusion,
            fusion_on: true,
            last_backend: None,
            copies: Copies::default(),
            inputs,
            steps,
            outputs: outputs
                .into_iter()
                .map(|(name, (from, o))| (name, position[&from], o))
                .collect(),
        })
    }

    /// The recipe's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The nodes' ids, in the order they run.
    pub fn order(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.id.as_str()).collect()
    }

    /// The recipe's inputs and their types, by name.
    pub fn inputs(&self) -> impl Iterator<Item = (&str, &PortType)> {
        self.inputs.iter().map(|(n, t)| (n.as_str(), t))
    }

    /// The ids of the nodes the fused kernel takes the place of on the GPU, in order; none
    /// when the recipe has no live chain the kernel can run.
    pub fn fused(&self) -> Vec<&str> {
        self.fusion
            .as_ref()
            .map(|f| f.chain.iter().map(|&i| self.steps[i].id.as_str()).collect())
            .unwrap_or_default()
    }

    /// Whether the fused kernel runs in the live chain's place on the GPU, which it does unless
    /// told not to.
    pub fn set_fusion(&mut self, on: bool) {
        if self.fusion_on != on {
            self.fusion_on = on;
            self.constants.iter_mut().for_each(|c| *c = None);
        }
    }
}

/// The recipe's outputs as steps' indices in the order.
fn outputs_planned(
    outputs: &[(String, (usize, usize))],
    position: &BTreeMap<usize, usize>,
) -> Vec<(usize, usize)> {
    outputs
        .iter()
        .map(|(_, (from, o))| (position[from], *o))
        .collect()
}

/// A reference resolved: a recipe input's index, or a node's (by its index in the recipe) and
/// its output's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resolved {
    Input(usize),
    Output(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Context, Descriptor, NoParameters, Node, parameters};
    use crate::port::{PortSpec, SampleType, Table, Value};
    use std::sync::{Arc, Mutex};
    use tetera_compute::Compute;
    use tetera_imaging::Image;

    /// A test node: `test.add` adds its two scalars and a parameter; it logs each run.
    struct Add(Descriptor, Arc<Mutex<Vec<f64>>>);

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct Offset {
        #[serde(default)]
        offset: f64,
    }

    struct AddStep(f64, Arc<Mutex<Vec<f64>>>);

    impl Node for Add {
        fn descriptor(&self) -> &Descriptor {
            &self.0
        }

        fn prepare(&self, params: &serde_json::Value) -> Result<Box<dyn Step>> {
            let p: Offset = parameters(params)?;
            Ok(Box::new(AddStep(p.offset, self.1.clone())))
        }
    }

    impl Step for AddStep {
        fn run(&mut self, _: &Context, inputs: &[&Value]) -> Result<Vec<Value>> {
            let [Value::Scalar(a), Value::Scalar(b)] = inputs else {
                return Err(Error::internal("add takes two scalars"));
            };
            let sum = a + b + self.0;
            self.1.lock().unwrap().push(sum);
            Ok(vec![Value::Scalar(sum)])
        }
    }

    /// `test.wrong` says it gives a scalar, and gives a table.
    struct Wrong(Descriptor);

    impl Node for Wrong {
        fn descriptor(&self) -> &Descriptor {
            &self.0
        }

        fn prepare(&self, params: &serde_json::Value) -> Result<Box<dyn Step>> {
            let _: NoParameters = parameters(params)?;
            Ok(Box::new(WrongStep))
        }
    }

    struct WrongStep;

    impl Step for WrongStep {
        fn run(&mut self, _: &Context, _: &[&Value]) -> Result<Vec<Value>> {
            Ok(vec![Value::Table(Table::new())])
        }
    }

    fn registry() -> (Registry, Arc<Mutex<Vec<f64>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut registry = Registry::stock();
        let scalar = |name| PortSpec::new(name, PortType::Scalar, "a number");
        let add = Descriptor::new("test.add", 1, "adds")
            .input(scalar("a"))
            .input(scalar("b"))
            .output(scalar("sum"))
            .parameters::<Offset>();
        registry.register(Arc::new(Add(add, log.clone()))).unwrap();
        let wrong = Descriptor::new("test.wrong", 1, "lies").output(scalar("out"));
        registry.register(Arc::new(Wrong(wrong))).unwrap();
        (registry, log)
    }

    fn recipe(text: &str) -> Recipe {
        Recipe::from_json(text).unwrap()
    }

    /// Every problem `text` has, as strings.
    fn problems(text: &str) -> Vec<String> {
        let (registry, _) = registry();
        Plan::check(&recipe(text), &registry)
            .unwrap_err()
            .iter()
            .map(Problem::to_string)
            .collect()
    }

    const IMAGE_RECIPE: &str = r#"{
        "name": "measure",
        "inputs": { "frame": { "kind": "image", "channels": 3, "sample": "u16" } },
        "nodes": [
            { "id": "stats", "node": "tetera.statistics", "inputs": { "image": "crop.image" } },
            { "id": "crop", "node": "tetera.crop",
              "params": { "x": 1, "y": 0, "width": 2, "height": 1 }, "inputs": { "image": "frame" } }
        ],
        "outputs": { "statistics": "stats.table", "cropped": "crop.image" }
    }"#;

    #[test]
    fn a_recipe_runs_its_nodes_after_those_that_feed_them() {
        let (registry, _) = registry();
        let mut plan = Plan::new(&recipe(IMAGE_RECIPE), &registry).unwrap();
        assert_eq!(
            plan.order(),
            ["crop", "stats"],
            "crop first, though listed second"
        );
        assert_eq!(plan.name(), "measure");
        assert_eq!(
            plan.inputs().collect::<Vec<_>>(),
            [("frame", &PortType::image(3, SampleType::U16))]
        );
        let frame = Image::new(3, 1, 3, vec![1u16, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let compute = Compute::cpu_only().unwrap();
        let out = plan
            .run(
                &Context { compute: &compute },
                BTreeMap::from([("frame".to_string(), Value::U16(frame))]),
            )
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            out["cropped"],
            Value::U16(Image::new(2, 1, 3, vec![4, 5, 6, 7, 8, 9]).unwrap())
        );
        let Value::Table(t) = &out["statistics"] else {
            panic!("a table")
        };
        assert_eq!((t["mean.0"], t["mean.2"], t["pixels"]), (5.5, 7.5, 2.0));
        assert!(format!("{plan:?}").contains("[\"crop\", \"stats\"]"));
    }

    #[test]
    fn a_diamond_runs_each_node_once_and_ties_keep_the_recipes_order() {
        let (registry, log) = registry();
        let text = r#"{
            "name": "diamond",
            "inputs": { "x": { "kind": "scalar" }, "y": { "kind": "scalar" } },
            "nodes": [
                { "id": "last", "node": "test.add", "inputs": { "a": "left.sum", "b": "right.sum" } },
                { "id": "right", "node": "test.add", "params": { "offset": 100 }, "inputs": { "a": "x", "b": "y" } },
                { "id": "left", "node": "test.add", "inputs": { "a": "x", "b": "x" } }
            ],
            "outputs": { "total": "last.sum", "left": "left.sum" }
        }"#;
        let mut plan = Plan::new(&recipe(text), &registry).unwrap();
        assert_eq!(plan.order(), ["right", "left", "last"]);
        let compute = Compute::cpu_only().unwrap();
        let inputs = BTreeMap::from([
            ("x".to_string(), Value::Scalar(1.0)),
            ("y".to_string(), Value::Scalar(2.0)),
        ]);
        let out = plan
            .run(&Context { compute: &compute }, inputs.clone())
            .unwrap();
        assert_eq!(
            (&out["total"], &out["left"]),
            (&Value::Scalar(105.0), &Value::Scalar(2.0))
        );
        assert_eq!(
            *log.lock().unwrap(),
            [103.0, 2.0, 105.0],
            "each node once, in order"
        );
        // Run again: the same steps, so a node's state would carry over.
        plan.run(&Context { compute: &compute }, inputs).unwrap();
        assert_eq!(log.lock().unwrap().len(), 6);
    }

    #[test]
    fn each_kind_of_problem_is_said_precisely() {
        let cases: &[(&str, &[&str])] = &[
            (
                r#"{"name": "r", "nodes": [{"id": "a", "node": "test.ad"}], "outputs": {"o": "a.sum"}}"#,
                &["node `a`: no node `test.ad` is registered"],
            ),
            (
                r#"{"name": "r", "nodes": [{"id": "a", "node": "test.add", "version": 2}], "outputs": {"o": "a.sum"}}"#,
                &["node `a`: `test.add` has no version 2; the registry has 1"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "x"}}], "outputs": {"o": "a.sum"}}"#,
                &["node `a`, input `b`: nothing feeds it"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "x", "b": "x", "c": "x"}}], "outputs": {"o": "a.sum"}}"#,
                &["node `a`, input `c`: `test.add` has no such input; its inputs are `a` and `b`"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "x", "b": "z"}}], "outputs": {"o": "a.sum"}}"#,
                &["node `a`, input `b`: `z` is no input of the recipe; its inputs are `x`"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "x", "b": "q.sum"}}], "outputs": {"o": "a.sum"}}"#,
                &["node `a`, input `b`: `q.sum` names no node of the recipe"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "x", "b": "x"}},
                              {"id": "b", "node": "test.add", "inputs": {"a": "a.total", "b": "x"}}],
                    "outputs": {"o": "b.sum"}}"#,
                &["node `b`, input `a`: `a` has no output `total`; its outputs are `sum`"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "params": {"offset": "big"}, "inputs": {"a": "x", "b": "x"}}],
                    "outputs": {"o": "a.sum"}}"#,
                &["node `a`, parameters: invalid type: string \"big\", expected f64"],
            ),
            (
                r#"{"name": "r", "inputs": {"f": {"kind": "image", "channels": 1, "sample": "u8"}},
                    "nodes": [{"id": "c", "node": "tetera.crop", "params": {"x": 0, "y": 0, "width": 1}, "inputs": {"image": "f"}}],
                    "outputs": {"o": "c.image"}}"#,
                &["node `c`, parameters: missing field `height`"],
            ),
            (
                r#"{"name": "r", "inputs": {"f": {"kind": "image", "channels": 1, "sample": "u8"}},
                    "nodes": [{"id": "s", "node": "tetera.statistics", "inputs": {"image": "f"}},
                              {"id": "a", "node": "test.add", "inputs": {"a": "s.table", "b": "f"}}],
                    "outputs": {"o": "a.sum"}}"#,
                &[
                    "node `a`, input `a`: it takes a scalar, but `s.table` gives a table",
                    "node `a`, input `b`: it takes a scalar, but `f` gives an image of 1 channel of u8",
                ],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "c.sum", "b": "x"}},
                              {"id": "b", "node": "test.add", "inputs": {"a": "a.sum", "b": "x"}},
                              {"id": "c", "node": "test.add", "inputs": {"a": "b.sum", "b": "x"}},
                              {"id": "d", "node": "test.add", "inputs": {"a": "c.sum", "b": "x"}}],
                    "outputs": {"o": "d.sum"}}"#,
                &["the recipe: its nodes `a` → `b` → `c` → `a` form a cycle"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "a.sum", "b": "x"}}],
                    "outputs": {"o": "a.sum"}}"#,
                &["the recipe: its nodes `a` → `a` form a cycle"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "x", "b": "x"}},
                              {"id": "a", "node": "test.add", "inputs": {"a": "x", "b": "x"}}],
                    "outputs": {"o": "a.sum"}}"#,
                &["node `a`: another node has this id"],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "x", "node": "test.add", "inputs": {"a": "x", "b": "x"}},
                              {"id": "a.b", "node": "test.add", "inputs": {"a": "x", "b": "x"}},
                              {"id": "", "node": "test.add", "inputs": {"a": "x", "b": "x"}}],
                    "outputs": {"o": "x.sum"}}"#,
                &[
                    "node `x`: the recipe has an input of this name",
                    "node `a.b`: an id can't be empty or hold a `.`",
                    "node ``: an id can't be empty or hold a `.`",
                ],
            ),
            (
                r#"{"name": "r", "inputs": {"x.y": {"kind": "scalar"}, "f": {"kind": "image", "sample": "u8"}},
                    "nodes": [], "outputs": {}}"#,
                &[
                    "input `f`: an image of u8 leaves its channels or sample type open",
                    "input `x.y`: a name can't be empty or hold a `.`",
                    "the recipe: it gives no outputs",
                ],
            ),
            (
                r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}},
                    "nodes": [{"id": "a", "node": "test.add", "inputs": {"a": "x", "b": "x"}}],
                    "outputs": {"o": "x", "p": "a.total", "q": "b.sum"}}"#,
                &[
                    "output `o`: `x` is an input; an output comes from a node, `node.port`",
                    "output `p`: `a` has no output `total`; its outputs are `sum`",
                    "output `q`: `b.sum` names no node of the recipe",
                ],
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(problems(text), *expected, "{text}");
        }
    }

    #[test]
    fn a_problem_is_said_once_not_again_downstream() {
        // `a`'s node is unknown, so `b`, fed by it, isn't reported; `c`'s input is the wrong
        // type, so `d`, fed by `c`, isn't checked against an unknown type either.
        let text = r#"{"name": "r", "inputs": {"x": {"kind": "scalar"}, "f": {"kind": "image", "channels": 1, "sample": "u8"}},
            "nodes": [{"id": "a", "node": "test.nothing"},
                      {"id": "b", "node": "test.add", "inputs": {"a": "a.sum", "b": "x"}},
                      {"id": "c", "node": "test.add", "inputs": {"a": "f", "b": "x"}},
                      {"id": "d", "node": "tetera.statistics", "inputs": {"image": "c.sum"}},
                      {"id": "e", "node": "test.add", "params": {"offset": []}, "inputs": {"a": "x", "b": "x"}},
                      {"id": "g", "node": "test.add", "inputs": {"a": "e.sum", "b": "x"}}],
            "outputs": {"o": "b.sum", "p": "a.sum", "q": "g.sum"}}"#;
        assert_eq!(
            problems(text),
            [
                "node `a`: no node `test.nothing` is registered",
                "node `e`, parameters: invalid type: sequence, expected f64",
                "node `c`, input `a`: it takes a scalar, but `f` gives an image of 1 channel of u8",
            ]
        );
    }

    #[test]
    fn all_the_problems_make_one_error() {
        let (registry, _) = registry();
        let one = r#"{"name": "single", "nodes": [{"id": "a", "node": "test.ad"}], "outputs": {"o": "a.sum"}}"#;
        assert_eq!(
            Plan::new(&recipe(one), &registry).unwrap_err().to_string(),
            "recipe `single` has 1 problem: node `a`: no node `test.ad` is registered"
        );
        let two = r#"{"name": "double", "nodes": [{"id": "a", "node": "test.ad"}], "outputs": {}}"#;
        assert_eq!(
            Plan::new(&recipe(two), &registry).unwrap_err().to_string(),
            "recipe `double` has 2 problems: the recipe: it gives no outputs; node `a`: no node `test.ad` is registered"
        );
    }

    #[test]
    fn running_checks_what_it_is_given_and_what_nodes_give() {
        let (registry, _) = registry();
        let mut plan = Plan::new(&recipe(IMAGE_RECIPE), &registry).unwrap();
        let compute = Compute::cpu_only().unwrap();
        let context = Context { compute: &compute };
        let err = |plan: &mut Plan, inputs: Vec<(&str, Value)>| {
            let inputs = inputs
                .into_iter()
                .map(|(n, v)| (n.to_string(), v))
                .collect();
            plan.run(&context, inputs).unwrap_err().to_string()
        };
        let rgb8 = Value::U8(Image::new(3, 1, 3, vec![0; 9]).unwrap());
        let rgb16 = Value::U16(Image::new(3, 1, 3, vec![0; 9]).unwrap());
        assert_eq!(
            err(&mut plan, vec![]),
            "recipe `measure` takes input `frame`, which wasn't given"
        );
        assert_eq!(
            err(&mut plan, vec![("frame", rgb8)]),
            "recipe `measure` takes an image of 3 channels of u16 as input `frame`, not an image of 3 channels of u8"
        );
        assert_eq!(
            err(
                &mut plan,
                vec![("frame", rgb16.clone()), ("extra", Value::Scalar(1.0))]
            ),
            "recipe `measure` has no input `extra`"
        );
        // A node's own error comes through: this frame is too small for the crop.
        let small = Value::U16(Image::new(2, 1, 3, vec![0; 6]).unwrap());
        assert!(err(&mut plan, vec![("frame", small)]).contains("runs past the 2 × 1 image"));
        // A node that gives what it didn't declare is caught.
        let lying = r#"{"name": "lies", "nodes": [{"id": "w", "node": "test.wrong"}], "outputs": {"o": "w.out"}}"#;
        let mut plan = Plan::new(&recipe(lying), &registry).unwrap();
        assert_eq!(
            err(&mut plan, vec![]),
            "node `w` (test.wrong) gave a table instead of a scalar"
        );
    }

    #[test]
    fn the_shipped_recipes_check_as_the_example_shows() {
        let registry = Registry::stock();
        let good = Recipe::from_json(include_str!("../recipes/crop-and-measure.json")).unwrap();
        assert_eq!(
            Plan::new(&good, &registry).unwrap().order(),
            ["crop", "measure"]
        );
        let broken = Recipe::from_json(include_str!("../recipes/broken.json")).unwrap();
        let problems: Vec<String> = Plan::check(&broken, &registry)
            .unwrap_err()
            .iter()
            .map(Problem::to_string)
            .collect();
        assert_eq!(
            problems,
            [
                "node `crop`, parameters: missing field `height`",
                "node `blur`: no node `tetera.gaussian` is registered",
                "node `second`, input `image`: nothing feeds it",
                "the recipe: its nodes `measure` → `again` → `measure` form a cycle",
                "node `crop_the_table`, input `image`: it takes an image, but `stats.table` gives a table",
                "output `small`: `second` has no output `picture`; its outputs are `image`",
            ]
        );
    }
}
