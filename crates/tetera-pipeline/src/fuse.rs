//! The fusion pass (P1.08c): finds the live chain in a plan, and runs P1.06b's fused kernel in
//! its place on the GPU (ADR-0011), which gives exactly what the chain's nodes give.

use crate::node::{Context, Step, parameters};
use crate::plan::{Planned, Source};
use crate::port::Value;
use crate::stock::{
    BinModeParameter, BinParameters, ColourParameters, DemosaicParameters, ResizeParameters,
};
use tetera_core::{Error, Result};
use tetera_imaging::{ColourCorrection, IspParams, Levels, Reduce};

/// The live chain found in a plan, and the fused step to run in its place.
pub(crate) struct Fusion {
    /// The chain's steps, in order; the fused step's outputs are the last one's.
    pub(crate) chain: Vec<usize>,
    /// The fused step's inputs: the raw frame, then the master dark and the master flat, each
    /// when the chain has its node.
    pub(crate) sources: Vec<Source>,
    pub(crate) step: Box<dyn Step>,
}

/// The live chain in `steps`, when there is one the fused kernel can run: `[bin →] [dark →]
/// [flat →] demosaic → resize → colour`, each step's frame or image feeding only the next.
/// Binning fuses by the mean alone, as the kernel bins; by the sum it stays a step of its own.
/// `outputs` are the recipe's, as steps and their outputs, which read what they name too.
pub(crate) fn find(steps: &[Planned], outputs: &[(usize, usize)]) -> Option<Fusion> {
    let mut readers = vec![0usize; steps.len()];
    for source in steps.iter().flat_map(|s| &s.sources) {
        if let Source::Output(j, _) = source {
            readers[*j] += 1;
        }
    }
    for (j, _) in outputs {
        readers[*j] += 1;
    }
    let is = |i: usize, id: &str| steps[i].descriptor.id == id;
    // The step whose only output feeds step `i`'s first input, and nothing else.
    let feeder = |i: usize, id: &str| match steps[i].sources.first() {
        Some(&Source::Output(j, 0)) if is(j, id) && readers[j] == 1 => Some(j),
        _ => None,
    };
    let colour = (0..steps.len()).find(|&i| is(i, "tetera.colour"))?;
    let resize = feeder(colour, "tetera.resize")?;
    let demosaic = feeder(resize, "tetera.demosaic")?;
    let mut chain = vec![colour, resize, demosaic];
    let flat = feeder(demosaic, "tetera.flat");
    let mut first = flat.unwrap_or(demosaic);
    chain.extend(flat);
    let dark = feeder(first, "tetera.dark");
    first = dark.unwrap_or(first);
    chain.extend(dark);
    let bin = feeder(first, "tetera.bin").filter(|&b| {
        parameters::<BinParameters>(&steps[b].params)
            .is_ok_and(|p| p.mode == BinModeParameter::Mean)
    });
    chain.extend(bin);
    chain.reverse();

    let reduce = match bin {
        Some(b) => match parameters::<BinParameters>(&steps[b].params).ok()?.factor {
            1 => Reduce::Full,
            factor => Reduce::Bin(factor),
        },
        None => Reduce::Full,
    };
    let c: ColourParameters = parameters(&steps[colour].params).ok()?;
    let r: ResizeParameters = parameters(&steps[resize].params).ok()?;
    let d: DemosaicParameters = parameters(&steps[demosaic].params).ok()?;
    let template = IspParams {
        reduce,
        dark: None,
        flat: None,
        demosaic: d.method.into(),
        colour: ColourCorrection {
            white_balance: c.white_balance,
            matrix: c.matrix,
        },
        levels: c.levels.map(Levels::from),
        curve: c.curve.into(),
        output: (r.width, r.height),
    };
    let mut sources = vec![steps[chain[0]].sources[0]];
    sources.extend(dark.map(|i| steps[i].sources[1]));
    sources.extend(flat.map(|i| steps[i].sources[1]));
    Some(Fusion {
        chain,
        sources,
        step: Box::new(FusedStep {
            template,
            dark: dark.is_some(),
            flat: flat.is_some(),
            #[cfg(feature = "cuda")]
            tables: None,
        }),
    })
}

/// The fused ISP: the chain's parameters, and its tables on the GPU, made again when the raw
/// frame's layout, the master dark or the master flat changes.
struct FusedStep {
    template: IspParams,
    dark: bool,
    flat: bool,
    #[cfg(feature = "cuda")]
    tables: Option<Tables>,
}

#[cfg(feature = "cuda")]
type Master = crate::port::OnDevice<tetera_imaging::DeviceMosaic>;

/// The tables made, and what they were made for.
#[cfg(feature = "cuda")]
struct Tables {
    layout: tetera_imaging::MosaicLayout,
    dark: Option<Master>,
    flat: Option<Master>,
    isp: tetera_imaging::Isp,
    device: tetera_imaging::isp::DeviceIsp,
}

impl Step for FusedStep {
    #[cfg(feature = "cuda")]
    fn run(&mut self, context: &Context, inputs: &[&Value]) -> Result<Vec<Value>> {
        use crate::port::OnDevice;
        let gpu = context
            .compute
            .cuda()
            .ok_or_else(|| Error::internal("the fused ISP runs on the GPU"))?;
        let on_device = |v: &Value| match v {
            Value::DeviceRaw(m) => Ok(m.clone()),
            other => Err(Error::internal(format!(
                "the fused ISP was given {} on the host",
                other.port_type()
            ))),
        };
        let raw = on_device(inputs[0])?;
        let mut rest = inputs[1..].iter();
        let mut master = |wanted: bool| {
            wanted
                .then(|| on_device(rest.next().expect("an input for each master")))
                .transpose()
        };
        let (dark, flat) = (master(self.dark)?, master(self.flat)?);
        let current = self
            .tables
            .as_ref()
            .is_some_and(|t| t.layout == *raw.layout() && t.dark == dark && t.flat == flat);
        if !current {
            let params = IspParams {
                dark: dark.as_ref().map(|d| d.download(gpu)).transpose()?,
                flat: flat
                    .as_ref()
                    .map(|f| tetera_imaging::FlatField::new(&f.download(gpu)?))
                    .transpose()?,
                ..self.template.clone()
            };
            let isp = tetera_imaging::Isp::new(params, *raw.layout())?;
            let device = tetera_imaging::isp::DeviceIsp::upload(gpu, &isp)?;
            self.tables = Some(Tables {
                layout: *raw.layout(),
                dark,
                flat,
                isp,
                device,
            });
        }
        let t = self.tables.as_ref().expect("made above");
        let preview = t.device.run_fused(gpu, &t.isp, &raw)?;
        Ok(vec![Value::DeviceU8(OnDevice::new(preview))])
    }

    #[cfg(not(feature = "cuda"))]
    fn run(&mut self, _: &Context, _: &[&Value]) -> Result<Vec<Value>> {
        let _ = (&self.template, self.dark, self.flat);
        Err(Error::internal(
            "the fused ISP runs on the GPU, which this build hasn't",
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::plan::Plan;
    use crate::recipe::Recipe;
    use crate::registry::Registry;

    /// The nodes a recipe of `nodes` fuses, its inputs a raw frame, a dark and a flat.
    fn fused(nodes: serde_json::Value, outputs: serde_json::Value) -> Vec<String> {
        let recipe = serde_json::json!({
            "name": "test",
            "inputs": {"raw": {"kind": "raw"}, "dark": {"kind": "raw"}, "flat": {"kind": "raw"}},
            "nodes": nodes,
            "outputs": outputs,
        });
        let recipe = Recipe::from_json(&recipe.to_string()).unwrap();
        let plan = Plan::new(&recipe, &Registry::stock()).unwrap();
        plan.fused().iter().map(ToString::to_string).collect()
    }

    fn colour() -> serde_json::Value {
        serde_json::json!({"levels": [{"black": 0, "white": 4095}, {"black": 0, "white": 4095}, {"black": 0, "white": 4095}]})
    }

    /// The chain from `first`'s raw frame: demosaic, resize, colour.
    fn tail(first: &str) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({"id": "rgb", "node": "tetera.demosaic", "inputs": {"raw": first}}),
            serde_json::json!({"id": "small", "node": "tetera.resize", "params": {"width": 8, "height": 6}, "inputs": {"image": "rgb.image"}}),
            serde_json::json!({"id": "colour", "node": "tetera.colour", "params": colour(), "inputs": {"image": "small.image"}}),
        ]
    }

    fn with(head: Vec<serde_json::Value>, first: &str) -> serde_json::Value {
        serde_json::Value::Array(head.into_iter().chain(tail(first)).collect())
    }

    fn preview() -> serde_json::Value {
        serde_json::json!({"preview": "colour.image"})
    }

    #[test]
    fn the_live_chain_fuses_whole() {
        let recipe = Recipe::from_json(include_str!("../recipes/live-chain.json")).unwrap();
        let plan = Plan::new(&recipe, &Registry::stock()).unwrap();
        assert_eq!(
            plan.fused(),
            ["less_dark", "even", "rgb", "small", "colour"]
        );
    }

    #[test]
    fn the_chain_fuses_with_or_without_its_optional_steps() {
        assert_eq!(
            fused(with(vec![], "raw"), preview()),
            ["rgb", "small", "colour"]
        );
        let dark = serde_json::json!({"id": "less_dark", "node": "tetera.dark", "inputs": {"raw": "raw", "dark": "dark"}});
        assert_eq!(
            fused(with(vec![dark.clone()], "less_dark.raw"), preview()),
            ["less_dark", "rgb", "small", "colour"]
        );
        let flat = serde_json::json!({"id": "even", "node": "tetera.flat", "inputs": {"raw": "raw", "flat": "flat"}});
        assert_eq!(
            fused(with(vec![flat], "even.raw"), preview()),
            ["even", "rgb", "small", "colour"]
        );
        let mean = serde_json::json!({"id": "bin", "node": "tetera.bin", "params": {"factor": 2}, "inputs": {"raw": "raw"}});
        let dark_after = serde_json::json!({"id": "less_dark", "node": "tetera.dark", "inputs": {"raw": "bin.raw", "dark": "dark"}});
        assert_eq!(
            fused(
                with(vec![mean, dark_after.clone()], "less_dark.raw"),
                preview()
            ),
            ["bin", "less_dark", "rgb", "small", "colour"]
        );
        // The kernel bins by the mean: binning by the sum stays a step of its own.
        let sum = serde_json::json!({"id": "bin", "node": "tetera.bin", "params": {"factor": 2, "mode": "sum"}, "inputs": {"raw": "raw"}});
        assert_eq!(
            fused(with(vec![sum, dark_after], "less_dark.raw"), preview()),
            ["less_dark", "rgb", "small", "colour"]
        );
    }

    #[test]
    fn a_chain_whose_middle_is_read_elsewhere_isnt_fused() {
        let also_rgb = serde_json::json!({"preview": "colour.image", "rgb": "rgb.image"});
        assert!(fused(with(vec![], "raw"), also_rgb).is_empty());
        let also_small = serde_json::json!({"preview": "colour.image", "small": "small.image"});
        assert!(fused(with(vec![], "raw"), also_small).is_empty());
        // A step outside the chain reading its middle.
        let mut nodes = with(vec![], "raw").as_array().unwrap().clone();
        nodes.push(serde_json::json!({"id": "stats", "node": "tetera.statistics", "inputs": {"image": "rgb.image"}}));
        let outputs = serde_json::json!({"preview": "colour.image", "stats": "stats.table"});
        assert!(fused(serde_json::Value::Array(nodes), outputs).is_empty());
        // A dark read elsewhere stays out, and the chain fuses from the demosaic.
        let dark = serde_json::json!({"id": "less_dark", "node": "tetera.dark", "inputs": {"raw": "raw", "dark": "dark"}});
        let outputs = serde_json::json!({"preview": "colour.image", "dark": "less_dark.raw"});
        assert_eq!(
            fused(with(vec![dark], "less_dark.raw"), outputs),
            ["rgb", "small", "colour"]
        );
    }

    #[test]
    fn a_recipe_without_the_chain_fuses_nothing() {
        let display = serde_json::json!([
            {"id": "rgb", "node": "tetera.demosaic", "inputs": {"raw": "raw"}},
            {"id": "show", "node": "tetera.display", "params": {"levels": [{"black": 0, "white": 4095}]}, "inputs": {"image": "rgb.image"}}
        ]);
        assert!(fused(display, serde_json::json!({"out": "show.image"})).is_empty());
        let no_resize = serde_json::json!([
            {"id": "rgb", "node": "tetera.demosaic", "inputs": {"raw": "raw"}},
            {"id": "colour", "node": "tetera.colour", "params": colour(), "inputs": {"image": "rgb.image"}}
        ]);
        assert!(fused(no_resize, preview()).is_empty());
        let superpixel = serde_json::json!([
            {"id": "rgb", "node": "tetera.superpixel", "inputs": {"raw": "raw"}},
            {"id": "small", "node": "tetera.resize", "params": {"width": 8, "height": 6}, "inputs": {"image": "rgb.image"}},
            {"id": "colour", "node": "tetera.colour", "params": colour(), "inputs": {"image": "small.image"}}
        ]);
        assert!(fused(superpixel, preview()).is_empty());
    }
}
