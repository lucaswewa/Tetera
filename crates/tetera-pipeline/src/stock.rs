//! The raw path's stock nodes (P1.08b): binning, the Bayer split, the superpixel,
//! demosaicking, dark subtraction, the flat field, resizing, colour, the display transform and
//! JPEG. Each wraps its `tetera-imaging` function, which runs on whichever backend `Compute`
//! chose.

use crate::node::{Context, Descriptor, NoParameters, Node, Step, parameters};
use crate::port::{Encoding, PortSpec, PortType, SampleType, Value};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;
use tetera_core::frame::format::Channel;
use tetera_core::{Error, Result};
use tetera_imaging::bayer::{bin as bin_mosaic, split, superpixel as superpixel_of};
use tetera_imaging::colour::colour as apply_colour;
use tetera_imaging::correct::{apply_flat, subtract_dark};
use tetera_imaging::demosaic::demosaic as demosaic_mosaic;
use tetera_imaging::display::display as apply_display;
use tetera_imaging::resize::resize as resize_image;
use tetera_imaging::{
    BinMode, Colour, ColourCorrection, Curve, Demosaic, DisplayTransform, FlatField, Image,
    JpegEncoder, Levels, Mosaic,
};
#[cfg(feature = "cuda")]
use {
    crate::port::OnDevice,
    tetera_compute::CudaBackend,
    tetera_imaging::bayer::{bin_device, superpixel_device},
    tetera_imaging::colour::{DeviceColour, colour_device},
    tetera_imaging::correct::{apply_flat_device, subtract_dark_device},
    tetera_imaging::demosaic::demosaic_device,
    tetera_imaging::display::{DeviceDisplay, display_device},
    tetera_imaging::resize::{DeviceResize, resize_device},
};

/// The GPU, for a step given values there.
#[cfg(feature = "cuda")]
fn gpu<'a>(context: &'a Context) -> Result<&'a CudaBackend> {
    context
        .compute
        .cuda()
        .ok_or_else(|| Error::internal("values on the GPU, and no GPU to run on"))
}

/// The raw path's nodes, for the registry.
pub fn nodes() -> Vec<Arc<dyn Node>> {
    vec![
        bin(),
        bayer_split(),
        superpixel(),
        demosaic(),
        dark(),
        flat(),
        resize(),
        colour(),
        display(),
        jpeg(),
    ]
}

/// A stock node: its descriptor, and how its parameters make its step.
struct Stock {
    descriptor: Descriptor,
    prepare: fn(&serde_json::Value) -> Result<Box<dyn Step>>,
}

impl Node for Stock {
    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn prepare(&self, params: &serde_json::Value) -> Result<Box<dyn Step>> {
        (self.prepare)(params)
    }
}

/// A step that is a closure.
struct FnStep<F>(F);

impl<F> Step for FnStep<F>
where
    F: FnMut(&Context, &[&Value]) -> Result<Vec<Value>> + Send,
{
    fn run(&mut self, context: &Context, inputs: &[&Value]) -> Result<Vec<Value>> {
        (self.0)(context, inputs)
    }
}

fn step(
    f: impl FnMut(&Context, &[&Value]) -> Result<Vec<Value>> + Send + 'static,
) -> Result<Box<dyn Step>> {
    Ok(Box::new(FnStep(f)))
}

/// A stock node, on either backend.
fn node(
    descriptor: Descriptor,
    prepare: fn(&serde_json::Value) -> Result<Box<dyn Step>>,
) -> Arc<dyn Node> {
    Arc::new(Stock {
        descriptor: descriptor.on_cuda(),
        prepare,
    })
}

const RGB16: PortType = PortType::Image {
    channels: Some(3),
    sample: Some(SampleType::U16),
};
const RGB8: PortType = PortType::Image {
    channels: Some(3),
    sample: Some(SampleType::U8),
};
const ANY16: PortType = PortType::Image {
    channels: None,
    sample: Some(SampleType::U16),
};
const ANY8: PortType = PortType::Image {
    channels: None,
    sample: Some(SampleType::U8),
};

fn raw(name: &'static str, summary: &'static str) -> PortSpec {
    PortSpec::new(name, PortType::Raw, summary)
}

fn invalid(message: String) -> Error {
    Error::invalid_argument(message)
}

/// What validation promised an input would be, or a bug's error.
fn given(expected: &str, value: &Value) -> Error {
    Error::internal(format!(
        "expected {expected}, was given {}",
        value.port_type()
    ))
}

fn mosaic(value: &Value) -> Result<&Mosaic> {
    match value {
        Value::Raw(m) => Ok(m),
        other => Err(given("a raw frame", other)),
    }
}

fn image16(value: &Value) -> Result<&Image<u16>> {
    match value {
        Value::U16(i) => Ok(i),
        other => Err(given("an image of u16", other)),
    }
}

fn image8(value: &Value) -> Result<&Image<u8>> {
    match value {
        Value::U8(i) => Ok(i),
        other => Err(given("an image of u8", other)),
    }
}

/// How binning combines a block's sites of one colour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum BinModeParameter {
    /// Their sum, in more bits.
    Sum,
    /// Their mean, in the frame's own bits.
    #[default]
    Mean,
}

/// `tetera.bin`'s parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BinParameters {
    /// The factor, 1 to 4.
    pub factor: u32,
    /// Their sum or their mean; the mean when left out.
    #[serde(default)]
    pub mode: BinModeParameter,
}

fn bin() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.bin",
        1,
        "A raw frame binned by 2, 3 or 4, keeping its Bayer pattern.",
    )
    .input(raw("raw", "The raw frame."))
    .output(raw("raw", "It binned."))
    .parameters::<BinParameters>();
    node(d, |params| {
        let p: BinParameters = parameters(params)?;
        if !(1..=4).contains(&p.factor) {
            return Err(invalid(format!("binning is by 1 to 4, not {}", p.factor)));
        }
        let mode = match p.mode {
            BinModeParameter::Sum => BinMode::Sum,
            BinModeParameter::Mean => BinMode::Mean,
        };
        step(move |ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let Value::DeviceRaw(m) = inputs[0] {
                let binned = bin_device(gpu(ctx)?, m, p.factor, mode)?;
                return Ok(vec![Value::DeviceRaw(OnDevice::new(binned))]);
            }
            let binned = bin_mosaic(ctx.compute, mosaic(inputs[0])?, p.factor, mode)?;
            Ok(vec![Value::Raw(binned)])
        })
    })
}

fn bayer_split() -> Arc<dyn Node> {
    let plane = |name, summary| PortSpec::new(name, PortType::image(1, SampleType::U16), summary);
    let d = Descriptor::new(
        "tetera.bayer_split",
        1,
        "A raw frame's four colour planes, each half its width and height.",
    )
    .input(raw("raw", "The raw frame."))
    .output(plane("red", "Its red sites."))
    .output(plane("green_on_red", "Its green sites on rows with red."))
    .output(plane("green_on_blue", "Its green sites on rows with blue."))
    .output(plane("blue", "Its blue sites."));
    node(d, |params| {
        let _: NoParameters = parameters(params)?;
        step(|ctx, inputs| {
            let planes = split(ctx.compute, mosaic(inputs[0])?)?;
            [
                Channel::Red,
                Channel::GreenOnRed,
                Channel::GreenOnBlue,
                Channel::Blue,
            ]
            .into_iter()
            .map(|c| {
                let data = planes.plane(c).to_vec();
                Ok(Value::U16(Image::new(
                    planes.width(),
                    planes.height(),
                    1,
                    data,
                )?))
            })
            .collect()
        })
    })
}

fn superpixel() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.superpixel",
        1,
        "RGB at half the width and height, each pixel from one 2 × 2 tile, its greens averaged.",
    )
    .input(raw("raw", "The raw frame."))
    .output(PortSpec::new("image", RGB16, "Its superpixels."));
    node(d, |params| {
        let _: NoParameters = parameters(params)?;
        step(|ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let Value::DeviceRaw(m) = inputs[0] {
                let rgb = superpixel_device(gpu(ctx)?, m)?;
                return Ok(vec![Value::DeviceU16(OnDevice::new(rgb))]);
            }
            Ok(vec![Value::U16(superpixel_of(
                ctx.compute,
                mosaic(inputs[0])?,
            )?)])
        })
    })
}

/// A demosaicking method.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DemosaicParameter {
    /// Each missing colour the mean of its nearest sites of that colour.
    Bilinear,
    /// Malvar, He and Cutler's, corrected by the site's own colour's gradient.
    #[default]
    MalvarHeCutler,
}

impl From<DemosaicParameter> for Demosaic {
    fn from(p: DemosaicParameter) -> Self {
        match p {
            DemosaicParameter::Bilinear => Demosaic::Bilinear,
            DemosaicParameter::MalvarHeCutler => Demosaic::MalvarHeCutler,
        }
    }
}

/// `tetera.demosaic`'s parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DemosaicParameters {
    /// The method; Malvar-He-Cutler when left out.
    #[serde(default)]
    pub method: DemosaicParameter,
}

fn demosaic() -> Arc<dyn Node> {
    let d = Descriptor::new("tetera.demosaic", 1, "A raw frame to RGB, in 16 bits.")
        .input(raw("raw", "The raw frame."))
        .output(PortSpec::new("image", RGB16, "Its RGB."))
        .parameters::<DemosaicParameters>();
    node(d, |params| {
        let p: DemosaicParameters = parameters(params)?;
        let method = Demosaic::from(p.method);
        step(move |ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let Value::DeviceRaw(m) = inputs[0] {
                let rgb = demosaic_device::<u16>(gpu(ctx)?, m, method)?;
                return Ok(vec![Value::DeviceU16(OnDevice::new(rgb))]);
            }
            let rgb = demosaic_mosaic::<u16>(ctx.compute, mosaic(inputs[0])?, method)?;
            Ok(vec![Value::U16(rgb)])
        })
    })
}

fn dark() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.dark",
        1,
        "A raw frame less a master dark of its layout, held at zero.",
    )
    .input(raw("raw", "The raw frame."))
    .input(raw("dark", "The master dark."))
    .output(raw("raw", "It, the dark subtracted."));
    node(d, |params| {
        let _: NoParameters = parameters(params)?;
        step(|ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let (Value::DeviceRaw(m), Value::DeviceRaw(d)) = (inputs[0], inputs[1]) {
                let out = subtract_dark_device(gpu(ctx)?, m, d.buffer())?;
                return Ok(vec![Value::DeviceRaw(OnDevice::new(out))]);
            }
            let out = subtract_dark(ctx.compute, mosaic(inputs[0])?, mosaic(inputs[1])?)?;
            Ok(vec![Value::Raw(out)])
        })
    })
}

fn flat() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.flat",
        1,
        "A raw frame evened out by the flat field of a master flat, dark-subtracted.",
    )
    .input(raw("raw", "The raw frame, dark-subtracted."))
    .input(raw("flat", "The master flat, dark-subtracted."))
    .output(raw("raw", "It, flat-fielded."));
    node(d, |params| {
        let _: NoParameters = parameters(params)?;
        // The field is made once for each master flat, not for each frame: on the GPU, for
        // each master there, known by its buffer.
        let mut field: Option<(Mosaic, FlatField)> = None;
        #[cfg(feature = "cuda")]
        let mut gains: Option<(
            OnDevice<tetera_imaging::DeviceMosaic>,
            tetera_compute::DeviceBuffer<u16>,
        )> = None;
        step(move |ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let (Value::DeviceRaw(frame), Value::DeviceRaw(master)) = (inputs[0], inputs[1]) {
                let gpu = gpu(ctx)?;
                if gains.as_ref().is_none_or(|(m, _)| m != master) {
                    let field = FlatField::new(&master.download(gpu)?)?;
                    gains = Some((master.clone(), gpu.upload(field.gains())?));
                }
                let (_, g) = gains.as_ref().expect("made above");
                let out = apply_flat_device(gpu, frame, g)?;
                return Ok(vec![Value::DeviceRaw(OnDevice::new(out))]);
            }
            let (frame, master) = (mosaic(inputs[0])?, mosaic(inputs[1])?);
            if field.as_ref().is_none_or(|(m, _)| m != master) {
                field = Some((master.clone(), FlatField::new(master)?));
            }
            let (_, f) = field.as_ref().expect("made above");
            Ok(vec![Value::Raw(apply_flat(ctx.compute, frame, f)?)])
        })
    })
}

/// `tetera.resize`'s parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResizeParameters {
    /// The width out, at least 1.
    pub width: u32,
    /// The height out, at least 1.
    pub height: u32,
}

fn resize() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.resize",
        1,
        "An image resized: area averaging to shrink, bilinear to enlarge.",
    )
    .input(PortSpec::new("image", ANY16, "The image."))
    .output(PortSpec::new("image", ANY16, "It resized.").following("image"))
    .parameters::<ResizeParameters>();
    node(d, |params| {
        let p: ResizeParameters = parameters(params)?;
        if p.width == 0 || p.height == 0 {
            return Err(invalid(format!(
                "an image resizes to at least 1 × 1, not {} × {}",
                p.width, p.height
            )));
        }
        // On the GPU, the tables for one size in, kept while that size comes.
        #[cfg(feature = "cuda")]
        let mut tables: Option<((u32, u32), DeviceResize)> = None;
        step(move |ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let Value::DeviceU16(i) = inputs[0] {
                let gpu = gpu(ctx)?;
                let size = (i.width(), i.height());
                if tables.as_ref().is_none_or(|(s, _)| *s != size) {
                    let plan = tetera_imaging::Resize::new(size.0, size.1, p.width, p.height)?;
                    tables = Some((size, DeviceResize::upload(gpu, &plan)?));
                }
                let (_, t) = tables.as_ref().expect("made above");
                return Ok(vec![Value::DeviceU16(OnDevice::new(resize_device(
                    gpu, i, t,
                )?))]);
            }
            let out = resize_image(ctx.compute, image16(inputs[0])?, p.width, p.height)?;
            Ok(vec![Value::U16(out)])
        })
    })
}

/// A channel's black and white levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LevelsParameter {
    /// The value shown as black, and below.
    pub black: u16,
    /// The value shown as white, and above.
    pub white: u16,
}

impl From<LevelsParameter> for Levels {
    fn from(l: LevelsParameter) -> Self {
        Levels {
            black: l.black,
            white: l.white,
        }
    }
}

/// A tone curve: `"linear"`, `"srgb"`, or `{"gamma": 2.2}`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum CurveParameter {
    /// As it is.
    Linear,
    /// The sRGB transfer function.
    #[default]
    Srgb,
    /// `t^(1/γ)`.
    Gamma(f64),
}

impl From<CurveParameter> for Curve {
    fn from(c: CurveParameter) -> Self {
        match c {
            CurveParameter::Linear => Curve::Linear,
            CurveParameter::Srgb => Curve::Srgb,
            CurveParameter::Gamma(g) => Curve::Gamma(g),
        }
    }
}

fn unity() -> [f64; 3] {
    [1.0; 3]
}

fn identity() -> [[f64; 3]; 3] {
    ColourCorrection::IDENTITY.matrix
}

/// `tetera.colour`'s parameters.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ColourParameters {
    /// Red's, green's and blue's gains; 1 when left out.
    #[serde(default = "unity")]
    pub white_balance: [f64; 3],
    /// The matrix to linear sRGB, row by row; the identity when left out.
    #[serde(default = "identity")]
    pub matrix: [[f64; 3]; 3],
    /// Red's, green's and blue's levels, after the matrix.
    pub levels: [LevelsParameter; 3],
    /// The curve; sRGB's when left out.
    #[serde(default)]
    pub curve: CurveParameter,
}

fn colour() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.colour",
        1,
        "RGB white-balanced, through a colour matrix, then levels and a curve, to 8 bits.",
    )
    .input(PortSpec::new("image", RGB16, "Linear RGB."))
    .output(PortSpec::new("image", RGB8, "It, for display."))
    .parameters::<ColourParameters>();
    node(d, |params| {
        let p: ColourParameters = parameters(params)?;
        let correction = ColourCorrection {
            white_balance: p.white_balance,
            matrix: p.matrix,
        };
        let colour = Colour::new(&correction, p.levels.map(Levels::from), p.curve.into())?;
        #[cfg(feature = "cuda")]
        let mut tables: Option<DeviceColour> = None;
        step(move |ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let Value::DeviceU16(i) = inputs[0] {
                let gpu = gpu(ctx)?;
                if tables.is_none() {
                    tables = Some(DeviceColour::upload(gpu, &colour)?);
                }
                let out = colour_device(gpu, i, tables.as_ref().expect("made above"))?;
                return Ok(vec![Value::DeviceU8(OnDevice::new(out))]);
            }
            Ok(vec![Value::U8(apply_colour(
                ctx.compute,
                image16(inputs[0])?,
                &colour,
            )?)])
        })
    })
}

/// `tetera.display`'s parameters.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DisplayParameters {
    /// Each channel's levels, in order; or one for every channel.
    pub levels: Vec<LevelsParameter>,
    /// The curve; sRGB's when left out.
    #[serde(default)]
    pub curve: CurveParameter,
}

fn display() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.display",
        1,
        "An image's levels and a tone curve, to 8 bits, by a table for each channel.",
    )
    .input(PortSpec::new("image", ANY16, "The image."))
    .output(PortSpec::new("image", ANY8, "It, for display.").following("image"))
    .parameters::<DisplayParameters>();
    node(d, |params| {
        let p: DisplayParameters = parameters(params)?;
        let levels: Vec<Levels> = p.levels.iter().copied().map(Levels::from).collect();
        let curve = Curve::from(p.curve);
        // Checks the levels and curve now; the tables are made for an image's channels.
        let mut transform = DisplayTransform::new(&levels, curve)?;
        #[cfg(feature = "cuda")]
        let mut tables: Option<DeviceDisplay> = None;
        step(move |ctx, inputs| {
            let PortType::Image {
                channels: Some(channels),
                sample: Some(SampleType::U16),
            } = inputs[0].port_type()
            else {
                return Err(given("an image of u16", inputs[0]));
            };
            if levels.len() != 1 && levels.len() != usize::from(channels) {
                return Err(invalid(format!(
                    "the display transform has levels for {} channels, and the image has {channels}",
                    levels.len()
                )));
            }
            // Levels for each channel made the tables already; one level for all makes them
            // for each image's channels.
            if transform.channels() != channels {
                transform = DisplayTransform::uniform(channels, levels[0], curve)?;
                #[cfg(feature = "cuda")]
                {
                    tables = None;
                }
            }
            #[cfg(feature = "cuda")]
            if let Value::DeviceU16(i) = inputs[0] {
                let gpu = gpu(ctx)?;
                if tables.is_none() {
                    tables = Some(DeviceDisplay::upload(gpu, &transform)?);
                }
                let out = display_device(gpu, i, tables.as_ref().expect("made above"))?;
                return Ok(vec![Value::DeviceU8(OnDevice::new(out))]);
            }
            Ok(vec![Value::U8(apply_display(
                ctx.compute,
                image16(inputs[0])?,
                &transform,
            )?)])
        })
    })
}

/// `tetera.jpeg`'s parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JpegParameters {
    /// The quality, 1 to 100; 80 when left out.
    #[serde(default = "eighty")]
    pub quality: u8,
}

fn eighty() -> u8 {
    80
}

fn jpeg() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.jpeg",
        1,
        "An 8-bit image, RGB or grey, as a JPEG, through libjpeg-turbo on the CPU.",
    )
    .input(PortSpec::new(
        "image",
        ANY8,
        "The image, of 1 or 3 channels.",
    ))
    .output(PortSpec::new(
        "jpeg",
        PortType::Encoded {
            format: Encoding::Jpeg,
        },
        "It as a JPEG file.",
    ))
    .parameters::<JpegParameters>();
    // libjpeg-turbo runs on the CPU, so the node does, whatever the backend.
    Arc::new(Stock {
        descriptor: d,
        prepare: |params| {
            let p: JpegParameters = parameters(params)?;
            let mut encoder = JpegEncoder::new(p.quality)?;
            step(move |_, inputs| {
                let bytes = encoder.encode(image8(inputs[0])?)?;
                Ok(vec![Value::Encoded(Encoding::Jpeg, bytes)])
            })
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Plan;
    use crate::recipe::Recipe;
    use crate::registry::Registry;
    use std::collections::BTreeMap;
    use tetera_compute::Compute;
    use tetera_imaging::pattern::{Sensor, test_pattern};
    use tetera_imaging::{Isp, IspParams};

    fn compute() -> Compute {
        Compute::cpu_only().unwrap()
    }

    /// The registry's node `id`, prepared with `params`, run once on `inputs`.
    fn run(id: &str, params: serde_json::Value, inputs: &[&Value]) -> Result<Vec<Value>> {
        let node = Registry::stock().find(id, None).unwrap();
        let mut step = node.prepare(&params)?;
        step.run(
            &Context {
                compute: &compute(),
            },
            inputs,
        )
    }

    fn refused(id: &str, params: serde_json::Value) -> String {
        Registry::stock()
            .find(id, None)
            .unwrap()
            .prepare(&params)
            .err()
            .unwrap()
            .to_string()
    }

    /// A sensor with an offset, vignetting and unequal channels, as the ISP's tests use.
    fn sensor() -> Sensor {
        Sensor {
            offset: 128,
            vignetting: 0.3,
            response: [0.7, 1.0, 0.5],
            ..Sensor::default()
        }
    }

    fn raw(width: u32, height: u32) -> Mosaic {
        sensor().expose(&test_pattern(width, height, 12)).unwrap()
    }

    /// The master dark and an even exposure, for a flat.
    fn calibration(width: u32, height: u32) -> (Mosaic, Mosaic) {
        let even = Image::new(
            width,
            height,
            3,
            [3_000u16; 3].repeat((width * height) as usize),
        )
        .unwrap();
        (
            sensor().dark(width, height).unwrap(),
            sensor().expose(&even).unwrap(),
        )
    }

    fn one(mut values: Vec<Value>) -> Value {
        assert_eq!(values.len(), 1);
        values.remove(0)
    }

    #[test]
    fn the_raw_nodes_match_the_functions_they_wrap() {
        let c = compute();
        let frame = raw(64, 48);
        let r = Value::Raw(frame.clone());
        for (mode, bin_mode) in [("mean", BinMode::Mean), ("sum", BinMode::Sum)] {
            let out = one(run(
                "tetera.bin",
                serde_json::json!({"factor": 2, "mode": mode}),
                &[&r],
            )
            .unwrap());
            assert_eq!(
                out,
                Value::Raw(bin_mosaic(&c, &frame, 2, bin_mode).unwrap()),
                "{mode}"
            );
        }
        let default = one(run("tetera.bin", serde_json::json!({"factor": 3}), &[&r]).unwrap());
        assert_eq!(
            default,
            Value::Raw(bin_mosaic(&c, &frame, 3, BinMode::Mean).unwrap()),
            "the mean by default"
        );
        let planes = split(&c, &frame).unwrap();
        let out = run("tetera.bayer_split", serde_json::json!({}), &[&r]).unwrap();
        for (value, channel) in out.iter().zip([
            Channel::Red,
            Channel::GreenOnRed,
            Channel::GreenOnBlue,
            Channel::Blue,
        ]) {
            assert_eq!(
                *value,
                Value::U16(Image::new(32, 24, 1, planes.plane(channel).to_vec()).unwrap())
            );
        }
        assert_eq!(out.len(), 4);
        let out = one(run("tetera.superpixel", serde_json::json!({}), &[&r]).unwrap());
        assert_eq!(out, Value::U16(superpixel_of(&c, &frame).unwrap()));
        for (name, method) in [
            ("bilinear", Demosaic::Bilinear),
            ("malvar_he_cutler", Demosaic::MalvarHeCutler),
        ] {
            let out = one(run(
                "tetera.demosaic",
                serde_json::json!({"method": name}),
                &[&r],
            )
            .unwrap());
            assert_eq!(
                out,
                Value::U16(demosaic_mosaic::<u16>(&c, &frame, method).unwrap()),
                "{name}"
            );
        }
        let out = one(run("tetera.demosaic", serde_json::json!({}), &[&r]).unwrap());
        let mhc = demosaic_mosaic::<u16>(&c, &frame, Demosaic::MalvarHeCutler).unwrap();
        assert_eq!(out, Value::U16(mhc), "Malvar-He-Cutler by default");
    }

    #[test]
    fn dark_and_flat_match_their_functions_the_field_made_once_per_master() {
        let c = compute();
        let frame = raw(64, 48);
        let (dark, even) = calibration(64, 48);
        let out = one(run(
            "tetera.dark",
            serde_json::json!({}),
            &[&Value::Raw(frame.clone()), &Value::Raw(dark.clone())],
        )
        .unwrap());
        assert_eq!(out, Value::Raw(subtract_dark(&c, &frame, &dark).unwrap()));
        let master = subtract_dark(&c, &even, &dark).unwrap();
        let node = Registry::stock().find("tetera.flat", None).unwrap();
        let mut step = node.prepare(&serde_json::json!({})).unwrap();
        let context = Context { compute: &c };
        let expected = apply_flat(&c, &frame, &FlatField::new(&master).unwrap()).unwrap();
        for _ in 0..2 {
            let out = one(step
                .run(
                    &context,
                    &[&Value::Raw(frame.clone()), &Value::Raw(master.clone())],
                )
                .unwrap());
            assert_eq!(out, Value::Raw(expected.clone()));
        }
        // A new master makes a new field.
        let other = subtract_dark(&c, &frame, &dark).unwrap();
        let out = one(step
            .run(
                &context,
                &[&Value::Raw(frame.clone()), &Value::Raw(other.clone())],
            )
            .unwrap());
        assert_eq!(
            out,
            Value::Raw(apply_flat(&c, &frame, &FlatField::new(&other).unwrap()).unwrap())
        );
        assert_ne!(out, Value::Raw(expected));
    }

    #[test]
    fn the_image_nodes_match_the_functions_they_wrap() {
        let c = compute();
        let rgb = demosaic_mosaic::<u16>(&c, &raw(64, 48), Demosaic::MalvarHeCutler).unwrap();
        let image = Value::U16(rgb.clone());
        let out = one(run(
            "tetera.resize",
            serde_json::json!({"width": 40, "height": 30}),
            &[&image],
        )
        .unwrap());
        assert_eq!(out, Value::U16(resize_image(&c, &rgb, 40, 30).unwrap()));
        let levels = [(0, 4095), (100, 3000), (50, 2000)];
        let params = serde_json::json!({
            "white_balance": [1.5, 1.0, 2.0],
            "matrix": [[1.2, -0.1, -0.1], [-0.1, 1.2, -0.1], [0.0, -0.2, 1.2]],
            "levels": levels.map(|(b, w)| serde_json::json!({"black": b, "white": w})),
            "curve": {"gamma": 2.2}
        });
        let correction = ColourCorrection {
            white_balance: [1.5, 1.0, 2.0],
            matrix: [[1.2, -0.1, -0.1], [-0.1, 1.2, -0.1], [0.0, -0.2, 1.2]],
        };
        let lv = levels.map(|(black, white)| Levels { black, white });
        let colour = Colour::new(&correction, lv, Curve::Gamma(2.2)).unwrap();
        let out = one(run("tetera.colour", params, &[&image]).unwrap());
        assert_eq!(out, Value::U8(apply_colour(&c, &rgb, &colour).unwrap()));
        let srgb = Colour::new(
            &ColourCorrection::IDENTITY,
            [Levels {
                black: 0,
                white: 4095,
            }; 3],
            Curve::Srgb,
        )
        .unwrap();
        let out = one(run(
            "tetera.colour",
            serde_json::json!({"levels": [{"black": 0, "white": 4095}, {"black": 0, "white": 4095}, {"black": 0, "white": 4095}]}),
            &[&image],
        )
        .unwrap());
        assert_eq!(
            out,
            Value::U8(apply_colour(&c, &rgb, &srgb).unwrap()),
            "the identity and sRGB by default"
        );
        let each = DisplayTransform::new(&lv, Curve::Linear).unwrap();
        let params = serde_json::json!({"levels": levels.map(|(b, w)| serde_json::json!({"black": b, "white": w})), "curve": "linear"});
        let out = one(run("tetera.display", params, &[&image]).unwrap());
        assert_eq!(out, Value::U8(apply_display(&c, &rgb, &each).unwrap()));
        // One level serves every channel, of any image, the tables made again when that changes.
        let grey = Image::new(4, 1, 1, vec![0u16, 1000, 2000, 4095]).unwrap();
        let node = Registry::stock().find("tetera.display", None).unwrap();
        let mut step = node
            .prepare(&serde_json::json!({"levels": [{"black": 0, "white": 4095}]}))
            .unwrap();
        let context = Context { compute: &c };
        let uniform = DisplayTransform::uniform(
            3,
            Levels {
                black: 0,
                white: 4095,
            },
            Curve::Srgb,
        )
        .unwrap();
        assert_eq!(
            one(step.run(&context, &[&image]).unwrap()),
            Value::U8(apply_display(&c, &rgb, &uniform).unwrap())
        );
        let single = DisplayTransform::uniform(
            1,
            Levels {
                black: 0,
                white: 4095,
            },
            Curve::Srgb,
        )
        .unwrap();
        assert_eq!(
            one(step.run(&context, &[&Value::U16(grey.clone())]).unwrap()),
            Value::U8(apply_display(&c, &grey, &single).unwrap())
        );
        let two =
            serde_json::json!({"levels": [{"black": 0, "white": 9}, {"black": 0, "white": 9}]});
        assert_eq!(
            run("tetera.display", two, &[&image])
                .unwrap_err()
                .to_string(),
            "the display transform has levels for 2 channels, and the image has 3"
        );
    }

    #[test]
    fn jpeg_encodes_rgb_and_grey() {
        let c = compute();
        let rgb = Colour::new(
            &ColourCorrection::IDENTITY,
            [Levels {
                black: 0,
                white: 4095,
            }; 3],
            Curve::Srgb,
        )
        .unwrap();
        let preview = apply_colour(
            &c,
            &demosaic_mosaic::<u16>(&c, &raw(64, 48), Demosaic::Bilinear).unwrap(),
            &rgb,
        )
        .unwrap();
        let out = one(run(
            "tetera.jpeg",
            serde_json::json!({"quality": 90}),
            &[&Value::U8(preview.clone())],
        )
        .unwrap());
        let Value::Encoded(Encoding::Jpeg, bytes) = &out else {
            panic!("a JPEG")
        };
        assert_eq!(&bytes[..2], [0xFF, 0xD8]);
        assert_eq!(*bytes, tetera_imaging::jpeg::encode(&preview, 90).unwrap());
        let decoded = tetera_imaging::jpeg::decode_rgb(bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (64, 48));
        let grey = Image::new(8, 8, 1, vec![128u8; 64]).unwrap();
        let out = one(run(
            "tetera.jpeg",
            serde_json::json!({}),
            &[&Value::U8(grey.clone())],
        )
        .unwrap());
        assert_eq!(
            out,
            Value::Encoded(
                Encoding::Jpeg,
                tetera_imaging::jpeg::encode(&grey, 80).unwrap()
            ),
            "80 by default"
        );
        let two = Image::new(2, 2, 2, vec![0u8; 8]).unwrap();
        assert!(run("tetera.jpeg", serde_json::json!({}), &[&Value::U8(two)]).is_err());
    }

    #[test]
    fn bad_parameters_are_refused_before_anything_runs() {
        assert_eq!(
            refused("tetera.bin", serde_json::json!({"factor": 5})),
            "binning is by 1 to 4, not 5"
        );
        assert!(refused("tetera.bin", serde_json::json!({"factor": 0})).contains("not 0"));
        assert!(
            refused(
                "tetera.bin",
                serde_json::json!({"factor": 2, "mode": "median"})
            )
            .contains("unknown variant `median`")
        );
        assert_eq!(
            refused(
                "tetera.resize",
                serde_json::json!({"width": 0, "height": 10})
            ),
            "an image resizes to at least 1 × 1, not 0 × 10"
        );
        assert!(
            refused(
                "tetera.resize",
                serde_json::json!({"width": 10, "height": 0})
            )
            .contains("not 10 × 0")
        );
        assert!(
            refused("tetera.demosaic", serde_json::json!({"method": "nearest"}))
                .contains("unknown variant `nearest`")
        );
        let levels = serde_json::json!([{"black": 0, "white": 9}, {"black": 0, "white": 9}, {"black": 0, "white": 9}]);
        assert!(
            refused(
                "tetera.colour",
                serde_json::json!({"levels": levels, "curve": {"gamma": -1.0}})
            )
            .contains("gamma")
        );
        let huge = serde_json::json!({"levels": levels, "matrix": [[100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]});
        assert!(refused("tetera.colour", huge).contains("within ±64"));
        assert!(refused("tetera.colour", serde_json::json!({})).contains("missing field `levels`"));
        assert!(
            refused("tetera.display", serde_json::json!({"levels": []}))
                .contains("1 to 4 channels")
        );
        assert_eq!(
            refused("tetera.jpeg", serde_json::json!({"quality": 0})),
            "a JPEG's quality is 1 to 100, not 0"
        );
        assert!(refused("tetera.jpeg", serde_json::json!({"quality": 101})).contains("not 101"));
        for id in [
            "tetera.bayer_split",
            "tetera.superpixel",
            "tetera.dark",
            "tetera.flat",
        ] {
            assert!(
                refused(id, serde_json::json!({"x": 1})).contains("unknown field `x`"),
                "{id}"
            );
        }
    }

    #[test]
    fn a_node_given_the_wrong_value_says_so() {
        let image = Value::U16(Image::new(1, 1, 3, vec![0; 3]).unwrap());
        assert_eq!(
            run("tetera.superpixel", serde_json::json!({}), &[&image])
                .unwrap_err()
                .to_string(),
            "expected a raw frame, was given an image of 3 channels of u16"
        );
        assert!(
            run(
                "tetera.resize",
                serde_json::json!({"width": 1, "height": 1}),
                &[&Value::Scalar(1.0)]
            )
            .unwrap_err()
            .to_string()
            .contains("expected an image of u16")
        );
        assert!(
            run("tetera.jpeg", serde_json::json!({}), &[&image])
                .unwrap_err()
                .to_string()
                .contains("expected an image of u8")
        );
    }

    #[test]
    fn the_live_chain_recipe_gives_the_isps_preview() {
        let c = compute();
        let (width, height, output) = (96, 64, (48, 32));
        let frame = raw(width, height);
        let (dark, even) = calibration(width, height);
        let mut recipe = Recipe::from_json(include_str!("../recipes/live-chain.json")).unwrap();
        let small = recipe.nodes.iter_mut().find(|n| n.id == "small").unwrap();
        small.params = serde_json::json!({"width": output.0, "height": output.1});
        let mut plan = Plan::new(&recipe, &Registry::stock()).unwrap();
        assert_eq!(
            plan.order(),
            [
                "less_dark",
                "flat_less_dark",
                "even",
                "rgb",
                "small",
                "colour",
                "jpeg"
            ]
        );
        let inputs = BTreeMap::from([
            ("raw".to_string(), Value::Raw(frame.clone())),
            ("dark".to_string(), Value::Raw(dark.clone())),
            ("flat".to_string(), Value::Raw(even.clone())),
        ]);
        let out = plan.run(&Context { compute: &c }, inputs).unwrap();
        let isp = Isp::new(
            IspParams {
                dark: Some(dark.clone()),
                flat: Some(FlatField::new(&subtract_dark(&c, &even, &dark).unwrap()).unwrap()),
                colour: ColourCorrection {
                    white_balance: [1.0 / 0.7, 1.0, 2.0],
                    ..ColourCorrection::IDENTITY
                },
                ..IspParams::preview(4_095, output.0, output.1)
            },
            *frame.layout(),
        )
        .unwrap();
        let preview = isp.run(&c, &frame).unwrap();
        assert_eq!(
            out["preview"],
            Value::U8(preview.clone()),
            "the ISP's own preview, exactly"
        );
        assert_eq!(
            out["jpeg"],
            Value::Encoded(
                Encoding::Jpeg,
                tetera_imaging::jpeg::encode(&preview, 80).unwrap()
            )
        );
        // The shipped recipe, at 1920 × 1280, checks as it is.
        let shipped = Recipe::from_json(include_str!("../recipes/live-chain.json")).unwrap();
        assert!(Plan::new(&shipped, &Registry::stock()).is_ok());
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::node::Context;
    use crate::plan::Plan;
    use crate::port::Value;
    use crate::recipe::Recipe;
    use crate::registry::Registry;
    use std::collections::BTreeMap;
    use tetera_compute::Compute;
    use tetera_imaging::Image;
    use tetera_imaging::pattern::{Sensor, test_pattern};

    #[test]
    fn on_the_gpu_every_raw_node_gives_what_it_gives_on_the_cpu() {
        let _gpu = tetera_compute::cuda_or_skip!();
        let gpu = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = Compute::cpu_only().unwrap();
        let sensor = Sensor {
            offset: 128,
            vignetting: 0.3,
            response: [0.7, 1.0, 0.5],
            ..Sensor::default()
        };
        let (w, h) = (192, 128);
        let even = Image::new(w, h, 3, [3_000u16; 3].repeat((w * h) as usize)).unwrap();
        let inputs = BTreeMap::from([
            (
                "raw".to_string(),
                Value::Raw(sensor.expose(&test_pattern(w, h, 12)).unwrap()),
            ),
            ("dark".to_string(), Value::Raw(sensor.dark(w, h).unwrap())),
            (
                "flat".to_string(),
                Value::Raw(sensor.expose(&even).unwrap()),
            ),
        ]);
        let mut recipe = Recipe::from_json(include_str!("../recipes/live-chain.json")).unwrap();
        recipe
            .nodes
            .iter_mut()
            .find(|n| n.id == "small")
            .unwrap()
            .params = serde_json::json!({"width": 96, "height": 64});
        let registry = Registry::stock();
        let run = |compute: &Compute| {
            let mut plan = Plan::new(&recipe, &registry).unwrap();
            plan.run(&Context { compute }, inputs.clone()).unwrap()
        };
        assert_eq!(run(&gpu), run(&cpu), "the live chain, node by node");
        // The nodes the chain doesn't use.
        let raw = &inputs["raw"];
        for (id, params) in [
            ("tetera.bin", serde_json::json!({"factor": 2})),
            ("tetera.bayer_split", serde_json::json!({})),
            ("tetera.superpixel", serde_json::json!({})),
            ("tetera.demosaic", serde_json::json!({"method": "bilinear"})),
        ] {
            let node = registry.find(id, None).unwrap();
            let on = |compute: &Compute| {
                node.prepare(&params)
                    .unwrap()
                    .run(&Context { compute }, &[raw])
                    .unwrap()
            };
            assert_eq!(on(&gpu), on(&cpu), "{id}");
        }
        let rgb = Value::U16(test_pattern(64, 48, 12));
        let node = registry.find("tetera.display", None).unwrap();
        let params =
            serde_json::json!({"levels": [{"black": 100, "white": 4000}], "curve": {"gamma": 2.2}});
        let on = |compute: &Compute| {
            node.prepare(&params)
                .unwrap()
                .run(&Context { compute }, &[&rgb])
                .unwrap()
        };
        assert_eq!(on(&gpu), on(&cpu), "tetera.display");
    }
}
