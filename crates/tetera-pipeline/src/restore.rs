//! The denoise and deconvolution nodes (P1.08d): temporal averaging, the bilateral filter and
//! Richardson–Lucy, each wrapping its `tetera-imaging` function.

use crate::node::{Descriptor, Node, parameters};
use crate::port::{PortSpec, PortType, SampleType, Value};
use crate::stock::{ANY16, given, invalid, mosaic, node, raw, step};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;
use tetera_imaging::deconvolve::richardson_lucy as deconvolve;
use tetera_imaging::denoise::bilateral as filter;
use tetera_imaging::{Bilateral, Image, Optics, Psf, RichardsonLucy, TemporalMean, Volume};
#[cfg(feature = "cuda")]
use {
    crate::port::OnDevice,
    crate::stock::gpu,
    tetera_imaging::denoise::{DeviceBilateral, DeviceTemporalMean, bilateral_device},
};

/// The denoise and deconvolution nodes, for the registry.
pub fn nodes() -> Vec<Arc<dyn Node>> {
    vec![temporal_mean(), bilateral(), richardson_lucy()]
}

/// `tetera.temporal_mean`'s parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TemporalParameters {
    /// The frames averaged, 1 to 64.
    pub window: u32,
}

fn temporal_mean() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.temporal_mean",
        1,
        "The mean of the last frames, exactly; a new layout starts it again.",
    )
    .input(raw("raw", "The raw frame."))
    .output(raw("raw", "The mean of it and the frames before it."))
    .parameters::<TemporalParameters>()
    .stateful();
    node(d, |params| {
        let p: TemporalParameters = parameters(params)?;
        let mut host = TemporalMean::new(p.window)?;
        #[cfg(feature = "cuda")]
        let mut device = DeviceTemporalMean::new(p.window)?;
        step(move |ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let Value::DeviceRaw(m) = inputs[0] {
                let mean = device.push(gpu(ctx)?, m, false)?;
                return Ok(vec![Value::DeviceRaw(OnDevice::new(mean))]);
            }
            let mean = host.push_cpu(ctx.compute.cpu(), mosaic(inputs[0])?, false)?;
            Ok(vec![Value::Raw(mean)])
        })
    })
}

/// `tetera.bilateral`'s parameters.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BilateralParameters {
    /// The spatial σ, in pixels; the window reaches 2σ, at most 7.
    pub sigma_spatial: f64,
    /// The range σ, in the image's values: differences much above it are kept as edges.
    pub sigma_range: f64,
}

fn bilateral() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.bilateral",
        1,
        "The bilateral filter: noise smoothed, edges higher than the range σ kept.",
    )
    .input(PortSpec::new("image", ANY16, "The image, grey or RGB."))
    .output(PortSpec::new("image", ANY16, "It filtered.").following("image"))
    .parameters::<BilateralParameters>();
    node(d, |params| {
        let p: BilateralParameters = parameters(params)?;
        let f = Bilateral::new(p.sigma_spatial, p.sigma_range)?;
        #[cfg(feature = "cuda")]
        let mut tables: Option<DeviceBilateral> = None;
        step(move |ctx, inputs| {
            #[cfg(feature = "cuda")]
            if let Value::DeviceU16(i) = inputs[0] {
                let gpu = gpu(ctx)?;
                if tables.is_none() {
                    tables = Some(DeviceBilateral::upload(gpu, &f)?);
                }
                let out = bilateral_device(gpu, i, tables.as_ref().expect("made above"))?;
                return Ok(vec![Value::DeviceU16(OnDevice::new(out))]);
            }
            let image = match inputs[0] {
                Value::U16(i) => i,
                other => return Err(given("an image of u16", other)),
            };
            Ok(vec![Value::U16(filter(ctx.compute, image, &f)?)])
        })
    })
}

fn one() -> f64 {
    1.0
}

/// `tetera.richardson_lucy`'s parameters: the optics for the PSF, and the iterations.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeconvolveParameters {
    /// The objective's numerical aperture.
    pub na: f64,
    /// The wavelength, in µm.
    pub wavelength_um: f64,
    /// A pixel's size at the sample, in µm.
    pub pixel_um: f64,
    /// The medium's refractive index; 1, a dry objective's, when left out.
    #[serde(default = "one")]
    pub refractive_index: f64,
    /// The iterations, 1 to 1,000.
    pub iterations: u32,
    /// Total variation's weight, about 0.002; 0, none, when left out.
    #[serde(default)]
    pub total_variation: f32,
}

fn richardson_lucy() -> Arc<dyn Node> {
    let d = Descriptor::new(
        "tetera.richardson_lucy",
        1,
        "Richardson–Lucy deconvolution by the widefield PSF of the optics given.",
    )
    .input(PortSpec::new(
        "image",
        PortType::Image {
            channels: Some(1),
            sample: None,
        },
        "A grey image.",
    ))
    .output(PortSpec::new(
        "image",
        PortType::image(1, SampleType::F32),
        "It deconvolved, in floats.",
    ))
    .parameters::<DeconvolveParameters>();
    // It takes its image on the host; on a GPU, `tetera-imaging` deconvolves it there, with its
    // own copies.
    Arc::new(crate::stock::Stock::new(d, |params| {
        let p: DeconvolveParameters = parameters(params)?;
        if !(1..=1_000).contains(&p.iterations) {
            return Err(invalid(format!(
                "Richardson–Lucy runs 1 to 1,000 iterations, not {}",
                p.iterations
            )));
        }
        let optics = Optics {
            na: p.na,
            wavelength_um: p.wavelength_um,
            pixel_um: p.pixel_um,
            refractive_index: p.refractive_index,
        };
        let psf = Psf::widefield_2d(&optics)?;
        let settings = RichardsonLucy::new(p.iterations).with_total_variation(p.total_variation);
        if !(settings.total_variation.is_finite() && settings.total_variation >= 0.0) {
            return Err(invalid(format!(
                "total variation's weight is 0 or more, not {}",
                p.total_variation
            )));
        }
        step(move |ctx, inputs| {
            let (w, h, values): (u32, u32, Vec<f32>) = match inputs[0] {
                Value::U8(i) => (
                    i.width(),
                    i.height(),
                    i.data().iter().map(|&v| f32::from(v)).collect(),
                ),
                Value::U16(i) => (
                    i.width(),
                    i.height(),
                    i.data().iter().map(|&v| f32::from(v)).collect(),
                ),
                Value::F32(i) => (i.width(), i.height(), i.data().to_vec()),
                other => return Err(given("a grey image", other)),
            };
            let out = deconvolve(ctx.compute, &Volume::new(w, h, 1, values)?, &psf, &settings)?;
            Ok(vec![Value::F32(Image::new(
                w,
                h,
                1,
                out.values().to_vec(),
            )?)])
        })
    }))
}

#[cfg(test)]
mod tests {
    use crate::node::Context;
    use crate::port::{SampleType, Value};
    use crate::registry::Registry;
    use tetera_compute::Compute;
    use tetera_imaging::deconvolve::richardson_lucy_cpu;
    use tetera_imaging::denoise::bilateral_cpu;
    use tetera_imaging::pattern::{Sensor, noisy, test_pattern};
    use tetera_imaging::{
        Bilateral, Image, Mosaic, Optics, Psf, RichardsonLucy, TemporalMean, Volume,
    };

    fn prepare(id: &str, params: serde_json::Value) -> Box<dyn crate::node::Step> {
        Registry::stock()
            .find(id, None)
            .unwrap()
            .prepare(&params)
            .unwrap()
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

    #[test]
    fn temporal_averaging_keeps_its_window_from_run_to_run() {
        let compute = Compute::cpu_only().unwrap();
        let context = Context { compute: &compute };
        let clean = Sensor::default().expose(&test_pattern(32, 16, 12)).unwrap();
        let frames: Vec<Mosaic> = (0..5)
            .map(|seed| {
                Mosaic::new(*clean.layout(), noisy(clean.sites(), 50.0, 4_095, seed)).unwrap()
            })
            .collect();
        let mut step = prepare("tetera.temporal_mean", serde_json::json!({"window": 3}));
        let mut expected = TemporalMean::new(3).unwrap();
        for frame in &frames {
            let out = step.run(&context, &[&Value::Raw(frame.clone())]).unwrap();
            assert_eq!(
                out,
                [Value::Raw(
                    expected.push_cpu(compute.cpu(), frame, false).unwrap()
                )]
            );
        }
        assert!(
            refused("tetera.temporal_mean", serde_json::json!({"window": 0})).contains("1 to 64")
        );
        assert!(
            refused("tetera.temporal_mean", serde_json::json!({"window": 65})).contains("not 65")
        );
    }

    #[test]
    fn the_bilateral_filter_matches_its_function() {
        let compute = Compute::cpu_only().unwrap();
        let image = Image::new(
            24,
            16,
            3,
            noisy(test_pattern(24, 16, 12).data(), 60.0, 4_095, 4),
        )
        .unwrap();
        let mut step = prepare(
            "tetera.bilateral",
            serde_json::json!({"sigma_spatial": 1.5, "sigma_range": 120.0}),
        );
        let out = step
            .run(
                &Context { compute: &compute },
                &[&Value::U16(image.clone())],
            )
            .unwrap();
        let expected =
            bilateral_cpu(compute.cpu(), &image, &Bilateral::new(1.5, 120.0).unwrap()).unwrap();
        assert_eq!(out, [Value::U16(expected)]);
        assert!(
            refused(
                "tetera.bilateral",
                serde_json::json!({"sigma_spatial": 0.0, "sigma_range": 1.0})
            )
            .len()
                > 10
        );
        assert!(
            refused(
                "tetera.bilateral",
                serde_json::json!({"sigma_spatial": 1.0})
            )
            .contains("missing field `sigma_range`")
        );
        let bytes = Value::U8(Image::new(1, 1, 1, vec![0u8]).unwrap());
        assert!(
            step.run(&Context { compute: &compute }, &[&bytes])
                .unwrap_err()
                .to_string()
                .contains("expected an image of u16")
        );
    }

    #[test]
    fn richardson_lucy_deconvolves_any_grey_image_by_its_optics() {
        let compute = Compute::cpu_only().unwrap();
        let params =
            serde_json::json!({"na": 0.5, "wavelength_um": 0.55, "pixel_um": 0.2, "iterations": 5});
        let mut step = prepare("tetera.richardson_lucy", params);
        let optics = Optics {
            na: 0.5,
            wavelength_um: 0.55,
            pixel_um: 0.2,
            refractive_index: 1.0,
        };
        let psf = Psf::widefield_2d(&optics).unwrap();
        let grey: Vec<u16> = test_pattern(32, 24, 12)
            .data()
            .iter()
            .step_by(3)
            .copied()
            .collect();
        let volume = Volume::new(32, 24, 1, grey.iter().map(|&v| f32::from(v)).collect()).unwrap();
        let expected =
            richardson_lucy_cpu(compute.cpu(), &volume, &psf, &RichardsonLucy::new(5)).unwrap();
        let expected = Value::F32(Image::new(32, 24, 1, expected.values().to_vec()).unwrap());
        let context = Context { compute: &compute };
        let as_u16 = Value::U16(Image::new(32, 24, 1, grey.clone()).unwrap());
        assert_eq!(
            step.run(&context, &[&as_u16]).unwrap(),
            std::slice::from_ref(&expected)
        );
        let as_f32 = Value::F32(
            Image::new(32, 24, 1, grey.iter().map(|&v| f32::from(v)).collect()).unwrap(),
        );
        assert_eq!(step.run(&context, &[&as_f32]).unwrap(), [expected]);
        let small: Vec<u8> = grey.iter().map(|&v| (v >> 4) as u8).collect();
        let as_u8 = Value::U8(Image::new(32, 24, 1, small.clone()).unwrap());
        let volume8 =
            Volume::new(32, 24, 1, small.iter().map(|&v| f32::from(v)).collect()).unwrap();
        let expected8 =
            richardson_lucy_cpu(compute.cpu(), &volume8, &psf, &RichardsonLucy::new(5)).unwrap();
        assert_eq!(
            step.run(&context, &[&as_u8]).unwrap(),
            [Value::F32(
                Image::new(32, 24, 1, expected8.values().to_vec()).unwrap()
            )]
        );
        let out = step.run(&context, &[&as_u16]).unwrap();
        assert_eq!(
            out[0].port_type(),
            crate::port::PortType::image(1, SampleType::F32)
        );
        // With total variation, and the medium's index.
        let tv = serde_json::json!({"na": 1.2, "wavelength_um": 0.55, "pixel_um": 0.1, "refractive_index": 1.5, "iterations": 3, "total_variation": 0.002});
        let optics = Optics {
            na: 1.2,
            wavelength_um: 0.55,
            pixel_um: 0.1,
            refractive_index: 1.5,
        };
        let psf = Psf::widefield_2d(&optics).unwrap();
        let settings = RichardsonLucy::new(3).with_total_variation(0.002);
        let expected = richardson_lucy_cpu(compute.cpu(), &volume, &psf, &settings).unwrap();
        let out = prepare("tetera.richardson_lucy", tv)
            .run(&context, &[&as_u16])
            .unwrap();
        assert_eq!(
            out,
            [Value::F32(
                Image::new(32, 24, 1, expected.values().to_vec()).unwrap()
            )]
        );
        let base = |extra: serde_json::Value| {
            let mut p = serde_json::json!({"na": 0.5, "wavelength_um": 0.55, "pixel_um": 0.2, "iterations": 5});
            p.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            p
        };
        assert_eq!(
            refused(
                "tetera.richardson_lucy",
                base(serde_json::json!({"iterations": 0}))
            ),
            "Richardson–Lucy runs 1 to 1,000 iterations, not 0"
        );
        assert!(
            refused(
                "tetera.richardson_lucy",
                base(serde_json::json!({"iterations": 1_001}))
            )
            .contains("not 1001")
        );
        assert!(
            refused(
                "tetera.richardson_lucy",
                base(serde_json::json!({"total_variation": -0.1}))
            )
            .contains("0 or more")
        );
        assert!(
            refused(
                "tetera.richardson_lucy",
                base(serde_json::json!({"na": 1.5}))
            )
            .contains("NA")
        );
        let rgb = Value::U16(Image::new(1, 1, 3, vec![0; 3]).unwrap());
        assert!(
            step.run(&context, &[&Value::Scalar(1.0)])
                .unwrap_err()
                .to_string()
                .contains("a grey image")
        );
        assert!(step.run(&context, &[&rgb]).is_err());
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::node::Context;
    use crate::port::Value;
    use crate::registry::Registry;
    use tetera_compute::Compute;
    use tetera_imaging::pattern::{Sensor, noisy, test_pattern};
    use tetera_imaging::{Image, Mosaic};

    #[test]
    fn on_the_gpu_the_denoise_nodes_give_what_they_give_on_the_cpu() {
        let _gpu = tetera_compute::cuda_or_skip!();
        let gpu = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = Compute::cpu_only().unwrap();
        let device = gpu.cuda().unwrap();
        let registry = Registry::stock();
        let clean = Sensor::default().expose(&test_pattern(64, 32, 12)).unwrap();
        let node = registry.find("tetera.temporal_mean", None).unwrap();
        let params = serde_json::json!({"window": 4});
        let (mut on_gpu, mut on_cpu) = (
            node.prepare(&params).unwrap(),
            node.prepare(&params).unwrap(),
        );
        for seed in 0..6 {
            let frame = Value::Raw(
                Mosaic::new(*clean.layout(), noisy(clean.sites(), 40.0, 4_095, seed)).unwrap(),
            );
            let up = frame.upload(device).unwrap().unwrap().0;
            let out = on_gpu.run(&Context { compute: &gpu }, &[&up]).unwrap();
            let down = out[0].download(device).unwrap().unwrap().0;
            assert_eq!(
                down,
                on_cpu.run(&Context { compute: &cpu }, &[&frame]).unwrap()[0],
                "frame {seed}"
            );
        }
        let node = registry.find("tetera.bilateral", None).unwrap();
        let params = serde_json::json!({"sigma_spatial": 1.0, "sigma_range": 100.0});
        let (mut on_gpu, mut on_cpu) = (
            node.prepare(&params).unwrap(),
            node.prepare(&params).unwrap(),
        );
        for seed in 0..2 {
            let image = Value::U16(
                Image::new(
                    48,
                    32,
                    3,
                    noisy(test_pattern(48, 32, 12).data(), 50.0, 4_095, seed),
                )
                .unwrap(),
            );
            let up = image.upload(device).unwrap().unwrap().0;
            let out = on_gpu.run(&Context { compute: &gpu }, &[&up]).unwrap();
            let down = out[0].download(device).unwrap().unwrap().0;
            assert_eq!(
                down,
                on_cpu.run(&Context { compute: &cpu }, &[&image]).unwrap()[0]
            );
        }
    }
}
