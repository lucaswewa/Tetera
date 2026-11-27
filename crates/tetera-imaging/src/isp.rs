//! The ISP: a raw frame to an 8-bit preview, stage by stage, on either backend.
//!
//! The chain: crop or bin → dark → flat → demosaic → resize → white balance and CCM →
//! levels and the transfer curve → 8 bits. It resizes before colour, in linear light: the two
//! are linear, so they commute but for rounding, and colour then works on the preview's
//! pixels rather than the frame's. This unfused chain is the reference for the fused kernel
//! (P1.06b).
//!
//! [`IspParams`] holds the calibration inputs; [`Isp`] prepares them for one raw layout, once;
//! [`DeviceIsp`] holds the prepared tables on the GPU.

use crate::bayer::{BinMode, bin_cpu};
use crate::colour::{Colour, ColourCorrection, colour_cpu};
use crate::correct::{FlatField, apply_flat_cpu, subtract_dark_cpu};
use crate::demosaic::{Demosaic, demosaic_cpu};
use crate::display::{Curve, Levels};
use crate::image::{Image, Mosaic, MosaicLayout};
use crate::resize::{Resize, resize_cpu};
use tetera_compute::{Compute, CpuBackend};
use tetera_core::{Error, Rect, Result};

/// How the chain reduces a raw frame first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduce {
    /// The whole frame.
    Full,
    /// A region of it, its pattern moved for an odd column or row.
    Crop(Rect),
    /// Binned by 2, 3 or 4, by the mean, keeping the Bayer layout.
    Bin(u32),
}

/// The ISP's inputs: what to reduce, the calibrations, and the output.
#[derive(Debug, Clone, PartialEq)]
pub struct IspParams {
    /// How to reduce a raw frame first.
    pub reduce: Reduce,
    /// The master dark, at the reduced layout, or none.
    pub dark: Option<Mosaic>,
    /// The flat field, at the reduced layout, or none.
    pub flat: Option<FlatField>,
    /// The demosaicking method.
    pub demosaic: Demosaic,
    /// White balance and the colour matrix.
    pub colour: ColourCorrection,
    /// Black and white levels for red, green and blue, after colour.
    pub levels: [Levels; 3],
    /// The transfer curve.
    pub curve: Curve,
    /// The preview's width and height.
    pub output: (u32, u32),
}

impl IspParams {
    /// The chain's usual parameters for a mosaic of `max`, to `width × height`: the whole
    /// frame, no calibrations, Malvar-He-Cutler, sRGB.
    pub fn preview(max: u16, width: u32, height: u32) -> Self {
        Self {
            reduce: Reduce::Full,
            dark: None,
            flat: None,
            demosaic: Demosaic::MalvarHeCutler,
            colour: ColourCorrection::IDENTITY,
            levels: [Levels {
                black: 0,
                white: max,
            }; 3],
            curve: Curve::Srgb,
            output: (width, height),
        }
    }

    /// The layout the chain works at, after reducing a raw frame of `raw`.
    pub fn reduced(&self, raw: &MosaicLayout) -> Result<MosaicLayout> {
        match self.reduce {
            Reduce::Full => Ok(*raw),
            Reduce::Crop(roi) => {
                let whole = Rect::new(0, 0, raw.width, raw.height);
                if roi.is_empty() || !whole.contains_rect(&roi) {
                    return Err(Error::invalid_argument(format!(
                        "{roi:?} isn't inside the {} × {} frame",
                        raw.width, raw.height
                    )));
                }
                Ok(MosaicLayout {
                    width: roi.width,
                    height: roi.height,
                    pattern: raw.pattern.cropped(roi.x, roi.y),
                    ..*raw
                })
            }
            Reduce::Bin(factor) => raw.binned(factor, BinMode::Mean),
        }
    }
}

/// The chain, prepared for raw frames of one layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Isp {
    params: IspParams,
    raw: MosaicLayout,
    reduced: MosaicLayout,
    resize: Resize,
    colour: Colour,
}

impl Isp {
    /// `params` prepared for raw frames of `raw`: the calibrations checked against the reduced
    /// layout, the resize's tables and colour's made.
    pub fn new(params: IspParams, raw: MosaicLayout) -> Result<Self> {
        let reduced = params.reduced(&raw)?;
        let shape = |l: &MosaicLayout| (l.width, l.height, l.pattern);
        if let Some(dark) = &params.dark
            && shape(dark.layout()) != shape(&reduced)
        {
            return Err(Error::invalid_argument(format!(
                "the dark is {} × {} {:?}, but the chain works at {} × {} {:?}",
                dark.layout().width,
                dark.layout().height,
                dark.layout().pattern,
                reduced.width,
                reduced.height,
                reduced.pattern
            )));
        }
        if let Some(flat) = &params.flat
            && shape(flat.layout()) != shape(&reduced)
        {
            return Err(Error::invalid_argument(format!(
                "the flat field is {} × {} {:?}, but the chain works at {} × {} {:?}",
                flat.layout().width,
                flat.layout().height,
                flat.layout().pattern,
                reduced.width,
                reduced.height,
                reduced.pattern
            )));
        }
        let min = params.demosaic.min_size();
        if reduced.width < min || reduced.height < min {
            return Err(Error::invalid_argument(format!(
                "{:?} needs at least {min} × {min}, but the chain works at {} × {}",
                params.demosaic, reduced.width, reduced.height
            )));
        }
        let resize = Resize::new(
            reduced.width,
            reduced.height,
            params.output.0,
            params.output.1,
        )?;
        let colour = Colour::new(&params.colour, params.levels, params.curve)?;
        Ok(Self {
            params,
            raw,
            reduced,
            resize,
            colour,
        })
    }

    /// Its parameters.
    pub fn params(&self) -> &IspParams {
        &self.params
    }

    /// The raw frames it takes.
    pub fn raw(&self) -> &MosaicLayout {
        &self.raw
    }

    /// The layout it works at, after reducing.
    pub fn reduced(&self) -> &MosaicLayout {
        &self.reduced
    }

    /// Its resize.
    pub fn resize(&self) -> &Resize {
        &self.resize
    }

    /// Its colour.
    pub fn colour(&self) -> &Colour {
        &self.colour
    }

    fn check_raw(&self, raw: &MosaicLayout) -> Result<()> {
        if *raw != self.raw {
            return Err(Error::invalid_argument(format!(
                "the ISP was prepared for {} × {} {:?} of {} bits, not {} × {} {:?} of {}",
                self.raw.width,
                self.raw.height,
                self.raw.pattern,
                self.raw.significant_bits,
                raw.width,
                raw.height,
                raw.pattern,
                raw.significant_bits
            )));
        }
        Ok(())
    }

    /// The first stage on the CPU: the raw frame reduced.
    pub fn reduce_cpu(&self, cpu: &CpuBackend, raw: &Mosaic) -> Result<Mosaic> {
        self.check_raw(raw.layout())?;
        match self.params.reduce {
            Reduce::Full => Ok(raw.clone()),
            Reduce::Crop(roi) => raw.crop(roi),
            Reduce::Bin(factor) => bin_cpu(cpu, raw, factor, BinMode::Mean),
        }
    }

    /// The whole chain on the CPU: the reference.
    pub fn run_cpu(&self, cpu: &CpuBackend, raw: &Mosaic) -> Result<Image<u8>> {
        let mut mosaic = self.reduce_cpu(cpu, raw)?;
        if let Some(dark) = &self.params.dark {
            mosaic = subtract_dark_cpu(cpu, &mosaic, dark)?;
        }
        if let Some(flat) = &self.params.flat {
            mosaic = apply_flat_cpu(cpu, &mosaic, flat)?;
        }
        let rgb = demosaic_cpu::<u16>(cpu, &mosaic, self.params.demosaic)?;
        let small = resize_cpu(cpu, &rgb, &self.resize)?;
        colour_cpu(cpu, &small, &self.colour)
    }

    /// The whole chain, on whichever backend `compute` chose. On CUDA, a crop is made on the
    /// host, before the upload; the rest on the device.
    pub fn run(&self, compute: &Compute, raw: &Mosaic) -> Result<Image<u8>> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = compute.cuda() {
            self.check_raw(raw.layout())?;
            let input = match self.params.reduce {
                Reduce::Crop(roi) => raw.crop(roi)?,
                _ => raw.clone(),
            };
            let on_device = crate::DeviceMosaic::upload(gpu, &input)?;
            let prepared = DeviceIsp::upload(gpu, self)?;
            return prepared.run(gpu, self, &on_device)?.download(gpu);
        }
        self.run_cpu(compute.cpu(), raw)
    }
}

/// An [`Isp`]'s calibrations and tables on the device, uploaded once.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeviceIsp {
    dark: Option<tetera_compute::DeviceBuffer<u16>>,
    gains: Option<tetera_compute::DeviceBuffer<u16>>,
    resize: crate::resize::DeviceResize,
    colour: crate::colour::DeviceColour,
}

#[cfg(feature = "cuda")]
impl DeviceIsp {
    /// `isp`'s calibrations and tables, copied to `gpu`.
    pub fn upload(gpu: &tetera_compute::CudaBackend, isp: &Isp) -> Result<Self> {
        Ok(Self {
            dark: isp
                .params
                .dark
                .as_ref()
                .map(|d| gpu.upload(d.sites()))
                .transpose()?,
            gains: isp
                .params
                .flat
                .as_ref()
                .map(|f| gpu.upload(f.gains()))
                .transpose()?,
            resize: crate::resize::DeviceResize::upload(gpu, &isp.resize)?,
            colour: crate::colour::DeviceColour::upload(gpu, &isp.colour)?,
        })
    }

    /// The chain on the device. `input` is the raw frame, or, for a crop, the cropped frame.
    pub fn run(
        &self,
        gpu: &tetera_compute::CudaBackend,
        isp: &Isp,
        input: &crate::DeviceMosaic,
    ) -> Result<crate::DeviceImage<u8>> {
        let expected = match isp.params.reduce {
            Reduce::Crop(_) => isp.reduced,
            _ => isp.raw,
        };
        if *input.layout() != expected {
            return Err(Error::invalid_argument(format!(
                "the ISP on the device takes {} × {} {:?}, not {} × {} {:?}",
                expected.width,
                expected.height,
                expected.pattern,
                input.layout().width,
                input.layout().height,
                input.layout().pattern
            )));
        }
        let binned;
        let mut mosaic = match isp.params.reduce {
            Reduce::Bin(factor) => {
                binned = crate::bayer::bin_device(gpu, input, factor, BinMode::Mean)?;
                &binned
            }
            _ => input,
        };
        let darkened;
        if let Some(dark) = &self.dark {
            darkened = crate::correct::subtract_dark_device(gpu, mosaic, dark)?;
            mosaic = &darkened;
        }
        let flattened;
        if let Some(gains) = &self.gains {
            flattened = crate::correct::apply_flat_device(gpu, mosaic, gains)?;
            mosaic = &flattened;
        }
        let rgb = crate::demosaic::demosaic_device::<u16>(gpu, mosaic, isp.params.demosaic)?;
        let small = crate::resize::resize_device(gpu, &rgb, &self.resize)?;
        crate::colour::colour_device(gpu, &small, &self.colour)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correct::subtract_dark_cpu;
    use crate::pattern::{Sensor, test_pattern};
    use tetera_core::frame::format::BayerPattern;

    fn cpu() -> CpuBackend {
        CpuBackend::new(3).unwrap()
    }

    fn sensor() -> Sensor {
        Sensor {
            offset: 128,
            vignetting: 0.3,
            response: [0.7, 1.0, 0.5],
            ..Sensor::default()
        }
    }

    /// The sensor's dark, and its flat field from an even exposure, at `w × h`.
    fn calibrations(s: &Sensor, w: u32, h: u32) -> (Mosaic, FlatField) {
        let dark = s.dark(w, h).unwrap();
        let even = s
            .expose(&Image::new(w, h, 3, [3_000u16; 3].repeat((w * h) as usize)).unwrap())
            .unwrap();
        (
            dark.clone(),
            FlatField::new(&subtract_dark_cpu(&cpu(), &even, &dark).unwrap()).unwrap(),
        )
    }

    #[test]
    fn reducing_crops_moving_the_pattern_or_bins_by_the_mean() {
        let raw = MosaicLayout {
            width: 20,
            height: 12,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        };
        let crop = IspParams {
            reduce: Reduce::Crop(Rect::new(3, 1, 10, 8)),
            ..IspParams::preview(4_095, 4, 4)
        };
        assert_eq!(
            crop.reduced(&raw).unwrap(),
            MosaicLayout {
                width: 10,
                height: 8,
                pattern: BayerPattern::Bggr,
                ..raw
            }
        );
        let bin = IspParams {
            reduce: Reduce::Bin(2),
            ..IspParams::preview(4_095, 4, 4)
        };
        assert_eq!(
            bin.reduced(&raw).unwrap(),
            MosaicLayout {
                width: 10,
                height: 6,
                ..raw
            },
            "the mean keeps 12 bits"
        );
        assert_eq!(IspParams::preview(4_095, 4, 4).reduced(&raw).unwrap(), raw);
        let outside = IspParams {
            reduce: Reduce::Crop(Rect::new(15, 0, 10, 8)),
            ..IspParams::preview(4_095, 4, 4)
        };
        let err = outside.reduced(&raw).unwrap_err();
        assert!(
            err.to_string().ends_with("isn't inside the 20 × 12 frame"),
            "{err}"
        );
        let empty = IspParams {
            reduce: Reduce::Crop(Rect::new(0, 0, 0, 8)),
            ..IspParams::preview(4_095, 4, 4)
        };
        assert!(empty.reduced(&raw).is_err());
        assert!(
            IspParams {
                reduce: Reduce::Bin(5),
                ..IspParams::preview(4_095, 4, 4)
            }
            .reduced(&raw)
            .is_err()
        );
    }

    #[test]
    fn the_chain_runs_its_stages_in_order() {
        let s = sensor();
        let (w, h) = (64, 48);
        let raw = s.expose(&test_pattern(w, h, 12)).unwrap();
        let (dark, flat) = calibrations(&s, w, h);
        let params = IspParams {
            dark: Some(dark.clone()),
            flat: Some(flat.clone()),
            ..IspParams::preview(4_095, 32, 24)
        };
        let isp = Isp::new(params, *raw.layout()).unwrap();
        let by_hand = {
            let m = apply_flat_cpu(
                &cpu(),
                &subtract_dark_cpu(&cpu(), &raw, &dark).unwrap(),
                &flat,
            )
            .unwrap();
            let rgb = demosaic_cpu::<u16>(&cpu(), &m, Demosaic::MalvarHeCutler).unwrap();
            colour_cpu(
                &cpu(),
                &resize_cpu(&cpu(), &rgb, isp.resize()).unwrap(),
                isp.colour(),
            )
            .unwrap()
        };
        let out = isp.run_cpu(&cpu(), &raw).unwrap();
        assert_eq!(out, by_hand);
        assert_eq!((out.width(), out.height(), out.channels()), (32, 24, 3));
        assert_eq!(
            (isp.raw(), isp.reduced(), isp.params().output),
            (raw.layout(), raw.layout(), (32, 24))
        );
    }

    #[test]
    fn dark_and_flat_take_the_sensors_faults_away() {
        // An even grey through the faulty sensor, white-balanced back: with the dark and the
        // flat field, the preview is even; without them, its corners are darker.
        let s = sensor();
        let (w, h) = (64, 48);
        let raw = s
            .expose(&Image::new(w, h, 3, [2_000u16; 3].repeat((w * h) as usize)).unwrap())
            .unwrap();
        let (dark, flat) = calibrations(&s, w, h);
        let balance = ColourCorrection {
            white_balance: [1.0 / 0.7, 1.0, 1.0 / 0.5],
            ..ColourCorrection::IDENTITY
        };
        let base = IspParams {
            colour: balance,
            curve: Curve::Linear,
            ..IspParams::preview(4_095, 16, 12)
        };
        let corrected = Isp::new(
            IspParams {
                dark: Some(dark),
                flat: Some(flat),
                ..base.clone()
            },
            *raw.layout(),
        )
        .unwrap();
        let spread = |image: &Image<u8>| {
            let (lo, hi) = image
                .data()
                .iter()
                .fold((255, 0), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            hi - lo
        };
        let even = corrected.run_cpu(&cpu(), &raw).unwrap();
        assert!(spread(&even) <= 2, "{:?}", &even.data()[..12]);
        let raw_only = Isp::new(base, *raw.layout())
            .unwrap()
            .run_cpu(&cpu(), &raw)
            .unwrap();
        assert!(
            spread(&raw_only) > 20,
            "vignetted without them: {}",
            spread(&raw_only)
        );
    }

    #[test]
    fn calibrations_must_match_the_reduced_layout() {
        let raw = MosaicLayout {
            width: 40,
            height: 30,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        };
        let (dark, flat) = calibrations(&sensor(), 40, 30);
        let binned = IspParams {
            reduce: Reduce::Bin(2),
            dark: Some(dark),
            ..IspParams::preview(4_095, 10, 10)
        };
        let err = Isp::new(binned, raw).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the dark is 40 × 30 Rggb, but the chain works at 20 × 14 Rggb"
        );
        let cropped = IspParams {
            reduce: Reduce::Crop(Rect::new(1, 0, 40 - 1, 30)),
            flat: Some(flat),
            ..IspParams::preview(4_095, 10, 10)
        };
        let err = Isp::new(cropped, raw).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the flat field is 40 × 30 Rggb, but the chain works at 39 × 30 Grbg"
        );
        let tiny = IspParams {
            reduce: Reduce::Crop(Rect::new(0, 0, 2, 2)),
            ..IspParams::preview(4_095, 1, 1)
        };
        let err = Isp::new(tiny, raw).unwrap_err();
        assert_eq!(
            err.to_string(),
            "MalvarHeCutler needs at least 3 × 3, but the chain works at 2 × 2"
        );
        assert!(
            Isp::new(IspParams::preview(4_095, 0, 10), raw).is_err(),
            "no output"
        );
        let isp = Isp::new(IspParams::preview(4_095, 10, 10), raw).unwrap();
        let other = Mosaic::new(
            MosaicLayout {
                significant_bits: 10,
                ..raw
            },
            vec![0; 1_200],
        )
        .unwrap();
        let err = isp.run_cpu(&cpu(), &other).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the ISP was prepared for 40 × 30 Rggb of 12 bits, not 40 × 30 Rggb of 10"
        );
    }

    #[test]
    fn a_crop_and_a_bin_reduce_before_the_rest() {
        let raw = sensor().expose(&test_pattern(40, 30, 12)).unwrap();
        let crop = Isp::new(
            IspParams {
                reduce: Reduce::Crop(Rect::new(5, 3, 20, 16)),
                ..IspParams::preview(4_095, 10, 8)
            },
            *raw.layout(),
        )
        .unwrap();
        let cropped = raw.crop(Rect::new(5, 3, 20, 16)).unwrap();
        assert_eq!(crop.reduce_cpu(&cpu(), &raw).unwrap(), cropped);
        let whole = Isp::new(IspParams::preview(4_095, 10, 8), *cropped.layout()).unwrap();
        assert_eq!(
            crop.run_cpu(&cpu(), &raw).unwrap(),
            whole.run_cpu(&cpu(), &cropped).unwrap()
        );
        let bin = Isp::new(
            IspParams {
                reduce: Reduce::Bin(2),
                ..IspParams::preview(4_095, 10, 8)
            },
            *raw.layout(),
        )
        .unwrap();
        assert_eq!(
            bin.reduce_cpu(&cpu(), &raw).unwrap(),
            bin_cpu(&cpu(), &raw, 2, BinMode::Mean).unwrap()
        );
    }

    #[test]
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let raw = sensor().expose(&test_pattern(24, 16, 12)).unwrap();
        let isp = Isp::new(IspParams::preview(4_095, 12, 8), *raw.layout()).unwrap();
        assert_eq!(
            isp.run(&compute, &raw).unwrap(),
            isp.run_cpu(compute.cpu(), &raw).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_chain_on_the_gpu_and_its_twin_agree_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = CpuBackend::new(0).unwrap();
        let s = sensor();
        let colour = ColourCorrection {
            white_balance: [1.4, 1.0, 1.9],
            matrix: [[1.5, -0.3, -0.2], [-0.2, 1.4, -0.2], [0.0, -0.4, 1.4]],
        };
        for (w, h, reduce, output) in [
            (64, 48, Reduce::Full, (40, 30)),
            (64, 48, Reduce::Crop(Rect::new(3, 5, 41, 33)), (20, 16)),
            (5_472, 3_648, Reduce::Full, (1_920, 1_280)),
            (5_472, 3_648, Reduce::Bin(2), (1_920, 1_280)),
        ] {
            let raw = s.expose(&test_pattern(w, h, 12)).unwrap();
            let base = IspParams {
                reduce,
                colour,
                ..IspParams::preview(4_095, output.0, output.1)
            };
            // Calibrations at the reduced layout, its pattern moved by an odd crop.
            let reduced = base.reduced(raw.layout()).unwrap();
            let at_reduced = Sensor {
                pattern: reduced.pattern,
                ..s
            };
            let (dark, flat) = calibrations(&at_reduced, reduced.width, reduced.height);
            let params = IspParams {
                dark: Some(dark),
                flat: Some(flat),
                ..base
            };
            let isp = Isp::new(params, *raw.layout()).unwrap();
            assert_eq!(
                isp.run(&compute, &raw).unwrap(),
                isp.run_cpu(&cpu, &raw).unwrap(),
                "{w} × {h} {reduce:?}"
            );
        }
        let raw = s.expose(&test_pattern(16, 12, 12)).unwrap();
        let isp = Isp::new(IspParams::preview(4_095, 8, 6), *raw.layout()).unwrap();
        let prepared = DeviceIsp::upload(&gpu, &isp).unwrap();
        let wrong =
            crate::DeviceMosaic::upload(&gpu, &raw.crop(Rect::new(0, 0, 8, 6)).unwrap()).unwrap();
        assert!(prepared.run(&gpu, &isp, &wrong).is_err());
    }
}
