//! Corrections for the sensor and the light, on the mosaic, before demosaicking: dark
//! subtraction, and the flat field.
//!
//! **Dark subtraction** takes away a master dark, what the sensor gives with no light at one
//! exposure, gain and mode (its offset and dark current), site by site, clamped at zero. A
//! [`DarkLibrary`] picks the master that matches a frame's metadata.
//!
//! **The flat field** evens out what the optics and the sensor make uneven, vignetting above
//! all. A master flat is a dark-subtracted frame of an even field. Each site's gain is its
//! Bayer channel's mean over its own value, so a corrected even field is even again. Gains are
//! fixed point, in 1/4096ths, so the CPU twin and the CUDA kernel (`tetera-compute`'s
//! `isp.cu`) agree exactly. A site below 5 % of its channel's mean, dust or a dead pixel,
//! keeps a gain of 1 rather than one that would amplify its noise without bound.

use crate::image::{Mosaic, MosaicLayout};
use rayon::prelude::*;
use std::time::Duration;
use tetera_compute::isp::GAIN_ONE;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::frame::format::{Binning, Channel};
use tetera_core::{Error, FrameMeta, Result};

/// A master dark: the mosaic the sensor gives with no light, at one exposure, gain, mode and
/// binning (P1.18 makes them).
#[derive(Debug, Clone, PartialEq)]
pub struct MasterDark {
    /// The exposure it was taken at.
    pub exposure: Duration,
    /// The gain.
    pub gain: f64,
    /// The camera's mode.
    pub mode: u32,
    /// The binning.
    pub binning: Binning,
    /// Its sites.
    pub mosaic: Mosaic,
}

/// Master darks, and how one is chosen for a frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DarkLibrary {
    darks: Vec<MasterDark>,
}

impl DarkLibrary {
    /// How far a dark's exposure may be from a frame's, as a share of the frame's: dark current
    /// grows with the exposure, and 2 % of it is below the sensor's noise.
    pub const EXPOSURE_TOLERANCE: f64 = 0.02;

    /// A library of `darks`.
    pub fn new(darks: Vec<MasterDark>) -> Self {
        Self { darks }
    }

    /// Its darks.
    pub fn darks(&self) -> &[MasterDark] {
        &self.darks
    }

    /// The dark for a frame of `meta` and `layout`: the same mode, binning, gain and size,
    /// and an exposure within [`Self::EXPOSURE_TOLERANCE`]; the nearest exposure, the first of
    /// equals, if several match.
    pub fn select(&self, meta: &FrameMeta, layout: &MosaicLayout) -> Result<&MasterDark> {
        let close = |dark: &MasterDark| {
            let (a, b) = (dark.exposure.as_secs_f64(), meta.exposure.as_secs_f64());
            (a - b).abs() <= Self::EXPOSURE_TOLERANCE * b
        };
        self.darks
            .iter()
            .filter(|d| d.mode == meta.mode && d.binning == meta.binning && d.gain == meta.gain)
            .filter(|d| {
                (d.mosaic.layout().width, d.mosaic.layout().height) == (layout.width, layout.height)
            })
            .filter(|d| close(d))
            .min_by_key(|d| d.exposure.abs_diff(meta.exposure))
            .ok_or_else(|| {
                Error::not_found(format!(
                    "no master dark for mode {}, binning {} × {}, gain {}, {:.3} ms and {} × {}",
                    meta.mode,
                    meta.binning.x,
                    meta.binning.y,
                    meta.gain,
                    meta.exposure.as_secs_f64() * 1e3,
                    layout.width,
                    layout.height
                ))
            })
    }
}

fn same_shape(what: &str, a: &MosaicLayout, b: &MosaicLayout) -> Result<()> {
    if (a.width, a.height, a.pattern) != (b.width, b.height, b.pattern) {
        return Err(Error::invalid_argument(format!(
            "{what} is {} × {} {:?}, but the frame is {} × {} {:?}",
            b.width, b.height, b.pattern, a.width, a.height, a.pattern
        )));
    }
    Ok(())
}

/// The CPU twin of dark subtraction: `mosaic − dark`, site by site, clamped at zero.
pub fn subtract_dark_cpu(cpu: &CpuBackend, mosaic: &Mosaic, dark: &Mosaic) -> Result<Mosaic> {
    same_shape("the dark", mosaic.layout(), dark.layout())?;
    let mut sites = vec![0u16; mosaic.sites().len()];
    cpu.install(|| {
        sites
            .par_iter_mut()
            .zip(mosaic.sites().par_iter().zip(dark.sites()))
            .for_each(|(out, (&v, &d))| *out = v.saturating_sub(d));
    });
    Mosaic::new(*mosaic.layout(), sites)
}

/// Dark subtraction, on whichever backend `compute` chose.
pub fn subtract_dark(compute: &Compute, mosaic: &Mosaic, dark: &Mosaic) -> Result<Mosaic> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        same_shape("the dark", mosaic.layout(), dark.layout())?;
        let on_device = crate::DeviceMosaic::upload(gpu, mosaic)?;
        let dark = gpu.upload(dark.sites())?;
        return subtract_dark_device(gpu, &on_device, &dark)?.download(gpu);
    }
    subtract_dark_cpu(compute.cpu(), mosaic, dark)
}

/// Dark subtraction on the device, the result left there. `dark` is the master's sites.
#[cfg(feature = "cuda")]
pub fn subtract_dark_device(
    gpu: &tetera_compute::CudaBackend,
    mosaic: &crate::DeviceMosaic,
    dark: &tetera_compute::DeviceBuffer<u16>,
) -> Result<crate::DeviceMosaic> {
    let mut out = gpu.zeros::<u16>(mosaic.layout().sites())?;
    gpu.subtract_dark(mosaic.buffer(), dark, &mut out)?;
    Ok(crate::DeviceMosaic::from_parts(*mosaic.layout(), out))
}

/// A flat field: a gain for each site, in 1/4096ths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatField {
    layout: MosaicLayout,
    gains: Vec<u16>,
}

impl FlatField {
    /// A site below this share of its channel's mean keeps a gain of 1.
    pub const GUARD: f64 = 0.05;

    /// The flat field of `flat`, a master flat already dark-subtracted: each site's gain is
    /// its Bayer channel's mean over its own value, up to the largest a gain holds, just under
    /// 16. A site below [`Self::GUARD`] of its channel's mean keeps a gain of 1.
    pub fn new(flat: &Mosaic) -> Result<Self> {
        let layout = *flat.layout();
        let channel = |i: usize| {
            let (x, y) = (
                (i % layout.width as usize) as u32,
                (i / layout.width as usize) as u32,
            );
            match layout.pattern.channel_at(x, y) {
                Channel::Red => 0,
                Channel::GreenOnRed => 1,
                Channel::GreenOnBlue => 2,
                Channel::Blue => 3,
            }
        };
        let (mut sums, mut counts) = ([0f64; 4], [0usize; 4]);
        for (i, &v) in flat.sites().iter().enumerate() {
            sums[channel(i)] += f64::from(v);
            counts[channel(i)] += 1;
        }
        let names = ["red", "green on red rows", "green on blue rows", "blue"];
        let mut means = [0f64; 4];
        for c in 0..4 {
            if counts[c] > 0 {
                means[c] = sums[c] / counts[c] as f64;
                if means[c] < 1.0 {
                    return Err(Error::invalid_argument(format!(
                        "the flat's {} sites are dark, a mean of {:.2}: it can't even out a frame",
                        names[c], means[c]
                    )));
                }
            }
        }
        let gains = flat
            .sites()
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let mean = means[channel(i)];
                if f64::from(v) < Self::GUARD * mean {
                    GAIN_ONE
                } else {
                    (mean / f64::from(v) * f64::from(GAIN_ONE))
                        .round()
                        .min(f64::from(u16::MAX)) as u16
                }
            })
            .collect();
        Ok(Self { layout, gains })
    }

    /// Its shape.
    pub fn layout(&self) -> &MosaicLayout {
        &self.layout
    }

    /// Each site's gain, in 1/4096ths.
    pub fn gains(&self) -> &[u16] {
        &self.gains
    }
}

/// The CPU twin of the flat field: each site times its gain, rounded and clamped to the
/// mosaic's largest value.
pub fn apply_flat_cpu(cpu: &CpuBackend, mosaic: &Mosaic, flat: &FlatField) -> Result<Mosaic> {
    same_shape("the flat field", mosaic.layout(), &flat.layout)?;
    let max = u32::from(mosaic.layout().max());
    let mut sites = vec![0u16; mosaic.sites().len()];
    cpu.install(|| {
        sites
            .par_iter_mut()
            .zip(mosaic.sites().par_iter().zip(&flat.gains))
            .for_each(|(out, (&v, &g))| {
                *out = ((u32::from(v) * u32::from(g) + 2_048) >> 12).min(max) as u16
            });
    });
    Mosaic::new(*mosaic.layout(), sites)
}

/// The flat field, on whichever backend `compute` chose.
pub fn apply_flat(compute: &Compute, mosaic: &Mosaic, flat: &FlatField) -> Result<Mosaic> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        same_shape("the flat field", mosaic.layout(), &flat.layout)?;
        let on_device = crate::DeviceMosaic::upload(gpu, mosaic)?;
        let gains = gpu.upload(&flat.gains)?;
        return apply_flat_device(gpu, &on_device, &gains)?.download(gpu);
    }
    apply_flat_cpu(compute.cpu(), mosaic, flat)
}

/// The flat field on the device, the result left there. `gains` is the flat field's.
#[cfg(feature = "cuda")]
pub fn apply_flat_device(
    gpu: &tetera_compute::CudaBackend,
    mosaic: &crate::DeviceMosaic,
    gains: &tetera_compute::DeviceBuffer<u16>,
) -> Result<crate::DeviceMosaic> {
    let mut out = gpu.zeros::<u16>(mosaic.layout().sites())?;
    gpu.apply_flat(mosaic.buffer(), gains, mosaic.layout().max(), &mut out)?;
    Ok(crate::DeviceMosaic::from_parts(*mosaic.layout(), out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::Sensor;
    use tetera_core::frame::format::BayerPattern;

    fn cpu() -> CpuBackend {
        CpuBackend::new(2).unwrap()
    }

    fn layout(width: u32, height: u32) -> MosaicLayout {
        MosaicLayout {
            width,
            height,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        }
    }

    fn flat_mosaic(width: u32, height: u32, value: u16) -> Mosaic {
        Mosaic::new(
            layout(width, height),
            vec![value; (width * height) as usize],
        )
        .unwrap()
    }

    fn dark(exposure_ms: u64, gain: f64, mode: u32, value: u16) -> MasterDark {
        MasterDark {
            exposure: Duration::from_millis(exposure_ms),
            gain,
            mode,
            binning: Binning::NONE,
            mosaic: flat_mosaic(4, 4, value),
        }
    }

    fn meta(exposure_ms: u64, gain: f64, mode: u32) -> FrameMeta {
        FrameMeta {
            exposure: Duration::from_millis(exposure_ms),
            gain,
            mode,
            ..FrameMeta::default()
        }
    }

    #[test]
    fn a_dark_is_chosen_by_mode_binning_gain_size_and_the_nearest_exposure() {
        let library = DarkLibrary::new(vec![
            dark(100, 1.0, 0, 1),
            dark(101, 1.0, 0, 2),
            dark(100, 2.0, 0, 3),
            dark(100, 1.0, 1, 4),
            dark(200, 1.0, 0, 5),
        ]);
        let pick = |m: &FrameMeta| {
            library
                .select(m, &layout(4, 4))
                .map(|d| d.mosaic.sites()[0])
        };
        assert_eq!(pick(&meta(100, 1.0, 0)).unwrap(), 1, "exact");
        assert_eq!(
            pick(&meta(102, 1.0, 0)).unwrap(),
            2,
            "the nearest of two within 2 %"
        );
        assert_eq!(pick(&meta(100, 2.0, 0)).unwrap(), 3, "by gain");
        assert_eq!(pick(&meta(100, 1.0, 1)).unwrap(), 4, "by mode");
        assert_eq!(
            pick(&meta(197, 1.0, 0)).unwrap(),
            5,
            "200 is within 2 % of 197"
        );
        assert!(
            pick(&meta(196, 1.0, 0)).is_err(),
            "but not of 196: 2 % of the frame's"
        );
        let err = pick(&meta(150, 1.0, 0)).unwrap_err();
        assert_eq!(err.kind(), tetera_core::ErrorKind::NotFound);
        assert_eq!(
            err.to_string(),
            "no master dark for mode 0, binning 1 × 1, gain 1, 150.000 ms and 4 × 4"
        );
        assert!(pick(&meta(100, 1.5, 0)).is_err());
        let binned = FrameMeta {
            binning: Binning::new(2, 2).unwrap(),
            ..meta(100, 1.0, 0)
        };
        assert!(pick(&binned).is_err(), "by binning");
        assert!(
            library.select(&meta(100, 1.0, 0), &layout(8, 4)).is_err(),
            "by size"
        );
        assert_eq!(library.darks().len(), 5);
        assert!(
            DarkLibrary::default()
                .select(&meta(100, 1.0, 0), &layout(4, 4))
                .is_err()
        );
    }

    #[test]
    fn dark_subtraction_clamps_at_zero() {
        let frame = Mosaic::new(layout(2, 2), vec![100, 50, 0, 4_095]).unwrap();
        let dark = Mosaic::new(layout(2, 2), vec![40, 60, 0, 95]).unwrap();
        let out = subtract_dark_cpu(&cpu(), &frame, &dark).unwrap();
        assert_eq!(out.sites(), [60, 0, 0, 4_000]);
        assert_eq!(out.layout(), frame.layout());
        let other = Mosaic::new(
            MosaicLayout {
                pattern: BayerPattern::Bggr,
                ..layout(2, 2)
            },
            vec![0; 4],
        )
        .unwrap();
        let err = subtract_dark_cpu(&cpu(), &frame, &other).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the dark is 2 × 2 Bggr, but the frame is 2 × 2 Rggb"
        );
    }

    #[test]
    fn the_flat_fields_gains_even_out_each_channel() {
        // Red sites 1,000 and 500, the rest 2,000: red's mean is 750.
        let flat = Mosaic::new(
            layout(4, 2),
            vec![1_000, 2_000, 500, 2_000, 2_000, 2_000, 2_000, 2_000],
        )
        .unwrap();
        let field = FlatField::new(&flat).unwrap();
        // 750/1,000 = 0.75 and 750/500 = 1.5, in 1/4096ths; the greens and blue, 1.
        assert_eq!(
            field.gains(),
            [3_072, 4_096, 6_144, 4_096, 4_096, 4_096, 4_096, 4_096]
        );
        let corrected = apply_flat_cpu(&cpu(), &flat, &field).unwrap();
        assert_eq!(
            corrected.sites(),
            [750, 2_000, 750, 2_000, 2_000, 2_000, 2_000, 2_000]
        );
        assert_eq!(field.layout(), flat.layout());
    }

    #[test]
    fn near_zero_flat_sites_keep_a_gain_of_one() {
        // A dead red site, 0, and one at 4 % of the mean, among reds of 1,000.
        let mut sites = vec![1_000u16; 8 * 8];
        sites[0] = 0;
        sites[2] = 40;
        let flat = Mosaic::new(layout(8, 8), sites).unwrap();
        let field = FlatField::new(&flat).unwrap();
        assert_eq!((field.gains()[0], field.gains()[2]), (GAIN_ONE, GAIN_ONE));
        // At 6 % of the mean, a site is corrected: 941.25 / 60 = 15.6875, in 1/4096ths.
        let mut sites = vec![1_000u16; 8 * 8];
        sites[0] = 60;
        let field = FlatField::new(&Mosaic::new(layout(8, 8), sites).unwrap()).unwrap();
        assert_eq!(field.gains()[0], 64_256);
        // At 5.3 %, its gain, 18.8, is past the largest a gain holds, just under 16.
        let mut sites = vec![1_000u16; 8 * 8];
        sites[0] = 50;
        let field = FlatField::new(&Mosaic::new(layout(8, 8), sites).unwrap()).unwrap();
        assert_eq!(field.gains()[0], u16::MAX);
        let dark = flat_mosaic(4, 4, 0);
        let err = FlatField::new(&dark).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the flat's red sites are dark, a mean of 0.00: it can't even out a frame"
        );
    }

    #[test]
    fn the_flat_field_rounds_and_clamps_to_the_mosaics_range() {
        let field = FlatField {
            layout: layout(2, 2),
            gains: vec![6_144, 2_048, 8_192, 4_096],
        };
        let frame = Mosaic::new(layout(2, 2), vec![1_000, 3, 4_000, 7]).unwrap();
        assert_eq!(
            apply_flat_cpu(&cpu(), &frame, &field).unwrap().sites(),
            [1_500, 2, 4_095, 7]
        );
        let small = flat_mosaic(2, 4, 1);
        assert!(apply_flat_cpu(&cpu(), &small, &field).is_err());
    }

    #[test]
    fn dark_and_flat_leave_an_even_field_flat_within_a_percent() {
        // An even field, through a sensor with an offset, vignetting and unequal channels.
        let sensor = Sensor {
            offset: 160,
            vignetting: 0.35,
            response: [0.8, 1.0, 0.6],
            ..Sensor::default()
        };
        let even = crate::Image::new(96, 64, 3, [3_000u16; 3].repeat(96 * 64)).unwrap();
        let raw = sensor.expose(&even).unwrap();
        let dark = sensor.dark(96, 64).unwrap();
        let flat = subtract_dark_cpu(
            &cpu(),
            &sensor
                .expose(&crate::Image::new(96, 64, 3, [3_500u16; 3].repeat(96 * 64)).unwrap())
                .unwrap(),
            &dark,
        )
        .unwrap();
        let field = FlatField::new(&flat).unwrap();
        let corrected = apply_flat_cpu(
            &cpu(),
            &subtract_dark_cpu(&cpu(), &raw, &dark).unwrap(),
            &field,
        )
        .unwrap();
        let before = subtract_dark_cpu(&cpu(), &raw, &dark).unwrap();
        for channel in [
            Channel::Red,
            Channel::GreenOnRed,
            Channel::GreenOnBlue,
            Channel::Blue,
        ] {
            let values = |m: &Mosaic| -> Vec<f64> {
                m.sites()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        BayerPattern::Rggb.channel_at((*i % 96) as u32, (*i / 96) as u32) == channel
                    })
                    .map(|(_, &v)| f64::from(v))
                    .collect()
            };
            let spread = |v: &[f64]| {
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                v.iter()
                    .map(|x| (x - mean).abs() / mean)
                    .fold(0.0, f64::max)
            };
            assert!(
                spread(&values(&before)) > 0.1,
                "{channel:?}: vignetted beforehand"
            );
            assert!(
                spread(&values(&corrected)) < 0.01,
                "{channel:?}: {:.4}",
                spread(&values(&corrected))
            );
        }
    }

    #[test]
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let frame = Mosaic::new(layout(2, 2), vec![100, 50, 0, 4_095]).unwrap();
        let dark = Mosaic::new(layout(2, 2), vec![40, 60, 0, 95]).unwrap();
        assert_eq!(
            subtract_dark(&compute, &frame, &dark).unwrap(),
            subtract_dark_cpu(compute.cpu(), &frame, &dark).unwrap()
        );
        let field =
            FlatField::new(&Mosaic::new(layout(2, 2), vec![900, 1_000, 1_000, 1_100]).unwrap())
                .unwrap();
        assert_eq!(
            apply_flat(&compute, &frame, &field).unwrap(),
            apply_flat_cpu(compute.cpu(), &frame, &field).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernels_and_their_twins_agree_exactly() {
        let _gpu = tetera_compute::cuda_or_skip!();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = CpuBackend::new(0).unwrap();
        let sensor = Sensor {
            offset: 200,
            vignetting: 0.4,
            response: [0.9, 1.0, 0.7],
            ..Sensor::default()
        };
        for (w, h) in [(2, 2), (33, 17), (5_472, 3_648)] {
            let raw = sensor
                .expose(&crate::pattern::test_pattern(w, h, 12))
                .unwrap();
            let dark = sensor.dark(w, h).unwrap();
            let flat_raw = sensor
                .expose(
                    &crate::Image::new(w, h, 3, [3_000u16; 3].repeat((w * h) as usize)).unwrap(),
                )
                .unwrap();
            let field =
                FlatField::new(&subtract_dark_cpu(&cpu, &flat_raw, &dark).unwrap()).unwrap();
            let on_cpu = subtract_dark_cpu(&cpu, &raw, &dark).unwrap();
            assert_eq!(
                subtract_dark(&compute, &raw, &dark).unwrap(),
                on_cpu,
                "dark, {w} × {h}"
            );
            assert_eq!(
                apply_flat(&compute, &on_cpu, &field).unwrap(),
                apply_flat_cpu(&cpu, &on_cpu, &field).unwrap(),
                "flat, {w} × {h}"
            );
        }
    }
}
