//! The display transform: 16-bit values to 8 bits a person sees, channel by channel. Each
//! channel has black and white levels, below and above which it is black and white; between
//! them, a tone curve.
//!
//! A transform is a lookup table of 65,536 entries per channel, made once on the host. The CPU
//! twin and the CUDA kernel (`tetera-compute`'s `display.cu`) look values up in the same
//! tables, so they agree exactly.

use crate::image::Image;
use rayon::prelude::*;
use tetera_compute::imaging::TABLE_SIZE;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::{Error, Result};

/// A tone curve, from 0..=1 to 0..=1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Curve {
    /// As it is.
    Linear,
    /// `t^(1/γ)`, for γ above 0.
    Gamma(f64),
    /// The sRGB transfer function (IEC 61966-2-1): linear near black, then a power of 1/2.4.
    Srgb,
}

impl Curve {
    fn apply(self, t: f64) -> f64 {
        match self {
            Self::Linear => t,
            Self::Gamma(gamma) => t.powf(1.0 / gamma),
            Self::Srgb if t <= 0.003_130_8 => 12.92 * t,
            Self::Srgb => 1.055 * t.powf(1.0 / 2.4) - 0.055,
        }
    }
}

/// A channel's black and white levels: at or below `black` it is 0, at or above `white` 255.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Levels {
    /// The darkest value shown as black.
    pub black: u16,
    /// The brightest value shown as white.
    pub white: u16,
}

/// A display transform: a table for each channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayTransform {
    channels: u8,
    tables: Vec<u8>,
}

impl DisplayTransform {
    /// A transform with `levels` for each channel, in order, and one `curve`.
    pub fn new(levels: &[Levels], curve: Curve) -> Result<Self> {
        if levels.is_empty() || levels.len() > 4 {
            return Err(Error::invalid_argument(format!(
                "a display transform takes 1 to 4 channels' levels, not {}",
                levels.len()
            )));
        }
        if let Curve::Gamma(gamma) = curve
            && !(gamma.is_finite() && gamma > 0.0)
        {
            return Err(Error::invalid_argument(format!(
                "a gamma must be above 0, not {gamma}"
            )));
        }
        let mut tables = Vec::with_capacity(levels.len() * TABLE_SIZE);
        for (channel, level) in levels.iter().enumerate() {
            if level.white <= level.black {
                return Err(Error::invalid_argument(format!(
                    "channel {channel}'s white, {}, must be above its black, {}",
                    level.white, level.black
                )));
            }
            let span = f64::from(level.white - level.black);
            tables.extend((0..TABLE_SIZE).map(|v| {
                let t = ((v as f64 - f64::from(level.black)) / span).clamp(0.0, 1.0);
                (curve.apply(t) * 255.0).round().clamp(0.0, 255.0) as u8
            }));
        }
        Ok(Self {
            channels: levels.len() as u8,
            tables,
        })
    }

    /// One `level` for each of `channels`.
    pub fn uniform(channels: u8, level: Levels, curve: Curve) -> Result<Self> {
        Self::new(&vec![level; usize::from(channels)], curve)
    }

    /// Channels it transforms.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Its tables, [`TABLE_SIZE`] entries a channel.
    pub fn tables(&self) -> &[u8] {
        &self.tables
    }

    fn check(&self, channels: u8) -> Result<()> {
        if channels != self.channels {
            return Err(Error::invalid_argument(format!(
                "a display transform of {} channels can't take an image of {channels}",
                self.channels
            )));
        }
        Ok(())
    }
}

/// The CPU twin: each value looked up in its channel's table.
pub fn display_cpu(
    cpu: &CpuBackend,
    image: &Image<u16>,
    transform: &DisplayTransform,
) -> Result<Image<u8>> {
    transform.check(image.channels())?;
    let c = usize::from(image.channels());
    let mut data = vec![0u8; image.data().len()];
    cpu.install(|| {
        data.par_chunks_mut(c * 4_096)
            .zip(image.data().par_chunks(c * 4_096))
            .for_each(|(out, values)| {
                for (i, (o, &v)) in out.iter_mut().zip(values).enumerate() {
                    *o = transform.tables[(i % c) * TABLE_SIZE + usize::from(v)];
                }
            });
    });
    Image::new(image.width(), image.height(), image.channels(), data)
}

/// The display transform, on whichever backend `compute` chose.
pub fn display(
    compute: &Compute,
    image: &Image<u16>,
    transform: &DisplayTransform,
) -> Result<Image<u8>> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let on_device = crate::DeviceImage::upload(gpu, image)?;
        let tables = DeviceDisplay::upload(gpu, transform)?;
        return display_device(gpu, &on_device, &tables)?.download(gpu);
    }
    display_cpu(compute.cpu(), image, transform)
}

/// A display transform's tables on the device, uploaded once.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeviceDisplay {
    channels: u8,
    tables: tetera_compute::DeviceBuffer<u8>,
}

#[cfg(feature = "cuda")]
impl DeviceDisplay {
    /// `transform`'s tables, copied to `gpu`.
    pub fn upload(gpu: &tetera_compute::CudaBackend, transform: &DisplayTransform) -> Result<Self> {
        Ok(Self {
            channels: transform.channels,
            tables: gpu.upload(&transform.tables)?,
        })
    }
}

/// The display transform of an image on the device, the result left there.
#[cfg(feature = "cuda")]
pub fn display_device(
    gpu: &tetera_compute::CudaBackend,
    image: &crate::DeviceImage<u16>,
    transform: &DeviceDisplay,
) -> Result<crate::DeviceImage<u8>> {
    if image.channels() != transform.channels {
        return Err(Error::invalid_argument(format!(
            "a display transform of {} channels can't take an image of {}",
            transform.channels,
            image.channels()
        )));
    }
    let mut out = crate::DeviceImage::zeros(gpu, image.width(), image.height(), image.channels())?;
    gpu.display(
        image.buffer(),
        image.channels(),
        &transform.tables,
        out.buffer_mut(),
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWELVE: Levels = Levels {
        black: 0,
        white: 4_095,
    };

    fn cpu() -> CpuBackend {
        CpuBackend::new(2).unwrap()
    }

    #[test]
    fn the_tables_ends_are_black_and_white_and_it_never_falls() {
        for curve in [
            Curve::Linear,
            Curve::Gamma(2.2),
            Curve::Gamma(0.5),
            Curve::Srgb,
        ] {
            let t = DisplayTransform::new(
                &[Levels {
                    black: 100,
                    white: 3_000,
                }],
                curve,
            )
            .unwrap();
            let table = t.tables();
            assert_eq!(table.len(), TABLE_SIZE);
            assert!(
                table[..=100].iter().all(|&v| v == 0),
                "{curve:?}: black at and below 100"
            );
            assert!(
                table[3_000..].iter().all(|&v| v == 255),
                "{curve:?}: white from 3,000"
            );
            assert!(
                table[101] < 255 && table[2_999] > 0,
                "{curve:?}: between them, between"
            );
            assert!(
                table.windows(2).all(|w| w[0] <= w[1]),
                "{curve:?} never falls"
            );
        }
    }

    #[test]
    fn the_curves_are_what_they_say() {
        let at = |curve, v: u16| {
            DisplayTransform::new(&[TWELVE], curve).unwrap().tables()[usize::from(v)]
        };
        // Halfway, 2,047.5 of 4,095 rounds to 2,048: linear 127.5 → 128; γ 2.2 → 186.
        assert_eq!(at(Curve::Linear, 2_048), 128);
        assert_eq!(
            at(Curve::Gamma(2.2), 2_048),
            (255.0 * (2_048.0f64 / 4_095.0).powf(1.0 / 2.2)).round() as u8
        );
        assert_eq!(at(Curve::Gamma(1.0), 1_000), at(Curve::Linear, 1_000));
        // sRGB: 0.18 grey shows as 118.
        assert_eq!(at(Curve::Srgb, (0.18f64 * 4_095.0).round() as u16), 118);
        // Near black, sRGB is linear, 12.92 times: 0.003 → 0.03876 × 255 = 9.88 → 10.
        assert_eq!(Curve::Srgb.apply(0.003), 12.92 * 0.003);
        // Where its pieces meet, the standard's constants leave a step of about 1e-5.
        assert!((Curve::Srgb.apply(0.003_130_8) - Curve::Srgb.apply(0.003_130_9)).abs() < 1e-4);
        assert!((Curve::Srgb.apply(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn each_channel_has_its_own_levels() {
        let t = DisplayTransform::new(
            &[
                Levels {
                    black: 0,
                    white: 1_000,
                },
                Levels {
                    black: 0,
                    white: 2_000,
                },
                Levels {
                    black: 0,
                    white: 4_000,
                },
            ],
            Curve::Linear,
        )
        .unwrap();
        let image = Image::new(2, 1, 3, vec![1_000u16, 1_000, 1_000, 500, 500, 500]).unwrap();
        let out = display_cpu(&cpu(), &image, &t).unwrap();
        assert_eq!(out.data(), [255, 128, 64, 128, 64, 32]);
        assert_eq!((out.width(), out.height(), out.channels()), (2, 1, 3));
        assert_eq!(t.channels(), 3);
        let grey = DisplayTransform::uniform(1, TWELVE, Curve::Linear).unwrap();
        assert_eq!(grey.channels(), 1);
    }

    #[test]
    fn every_value_finds_its_own_channels_table_however_large_the_image() {
        let t = DisplayTransform::new(
            &[
                Levels {
                    black: 0,
                    white: 1_000,
                },
                Levels {
                    black: 500,
                    white: 3_000,
                },
                Levels {
                    black: 0,
                    white: 4_095,
                },
            ],
            Curve::Gamma(2.2),
        )
        .unwrap();
        // 100 × 50 × 3: several of the CPU's chunks.
        let image = crate::pattern::test_pattern(100, 50, 12);
        let out = display_cpu(&cpu(), &image, &t).unwrap();
        for (i, (&o, &v)) in out.data().iter().zip(image.data()).enumerate() {
            assert_eq!(
                o,
                t.tables()[(i % 3) * TABLE_SIZE + usize::from(v)],
                "value {i}"
            );
        }
    }

    #[test]
    fn bad_levels_curves_and_images_are_refused() {
        let err = DisplayTransform::new(
            &[Levels {
                black: 10,
                white: 10,
            }],
            Curve::Linear,
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "channel 0's white, 10, must be above its black, 10"
        );
        let err = DisplayTransform::new(&[], Curve::Linear).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a display transform takes 1 to 4 channels' levels, not 0"
        );
        assert!(DisplayTransform::new(&[TWELVE; 5], Curve::Linear).is_err());
        assert!(DisplayTransform::new(&[TWELVE; 4], Curve::Linear).is_ok());
        for gamma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                DisplayTransform::new(&[TWELVE], Curve::Gamma(gamma)).is_err(),
                "{gamma}"
            );
        }
        let err = DisplayTransform::new(&[TWELVE], Curve::Gamma(0.0)).unwrap_err();
        assert_eq!(err.to_string(), "a gamma must be above 0, not 0");
        let t = DisplayTransform::uniform(3, TWELVE, Curve::Srgb).unwrap();
        let grey = Image::new(1, 1, 1, vec![0u16]).unwrap();
        let err = display_cpu(&cpu(), &grey, &t).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a display transform of 3 channels can't take an image of 1"
        );
    }

    #[test]
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let image = crate::pattern::test_pattern(20, 12, 12);
        let t = DisplayTransform::uniform(3, TWELVE, Curve::Srgb).unwrap();
        assert_eq!(
            display(&compute, &image, &t).unwrap(),
            display_cpu(compute.cpu(), &image, &t).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernel_and_its_twin_agree_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let t = DisplayTransform::new(
            &[
                Levels {
                    black: 64,
                    white: 4_000,
                },
                Levels {
                    black: 0,
                    white: 4_095,
                },
                Levels {
                    black: 200,
                    white: 3_500,
                },
            ],
            Curve::Srgb,
        )
        .unwrap();
        for (w, h) in [(1, 1), (33, 17), (1_920, 1_280)] {
            let image = crate::pattern::test_pattern(w, h, 12);
            let on_cpu = display_cpu(&CpuBackend::new(0).unwrap(), &image, &t).unwrap();
            assert_eq!(display(&compute, &image, &t).unwrap(), on_cpu, "{w} × {h}");
        }
        let grey =
            crate::DeviceImage::upload(&gpu, &Image::new(1, 1, 1, vec![0u16]).unwrap()).unwrap();
        let tables = DeviceDisplay::upload(&gpu, &t).unwrap();
        assert!(display_device(&gpu, &grey, &tables).is_err());
    }
}
