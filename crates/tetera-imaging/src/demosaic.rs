//! Demosaicking: a Bayer mosaic to interleaved RGB, by bilinear interpolation, or by Malvar,
//! He and Cutler's gradient-corrected linear interpolation ("High-quality linear
//! interpolation for demosaicing of Bayer-patterned color images", ICASSP 2004).
//!
//! Both are linear filters whose weights, over a common power of two, are whole numbers:
//! quarters for bilinear, sixteenths for Malvar-He-Cutler, whose paper gives eighths and
//! halves of them. So each value is an integer sum. In 16 bits it is rounded and clamped to the
//! mosaic's range; as a float it is divided exactly, and keeps any overshoot. Beyond the edges
//! the mosaic is mirrored without repeating the edge, which keeps every site's colour.
//!
//! The CPU twin here applies the filters as tables of taps. The CUDA kernel
//! (`tetera-compute`'s `demosaic.cu`) computes the same sums written out, so the two are
//! independent, and agree bit for bit.

use crate::image::{Image, Mosaic, MosaicLayout};
use rayon::prelude::*;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::frame::format::Channel;
use tetera_core::{Error, Result};

pub use tetera_compute::imaging::Demosaic;

/// A tap of a filter: the site at (`dx`, `dy`) from the centre, and its weight.
type Tap = (i32, i32, i32);

/// The site itself, in quarters and in sixteenths.
const SELF_4: &[Tap] = &[(0, 0, 4)];
const SELF_16: &[Tap] = &[(0, 0, 16)];

/// Bilinear, in quarters: the four nearest across and down, the four diagonal, or the two in
/// the row or in the column.
const CROSS: &[Tap] = &[(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)];
const DIAGONAL: &[Tap] = &[(-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)];
const ROW: &[Tap] = &[(-1, 0, 2), (1, 0, 2)];
const COLUMN: &[Tap] = &[(0, -1, 2), (0, 1, 2)];

/// Malvar-He-Cutler's filters, in sixteenths: green at red or blue; red or blue at the other;
/// red or blue at green from the row's neighbours, and from the column's.
const GREEN_AT_RB: &[Tap] = &[
    (0, 0, 8),
    (-1, 0, 4),
    (1, 0, 4),
    (0, -1, 4),
    (0, 1, 4),
    (-2, 0, -2),
    (2, 0, -2),
    (0, -2, -2),
    (0, 2, -2),
];
const RB_AT_BR: &[Tap] = &[
    (0, 0, 12),
    (-1, -1, 4),
    (1, -1, 4),
    (-1, 1, 4),
    (1, 1, 4),
    (-2, 0, -3),
    (2, 0, -3),
    (0, -2, -3),
    (0, 2, -3),
];
const RB_AT_G_ROW: &[Tap] = &[
    (0, 0, 10),
    (-1, 0, 8),
    (1, 0, 8),
    (-2, 0, -2),
    (2, 0, -2),
    (-1, -1, -2),
    (1, -1, -2),
    (-1, 1, -2),
    (1, 1, -2),
    (0, -2, 1),
    (0, 2, 1),
];
const RB_AT_G_COLUMN: &[Tap] = &[
    (0, 0, 10),
    (0, -1, 8),
    (0, 1, 8),
    (0, -2, -2),
    (0, 2, -2),
    (-1, -1, -2),
    (1, -1, -2),
    (-1, 1, -2),
    (1, 1, -2),
    (-2, 0, 1),
    (2, 0, 1),
];

/// The filters for red, green and blue at a site of `channel`, and their denominator's
/// power of two.
fn filters(method: Demosaic, channel: Channel) -> ([&'static [Tap]; 3], u32) {
    match method {
        Demosaic::Bilinear => (
            match channel {
                Channel::Red => [SELF_4, CROSS, DIAGONAL],
                Channel::Blue => [DIAGONAL, CROSS, SELF_4],
                Channel::GreenOnRed => [ROW, SELF_4, COLUMN],
                Channel::GreenOnBlue => [COLUMN, SELF_4, ROW],
            },
            2,
        ),
        Demosaic::MalvarHeCutler => (
            match channel {
                Channel::Red => [SELF_16, GREEN_AT_RB, RB_AT_BR],
                Channel::Blue => [RB_AT_BR, GREEN_AT_RB, SELF_16],
                Channel::GreenOnRed => [RB_AT_G_ROW, SELF_16, RB_AT_G_COLUMN],
                Channel::GreenOnBlue => [RB_AT_G_COLUMN, SELF_16, RB_AT_G_ROW],
            },
            4,
        ),
    }
}

#[cfg(feature = "cuda")]
mod sample_bound {
    /// What the CUDA kernels can write too.
    pub trait DeviceSample: tetera_compute::imaging::DemosaicOutput {}
    impl<T: tetera_compute::imaging::DemosaicOutput> DeviceSample for T {}
}

#[cfg(not(feature = "cuda"))]
mod sample_bound {
    /// Without the `cuda` feature, any type.
    pub trait DeviceSample {}
    impl<T> DeviceSample for T {}
}

pub use sample_bound::DeviceSample;

/// A value demosaicking can write: 16 bits, rounded and clamped to the mosaic's range, or a
/// float, exact and unclamped.
pub trait Sample: DeviceSample + Copy + Default + Send + Sync + 'static {
    /// `sum / 2^shift`, for a mosaic whose largest value is `max`.
    fn from_sum(sum: i32, shift: u32, max: u16) -> Self;
}

impl Sample for u16 {
    fn from_sum(sum: i32, shift: u32, max: u16) -> Self {
        ((sum.max(0) + (1 << (shift - 1))) >> shift).min(i32::from(max)) as u16
    }
}

impl Sample for f32 {
    fn from_sum(sum: i32, shift: u32, _max: u16) -> Self {
        // Exact: |sum| is below 2^24, and the divisor a power of two.
        sum as f32 / (1u32 << shift) as f32
    }
}

fn check_size(layout: &MosaicLayout, method: Demosaic) -> Result<()> {
    let min = method.min_size();
    if layout.width < min || layout.height < min {
        return Err(Error::invalid_argument(format!(
            "{method:?} needs a mosaic of at least {min} × {min}, not {} × {}",
            layout.width, layout.height
        )));
    }
    Ok(())
}

/// The CPU twin: `mosaic` demosaicked by `method` to interleaved RGB of its size.
pub fn demosaic_cpu<T: Sample>(
    cpu: &CpuBackend,
    mosaic: &Mosaic,
    method: Demosaic,
) -> Result<Image<T>> {
    let layout = *mosaic.layout();
    check_size(&layout, method)?;
    let width = layout.width as usize;
    let max = layout.max();
    let mut data = vec![T::default(); 3 * layout.sites()];
    cpu.install(|| {
        data.par_chunks_mut(3 * width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let (x, y) = (x as i32, y as i32);
                    let (rgb, shift) =
                        filters(method, layout.pattern.channel_at(x as u32, y as u32));
                    for (value, taps) in pixel.iter_mut().zip(rgb) {
                        let sum: i32 = taps
                            .iter()
                            .map(|&(dx, dy, weight)| {
                                weight * i32::from(mosaic.mirrored(x + dx, y + dy))
                            })
                            .sum();
                        *value = T::from_sum(sum, shift, max);
                    }
                }
            });
    });
    Image::new(layout.width, layout.height, 3, data)
}

/// Demosaicking on whichever backend `compute` chose, from and to the host.
pub fn demosaic<T: Sample>(
    compute: &Compute,
    mosaic: &Mosaic,
    method: Demosaic,
) -> Result<Image<T>> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let on_device = crate::DeviceMosaic::upload(gpu, mosaic)?;
        return demosaic_device::<T>(gpu, &on_device, method)?.download(gpu);
    }
    demosaic_cpu(compute.cpu(), mosaic, method)
}

/// Demosaicking a mosaic on the device, the result left there.
#[cfg(feature = "cuda")]
pub fn demosaic_device<T: Sample>(
    gpu: &tetera_compute::CudaBackend,
    mosaic: &crate::DeviceMosaic,
    method: Demosaic,
) -> Result<crate::DeviceImage<T>> {
    let layout = mosaic.layout();
    let mut out = crate::DeviceImage::<T>::zeros(gpu, layout.width, layout.height, 3)?;
    gpu.demosaic(method, mosaic.buffer(), layout, out.buffer_mut())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayer::mosaic;
    use crate::pattern::test_pattern;
    use tetera_core::Rect;
    use tetera_core::frame::format::BayerPattern;

    const PATTERNS: [BayerPattern; 4] = [
        BayerPattern::Rggb,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
        BayerPattern::Bggr,
    ];
    const METHODS: [Demosaic; 2] = [Demosaic::Bilinear, Demosaic::MalvarHeCutler];
    const CHANNELS: [Channel; 4] = [
        Channel::Red,
        Channel::GreenOnRed,
        Channel::GreenOnBlue,
        Channel::Blue,
    ];

    fn cpu() -> CpuBackend {
        CpuBackend::new(4).unwrap()
    }

    fn flat(width: u32, height: u32, rgb: [u16; 3]) -> Image<u16> {
        Image::new(
            width,
            height,
            3,
            rgb.repeat(width as usize * height as usize),
        )
        .unwrap()
    }

    #[test]
    fn every_filter_weighs_to_its_denominator_and_takes_its_own_site_whole() {
        for method in METHODS {
            for channel in CHANNELS {
                let (rgb, shift) = filters(method, channel);
                for (k, taps) in rgb.iter().enumerate() {
                    let total: i32 = taps.iter().map(|t| t.2).sum();
                    assert_eq!(total, 1 << shift, "{method:?} at {channel:?}, channel {k}");
                    assert!(
                        taps.iter()
                            .all(|&(dx, dy, _)| dx.abs() <= 2 && dy.abs() <= 2)
                    );
                }
                // The site's own colour is the site itself.
                let own = match channel {
                    Channel::Red => 0,
                    Channel::GreenOnRed | Channel::GreenOnBlue => 1,
                    Channel::Blue => 2,
                };
                assert_eq!(rgb[own], [(0, 0, 1 << shift)], "{method:?} at {channel:?}");
            }
        }
    }

    #[test]
    fn the_filters_read_only_sites_of_the_colour_they_make() {
        // Away from the site itself, a filter for a colour reads that colour's sites, or, for
        // Malvar-He-Cutler's correction, the site's own colour: either green, at a green site.
        let colour = |c: Channel| match c {
            Channel::Red => 0,
            Channel::GreenOnRed | Channel::GreenOnBlue => 1,
            Channel::Blue => 2,
        };
        for method in METHODS {
            for pattern in PATTERNS {
                for (x, y) in [(2, 2), (3, 2), (2, 3), (3, 3)] {
                    let channel = pattern.channel_at(x, y);
                    let (rgb, _) = filters(method, channel);
                    for (k, taps) in rgb.iter().enumerate() {
                        for &(dx, dy, _) in taps.iter().filter(|t| (t.0, t.1) != (0, 0)) {
                            let read =
                                pattern.channel_at((x as i32 + dx) as u32, (y as i32 + dy) as u32);
                            assert!(
                                colour(read) == k || colour(read) == colour(channel),
                                "{method:?} {pattern:?} at ({x}, {y}) makes {k} from {read:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn flat_colours_survive_mosaicking_and_demosaicking_exactly_to_the_edges() {
        for pattern in PATTERNS {
            for method in METHODS {
                for (width, height) in [(3, 3), (4, 5), (9, 6)] {
                    let rgb = flat(width, height, [1_000, 2_500, 4_000]);
                    let m = mosaic(&rgb, pattern, 12).unwrap();
                    let out: Image<u16> = demosaic_cpu(&cpu(), &m, method).unwrap();
                    assert_eq!(out, rgb, "{method:?} {pattern:?} {width} × {height}");
                    let floats: Image<f32> = demosaic_cpu(&cpu(), &m, method).unwrap();
                    assert!(
                        floats
                            .data()
                            .iter()
                            .zip(rgb.data())
                            .all(|(&f, &v)| f == f32::from(v))
                    );
                }
            }
        }
    }

    #[test]
    fn a_crop_at_an_odd_offset_demosaicks_like_the_whole_frame() {
        let rgb = test_pattern(40, 30, 12);
        for pattern in PATTERNS {
            let whole = mosaic(&rgb, pattern, 12).unwrap();
            for method in METHODS {
                let full: Image<u16> = demosaic_cpu(&cpu(), &whole, method).unwrap();
                for (cx, cy) in [(1, 0), (0, 1), (3, 5), (8, 8)] {
                    let roi = Rect::new(cx, cy, 20, 16);
                    let cropped: Image<u16> =
                        demosaic_cpu(&cpu(), &whole.crop(roi).unwrap(), method).unwrap();
                    // Away from the crop's own edges, which it mirrors, the two agree.
                    for y in 2..14 {
                        for x in 2..18 {
                            assert_eq!(
                                cropped.pixel(x, y),
                                full.pixel(cx + x, cy + y),
                                "{method:?} {pattern:?} crop at ({cx}, {cy}), pixel ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn values_are_rounded_and_clamped_in_sixteen_bits_and_exact_as_floats() {
        assert_eq!(u16::from_sum(6, 2, 4_095), 2, "1.5 rounds up");
        assert_eq!(u16::from_sum(5, 2, 4_095), 1, "1.25 rounds down");
        assert_eq!(u16::from_sum(-1, 4, 4_095), 0);
        assert_eq!(u16::from_sum(16 * 5_000, 4, 4_095), 4_095);
        assert_eq!(u16::from_sum(16 * 4_095, 4, 4_095), 4_095);
        assert_eq!(f32::from_sum(-8_000, 4, 4_095), -500.0);
        assert_eq!(f32::from_sum(16 * 5_000 + 1, 4, 4_095), 5_000.062_5);
        // A bright red site among dark ones, away from the edges: Malvar-He-Cutler's green two
        // sites away is pulled below zero, and bilinear's isn't.
        let layout = MosaicLayout {
            width: 9,
            height: 9,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        };
        let mut sites = vec![0u16; 81];
        sites[4 * 9 + 4] = 4_000;
        let m = Mosaic::new(layout, sites).unwrap();
        let floats: Image<f32> = demosaic_cpu(&cpu(), &m, Demosaic::MalvarHeCutler).unwrap();
        // At the red site: itself; green 8/16 of it; blue 12/16 of it, as its diagonals are 0.
        assert_eq!(floats.pixel(4, 4), [4_000.0, 2_000.0, 3_000.0]);
        assert_eq!(floats.pixel(2, 4)[1], -500.0);
        let clamped: Image<u16> = demosaic_cpu(&cpu(), &m, Demosaic::MalvarHeCutler).unwrap();
        assert_eq!(clamped.pixel(2, 4)[1], 0);
        let bilinear: Image<f32> = demosaic_cpu(&cpu(), &m, Demosaic::Bilinear).unwrap();
        assert_eq!(bilinear.pixel(4, 4), [4_000.0, 0.0, 0.0]);
        assert_eq!(
            bilinear.pixel(5, 4),
            [2_000.0, 0.0, 0.0],
            "the mean of its row's reds"
        );
    }

    #[test]
    fn a_grey_ramp_comes_through_exactly_away_from_the_edges() {
        // Both methods reproduce a linear signal: bilinear averages it symmetrically, and
        // Malvar-He-Cutler's correction, a second difference, is zero on it. A tap in the wrong
        // place or with the wrong sign shifts the value. Mirrored edges aren't linear.
        let (width, height) = (12u32, 10u32);
        let value = |x: u32, y: u32| (100 + 8 * x + 24 * y) as u16;
        let ramp: Vec<u16> = (0..height)
            .flat_map(|y| (0..width).flat_map(move |x| [value(x, y); 3]))
            .collect();
        let rgb = Image::new(width, height, 3, ramp).unwrap();
        for pattern in PATTERNS {
            let m = mosaic(&rgb, pattern, 12).unwrap();
            for method in METHODS {
                let out: Image<u16> = demosaic_cpu(&cpu(), &m, method).unwrap();
                for y in 2..height - 2 {
                    for x in 2..width - 2 {
                        assert_eq!(
                            out.pixel(x, y),
                            [value(x, y); 3],
                            "{method:?} {pattern:?} at ({x}, {y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mosaics_too_small_for_the_window_are_refused() {
        let m = mosaic(&flat(2, 5, [1, 2, 3]), BayerPattern::Rggb, 12).unwrap();
        let err = demosaic_cpu::<u16>(&cpu(), &m, Demosaic::MalvarHeCutler).unwrap_err();
        assert_eq!(
            err.to_string(),
            "MalvarHeCutler needs a mosaic of at least 3 × 3, not 2 × 5"
        );
        assert!(demosaic_cpu::<u16>(&cpu(), &m, Demosaic::Bilinear).is_ok());
        let low = mosaic(&flat(5, 2, [1, 2, 3]), BayerPattern::Rggb, 12).unwrap();
        assert!(demosaic_cpu::<u16>(&cpu(), &low, Demosaic::MalvarHeCutler).is_err());
        let single = mosaic(&flat(1, 4, [1, 2, 3]), BayerPattern::Rggb, 12).unwrap();
        assert!(demosaic_cpu::<f32>(&cpu(), &single, Demosaic::Bilinear).is_err());
    }

    #[test]
    fn malvar_he_cutler_beats_bilinear_on_the_test_pattern() {
        let rgb = test_pattern(320, 240, 12);
        for pattern in PATTERNS {
            let m = mosaic(&rgb, pattern, 12).unwrap();
            let psnr = |method| {
                let out: Image<u16> = demosaic_cpu(&cpu(), &m, method).unwrap();
                crate::pattern::psnr(&rgb, &out, 4_095).unwrap()
            };
            let (bilinear, mhc) = (psnr(Demosaic::Bilinear), psnr(Demosaic::MalvarHeCutler));
            assert!(
                mhc > bilinear + 1.0,
                "{pattern:?}: {mhc:.2} dB against {bilinear:.2} dB"
            );
        }
    }

    #[test]
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let m = mosaic(&test_pattern(16, 12, 12), BayerPattern::Gbrg, 12).unwrap();
        let chosen: Image<u16> = demosaic(&compute, &m, Demosaic::MalvarHeCutler).unwrap();
        assert_eq!(
            chosen,
            demosaic_cpu(compute.cpu(), &m, Demosaic::MalvarHeCutler).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernels_and_their_twins_agree_bit_for_bit() {
        use tetera_compute::twin::{Tolerance, compare};
        let gpu = tetera_compute::cuda_or_skip!();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = CpuBackend::new(0).unwrap();
        // Every pattern at small sizes, either side of a block's 16; the camera's at full frame.
        let cases: [(u32, u32, &[BayerPattern]); 5] = [
            (3, 3, &PATTERNS),
            (2, 7, &PATTERNS),
            (17, 16, &PATTERNS),
            (33, 31, &PATTERNS),
            (5_472, 3_648, &[BayerPattern::Rggb]),
        ];
        for (width, height, patterns) in cases {
            let rgb = test_pattern(width, height, 12);
            for &pattern in patterns {
                let m = mosaic(&rgb, pattern, 12).unwrap();
                for method in METHODS {
                    if width < method.min_size() || height < method.min_size() {
                        continue;
                    }
                    let what = format!("{method:?} {pattern:?} {width} × {height}");
                    let on_cpu: Image<u16> = demosaic_cpu(&cpu, &m, method).unwrap();
                    let on_gpu: Image<u16> = demosaic(&compute, &m, method).unwrap();
                    let c = compare(on_cpu.data(), on_gpu.data(), Tolerance::EXACT).unwrap();
                    assert!(c.passes(), "{what}, 16 bits: {c}");
                    let on_cpu: Image<f32> = demosaic_cpu(&cpu, &m, method).unwrap();
                    let on_device = crate::DeviceMosaic::upload(&gpu, &m).unwrap();
                    let on_gpu = demosaic_device::<f32>(&gpu, &on_device, method)
                        .unwrap()
                        .download(&gpu)
                        .unwrap();
                    let c = compare(on_cpu.data(), on_gpu.data(), Tolerance::EXACT).unwrap();
                    assert!(c.passes(), "{what}, floats: {c}");
                }
            }
        }
    }
}
