//! The Bayer mosaic taken apart: its four colour planes, and a half-resolution RGB
//! superpixel, one pixel per 2 × 2 tile. Binned, keeping its layout (P1.05b). And the other
//! way, an RGB image sampled into a mosaic, as a sensor would see it, for tests and test
//! patterns.
//!
//! Each runs on the CPU here, and on CUDA through `tetera-compute`'s kernels (`bayer.cu`),
//! which compute the same.

use crate::image::{BayerPlanes, Image, Mosaic, MosaicLayout};
use rayon::prelude::*;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::frame::format::{BayerPattern, Channel};
use tetera_core::{Error, Result};

pub use tetera_compute::imaging::BinMode;

fn channel_index(channel: Channel) -> usize {
    match channel {
        Channel::Red => 0,
        Channel::GreenOnRed => 1,
        Channel::GreenOnBlue => 2,
        Channel::Blue => 3,
    }
}

/// `rgb`, an image of three channels, sampled into a mosaic of `pattern`: each site keeps
/// only its own colour. Values above the largest `significant_bits` hold are clamped.
pub fn mosaic(rgb: &Image<u16>, pattern: BayerPattern, significant_bits: u8) -> Result<Mosaic> {
    if rgb.channels() != 3 {
        return Err(Error::invalid_argument(format!(
            "a mosaic is sampled from RGB, not {} channels",
            rgb.channels()
        )));
    }
    let layout = MosaicLayout {
        width: rgb.width(),
        height: rgb.height(),
        pattern,
        significant_bits,
    };
    let max = layout.max();
    let mut sites = Vec::with_capacity(layout.sites());
    for y in 0..rgb.height() {
        for x in 0..rgb.width() {
            let pixel = rgb.pixel(x, y);
            let value = match pattern.channel_at(x, y) {
                Channel::Red => pixel[0],
                Channel::GreenOnRed | Channel::GreenOnBlue => pixel[1],
                Channel::Blue => pixel[2],
            };
            sites.push(value.min(max));
        }
    }
    Mosaic::new(layout, sites)
}

/// The four tiles' sites at plane position (`px`, `py`), in channel order.
fn tile(mosaic: &Mosaic, px: usize, py: usize) -> [u16; 4] {
    let layout = mosaic.layout();
    let width = layout.width as usize;
    let mut sites = [0; 4];
    for dy in 0..2 {
        for dx in 0..2 {
            let (x, y) = (2 * px + dx, 2 * py + dy);
            let channel = layout.pattern.channel_at(x as u32, y as u32);
            sites[channel_index(channel)] = mosaic.sites()[y * width + x];
        }
    }
    sites
}

/// The CPU twin of the split: four planes of `width / 2` × `height / 2`. An odd last column
/// or row has no tile of its own and is left out.
pub fn split_cpu(cpu: &CpuBackend, mosaic: &Mosaic) -> BayerPlanes {
    let layout = mosaic.layout();
    let (pw, ph) = ((layout.width / 2) as usize, (layout.height / 2) as usize);
    let size = pw * ph;
    let mut data = vec![0u16; 4 * size];
    if size > 0 {
        let (red, rest) = data.split_at_mut(size);
        let (green_on_red, rest) = rest.split_at_mut(size);
        let (green_on_blue, blue) = rest.split_at_mut(size);
        cpu.install(|| {
            red.par_chunks_mut(pw)
                .zip(green_on_red.par_chunks_mut(pw))
                .zip(green_on_blue.par_chunks_mut(pw))
                .zip(blue.par_chunks_mut(pw))
                .enumerate()
                .for_each(|(py, (((r, gr), gb), b))| {
                    for px in 0..pw {
                        [r[px], gr[px], gb[px], b[px]] = tile(mosaic, px, py);
                    }
                });
        });
    }
    BayerPlanes::from_data(pw as u32, ph as u32, data)
}

/// The CPU twin of the superpixel: one RGB pixel per tile, its red, the rounded mean of its
/// greens, and its blue.
pub fn superpixel_cpu(cpu: &CpuBackend, mosaic: &Mosaic) -> Image<u16> {
    let layout = mosaic.layout();
    let (pw, ph) = ((layout.width / 2) as usize, (layout.height / 2) as usize);
    let mut data = vec![0u16; 3 * pw * ph];
    if pw > 0 {
        cpu.install(|| {
            data.par_chunks_mut(3 * pw)
                .enumerate()
                .for_each(|(py, row)| {
                    for (px, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                        let [r, g1, g2, b] = tile(mosaic, px, py);
                        *pixel = [r, ((u32::from(g1) + u32::from(g2) + 1) >> 1) as u16, b];
                    }
                });
        });
    }
    Image::new(pw as u32, ph as u32, 3, data).expect("sized to fit")
}

/// The CPU twin of binning by `factor`, 1 to 4, keeping the Bayer layout: each site of the
/// result sums, or averages rounding halves up, the `factor × factor` sites of its own colour
/// in the block of tiles it stands for (see [`MosaicLayout::binned`]).
pub fn bin_cpu(cpu: &CpuBackend, mosaic: &Mosaic, factor: u32, mode: BinMode) -> Result<Mosaic> {
    let layout = mosaic.layout();
    let binned = layout.binned(factor, mode)?;
    let (width, out_width, k) = (
        layout.width as usize,
        binned.width as usize,
        factor as usize,
    );
    let n = (k * k) as u32;
    let mut sites = vec![0u16; binned.sites()];
    if out_width > 0 {
        cpu.install(|| {
            sites
                .par_chunks_mut(out_width)
                .enumerate()
                .for_each(|(y, row)| {
                    let (dy, ty) = (y & 1, (y >> 1) * k);
                    for (x, site) in row.iter_mut().enumerate() {
                        let (dx, tx) = (x & 1, (x >> 1) * k);
                        let mut sum = 0u32;
                        for j in 0..k {
                            let start = (2 * (ty + j) + dy) * width + 2 * tx + dx;
                            sum += (0..k)
                                .map(|i| u32::from(mosaic.sites()[start + 2 * i]))
                                .sum::<u32>();
                        }
                        *site = match mode {
                            BinMode::Sum => sum as u16,
                            BinMode::Mean => ((sum + n / 2) / n) as u16,
                        };
                    }
                });
        });
    }
    Mosaic::new(binned, sites)
}

/// Binning, on whichever backend `compute` chose.
pub fn bin(compute: &Compute, mosaic: &Mosaic, factor: u32, mode: BinMode) -> Result<Mosaic> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let on_device = crate::DeviceMosaic::upload(gpu, mosaic)?;
        return bin_device(gpu, &on_device, factor, mode)?.download(gpu);
    }
    bin_cpu(compute.cpu(), mosaic, factor, mode)
}

/// Binning a mosaic on the device, the result left there.
#[cfg(feature = "cuda")]
pub fn bin_device(
    gpu: &tetera_compute::CudaBackend,
    mosaic: &crate::DeviceMosaic,
    factor: u32,
    mode: BinMode,
) -> Result<crate::DeviceMosaic> {
    let binned = mosaic.layout().binned(factor, mode)?;
    let mut out = gpu.zeros::<u16>(binned.sites())?;
    gpu.bayer_bin(mosaic.buffer(), mosaic.layout(), factor, mode, &mut out)?;
    Ok(crate::DeviceMosaic::from_parts(binned, out))
}

/// The split, on whichever backend `compute` chose.
pub fn split(compute: &Compute, mosaic: &Mosaic) -> Result<BayerPlanes> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let layout = mosaic.layout();
        let input = gpu.upload(mosaic.sites())?;
        let mut out = gpu.zeros::<u16>(4 * layout.tiles())?;
        gpu.bayer_split(&input, layout, &mut out)?;
        return Ok(BayerPlanes::from_data(
            layout.width / 2,
            layout.height / 2,
            gpu.download(&out)?,
        ));
    }
    Ok(split_cpu(compute.cpu(), mosaic))
}

/// The superpixel, on whichever backend `compute` chose.
pub fn superpixel(compute: &Compute, mosaic: &Mosaic) -> Result<Image<u16>> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let on_device = crate::DeviceMosaic::upload(gpu, mosaic)?;
        return superpixel_device(gpu, &on_device)?.download(gpu);
    }
    Ok(superpixel_cpu(compute.cpu(), mosaic))
}

/// The superpixel of a mosaic on the device, left there.
#[cfg(feature = "cuda")]
pub fn superpixel_device(
    gpu: &tetera_compute::CudaBackend,
    mosaic: &crate::DeviceMosaic,
) -> Result<crate::DeviceImage<u16>> {
    let layout = mosaic.layout();
    let mut out = crate::DeviceImage::zeros(gpu, layout.width / 2, layout.height / 2, 3)?;
    gpu.bayer_superpixel(mosaic.buffer(), layout, out.buffer_mut())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATTERNS: [BayerPattern; 4] = [
        BayerPattern::Rggb,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
        BayerPattern::Bggr,
    ];

    fn cpu() -> CpuBackend {
        CpuBackend::new(3).unwrap()
    }

    /// A mosaic of `width × height` whose sites are numbered row by row.
    fn numbered(width: u32, height: u32, pattern: BayerPattern) -> Mosaic {
        let layout = MosaicLayout {
            width,
            height,
            pattern,
            significant_bits: 16,
        };
        Mosaic::new(
            layout,
            (0..layout.sites() as u32).map(|v| v as u16).collect(),
        )
        .unwrap()
    }

    #[test]
    fn sampling_keeps_each_sites_own_colour() {
        let rgb = Image::new(
            2,
            2,
            3,
            vec![10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33],
        )
        .unwrap();
        assert_eq!(
            mosaic(&rgb, BayerPattern::Rggb, 12).unwrap().sites(),
            [10, 21, 22, 33]
        );
        assert_eq!(
            mosaic(&rgb, BayerPattern::Bggr, 12).unwrap().sites(),
            [30, 21, 22, 13]
        );
        assert_eq!(
            mosaic(&rgb, BayerPattern::Grbg, 12).unwrap().sites(),
            [20, 11, 32, 23]
        );
        let bright = Image::new(1, 1, 3, vec![5_000, 0, 0]).unwrap();
        assert_eq!(
            mosaic(&bright, BayerPattern::Rggb, 12).unwrap().sites(),
            [4_095],
            "clamped"
        );
        let grey = Image::new(1, 1, 1, vec![0u16]).unwrap();
        let err = mosaic(&grey, BayerPattern::Rggb, 12).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a mosaic is sampled from RGB, not 1 channels"
        );
    }

    #[test]
    fn the_split_puts_each_colour_in_its_plane_for_every_pattern() {
        for pattern in PATTERNS {
            let m = numbered(6, 4, pattern);
            let planes = split_cpu(&cpu(), &m);
            assert_eq!((planes.width(), planes.height()), (3, 2));
            for (i, channel) in [
                Channel::Red,
                Channel::GreenOnRed,
                Channel::GreenOnBlue,
                Channel::Blue,
            ]
            .into_iter()
            .enumerate()
            {
                let plane = planes.plane(channel);
                for py in 0..2u32 {
                    for px in 0..3u32 {
                        let site = plane[(py * 3 + px) as usize];
                        let (x, y) = (u32::from(site) % 6, u32::from(site) / 6);
                        assert_eq!((x / 2, y / 2), (px, py), "{pattern:?}: from its own tile");
                        assert_eq!(pattern.channel_at(x, y), channel, "{pattern:?} plane {i}");
                    }
                }
            }
        }
        let rggb = split_cpu(&cpu(), &numbered(4, 4, BayerPattern::Rggb));
        assert_eq!(
            rggb.data(),
            [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15]
        );
    }

    #[test]
    fn an_odd_edge_is_left_out_and_a_mosaic_without_tiles_has_empty_planes() {
        let planes = split_cpu(&cpu(), &numbered(5, 3, BayerPattern::Rggb));
        assert_eq!(planes.data(), [0, 2, 1, 3, 5, 7, 6, 8]);
        let none = split_cpu(&cpu(), &numbered(3, 1, BayerPattern::Rggb));
        assert_eq!((none.width(), none.height(), none.data().len()), (1, 0, 0));
        let narrow = split_cpu(&cpu(), &numbered(1, 4, BayerPattern::Rggb));
        assert_eq!(
            (narrow.width(), narrow.height(), narrow.data().len()),
            (0, 2, 0)
        );
        let image = superpixel_cpu(&cpu(), &numbered(1, 4, BayerPattern::Rggb));
        assert_eq!(
            (image.width(), image.height(), image.data().len()),
            (0, 2, 0)
        );
    }

    #[test]
    fn the_superpixel_averages_the_greens_rounding_halves_up() {
        let image = superpixel_cpu(&cpu(), &numbered(4, 4, BayerPattern::Rggb));
        assert_eq!(image.data(), [0, 3, 5, 2, 5, 7, 8, 11, 13, 10, 13, 15]);
        let gbrg = superpixel_cpu(&cpu(), &numbered(4, 2, BayerPattern::Gbrg));
        // GBRG's tile (0, 0): green 0, blue 1, red 4, green 5.
        assert_eq!(gbrg.pixel(0, 0), [4, 3, 1]);
        let layout = MosaicLayout {
            width: 2,
            height: 2,
            pattern: BayerPattern::Rggb,
            significant_bits: 16,
        };
        let top = Mosaic::new(layout, vec![0, u16::MAX, u16::MAX, 0]).unwrap();
        assert_eq!(
            superpixel_cpu(&cpu(), &top).data(),
            [0, u16::MAX, 0],
            "no overflow"
        );
    }

    #[test]
    fn binning_keeps_each_colour_to_itself() {
        // Each colour a constant of its own: a bin that mixed colours would show it.
        for pattern in PATTERNS {
            let layout = MosaicLayout {
                width: 12,
                height: 8,
                pattern,
                significant_bits: 12,
            };
            let sites = (0..8u32)
                .flat_map(|y| (0..12u32).map(move |x| (x, y)))
                .map(|(x, y)| match pattern.channel_at(x, y) {
                    Channel::Red => 100,
                    Channel::GreenOnRed => 200,
                    Channel::GreenOnBlue => 300,
                    Channel::Blue => 400,
                })
                .collect();
            let m = Mosaic::new(layout, sites).unwrap();
            for factor in 1..=4 {
                let sum = bin_cpu(&cpu(), &m, factor, BinMode::Sum).unwrap();
                let mean = bin_cpu(&cpu(), &m, factor, BinMode::Mean).unwrap();
                assert_eq!(sum.layout().pattern, pattern);
                let n = (factor * factor) as u16;
                for (i, (&s, &a)) in sum.sites().iter().zip(mean.sites()).enumerate() {
                    let (x, y) = (i as u32 % sum.layout().width, i as u32 / sum.layout().width);
                    let own = match pattern.channel_at(x, y) {
                        Channel::Red => 100,
                        Channel::GreenOnRed => 200,
                        Channel::GreenOnBlue => 300,
                        Channel::Blue => 400,
                    };
                    assert_eq!(
                        (s, a),
                        (own * n, own),
                        "{pattern:?} by {factor} at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn binning_sums_and_averages_the_right_sites() {
        // 8 × 4, numbered: output (0, 0), red, sums sites (0, 0), (2, 0), (0, 2), (2, 2).
        let m = numbered(8, 4, BayerPattern::Rggb);
        let narrow = Mosaic::new(
            MosaicLayout {
                significant_bits: 12,
                ..*m.layout()
            },
            m.sites().to_vec(),
        )
        .unwrap();
        let sum = bin_cpu(&cpu(), &narrow, 2, BinMode::Sum).unwrap();
        assert_eq!(sum.sites(), [36, 40, 52, 56, 68, 72, 84, 88]);
        assert_eq!(
            (
                sum.layout().width,
                sum.layout().height,
                sum.layout().significant_bits
            ),
            (4, 2, 14)
        );
        let mean = bin_cpu(&cpu(), &narrow, 2, BinMode::Mean).unwrap();
        assert_eq!(mean.sites(), [9, 10, 13, 14, 17, 18, 21, 22]);
        // A mean of 2.5 rounds up: sites 1, 2, 3, 4 at one colour's places.
        let layout = MosaicLayout {
            width: 4,
            height: 4,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        };
        let mut sites = vec![0u16; 16];
        for (i, v) in [(0, 1), (2, 2), (8, 3), (10, 4)] {
            sites[i] = v;
        }
        let up = bin_cpu(
            &cpu(),
            &Mosaic::new(layout, sites).unwrap(),
            2,
            BinMode::Mean,
        )
        .unwrap();
        assert_eq!(up.sites()[0], 3);
        let nothing =
            bin_cpu(&cpu(), &numbered(1, 4, BayerPattern::Rggb), 1, BinMode::Sum).unwrap();
        assert_eq!(nothing.sites().len(), 0);
        let err =
            bin_cpu(&cpu(), &numbered(4, 4, BayerPattern::Rggb), 2, BinMode::Sum).unwrap_err();
        assert_eq!(
            err.to_string(),
            "sums of 2 × 2 sites of 16 bits need 18: bin by the mean"
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]
        #[test]
        fn binned_sums_keep_each_colours_total(
            tiles_w in 1u32..12, tiles_h in 1u32..10, factor in 1u32..=4,
            pattern in 0usize..4, seed in proptest::prelude::any::<u64>(),
        ) {
            let pattern = PATTERNS[pattern];
            let layout = MosaicLayout { width: 2 * tiles_w + 1, height: 2 * tiles_h, pattern, significant_bits: 12 };
            let mut state = seed | 1;
            let sites: Vec<u16> = (0..layout.sites())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % 4_096) as u16
                })
                .collect();
            let m = Mosaic::new(layout, sites).unwrap();
            let sum = bin_cpu(&cpu(), &m, factor, BinMode::Sum).unwrap();
            // The binned area: whole blocks of tiles, from the top left.
            let (cover_w, cover_h) = (sum.layout().width * factor, sum.layout().height * factor);
            for channel in [Channel::Red, Channel::GreenOnRed, Channel::GreenOnBlue, Channel::Blue] {
                let total = |mosaic: &Mosaic, w: u32, h: u32| -> u64 {
                    let width = mosaic.layout().width;
                    (0..h).flat_map(|y| (0..w).map(move |x| (x, y)))
                        .filter(|&(x, y)| pattern.channel_at(x, y) == channel)
                        .map(|(x, y)| u64::from(mosaic.sites()[(y * width + x) as usize]))
                        .sum()
                };
                proptest::prop_assert_eq!(
                    total(&sum, sum.layout().width, sum.layout().height),
                    total(&m, cover_w, cover_h),
                    "{:?} by {}", channel, factor
                );
            }
        }
    }

    #[test]
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let m = numbered(4, 4, BayerPattern::Grbg);
        assert_eq!(split(&compute, &m).unwrap(), split_cpu(compute.cpu(), &m));
        let twelve = Mosaic::new(
            MosaicLayout {
                significant_bits: 12,
                ..*m.layout()
            },
            m.sites().to_vec(),
        )
        .unwrap();
        assert_eq!(
            bin(&compute, &twelve, 2, BinMode::Sum).unwrap(),
            bin_cpu(compute.cpu(), &twelve, 2, BinMode::Sum).unwrap()
        );
        assert_eq!(
            superpixel(&compute, &m).unwrap(),
            superpixel_cpu(compute.cpu(), &m)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernels_and_their_twins_agree_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        assert!(compute.cuda().is_some(), "{}", compute.why());
        let cpu = CpuBackend::new(0).unwrap();
        // Every pattern at small sizes; the camera's at full frame.
        let cases: [(u32, u32, &[BayerPattern]); 4] = [
            (2, 2, &PATTERNS),
            (5, 3, &PATTERNS),
            (33, 17, &PATTERNS),
            (5_472, 3_648, &[BayerPattern::Rggb]),
        ];
        for (width, height, patterns) in cases {
            let rgb = crate::pattern::test_pattern(width, height, 12);
            for &pattern in patterns {
                let m = mosaic(&rgb, pattern, 12).unwrap();
                assert_eq!(
                    split(&compute, &m).unwrap(),
                    split_cpu(&cpu, &m),
                    "{width} × {height} {pattern:?}"
                );
                assert_eq!(superpixel(&compute, &m).unwrap(), superpixel_cpu(&cpu, &m));
                for factor in 1..=4 {
                    for mode in [BinMode::Sum, BinMode::Mean] {
                        assert_eq!(
                            bin(&compute, &m, factor, mode).unwrap(),
                            bin_cpu(&cpu, &m, factor, mode).unwrap(),
                            "{width} × {height} {pattern:?} by {factor}, {mode:?}"
                        );
                    }
                }
                let on_device = crate::DeviceMosaic::upload(&gpu, &m).unwrap();
                assert_eq!(on_device.download(&gpu).unwrap(), m);
            }
        }
    }
}
