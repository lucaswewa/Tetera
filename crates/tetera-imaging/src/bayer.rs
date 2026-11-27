//! The Bayer mosaic taken apart: its four colour planes, and a half-resolution RGB
//! superpixel, one pixel per 2 × 2 tile. And the other way, an RGB image sampled into a
//! mosaic, as a sensor would see it, for tests and test patterns.
//!
//! Each runs on the CPU here, and on CUDA through `tetera-compute`'s kernels (`bayer.cu`),
//! which compute the same.

use crate::image::{BayerPlanes, Image, Mosaic, MosaicLayout};
use rayon::prelude::*;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::frame::format::{BayerPattern, Channel};
use tetera_core::{Error, Result};

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
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let m = numbered(4, 4, BayerPattern::Grbg);
        assert_eq!(split(&compute, &m).unwrap(), split_cpu(compute.cpu(), &m));
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
                let on_device = crate::DeviceMosaic::upload(&gpu, &m).unwrap();
                assert_eq!(on_device.download(&gpu).unwrap(), m);
            }
        }
    }
}
