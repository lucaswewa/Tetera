//! Imaging kernels (P1.05a): the Bayer split, the superpixel and demosaicking on a CUDA
//! backend. Their CPU twins, and the images both work on, are in `tetera-imaging`. The kernels
//! live here because launching one is unsafe, and this is the crate allowed it (plan §5.3).
//!
//! A mosaic is [`MosaicLayout::sites`] 16-bit values, row by row, with no padding. RGB is
//! interleaved, three values a pixel.

use tetera_core::frame::format::BayerPattern;
#[cfg(feature = "cuda")]
use tetera_core::{Error, Result};

/// A demosaicking method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Demosaic {
    /// Each missing colour is the mean of its nearest two or four sites of that colour.
    Bilinear,
    /// Malvar, He and Cutler's gradient-corrected linear interpolation (ICASSP 2004): bilinear,
    /// corrected by the site's own colour's gradient, over a 5 × 5 window.
    MalvarHeCutler,
}

impl Demosaic {
    /// The smallest width and height it takes: its window reaches this far beyond a site,
    /// less one, mirrored at the edges.
    pub fn min_size(self) -> u32 {
        match self {
            Self::Bilinear => 2,
            Self::MalvarHeCutler => 3,
        }
    }
}

/// A Bayer mosaic's shape: its size, its pattern, and how many of each site's 16 bits carry
/// a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MosaicLayout {
    /// Sites in a row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Its first 2 × 2 tile.
    pub pattern: BayerPattern,
    /// Bits that carry a value, 1 to 16.
    pub significant_bits: u8,
}

impl MosaicLayout {
    /// How many sites it has.
    pub fn sites(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// How many whole 2 × 2 tiles it has: an odd last column or row has none of its own.
    pub fn tiles(&self) -> usize {
        (self.width / 2) as usize * (self.height / 2) as usize
    }

    /// The largest value a site can hold.
    pub fn max(&self) -> u16 {
        u16::MAX >> (16 - self.significant_bits.clamp(1, 16))
    }
}

/// A pattern's first site, as its column and row in an RGGB tile: how the kernels take it.
pub fn pattern_offset(pattern: BayerPattern) -> (i32, i32) {
    match pattern {
        BayerPattern::Rggb => (0, 0),
        BayerPattern::Grbg => (1, 0),
        BayerPattern::Gbrg => (0, 1),
        BayerPattern::Bggr => (1, 1),
    }
}

#[cfg(feature = "cuda")]
pub use device::{DemosaicOutput, TILE};

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::kernels::{BAYER, DEMOSAIC};
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

    /// Threads along each side of a block, for two-dimensional kernels.
    pub const TILE: u32 = 16;

    /// What a demosaic kernel writes: 16-bit samples, rounded and clamped, or exact floats.
    pub trait DemosaicOutput: DeviceRepr + ValidAsZeroBits + Copy + Unpin {
        /// The kernels' suffix for it.
        const SUFFIX: &'static str;
    }

    impl DemosaicOutput for u16 {
        const SUFFIX: &'static str = "u16";
    }

    impl DemosaicOutput for f32 {
        const SUFFIX: &'static str = "f32";
    }

    /// The mosaic's size as the kernels' `int`s, checked to fit them, and against its buffer
    /// and the output's.
    fn checked(
        input: usize,
        layout: &MosaicLayout,
        output: usize,
        needed: usize,
    ) -> Result<(i32, i32)> {
        let (width, height) = (layout.width, layout.height);
        let (Ok(w), Ok(h), Ok(_)) = (
            i32::try_from(width),
            i32::try_from(height),
            i32::try_from(layout.sites() * 3),
        ) else {
            return Err(Error::invalid_argument(format!(
                "a {width} × {height} mosaic is too large for one launch"
            )));
        };
        if input != layout.sites() {
            return Err(Error::invalid_argument(format!(
                "a {width} × {height} mosaic has {} sites, not {input}",
                layout.sites()
            )));
        }
        if output != needed {
            return Err(Error::invalid_argument(format!(
                "the output holds {output} values, not the {needed} it needs"
            )));
        }
        Ok((w, h))
    }

    fn grid(cols: u32, rows: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (cols.div_ceil(TILE), rows.div_ceil(TILE), 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        }
    }

    fn launch_failed(what: &str, e: cudarc::driver::DriverError) -> Error {
        Error::hardware(format!("the {what} kernel wouldn't launch: {e:?}"))
    }

    impl CudaBackend {
        /// The mosaic's four colour planes, back to back in the order red, green on a red row,
        /// green on a blue row, blue; each `width / 2` × `height / 2`. An odd last column or
        /// row has no tile of its own and is left out.
        pub fn bayer_split(
            &self,
            input: &DeviceBuffer<u16>,
            layout: &MosaicLayout,
            out: &mut DeviceBuffer<u16>,
        ) -> Result<()> {
            let dims = checked(input.len(), layout, out.len(), 4 * layout.tiles())?;
            self.tiles_kernel("bayer_split_u16", input, dims, layout, out)
        }

        /// One RGB pixel per 2 × 2 tile: its red, the rounded mean of its two greens, and its
        /// blue; `width / 2` × `height / 2` pixels.
        pub fn bayer_superpixel(
            &self,
            input: &DeviceBuffer<u16>,
            layout: &MosaicLayout,
            out: &mut DeviceBuffer<u16>,
        ) -> Result<()> {
            let dims = checked(input.len(), layout, out.len(), 3 * layout.tiles())?;
            self.tiles_kernel("bayer_superpixel_u16", input, dims, layout, out)
        }

        fn tiles_kernel(
            &self,
            name: &str,
            input: &DeviceBuffer<u16>,
            (w, h): (i32, i32),
            layout: &MosaicLayout,
            out: &mut DeviceBuffer<u16>,
        ) -> Result<()> {
            if layout.tiles() == 0 {
                return Ok(());
            }
            let kernel = self.module(&BAYER)?.function(name)?;
            let (ox, oy) = pattern_offset(layout.pattern);
            let config = grid(layout.width / 2, layout.height / 2);
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(&w)
                .arg(&h)
                .arg(&ox)
                .arg(&oy);
            // SAFETY: both kernels take (unsigned short* out, const unsigned short* in, int
            // width, int height, int ox, int oy), matched here in order and type. A thread
            // touches only its tile, (2px..2px+2, 2py..2py+2) of the mosaic, for px < width / 2
            // and py < height / 2, so it reads inside `input`, which holds width × height
            // sites (checked above). It writes 4 values (split) or 3 (superpixel) per tile,
            // inside `out`, whose length was checked to be exactly that.
            unsafe { launch.launch(config) }.map_err(|e| launch_failed(name, e))?;
            Ok(())
        }

        /// The mosaic demosaicked to interleaved RGB, `width × height` pixels. In 16 bits,
        /// each value is rounded and clamped to `0..=layout.max()`; as floats, it is exact,
        /// and may fall below 0 or above the maximum.
        pub fn demosaic<T: DemosaicOutput>(
            &self,
            method: Demosaic,
            input: &DeviceBuffer<u16>,
            layout: &MosaicLayout,
            out: &mut DeviceBuffer<T>,
        ) -> Result<()> {
            let (w, h) = checked(input.len(), layout, out.len(), 3 * layout.sites())?;
            let min = method.min_size();
            if layout.width < min || layout.height < min {
                return Err(Error::invalid_argument(format!(
                    "{method:?} needs a mosaic of at least {min} × {min}, not {} × {}",
                    layout.width, layout.height
                )));
            }
            let name = match method {
                Demosaic::Bilinear => format!("demosaic_bilinear_{}", T::SUFFIX),
                Demosaic::MalvarHeCutler => format!("demosaic_mhc_{}", T::SUFFIX),
            };
            let kernel = self.module(&DEMOSAIC)?.function(&name)?;
            let (ox, oy) = pattern_offset(layout.pattern);
            let maxv = i32::from(layout.max());
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(&w)
                .arg(&h)
                .arg(&ox)
                .arg(&oy)
                .arg(&maxv);
            // SAFETY: the four kernels take (T* out, const unsigned short* in, int width, int
            // height, int ox, int oy, int maxv), with T unsigned short for `_u16` and float for
            // `_f32`, the name chosen from `T`; matched here in order and type. A thread writes
            // its pixel's three values, at (y × width + x) × 3, for x < width and y < height,
            // inside `out`, checked to hold width × height × 3. It reads sites mirrored into
            // 0..width and 0..height, which needs width and height of at least the method's
            // reach plus one, checked above, inside `input`, checked to hold width × height.
            unsafe { launch.launch(grid(layout.width, layout.height)) }
                .map_err(|e| launch_failed(&name, e))?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_map_to_their_offset_in_an_rggb_tile() {
        for pattern in [
            BayerPattern::Rggb,
            BayerPattern::Grbg,
            BayerPattern::Gbrg,
            BayerPattern::Bggr,
        ] {
            let (ox, oy) = pattern_offset(pattern);
            // The pattern's first site is where RGGB's own tile has that colour.
            assert_eq!(
                pattern.channel_at(0, 0),
                BayerPattern::Rggb.channel_at(ox as u32, oy as u32),
                "{pattern:?}"
            );
        }
        assert_eq!(pattern_offset(BayerPattern::Bggr), (1, 1));
        assert_eq!(
            (
                Demosaic::Bilinear.min_size(),
                Demosaic::MalvarHeCutler.min_size()
            ),
            (2, 3)
        );
    }

    #[test]
    fn a_layout_counts_its_sites_tiles_and_largest_value() {
        let layout = MosaicLayout {
            width: 9,
            height: 7,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        };
        assert_eq!(
            (layout.sites(), layout.tiles(), layout.max()),
            (63, 12, 4_095)
        );
        let full = MosaicLayout {
            significant_bits: 16,
            ..layout
        };
        assert_eq!(full.max(), u16::MAX);
        assert_eq!(
            MosaicLayout {
                significant_bits: 1,
                ..layout
            }
            .max(),
            1
        );
        assert_eq!(
            MosaicLayout {
                significant_bits: 0,
                ..layout
            }
            .max(),
            1,
            "read as 1 bit"
        );
    }

    #[cfg(feature = "cuda")]
    mod on_the_gpu {
        use super::*;
        use tetera_core::ErrorKind;

        fn layout(width: u32, height: u32, pattern: BayerPattern) -> MosaicLayout {
            MosaicLayout {
                width,
                height,
                pattern,
                significant_bits: 12,
            }
        }

        #[test]
        fn the_split_and_the_superpixel_take_each_tile_apart() {
            let gpu = crate::cuda_or_skip!();
            // A 4 × 4 mosaic whose sites are numbered 0 to 15, row by row.
            let input = gpu.upload(&(0..16).collect::<Vec<u16>>()).unwrap();
            let mut planes = gpu.zeros::<u16>(16).unwrap();
            gpu.bayer_split(&input, &layout(4, 4, BayerPattern::Rggb), &mut planes)
                .unwrap();
            let expected = [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15];
            assert_eq!(gpu.download(&planes).unwrap(), expected);
            // BGGR's first site is blue: the planes come out in the other order.
            gpu.bayer_split(&input, &layout(4, 4, BayerPattern::Bggr), &mut planes)
                .unwrap();
            let bggr = [5, 7, 13, 15, 4, 6, 12, 14, 1, 3, 9, 11, 0, 2, 8, 10];
            assert_eq!(gpu.download(&planes).unwrap(), bggr);
            let mut rgb = gpu.zeros::<u16>(12).unwrap();
            gpu.bayer_superpixel(&input, &layout(4, 4, BayerPattern::Rggb), &mut rgb)
                .unwrap();
            // Tile (0, 0): red 0, greens 1 and 4 → 3 (2.5 rounded up), blue 5.
            assert_eq!(
                gpu.download(&rgb).unwrap(),
                [0, 3, 5, 2, 5, 7, 8, 11, 13, 10, 13, 15]
            );
        }

        #[test]
        fn an_odd_edge_is_left_out_and_a_mosaic_without_tiles_launches_nothing() {
            let gpu = crate::cuda_or_skip!();
            // 5 × 3: two tiles, (0..2, 0..2) and (2..4, 0..2); column 4 and row 2 are left out.
            let input = gpu.upload(&(0..15).collect::<Vec<u16>>()).unwrap();
            let mut planes = gpu.zeros::<u16>(8).unwrap();
            gpu.bayer_split(&input, &layout(5, 3, BayerPattern::Rggb), &mut planes)
                .unwrap();
            assert_eq!(gpu.download(&planes).unwrap(), [0, 2, 1, 3, 5, 7, 6, 8]);
            let one_row = gpu.upload(&[1u16, 2, 3]).unwrap();
            let mut nothing = gpu.zeros::<u16>(0).unwrap();
            let flat = layout(3, 1, BayerPattern::Rggb);
            gpu.bayer_split(&one_row, &flat, &mut nothing).unwrap();
            gpu.bayer_superpixel(&one_row, &flat, &mut nothing).unwrap();
        }

        #[test]
        fn a_flat_grey_mosaic_demosaicks_to_flat_grey() {
            let gpu = crate::cuda_or_skip!();
            let input = gpu.upload(&[1_000u16; 7 * 5]).unwrap();
            let grbg = layout(7, 5, BayerPattern::Grbg);
            for method in [Demosaic::Bilinear, Demosaic::MalvarHeCutler] {
                let mut out = gpu.zeros::<u16>(7 * 5 * 3).unwrap();
                gpu.demosaic(method, &input, &grbg, &mut out).unwrap();
                assert!(gpu.download(&out).unwrap().iter().all(|&v| v == 1_000));
                let mut floats = gpu.zeros::<f32>(7 * 5 * 3).unwrap();
                gpu.demosaic(method, &input, &grbg, &mut floats).unwrap();
                assert!(gpu.download(&floats).unwrap().iter().all(|&v| v == 1_000.0));
            }
        }

        #[test]
        fn sixteen_bit_values_are_clamped_and_floats_are_not() {
            let gpu = crate::cuda_or_skip!();
            // A bright red site among dark ones, at (4, 4) of 9 × 9, away from the edges:
            // Malvar-He-Cutler overshoots below zero two sites away.
            let mut sites = vec![0u16; 81];
            sites[4 * 9 + 4] = 4_000;
            let input = gpu.upload(&sites).unwrap();
            let mut out = gpu.zeros::<u16>(243).unwrap();
            let mut floats = gpu.zeros::<f32>(243).unwrap();
            let mhc = Demosaic::MalvarHeCutler;
            let eight_bits = MosaicLayout {
                significant_bits: 8,
                ..layout(9, 9, BayerPattern::Rggb)
            };
            gpu.demosaic(mhc, &input, &eight_bits, &mut out).unwrap();
            gpu.demosaic(mhc, &input, &eight_bits, &mut floats).unwrap();
            let (out, floats) = (gpu.download(&out).unwrap(), gpu.download(&floats).unwrap());
            // At the red site: red 4,000, clamped to 255 in 16 bits; green 8 × 4,000 / 16.
            let site = (4 * 9 + 4) * 3;
            assert_eq!((out[site], floats[site]), (255, 4_000.0));
            assert_eq!(floats[site + 1], 2_000.0);
            // At (2, 4), red too, green from the bright site: −2 × 4,000 / 16 = −500, clamped
            // to 0.
            let two_left = (4 * 9 + 2) * 3 + 1;
            assert_eq!((out[two_left], floats[two_left]), (0, -500.0));
        }

        #[test]
        fn mismatched_and_undersized_mosaics_are_refused() {
            let gpu = crate::cuda_or_skip!();
            let bilinear = Demosaic::Bilinear;
            let input = gpu.upload(&[0u16; 12]).unwrap();
            let mut out = gpu.zeros::<u16>(36).unwrap();
            let err = gpu
                .demosaic(
                    bilinear,
                    &input,
                    &layout(4, 4, BayerPattern::Rggb),
                    &mut out,
                )
                .unwrap_err();
            assert_eq!(err.to_string(), "a 4 × 4 mosaic has 16 sites, not 12");
            let mut short = gpu.zeros::<u16>(35).unwrap();
            let err = gpu
                .demosaic(
                    bilinear,
                    &input,
                    &layout(4, 3, BayerPattern::Rggb),
                    &mut short,
                )
                .unwrap_err();
            assert_eq!(
                err.to_string(),
                "the output holds 35 values, not the 36 it needs"
            );
            let narrow = gpu.upload(&[0u16; 8]).unwrap();
            let two_by_four = layout(2, 4, BayerPattern::Rggb);
            let mut out = gpu.zeros::<u16>(24).unwrap();
            let err = gpu
                .demosaic(Demosaic::MalvarHeCutler, &narrow, &two_by_four, &mut out)
                .unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidArgument);
            assert_eq!(
                err.to_string(),
                "MalvarHeCutler needs a mosaic of at least 3 × 3, not 2 × 4"
            );
            assert!(
                gpu.demosaic(bilinear, &narrow, &two_by_four, &mut out)
                    .is_ok()
            );
            let four_by_two = layout(4, 2, BayerPattern::Rggb);
            let narrow_rows = gpu.upload(&[0u16; 8]).unwrap();
            let err = gpu
                .demosaic(
                    Demosaic::MalvarHeCutler,
                    &narrow_rows,
                    &four_by_two,
                    &mut out,
                )
                .unwrap_err();
            assert!(err.to_string().ends_with("not 4 × 2"), "{err}");
            let mut out = gpu.zeros::<u16>(36).unwrap();
            let four_by_three = gpu.upload(&[0u16; 12]).unwrap();
            let smallest = layout(4, 3, BayerPattern::Rggb);
            assert!(
                gpu.demosaic(
                    Demosaic::MalvarHeCutler,
                    &four_by_three,
                    &smallest,
                    &mut out
                )
                .is_ok(),
                "3 rows are enough"
            );
            // 30,000 × 30,000 sites fit an int, but three values each don't.
            let huge = layout(30_000, 30_000, BayerPattern::Rggb);
            let err = gpu
                .demosaic(Demosaic::Bilinear, &narrow, &huge, &mut out)
                .unwrap_err();
            assert_eq!(
                err.to_string(),
                "a 30000 × 30000 mosaic is too large for one launch"
            );
            let wide = layout(u32::MAX, 1, BayerPattern::Rggb);
            assert!(gpu.bayer_split(&narrow, &wide, &mut out).is_err());
            let tall = layout(1, u32::MAX, BayerPattern::Rggb);
            assert!(gpu.bayer_superpixel(&narrow, &tall, &mut out).is_err());
            let mut planes = gpu.zeros::<u16>(3).unwrap();
            assert!(gpu.bayer_split(&narrow, &two_by_four, &mut planes).is_err());
            assert!(
                gpu.bayer_superpixel(&narrow, &two_by_four, &mut planes)
                    .is_err()
            );
        }
    }
}
