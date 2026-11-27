//! Imaging kernels: the Bayer split, the superpixel and demosaicking (P1.05a); binning,
//! resizing and the display transform (P1.05b), on a CUDA backend. Their CPU twins, and the images both work on, are in `tetera-imaging`. The kernels
//! live here because launching one is unsafe, and this is the crate allowed it (plan §5.3).
//!
//! A mosaic is [`MosaicLayout::sites`] 16-bit values, row by row, with no padding. RGB is
//! interleaved, three values a pixel.

use tetera_core::frame::format::BayerPattern;
use tetera_core::{Error, Rect, Result};

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

/// How binning combines a block's sites of one colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinMode {
    /// Their sum, which needs more significant bits.
    Sum,
    /// Their mean, rounded halves up, in the mosaic's own bits.
    Mean,
}

impl MosaicLayout {
    /// The layout after binning by `factor`, 1 to 4, keeping the Bayer pattern: each 2 × 2
    /// tile of the result stands for `factor × factor` tiles, and the tiles left over at the
    /// right and the bottom are left out. A sum adds the bits it needs, up to 16.
    pub fn binned(&self, factor: u32, mode: BinMode) -> Result<Self> {
        if !(1..=4).contains(&factor) {
            return Err(Error::invalid_argument(format!(
                "binning takes a factor of 1 to 4, not {factor}"
            )));
        }
        let extra = match mode {
            BinMode::Mean => 0,
            // The bits that factor² more values need: ⌈log2(factor²)⌉.
            BinMode::Sum => 32 - (factor * factor - 1).leading_zeros() as u8,
        };
        let bits = self.significant_bits + extra;
        if bits > 16 {
            return Err(Error::invalid_argument(format!(
                "sums of {factor} × {factor} sites of {} bits need {bits}: bin by the mean",
                self.significant_bits
            )));
        }
        Ok(Self {
            width: 2 * (self.width / 2 / factor),
            height: 2 * (self.height / 2 / factor),
            pattern: self.pattern,
            significant_bits: bits,
        })
    }
}

/// How the ISP reduces a raw frame first (P1.06a).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduce {
    /// The whole frame.
    Full,
    /// A region of it, its pattern moved for an odd column or row.
    Crop(Rect),
    /// Binned by 2, 3 or 4, by the mean, keeping the Bayer layout.
    Bin(u32),
}

impl MosaicLayout {
    /// The layout after `reduce`.
    pub fn reduced(&self, reduce: Reduce) -> Result<Self> {
        match reduce {
            Reduce::Full => Ok(*self),
            Reduce::Crop(roi) => {
                let whole = Rect::new(0, 0, self.width, self.height);
                if roi.is_empty() || !whole.contains_rect(&roi) {
                    return Err(Error::invalid_argument(format!(
                        "{roi:?} isn't inside the {} × {} frame",
                        self.width, self.height
                    )));
                }
                Ok(Self {
                    width: roi.width,
                    height: roi.height,
                    pattern: self.pattern.cropped(roi.x, roi.y),
                    ..*self
                })
            }
            Reduce::Bin(factor) => self.binned(factor, BinMode::Mean),
        }
    }
}

/// A resize weight's unit: an axis's weights for one output sum to this.
pub const WEIGHT_ONE: u16 = 16_384;

/// Entries in one channel's display table: one for every 16-bit value.
pub const TABLE_SIZE: usize = 65_536;

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
pub use device::{DemosaicOutput, DeviceAxis, DeviceElement, TILE};

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::kernels::{BAYER, DEMOSAIC, DISPLAY, RESIZE};
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

    /// Threads along each side of a block, for two-dimensional kernels.
    pub const TILE: u32 = 16;

    /// What a device buffer of an image may hold.
    pub trait DeviceElement: DeviceRepr + ValidAsZeroBits + Copy + Unpin {}

    impl<T: DeviceRepr + ValidAsZeroBits + Copy + Unpin> DeviceElement for T {}

    /// What a demosaic kernel writes: 16-bit samples, rounded and clamped, or exact floats.
    pub trait DemosaicOutput: DeviceElement {
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

    fn int(value: usize, what: &str) -> Result<i32> {
        i32::try_from(value).map_err(|_| {
            Error::invalid_argument(format!("{what} of {value} is too large for one launch"))
        })
    }

    /// One axis of a resize, on the device: for each of `output` columns (or rows), the first
    /// of `input` it reads, and `taps` weights. Every window was checked to lie inside the
    /// input when it was uploaded.
    #[derive(Debug)]
    pub struct DeviceAxis {
        input: u32,
        output: u32,
        taps: u32,
        pub(crate) start: DeviceBuffer<i32>,
        pub(crate) weights: DeviceBuffer<u16>,
    }

    impl DeviceAxis {
        /// Columns or rows read.
        pub fn input(&self) -> u32 {
            self.input
        }

        /// Columns or rows written.
        pub fn output(&self) -> u32 {
            self.output
        }

        /// Weights for each.
        pub fn taps(&self) -> u32 {
            self.taps
        }
    }

    impl CudaBackend {
        /// The mosaic binned by `factor`, keeping its Bayer layout, into `out`, which holds
        /// the binned layout's sites. It gives that layout.
        pub fn bayer_bin(
            &self,
            input: &DeviceBuffer<u16>,
            layout: &MosaicLayout,
            factor: u32,
            mode: BinMode,
            out: &mut DeviceBuffer<u16>,
        ) -> Result<MosaicLayout> {
            let binned = layout.binned(factor, mode)?;
            let (w, _) = checked(input.len(), layout, out.len(), binned.sites())?;
            if binned.sites() == 0 {
                return Ok(binned);
            }
            let kernel = self.module(&BAYER)?.function("bayer_bin_u16")?;
            let (k, mean) = (factor as i32, i32::from(mode == BinMode::Mean));
            let (ow, oh) = (binned.width as i32, binned.height as i32);
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(&w)
                .arg(&k)
                .arg(&mean)
                .arg(&ow)
                .arg(&oh);
            // SAFETY: the kernel takes (unsigned short* out, const unsigned short* in, int
            // width, int factor, int mean, int out_width, int out_height), matched here in
            // order and type. A thread writes its one site, y × out_width + x for x < out_width
            // and y < out_height, inside `out`, checked to hold out_width × out_height. It
            // reads columns up to 2 × (out_width / 2) × factor − 1, which is less than width,
            // since out_width / 2 = width / 2 / factor; rows likewise. So it reads inside
            // `input`, checked to hold width × height.
            unsafe { launch.launch(grid(binned.width, binned.height)) }
                .map_err(|e| launch_failed("bayer_bin_u16", e))?;
            Ok(binned)
        }

        /// One axis of a resize, uploaded once it is checked: `start` holds an entry per
        /// output, `weights` holds `taps` per output, and each window lies inside the input.
        pub fn upload_axis(
            &self,
            input: u32,
            output: u32,
            taps: u32,
            start: &[i32],
            weights: &[u16],
        ) -> Result<DeviceAxis> {
            if taps == 0 || taps > input {
                return Err(Error::invalid_argument(format!(
                    "{taps} taps can't read from {input} inputs"
                )));
            }
            if start.len() != output as usize || weights.len() != output as usize * taps as usize {
                return Err(Error::invalid_argument(format!(
                    "an axis of {output} outputs and {taps} taps needs {output} starts and {} weights, not {} and {}",
                    output as usize * taps as usize,
                    start.len(),
                    weights.len()
                )));
            }
            let last = i64::from(input) - i64::from(taps);
            if let Some((i, &s)) = start
                .iter()
                .enumerate()
                .find(|&(_, &s)| s < 0 || i64::from(s) > last)
            {
                return Err(Error::invalid_argument(format!(
                    "output {i}'s window starts at {s}, outside 0..={last}"
                )));
            }
            Ok(DeviceAxis {
                input,
                output,
                taps,
                start: self.upload(start)?,
                weights: self.upload(weights)?,
            })
        }

        /// `input`, `x.input() × y.input()` pixels of `channels` interleaved values, resized to
        /// `x.output() × y.output()` into `out`.
        pub fn resize(
            &self,
            input: &DeviceBuffer<u16>,
            channels: u8,
            x: &DeviceAxis,
            y: &DeviceAxis,
            out: &mut DeviceBuffer<u16>,
        ) -> Result<()> {
            let c = usize::from(channels);
            let (inputs, outputs) = (
                x.input as usize * y.input as usize * c,
                x.output as usize * y.output as usize * c,
            );
            if channels == 0 || input.len() != inputs || out.len() != outputs {
                return Err(Error::invalid_argument(format!(
                    "a resize from {} × {} to {} × {} of {channels} channels needs {inputs} values in and {outputs} out, not {} and {}",
                    x.input,
                    y.input,
                    x.output,
                    y.output,
                    input.len(),
                    out.len()
                )));
            }
            if outputs == 0 {
                return Ok(());
            }
            let (in_width, ow, oh) = (
                int(x.input as usize, "a width")?,
                x.output as i32,
                y.output as i32,
            );
            let (channels, x_taps, y_taps) = (i32::from(channels), x.taps as i32, y.taps as i32);
            int(inputs.max(outputs), "an image")?;
            let kernel = self.module(&RESIZE)?.function("resize_u16")?;
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(&in_width)
                .arg(&channels)
                .arg(&ow)
                .arg(&oh)
                .arg(x.start.slice())
                .arg(x.weights.slice())
                .arg(&x_taps)
                .arg(y.start.slice())
                .arg(y.weights.slice())
                .arg(&y_taps);
            // SAFETY: the kernel takes (unsigned short* out, const unsigned short* in, int
            // in_width, int channels, int out_width, int out_height, const int* x_start, const
            // unsigned short* x_weights, int x_taps, const int* y_start, const unsigned short*
            // y_weights, int y_taps), matched here in order and type. A thread writes its
            // pixel's `channels` values inside `out`, checked to hold out_width × out_height ×
            // channels. It reads x_start[x] and y_start[y], and x_taps (y_taps) weights from
            // x × x_taps (y × y_taps), inside the tables, whose lengths `upload_axis` checked.
            // It reads input columns x_start[x]..x_start[x] + x_taps and rows likewise, which
            // `upload_axis` checked lie inside the axes' inputs, whose product × channels
            // `input` was checked to hold.
            unsafe { launch.launch(grid(x.output, y.output)) }
                .map_err(|e| launch_failed("resize_u16", e))?;
            Ok(())
        }

        /// The display transform: `input`, interleaved values of `channels`, each looked up in
        /// its channel's table of `tables`, [`TABLE_SIZE`] entries a channel, into `out`.
        pub fn display(
            &self,
            input: &DeviceBuffer<u16>,
            channels: u8,
            tables: &DeviceBuffer<u8>,
            out: &mut DeviceBuffer<u8>,
        ) -> Result<()> {
            let c = usize::from(channels);
            if channels == 0
                || !input.len().is_multiple_of(c)
                || out.len() != input.len()
                || tables.len() != c * TABLE_SIZE
            {
                return Err(Error::invalid_argument(format!(
                    "a display transform of {channels} channels needs whole pixels, an output as long as its input, and {} table entries; it has {} values in, {} out and {} entries",
                    c * TABLE_SIZE,
                    input.len(),
                    out.len(),
                    tables.len()
                )));
            }
            if input.is_empty() {
                return Ok(());
            }
            let kernel = self.module(&DISPLAY)?.function("display_u16_u8")?;
            let (values, channels) = (input.len() as i64, i32::from(channels));
            let config = LaunchConfig {
                grid_dim: ((input.len() as u64).div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(&values)
                .arg(&channels)
                .arg(tables.slice());
            // SAFETY: the kernel takes (unsigned char* out, const unsigned short* in, long long
            // values, int channels, const unsigned char* tables), matched here in order and
            // type. Thread i < values reads in[i] and writes out[i], inside both, which hold
            // `values` each (checked). It reads tables[(i % channels) × 65536 + in[i]], below
            // channels × 65536, the tables' checked length, since in[i] is below 65536.
            unsafe { launch.launch(config) }.map_err(|e| launch_failed("display_u16_u8", e))?;
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

    #[test]
    fn binning_keeps_the_pattern_and_counts_the_bits_a_sum_needs() {
        let layout = MosaicLayout {
            width: 21,
            height: 13,
            pattern: BayerPattern::Gbrg,
            significant_bits: 12,
        };
        let two = layout.binned(2, BinMode::Sum).unwrap();
        // 10 tiles across, 6 down; binned by 2, 5 and 3: 10 × 6 sites.
        assert_eq!(
            (two.width, two.height, two.pattern, two.significant_bits),
            (10, 6, BayerPattern::Gbrg, 14)
        );
        let three = layout.binned(3, BinMode::Sum).unwrap();
        assert_eq!(
            (three.width, three.height, three.significant_bits),
            (6, 4, 16)
        );
        assert_eq!(layout.binned(4, BinMode::Sum).unwrap().significant_bits, 16);
        assert_eq!(
            layout.binned(4, BinMode::Mean).unwrap().significant_bits,
            12
        );
        assert_eq!(
            layout.binned(1, BinMode::Sum).unwrap(),
            MosaicLayout {
                width: 20,
                height: 12,
                ..layout
            }
        );
        let fourteen = MosaicLayout {
            significant_bits: 14,
            ..layout
        };
        let err = fourteen.binned(3, BinMode::Sum).unwrap_err();
        assert_eq!(
            err.to_string(),
            "sums of 3 × 3 sites of 14 bits need 18: bin by the mean"
        );
        assert_eq!(
            fourteen.binned(2, BinMode::Sum).unwrap().significant_bits,
            16
        );
        for factor in [0, 5] {
            let err = layout.binned(factor, BinMode::Mean).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!("binning takes a factor of 1 to 4, not {factor}")
            );
        }
        let small = MosaicLayout {
            width: 5,
            height: 3,
            ..layout
        };
        assert_eq!(
            small.binned(2, BinMode::Mean).unwrap().sites(),
            0,
            "no whole block"
        );
    }

    #[test]
    fn reducing_crops_moving_the_pattern_or_bins_by_the_mean() {
        let raw = MosaicLayout {
            width: 20,
            height: 12,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        };
        assert_eq!(raw.reduced(Reduce::Full).unwrap(), raw);
        let crop = raw.reduced(Reduce::Crop(Rect::new(3, 1, 10, 8))).unwrap();
        assert_eq!(
            (crop.width, crop.height, crop.pattern, crop.significant_bits),
            (10, 8, BayerPattern::Bggr, 12)
        );
        assert_eq!(
            raw.reduced(Reduce::Crop(Rect::new(0, 0, 20, 12))).unwrap(),
            raw,
            "the whole frame"
        );
        let err = raw
            .reduced(Reduce::Crop(Rect::new(15, 0, 10, 8)))
            .unwrap_err();
        assert!(
            err.to_string().ends_with("isn't inside the 20 × 12 frame"),
            "{err}"
        );
        assert!(
            raw.reduced(Reduce::Crop(Rect::new(0, 0, 0, 8))).is_err(),
            "empty"
        );
        assert!(
            raw.reduced(Reduce::Crop(Rect::new(0, 0, 8, 0))).is_err(),
            "empty"
        );
        let bin = raw.reduced(Reduce::Bin(2)).unwrap();
        assert_eq!(
            (bin.width, bin.height, bin.significant_bits),
            (10, 6, 12),
            "the mean keeps the bits"
        );
        assert!(raw.reduced(Reduce::Bin(5)).is_err());
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
        fn binning_sums_or_averages_each_colour_within_its_block() {
            let gpu = crate::cuda_or_skip!();
            // 8 × 4, numbered row by row: binned by 2, one block of 2 × 2 tiles a side.
            let input = gpu.upload(&(0..32).collect::<Vec<u16>>()).unwrap();
            let layout = layout(8, 4, BayerPattern::Rggb);
            let mut out = gpu.zeros::<u16>(8).unwrap();
            let binned = gpu
                .bayer_bin(&input, &layout, 2, BinMode::Sum, &mut out)
                .unwrap();
            assert_eq!(
                (binned.width, binned.height, binned.significant_bits),
                (4, 2, 14)
            );
            // Output (0, 0), red: sites (0, 0), (2, 0), (0, 2), (2, 2) = 0 + 2 + 16 + 18.
            assert_eq!(
                gpu.download(&out).unwrap(),
                [36, 40, 52, 56, 68, 72, 84, 88]
            );
            gpu.bayer_bin(&input, &layout, 2, BinMode::Mean, &mut out)
                .unwrap();
            assert_eq!(gpu.download(&out).unwrap(), [9, 10, 13, 14, 17, 18, 21, 22]);
            // 4 × 4 by 2: one tile; 1 × 4 by 1: no tiles, nothing launched.
            let mut nothing = gpu.zeros::<u16>(0).unwrap();
            let narrow = gpu.upload(&[1u16, 2, 3, 4]).unwrap();
            gpu.bayer_bin(
                &narrow,
                &MosaicLayout {
                    width: 1,
                    height: 4,
                    ..layout
                },
                1,
                BinMode::Sum,
                &mut nothing,
            )
            .unwrap();
            let mut short = gpu.zeros::<u16>(7).unwrap();
            let err = gpu
                .bayer_bin(&input, &layout, 2, BinMode::Sum, &mut short)
                .unwrap_err();
            assert_eq!(
                err.to_string(),
                "the output holds 7 values, not the 8 it needs"
            );
            assert!(
                gpu.bayer_bin(&input, &layout, 5, BinMode::Sum, &mut out)
                    .is_err()
            );
        }

        #[test]
        fn a_resize_weighs_its_window_and_rounds_once() {
            let gpu = crate::cuda_or_skip!();
            // 4 × 1 of two channels to 2 × 1: each output averages two inputs, half each.
            let input = gpu
                .upload(&[10u16, 100, 20, 200, 31, 300, 41, 400])
                .unwrap();
            let x = gpu
                .upload_axis(4, 2, 2, &[0, 2], &[8_192, 8_192, 8_192, 8_192])
                .unwrap();
            let y = gpu.upload_axis(1, 1, 1, &[0], &[16_384]).unwrap();
            assert_eq!((x.input(), x.output(), x.taps()), (4, 2, 2));
            let mut out = gpu.zeros::<u16>(4).unwrap();
            gpu.resize(&input, 2, &x, &y, &mut out).unwrap();
            // (10 + 20) / 2 = 15 and (31 + 41) / 2 = 36, exactly.
            assert_eq!(gpu.download(&out).unwrap(), [15, 150, 36, 350]);
            // 2 × 2 to 1 × 1: both rows are read, a quarter each: 25.
            let square = gpu.upload(&[10u16, 20, 30, 40]).unwrap();
            let half = gpu.upload_axis(2, 1, 2, &[0], &[8_192, 8_192]).unwrap();
            let mut mean = gpu.zeros::<u16>(1).unwrap();
            gpu.resize(&square, 1, &half, &half, &mut mean).unwrap();
            assert_eq!(gpu.download(&mean).unwrap(), [25]);
            // 2 × 2 to 2 × 2, each output its own input: two rows out.
            let copy = gpu
                .upload_axis(2, 2, 1, &[0, 1], &[16_384, 16_384])
                .unwrap();
            let mut same = gpu.zeros::<u16>(4).unwrap();
            gpu.resize(&square, 1, &copy, &copy, &mut same).unwrap();
            assert_eq!(gpu.download(&same).unwrap(), [10, 20, 30, 40]);
            // Four inputs of a quarter each: 1 to 4 average 2.5, which rounds up.
            let ramp = gpu.upload(&[1u16, 2, 3, 4]).unwrap();
            let x4 = gpu.upload_axis(4, 1, 4, &[0], &[4_096; 4]).unwrap();
            let mut one = gpu.zeros::<u16>(1).unwrap();
            gpu.resize(&ramp, 1, &x4, &y, &mut one).unwrap();
            assert_eq!(gpu.download(&one).unwrap(), [3]);
        }

        #[test]
        fn axes_and_resizes_that_dont_fit_are_refused() {
            let gpu = crate::cuda_or_skip!();
            let err = gpu.upload_axis(3, 2, 2, &[0, 2], &[1; 4]).unwrap_err();
            assert_eq!(
                err.to_string(),
                "output 1's window starts at 2, outside 0..=1"
            );
            assert!(gpu.upload_axis(3, 2, 2, &[-1, 0], &[1; 4]).is_err());
            let err = gpu.upload_axis(3, 2, 4, &[0, 0], &[1; 8]).unwrap_err();
            assert_eq!(err.to_string(), "4 taps can't read from 3 inputs");
            assert!(gpu.upload_axis(3, 2, 0, &[0, 0], &[]).is_err());
            let err = gpu.upload_axis(3, 2, 2, &[0], &[1; 4]).unwrap_err();
            assert_eq!(
                err.to_string(),
                "an axis of 2 outputs and 2 taps needs 2 starts and 4 weights, not 1 and 4"
            );
            assert!(gpu.upload_axis(3, 2, 2, &[0, 1], &[1; 3]).is_err());
            assert!(
                gpu.upload_axis(3, 2, 2, &[0, 1], &[1; 4]).is_ok(),
                "the last window may end at the edge"
            );
            assert!(
                gpu.upload_axis(5, 1, 2, &[3], &[1; 2]).is_ok(),
                "up to input − taps"
            );
            let x = gpu.upload_axis(2, 1, 2, &[0], &[8_192; 2]).unwrap();
            let y = gpu.upload_axis(1, 1, 1, &[0], &[16_384]).unwrap();
            let input = gpu.upload(&[1u16; 6]).unwrap();
            let mut out = gpu.zeros::<u16>(3).unwrap();
            gpu.resize(&input, 3, &x, &y, &mut out).unwrap();
            assert_eq!(gpu.download(&out).unwrap(), [1, 1, 1]);
            let err = gpu.resize(&input, 2, &x, &y, &mut out).unwrap_err();
            assert_eq!(
                err.to_string(),
                "a resize from 2 × 1 to 1 × 1 of 2 channels needs 4 values in and 2 out, not 6 and 3"
            );
            assert!(gpu.resize(&input, 0, &x, &y, &mut out).is_err());
            let nothing = gpu.upload::<u16>(&[]).unwrap();
            let mut none = gpu.zeros::<u16>(0).unwrap();
            assert!(
                gpu.resize(&nothing, 0, &x, &y, &mut none).is_err(),
                "no channels, even with nothing to do"
            );
            let mut long = gpu.zeros::<u16>(4).unwrap();
            assert!(gpu.resize(&input, 3, &x, &y, &mut long).is_err());
            let empty_axis = gpu.upload_axis(2, 0, 1, &[], &[]).unwrap();
            let mut none = gpu.zeros::<u16>(0).unwrap();
            gpu.resize(&input, 3, &empty_axis, &y, &mut none).unwrap();
        }

        #[test]
        fn the_display_transform_looks_each_channel_up_in_its_own_table() {
            let gpu = crate::cuda_or_skip!();
            let mut tables = vec![0u8; 2 * TABLE_SIZE];
            tables[7] = 70;
            tables[TABLE_SIZE + 7] = 170;
            tables[TABLE_SIZE - 1] = 1;
            tables[2 * TABLE_SIZE - 1] = 2;
            let tables = gpu.upload(&tables).unwrap();
            let input = gpu.upload(&[7u16, 7, u16::MAX, u16::MAX, 0, 0]).unwrap();
            let mut out = gpu.zeros::<u8>(6).unwrap();
            gpu.display(&input, 2, &tables, &mut out).unwrap();
            assert_eq!(gpu.download(&out).unwrap(), [70, 170, 1, 2, 0, 0]);
            let err = gpu.display(&input, 4, &tables, &mut out).unwrap_err();
            assert!(
                err.to_string()
                    .starts_with("a display transform of 4 channels needs"),
                "{err}"
            );
            assert!(gpu.display(&input, 0, &tables, &mut out).is_err());
            let (nothing, no_tables) = (
                gpu.upload::<u16>(&[]).unwrap(),
                gpu.upload::<u8>(&[]).unwrap(),
            );
            let mut none = gpu.zeros::<u8>(0).unwrap();
            assert!(
                gpu.display(&nothing, 0, &no_tables, &mut none).is_err(),
                "no channels, even with nothing to do"
            );
            assert!(
                gpu.display(&input, 3, &tables, &mut out).is_err(),
                "3 tables needed"
            );
            let mut short = gpu.zeros::<u8>(5).unwrap();
            assert!(gpu.display(&input, 2, &tables, &mut short).is_err());
            let empty = gpu.upload::<u16>(&[]).unwrap();
            let mut none = gpu.zeros::<u8>(0).unwrap();
            gpu.display(&empty, 2, &tables, &mut none).unwrap();
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
            assert!(
                gpu.bayer_bin(&narrow, &tall, 2, BinMode::Mean, &mut planes)
                    .is_err(),
                "too large"
            );
            assert!(gpu.bayer_split(&narrow, &two_by_four, &mut planes).is_err());
            assert!(
                gpu.bayer_superpixel(&narrow, &two_by_four, &mut planes)
                    .is_err()
            );
        }
    }
}
