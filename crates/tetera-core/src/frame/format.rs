//! Pixel formats and frame layouts (ADR-0008).
//!
//! A raw frame is the sensor's Bayer mosaic in 16-bit words, with its significant bits
//! recorded (the MIChrome 20's are 12). Cropping at an odd column or row shifts which colour
//! comes first, so a format carries its Bayer pattern and [`PixelFormat::cropped`] moves it.

use crate::error::{Error, Result};
use crate::geometry::Rect;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The colour of one site of a Bayer mosaic. The two greens are told apart by their row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    /// Red.
    Red,
    /// Green, on a row with red.
    GreenOnRed,
    /// Green, on a row with blue.
    GreenOnBlue,
    /// Blue.
    Blue,
}

/// A Bayer mosaic's 2 × 2 tile, read across its first row then its second: RGGB is red and
/// green, then green and blue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "UPPERCASE")]
pub enum BayerPattern {
    /// Red, green / green, blue.
    Rggb,
    /// Green, red / blue, green.
    Grbg,
    /// Green, blue / red, green.
    Gbrg,
    /// Blue, green / green, red.
    Bggr,
}

impl BayerPattern {
    /// Where this pattern's first site falls in an RGGB tile.
    fn offset(self) -> (u32, u32) {
        match self {
            Self::Rggb => (0, 0),
            Self::Grbg => (1, 0),
            Self::Gbrg => (0, 1),
            Self::Bggr => (1, 1),
        }
    }

    fn from_offset(x: u32, y: u32) -> Self {
        match (x % 2, y % 2) {
            (0, 0) => Self::Rggb,
            (1, 0) => Self::Grbg,
            (0, _) => Self::Gbrg,
            _ => Self::Bggr,
        }
    }

    /// The colour at column `x`, row `y`.
    pub fn channel_at(self, x: u32, y: u32) -> Channel {
        let (dx, dy) = self.offset();
        match ((x % 2 + dx) % 2, (y % 2 + dy) % 2) {
            (0, 0) => Channel::Red,
            (1, 0) => Channel::GreenOnRed,
            (0, _) => Channel::GreenOnBlue,
            _ => Channel::Blue,
        }
    }

    /// The pattern of an image cropped from this one at column `x`, row `y`.
    pub fn cropped(self, x: u32, y: u32) -> Self {
        let (dx, dy) = self.offset();
        Self::from_offset(dx + x % 2, dy + y % 2)
    }
}

/// How a frame's pixels are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PixelFormat {
    /// A Bayer mosaic, one byte a site.
    Bayer8 {
        /// The mosaic's tile.
        pattern: BayerPattern,
    },
    /// A Bayer mosaic in 16-bit words, of which the low `significant_bits` carry the value.
    Bayer16 {
        /// The mosaic's tile.
        pattern: BayerPattern,
        /// How many bits the sensor gives, 1 to 16.
        significant_bits: u8,
    },
    /// Grey, one byte a pixel.
    Mono8,
    /// Grey, one 16-bit word a pixel.
    Mono16,
    /// Red, green and blue bytes, interleaved.
    Rgb8,
    /// Red, green and blue 16-bit words, interleaved.
    Rgb16,
    /// Red, green, blue and alpha bytes, interleaved.
    Rgba8,
    /// Planes of 32-bit floats, one after another: for processing.
    PlanarF32 {
        /// How many planes, 1 or more.
        planes: u8,
    },
}

impl PixelFormat {
    /// A raw Bayer frame of `significant_bits` in 16-bit words, checked.
    pub fn bayer16(pattern: BayerPattern, significant_bits: u8) -> Result<Self> {
        if !(1..=16).contains(&significant_bits) {
            return Err(Error::invalid_argument(format!(
                "a 16-bit word holds 1 to 16 significant bits, not {significant_bits}"
            )));
        }
        Ok(Self::Bayer16 {
            pattern,
            significant_bits,
        })
    }

    /// Bytes in one sample.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Bayer8 { .. } | Self::Mono8 | Self::Rgb8 | Self::Rgba8 => 1,
            Self::Bayer16 { .. } | Self::Mono16 | Self::Rgb16 => 2,
            Self::PlanarF32 { .. } => 4,
        }
    }

    /// Samples a pixel holds within one plane.
    pub fn samples_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 | Self::Rgb16 => 3,
            Self::Rgba8 => 4,
            _ => 1,
        }
    }

    /// Bytes a pixel takes within one plane.
    pub fn bytes_per_pixel(self) -> usize {
        self.bytes_per_sample() * self.samples_per_pixel()
    }

    /// How many planes a frame has.
    pub fn planes(self) -> usize {
        match self {
            Self::PlanarF32 { planes } => usize::from(planes),
            _ => 1,
        }
    }

    /// The bits that carry a value in each integer sample; none for floats.
    pub fn significant_bits(self) -> Option<u8> {
        match self {
            Self::Bayer8 { .. } | Self::Mono8 | Self::Rgb8 | Self::Rgba8 => Some(8),
            Self::Bayer16 {
                significant_bits, ..
            } => Some(significant_bits),
            Self::Mono16 | Self::Rgb16 => Some(16),
            Self::PlanarF32 { .. } => None,
        }
    }

    /// The mosaic's tile, for a Bayer format.
    pub fn bayer_pattern(self) -> Option<BayerPattern> {
        match self {
            Self::Bayer8 { pattern } | Self::Bayer16 { pattern, .. } => Some(pattern),
            _ => None,
        }
    }

    /// The format of a region cropped at column `x`, row `y`: a Bayer pattern shifts at an
    /// odd offset; every other format stays as it is.
    pub fn cropped(self, x: u32, y: u32) -> Self {
        match self {
            Self::Bayer8 { pattern } => Self::Bayer8 {
                pattern: pattern.cropped(x, y),
            },
            Self::Bayer16 {
                pattern,
                significant_bits,
            } => Self::Bayer16 {
                pattern: pattern.cropped(x, y),
                significant_bits,
            },
            other => other,
        }
    }
}

/// Binning: how many sensor pixels across and down make one, 1 to 4 each way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Binning {
    /// Across.
    pub x: u8,
    /// Down.
    pub y: u8,
}

impl Binning {
    /// No binning.
    pub const NONE: Self = Self { x: 1, y: 1 };

    /// `x` across by `y` down, checked.
    pub fn new(x: u8, y: u8) -> Result<Self> {
        if !(1..=4).contains(&x) || !(1..=4).contains(&y) {
            return Err(Error::invalid_argument(format!(
                "binning is 1 to 4 each way, not {x} × {y}"
            )));
        }
        Ok(Self { x, y })
    }

    /// The size a `width` × `height` image of `format` bins to: whole bins only, and for a
    /// Bayer mosaic an even size, so its tiles stay whole.
    pub fn size(self, width: u32, height: u32, format: PixelFormat) -> (u32, u32) {
        let (w, h) = (
            width / u32::from(self.x.max(1)),
            height / u32::from(self.y.max(1)),
        );
        if format.bayer_pattern().is_some() {
            (w & !1, h & !1)
        } else {
            (w, h)
        }
    }
}

impl Default for Binning {
    fn default() -> Self {
        Self::NONE
    }
}

/// Every row starts on a multiple of this many bytes, for SIMD and DMA.
pub const ROW_ALIGN: usize = 64;

/// How a frame's bytes are laid out: its format, its size, and the bytes from one row's
/// start to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct FrameLayout {
    /// The pixel format.
    pub format: PixelFormat,
    /// Pixels across.
    pub width: u32,
    /// Pixels down.
    pub height: u32,
    /// Bytes from one row's start to the next's, a multiple of [`ROW_ALIGN`].
    pub stride: usize,
}

impl FrameLayout {
    /// The layout of a `width` × `height` frame of `format`, its rows aligned.
    pub fn new(format: PixelFormat, width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 || format.planes() == 0 {
            return Err(Error::invalid_argument(format!(
                "a frame needs pixels and planes, not {width} × {height} with {} planes",
                format.planes()
            )));
        }
        let too_big =
            || Error::invalid_argument(format!("a {width} × {height} frame is too large to hold"));
        let row = (width as usize)
            .checked_mul(format.bytes_per_pixel())
            .ok_or_else(too_big)?;
        let stride = row
            .checked_next_multiple_of(ROW_ALIGN)
            .ok_or_else(too_big)?;
        stride
            .checked_mul(height as usize)
            .and_then(|plane| plane.checked_mul(format.planes()))
            .ok_or_else(too_big)?;
        Ok(Self {
            format,
            width,
            height,
            stride,
        })
    }

    /// The bytes of one row that hold pixels; the rest of the stride is padding.
    pub fn row_bytes(&self) -> usize {
        self.width as usize * self.format.bytes_per_pixel()
    }

    /// The bytes of one plane.
    pub fn plane_bytes(&self) -> usize {
        self.stride * self.height as usize
    }

    /// The bytes of the whole frame.
    pub fn bytes(&self) -> usize {
        self.plane_bytes() * self.format.planes()
    }

    /// The layout of the region `roi` of this frame: its size, and its Bayer phase.
    pub fn cropped(&self, roi: Rect) -> Result<Self> {
        let whole = Rect::new(0, 0, self.width, self.height);
        if roi.is_empty() || !whole.contains_rect(&roi) {
            return Err(Error::invalid_argument(format!(
                "{roi:?} isn't a region of a {} × {} frame",
                self.width, self.height
            )));
        }
        Self::new(self.format.cropped(roi.x, roi.y), roi.width, roi.height)
    }

    /// The layout of this frame binned by `binning`.
    pub fn binned(&self, binning: Binning) -> Result<Self> {
        let (width, height) = binning.size(self.width, self.height, self.format);
        Self::new(self.format, width, height)
    }
}

/// `value`, of `bits` significant bits, scaled to the full 16-bit range: 0 stays 0, and the
/// largest `bits` value becomes 65,535. A value above that largest one counts as it.
pub fn to_full_range(value: u16, bits: u8) -> u16 {
    let max = max_value(bits);
    let v = u32::from(value).min(max);
    ((v * 65_535 + max / 2) / max) as u16
}

/// A full-range 16-bit value scaled back to `bits` significant bits, to the nearest.
/// [`to_full_range`] then this gives back every `bits` value.
pub fn from_full_range(value: u16, bits: u8) -> u16 {
    let max = max_value(bits);
    ((u32::from(value) * max + 32_767) / 65_535) as u16
}

/// The largest value `bits` hold, `bits` clamped to 1 to 16.
fn max_value(bits: u8) -> u32 {
    (1u32 << bits.clamp(1, 16)) - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const PATTERNS: [BayerPattern; 4] = [
        BayerPattern::Rggb,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
        BayerPattern::Bggr,
    ];

    fn raw() -> PixelFormat {
        PixelFormat::bayer16(BayerPattern::Rggb, 12).unwrap()
    }

    #[test]
    fn each_pattern_reads_its_name() {
        use Channel::*;
        let tile = |p: BayerPattern| {
            [
                p.channel_at(0, 0),
                p.channel_at(1, 0),
                p.channel_at(0, 1),
                p.channel_at(1, 1),
            ]
        };
        assert_eq!(
            tile(BayerPattern::Rggb),
            [Red, GreenOnRed, GreenOnBlue, Blue]
        );
        assert_eq!(
            tile(BayerPattern::Grbg),
            [GreenOnRed, Red, Blue, GreenOnBlue]
        );
        assert_eq!(
            tile(BayerPattern::Gbrg),
            [GreenOnBlue, Blue, Red, GreenOnRed]
        );
        assert_eq!(
            tile(BayerPattern::Bggr),
            [Blue, GreenOnBlue, GreenOnRed, Red]
        );
        assert_eq!(BayerPattern::Rggb.channel_at(5_470, 3_647), GreenOnBlue);
    }

    #[test]
    fn an_odd_crop_shifts_the_pattern_and_an_even_one_doesnt() {
        assert_eq!(BayerPattern::Rggb.cropped(1, 0), BayerPattern::Grbg);
        assert_eq!(BayerPattern::Rggb.cropped(0, 1), BayerPattern::Gbrg);
        assert_eq!(BayerPattern::Rggb.cropped(1, 1), BayerPattern::Bggr);
        assert_eq!(BayerPattern::Bggr.cropped(3, 5), BayerPattern::Rggb);
        assert_eq!(BayerPattern::Grbg.cropped(10, 20), BayerPattern::Grbg);
        assert_eq!(
            raw().cropped(1, 0).bayer_pattern(),
            Some(BayerPattern::Grbg)
        );
        assert_eq!(PixelFormat::Rgb8.cropped(1, 1), PixelFormat::Rgb8);
    }

    #[test]
    fn the_largest_coordinates_dont_overflow() {
        assert_eq!(
            BayerPattern::Rggb.channel_at(u32::MAX, u32::MAX),
            Channel::Blue
        );
        assert_eq!(
            BayerPattern::Bggr.channel_at(u32::MAX, u32::MAX - 1),
            Channel::GreenOnBlue
        );
        assert_eq!(
            BayerPattern::Bggr.cropped(u32::MAX, u32::MAX),
            BayerPattern::Rggb
        );
        assert_eq!(
            BayerPattern::Rggb.cropped(u32::MAX - 1, u32::MAX),
            BayerPattern::Gbrg
        );
    }

    #[test]
    fn a_full_michrome_20_frame_takes_39_9_mb() {
        let layout = FrameLayout::new(raw(), 5_472, 3_648).unwrap();
        assert_eq!(layout.stride, 10_944);
        assert_eq!(layout.bytes(), 39_923_712);
    }

    #[test]
    fn each_format_knows_its_samples_and_bits() {
        let cases: [(PixelFormat, usize, usize, usize, Option<u8>); 8] = [
            (
                PixelFormat::Bayer8 {
                    pattern: BayerPattern::Rggb,
                },
                1,
                1,
                1,
                Some(8),
            ),
            (raw(), 2, 1, 1, Some(12)),
            (PixelFormat::Mono8, 1, 1, 1, Some(8)),
            (PixelFormat::Mono16, 2, 1, 1, Some(16)),
            (PixelFormat::Rgb8, 1, 3, 1, Some(8)),
            (PixelFormat::Rgb16, 2, 3, 1, Some(16)),
            (PixelFormat::Rgba8, 1, 4, 1, Some(8)),
            (PixelFormat::PlanarF32 { planes: 3 }, 4, 1, 3, None),
        ];
        for (format, bytes, samples, planes, bits) in cases {
            assert_eq!(format.bytes_per_sample(), bytes, "{format:?}");
            assert_eq!(format.samples_per_pixel(), samples, "{format:?}");
            assert_eq!(format.bytes_per_pixel(), bytes * samples, "{format:?}");
            assert_eq!(format.planes(), planes, "{format:?}");
            assert_eq!(format.significant_bits(), bits, "{format:?}");
            assert_eq!(
                format.bayer_pattern().is_some(),
                matches!(
                    format,
                    PixelFormat::Bayer8 { .. } | PixelFormat::Bayer16 { .. }
                )
            );
        }
    }

    #[test]
    fn impossible_formats_and_layouts_are_refused() {
        assert!(PixelFormat::bayer16(BayerPattern::Rggb, 0).is_err());
        assert!(PixelFormat::bayer16(BayerPattern::Rggb, 17).is_err());
        assert!(PixelFormat::bayer16(BayerPattern::Rggb, 16).is_ok());
        assert!(FrameLayout::new(PixelFormat::Mono8, 0, 10).is_err());
        assert!(FrameLayout::new(PixelFormat::Mono8, 10, 0).is_err());
        assert!(FrameLayout::new(PixelFormat::PlanarF32 { planes: 0 }, 10, 10).is_err());
        assert!(
            FrameLayout::new(PixelFormat::PlanarF32 { planes: 255 }, u32::MAX, u32::MAX).is_err()
        );
        assert!(Binning::new(0, 1).is_err() && Binning::new(1, 5).is_err());
        assert_eq!(Binning::new(4, 2).unwrap(), Binning { x: 4, y: 2 });
    }

    #[test]
    fn regions_crop_and_bin_with_their_phase_and_size() {
        let sensor = FrameLayout::new(raw(), 5_472, 3_648).unwrap();
        let roi = sensor.cropped(Rect::new(101, 50, 1_000, 800)).unwrap();
        assert_eq!((roi.width, roi.height), (1_000, 800));
        assert_eq!(roi.format.bayer_pattern(), Some(BayerPattern::Grbg));
        assert!(
            sensor.cropped(Rect::new(5_000, 0, 500, 10)).is_err(),
            "past the edge"
        );
        assert!(sensor.cropped(Rect::new(0, 0, 0, 10)).is_err(), "empty");
        let binned = sensor.binned(Binning::new(2, 2).unwrap()).unwrap();
        assert_eq!((binned.width, binned.height), (2_736, 1_824));
        assert_eq!(
            Binning::new(3, 3).unwrap().size(5_472, 3_649, raw()),
            (1_824, 1_216)
        );
        assert_eq!(
            Binning::new(3, 3).unwrap().size(7, 7, raw()),
            (2, 2),
            "a Bayer mosaic stays even"
        );
        assert_eq!(
            Binning::new(3, 3).unwrap().size(7, 7, PixelFormat::Mono8),
            (2, 2)
        );
        assert_eq!(
            Binning::new(2, 1).unwrap().size(7, 7, PixelFormat::Mono8),
            (3, 7)
        );
        assert_eq!(Binning::default().size(7, 5, PixelFormat::Mono8), (7, 5));
    }

    #[test]
    fn formats_serialise_with_their_kind() {
        let json = serde_json::to_string(&raw()).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"bayer16","pattern":"RGGB","significant_bits":12}"#
        );
        assert_eq!(serde_json::from_str::<PixelFormat>(&json).unwrap(), raw());
        assert_eq!(
            serde_json::to_string(&PixelFormat::Mono8).unwrap(),
            r#"{"kind":"mono8"}"#
        );
    }

    #[test]
    fn every_value_of_every_depth_scales_up_and_back() {
        for bits in 1..=16u8 {
            let max = (1u32 << bits) - 1;
            assert_eq!(to_full_range(max as u16, bits), 65_535, "{bits}");
            assert_eq!(to_full_range(0, bits), 0);
            for v in 0..=max {
                assert_eq!(
                    from_full_range(to_full_range(v as u16, bits), bits),
                    v as u16,
                    "{v} of {bits} bits"
                );
            }
        }
        assert_eq!(to_full_range(4_095, 12), 65_535);
        assert_eq!(to_full_range(2_048, 12), 32_776);
        assert_eq!(
            to_full_range(9_000, 12),
            65_535,
            "beyond 12 bits counts as the largest"
        );
        assert_eq!(from_full_range(65_535, 12), 4_095);
    }

    proptest! {
        #[test]
        fn layouts_hold_every_row_aligned(
            format in prop::sample::select(vec![
                PixelFormat::Bayer8 { pattern: BayerPattern::Gbrg }, PixelFormat::Mono8, PixelFormat::Mono16,
                PixelFormat::Rgb8, PixelFormat::Rgb16, PixelFormat::Rgba8, PixelFormat::PlanarF32 { planes: 4 },
            ]),
            width in 1u32..6_000,
            height in 1u32..4_000,
        ) {
            let layout = FrameLayout::new(format, width, height).unwrap();
            prop_assert_eq!(layout.stride % ROW_ALIGN, 0);
            prop_assert!(layout.stride >= layout.row_bytes());
            prop_assert!(layout.stride < layout.row_bytes() + ROW_ALIGN);
            prop_assert_eq!(layout.row_bytes(), width as usize * format.bytes_per_pixel());
            prop_assert_eq!(layout.bytes(), layout.stride * height as usize * format.planes());
        }

        #[test]
        fn a_crop_keeps_every_sites_colour(p in 0usize..4, cx in 0u32..9, cy in 0u32..9, x in 0u32..9, y in 0u32..9) {
            let pattern = PATTERNS[p];
            prop_assert_eq!(pattern.cropped(cx, cy).channel_at(x, y), pattern.channel_at(cx + x, cy + y));
        }
    }
}
