//! A test pattern with a known answer, and how close an image comes to it: the peak
//! signal-to-noise ratio. And a synthetic [`Sensor`] that records a linear image as a raw
//! mosaic, with an offset, vignetting and unequal channels, for testing the ISP.

use crate::image::{Image, Mosaic, MosaicLayout};
use std::f64::consts::{PI, TAU};
use tetera_core::frame::format::{BayerPattern, Channel};
use tetera_core::{Error, Result};

/// A synthetic sensor: what a linear image becomes as a raw mosaic, with the faults the ISP
/// corrects. Noise-free, so tests know the answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sensor {
    /// Its Bayer pattern.
    pub pattern: BayerPattern,
    /// Its significant bits.
    pub bits: u8,
    /// What every site reads with no light: the black level and dark current.
    pub offset: u16,
    /// How much darker the corners are: `1 − vignetting × r²`, with r 1 at a corner.
    pub vignetting: f64,
    /// Each channel's response to its light: red, green and blue.
    pub response: [f64; 3],
}

impl Default for Sensor {
    fn default() -> Self {
        Self {
            pattern: BayerPattern::Rggb,
            bits: 12,
            offset: 0,
            vignetting: 0.0,
            response: [1.0; 3],
        }
    }
}

impl Sensor {
    fn layout(&self, width: u32, height: u32) -> MosaicLayout {
        MosaicLayout {
            width,
            height,
            pattern: self.pattern,
            significant_bits: self.bits,
        }
    }

    /// `rgb`, a linear image of three channels, as this sensor records it: each site's own
    /// colour times its response and the vignetting, rounded, plus the offset, clamped.
    pub fn expose(&self, rgb: &Image<u16>) -> Result<Mosaic> {
        if rgb.channels() != 3 {
            return Err(Error::invalid_argument(format!(
                "a sensor records RGB, not {} channels",
                rgb.channels()
            )));
        }
        let layout = self.layout(rgb.width(), rgb.height());
        let (cx, cy) = (f64::from(rgb.width()) / 2.0, f64::from(rgb.height()) / 2.0);
        let corner = cx * cx + cy * cy;
        let max = f64::from(layout.max());
        let mut sites = Vec::with_capacity(layout.sites());
        for y in 0..rgb.height() {
            for x in 0..rgb.width() {
                let (dx, dy) = (f64::from(x) + 0.5 - cx, f64::from(y) + 0.5 - cy);
                let shade = 1.0 - self.vignetting * (dx * dx + dy * dy) / corner;
                let c = match self.pattern.channel_at(x, y) {
                    Channel::Red => 0,
                    Channel::GreenOnRed | Channel::GreenOnBlue => 1,
                    Channel::Blue => 2,
                };
                let light = (f64::from(rgb.pixel(x, y)[c]) * self.response[c] * shade).round();
                sites.push((light + f64::from(self.offset)).clamp(0.0, max) as u16);
            }
        }
        Mosaic::new(layout, sites)
    }

    /// The master dark it gives at `width × height`: the offset at every site.
    pub fn dark(&self, width: u32, height: u32) -> Result<Mosaic> {
        Mosaic::new(
            self.layout(width, height),
            vec![self.offset; width as usize * height as usize],
        )
    }
}

/// The finest stripes of the zone plate, in cycles a pixel: half the Bayer mosaic's
/// Nyquist limit for red and blue, so that every method can follow them, some better.
const FINEST: f64 = 0.125;

/// Colour bars at 75 %: white, yellow, cyan, green, magenta, red, blue, black.
const BARS: [[f64; 3]; 8] = [
    [0.75, 0.75, 0.75],
    [0.75, 0.75, 0.0],
    [0.0, 0.75, 0.75],
    [0.0, 0.75, 0.0],
    [0.75, 0.0, 0.75],
    [0.75, 0.0, 0.0],
    [0.0, 0.0, 0.75],
    [0.0, 0.0, 0.0],
];

fn hsv(hue: f64, saturation: f64, value: f64) -> [f64; 3] {
    let h = hue.rem_euclid(1.0) * 6.0;
    let f = h - h.floor();
    let (p, q, t) = (
        value * (1.0 - saturation),
        value * (1.0 - saturation * f),
        value * (1.0 - saturation * (1.0 - f)),
    );
    match h as u32 {
        0 => [value, t, p],
        1 => [q, value, p],
        2 => [p, value, t],
        3 => [p, q, value],
        4 => [t, p, value],
        _ => [value, p, q],
    }
}

/// A `width × height` RGB test pattern in values of `significant_bits` bits:
/// - **the top third:** hues across, brightness down, all smooth;
/// - **the middle third:** a grey zone plate, whose rings grow finer outwards to an eighth of
///   a cycle a pixel: fine detail, where demosaicking errs most;
/// - **the bottom third:** eight colour bars with sharp edges, under fine diagonal ripples.
///
/// The same arguments always give the same pattern.
pub fn test_pattern(width: u32, height: u32, significant_bits: u8) -> Image<u16> {
    let max = f64::from(u16::MAX >> (16 - significant_bits.clamp(1, 16)));
    let (w, h) = (f64::from(width), f64::from(height));
    let (cx, cy) = (w / 2.0, h / 2.0);
    let radius = (w / 2.0).min(h / 6.0).max(1.0);
    let mut data = Vec::with_capacity(3 * width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let rgb = if fy < h / 3.0 {
                hsv(fx / w, 0.8, 0.25 + 0.75 * fy / (h / 3.0))
            } else if fy < 2.0 * h / 3.0 {
                let r = (fx - cx).hypot(fy - cy);
                let phase = if r <= radius {
                    PI * FINEST * r * r / radius
                } else {
                    PI * FINEST * radius + TAU * FINEST * (r - radius)
                };
                [0.5 + 0.4 * phase.cos(); 3]
            } else {
                let bar = BARS[((fx / w * 8.0) as usize).min(7)];
                let ripple = 0.8 + 0.2 * (TAU * (fx + fy) / 12.0).sin();
                bar.map(|c| c * ripple + 0.1)
            };
            data.extend(rgb.map(|c| (c.clamp(0.0, 1.0) * max).round() as u16));
        }
    }
    Image::new(width, height, 3, data).expect("sized to fit")
}

/// The peak signal-to-noise ratio of `actual` against `expected`, in decibels, for values up
/// to `max`: `10 log10(max² / mean squared error)`, over every value. Infinite when the two
/// are the same.
pub fn psnr(expected: &Image<u16>, actual: &Image<u16>, max: u16) -> Result<f64> {
    let same_shape = (expected.width(), expected.height(), expected.channels())
        == (actual.width(), actual.height(), actual.channels());
    if !same_shape || expected.data().is_empty() {
        return Err(Error::invalid_argument(format!(
            "PSNR needs two images of one shape, not {} × {} × {} and {} × {} × {}",
            expected.width(),
            expected.height(),
            expected.channels(),
            actual.width(),
            actual.height(),
            actual.channels()
        )));
    }
    let squares: f64 = expected
        .data()
        .iter()
        .zip(actual.data())
        .map(|(&e, &a)| (f64::from(e) - f64::from(a)).powi(2))
        .sum();
    let mse = squares / expected.data().len() as f64;
    Ok(10.0 * (f64::from(max).powi(2) / mse).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pattern_has_its_three_bands_in_range() {
        let image = test_pattern(96, 60, 12);
        assert_eq!(
            (image.width(), image.height(), image.channels()),
            (96, 60, 3)
        );
        assert!(image.data().iter().all(|&v| v <= 4_095));
        assert_eq!(image, test_pattern(96, 60, 12), "the same every time");
        // The top band is saturated colour: its channels differ.
        let top = image.pixel(10, 15);
        assert!(top[0] != top[1] || top[1] != top[2], "{top:?}");
        // The middle band is grey.
        let middle = image.pixel(70, 30);
        assert!(
            middle[0] == middle[1] && middle[1] == middle[2],
            "{middle:?}"
        );
        // The bottom band's first bar is grey too, and its sixth red.
        let white = image.pixel(2, 50);
        assert!(
            white[0] == white[1] && white[1] == white[2] && white[0] > 1_000,
            "{white:?}"
        );
        let red = image.pixel(66, 50);
        assert!(red[0] > red[1] && red[1] == red[2], "{red:?}");
        let sixteen = test_pattern(8, 6, 16);
        assert!(
            sixteen.data().iter().any(|&v| v > 4_095),
            "16 bits use their range"
        );
        let one = test_pattern(8, 6, 1);
        assert!(one.data().iter().all(|&v| v <= 1));
    }

    #[test]
    fn the_zone_plate_is_finer_outwards() {
        // Along the middle row from the centre, count the crossings of mid-grey in two
        // stretches of equal length: the outer one has more.
        let image = test_pattern(600, 600, 12);
        let row: Vec<f64> = (0..600)
            .map(|x| f64::from(image.pixel(x, 299)[1]) - 2_048.0)
            .collect();
        let crossings = |range: std::ops::Range<usize>| {
            row[range]
                .windows(2)
                .filter(|w| w[0].signum() != w[1].signum())
                .count()
        };
        let (inner, outer) = (crossings(300..350), crossings(350..400));
        assert!(outer > inner, "{inner} inner, {outer} outer");
    }

    #[test]
    fn hues_turn_through_the_colours() {
        // Hues that are exact in binary, one in each sixth of the turn but two.
        assert_eq!(hsv(0.0, 1.0, 1.0), [1.0, 0.0, 0.0]);
        assert_eq!(hsv(0.125, 1.0, 1.0), [1.0, 0.75, 0.0]);
        assert_eq!(hsv(0.25, 1.0, 1.0), [0.5, 1.0, 0.0]);
        assert_eq!(hsv(0.5, 1.0, 1.0), [0.0, 1.0, 1.0]);
        assert_eq!(hsv(0.75, 1.0, 1.0), [0.5, 0.0, 1.0]);
        assert_eq!(hsv(0.875, 1.0, 1.0), [1.0, 0.0, 0.75]);
        assert_eq!(hsv(1.0, 1.0, 1.0), hsv(0.0, 1.0, 1.0), "a full turn");
        assert_eq!(hsv(0.3, 0.0, 0.5), [0.5, 0.5, 0.5], "no saturation is grey");
    }

    #[test]
    fn the_pattern_is_pinned() {
        // Its sum, and a sum weighted by position, recorded when it was made: the PSNRs in
        // the notes are against this pattern. `sin`, `cos` and `hypot` are within an ulp on
        // any platform, and a value rounds differently only at an exact half.
        let image = test_pattern(97, 61, 12);
        let sum: u64 = image.data().iter().map(|&v| u64::from(v)).sum();
        let weighted: u64 = image
            .data()
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u64 + 1) * u64::from(v))
            .sum();
        assert_eq!((sum, weighted), (30_884_548, 283_145_247_989));
        assert_eq!(image.pixel(48, 30), [3_686, 3_686, 3_686]);
        assert_eq!(image.pixel(5, 5), [1_854, 876, 371]);
        assert_eq!(image.pixel(90, 55), [410, 410, 410]);
        // Tall and narrow, the zone plate's radius is half the width, not a sixth of the height.
        let tall = test_pattern(12, 120, 12);
        let sum: u64 = tall.data().iter().map(|&v| u64::from(v)).sum();
        assert_eq!(sum, 7_476_267);
    }

    #[test]
    fn the_sensor_shades_each_channel_and_adds_its_offset() {
        let sensor = Sensor {
            offset: 100,
            vignetting: 0.5,
            response: [0.5, 1.0, 2.0],
            ..Sensor::default()
        };
        let even = Image::new(4, 2, 3, [1_000u16; 3].repeat(8)).unwrap();
        // 4 × 2: the centre is (2, 1), and a corner 2² + 1² = 5 from it. Sites (0, 0) and
        // (3, 0) are 2.5/5 of the way out, shaded 0.75; (1, 0) and (2, 0), 0.5/5, shaded 0.95.
        let m = sensor.expose(&even).unwrap();
        assert_eq!(m.sites(), [475, 1_050, 575, 850, 850, 2_000, 1_050, 1_600]);
        assert_eq!(
            *m.layout(),
            MosaicLayout {
                width: 4,
                height: 2,
                pattern: BayerPattern::Rggb,
                significant_bits: 12
            }
        );
        let bright = Sensor {
            response: [6.0; 3],
            ..sensor
        }
        .expose(&even)
        .unwrap();
        assert!(
            bright.sites().iter().all(|&v| v == 4_095),
            "clamped to 12 bits"
        );
        let gbrg = Sensor {
            pattern: BayerPattern::Gbrg,
            ..sensor
        }
        .expose(&even)
        .unwrap();
        assert_eq!(
            gbrg.sites()[1],
            2_000,
            "blue at (1, 0): 1,000 × 2 × 0.95 + 100"
        );
        // 2 × 4, the other way up: rows 0 and 3 shaded 0.75, rows 1 and 2, 0.95.
        let tall = sensor
            .expose(&Image::new(2, 4, 3, [1_000u16; 3].repeat(8)).unwrap())
            .unwrap();
        assert_eq!(
            tall.sites(),
            [475, 850, 1_050, 2_000, 575, 1_050, 850, 1_600]
        );
        assert_eq!(sensor.dark(3, 2).unwrap().sites(), [100; 6]);
        let grey = Image::new(1, 1, 1, vec![0u16]).unwrap();
        assert_eq!(
            sensor.expose(&grey).unwrap_err().to_string(),
            "a sensor records RGB, not 1 channels"
        );
    }

    #[test]
    fn psnr_follows_the_mean_squared_error() {
        let a = Image::new(2, 1, 1, vec![100u16, 200]).unwrap();
        assert_eq!(psnr(&a, &a, 255).unwrap(), f64::INFINITY);
        let b = Image::new(2, 1, 1, vec![110u16, 200]).unwrap();
        // MSE = 10² / 2 = 50; 10 log10(255² / 50) ≈ 31.14 dB.
        let db = psnr(&a, &b, 255).unwrap();
        assert!(
            (db - 10.0 * (255.0f64 * 255.0 / 50.0).log10()).abs() < 1e-12,
            "{db}"
        );
        assert!((db - 31.1411).abs() < 1e-3, "{db}");
        assert_eq!(psnr(&b, &a, 255).unwrap(), db, "symmetric");
        let c = Image::new(1, 2, 1, vec![100u16, 200]).unwrap();
        let err = psnr(&a, &c, 255).unwrap_err();
        assert_eq!(
            err.to_string(),
            "PSNR needs two images of one shape, not 2 × 1 × 1 and 1 × 2 × 1"
        );
        let empty = Image::new(0, 0, 3, Vec::<u16>::new()).unwrap();
        assert!(psnr(&empty, &empty, 255).is_err());
    }
}
