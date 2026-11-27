//! Optics and light (P1.12a, ADR-0016): the field the sensor sees. Where each sensor pixel
//! looks on the slide; the section in depth groups, each blurred by its defocus and the
//! objective's diffraction; vignetting; and the illumination, its colour temperature and its
//! unevenness. The result is each pixel's irradiance in red, green and blue.
//!
//! Distances are µm on the slide unless they say pixels.

use crate::he::{EOSIN, HAEMATOXYLIN, HeSample, Stains};
use crate::render::{Depth, LAYER, Region, render};
use crate::tiles::{FINEST, TILE, TileKey};
use rayon::prelude::*;
use tetera_core::{Error, Result};

/// A camera's sensor, as the optics see it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sensor {
    /// Its width, pixels.
    pub width: u32,
    /// Its height, pixels.
    pub height: u32,
    /// Its pixel pitch, µm.
    pub pixel: f64,
}

/// The MIChrome 20's sensor: 5472 × 3648 pixels of 2.4 µm.
pub const MICHROME_20: Sensor = Sensor {
    width: 5472,
    height: 3648,
    pixel: 2.4,
};

/// The objective and the camera's mounting, as they shape the field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Optics {
    /// The sensor.
    pub sensor: Sensor,
    /// The objective's magnification.
    pub magnification: f64,
    /// Its numerical aperture.
    pub numerical_aperture: f64,
    /// The sensor's rows against the stage's x axis, radians.
    pub mounting_angle: f64,
    /// The light's middle wavelength, µm.
    pub wavelength: f64,
    /// How much darker the field's corners are than its middle, 0 to 1.
    pub vignetting: f32,
}

impl Optics {
    /// The optics of `objective` on the MIChrome 20, mounted at `mounting_angle`.
    pub fn new(magnification: f64, numerical_aperture: f64, mounting_angle: f64) -> Result<Self> {
        if !(magnification.is_finite() && magnification > 0.0)
            || !(numerical_aperture > 0.0 && numerical_aperture < 1.0)
            || !mounting_angle.is_finite()
        {
            return Err(Error::invalid_argument(format!(
                "a magnification of {magnification}, NA {numerical_aperture}, mounted at {mounting_angle} rad"
            )));
        }
        Ok(Self {
            sensor: MICHROME_20,
            magnification,
            numerical_aperture,
            mounting_angle,
            wavelength: 0.55,
            vignetting: 0.15,
        })
    }

    /// A sensor pixel's size on the slide, µm.
    pub fn pixel(&self) -> f64 {
        self.sensor.pixel / self.magnification
    }

    /// The depth of field, µm: the wave term `λ / NA²` and the geometric term
    /// `e / (M · NA)`, e the sensor's pixel.
    pub fn depth_of_field(&self) -> f64 {
        let na = self.numerical_aperture;
        self.wavelength / (na * na) + self.sensor.pixel / (self.magnification * na)
    }

    /// The blur of a point `defocus` µm from focus, as a Gaussian's σ in µm: the Airy disc's,
    /// `0.21 λ / NA`, and the defocus's circle of confusion, of radius
    /// `|defocus| · NA / √(1 − NA²)`, whose σ is half its radius, added in quadrature.
    pub fn blur(&self, defocus: f64) -> f64 {
        let na = self.numerical_aperture;
        let airy = 0.21 * self.wavelength / na;
        let circle = defocus.abs() * na / (1.0 - na * na).sqrt();
        (airy * airy + circle * circle / 4.0).sqrt()
    }
}

/// Where the sensor looks: its middle on the slide, and its optics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldOfView {
    /// Where its middle looks, µm.
    pub centre: (f64, f64),
    /// The optics.
    pub optics: Optics,
}

impl FieldOfView {
    /// Where the middle of sensor pixel (`col`, `row`) looks, µm.
    pub fn position(&self, col: f64, row: f64) -> (f64, f64) {
        let s = self.optics.sensor;
        let p = self.optics.pixel();
        let u = (col + 0.5 - f64::from(s.width) / 2.0) * p;
        let v = (row + 0.5 - f64::from(s.height) / 2.0) * p;
        let (sin, cos) = self.optics.mounting_angle.sin_cos();
        (
            self.centre.0 + u * cos - v * sin,
            self.centre.1 + u * sin + v * cos,
        )
    }

    /// The smallest box on the slide that holds the whole field, `margin` µm wider each way:
    /// (x0, y0, x1, y1).
    pub fn bounds(&self, margin: f64) -> (f64, f64, f64, f64) {
        let s = self.optics.sensor;
        let (w, h) = (f64::from(s.width), f64::from(s.height));
        let corners = [
            (-0.5, -0.5),
            (w - 0.5, -0.5),
            (-0.5, h - 0.5),
            (w - 0.5, h - 0.5),
        ]
        .map(|(c, r)| self.position(c, r));
        let xs = corners.map(|p| p.0);
        let ys = corners.map(|p| p.1);
        let min = |v: [f64; 4]| v.iter().copied().fold(f64::INFINITY, f64::min);
        let max = |v: [f64; 4]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (
            min(xs) - margin,
            min(ys) - margin,
            max(xs) + margin,
            max(ys) + margin,
        )
    }
}

/// A group of the section's depths that the field takes together, and its blur.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthGroup {
    /// Its first layer, from the section's lower face: layers are 0.5 µm.
    pub first: u8,
    /// How many layers.
    pub count: u8,
    /// Its blur, a Gaussian's σ in sensor pixels.
    pub sigma: f32,
}

impl DepthGroup {
    /// Its depths, µm above the section's lower face.
    pub fn range(&self) -> (f32, f32) {
        (
            f32::from(self.first) * LAYER,
            f32::from(self.first + self.count) * LAYER,
        )
    }
}

/// The section's depth, `thickness` µm, in groups as thick as `optics`' depth of field, to
/// whole layers, each blurred for its middle's distance from `focus` µm above the section's
/// lower face.
pub fn depth_groups(optics: &Optics, focus: f64, thickness: f32) -> Vec<DepthGroup> {
    let layers = (thickness / LAYER).ceil() as u8;
    let per = ((optics.depth_of_field() / f64::from(LAYER)).floor() as u8).clamp(1, layers.max(1));
    let mut groups = Vec::new();
    let mut first = 0;
    while first < layers {
        let count = per.min(layers - first);
        let middle = (f64::from(first) + f64::from(count) / 2.0) * f64::from(LAYER);
        groups.push(DepthGroup {
            first,
            count,
            sigma: (optics.blur(middle - focus) / optics.pixel()) as f32,
        });
        first += count;
    }
    groups
}

/// A Gaussian's weights for σ `sigma` pixels, over ±⌈3σ⌉, summing to 1; one weight of 1 for no
/// blur.
pub fn gaussian(sigma: f32) -> Vec<f32> {
    if sigma.is_nan() || sigma < 1e-3 {
        return vec![1.0];
    }
    let radius = (3.0 * sigma).ceil().max(0.0) as i32;
    let raw: Vec<f32> = (-radius..=radius)
        .map(|k| (-((k * k) as f32) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = raw.iter().sum();
    raw.into_iter().map(|w| w / sum).collect()
}

/// The light: how bright, how warm, how even.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    /// The irradiance through bare glass at the field's middle, in photoelectrons a ms on a
    /// green pixel of quantum efficiency 1.
    pub intensity: f32,
    /// Its colour temperature, K.
    pub colour_temperature: f32,
    /// How uneven the field is: the largest difference from even, as a fraction.
    pub unevenness: f32,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            intensity: 1_000.0,
            colour_temperature: 3_200.0,
            unevenness: 0.05,
        }
    }
}

/// Red's, green's and blue's middle wavelengths, µm.
const WAVELENGTHS: [f64; 3] = [0.610, 0.540, 0.465];

/// A blackbody's radiance at `kelvin` in red, green and blue, by Planck's law at each
/// channel's middle wavelength, relative to green.
pub fn blackbody(kelvin: f32) -> [f32; 3] {
    // The second radiation constant, hc / k, µm·K.
    const C2: f64 = 14_387.77;
    let planck = |lambda: f64| lambda.powi(-5) / ((C2 / (lambda * f64::from(kelvin))).exp() - 1.0);
    let green = planck(WAVELENGTHS[1]);
    WAVELENGTHS.map(|l| (planck(l) / green) as f32)
}

/// How bright the light is at (`u`, `v`), the field's place from −1 to 1 each way, relative to
/// its middle: a gentle slope and twist, `unevenness` at most.
pub fn unevenness(u: f32, v: f32, unevenness: f32) -> f32 {
    1.0 + unevenness * (0.6 * u - 0.3 * v + 0.1 * u * v)
}

/// The vignetting at (`u`, `v`): `vignetting` darker in the corners, falling as the radius
/// squared.
pub fn vignette(u: f32, v: f32, vignetting: f32) -> f32 {
    1.0 - vignetting * (u * u + v * v) / 2.0
}

/// The field: each sensor pixel's irradiance in red, green and blue, photoelectrons a ms at
/// quantum efficiency 1, row by row.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Its width, pixels.
    pub width: u32,
    /// Its height, pixels.
    pub height: u32,
    /// Each pixel's irradiance.
    pub rgb: Vec<[f32; 3]>,
}

impl Field {
    /// As 8-bit sRGB, `white` photoelectrons a ms the brightest.
    pub fn srgb8(&self, white: f32) -> Vec<u8> {
        self.rgb
            .iter()
            .flat_map(|p| crate::he::srgb8(p.map(|v| v / white)))
            .collect()
    }
}

/// The rendering grid a field takes its stains from: the level just finer than its pixel,
/// and the box of level pixels that covers it with room for its blur and resampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    /// The level of detail.
    pub level: u8,
    /// Its first level pixel, each way.
    pub first: (i64, i64),
    /// How many level pixels, each way.
    pub size: (u32, u32),
}

impl Grid {
    /// The grid for `fov`, its widest blur `sigma` sensor pixels.
    pub fn new(fov: &FieldOfView, sigma: f32) -> Self {
        let level = TileKey::level_for(fov.optics.pixel());
        let p = FINEST * f64::from(1u32 << level);
        let margin = (f64::from(sigma) * 3.0 + 2.0) * fov.optics.pixel() + 2.0 * p;
        let (x0, y0, x1, y1) = fov.bounds(margin);
        let first = ((x0 / p).floor() as i64, (y0 / p).floor() as i64);
        let last = ((x1 / p).ceil() as i64, (y1 / p).ceil() as i64);
        Self {
            level,
            first,
            size: ((last.0 - first.0) as u32, (last.1 - first.1) as u32),
        }
    }

    /// Its level's pixel, µm.
    pub fn pixel(&self) -> f64 {
        FINEST * f64::from(1u32 << self.level)
    }

    /// The tiles it touches: their first column and row, and how many each way.
    pub fn tiles(&self) -> ((i32, i32), u32, u32) {
        let t = i64::from(TILE);
        let c0 = self.first.0.div_euclid(t);
        let r0 = self.first.1.div_euclid(t);
        let c1 = (self.first.0 + i64::from(self.size.0) - 1).div_euclid(t);
        let r1 = (self.first.1 + i64::from(self.size.1) - 1).div_euclid(t);
        (
            (c0 as i32, r0 as i32),
            (c1 - c0 + 1) as u32,
            (r1 - r0 + 1) as u32,
        )
    }
}

/// Bilinear: `plane` (`w` × `h`) at (`x`, `y`) in its pixels, its pixels' middles at whole
/// numbers plus a half, held at its edges.
fn bilinear(plane: &[Stains], w: u32, h: u32, x: f64, y: f64) -> Stains {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (ix, iy) = (fx.floor(), fy.floor());
    let (tx, ty) = ((fx - ix) as f32, (fy - iy) as f32);
    let at = |i: f64, j: f64| {
        let c = (i as i64).clamp(0, i64::from(w) - 1) as usize;
        let r = (j as i64).clamp(0, i64::from(h) - 1) as usize;
        plane[r * w as usize + c]
    };
    let mix = |a: Stains, b: Stains, t: f32| Stains {
        haematoxylin: a.haematoxylin + t * (b.haematoxylin - a.haematoxylin),
        eosin: a.eosin + t * (b.eosin - a.eosin),
    };
    mix(
        mix(at(ix, iy), at(ix + 1.0, iy), tx),
        mix(at(ix, iy + 1.0), at(ix + 1.0, iy + 1.0), tx),
        ty,
    )
}

/// `plane` (`w` × `h`) blurred by `weights` along its rows, then its columns, held at its edges.
pub fn blur(plane: &[Stains], w: u32, h: u32, weights: &[f32]) -> Vec<Stains> {
    let r = (weights.len() / 2) as i64;
    let (w, h) = (i64::from(w), i64::from(h));
    let pass = |src: &[Stains], along_rows: bool| -> Vec<Stains> {
        (0..h * w)
            .into_par_iter()
            .map(|k| {
                let (row, col) = (k / w, k % w);
                let mut s = Stains::default();
                for (i, weight) in weights.iter().enumerate() {
                    let d = i as i64 - r;
                    let (c, rr) = if along_rows {
                        ((col + d).clamp(0, w - 1), row)
                    } else {
                        (col, (row + d).clamp(0, h - 1))
                    };
                    let p = src[(rr * w + c) as usize];
                    s.haematoxylin += weight * p.haematoxylin;
                    s.eosin += weight * p.eosin;
                }
                s
            })
            .collect()
    };
    pass(&pass(plane, true), false)
}

/// The light at sensor pixel `at` of a sensor `size` through `stains`, summed over the section:
/// irradiance in red, green and blue.
fn lit(
    light: &Light,
    warmth: [f32; 3],
    vignetting: f32,
    stains: Stains,
    at: (u32, u32),
    size: (u32, u32),
) -> [f32; 3] {
    let ((col, row), (w, h)) = (at, size);
    let u = (col as f32 + 0.5) / w as f32 * 2.0 - 1.0;
    let v = (row as f32 + 0.5) / h as f32 * 2.0 - 1.0;
    let k = light.intensity * unevenness(u, v, light.unevenness) * vignette(u, v, vignetting);
    std::array::from_fn(|c| {
        let od = stains.haematoxylin * HAEMATOXYLIN[c] + stains.eosin * EOSIN[c];
        k * warmth[c] * 10f32.powf(-od)
    })
}

/// The field on the CPU, the reference for the GPU's: `sample` seen by `fov`, focused `focus`
/// µm above the section's lower face, under `light`. Each depth group is rendered on the
/// grid at its level, resampled into the sensor's pixels, and blurred; the groups' stains
/// are summed, and lit.
pub fn field(sample: &HeSample, fov: &FieldOfView, light: &Light, focus: f64) -> Result<Field> {
    let s = fov.optics.sensor;
    let groups = depth_groups(&fov.optics, focus, sample.section().thickness);
    let widest = groups.iter().map(|g| g.sigma).fold(0.0, f32::max);
    let grid = Grid::new(fov, widest);
    let p = grid.pixel();
    let n = s.width as usize * s.height as usize;
    let mut total = vec![Stains::default(); n];
    for group in &groups {
        let (from, to) = group.range();
        let rendered = render(
            sample,
            &Region {
                origin: (grid.first.0 as f64 * p, grid.first.1 as f64 * p),
                width: grid.size.0,
                height: grid.size.1,
                pixel: p,
                depth: Depth::Range(from, to),
            },
        )?;
        let resampled: Vec<Stains> = (0..n)
            .into_par_iter()
            .map(|k| {
                let (col, row) = ((k % s.width as usize) as f64, (k / s.width as usize) as f64);
                let (x, y) = fov.position(col, row);
                bilinear(
                    &rendered.stains,
                    grid.size.0,
                    grid.size.1,
                    x / p - grid.first.0 as f64,
                    y / p - grid.first.1 as f64,
                )
            })
            .collect();
        let blurred = blur(&resampled, s.width, s.height, &gaussian(group.sigma));
        for (t, b) in total.iter_mut().zip(&blurred) {
            t.haematoxylin += b.haematoxylin;
            t.eosin += b.eosin;
        }
    }
    let warmth = blackbody(light.colour_temperature);
    let rgb = total
        .par_iter()
        .enumerate()
        .map(|(k, st)| {
            let at = ((k % s.width as usize) as u32, (k / s.width as usize) as u32);
            lit(
                light,
                warmth,
                fov.optics.vignetting,
                *st,
                at,
                (s.width, s.height),
            )
        })
        .collect();
    Ok(Field {
        width: s.width,
        height: s.height,
        rgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::he::SectionConfig;

    /// A scaled-down sensor for tests: 192 × 128 pixels of 2.4 µm.
    fn small(magnification: f64, na: f64, angle: f64) -> Optics {
        Optics {
            sensor: Sensor {
                width: 192,
                height: 128,
                pixel: 2.4,
            },
            ..Optics::new(magnification, na, angle).unwrap()
        }
    }

    fn near(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn optics_are_a_magnification_an_na_and_an_angle() {
        for (m, na, angle) in [
            (0.0, 0.5, 0.0),
            (10.0, 1.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 0.5, f64::NAN),
            (f64::INFINITY, 0.5, 0.0),
        ] {
            assert!(Optics::new(m, na, angle).is_err(), "{m} {na} {angle}");
        }
        let o = Optics::new(20.0, 0.42, 0.01).unwrap();
        assert_eq!(
            (o.sensor, o.wavelength, o.vignetting),
            (MICHROME_20, 0.55, 0.15)
        );
        assert!(near(o.pixel(), 0.12, 1e-12));
    }

    #[test]
    fn the_field_of_view_follows_the_magnification_and_the_angle() {
        let fov = FieldOfView {
            centre: (1_000.0, 2_000.0),
            optics: small(20.0, 0.42, 0.0),
        };
        // The middle of the sensor looks at the centre; its width is 192 pixels of 0.12 µm.
        let (x, y) = fov.position(95.5, 63.5);
        assert!(near(x, 1_000.0, 1e-9) && near(y, 2_000.0, 1e-9));
        let (x0, y0) = fov.position(0.0, 0.0);
        let (x1, _) = fov.position(191.0, 0.0);
        assert!(
            near(x1 - x0, 191.0 * 0.12, 1e-9)
                && near(x0, 1_000.0 - 95.5 * 0.12, 1e-9)
                && near(y0, 2_000.0 - 63.5 * 0.12, 1e-9)
        );
        assert_eq!(
            fov.bounds(1.0),
            (
                1_000.0 - 11.52 - 1.0,
                2_000.0 - 7.68 - 1.0,
                1_000.0 + 11.52 + 1.0,
                2_000.0 + 7.68 + 1.0
            )
        );
        // At 10×, twice as wide.
        let ten = FieldOfView {
            optics: small(10.0, 0.28, 0.0),
            ..fov
        };
        assert!(near(
            ten.position(191.0, 0.0).0 - ten.position(0.0, 0.0).0,
            2.0 * 191.0 * 0.12,
            1e-9
        ));
        // Mounted a quarter turn round, the sensor's rows run along the stage's y.
        let turned = FieldOfView {
            optics: small(20.0, 0.42, std::f64::consts::FRAC_PI_2),
            ..fov
        };
        let (a, b) = (turned.position(0.0, 0.0), turned.position(10.0, 0.0));
        assert!(near(b.0, a.0, 1e-9) && near(b.1 - a.1, 1.2, 1e-9));
        // At any angle, the box of its four corners.
        let tilted = FieldOfView {
            optics: small(20.0, 0.42, 0.3),
            ..fov
        };
        let corners = [(-0.5, -0.5), (191.5, -0.5), (-0.5, 127.5), (191.5, 127.5)]
            .map(|(c, r)| tilted.position(c, r));
        let (tx0, ty0, tx1, ty1) = tilted.bounds(0.0);
        assert!(near(
            tx0,
            corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min),
            1e-9
        ));
        assert!(near(
            ty0,
            corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min),
            1e-9
        ));
        assert!(near(
            tx1,
            corners
                .iter()
                .map(|p| p.0)
                .fold(f64::NEG_INFINITY, f64::max),
            1e-9
        ));
        assert!(near(
            ty1,
            corners
                .iter()
                .map(|p| p.1)
                .fold(f64::NEG_INFINITY, f64::max),
            1e-9
        ));
        let (bx0, by0, bx1, by1) = turned.bounds(0.0);
        assert!(
            near(bx1 - bx0, 128.0 * 0.12, 1e-9) && near(by1 - by0, 192.0 * 0.12, 1e-9),
            "its box turns with it"
        );
    }

    #[test]
    fn the_depth_of_field_groups_the_section() {
        let groups = |m: f64, na: f64, focus: f64| depth_groups(&small(m, na, 0.0), focus, 4.0);
        let counts = |g: &[DepthGroup]| g.iter().map(|d| d.count).collect::<Vec<_>>();
        assert_eq!(counts(&groups(5.0, 0.14, 2.0)), [8], "5×: all in focus");
        assert_eq!(counts(&groups(10.0, 0.28, 2.0)), [8], "10×");
        assert_eq!(counts(&groups(20.0, 0.42, 2.0)), [6, 2], "20×");
        assert_eq!(counts(&groups(50.0, 0.55, 2.0)), [3, 3, 2], "50×");
        let fifty = groups(50.0, 0.55, 0.75);
        assert_eq!(fifty[1].range(), (1.5, 3.0));
        assert!(
            fifty[0].sigma < fifty[1].sigma && fifty[1].sigma < fifty[2].sigma,
            "blurrier away from focus: {fifty:?}"
        );
        assert!(near(
            depth_groups(&small(5.0, 0.14, 0.0), 2.0, 4.0)[0].sigma as f64,
            small(5.0, 0.14, 0.0).blur(0.0) / 0.48,
            1e-6
        ));
        let o = small(50.0, 0.55, 0.0);
        assert!(near(o.depth_of_field(), 0.55 / 0.3025 + 2.4 / 27.5, 1e-12));
    }

    #[test]
    fn the_blur_grows_with_the_distance_from_focus() {
        let o = small(50.0, 0.55, 0.0);
        assert!(
            near(o.blur(0.0), 0.21 * 0.55 / 0.55, 1e-12),
            "the Airy disc's"
        );
        let mut last = o.blur(0.0);
        for k in 1..20 {
            let b = o.blur(f64::from(k) * 0.25);
            assert!(b > last, "{k}");
            assert_eq!(b, o.blur(-f64::from(k) * 0.25), "the same above and below");
            last = b;
        }
        let circle = 2.0 * 0.55 / (1.0f64 - 0.3025).sqrt();
        assert!(near(
            o.blur(2.0),
            (0.21f64.powi(2) + circle * circle / 4.0).sqrt(),
            1e-12
        ));
        assert!(
            small(10.0, 0.28, 0.0).blur(2.0) < o.blur(2.0),
            "a smaller NA, less defocus"
        );
    }

    #[test]
    fn a_gaussian_sums_to_one_about_its_middle() {
        assert_eq!(gaussian(0.0), [1.0]);
        assert_eq!(gaussian(f32::NAN), [1.0]);
        let w = gaussian(2.0);
        assert_eq!(w.len(), 13);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((0..6).all(|k| w[k] == w[12 - k] && w[k] < w[k + 1]));
        assert!(
            (w[6] / w[8] - 0.5f32.exp()).abs() < 1e-4,
            "e^(k² / 2σ²), k = 2"
        );
    }

    #[test]
    fn the_lights_colour_temperature_and_its_field() {
        let warm = blackbody(3_200.0);
        let cool = blackbody(6_500.0);
        assert_eq!((warm[1], cool[1]), (1.0, 1.0), "relative to green");
        assert!(warm[0] > 1.3 && warm[2] < 0.6, "warm: {warm:?}");
        assert!(cool[2] > 1.0 && cool[0] < 0.95, "cool: {cool:?}");
        // Hot enough that Planck's −1 matters: worked out on its own.
        let hot = blackbody(20_000.0);
        assert!(
            (hot[0] - 0.673323).abs() < 1e-5 && (hot[2] - 1.59323).abs() < 1e-5,
            "{hot:?}"
        );
        assert_eq!(unevenness(0.0, 0.0, 0.1), 1.0);
        assert!(near(f64::from(unevenness(1.0, 0.0, 0.1)), 1.06, 1e-6));
        assert!(near(f64::from(unevenness(1.0, -1.0, 0.1)), 1.08, 1e-6));
        assert_eq!(vignette(0.0, 0.0, 0.15), 1.0);
        assert!(
            near(f64::from(vignette(1.0, 1.0, 0.15)), 0.85, 1e-6),
            "darker by its vignetting in the corners"
        );
    }

    #[test]
    fn a_grid_counts_its_tiles_exactly_and_bilinear_holds_at_its_edges() {
        let grid = |first, size| Grid {
            level: 0,
            first,
            size,
        };
        assert_eq!(grid((0, 0), (256, 256)).tiles(), ((0, 0), 1, 1));
        assert_eq!(grid((0, 0), (257, 512)).tiles(), ((0, 0), 2, 2));
        assert_eq!(grid((-1, 255), (2, 2)).tiles(), ((-1, 0), 2, 2));
        let plane: Vec<Stains> = (0..4)
            .map(|k| Stains {
                haematoxylin: k as f32,
                eosin: 10.0 * k as f32,
            })
            .collect();
        assert_eq!(bilinear(&plane, 2, 2, 0.5, 0.5).haematoxylin, 0.0);
        assert_eq!(bilinear(&plane, 2, 2, 1.0, 0.5).haematoxylin, 0.5);
        assert_eq!(
            bilinear(&plane, 2, 2, 9.0, 9.0),
            plane[3],
            "held at the far edge"
        );
        assert_eq!(
            bilinear(&plane, 2, 2, -9.0, -9.0),
            plane[0],
            "and the near one"
        );
        assert_eq!(bilinear(&plane, 2, 2, 1.5, 9.0).haematoxylin, 3.0);
    }

    #[test]
    fn a_pixel_is_lit_by_its_place_and_its_stains() {
        let light = Light {
            intensity: 1_000.0,
            colour_temperature: 3_200.0,
            unevenness: 0.1,
        };
        let warmth = [1.4, 1.0, 0.6];
        // A 2 × 2 sensor: pixel (0, 1) sits at u = −½, v = ½.
        let p = lit(
            &light,
            warmth,
            0.2,
            Stains {
                haematoxylin: 0.3,
                eosin: 0.1,
            },
            (0, 1),
            (2, 2),
        );
        let k = 1_000.0 * (1.0 + 0.1 * (-0.3 - 0.15 - 0.025)) * (1.0 - 0.2 * 0.25);
        for c in 0..3 {
            let od = 0.3 * HAEMATOXYLIN[c] + 0.1 * EOSIN[c];
            assert!(
                (p[c] - k * warmth[c] * 10f32.powf(-od)).abs() < 1e-3,
                "{c}: {p:?}"
            );
        }
    }

    #[test]
    fn its_grid_covers_the_field_and_its_tiles_the_grid() {
        let fov = FieldOfView {
            centre: (-300.0, 4_000.0),
            optics: small(20.0, 0.42, 0.2),
        };
        let grid = Grid::new(&fov, 4.0);
        assert_eq!(grid.level, 1);
        let p = grid.pixel();
        let (x0, y0, x1, y1) = fov.bounds(0.0);
        assert!(grid.first.0 as f64 * p <= x0 && grid.first.1 as f64 * p <= y0);
        assert!((grid.first.0 + i64::from(grid.size.0)) as f64 * p >= x1);
        assert!((grid.first.1 + i64::from(grid.size.1)) as f64 * p >= y1);
        let ((c0, r0), columns, rows) = grid.tiles();
        let t = i64::from(TILE);
        assert!(i64::from(c0) * t <= grid.first.0 && i64::from(r0) * t <= grid.first.1);
        assert!(i64::from(c0 + columns as i32) * t >= grid.first.0 + i64::from(grid.size.0));
        assert!(i64::from(r0 + rows as i32) * t >= grid.first.1 + i64::from(grid.size.1));
        assert!(c0 < 0, "the box crosses x = 0: {c0}");
    }

    fn sharpness(f: &Field) -> f64 {
        let w = f.width as usize;
        (0..f.rgb.len() - 1)
            .filter(|k| (k + 1) % w != 0)
            .map(|k| f64::from(f.rgb[k + 1][1] - f.rgb[k][1]).powi(2))
            .sum()
    }

    #[test]
    fn the_field_is_sharpest_in_focus_and_bare_glass_is_the_light() {
        let sample = HeSample::new(7, SectionConfig::default()).unwrap();
        let light = Light {
            unevenness: 0.0,
            ..Light::default()
        };
        let fov = FieldOfView {
            centre: (1_000.0, 1_000.0),
            optics: small(50.0, 0.55, 0.3),
        };
        let focused = field(&sample, &fov, &light, 2.0).unwrap();
        assert_eq!(
            (focused.width, focused.height, focused.rgb.len()),
            (192, 128, 192 * 128)
        );
        let blurred = field(&sample, &fov, &light, 12.0).unwrap();
        assert!(
            sharpness(&focused) > 2.0 * sharpness(&blurred),
            "{} against {}",
            sharpness(&focused),
            sharpness(&blurred)
        );
        assert_eq!(
            field(&sample, &fov, &light, 2.0).unwrap(),
            focused,
            "the same twice"
        );
        // Somewhere bare: the light itself, dimmed only by the vignetting.
        let glass = (0..200)
            .map(|i| (f64::from(i) * 37.0, 3_000.0))
            .find(|&(x, y)| {
                (0..9).all(|k| {
                    sample.column(
                        x + f64::from(k % 3) * 15.0 - 15.0,
                        y + f64::from(k / 3) * 15.0 - 15.0,
                    ) == crate::he::Column::default()
                })
            })
            .expect("bare glass");
        let bare = field(
            &sample,
            &FieldOfView {
                centre: glass,
                optics: small(20.0, 0.42, 0.0),
            },
            &light,
            2.0,
        )
        .unwrap();
        let middle = bare.rgb[64 * 192 + 96];
        let warmth = blackbody(3_200.0);
        for c in 0..3 {
            assert!(
                near(
                    f64::from(middle[c]),
                    f64::from(1_000.0 * warmth[c] * vignette(1.0 / 192.0, 1.0 / 128.0, 0.15)),
                    1e-2
                ),
                "{middle:?}"
            );
        }
        let corner = bare.rgb[0][1];
        assert!(corner < middle[1] * 0.9, "the corners darker");
        let shown = bare.srgb8(1_000.0 * warmth[0]);
        assert_eq!(shown.len(), 192 * 128 * 3);
        let at = (64 * 192 + 96) * 3;
        assert_eq!(
            &shown[at..at + 3],
            &crate::he::srgb8(middle.map(|v| v / (1_000.0 * warmth[0])))
        );
    }

    #[test]
    fn changing_the_objective_changes_the_field_and_its_blur() {
        let sample = HeSample::new(7, SectionConfig::default()).unwrap();
        let light = Light::default();
        let at = |m: f64, na: f64| {
            field(
                &sample,
                &FieldOfView {
                    centre: (1_000.0, 1_000.0),
                    optics: small(m, na, 0.0),
                },
                &light,
                2.0,
            )
            .unwrap()
        };
        let (ten, twenty) = (at(10.0, 0.28), at(20.0, 0.42));
        assert_ne!(ten, twenty);
        let mean =
            |f: &Field| f.rgb.iter().map(|p| f64::from(p[1])).sum::<f64>() / f.rgb.len() as f64;
        assert!(mean(&ten) > 0.0 && mean(&twenty) > 0.0);
    }
}
