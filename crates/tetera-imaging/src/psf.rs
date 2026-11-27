//! Point spread functions (P1.07b; the behaviour spec is in its notes): the theoretical
//! widefield PSF from the objective's NA and the wavelength, 2D in focus and 3D through focus,
//! by scalar diffraction (Born and Wolf, §8.5 and §8.8); and a measured PSF, from a NumPy `.npy`
//! file. Every PSF is centred on its middle pixel and sums to 1.

use std::f64::consts::{FRAC_PI_4, TAU};
use std::path::Path;
use tetera_core::{Error, Result};

/// J₀(x), by Abramowitz and Stegun's 9.4.1 and 9.4.3, to about 1e-8.
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 3.0 {
        let t = (ax / 3.0).powi(2);
        1.0 + t
            * (-2.249_999_7
                + t * (1.265_620_8
                    + t * (-0.316_386_6
                        + t * (0.044_447_9 + t * (-0.003_944_4 + t * 0.000_210_0)))))
    } else {
        let t = 3.0 / ax;
        let f = 0.797_884_56
            + t * (-0.000_000_77
                + t * (-0.005_527_40
                    + t * (-0.000_095_12
                        + t * (0.001_372_37 + t * (-0.000_728_05 + t * 0.000_144_76)))));
        let theta = ax - FRAC_PI_4
            + t * (-0.041_663_97
                + t * (-0.000_039_54
                    + t * (0.002_625_73
                        + t * (-0.000_541_25 + t * (-0.000_293_33 + t * 0.000_135_58)))));
        f * theta.cos() / ax.sqrt()
    }
}

/// J₁(x), by Abramowitz and Stegun's 9.4.4 and 9.4.6, to about 1e-8.
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    let value = if ax <= 3.0 {
        let t = (ax / 3.0).powi(2);
        ax * (0.5
            + t * (-0.562_499_85
                + t * (0.210_935_73
                    + t * (-0.039_542_89
                        + t * (0.004_433_19 + t * (-0.000_317_61 + t * 0.000_011_09))))))
    } else {
        let t = 3.0 / ax;
        let f = 0.797_884_56
            + t * (0.000_001_56
                + t * (0.016_596_67
                    + t * (0.000_171_05
                        + t * (-0.002_495_11 + t * (0.001_136_53 + t * -0.000_200_33)))));
        let theta = ax - 3.0 * FRAC_PI_4
            + t * (0.124_996_12
                + t * (0.000_056_50
                    + t * (-0.006_378_79
                        + t * (0.000_743_48 + t * (0.000_798_24 + t * -0.000_291_66)))));
        f * theta.cos() / ax.sqrt()
    };
    if x.is_sign_negative() { -value } else { value }
}

/// The Airy pattern's intensity at optical radius v, `[2 J₁(v) / v]²`, 1 at the centre.
pub fn airy(v: f64) -> f64 {
    // J₁(v) is v times a polynomial, so the ratio is exact everywhere but at 0 itself.
    if v == 0.0 {
        return 1.0;
    }
    (2.0 * bessel_j1(v) / v).powi(2)
}

/// The scalar paraxial intensity through focus, at optical coordinates u (axial) and v
/// (lateral): `|2 ∫₀¹ J₀(vρ) exp(−i u ρ²/2) ρ dρ|²`, 1 at the focus.
pub fn defocused(u: f64, v: f64) -> f64 {
    let (re, im) = amplitude(u, v);
    re * re + im * im
}

/// The amplitude whose square `defocused` is, `2 ∫₀¹ J₀(vρ) exp(−i u ρ²/2) ρ dρ`, as its real
/// and imaginary parts, by Simpson's rule over 400 intervals.
fn amplitude(u: f64, v: f64) -> (f64, f64) {
    const N: usize = 400;
    let h = 1.0 / N as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for k in 0..=N {
        let rho = k as f64 * h;
        let weight = if k == 0 || k == N {
            1.0
        } else if k % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let amplitude = weight * bessel_j0(v * rho) * rho;
        let phase = -u * rho * rho / 2.0;
        re += amplitude * phase.cos();
        im += amplitude * phase.sin();
    }
    let scale = 2.0 * h / 3.0;
    (scale * re, scale * im)
}

/// The objective and the camera, as a PSF sees them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Optics {
    /// The numerical aperture.
    pub na: f64,
    /// The wavelength, in µm.
    pub wavelength_um: f64,
    /// A camera pixel's size at the sample, in µm.
    pub pixel_um: f64,
    /// The refractive index of the medium the objective looks through: 1 for a dry objective.
    pub refractive_index: f64,
}

impl Optics {
    fn check(&self) -> Result<()> {
        let fine = |v: f64| v.is_finite() && v > 0.0;
        if !(fine(self.na)
            && fine(self.wavelength_um)
            && fine(self.pixel_um)
            && fine(self.refractive_index))
            || self.na >= self.refractive_index
        {
            return Err(Error::invalid_argument(format!(
                "optics need a positive NA below the refractive index, and a positive wavelength and pixel: {self:?}"
            )));
        }
        Ok(())
    }

    /// Optical radius v at `r_um` from the axis.
    fn v(&self, r_um: f64) -> f64 {
        TAU * self.na * r_um / self.wavelength_um
    }

    /// Optical depth u at `z_um` from focus.
    fn u(&self, z_um: f64) -> f64 {
        TAU * self.na * self.na * z_um / (self.refractive_index * self.wavelength_um)
    }

    /// The PSF's half-width, in pixels: out to the Airy pattern's fourth dark ring, at most 63.
    fn half_width(&self) -> u32 {
        let ring = 13.323_7 * self.wavelength_um / (TAU * self.na);
        ((ring / self.pixel_um).ceil() as u32).clamp(1, 63)
    }
}

/// A point spread function: `width × height × depth` values, x fastest, centred on its middle
/// pixel, summing to 1. A 2D PSF has a depth of 1.
#[derive(Debug, Clone, PartialEq)]
pub struct Psf {
    width: u32,
    height: u32,
    depth: u32,
    values: Vec<f32>,
}

/// Points sampled across a pixel each way, to average its intensity.
const SUB: usize = 5;

impl Psf {
    /// A PSF of `values`, normalised to sum 1. Each size must be odd, so a middle pixel
    /// exists; every value finite and none negative; and some light in it.
    pub fn new(width: u32, height: u32, depth: u32, mut values: Vec<f32>) -> Result<Self> {
        let n = width as usize * height as usize * depth as usize;
        if values.len() != n
            || n == 0
            || width.is_multiple_of(2)
            || height.is_multiple_of(2)
            || depth.is_multiple_of(2)
        {
            return Err(Error::invalid_argument(format!(
                "a PSF is an odd size each way, {width} × {height} × {depth} isn't, or its {} values don't fill it",
                values.len()
            )));
        }
        if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::invalid_argument(
                "a PSF's values must be finite and none negative",
            ));
        }
        let total: f64 = values.iter().map(|&v| f64::from(v)).sum();
        if total <= 0.0 {
            return Err(Error::invalid_argument("a PSF must hold some light"));
        }
        for v in &mut values {
            *v = (f64::from(*v) / total) as f32;
        }
        Ok(Self {
            width,
            height,
            depth,
            values,
        })
    }

    /// The widefield PSF in focus: the Airy pattern, each pixel averaged over 5 × 5 points.
    pub fn widefield_2d(optics: &Optics) -> Result<Self> {
        Self::widefield_3d(optics, 1.0, 1)
    }

    /// The widefield PSF through focus: `planes` (odd) planes `z_step_um` apart, centred on
    /// focus. Each plane's radial profile is integrated once, then sampled across each pixel.
    pub fn widefield_3d(optics: &Optics, z_step_um: f64, planes: u32) -> Result<Self> {
        optics.check()?;
        if planes.is_multiple_of(2) || !(z_step_um.is_finite() && z_step_um > 0.0) {
            return Err(Error::invalid_argument(format!(
                "a 3D PSF takes an odd number of planes and a positive step, not {planes} and {z_step_um}"
            )));
        }
        let half = optics.half_width() as i64;
        let side = (2 * half + 1) as u32;
        let mut values = Vec::with_capacity((side * side * planes) as usize);
        for k in 0..i64::from(planes) {
            let z = (k - i64::from(planes) / 2) as f64 * z_step_um;
            // In focus, the closed form; out of it, the integral on a fine radial grid.
            let profile = if z == 0.0 {
                None
            } else {
                Some(Profile::new(optics, optics.u(z), half))
            };
            for y in -half..=half {
                for x in -half..=half {
                    let mut sum = 0.0;
                    for j in 0..SUB {
                        for i in 0..SUB {
                            let dx = x as f64 + (i as f64 + 0.5) / SUB as f64 - 0.5;
                            let dy = y as f64 + (j as f64 + 0.5) / SUB as f64 - 0.5;
                            let r = dx.hypot(dy) * optics.pixel_um;
                            sum += match &profile {
                                None => airy(optics.v(r)),
                                Some(p) => p.at(optics.v(r)),
                            };
                        }
                    }
                    // The sum serves for the mean: the PSF is normalised below.
                    values.push(sum as f32);
                }
            }
        }
        Self::new(side, side, planes, values)
    }

    /// A measured PSF from a NumPy `.npy` file's bytes: little-endian `f4` or `f8`, 2D (y, x)
    /// or 3D (z, y, x), in C order; taken as centred, and normalised.
    pub fn from_npy(bytes: &[u8]) -> Result<Self> {
        let bad = |why: &str| Error::invalid_argument(format!("not a PSF's .npy file: {why}"));
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return Err(bad("no NumPy magic"));
        }
        let (header_len, start) = match bytes[6] {
            1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
            2 | 3 => {
                let len = bytes.get(8..12).ok_or_else(|| bad("a short header"))?;
                (
                    u32::from_le_bytes(len.try_into().expect("four bytes")) as usize,
                    12,
                )
            }
            v => return Err(bad(&format!("version {v}"))),
        };
        let header = bytes
            .get(start..start + header_len)
            .ok_or_else(|| bad("a short header"))?;
        let header = std::str::from_utf8(header).map_err(|_| bad("a header that isn't text"))?;
        let field = |key: &str| {
            let at = header
                .find(&format!("'{key}'"))
                .ok_or_else(|| bad(&format!("no '{key}'")))?;
            Ok::<&str, Error>(
                header[at + key.len() + 2..]
                    .trim_start()
                    .trim_start_matches(':')
                    .trim_start(),
            )
        };
        let size = match field("descr")? {
            d if d.starts_with("'<f4'") => 4,
            d if d.starts_with("'<f8'") => 8,
            d => {
                return Err(bad(&format!(
                    "values of type {}",
                    d.split(',').next().unwrap_or(d)
                )));
            }
        };
        if !field("fortran_order")?.starts_with("False") {
            return Err(bad("Fortran order"));
        }
        let shape = field("shape")?;
        let shape = &shape[shape.find('(').ok_or_else(|| bad("no shape"))? + 1
            ..shape.find(')').ok_or_else(|| bad("no shape"))?];
        let dims: Vec<u32> = shape
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().map_err(|_| bad("a shape that isn't numbers")))
            .collect::<Result<_>>()?;
        let (depth, height, width) = match dims[..] {
            [h, w] => (1, h, w),
            [d, h, w] => (d, h, w),
            _ => return Err(bad(&format!("{} dimensions", dims.len()))),
        };
        let data = &bytes[start + header_len..];
        let n = width as usize * height as usize * depth as usize;
        if data.len() != n * size {
            return Err(bad(&format!("{} bytes of data for {n} values", data.len())));
        }
        let values = if size == 4 {
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect()
        } else {
            data.as_chunks::<8>()
                .0
                .iter()
                .map(|&b| f64::from_le_bytes(b) as f32)
                .collect()
        };
        Self::new(width, height, depth, values)
    }

    /// A measured PSF from a `.npy` file.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::not_found(format!("{}: {e}", path.display())))?;
        Self::from_npy(&bytes)
    }

    /// It as a NumPy `.npy` file: version 1.0, `<f4`, (z, y, x) or (y, x).
    pub fn to_npy(&self) -> Vec<u8> {
        let shape = if self.depth == 1 {
            format!("({}, {})", self.height, self.width)
        } else {
            format!("({}, {}, {})", self.depth, self.height, self.width)
        };
        let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
        // The header, with its newline, pads the preamble to a multiple of 64 bytes.
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        for v in &self.values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Its width, height and depth.
    pub fn size(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.depth)
    }

    /// Its values, x fastest, then y, then z.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The value at (x, y, z), counted from its first pixel.
    pub fn at(&self, x: u32, y: u32, z: u32) -> f32 {
        self.values[((z * self.height + y) * self.width + x) as usize]
    }
}

/// A plane's radial profile, integrated on a fine grid of v and interpolated between.
struct Profile {
    step: f64,
    samples: Vec<f64>,
}

impl Profile {
    fn new(optics: &Optics, u: f64, half: i64) -> Self {
        // Out to the PSF's corner, with a pixel to spare, in steps of a twentieth of a pixel.
        let reach = optics.v(((half + 1) as f64) * std::f64::consts::SQRT_2 * optics.pixel_um);
        let step = optics.v(optics.pixel_um / 20.0);
        let n = (reach / step).ceil() as usize + 2;
        Self {
            step,
            samples: (0..n).map(|i| defocused(u, i as f64 * step)).collect(),
        }
    }

    /// The profile at `v`, which `new`'s reach covers for every point of the PSF.
    fn at(&self, v: f64) -> f64 {
        let x = v / self.step;
        let i = x.floor() as usize;
        let t = x - i as f64;
        self.samples[i] * (1.0 - t) + self.samples[i + 1] * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn optics(na: f64) -> Optics {
        Optics {
            na,
            wavelength_um: 0.55,
            pixel_um: 0.1,
            refractive_index: 1.0,
        }
    }

    #[test]
    fn bessel_functions_match_their_tables() {
        for (x, j0, j1) in [
            (0.0, 1.0, 0.0),
            (1.0, 0.765_197_686_6, 0.440_050_585_7),
            (2.5, -0.048_383_776_5, 0.497_094_102_5),
            (5.0, -0.177_596_771_3, -0.327_579_137_6),
            (10.0, -0.245_935_764_5, 0.043_472_746_2),
            (20.0, 0.167_024_664_3, 0.066_833_124_2),
        ] {
            assert!(
                (bessel_j0(x) - j0).abs() < 1e-7,
                "J0({x}) = {}",
                bessel_j0(x)
            );
            assert!(
                (bessel_j1(x) - j1).abs() < 1e-7,
                "J1({x}) = {}",
                bessel_j1(x)
            );
        }
        assert_eq!(bessel_j0(-2.0), bessel_j0(2.0), "J0 is even");
        assert_eq!(bessel_j1(-2.0), -bessel_j1(2.0), "J1 is odd");
        assert!(bessel_j0(2.404_825_557_7).abs() < 1e-7, "J0's first zero");
        assert!(bessel_j1(3.831_705_970_2).abs() < 1e-7, "J1's first zero");
    }

    #[test]
    fn the_airy_pattern_is_one_at_its_centre_and_dark_at_j1s_zeros() {
        assert_eq!(airy(0.0), 1.0);
        for zero in [
            3.831_705_970_2,
            7.015_586_669_8,
            10.173_468_135_1,
            13.323_691_936_3,
        ] {
            assert!(airy(zero) < 1e-12, "{zero}");
        }
        assert!(
            (airy(1.616_339_948_3) - 0.5).abs() < 1e-6,
            "half the peak at v = 1.616"
        );
        assert_eq!(airy(-2.0), airy(2.0));
    }

    #[test]
    fn defocus_at_zero_is_the_airy_pattern_and_spreads_the_light() {
        for v in [0.0, 1.0, 2.5, 5.0, 8.0] {
            assert!((defocused(0.0, v) - airy(v)).abs() < 1e-6, "v = {v}");
        }
        assert!((defocused(0.0, 0.0) - 1.0).abs() < 1e-9);
        // On the axis, intensity falls as [sin(u/4) / (u/4)]² (Born and Wolf, §8.8).
        for u in [2.0, 4.0 * std::f64::consts::PI, 10.0] {
            let expected = ((u / 4.0).sin() / (u / 4.0)).powi(2);
            assert!((defocused(u, 0.0) - expected).abs() < 1e-6, "u = {u}");
        }
        assert_eq!(
            defocused(3.0, 2.0),
            defocused(-3.0, 2.0),
            "symmetric through focus"
        );
    }

    #[test]
    fn the_pupil_integral_has_its_closed_form_on_the_axis_and_in_focus() {
        // On the axis it is ∫₀¹ exp(−i u t / 2) dt: with a = u / 2, (sin a + i (cos a − 1)) / a.
        for u in [1.0, 5.0, -3.0] {
            let (re, im) = amplitude(u, 0.0);
            let a = u / 2.0;
            assert!((re - a.sin() / a).abs() < 1e-6, "u = {u}: {re}");
            assert!((im - (a.cos() - 1.0) / a).abs() < 1e-6, "u = {u}: {im}");
        }
        let (re, im) = amplitude(0.0, 2.0);
        assert!(
            (re - bessel_j1(2.0)).abs() < 1e-7 && im == 0.0,
            "2 J₁(v) / v at v = 2"
        );
    }

    #[test]
    fn a_profile_interpolates_the_integral_between_its_samples() {
        let p = Profile::new(&optics(0.5), 3.0, 10);
        for k in [0usize, 7, 40, 101] {
            let v = (k as f64 + 0.5) * p.step;
            assert!(
                (p.at(v) - defocused(3.0, v)).abs() < 2e-4,
                "between samples {k} and {}",
                k + 1
            );
        }
        assert!((p.at(5.0 * p.step) - p.samples[5]).abs() < 1e-12);
        assert!((p.at(5.25 * p.step) - (0.75 * p.samples[5] + 0.25 * p.samples[6])).abs() < 1e-12);
    }

    #[test]
    fn a_fine_psf_follows_the_airy_pattern_and_falls_through_focus_as_theory_says() {
        // At 0.05 µm a pixel, a pixel spans 0.29 in v, so its mean is near the pattern at its
        // centre.
        let o = Optics {
            pixel_um: 0.05,
            ..optics(0.5)
        };
        let flat = Psf::widefield_2d(&o).unwrap();
        let c = flat.size().0 / 2;
        let per_pixel = TAU * 0.5 * 0.05 / 0.55;
        for k in 0..14 {
            let got = f64::from(flat.at(c + k, c, 0) / flat.at(c, c, 0));
            let expected = airy(f64::from(k) * per_pixel);
            assert!(
                (got - expected).abs() < 0.01,
                "{k} pixels out: {got} against {expected}"
            );
        }
        // On the axis, [sin(u/4) / (u/4)]², u = 2π NA² z / (n λ).
        let stack = Psf::widefield_3d(&o, 0.5, 7).unwrap();
        for k in 0..7 {
            let u = TAU * 0.25 * (f64::from(k) - 3.0) * 0.5 / 0.55;
            let expected = if k == 3 {
                1.0
            } else {
                ((u / 4.0).sin() / (u / 4.0)).powi(2)
            };
            let got = f64::from(stack.at(c, c, k) / stack.at(c, c, 3));
            assert!(
                (got - expected).abs() < 0.01,
                "plane {k}: {got} against {expected}"
            );
        }
    }

    #[test]
    fn a_widefield_psf_is_normalised_centred_and_symmetric() {
        let psf = Psf::widefield_2d(&optics(0.5)).unwrap();
        let (w, h, d) = psf.size();
        assert!(w % 2 == 1 && w == h && d == 1);
        let total: f64 = psf.values().iter().map(|&v| f64::from(v)).sum();
        assert!((total - 1.0).abs() < 1e-5, "sums to {total}");
        let c = w / 2;
        let peak = psf.at(c, c, 0);
        assert!(
            psf.values().iter().all(|&v| v <= peak),
            "brightest at the centre"
        );
        for y in 0..h {
            for x in 0..w {
                let v = psf.at(x, y, 0);
                assert!(
                    (v - psf.at(w - 1 - x, y, 0)).abs() < 1e-7
                        && (v - psf.at(x, h - 1 - y, 0)).abs() < 1e-7
                );
                assert!(
                    (v - psf.at(y, x, 0)).abs() < 1e-7,
                    "symmetric across the diagonal"
                );
            }
        }
        // Its half-width reaches the fourth dark ring: 13.32 λ / 2π NA = 2.33 µm, 24 pixels.
        assert_eq!(w, 2 * 24 + 1);
    }

    #[test]
    fn a_wider_aperture_gives_a_narrower_psf() {
        let centre_share = |na: f64| {
            let p = Psf::widefield_2d(&optics(na)).unwrap();
            let c = p.size().0 / 2;
            p.at(c, c, 0)
        };
        assert!(centre_share(0.9) > centre_share(0.5) && centre_share(0.5) > centre_share(0.25));
        assert_eq!(
            Psf::widefield_2d(&Optics {
                pixel_um: 0.001,
                ..optics(0.1)
            })
            .unwrap()
            .size()
            .0,
            127,
            "capped"
        );
    }

    #[test]
    fn a_3d_psf_matches_airy_in_focus_and_spreads_out_of_it() {
        let o = optics(0.5);
        let stack = Psf::widefield_3d(&o, 0.5, 7).unwrap();
        let (w, h, d) = stack.size();
        assert_eq!(d, 7);
        let total: f64 = stack.values().iter().map(|&v| f64::from(v)).sum();
        assert!((total - 1.0).abs() < 1e-5);
        let flat = Psf::widefield_2d(&o).unwrap();
        let plane = |z: u32| {
            (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| stack.at(x, y, z))
                .collect::<Vec<_>>()
        };
        // The focal plane is the Airy pattern, scaled by its share of the stack's light.
        let focal = plane(3);
        let share: f32 = focal.iter().sum();
        for (a, b) in focal.iter().zip(flat.values()) {
            assert!((a / share - b).abs() < 2e-5, "{a} / {share} against {b}");
        }
        // Symmetric through focus, and dimmer at the centre away from it.
        for z in 0..3 {
            for (a, b) in plane(z).iter().zip(plane(6 - z)) {
                assert!((a - b).abs() < 1e-7);
            }
        }
        let c = w / 2;
        assert!(stack.at(c, c, 3) > stack.at(c, c, 2) && stack.at(c, c, 2) > stack.at(c, c, 1));
    }

    #[test]
    fn bad_optics_and_shapes_are_refused() {
        for o in [
            optics(0.0),
            optics(1.0),
            Optics {
                wavelength_um: -1.0,
                ..optics(0.5)
            },
            Optics {
                pixel_um: f64::NAN,
                ..optics(0.5)
            },
        ] {
            assert!(Psf::widefield_2d(&o).is_err(), "{o:?}");
        }
        assert!(
            Psf::widefield_2d(&Optics {
                refractive_index: 1.5,
                ..optics(1.3)
            })
            .is_ok(),
            "oil"
        );
        assert!(
            Psf::widefield_3d(&optics(0.5), 0.5, 4).is_err(),
            "an even number of planes"
        );
        assert!(Psf::widefield_3d(&optics(0.5), 0.0, 3).is_err());
        assert!(Psf::new(2, 3, 1, vec![1.0; 6]).is_err(), "an even width");
        assert!(Psf::new(3, 3, 1, vec![1.0; 8]).is_err(), "too few values");
        assert!(Psf::new(1, 1, 1, vec![-1.0]).is_err());
        assert!(Psf::new(1, 1, 1, vec![f32::NAN]).is_err());
        assert_eq!(
            Psf::new(1, 1, 1, vec![0.0]).unwrap_err().to_string(),
            "a PSF must hold some light"
        );
        let p = Psf::new(3, 1, 1, vec![1.0, 2.0, 1.0]).unwrap();
        assert_eq!(p.values(), [0.25, 0.5, 0.25], "normalised");
    }

    #[test]
    fn npy_files_round_trip_and_bad_ones_are_refused() {
        let psf = Psf::widefield_3d(&optics(0.5), 0.5, 3).unwrap();
        let bytes = psf.to_npy();
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        assert_eq!(
            (10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]))) % 64,
            0,
            "aligned"
        );
        let back = Psf::from_npy(&bytes).unwrap();
        assert_eq!(back.size(), psf.size());
        assert!(
            back.values()
                .iter()
                .zip(psf.values())
                .all(|(a, b)| (a - b).abs() < 1e-9)
        );
        let flat = Psf::new(3, 3, 1, vec![1.0; 9]).unwrap();
        assert_eq!(Psf::from_npy(&flat.to_npy()).unwrap(), flat, "2D");
        // f8, written by hand.
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 3), }";
        let mut f8 = b"\x93NUMPY\x01\x00".to_vec();
        f8.extend_from_slice(&(header.len() as u16).to_le_bytes());
        f8.extend_from_slice(header.as_bytes());
        for v in [1.0f64, 2.0, 1.0] {
            f8.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(Psf::from_npy(&f8).unwrap().values(), [0.25, 0.5, 0.25]);
        let err = |b: &[u8]| Psf::from_npy(b).unwrap_err().to_string();
        assert_eq!(
            err(b"not numpy at all"),
            "not a PSF's .npy file: no NumPy magic"
        );
        assert!(err(&bytes[..bytes.len() - 1]).contains("bytes of data"));
        // The header edited in place, keeping its length.
        let edited = |b: &[u8], from: &str, to: &str| {
            let at = b
                .windows(from.len())
                .position(|w| w == from.as_bytes())
                .unwrap();
            let mut b = b.to_vec();
            b[at..at + to.len()].copy_from_slice(to.as_bytes());
            b
        };
        assert!(err(&edited(&bytes, "<f4", "<i4")).contains("values of type"));
        assert!(err(&edited(&flat.to_npy(), "False", "True ")).contains("Fortran"));
        assert!(err(&edited(&bytes, "'shape'", "'shope'")).contains("no 'shape'"));
        // Version 2 has a four-byte header length.
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1), }";
        let mut v2 = b"\x93NUMPY\x02\x00".to_vec();
        v2.extend_from_slice(&(header.len() as u32).to_le_bytes());
        v2.extend_from_slice(header.as_bytes());
        v2.extend_from_slice(&2.0f32.to_le_bytes());
        assert_eq!(Psf::from_npy(&v2).unwrap().values(), [1.0]);
        assert_eq!(err(&v2[..11]), "not a PSF's .npy file: a short header");
        assert_eq!(err(&bytes[..9]), "not a PSF's .npy file: no NumPy magic");
        assert_eq!(
            err(b"\x93NUMPY\x01\x00\x00\x00"),
            "not a PSF's .npy file: no 'descr'"
        );
        let mut v9 = bytes.clone();
        v9[6] = 9;
        assert!(err(&v9).contains("version 9"));
        assert!(Psf::load(Path::new("/no/such/psf.npy")).is_err());
    }
}
