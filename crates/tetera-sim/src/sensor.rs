//! The sensor (P1.12b, ADR-0016): the field's light turned into a raw frame. Each pixel sees
//! its colour of an RGGB mosaic through its quantum efficiency; photoelectrons and dark
//! current arrive as Poisson counts, read noise adds a Gaussian; the well fills, the gain
//! and the black level map electrons to counts, 12 significant bits in 16-bit words, or 8.
//!
//! The noise comes from integer hashes of the seed, the pixel and the frame's number, so a
//! frame repeats exactly from its seed on the same backend.

use crate::noise::{Seed, hash, unit};
use rayon::prelude::*;
use std::time::Duration;
use tetera_core::{Error, Result};

/// The sensor's physics, per pixel of 2.4 µm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorModel {
    /// Quantum efficiency in red, green and blue.
    pub quantum_efficiency: [f32; 3],
    /// The full well, electrons.
    pub full_well: f32,
    /// Read noise, electrons rms.
    pub read_noise: f32,
    /// Dark current, electrons a second.
    pub dark_current: f32,
    /// The black level, in 12-bit counts.
    pub black_level: u16,
    /// Electrons a count at a gain of 1, in 12 bits.
    pub electrons_per_count: f32,
}

impl Default for SensorModel {
    /// A 20 MP back-illuminated sensor of 2.4 µm pixels, as the MIChrome 20's is.
    fn default() -> Self {
        Self {
            quantum_efficiency: [0.45, 0.62, 0.42],
            full_well: 15_000.0,
            read_noise: 2.2,
            dark_current: 0.5,
            black_level: 100,
            electrons_per_count: 3.76,
        }
    }
}

/// How a frame is exposed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exposure {
    /// How long.
    pub time: Duration,
    /// The gain, 1 or more.
    pub gain: f32,
    /// Significant bits: 12, in 16-bit words, or 8.
    pub bits: u8,
    /// Pixels binned each way: their light and charge summed.
    pub binning: u32,
}

impl Exposure {
    /// Whether it is one the sensor takes.
    pub fn check(&self) -> Result<()> {
        if !(self.gain.is_finite() && self.gain >= 1.0)
            || !matches!(self.bits, 8 | 12)
            || self.binning == 0
        {
            return Err(Error::invalid_argument(format!(
                "a gain of {}, {} bits, binned {}: a gain of 1 or more, 8 or 12 bits, binned 1 or more",
                self.gain, self.bits, self.binning
            )));
        }
        Ok(())
    }
}

/// The colour of pixel (`col`, `row`) of an RGGB mosaic: 0 red, 1 green, 2 blue.
pub fn bayer(col: u32, row: u32) -> usize {
    match (row % 2, col % 2) {
        (0, 0) => 0,
        (1, 1) => 2,
        _ => 1,
    }
}

const SHOT: u32 = 40;
const READ: u32 = 41;

/// A uniform draw in (0, 1] for pixel (`col`, `row`) of frame `frame`, the `k`th for `salt`.
fn draw(seed: Seed, salt: u32, col: u32, row: u32, frame: u64, k: u32) -> f32 {
    let h = hash(
        seed,
        salt.wrapping_add(k << 8) ^ (frame >> 32) as u32,
        col as i32,
        row as i32,
        frame as u32 as i32,
    );
    1.0 - unit(h)
}

/// A standard normal draw, by Box–Muller.
fn normal(seed: Seed, salt: u32, col: u32, row: u32, frame: u64) -> f32 {
    let (u1, u2) = (
        draw(seed, salt, col, row, frame, 0),
        draw(seed, salt, col, row, frame, 1),
    );
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// The largest mean drawn by inversion; above it, a Gaussian of the same mean and variance.
pub const POISSON_EXACT: f32 = 16.0;

/// A Poisson draw of mean `mean`: by inversion for a small mean, by a Gaussian above.
fn poisson(seed: Seed, col: u32, row: u32, frame: u64, mean: f32) -> f32 {
    if mean <= 0.0 {
        return 0.0;
    }
    if mean > POISSON_EXACT {
        return (mean + mean.sqrt() * normal(seed, SHOT, col, row, frame))
            .round()
            .max(0.0);
    }
    let u = draw(seed, SHOT + 1, col, row, frame, 0);
    let mut p = (-mean).exp();
    let mut cumulative = p;
    let mut k = 0.0;
    while u > cumulative && k < 64.0 {
        k += 1.0;
        p *= mean / k;
        cumulative += p;
    }
    k
}

/// One pixel's count: `irradiance` in its colour, photoelectrons a ms at quantum efficiency 1,
/// from `model`, exposed as `exposure`.
pub fn pixel(
    model: &SensorModel,
    exposure: &Exposure,
    seed: Seed,
    frame: u64,
    col: u32,
    row: u32,
    irradiance: f32,
) -> u16 {
    let b = exposure.binning as f32;
    let area = b * b;
    let ms = exposure.time.as_secs_f32() * 1e3;
    let colour = bayer(col, row);
    let mean =
        irradiance * model.quantum_efficiency[colour] * ms + model.dark_current * area * ms * 1e-3;
    let electrons = poisson(seed, col, row, frame, mean)
        + model.read_noise * b * normal(seed, READ, col, row, frame);
    // Read noise may take a dark pixel below its black level; only the well has a limit.
    let well = electrons.min(model.full_well * area);
    let counts = (f32::from(model.black_level) + well * exposure.gain / model.electrons_per_count)
        .round()
        .clamp(0.0, 4_095.0);
    if exposure.bits == 8 {
        (counts as u16) >> 4
    } else {
        counts as u16
    }
}

/// A raw frame, row by row, from a field of `width` × `height` pixels' irradiance in red,
/// green and blue: each pixel its colour of the mosaic, through `model`.
pub fn expose(
    rgb: &[[f32; 3]],
    width: u32,
    height: u32,
    model: &SensorModel,
    exposure: &Exposure,
    seed: u64,
    frame: u64,
) -> Result<Vec<u16>> {
    exposure.check()?;
    if rgb.len() != width as usize * height as usize {
        return Err(Error::invalid_argument(format!(
            "a field of {} pixels for a {width} × {height} frame",
            rgb.len()
        )));
    }
    let seed = Seed::new(seed);
    Ok((0..rgb.len())
        .into_par_iter()
        .map(|k| {
            let (col, row) = ((k % width as usize) as u32, (k / width as usize) as u32);
            let area = (exposure.binning * exposure.binning) as f32;
            pixel(
                model,
                exposure,
                seed,
                frame,
                col,
                row,
                rgb[k][bayer(col, row)] * area,
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exposure(ms: u64, gain: f32, bits: u8) -> Exposure {
        Exposure {
            time: Duration::from_millis(ms),
            gain,
            bits,
            binning: 1,
        }
    }

    #[test]
    fn the_mosaic_is_rggb() {
        assert_eq!(
            [
                bayer(0, 0),
                bayer(1, 0),
                bayer(0, 1),
                bayer(1, 1),
                bayer(2, 2),
                bayer(3, 5)
            ],
            [0, 1, 1, 2, 0, 2]
        );
    }

    #[test]
    fn an_exposure_is_checked() {
        for bad in [
            exposure(1, 0.5, 12),
            exposure(1, f32::NAN, 12),
            exposure(1, 1.0, 10),
            Exposure {
                binning: 0,
                ..exposure(1, 1.0, 12)
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
        exposure(1, 1.0, 8).check().unwrap();
        assert!(
            expose(
                &[[0.0; 3]; 3],
                2,
                2,
                &SensorModel::default(),
                &exposure(1, 1.0, 12),
                1,
                0
            )
            .is_err()
        );
    }

    /// The mean and variance of `n` × `n` pixels of one colour, lit evenly.
    fn statistics(model: &SensorModel, e: &Exposure, irradiance: f32, colour: usize) -> (f64, f64) {
        let n = 256u32;
        let field = vec![[irradiance; 3]; (n * n) as usize];
        let raw = expose(&field, n, n, model, e, 7, 3).unwrap();
        let values: Vec<f64> = (0..n * n)
            .filter(|k| bayer(k % n, k / n) == colour)
            .map(|k| f64::from(raw[k as usize]))
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        (mean, var)
    }

    #[test]
    fn noise_matches_the_model_in_mean_and_variance() {
        let m = SensorModel::default();
        for (irradiance, ms, gain, colour) in [
            (100.0f32, 10u64, 1.0f32, 1usize),
            (100.0, 10, 4.0, 0),
            (1.0, 2, 1.0, 2),
            (0.0, 50, 2.0, 1),
            (500.0, 20, 1.0, 1),
        ] {
            let e = exposure(ms, gain, 12);
            let t = ms as f64;
            let electrons = f64::from(irradiance) * f64::from(m.quantum_efficiency[colour]) * t
                + f64::from(m.dark_current) * t * 1e-3;
            let k = f64::from(gain / m.electrons_per_count);
            // Poisson's variance is its mean; read noise adds its square; counts round.
            let mean = f64::from(m.black_level) + electrons * k;
            let var = (electrons + f64::from(m.read_noise).powi(2)) * k * k + 1.0 / 12.0;
            let (got_mean, got_var) = statistics(&m, &e, irradiance, colour);
            assert!(
                (got_mean - mean).abs() < 0.02 * mean.max(1.0) + 0.1,
                "{irradiance} {ms} ms ×{gain}: mean {got_mean} against {mean}"
            );
            assert!(
                (got_var - var).abs() < 0.08 * var + 0.1,
                "{irradiance} {ms} ms ×{gain}: variance {got_var} against {var}"
            );
        }
    }

    #[test]
    fn small_counts_are_poisson_and_large_ones_gaussian() {
        let s = Seed::new(5);
        for mean in [0.5f32, 3.0, 12.0, POISSON_EXACT + 0.5, 400.0] {
            let draws: Vec<f64> = (0..40_000)
                .map(|k| f64::from(poisson(s, k % 200, k / 200, 9, mean)))
                .collect();
            let m = draws.iter().sum::<f64>() / draws.len() as f64;
            let v = draws.iter().map(|d| (d - m).powi(2)).sum::<f64>() / (draws.len() - 1) as f64;
            let mean = f64::from(mean);
            assert!(
                (m - mean).abs() < 0.03 * mean + 0.02 && (v - mean).abs() < 0.06 * mean + 0.05,
                "{mean}: {m}, {v}"
            );
            assert!(
                draws.iter().all(|d| *d >= 0.0 && d.fract() == 0.0),
                "whole counts"
            );
        }
        assert_eq!(poisson(s, 0, 0, 0, 0.0), 0.0);
        assert_eq!(poisson(s, 0, 0, 0, -1.0), 0.0);
    }

    #[test]
    fn the_well_fills_and_the_counts_end_at_their_bits() {
        let m = SensorModel::default();
        let bright = vec![[1e9f32; 3]; 4];
        let raw = expose(&bright, 2, 2, &m, &exposure(10, 1.0, 12), 1, 0).unwrap();
        let full = (f32::from(m.black_level) + m.full_well / m.electrons_per_count).round() as u16;
        assert_eq!(raw, vec![full; 4], "a full well");
        let gained = expose(&bright, 2, 2, &m, &exposure(10, 16.0, 12), 1, 0).unwrap();
        assert_eq!(gained, vec![4_095; 4], "12 bits at most");
        let eight = expose(&bright, 2, 2, &m, &exposure(10, 16.0, 8), 1, 0).unwrap();
        assert_eq!(eight, vec![255; 4], "8 bits");
        let dark = expose(
            &[[0.0; 3]; 4],
            2,
            2,
            &SensorModel {
                read_noise: 0.0,
                dark_current: 0.0,
                ..m
            },
            &exposure(10, 1.0, 12),
            1,
            0,
        )
        .unwrap();
        assert_eq!(dark, vec![m.black_level; 4], "the black level in the dark");
    }

    #[test]
    fn binning_sums_its_pixels_light_and_charge() {
        let m = SensorModel {
            read_noise: 0.0,
            dark_current: 0.0,
            ..SensorModel::default()
        };
        let one = Exposure {
            binning: 1,
            ..exposure(10, 1.0, 12)
        };
        let two = Exposure { binning: 2, ..one };
        let (a, _) = statistics(&m, &one, 20.0, 1);
        let (b, _) = statistics(&m, &two, 20.0, 1);
        let black = f64::from(m.black_level);
        assert!(
            ((b - black) / (a - black) - 4.0).abs() < 0.05,
            "four pixels' light: {a} then {b}"
        );
    }

    #[test]
    fn dark_current_grows_with_the_exposure_and_the_binning() {
        let m = SensorModel {
            read_noise: 0.0,
            dark_current: 50.0,
            ..SensorModel::default()
        };
        let black = f64::from(m.black_level);
        let k = f64::from(m.electrons_per_count);
        for (secs, binning, electrons) in [(1u64, 1u32, 50.0), (4, 1, 200.0), (2, 2, 400.0)] {
            let e = Exposure {
                time: Duration::from_secs(secs),
                binning,
                ..exposure(0, 1.0, 12)
            };
            let (mean, _) = statistics(&m, &e, 0.0, 1);
            assert!(
                ((mean - black) * k - electrons).abs() < 0.03 * electrons,
                "{secs} s binned {binning}: {mean} counts, {electrons} e⁻ expected"
            );
        }
    }

    #[test]
    fn a_frame_repeats_from_its_seed_and_differs_by_its_number() {
        let m = SensorModel::default();
        let field = vec![[200.0f32, 300.0, 150.0]; 64];
        let e = exposure(5, 2.0, 12);
        let a = expose(&field, 8, 8, &m, &e, 11, 4).unwrap();
        assert_eq!(a, expose(&field, 8, 8, &m, &e, 11, 4).unwrap());
        assert_ne!(
            a,
            expose(&field, 8, 8, &m, &e, 11, 5).unwrap(),
            "the next frame's noise"
        );
        assert_ne!(
            a,
            expose(&field, 8, 8, &m, &e, 12, 4).unwrap(),
            "another seed"
        );
        // Each of the high bits: 8 << 32 is one the salts' own bits share.
        for high in [1u64 << 40, 8 << 32] {
            assert_ne!(
                a,
                expose(&field, 8, 8, &m, &e, 11, 4 + high).unwrap(),
                "a frame number's high bits, {high:#x}"
            );
        }
    }
}
