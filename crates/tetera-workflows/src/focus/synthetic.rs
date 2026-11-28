//! A synthetic [`FocusProbe`](super::FocusProbe), for tests and examples: a stage whose
//! carriage lags its screw by its backlash and moves in whole steps, and a focus curve of a
//! known shape, peak and width, with noise.

use super::FocusProbe;
use tetera_core::Result;

/// A focus curve's shape around its peak, in units of its width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// `exp(−t² / 2)`.
    Gaussian,
    /// `1 / (1 + t²)`.
    Lorentzian,
    /// No peak: a blank field.
    Flat,
}

/// A stage with backlash and steps, and the measure of where its carriage really is:
/// `(base + height · shape) · (1 + noise · n)`, n a standard normal.
#[derive(Debug, Clone)]
pub struct Synthetic {
    /// Where the controller has put the screw, which is what it reports.
    screw: f64,
    /// Where the carriage is: within the backlash below the screw.
    carriage: f64,
    /// The play between screw and carriage, µm.
    pub backlash: f64,
    /// The stage's step, µm; 0 for none.
    pub resolution: f64,
    /// The carriage's z at focus, µm.
    pub focus: f64,
    /// The curve's width, µm.
    pub width: f64,
    /// Its shape.
    pub shape: Shape,
    /// The measure far from focus.
    pub base: f64,
    /// Its rise at focus.
    pub height: f64,
    /// Its noise, as a fraction of it.
    pub noise: f64,
    rng: u64,
    rose: bool,
    /// Each move, from and to, as the screw went.
    pub moves: Vec<(f64, f64)>,
    /// For each measure, whether the move before it rose.
    pub measured_rising: Vec<bool>,
}

impl Synthetic {
    /// A stage at `start`, its carriage there too, over a Gaussian curve 2 µm wide peaking at
    /// `focus`, from 1 to 4, without noise, backlash or steps; its noise from `seed`.
    pub fn new(start: f64, focus: f64, seed: u64) -> Self {
        Self {
            screw: start,
            carriage: start,
            backlash: 0.0,
            resolution: 0.0,
            focus,
            width: 2.0,
            shape: Shape::Gaussian,
            base: 1.0,
            height: 3.0,
            noise: 0.0,
            rng: seed,
            rose: false,
            moves: Vec::new(),
            measured_rising: Vec::new(),
        }
    }

    /// Where the carriage really is, µm.
    pub fn carriage(&self) -> f64 {
        self.carriage
    }

    /// A number in [0, 1), from its seed (SplitMix64).
    pub fn uniform(&mut self) -> f64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard normal, by Box and Muller.
    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(f64::MIN_POSITIVE);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

impl FocusProbe for Synthetic {
    fn z(&mut self) -> Result<f64> {
        Ok(self.screw)
    }

    fn move_to(&mut self, z: f64) -> Result<()> {
        let target = if self.resolution > 0.0 {
            (z / self.resolution).round() * self.resolution
        } else {
            z
        };
        if target > self.screw {
            self.carriage = self.carriage.max(target - self.backlash);
            self.rose = true;
        } else if target < self.screw {
            self.carriage = self.carriage.min(target);
            self.rose = false;
        }
        self.moves.push((self.screw, target));
        self.screw = target;
        Ok(())
    }

    fn measure(&mut self) -> Result<f64> {
        self.measured_rising.push(self.rose);
        let t = (self.carriage - self.focus) / self.width;
        let peak = match self.shape {
            Shape::Gaussian => (-t * t / 2.0).exp(),
            Shape::Lorentzian => 1.0 / (1.0 + t * t),
            Shape::Flat => 0.0,
        };
        let n = self.normal();
        Ok((self.base + self.height * peak) * (1.0 + self.noise * n))
    }
}
