//! The twin harness (ADR-0010): every CUDA kernel has a CPU twin, and the two must agree
//! within a stated tolerance. [`compare`] measures how far apart two results are; tests then
//! check that nothing is beyond the tolerance. `cuda_or_skip!()` gives a test the GPU, or
//! skips it when `TETERA_CUDA=1` isn't set (plan §5.4).

use std::fmt;
use tetera_core::{Error, Result};

/// How far a CUDA result may be from its CPU twin: `|actual − expected|` at most
/// `absolute + relative × |expected|`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// The difference allowed whatever the value.
    pub absolute: f64,
    /// The difference allowed as a fraction of the expected value.
    pub relative: f64,
}

impl Tolerance {
    /// No difference at all: bit for bit, for floats that aren't NaN.
    pub const EXACT: Self = Self {
        absolute: 0.0,
        relative: 0.0,
    };

    /// A difference of at most `absolute`.
    pub const fn absolute(absolute: f64) -> Self {
        Self {
            absolute,
            relative: 0.0,
        }
    }

    /// A difference of at most `relative` × the expected value.
    pub const fn relative(relative: f64) -> Self {
        Self {
            absolute: 0.0,
            relative,
        }
    }

    /// Whether `actual` is close enough to `expected`. NaN matches only NaN; an infinity
    /// matches only the same infinity, whatever the tolerance.
    pub fn allows(&self, expected: f64, actual: f64) -> bool {
        if expected == actual || (expected.is_nan() && actual.is_nan()) {
            return true;
        }
        // Otherwise a NaN or an infinity is never close: a relative tolerance of an infinity
        // would be infinite, and allow anything.
        if !(expected.is_finite() && actual.is_finite()) {
            return false;
        }
        (actual - expected).abs() <= self.absolute + self.relative * expected.abs()
    }
}

/// The value where two results are furthest apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Worst {
    /// Its index.
    pub index: usize,
    /// The CPU's value.
    pub expected: f64,
    /// The GPU's value.
    pub actual: f64,
}

/// How two results compare.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// How many values were compared.
    pub len: usize,
    /// How many are beyond the tolerance.
    pub beyond: usize,
    /// Where they are furthest apart, if they differ anywhere.
    pub worst: Option<Worst>,
}

impl Comparison {
    /// Whether every value is within the tolerance.
    pub fn passes(&self) -> bool {
        self.beyond == 0
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} values, {} beyond the tolerance",
            self.len, self.beyond
        )?;
        match self.worst {
            None => write!(f, "; identical"),
            Some(w) => write!(
                f,
                "; the largest difference is {:e} at {} ({} on the CPU, {} on CUDA)",
                (w.actual - w.expected).abs(),
                w.index,
                w.expected,
                w.actual
            ),
        }
    }
}

/// How far `actual` (CUDA's) is from `expected` (the CPU twin's), value by value.
pub fn compare<T: Copy + Into<f64>>(
    expected: &[T],
    actual: &[T],
    tolerance: Tolerance,
) -> Result<Comparison> {
    if expected.len() != actual.len() {
        return Err(Error::invalid_argument(format!(
            "twins of different lengths: {} on the CPU, {} on CUDA",
            expected.len(),
            actual.len()
        )));
    }
    let mut comparison = Comparison {
        len: expected.len(),
        beyond: 0,
        worst: None,
    };
    let mut largest = 0.0f64;
    for (index, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        let (e, a): (f64, f64) = (e.into(), a.into());
        if !tolerance.allows(e, a) {
            comparison.beyond += 1;
        }
        let both_nan = e.is_nan() && a.is_nan();
        if e != a && !both_nan {
            let difference = if e.is_nan() || a.is_nan() {
                f64::INFINITY
            } else {
                (a - e).abs()
            };
            if comparison.worst.is_none() || difference > largest {
                largest = difference;
                comparison.worst = Some(Worst {
                    index,
                    expected: e,
                    actual: a,
                });
            }
        }
    }
    Ok(comparison)
}

/// The GPU for a test, or `None` when `TETERA_CUDA=1` isn't set. With it set, a device that
/// won't start fails the test: the switch promises one.
#[cfg(feature = "cuda")]
pub fn gpu_for_tests() -> Option<crate::CudaBackend> {
    if !crate::select::cuda_tests_enabled() {
        return None;
    }
    Some(crate::CudaBackend::new(0).expect("TETERA_CUDA=1 promises a working CUDA device 0"))
}

/// The GPU for a test, or a note and a return when `TETERA_CUDA=1` isn't set.
#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! cuda_or_skip {
    () => {
        match $crate::twin::gpu_for_tests() {
            Some(gpu) => gpu,
            None => {
                eprintln!("skipped: set TETERA_CUDA=1 to run CUDA tests");
                return;
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_results_pass_exactly() {
        let c = compare(
            &[1.0f32, 2.0, f32::NAN],
            &[1.0, 2.0, f32::NAN],
            Tolerance::EXACT,
        )
        .unwrap();
        assert!(c.passes());
        assert_eq!((c.len, c.beyond, c.worst), (3, 0, None));
        assert_eq!(c.to_string(), "3 values, 0 beyond the tolerance; identical");
    }

    #[test]
    fn a_difference_beyond_the_tolerance_fails_and_is_located() {
        let c = compare(
            &[0u16, 10, 20, 30],
            &[0, 11, 20, 27],
            Tolerance::absolute(1.0),
        )
        .unwrap();
        assert!(!c.passes());
        assert_eq!(c.beyond, 1, "11 for 10 is within 1; 27 for 30 isn't");
        assert_eq!(
            c.worst,
            Some(Worst {
                index: 3,
                expected: 30.0,
                actual: 27.0
            })
        );
        assert_eq!(
            c.to_string(),
            "4 values, 1 beyond the tolerance; the largest difference is 3e0 at 3 (30 on the CPU, 27 on CUDA)"
        );
    }

    #[test]
    fn the_worst_is_the_largest_difference_even_within_the_tolerance() {
        let c = compare(&[1.0f64, 2.0], &[1.5, 2.25], Tolerance::absolute(1.0)).unwrap();
        assert!(c.passes());
        assert_eq!(c.worst.unwrap().index, 0);
        let first_of_equals =
            compare(&[1.0f64, 2.0], &[1.5, 2.5], Tolerance::absolute(1.0)).unwrap();
        assert_eq!(first_of_equals.worst.unwrap().index, 0);
    }

    #[test]
    fn tolerances_add_their_absolute_and_relative_parts() {
        let t = Tolerance {
            absolute: 0.5,
            relative: 0.01,
        };
        assert!(t.allows(100.0, 101.5), "0.5 + 1% of 100");
        assert!(!t.allows(100.0, 101.6));
        assert!(
            t.allows(-100.0, -101.5),
            "the relative part uses the magnitude"
        );
        assert!(Tolerance::relative(0.1).allows(10.0, 11.0));
        assert!(!Tolerance::relative(0.1).allows(10.0, 11.01));
        assert!(Tolerance::absolute(0.0).allows(3.0, 3.0));
        assert!(!Tolerance::EXACT.allows(1.0, 1.0 + f64::EPSILON));
    }

    #[test]
    fn nan_and_infinity_match_only_themselves() {
        let wide = Tolerance::absolute(1e300);
        assert!(wide.allows(f64::NAN, f64::NAN));
        assert!(!wide.allows(f64::NAN, 0.0) && !wide.allows(0.0, f64::NAN));
        assert!(wide.allows(f64::INFINITY, f64::INFINITY));
        assert!(!wide.allows(f64::INFINITY, f64::NEG_INFINITY));
        assert!(!wide.allows(f64::INFINITY, 1e300));
        let relative = Tolerance::relative(0.5);
        assert!(relative.allows(f64::INFINITY, f64::INFINITY));
        assert!(
            !relative.allows(f64::INFINITY, 1e300),
            "half of infinity is no limit"
        );
        assert!(!relative.allows(f64::INFINITY, f64::NEG_INFINITY));
        assert!(!relative.allows(f64::NAN, 1.0) && !relative.allows(1.0, f64::NAN));
        let c = compare(
            &[1.0f32, f32::NAN],
            &[f32::NAN, f32::NAN],
            Tolerance::absolute(10.0),
        )
        .unwrap();
        assert_eq!(c.beyond, 1);
        let worst = c.worst.unwrap();
        assert_eq!(worst.index, 0);
        assert!(worst.actual.is_nan());
    }

    #[test]
    fn a_nan_against_a_number_is_the_worst_difference() {
        let c = compare(&[1.0f64, 2.0], &[1.5, f64::NAN], Tolerance::absolute(1.0)).unwrap();
        assert_eq!(c.beyond, 1);
        assert_eq!(c.worst.unwrap().index, 1, "after a finite difference too");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_switch_decides_whether_cuda_tests_get_a_gpu() {
        let gpu = gpu_for_tests();
        assert_eq!(gpu.is_some(), crate::select::cuda_tests_enabled());
    }

    #[test]
    fn results_of_different_lengths_cant_be_compared() {
        let err = compare(&[1u16, 2], &[1], Tolerance::EXACT).unwrap_err();
        assert_eq!(
            err.to_string(),
            "twins of different lengths: 2 on the CPU, 1 on CUDA"
        );
    }
}
