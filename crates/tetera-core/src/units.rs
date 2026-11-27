//! Units (ADR-0006): lengths in micrometres as [`Um`], times as [`std::time::Duration`].
//! Every API speaks µm as `f64` and ms as `f64` (plan D18); the conversions live here, at
//! the edge, so nothing inside mixes them up.

use crate::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::time::Duration;

/// A length in micrometres.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, PartialOrd, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct Um(pub f64);

impl Um {
    /// No length.
    pub const ZERO: Self = Self(0.0);

    /// A length given in millimetres.
    pub fn from_mm(mm: f64) -> Self {
        Self(mm * 1_000.0)
    }

    /// A length given in nanometres.
    pub fn from_nm(nm: f64) -> Self {
        Self(nm / 1_000.0)
    }

    /// The length in micrometres.
    pub fn um(self) -> f64 {
        self.0
    }

    /// The length in millimetres.
    pub fn mm(self) -> f64 {
        self.0 / 1_000.0
    }

    /// The length in nanometres.
    pub fn nm(self) -> f64 {
        self.0 * 1_000.0
    }

    /// The length without its sign.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
}

impl fmt::Display for Um {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match f.precision() {
            Some(p) => write!(f, "{:.*} µm", p, self.0),
            None => write!(f, "{} µm", self.0),
        }
    }
}

impl Add for Um {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Um {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Um {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Um {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for Um {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Mul<f64> for Um {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for Um {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self(self.0 / rhs)
    }
}

/// The ratio of two lengths.
impl Div for Um {
    type Output = f64;
    fn div(self, rhs: Self) -> f64 {
        self.0 / rhs.0
    }
}

/// A duration given in milliseconds, as APIs give it. A negative, infinite or NaN value is
/// refused, as is one too large for a [`Duration`].
pub fn duration_from_ms(ms: f64) -> Result<Duration> {
    if !ms.is_finite() || ms < 0.0 {
        return Err(Error::invalid_argument(format!(
            "a duration must be zero or more milliseconds, not {ms}"
        )));
    }
    Duration::try_from_secs_f64(ms / 1_000.0)
        .map_err(|_| Error::invalid_argument(format!("{ms} ms is too long a duration")))
}

/// A duration in milliseconds, as APIs give it.
pub fn duration_ms(d: Duration) -> f64 {
    // Whole nanoseconds over a million: one correctly rounded division, so 12.5 ms reads 12.5.
    d.as_nanos() as f64 / 1_000_000.0
}

/// Serde for a [`Duration`] as milliseconds, as the API speaks it (ADR-0006):
/// `#[serde(with = "tetera_core::units::ms")]`.
pub mod ms {
    use super::{duration_from_ms, duration_ms};
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    /// Writes `d` as milliseconds.
    pub fn serialize<S: Serializer>(d: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(duration_ms(*d))
    }

    /// Reads milliseconds, refusing what [`duration_from_ms`] refuses.
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        duration_from_ms(f64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Serde for an optional [`Duration`] as milliseconds, or `null`.
pub mod ms_opt {
    use super::{duration_from_ms, duration_ms};
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    /// Writes `d` as milliseconds, or `null`.
    pub fn serialize<S: Serializer>(
        d: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match d {
            Some(d) => serializer.serialize_some(&duration_ms(*d)),
            None => serializer.serialize_none(),
        }
    }

    /// Reads milliseconds or `null`, refusing what [`duration_from_ms`] refuses.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        Option::<f64>::deserialize(deserializer)?
            .map(duration_from_ms)
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn lengths_convert_between_units() {
        assert_eq!(Um::from_mm(1.5), Um(1_500.0));
        assert_eq!(Um::from_nm(550.0), Um(0.55));
        assert_eq!(Um(2_500.0).mm(), 2.5);
        assert_eq!(Um(0.6).nm(), 600.0);
        assert_eq!(Um(-3.0).abs(), Um(3.0));
        assert_eq!(Um(4.0).um(), 4.0);
    }

    #[test]
    fn lengths_add_scale_and_compare() {
        let mut a = Um(10.0) + Um(2.5) - Um(0.5);
        assert_eq!(a, Um(12.0));
        a += Um(1.0);
        a -= Um(3.0);
        assert_eq!(a, Um(10.0));
        assert_eq!(-a, Um(-10.0));
        assert_eq!(a * 2.0, Um(20.0));
        assert_eq!(a / 4.0, Um(2.5));
        assert_eq!(Um(9.0) / Um(3.0), 3.0);
        assert!(Um(1.0) < Um(2.0));
        assert_eq!(Um::default(), Um::ZERO);
    }

    #[test]
    fn lengths_print_with_their_unit() {
        assert_eq!(Um(12.5).to_string(), "12.5 µm");
        assert_eq!(format!("{:.2}", Um(0.04812)), "0.05 µm");
    }

    #[test]
    fn lengths_serialise_as_plain_numbers() {
        assert_eq!(serde_json::to_string(&Um(1.25)).unwrap(), "1.25");
        assert_eq!(serde_json::from_str::<Um>("3").unwrap(), Um(3.0));
    }

    #[test]
    fn milliseconds_convert_to_durations_and_back() {
        assert_eq!(duration_from_ms(1.5).unwrap(), Duration::from_micros(1_500));
        assert_eq!(duration_from_ms(0.0).unwrap(), Duration::ZERO);
        assert_eq!(duration_ms(Duration::from_millis(250)), 250.0);
    }

    #[test]
    fn impossible_durations_are_refused() {
        for ms in [-1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            let err = duration_from_ms(ms).unwrap_err();
            assert_eq!(err.kind(), crate::ErrorKind::InvalidArgument, "{ms}");
        }
    }

    #[test]
    fn a_negative_or_unreadable_duration_is_named_as_such() {
        for ms in [-1.0, f64::NAN, f64::NEG_INFINITY] {
            let err = duration_from_ms(ms).unwrap_err();
            assert!(
                err.message().contains("zero or more milliseconds"),
                "{ms}: {err}"
            );
        }
        assert!(
            duration_from_ms(f64::MAX)
                .unwrap_err()
                .message()
                .contains("too long")
        );
    }

    #[test]
    fn durations_serialise_as_milliseconds() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Timing {
            #[serde(with = "ms")]
            exposure: Duration,
            #[serde(with = "ms_opt", default)]
            line: Option<Duration>,
        }
        let t = Timing {
            exposure: Duration::from_micros(12_500),
            line: Some(Duration::from_nanos(10_500)),
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#"{"exposure":12.5,"line":0.0105}"#);
        assert_eq!(serde_json::from_str::<Timing>(&json).unwrap(), t);
        let none: Timing = serde_json::from_str(r#"{"exposure":1,"line":null}"#).unwrap();
        assert_eq!(none.line, None);
        assert_eq!(
            serde_json::from_str::<Timing>(r#"{"exposure":2}"#)
                .unwrap()
                .line,
            None
        );
        assert!(serde_json::from_str::<Timing>(r#"{"exposure":-1}"#).is_err());
        assert!(serde_json::from_str::<Timing>(r#"{"exposure":1,"line":-2}"#).is_err());
    }

    proptest! {
        #[test]
        fn millimetres_round_trip(mm in -1.0e4f64..1.0e4) {
            prop_assert!((Um::from_mm(mm).mm() - mm).abs() <= 1e-12 * mm.abs().max(1.0));
        }

        #[test]
        fn nanometres_round_trip(nm in -1.0e7f64..1.0e7) {
            prop_assert!((Um::from_nm(nm).nm() - nm).abs() <= 1e-9 * nm.abs().max(1.0));
        }

        #[test]
        fn milliseconds_round_trip(ms in 0.0f64..1.0e9) {
            let back = duration_ms(duration_from_ms(ms).unwrap());
            prop_assert!((back - ms).abs() <= 1e-6, "{ms} -> {back}");
        }
    }
}
