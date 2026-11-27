//! Time (ADR-0006): one monotonic clock for every timestamp, mapped once to UTC.
//!
//! A [`Timestamp`] counts nanoseconds since its clock started. A clock maps its start to a
//! [`UtcTime`] once; [`Clock::drift`] then says how far the wall clock has moved from that
//! mapping, as WSL2's can after the host sleeps. Beyond [`DRIFT_LIMIT`], the drift is worth
//! a warning. P1.10a's simulated clock implements [`Clock`] too.

use crate::error::{Error, Result};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Nanoseconds since a clock started.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(transparent)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// The clock's start.
    pub const ZERO: Self = Self(0);

    /// The timestamp `d` after the clock's start, saturating at the largest one.
    pub fn from_duration(d: Duration) -> Self {
        Self(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
    }

    /// The time since the clock's start.
    pub fn as_duration(self) -> Duration {
        Duration::from_nanos(self.0)
    }

    /// The time from `earlier` to this, or zero if `earlier` is later.
    pub fn saturating_since(self, earlier: Timestamp) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }

    /// The timestamp `d` later, saturating at the largest one.
    pub fn saturating_add(self, d: Duration) -> Timestamp {
        Self::from_duration(self.as_duration().saturating_add(d))
    }

    /// The timestamp `d` later, unless it would overflow.
    pub fn checked_add(self, d: Duration) -> Option<Timestamp> {
        u64::try_from(d.as_nanos())
            .ok()
            .and_then(|n| self.0.checked_add(n))
            .map(Self)
    }
}

/// A moment in UTC: nanoseconds since 1970-01-01T00:00:00Z. It reads and writes as
/// RFC 3339, such as `2026-11-26T22:53:20.25Z`, and covers the years 1678 to 2261.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UtcTime(i64);

impl UtcTime {
    /// 1970-01-01T00:00:00Z.
    pub const UNIX_EPOCH: Self = Self(0);

    /// The moment `nanos` nanoseconds after the Unix epoch.
    pub const fn from_unix_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    /// Nanoseconds since the Unix epoch.
    pub const fn unix_nanos(self) -> i64 {
        self.0
    }

    /// The wall clock's time now.
    pub fn now() -> Self {
        let nanos = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(after) => i64::try_from(after.as_nanos()).unwrap_or(i64::MAX),
            Err(before) => i64::try_from(before.duration().as_nanos()).map_or(i64::MIN, |n| -n),
        };
        Self(nanos)
    }

    /// The moment `d` later, saturating at the range's end.
    pub fn saturating_add(self, d: Duration) -> Self {
        Self(
            self.0
                .saturating_add(i64::try_from(d.as_nanos()).unwrap_or(i64::MAX)),
        )
    }

    /// Nanoseconds from `earlier` to this: negative if `earlier` is later.
    pub fn nanos_since(self, earlier: UtcTime) -> i64 {
        self.0.saturating_sub(earlier.0)
    }
}

const NANOS_PER_SEC: i64 = 1_000_000_000;
const SECS_PER_DAY: i64 = 86_400;

/// Days since 1970-01-01 of a proleptic Gregorian date, after Howard Hinnant's
/// `days_from_civil` ("chrono-Compatible Low-Level Date Algorithms").
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The date of a day counted from 1970-01-01, after Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

impl fmt::Display for UtcTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.div_euclid(NANOS_PER_SEC);
        let nanos = self.0.rem_euclid(NANOS_PER_SEC);
        let (year, month, day) = civil_from_days(secs.div_euclid(SECS_PER_DAY));
        let of_day = secs.rem_euclid(SECS_PER_DAY);
        let (h, m, s) = (of_day / 3_600, of_day / 60 % 60, of_day % 60);
        write!(f, "{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}")?;
        if nanos != 0 {
            let fraction = format!("{nanos:09}");
            write!(f, ".{}", fraction.trim_end_matches('0'))?;
        }
        f.write_str("Z")
    }
}

impl FromStr for UtcTime {
    type Err = Error;

    /// Reads RFC 3339 in UTC: `YYYY-MM-DDTHH:MM:SS`, an optional fraction of up to nine
    /// digits, and `Z`.
    fn from_str(s: &str) -> Result<Self> {
        let bad = || {
            Error::invalid_argument(format!(
                "`{s}` isn't a UTC time such as 2026-11-26T22:53:20Z"
            ))
        };
        let body = s
            .strip_suffix('Z')
            .or_else(|| s.strip_suffix('z'))
            .ok_or_else(bad)?;
        let (main, fraction) = match body.split_once('.') {
            Some((main, fraction)) => (main, Some(fraction)),
            None => (body, None),
        };
        let b = main.as_bytes();
        let shape = b.len() == 19
            && [4, 7].iter().all(|&i| b[i] == b'-')
            && matches!(b[10], b'T' | b't')
            && [13, 16].iter().all(|&i| b[i] == b':');
        if !shape {
            return Err(bad());
        }
        let num = |range: std::ops::Range<usize>| -> Result<i64> {
            let part = &main[range];
            if part.bytes().all(|c| c.is_ascii_digit()) {
                part.parse().map_err(|_| bad())
            } else {
                Err(bad())
            }
        };
        let (year, month, day) = (num(0..4)?, num(5..7)?, num(8..10)?);
        let (hour, minute, second) = (num(11..13)?, num(14..16)?, num(17..19)?);
        if !(1..=12).contains(&month)
            || day < 1
            || day > days_in_month(year, month)
            || hour > 23
            || minute > 59
            || second > 59
        {
            return Err(bad());
        }
        let nanos = match fraction {
            None => 0,
            Some(f) if !f.is_empty() && f.len() <= 9 && f.bytes().all(|c| c.is_ascii_digit()) => {
                format!("{f:0<9}").parse::<i64>().map_err(|_| bad())?
            }
            Some(_) => return Err(bad()),
        };
        let secs =
            days_from_civil(year, month, day) * SECS_PER_DAY + hour * 3_600 + minute * 60 + second;
        secs.checked_mul(NANOS_PER_SEC)
            .and_then(|n| n.checked_add(nanos))
            .map(Self)
            .ok_or_else(|| {
                Error::invalid_argument(format!("`{s}` is outside the years 1678 to 2261"))
            })
    }
}

impl Serialize for UtcTime {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for UtcTime {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = Cow::<str>::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for UtcTime {
    fn schema_name() -> Cow<'static, str> {
        "UtcTime".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({ "type": "string", "format": "date-time", "description": "A moment in UTC, in RFC 3339" })
    }
}

/// A drift beyond this, either way, is worth a warning.
pub const DRIFT_LIMIT: Duration = Duration::from_secs(1);

/// Whether a drift of `drift_nanos` is beyond [`DRIFT_LIMIT`].
pub fn drift_exceeds_limit(drift_nanos: i64) -> bool {
    u128::from(drift_nanos.unsigned_abs()) > DRIFT_LIMIT.as_nanos()
}

/// A monotonic clock, mapped once to UTC.
pub trait Clock: Send + Sync {
    /// Now, on this clock. It never goes backwards.
    fn now(&self) -> Timestamp;

    /// The UTC time of this clock's start, as mapped when it started.
    fn origin(&self) -> UtcTime;

    /// The wall clock's UTC time now, for [`Clock::drift`].
    fn wall_now(&self) -> UtcTime;

    /// `t` in UTC, through the mapping made at the start.
    fn utc(&self, t: Timestamp) -> UtcTime {
        self.origin().saturating_add(t.as_duration())
    }

    /// Nanoseconds by which the wall clock is ahead of the mapping (negative when behind).
    fn drift(&self) -> i64 {
        self.wall_now().nanos_since(self.utc(self.now()))
    }
}

/// The host's monotonic clock, mapped to UTC when it is made.
#[derive(Debug, Clone)]
pub struct SystemClock {
    start: Instant,
    origin: UtcTime,
}

impl SystemClock {
    /// A clock that starts now.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            origin: UtcTime::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        Timestamp::from_duration(self.start.elapsed())
    }

    fn origin(&self) -> UtcTime {
        self.origin
    }

    fn wall_now(&self) -> UtcTime {
        UtcTime::now()
    }
}

/// A clock that moves only when told: for tests, and anything that must not depend on the
/// host's speed. Its wall clock can be offset, to stand for a drifting host.
#[derive(Debug)]
pub struct ManualClock {
    now: AtomicU64,
    origin: UtcTime,
    wall_offset: AtomicI64,
}

impl ManualClock {
    /// A clock at its start, which maps to `origin`.
    pub fn new(origin: UtcTime) -> Self {
        Self {
            now: AtomicU64::new(0),
            origin,
            wall_offset: AtomicI64::new(0),
        }
    }

    /// Moves the clock `d` forward.
    pub fn advance(&self, d: Duration) {
        let step = u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        let _ = self
            .now
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                Some(n.saturating_add(step))
            });
    }

    /// Moves the clock to `t`, unless `t` is earlier: the clock never goes back.
    pub fn set(&self, t: Timestamp) {
        self.now.fetch_max(t.0, Ordering::SeqCst);
    }

    /// Puts the wall clock `nanos` ahead of the mapping (behind, if negative).
    pub fn set_wall_offset(&self, nanos: i64) {
        self.wall_offset.store(nanos, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Timestamp {
        Timestamp(self.now.load(Ordering::SeqCst))
    }

    fn origin(&self) -> UtcTime {
        self.origin
    }

    fn wall_now(&self) -> UtcTime {
        UtcTime(
            self.utc(self.now())
                .0
                .saturating_add(self.wall_offset.load(Ordering::SeqCst)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn utc(s: &str) -> UtcTime {
        s.parse().unwrap()
    }

    #[test]
    fn known_moments_print_as_rfc_3339() {
        assert_eq!(UtcTime::UNIX_EPOCH.to_string(), "1970-01-01T00:00:00Z");
        assert_eq!(
            UtcTime::from_unix_nanos(1_000_000_000_000_000_000).to_string(),
            "2001-09-09T01:46:40Z"
        );
        assert_eq!(
            UtcTime::from_unix_nanos(1_709_164_800_250_000_000).to_string(),
            "2024-02-29T00:00:00.25Z"
        );
        assert_eq!(
            UtcTime::from_unix_nanos(-1).to_string(),
            "1969-12-31T23:59:59.999999999Z"
        );
        assert_eq!(
            UtcTime::from_unix_nanos(951_782_400_000_000_001).to_string(),
            "2000-02-29T00:00:00.000000001Z"
        );
    }

    #[test]
    fn rfc_3339_reads_back() {
        assert_eq!(utc("1970-01-01T00:00:00Z"), UtcTime::UNIX_EPOCH);
        assert_eq!(
            utc("2024-02-29T00:00:00.25Z").unix_nanos(),
            1_709_164_800_250_000_000
        );
        assert_eq!(
            utc("2026-11-26t22:53:20.5z"),
            utc("2026-11-26T22:53:20.500Z")
        );
        assert_eq!(
            utc("1900-03-01T00:00:00Z").unix_nanos(),
            -2_203_891_200 * NANOS_PER_SEC
        );
    }

    #[test]
    fn malformed_and_impossible_times_are_refused() {
        for bad in [
            "",
            "2026-11-26",
            "2026-11-26T22:53:20",
            "2026-11-26T22:53:20+01:00",
            "2026-13-01T00:00:00Z",
            "2026-02-29T00:00:00Z",
            "1900-02-29T00:00:00Z",
            "2026-04-31T00:00:00Z",
            "2026-11-26T24:00:00Z",
            "2026-11-26T23:60:00Z",
            "2026-11-26T23:59:60Z",
            "2026-11-26T23:59:59.Z",
            "2026-11-26T23:59:59.1234567890Z",
            "2026-11-26T23:59:5xZ",
            "+026-11-26T23:59:59Z",
            "9999-01-01T00:00:00Z",
        ] {
            let err = bad.parse::<UtcTime>().unwrap_err();
            assert_eq!(err.kind(), crate::ErrorKind::InvalidArgument, "{bad}");
        }
        assert!("2000-02-29T00:00:00Z".parse::<UtcTime>().is_ok());
    }

    #[test]
    fn utc_times_serialise_as_strings() {
        let t = utc("2026-11-26T22:53:20.125Z");
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "\"2026-11-26T22:53:20.125Z\"");
        assert_eq!(serde_json::from_str::<UtcTime>(&json).unwrap(), t);
        assert!(serde_json::from_str::<UtcTime>("\"yesterday\"").is_err());
        assert_eq!(UtcTime::schema_name(), "UtcTime");
    }

    #[test]
    fn right_length_wrong_separators_are_refused() {
        for bad in [
            "2026/11-26T22:53:20Z",
            "2026-11/26T22:53:20Z",
            "2026-11-26 22:53:20Z",
            "2026-11-26T22-53:20Z",
            "2026-11-26T22:53-20Z",
        ] {
            assert!(bad.parse::<UtcTime>().is_err(), "{bad}");
        }
    }

    #[test]
    fn every_day_from_the_year_minus_1000_to_3000_round_trips() {
        let (first, last) = (days_from_civil(-1000, 1, 1), days_from_civil(3000, 12, 31));
        let mut previous = civil_from_days(first - 1);
        for days in first..=last {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "{y}-{m}-{d}");
            let next_day = (y, m, d) == (previous.0, previous.1, previous.2 + 1)
                || (d == 1 && (m, y) == (previous.1 + 1, previous.0))
                || (d == 1 && m == 1 && y == previous.0 + 1);
            assert!(
                next_day && d <= days_in_month(y, m),
                "{previous:?} then {y}-{m}-{d}"
            );
            previous = (y, m, d);
        }
    }

    #[test]
    fn the_wall_clock_reads_today_and_the_system_clock_moves() {
        assert!(UtcTime::now() > utc("2026-01-01T00:00:00Z"));
        let clock = SystemClock::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while clock.now() < Timestamp::from_duration(Duration::from_millis(1)) {
            assert!(Instant::now() < deadline, "the clock didn't move");
        }
    }

    #[test]
    fn a_utc_times_schema_is_a_date_time_string() {
        let schema = schemars::schema_for!(UtcTime);
        assert_eq!(
            schema.get("format").and_then(|f| f.as_str()),
            Some("date-time")
        );
    }

    #[test]
    fn timestamps_measure_and_saturate() {
        let t = Timestamp::from_duration(Duration::from_millis(1_500));
        assert_eq!(t, Timestamp(1_500_000_000));
        assert_eq!(t.as_duration(), Duration::from_millis(1_500));
        assert_eq!(
            t.saturating_since(Timestamp(500_000_000)),
            Duration::from_secs(1)
        );
        assert_eq!(Timestamp::ZERO.saturating_since(t), Duration::ZERO);
        assert_eq!(
            t.checked_add(Duration::from_nanos(1)),
            Some(Timestamp(1_500_000_001))
        );
        assert_eq!(
            Timestamp(u64::MAX).checked_add(Duration::from_nanos(1)),
            None
        );
        assert_eq!(
            t.saturating_add(Duration::from_millis(1)),
            Timestamp(1_501_000_000)
        );
        assert_eq!(t.saturating_add(Duration::MAX), Timestamp(u64::MAX));
        assert_eq!(Timestamp::from_duration(Duration::MAX), Timestamp(u64::MAX));
    }

    #[test]
    fn the_system_clock_never_goes_back_and_maps_to_now() {
        let clock = SystemClock::new();
        let mut last = clock.now();
        for _ in 0..10_000 {
            let now = clock.now();
            assert!(now >= last);
            last = now;
        }
        let mapped = clock.utc(clock.now());
        assert!(
            UtcTime::now().nanos_since(mapped).abs() < 1_000_000_000,
            "{mapped}"
        );
        assert!(!drift_exceeds_limit(clock.drift()));
    }

    #[test]
    fn a_manual_clock_moves_only_forward_and_when_told() {
        let clock = ManualClock::new(utc("2026-11-26T00:00:00Z"));
        assert_eq!(clock.now(), Timestamp::ZERO);
        clock.advance(Duration::from_secs(90));
        assert_eq!(clock.utc(clock.now()), utc("2026-11-26T00:01:30Z"));
        clock.set(Timestamp(1));
        assert_eq!(
            clock.now(),
            Timestamp::from_duration(Duration::from_secs(90)),
            "it never goes back"
        );
        clock.set(Timestamp::from_duration(Duration::from_secs(120)));
        assert_eq!(clock.now().as_duration(), Duration::from_secs(120));
        assert_eq!(clock.drift(), 0);
    }

    #[test]
    fn a_drift_beyond_a_second_either_way_is_flagged() {
        let clock = ManualClock::new(UtcTime::UNIX_EPOCH);
        for (offset, flagged) in [
            (0, false),
            (999_999_999, false),
            (1_000_000_000, false),
            (1_000_000_001, true),
            (-1_000_000_001, true),
            (-500_000_000, false),
        ] {
            clock.set_wall_offset(offset);
            assert_eq!(clock.drift(), offset);
            assert_eq!(drift_exceeds_limit(clock.drift()), flagged, "{offset}");
        }
        assert!(drift_exceeds_limit(i64::MIN));
    }

    proptest! {
        #[test]
        fn every_representable_moment_round_trips(nanos in -9_000_000_000_000_000_000i64..9_000_000_000_000_000_000) {
            let t = UtcTime::from_unix_nanos(nanos);
            prop_assert_eq!(t.to_string().parse::<UtcTime>().unwrap(), t);
        }

        #[test]
        fn every_valid_date_round_trips_through_days(days in -100_000i64..100_000) {
            let (y, m, d) = civil_from_days(days);
            prop_assert!((1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m));
            prop_assert_eq!(days_from_civil(y, m, d), days);
        }
    }
}
