//! What a frame carries besides its pixels (ADR-0008): when it was exposed, how, through
//! what, and where the stage was.

use super::format::Binning;
use crate::geometry::{Rect, StagePos};
use crate::time::Timestamp;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// The light's state while a frame was exposed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LightState {
    /// Whether it was on.
    pub on: bool,
    /// Its intensity, from 0 to 1.
    pub intensity: f64,
    /// Its colour temperature in kelvin, where the light reports one.
    pub colour_temperature_k: Option<f64>,
}

/// A frame's metadata.
///
/// `exposure_start` and `exposure_end` are the first row's. With a rolling shutter, each
/// row starts `line_time` after the one above it: [`FrameMeta::row_midpoint`] places a row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FrameMeta {
    /// The frame's place in the camera's stream, from 0.
    pub seq: u64,
    /// When the first row's exposure started.
    pub exposure_start: Timestamp,
    /// When the first row's exposure ended.
    pub exposure_end: Timestamp,
    /// With a rolling shutter, the time from one row's start to the next's, in ms;
    /// `null` for a global shutter.
    #[serde(with = "crate::units::ms_opt", default)]
    #[schemars(with = "Option<f64>")]
    pub line_time: Option<Duration>,
    /// The exposure the camera was set to, in ms.
    #[serde(with = "crate::units::ms")]
    #[schemars(with = "f64")]
    pub exposure: Duration,
    /// The camera's gain.
    pub gain: f64,
    /// The camera's mode, as its driver numbers them.
    pub mode: u32,
    /// The binning.
    pub binning: Binning,
    /// The region of the sensor read out.
    pub roi: Rect,
    /// The black level, in the frame's sample units.
    pub black_level: u16,
    /// The objective fitted, by name.
    pub objective: Option<String>,
    /// The light's state.
    pub light: Option<LightState>,
    /// The stage's position at the exposure's midpoint, where it is known.
    pub position: Option<StagePos>,
    /// Frames the camera dropped just before this one.
    pub dropped_before: u64,
}

impl Default for FrameMeta {
    fn default() -> Self {
        Self {
            seq: 0,
            exposure_start: Timestamp::ZERO,
            exposure_end: Timestamp::ZERO,
            line_time: None,
            exposure: Duration::ZERO,
            gain: 1.0,
            mode: 0,
            binning: Binning::NONE,
            roi: Rect::default(),
            black_level: 0,
            objective: None,
            light: None,
            position: None,
            dropped_before: 0,
        }
    }
}

impl FrameMeta {
    /// The middle of the first row's exposure.
    pub fn midpoint(&self) -> Timestamp {
        let half = self.exposure_end.saturating_since(self.exposure_start) / 2;
        self.exposure_start
            .checked_add(half)
            .unwrap_or(self.exposure_end)
    }

    /// The middle of row `row`'s exposure: the first row's, plus `row` line times.
    pub fn row_midpoint(&self, row: u32) -> Timestamp {
        let delay = self.line_time.unwrap_or_default().saturating_mul(row);
        self.midpoint()
            .checked_add(delay)
            .unwrap_or(Timestamp(u64::MAX))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::Um;

    fn exposed(start_ms: u64, end_ms: u64) -> FrameMeta {
        FrameMeta {
            exposure_start: Timestamp::from_duration(Duration::from_millis(start_ms)),
            exposure_end: Timestamp::from_duration(Duration::from_millis(end_ms)),
            ..FrameMeta::default()
        }
    }

    #[test]
    fn the_midpoint_is_halfway_through_the_first_row() {
        let meta = exposed(100, 120);
        assert_eq!(meta.midpoint().as_duration(), Duration::from_millis(110));
        assert_eq!(
            exposed(5, 5).midpoint().as_duration(),
            Duration::from_millis(5)
        );
        assert_eq!(
            exposed(9, 3).midpoint().as_duration(),
            Duration::from_millis(9),
            "a backwards exposure has no length"
        );
    }

    #[test]
    fn a_rolling_shutter_delays_each_row() {
        let mut meta = exposed(100, 120);
        meta.line_time = Some(Duration::from_micros(10));
        assert_eq!(
            meta.row_midpoint(0).as_duration(),
            Duration::from_millis(110)
        );
        assert_eq!(
            meta.row_midpoint(3_647).as_duration(),
            Duration::from_millis(110) + Duration::from_micros(36_470)
        );
        meta.line_time = None;
        assert_eq!(
            meta.row_midpoint(3_647),
            meta.midpoint(),
            "a global shutter exposes every row together"
        );
        meta.line_time = Some(Duration::MAX);
        assert_eq!(meta.row_midpoint(2), Timestamp(u64::MAX), "it saturates");
    }

    #[test]
    fn metadata_round_trips_with_durations_in_milliseconds() {
        let meta = FrameMeta {
            seq: 41,
            line_time: Some(Duration::from_nanos(10_500)),
            exposure: Duration::from_micros(12_500),
            gain: 2.0,
            mode: 1,
            binning: Binning::new(2, 2).unwrap(),
            roi: Rect::new(0, 0, 2_736, 1_824),
            black_level: 2_000,
            objective: Some("10x".into()),
            light: Some(LightState {
                on: true,
                intensity: 0.8,
                colour_temperature_k: Some(3_200.0),
            }),
            position: Some(StagePos::new(Um(1.0), Um(2.0), Um(3.5))),
            dropped_before: 1,
            ..exposed(100, 112)
        };
        let json = serde_json::to_value(&meta).unwrap();
        assert_eq!(json["exposure"], 12.5);
        assert_eq!(json["line_time"], 0.0105);
        assert_eq!(json["exposure_start"], 100_000_000);
        assert_eq!(serde_json::from_value::<FrameMeta>(json).unwrap(), meta);
        let global = serde_json::to_value(FrameMeta::default()).unwrap();
        assert!(global["line_time"].is_null());
        assert_eq!(
            serde_json::from_value::<FrameMeta>(global).unwrap(),
            FrameMeta::default()
        );
    }
}
