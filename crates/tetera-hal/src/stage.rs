//! The stage: what one can do, where it is, and the trait every stage implements, a real
//! driver, the virtual stage, a fake, or a bridge to another process (ADR-0013).
//!
//! Moves, jogs and homing start and return; [`Stage::wait`] waits until the stage is still.
//! Positions are in micrometres and speeds in micrometres a second (plan D18).

use crate::camera::Span;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::Receiver;
use std::time::Duration;
use tetera_core::{Error, ErrorKind, Result, StagePos, Timestamp, Um};

/// One of the stage's axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    /// Across the sample.
    X,
    /// Along the sample.
    Y,
    /// Focus.
    Z,
}

impl Axis {
    /// All three, in order.
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    /// Its coordinate in `pos`.
    pub fn of(self, pos: &StagePos) -> Um {
        match self {
            Axis::X => pos.x,
            Axis::Y => pos.y,
            Axis::Z => pos.z,
        }
    }

    /// `pos` with its coordinate set to `to`.
    pub fn with(self, pos: StagePos, to: Um) -> StagePos {
        match self {
            Axis::X => StagePos { x: to, ..pos },
            Axis::Y => StagePos { y: to, ..pos },
            Axis::Z => StagePos { z: to, ..pos },
        }
    }
}

/// What one axis can do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AxisCapabilities {
    /// Which axis.
    pub axis: Axis,
    /// Its travel, in µm.
    pub travel: Span,
    /// Its fastest speed, in µm/s.
    pub max_speed: f64,
    /// How fast it speeds up and slows down, in µm/s²: with the speed, the time a move takes
    /// (ADR-0022).
    pub acceleration: f64,
}

/// What a stage can do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StageCapabilities {
    /// Its maker and model.
    pub model: String,
    /// Its axes.
    pub axes: Vec<AxisCapabilities>,
    /// Its smallest step, in µm: a move ends within this of its target.
    pub resolution: Um,
    /// Where homing takes it, if it homes.
    pub home: Option<StagePos>,
}

impl StageCapabilities {
    /// What `axis` can do.
    pub fn axis(&self, axis: Axis) -> Result<&AxisCapabilities> {
        self.axes
            .iter()
            .find(|a| a.axis == axis)
            .ok_or_else(|| Error::invalid_argument(format!("the stage has no {axis:?} axis")))
    }

    /// Whether `target` is within the travel of every axis, saying which coordinate isn't.
    pub fn check_target(&self, target: &StagePos) -> Result<()> {
        for axis in Axis::ALL {
            let to = axis.of(target).um();
            let travel = self.axis(axis)?.travel;
            if !travel.holds(to) {
                return Err(Error::invalid_argument(format!(
                    "{axis:?} {to} µm is beyond the stage's travel, {} to {} µm",
                    travel.min, travel.max
                )));
            }
        }
        Ok(())
    }

    /// Whether `speed`, in µm/s, is one a move or a jog can take: finite and above zero.
    /// The stage keeps each axis to its own fastest.
    pub fn check_speed(speed: f64) -> Result<()> {
        if speed.is_finite() && speed > 0.0 {
            Ok(())
        } else {
            Err(Error::invalid_argument(format!(
                "a speed of {speed} µm/s; a speed is above zero"
            )))
        }
    }
}

/// What a stage is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StageState {
    /// Still, and ready.
    Idle,
    /// Moving to a target.
    Moving,
    /// Jogging along an axis.
    Jogging,
    /// Homing.
    Homing,
    /// Still, after a fault or a limit: it must be homed before it moves again.
    Alarm,
}

impl StageState {
    /// Whether it is still: idle, or in alarm.
    pub fn is_still(self) -> bool {
        matches!(self, StageState::Idle | StageState::Alarm)
    }
}

/// Where a stage is, what it is doing, and when it was read.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StageReading {
    /// Its position.
    pub position: StagePos,
    /// Its state.
    pub state: StageState,
    /// When it was read, by the stage's clock.
    pub at: Timestamp,
}

/// What a stage tells its subscribers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StageEvent {
    /// Its state changed, at this position.
    State {
        /// The new state.
        state: StageState,
        /// Where it was.
        position: StagePos,
    },
    /// It failed.
    Error {
        /// The kind of failure.
        kind: ErrorKind,
        /// What went wrong.
        message: String,
    },
}

/// A stage. Its methods block only as long as the stage takes to answer, except `wait`, and
/// it is shared by every thread that uses it.
pub trait Stage: Send + Sync {
    /// What it can do.
    fn capabilities(&self) -> StageCapabilities;

    /// Where it is, what it is doing, and when.
    fn reading(&self) -> Result<StageReading>;

    /// Starts a move to `target`, at `speed` µm/s or as fast as it goes. A target beyond its
    /// travel is refused, as is a move while it moves (`Busy`) or in alarm (`Hardware`).
    fn move_to(&self, target: StagePos, speed: Option<f64>) -> Result<()>;

    /// Starts a move by `delta` from where it is.
    fn move_by(&self, delta: StagePos, speed: Option<f64>) -> Result<()> {
        let here = self.reading()?.position;
        let target = StagePos::new(here.x + delta.x, here.y + delta.y, here.z + delta.z);
        self.move_to(target, speed)
    }

    /// Starts moving along `axis` at `speed` µm/s, below zero to go back, until
    /// `stop_jog`, `stop`, or the end of its travel.
    fn jog(&self, axis: Axis, speed: f64) -> Result<()>;

    /// Ends a jog. Nothing, if it isn't jogging.
    fn stop_jog(&self) -> Result<()>;

    /// Stops whatever it is doing. Once it returns, the stage is still.
    fn stop(&self) -> Result<()>;

    /// Starts homing. A stage that doesn't home refuses.
    fn home(&self) -> Result<()>;

    /// Waits until it is still, and gives its reading then: an error if it is in alarm, or
    /// if `timeout` passes first.
    fn wait(&self, timeout: Duration) -> Result<StageReading>;

    /// A channel of its events from now on.
    fn subscribe(&self) -> Receiver<StageEvent>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> StageCapabilities {
        let axis = |axis, max: f64, max_speed| AxisCapabilities {
            axis,
            travel: Span { min: 0.0, max },
            max_speed,
            acceleration: 10.0 * max_speed,
        };
        StageCapabilities {
            model: "a stage".into(),
            axes: vec![
                axis(Axis::X, 100.0, 10.0),
                axis(Axis::Y, 50.0, 10.0),
                axis(Axis::Z, 10.0, 1.0),
            ],
            resolution: Um(0.1),
            home: Some(StagePos::default()),
        }
    }

    fn at(x: f64, y: f64, z: f64) -> StagePos {
        StagePos::new(Um(x), Um(y), Um(z))
    }

    #[test]
    fn each_axis_reads_and_sets_its_own_coordinate() {
        let pos = at(1.0, 2.0, 3.0);
        let read: Vec<f64> = Axis::ALL.iter().map(|a| a.of(&pos).um()).collect();
        assert_eq!(read, [1.0, 2.0, 3.0]);
        assert_eq!(Axis::X.with(pos, Um(9.0)), at(9.0, 2.0, 3.0));
        assert_eq!(Axis::Y.with(pos, Um(9.0)), at(1.0, 9.0, 3.0));
        assert_eq!(Axis::Z.with(pos, Um(9.0)), at(1.0, 2.0, 9.0));
    }

    #[test]
    fn targets_beyond_the_travel_and_speeds_at_or_below_zero_are_refused() {
        let caps = caps();
        caps.check_target(&at(0.0, 0.0, 0.0)).unwrap();
        caps.check_target(&at(100.0, 50.0, 10.0)).unwrap();
        for (target, says) in [
            (
                at(100.5, 0.0, 0.0),
                "X 100.5 µm is beyond the stage's travel, 0 to 100 µm",
            ),
            (at(0.0, -1.0, 0.0), "Y -1 µm"),
            (at(0.0, 0.0, 10.5), "Z 10.5 µm"),
            (at(f64::NAN, 0.0, 0.0), "X NaN µm"),
        ] {
            let e = caps.check_target(&target).unwrap_err();
            assert_eq!(e.kind(), ErrorKind::InvalidArgument);
            assert!(e.message().contains(says), "{e}");
        }
        let without_z = StageCapabilities {
            axes: caps.axes[..2].to_vec(),
            ..caps.clone()
        };
        assert!(
            without_z
                .check_target(&at(0.0, 0.0, 0.0))
                .unwrap_err()
                .message()
                .contains("no Z axis")
        );
        StageCapabilities::check_speed(0.001).unwrap();
        for speed in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                StageCapabilities::check_speed(speed).unwrap_err().kind(),
                ErrorKind::InvalidArgument
            );
        }
    }

    #[test]
    fn only_idle_and_alarm_are_still() {
        use StageState::*;
        let still: Vec<bool> = [Idle, Moving, Jogging, Homing, Alarm]
            .iter()
            .map(|s| s.is_still())
            .collect();
        assert_eq!(still, [true, false, false, false, true]);
    }

    #[test]
    fn capabilities_readings_and_events_round_trip_through_json() {
        let caps = caps();
        let json = serde_json::to_string(&caps).unwrap();
        assert!(json.contains(r#""axis":"z""#) && json.contains(r#""resolution":0.1"#));
        assert_eq!(
            serde_json::from_str::<StageCapabilities>(&json).unwrap(),
            caps
        );
        let reading = StageReading {
            position: at(1.0, 2.0, 3.0),
            state: StageState::Jogging,
            at: Timestamp::from_duration(Duration::from_millis(5)),
        };
        let json = serde_json::to_string(&reading).unwrap();
        assert!(json.contains(r#""state":"jogging""#));
        assert_eq!(
            serde_json::from_str::<StageReading>(&json).unwrap(),
            reading
        );
        for event in [
            StageEvent::State {
                state: StageState::Alarm,
                position: at(1.0, 2.0, 3.0),
            },
            StageEvent::Error {
                kind: ErrorKind::Hardware,
                message: "stalled".into(),
            },
        ] {
            let json = serde_json::to_string(&event).unwrap();
            assert_eq!(serde_json::from_str::<StageEvent>(&json).unwrap(), event);
        }
        assert_eq!(
            serde_json::to_string(&StageEvent::Error {
                kind: ErrorKind::Hardware,
                message: "x".into()
            })
            .unwrap(),
            r#"{"event":"error","kind":"hardware","message":"x"}"#
        );
    }
}
