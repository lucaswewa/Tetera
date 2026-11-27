//! The stage's contract (ADR-0013): what any stage must do, as checks that panic with what went
//! wrong. Each stage's tests run [`all`]: the fake now, the virtual stage (P1.10b) and the
//! grblHAL driver (M5) later. The checks move each axis a little from where it is.

use crate::stage::{Axis, Stage, StageCapabilities, StageEvent, StageReading, StageState};
use std::time::Duration;
use tetera_core::{ErrorKind, StagePos, Um};

/// How long the suite waits for a stage to be still.
const WAIT: Duration = Duration::from_secs(30);
/// How far the suite moves each axis, in µm: within any stage's travel.
const STEP: f64 = 100.0;
/// A speed any stage goes at, in µm/s, slow enough to be stopped partway.
const SLOW: f64 = 100.0;

/// Every check, each on a fresh stage from `make`.
pub fn all(make: &dyn Fn() -> Box<dyn Stage>) {
    capabilities_are_sound(&*make());
    a_move_ends_where_it_was_sent(&*make());
    a_move_by_is_relative(&*make());
    targets_beyond_the_travel_are_refused(&*make());
    a_stop_stills_a_move(&*make());
    a_jog_runs_until_stopped(&*make());
    homing_takes_it_home(&*make());
    stopping_a_still_stage_does_nothing(&*make());
}

/// The kind of error `result` is, if it is one.
fn kind<T>(result: tetera_core::Result<T>) -> Option<ErrorKind> {
    result.err().map(|e| e.kind())
}

/// Ready to move, homed first if it is in alarm: its reading then.
fn ready(stage: &dyn Stage) -> StageReading {
    let reading = stage.reading().expect("a reading");
    if reading.state != StageState::Alarm {
        return reading;
    }
    stage.home().expect("homing, out of alarm");
    stage.wait(WAIT).expect("homed")
}

/// `STEP` from `here` along each axis, back where its travel ends.
fn step_from(caps: &StageCapabilities, here: StagePos) -> StagePos {
    Axis::ALL.iter().fold(here, |pos, &axis| {
        let travel = caps.axis(axis).expect("an axis").travel;
        let from = axis.of(&here).um();
        let to = if from + STEP <= travel.max {
            from + STEP
        } else {
            from - STEP
        };
        axis.with(pos, Um(to))
    })
}

/// Whether `a` is within the stage's resolution of `b` along each axis.
fn close(caps: &StageCapabilities, a: &StagePos, b: &StagePos) -> bool {
    Axis::ALL
        .iter()
        .all(|axis| (axis.of(a) - axis.of(b)).abs() <= caps.resolution)
}

/// Still, and staying put: two readings 100 ms apart agree.
fn stays_put(stage: &dyn Stage, after: &str) -> StageReading {
    let first = stage.reading().expect("a reading");
    std::thread::sleep(Duration::from_millis(100));
    let then = stage.reading().expect("another");
    assert!(
        first.state.is_still() && then.state.is_still(),
        "{after}, still: {:?}",
        then.state
    );
    assert_eq!(first.position, then.position, "{after}, it stays put");
    assert!(then.at >= first.at, "readings in time order");
    then
}

/// It has x, y and z; home and where it is are within its travel; its capabilities survive
/// JSON.
pub fn capabilities_are_sound(stage: &dyn Stage) {
    let caps = stage.capabilities();
    for axis in Axis::ALL {
        assert!(
            caps.axis(axis).is_ok(),
            "a stage has x, y and z: no {axis:?}"
        );
    }
    if let Some(home) = caps.home {
        caps.check_target(&home).expect("home is within the travel");
    }
    let json = serde_json::to_string(&caps).expect("capabilities are JSON");
    assert_eq!(
        serde_json::from_str::<StageCapabilities>(&json).expect("and back"),
        caps
    );
    let reading = stage.reading().expect("a reading");
    caps.check_target(&reading.position)
        .expect("it is within its travel");
}

/// A move ends where it was sent, idle, and stays there; it says it set off and arrived.
pub fn a_move_ends_where_it_was_sent(stage: &dyn Stage) {
    let caps = stage.capabilities();
    let here = ready(stage).position;
    let events = stage.subscribe();
    let target = step_from(&caps, here);
    stage.move_to(target, None).expect("a move");
    let end = stage.wait(WAIT).expect("the move ends");
    assert!(
        close(&caps, &end.position, &target),
        "a move ends where it was sent, {target:?}, not {:?}",
        end.position
    );
    assert_eq!(end.state, StageState::Idle, "idle at its end");
    stays_put(stage, "after a move");
    let seen: Vec<StageEvent> = events.try_iter().collect();
    assert!(
        matches!(
            seen.first(),
            Some(StageEvent::State {
                state: StageState::Moving,
                ..
            })
        ),
        "it says it set off: {seen:?}"
    );
    assert!(
        matches!(seen.last(), Some(StageEvent::State { state: StageState::Idle, position }) if close(&caps, position, &target)),
        "it says it arrived: {seen:?}"
    );
}

/// A move by a displacement ends that far from where it began.
pub fn a_move_by_is_relative(stage: &dyn Stage) {
    let caps = stage.capabilities();
    let start = step_from(&caps, ready(stage).position);
    stage.move_to(start, None).expect("a move");
    let here = stage.wait(WAIT).expect("the move ends").position;
    let target = step_from(&caps, here);
    let delta = StagePos::new(target.x - here.x, target.y - here.y, target.z - here.z);
    stage.move_by(delta, None).expect("a move by");
    let end = stage.wait(WAIT).expect("it ends").position;
    assert!(
        close(&caps, &end, &target),
        "a move by {delta:?} from {here:?} ends at {target:?}, not {end:?}"
    );
}

/// A target beyond the travel, or a speed at or below zero, is refused, and nothing moves.
pub fn targets_beyond_the_travel_are_refused(stage: &dyn Stage) {
    let caps = stage.capabilities();
    let before = ready(stage);
    for axis in Axis::ALL {
        let travel = caps.axis(axis).expect("an axis").travel;
        for beyond in [travel.max.next_up(), travel.min.next_down(), f64::NAN] {
            assert_eq!(
                kind(stage.move_to(axis.with(before.position, Um(beyond)), None)),
                Some(ErrorKind::InvalidArgument),
                "{axis:?} to {beyond} µm, beyond its travel"
            );
        }
    }
    for speed in [0.0, -SLOW, f64::NAN] {
        assert_eq!(
            kind(stage.move_to(before.position, Some(speed))),
            Some(ErrorKind::InvalidArgument),
            "a move at {speed} µm/s"
        );
    }
    for speed in [0.0, f64::NAN] {
        assert_eq!(
            kind(stage.jog(Axis::X, speed)),
            Some(ErrorKind::InvalidArgument),
            "a jog at {speed} µm/s"
        );
    }
    let after = stays_put(stage, "after refusals");
    assert_eq!(
        after.position, before.position,
        "a refused move changes nothing"
    );
}

/// Once `stop` returns, the stage is still, wherever the move had got to.
pub fn a_stop_stills_a_move(stage: &dyn Stage) {
    let caps = stage.capabilities();
    let here = ready(stage).position;
    let far = caps.axis(Axis::X).expect("x").travel.max;
    stage
        .move_to(Axis::X.with(here, Um(far)), Some(SLOW))
        .expect("a slow move");
    std::thread::sleep(Duration::from_millis(50));
    stage.stop().expect("a stop");
    stays_put(stage, "after a stop");
    stage.wait(WAIT).expect("nothing to wait for");
}

/// A jog moves until it is stopped, refusing a move meanwhile.
pub fn a_jog_runs_until_stopped(stage: &dyn Stage) {
    let caps = stage.capabilities();
    let here = ready(stage).position;
    let speed = if here.x.um() < caps.axis(Axis::X).expect("x").travel.max {
        SLOW
    } else {
        -SLOW
    };
    stage.jog(Axis::X, speed).expect("a jog");
    std::thread::sleep(Duration::from_millis(100));
    let jogging = stage.reading().expect("a reading");
    assert_eq!(jogging.state, StageState::Jogging, "it jogs");
    assert_ne!(jogging.position.x, here.x, "a jog moves");
    assert_eq!(
        kind(stage.move_to(here, None)),
        Some(ErrorKind::Busy),
        "a move while jogging"
    );
    stage.stop_jog().expect("the jog stops");
    let after = stays_put(stage, "after a jog");
    assert_eq!(after.state, StageState::Idle, "idle after a jog");
}

/// Homing ends at home, idle; a stage that doesn't home refuses to.
pub fn homing_takes_it_home(stage: &dyn Stage) {
    let caps = stage.capabilities();
    let Some(home) = caps.home else {
        assert!(stage.home().is_err(), "a stage that doesn't home refuses");
        return;
    };
    stage
        .move_to(step_from(&caps, ready(stage).position), None)
        .expect("a move away");
    stage.wait(WAIT).expect("moved away");
    stage.home().expect("homing");
    let homed = stage.wait(WAIT).expect("homed");
    assert!(
        close(&caps, &homed.position, &home),
        "homing ends at home, {home:?}, not {:?}",
        homed.position
    );
    assert_eq!(homed.state, StageState::Idle, "idle once homed");
}

/// Stopping a still stage, or a jog that isn't there, does nothing and isn't an error.
pub fn stopping_a_still_stage_does_nothing(stage: &dyn Stage) {
    let before = ready(stage);
    stage.stop().expect("nothing to stop");
    stage.stop_jog().expect("no jog to stop");
    let after = stays_put(stage, "after stopping a still stage");
    assert_eq!(after.position, before.position, "nothing moved");
}

#[cfg(test)]
mod tests {
    //! The suite's own tests: a stage flawed on purpose, in one way for each check, fails it.
    use super::*;
    use crate::fake::stage::{FakeStage, Pace, StageFault};
    use std::sync::Arc;
    use std::sync::mpsc::{Receiver, channel};
    use tetera_core::{Error, Result, SystemClock};

    /// What is wrong with a flawed stage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Flaw {
        /// It has no z axis.
        NoZ,
        /// `wait` reports a move finished before it is.
        ReportsDoneEarly,
        /// It says nothing.
        Silent,
        /// It takes a move by a displacement as a move to it.
        MovesByAsTo,
        /// It moves to the end of its travel when sent beyond it.
        Clamps,
        /// `stop` doesn't.
        StopIgnored,
        /// `stop_jog` doesn't.
        StopJogIgnored,
        /// Homing does nothing.
        HomingIgnored,
        /// Stopping it when it is still is an error.
        FailsToStopWhenStill,
    }

    /// The timed fake stage, with a flaw.
    struct Flawed {
        stage: FakeStage,
        flaw: Flaw,
    }

    fn flawed(flaw: Flaw) -> Flawed {
        Flawed {
            stage: FakeStage::new(Arc::new(SystemClock::new()), Pace::Timed),
            flaw,
        }
    }

    impl Stage for Flawed {
        fn capabilities(&self) -> StageCapabilities {
            let mut caps = self.stage.capabilities();
            if self.flaw == Flaw::NoZ {
                caps.axes.retain(|a| a.axis != Axis::Z);
            }
            caps
        }

        fn reading(&self) -> Result<StageReading> {
            self.stage.reading()
        }

        fn move_to(&self, target: StagePos, speed: Option<f64>) -> Result<()> {
            if self.flaw != Flaw::Clamps {
                return self.stage.move_to(target, speed);
            }
            let caps = self.stage.capabilities();
            let clamped = Axis::ALL.iter().fold(target, |pos, &axis| {
                let travel = caps.axis(axis).expect("an axis").travel;
                let v = axis.of(&pos).um();
                let v = if v.is_nan() { travel.min } else { v };
                axis.with(pos, Um(v.clamp(travel.min, travel.max)))
            });
            self.stage.move_to(clamped, speed)
        }

        fn move_by(&self, delta: StagePos, speed: Option<f64>) -> Result<()> {
            if self.flaw == Flaw::MovesByAsTo {
                return self.stage.move_to(delta, speed);
            }
            self.stage.move_by(delta, speed)
        }

        fn jog(&self, axis: Axis, speed: f64) -> Result<()> {
            self.stage.jog(axis, speed)
        }

        fn stop_jog(&self) -> Result<()> {
            if self.flaw == Flaw::StopJogIgnored {
                return Ok(());
            }
            self.stage.stop_jog()
        }

        fn stop(&self) -> Result<()> {
            match self.flaw {
                Flaw::StopIgnored => Ok(()),
                Flaw::FailsToStopWhenStill if self.reading()?.state.is_still() => {
                    Err(Error::invalid_argument("the stage isn't moving"))
                }
                _ => self.stage.stop(),
            }
        }

        fn home(&self) -> Result<()> {
            if self.flaw == Flaw::HomingIgnored {
                return Ok(());
            }
            self.stage.home()
        }

        fn wait(&self, timeout: Duration) -> Result<StageReading> {
            if self.flaw != Flaw::ReportsDoneEarly {
                return self.stage.wait(timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
            Ok(StageReading {
                state: StageState::Idle,
                ..self.stage.reading()?
            })
        }

        fn subscribe(&self) -> Receiver<StageEvent> {
            if self.flaw == Flaw::Silent {
                return channel().1;
            }
            self.stage.subscribe()
        }
    }

    fn instant() -> FakeStage {
        FakeStage::new(Arc::new(SystemClock::new()), Pace::Instant)
    }

    #[test]
    fn the_suite_homes_a_stage_in_alarm_first() {
        all(&|| {
            let stage = instant();
            stage.inject(StageFault::Alarm);
            Box::new(stage)
        });
    }

    #[test]
    fn the_suite_works_from_the_far_end_of_the_travel() {
        all(&|| {
            let stage = instant();
            let caps = stage.capabilities();
            let far = Axis::ALL.iter().fold(StagePos::default(), |pos, &axis| {
                axis.with(pos, Um(caps.axis(axis).unwrap().travel.max))
            });
            stage.move_to(far, None).unwrap();
            Box::new(stage)
        });
    }

    #[test]
    #[should_panic(expected = "nothing to stop")]
    fn the_suite_runs_every_check_to_the_last() {
        all(&|| Box::new(flawed(Flaw::FailsToStopWhenStill)));
    }

    #[test]
    #[should_panic(expected = "a stage has x, y and z")]
    fn a_stage_without_z_fails() {
        capabilities_are_sound(&flawed(Flaw::NoZ));
    }

    #[test]
    #[should_panic(expected = "a move ends where it was sent")]
    fn a_stage_that_reports_a_move_done_early_fails() {
        a_move_ends_where_it_was_sent(&flawed(Flaw::ReportsDoneEarly));
    }

    #[test]
    #[should_panic(expected = "it says it set off")]
    fn a_silent_stage_fails() {
        a_move_ends_where_it_was_sent(&flawed(Flaw::Silent));
    }

    #[test]
    #[should_panic(expected = "a move by")]
    fn a_stage_that_moves_by_as_to_fails() {
        a_move_by_is_relative(&flawed(Flaw::MovesByAsTo));
    }

    #[test]
    #[should_panic(expected = "beyond its travel")]
    fn a_stage_that_clamps_its_targets_fails() {
        targets_beyond_the_travel_are_refused(&flawed(Flaw::Clamps));
    }

    #[test]
    #[should_panic(expected = "after a stop, still")]
    fn a_stage_that_doesnt_stop_fails() {
        a_stop_stills_a_move(&flawed(Flaw::StopIgnored));
    }

    #[test]
    #[should_panic(expected = "after a jog, still")]
    fn a_stage_that_doesnt_stop_jogging_fails() {
        a_jog_runs_until_stopped(&flawed(Flaw::StopJogIgnored));
    }

    #[test]
    #[should_panic(expected = "homing ends at home")]
    fn a_stage_that_doesnt_home_fails() {
        homing_takes_it_home(&flawed(Flaw::HomingIgnored));
    }

    #[test]
    #[should_panic(expected = "nothing to stop")]
    fn a_stage_that_cant_stop_when_still_fails() {
        stopping_a_still_stage_does_nothing(&flawed(Flaw::FailsToStopWhenStill));
    }
}
