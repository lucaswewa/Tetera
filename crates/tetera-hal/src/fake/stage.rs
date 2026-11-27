//! The fake stage (P1.09b): three axes that move at once or in time, on a straight line at
//! each axis's speed, that jog, home, and fail when asked. It stands in for a stage in tests
//! and examples; the virtual stage (P1.10) adds acceleration and grblHAL's ways.

use super::lock;
use crate::camera::Span;
use crate::event::Events;
use crate::stage::{
    Axis, AxisCapabilities, Stage, StageCapabilities, StageEvent, StageReading, StageState,
};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use tetera_core::{Clock, Error, ErrorKind, Result, StagePos, Um};

/// How the fake stage's moves and homing take their time. Jogs always take theirs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// Done the moment they start.
    Instant,
    /// At each axis's speed, in real time.
    Timed,
}

/// A fault the fake stage shows when asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageFault {
    /// The next move, jog or homing is refused, as a hardware error.
    FailMove,
    /// The next move stalls halfway: an error, and the stage stops there in alarm.
    Stall,
    /// Into alarm now, as at a limit switch: stopped, and refusing to move until homed.
    Alarm,
    /// Gone: every call fails, as a hardware error, until it comes back.
    Disconnect,
    /// Back again.
    Reconnect,
}

/// A motion: from where, to where, when it set off, and how long it takes.
#[derive(Debug, Clone, Copy)]
struct Motion {
    from: StagePos,
    to: StagePos,
    started: Instant,
    takes: Duration,
}

impl Motion {
    fn still(at: StagePos) -> Self {
        Self {
            from: at,
            to: at,
            started: Instant::now(),
            takes: Duration::ZERO,
        }
    }

    /// Where it is at `now`: along the straight line, in proportion to the time.
    fn at(&self, now: Instant) -> StagePos {
        let elapsed = now.saturating_duration_since(self.started);
        if elapsed >= self.takes {
            return self.to;
        }
        let f = elapsed.as_secs_f64() / self.takes.as_secs_f64();
        let along = |a: Um, b: Um| a + (b - a) * f;
        StagePos::new(
            along(self.from.x, self.to.x),
            along(self.from.y, self.to.y),
            along(self.from.z, self.to.z),
        )
    }
}

struct State {
    motion: Motion,
    state: StageState,
    /// Changed when a motion is cut short, or the stage dropped, so its finisher ends.
    generation: u64,
    /// The motion under way ends in alarm: a stall.
    stalls: bool,
    fail_move: bool,
    stall_next: bool,
    disconnected: bool,
}

impl State {
    fn connected(&self) -> Result<()> {
        if self.disconnected {
            return Err(Error::hardware("the stage is disconnected"));
        }
        Ok(())
    }

    /// Ready to set off: still, and not in alarm, unless it is homing.
    fn ready(&mut self, homing: bool) -> Result<()> {
        self.connected()?;
        if !self.state.is_still() {
            return Err(Error::busy(format!("the stage is {:?}", self.state)));
        }
        if self.state == StageState::Alarm && !homing {
            return Err(Error::hardware("the stage is in alarm; home it"));
        }
        if std::mem::take(&mut self.fail_move) {
            return Err(Error::hardware("the stage refused to move"));
        }
        Ok(())
    }
}

/// What the stage shares with its motions' finishers.
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    events: Events<StageEvent>,
}

impl Shared {
    /// Ends the motion under way, where it was going, and says so.
    fn finish(&self, s: &mut State) {
        let at = s.motion.to;
        s.motion = Motion::still(at);
        s.state = if std::mem::take(&mut s.stalls) {
            self.events.emit(StageEvent::Error {
                kind: ErrorKind::Hardware,
                message: "the stage stalled".into(),
            });
            StageState::Alarm
        } else {
            StageState::Idle
        };
        self.events.emit(StageEvent::State {
            state: s.state,
            position: at,
        });
        self.changed.notify_all();
    }

    /// Stops the motion under way where it is now, in `state`, and says so.
    fn halt(&self, s: &mut State, state: StageState) {
        let at = s.motion.at(Instant::now());
        s.motion = Motion::still(at);
        s.generation += 1;
        s.stalls = false;
        s.state = state;
        self.events.emit(StageEvent::State {
            state,
            position: at,
        });
        self.changed.notify_all();
    }

    /// Ends the motion under way when its time is up, unless it is cut short first.
    fn finish_on_time(&self, generation: u64) {
        let mut s = lock(&self.state);
        while s.generation == generation {
            let left = s.motion.takes.saturating_sub(s.motion.started.elapsed());
            if left.is_zero() {
                self.finish(&mut s);
                return;
            }
            s = self
                .changed
                .wait_timeout(s, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

/// A fake stage: 100 × 75 mm of travel in x and y at 10 mm/s, 10 mm in z at 2 mm/s, homing to
/// the origin, where it starts, idle. Its readings are timestamped by its clock.
pub struct FakeStage {
    capabilities: StageCapabilities,
    pace: Pace,
    clock: Arc<dyn Clock>,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for FakeStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeStage")
            .field("model", &self.capabilities.model)
            .field("pace", &self.pace)
            .finish_non_exhaustive()
    }
}

impl FakeStage {
    /// A fake stage at the origin, moving at `pace`, its readings timed by `clock`.
    pub fn new(clock: Arc<dyn Clock>, pace: Pace) -> Self {
        let axis = |axis, max: f64, max_speed| AxisCapabilities {
            axis,
            travel: Span { min: 0.0, max },
            max_speed,
        };
        Self {
            capabilities: StageCapabilities {
                model: "Tetera fake stage".into(),
                axes: vec![
                    axis(Axis::X, 100_000.0, 10_000.0),
                    axis(Axis::Y, 75_000.0, 10_000.0),
                    axis(Axis::Z, 10_000.0, 2_000.0),
                ],
                resolution: Um(0.1),
                home: Some(StagePos::default()),
            },
            pace,
            clock,
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    motion: Motion::still(StagePos::default()),
                    state: StageState::Idle,
                    generation: 0,
                    stalls: false,
                    fail_move: false,
                    stall_next: false,
                    disconnected: false,
                }),
                changed: Condvar::new(),
                events: Events::default(),
            }),
        }
    }

    /// Shows `fault`.
    pub fn inject(&self, fault: StageFault) {
        let mut s = lock(&self.shared.state);
        match fault {
            StageFault::FailMove => s.fail_move = true,
            StageFault::Stall => s.stall_next = true,
            StageFault::Alarm => {
                self.shared.events.emit(StageEvent::Error {
                    kind: ErrorKind::Hardware,
                    message: "the stage hit a limit".into(),
                });
                self.shared.halt(&mut s, StageState::Alarm);
            }
            StageFault::Disconnect => {
                s.disconnected = true;
                self.shared.changed.notify_all();
            }
            StageFault::Reconnect => s.disconnected = false,
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.shared.state)
    }

    /// How long a straight move from `from` to `to` takes at `speed`, or as fast as it
    /// goes, each axis kept to its own fastest.
    fn duration(&self, from: &StagePos, to: &StagePos, speed: Option<f64>) -> Duration {
        if self.pace == Pace::Instant {
            return Duration::ZERO;
        }
        let mut secs: f64 = 0.0;
        let mut length_squared = 0.0;
        for axis in &self.capabilities.axes {
            let d = (axis.axis.of(to) - axis.axis.of(from)).um().abs();
            secs = secs.max(d / axis.max_speed);
            length_squared += d * d;
        }
        if let Some(speed) = speed {
            secs = secs.max(length_squared.sqrt() / speed);
        }
        Duration::from_secs_f64(secs)
    }

    /// Sets off for `to` in `state`, taking `takes`; a thread of its own ends it on time.
    fn set_off(
        &self,
        mut s: MutexGuard<'_, State>,
        to: StagePos,
        takes: Duration,
        state: StageState,
    ) -> Result<()> {
        let now = Instant::now();
        let from = s.motion.at(now);
        s.motion = Motion {
            from,
            to,
            started: now,
            takes,
        };
        s.state = state;
        self.shared.events.emit(StageEvent::State {
            state,
            position: from,
        });
        if takes.is_zero() {
            self.shared.finish(&mut s);
            return Ok(());
        }
        let generation = s.generation;
        drop(s);
        let shared = self.shared.clone();
        std::thread::Builder::new()
            .name("fake stage".into())
            .spawn(move || shared.finish_on_time(generation))
            .map(drop)
            .map_err(|e| Error::internal(format!("couldn't move the fake stage: {e}")))
    }

    fn read(&self, s: &State) -> StageReading {
        StageReading {
            position: s.motion.at(Instant::now()),
            state: s.state,
            at: self.clock.now(),
        }
    }
}

impl Stage for FakeStage {
    fn capabilities(&self) -> StageCapabilities {
        self.capabilities.clone()
    }

    fn reading(&self) -> Result<StageReading> {
        let s = self.state();
        s.connected()?;
        Ok(self.read(&s))
    }

    fn move_to(&self, target: StagePos, speed: Option<f64>) -> Result<()> {
        self.capabilities.check_target(&target)?;
        if let Some(speed) = speed {
            StageCapabilities::check_speed(speed)?;
        }
        let mut s = self.state();
        s.ready(false)?;
        let here = s.motion.at(Instant::now());
        let mut to = target;
        if std::mem::take(&mut s.stall_next) {
            to = StagePos::new(
                (here.x + to.x) / 2.0,
                (here.y + to.y) / 2.0,
                (here.z + to.z) / 2.0,
            );
            s.stalls = true;
        }
        let takes = self.duration(&here, &to, speed);
        self.set_off(s, to, takes, StageState::Moving)
    }

    fn jog(&self, axis: Axis, speed: f64) -> Result<()> {
        StageCapabilities::check_speed(speed.abs())?;
        let caps = self.capabilities.axis(axis)?;
        let mut s = self.state();
        s.ready(false)?;
        let here = s.motion.at(Instant::now());
        let end = if speed > 0.0 {
            caps.travel.max
        } else {
            caps.travel.min
        };
        let distance = (end - axis.of(&here).um()).abs();
        let takes = Duration::from_secs_f64(distance / speed.abs().min(caps.max_speed));
        self.set_off(s, axis.with(here, Um(end)), takes, StageState::Jogging)
    }

    fn stop_jog(&self) -> Result<()> {
        let mut s = self.state();
        s.connected()?;
        if s.state == StageState::Jogging {
            self.shared.halt(&mut s, StageState::Idle);
        }
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        let mut s = self.state();
        s.connected()?;
        if !s.state.is_still() {
            self.shared.halt(&mut s, StageState::Idle);
        }
        Ok(())
    }

    fn home(&self) -> Result<()> {
        let home = self
            .capabilities
            .home
            .ok_or_else(|| Error::invalid_argument("the stage doesn't home"))?;
        let mut s = self.state();
        s.ready(true)?;
        let here = s.motion.at(Instant::now());
        let takes = self.duration(&here, &home, None);
        self.set_off(s, home, takes, StageState::Homing)
    }

    fn wait(&self, timeout: Duration) -> Result<StageReading> {
        let deadline = Instant::now() + timeout;
        let mut s = self.state();
        loop {
            s.connected()?;
            if s.state.is_still() {
                break;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(Error::timeout(format!(
                    "the stage was still {:?} after {timeout:?}",
                    s.state
                )));
            }
            s = self
                .shared
                .changed
                .wait_timeout(s, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        if s.state == StageState::Alarm {
            return Err(Error::hardware("the stage is in alarm; home it"));
        }
        Ok(self.read(&s))
    }

    fn subscribe(&self) -> Receiver<StageEvent> {
        self.shared.events.subscribe()
    }
}

impl Drop for FakeStage {
    fn drop(&mut self) {
        // Any motion's finisher sees it is stale, and ends.
        let mut s = self.state();
        s.generation += 1;
        self.shared.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::stage as contract;
    use tetera_core::SystemClock;

    fn fake(pace: Pace) -> FakeStage {
        FakeStage::new(Arc::new(SystemClock::new()), pace)
    }

    fn at(x: f64, y: f64, z: f64) -> StagePos {
        StagePos::new(Um(x), Um(y), Um(z))
    }

    #[test]
    fn the_fake_stage_keeps_the_contract_at_either_pace() {
        contract::all(&|| Box::new(fake(Pace::Instant)));
        contract::all(&|| Box::new(fake(Pace::Timed)));
    }

    #[test]
    fn a_move_takes_as_long_as_its_slowest_axis_or_its_speed() {
        let stage = fake(Pace::Timed);
        let origin = StagePos::default();
        let secs = |to: StagePos, speed| stage.duration(&origin, &to, speed).as_secs_f64();
        assert_eq!(secs(at(1_000.0, 0.0, 0.0), None), 0.1, "x at 10 mm/s");
        assert_eq!(secs(at(0.0, 500.0, 0.0), None), 0.05, "y at 10 mm/s");
        assert_eq!(secs(at(0.0, 0.0, 400.0), None), 0.2, "z at 2 mm/s");
        assert_eq!(secs(at(1_000.0, 0.0, 400.0), None), 0.2, "the slower axis");
        assert_eq!(
            secs(at(3_000.0, 4_000.0, 0.0), Some(1_000.0)),
            5.0,
            "5 mm at 1 mm/s"
        );
        assert_eq!(
            secs(at(3_000.0, 4_000.0, 0.0), Some(1e9)),
            0.4,
            "no faster than y allows"
        );
        assert_eq!(
            stage.duration(&at(3_000.0, 0.0, 0.0), &at(1_000.0, 0.0, 0.0), None),
            Duration::from_millis(200),
            "back 2 mm"
        );
        let instant = fake(Pace::Instant);
        assert_eq!(
            instant.duration(&origin, &at(1_000.0, 0.0, 0.0), None),
            Duration::ZERO
        );
    }

    #[test]
    fn a_motion_is_on_its_line_in_proportion_to_the_time() {
        let started = Instant::now();
        let motion = Motion {
            from: at(1_000.0, 0.0, 100.0),
            to: at(3_000.0, 1_000.0, 500.0),
            started,
            takes: Duration::from_millis(200),
        };
        let after = |ms| motion.at(started + Duration::from_millis(ms));
        assert_eq!(after(0), motion.from);
        assert_eq!(after(50), at(1_500.0, 250.0, 200.0), "a quarter of the way");
        assert_eq!(after(200), motion.to);
        assert_eq!(after(900), motion.to, "and there it stays");
        assert_eq!(motion.at(started - Duration::from_millis(5)), motion.from);
    }

    #[test]
    fn a_timed_move_is_under_way_on_its_line_then_arrives() {
        let stage = fake(Pace::Timed);
        let events = stage.subscribe();
        let target = at(2_000.0, 1_000.0, 0.0);
        let start = Instant::now();
        stage.move_to(target, None).unwrap();
        let during = stage.reading().unwrap();
        assert_eq!(during.state, StageState::Moving);
        let (x, y) = (during.position.x.um(), during.position.y.um());
        assert!(
            x < 2_000.0 && (x - 2.0 * y).abs() < 1e-6,
            "on the line: {x}, {y}"
        );
        let end = stage.wait(Duration::from_secs(5)).unwrap();
        assert!(
            start.elapsed() >= Duration::from_millis(200),
            "2 mm at 10 mm/s"
        );
        assert_eq!((end.position, end.state), (target, StageState::Idle));
        assert_eq!(
            events.try_iter().collect::<Vec<_>>(),
            [
                StageEvent::State {
                    state: StageState::Moving,
                    position: StagePos::default()
                },
                StageEvent::State {
                    state: StageState::Idle,
                    position: target
                }
            ]
        );
    }

    #[test]
    fn a_jog_ends_at_the_end_of_the_travel() {
        let stage = fake(Pace::Instant);
        stage.move_to(at(99_990.0, 0.0, 9_995.0), None).unwrap();
        stage.jog(Axis::X, 1e9).unwrap();
        let end = stage.wait(Duration::from_secs(5)).unwrap();
        assert_eq!(end.position, at(100_000.0, 0.0, 9_995.0));
        let start = Instant::now();
        stage.jog(Axis::Z, 1_000.0).unwrap();
        let end = stage.wait(Duration::from_secs(5)).unwrap();
        assert!(
            start.elapsed() >= Duration::from_millis(5),
            "5 µm at 1 mm/s"
        );
        assert_eq!(end.position.z, Um(10_000.0));
        stage.jog(Axis::Y, -1_000.0).unwrap();
        assert_eq!(
            stage.wait(Duration::from_secs(5)).unwrap().position.y,
            Um(0.0),
            "at its start already"
        );
    }

    #[test]
    fn it_is_busy_while_it_moves_and_waits_no_longer_than_asked() {
        let stage = fake(Pace::Timed);
        stage.move_to(at(100_000.0, 0.0, 0.0), Some(100.0)).unwrap();
        for refused in [
            stage.move_to(StagePos::default(), None),
            stage.jog(Axis::Y, 100.0),
            stage.home(),
        ] {
            assert_eq!(refused.unwrap_err().kind(), ErrorKind::Busy);
        }
        assert_eq!(
            stage.wait(Duration::from_millis(20)).unwrap_err().kind(),
            ErrorKind::Timeout
        );
        stage.stop_jog().unwrap();
        assert_eq!(
            stage.reading().unwrap().state,
            StageState::Moving,
            "not a jog"
        );
        stage.stop().unwrap();
        assert_eq!(stage.reading().unwrap().state, StageState::Idle);
    }

    #[test]
    fn faults_come_when_asked() {
        let stage = fake(Pace::Instant);
        let events = stage.subscribe();
        stage.inject(StageFault::FailMove);
        assert_eq!(
            stage.move_to(at(10.0, 0.0, 0.0), None).unwrap_err().kind(),
            ErrorKind::Hardware
        );
        stage.move_to(at(10.0, 0.0, 100.0), None).unwrap();
        // A stall: halfway, in alarm, refusing all but homing.
        stage.inject(StageFault::Stall);
        stage.move_to(at(1_010.0, 200.0, 300.0), None).unwrap();
        assert_eq!(
            stage.wait(Duration::from_secs(1)).unwrap_err().kind(),
            ErrorKind::Hardware
        );
        let stalled = stage.reading().unwrap();
        assert_eq!(
            (stalled.position, stalled.state),
            (at(510.0, 100.0, 200.0), StageState::Alarm)
        );
        assert!(events.try_iter().any(|e| matches!(
            e,
            StageEvent::Error {
                kind: ErrorKind::Hardware,
                ..
            }
        )));
        for refused in [
            stage.move_to(StagePos::default(), None),
            stage.jog(Axis::X, 100.0),
        ] {
            assert_eq!(refused.unwrap_err().kind(), ErrorKind::Hardware);
        }
        stage.home().unwrap();
        let homed = stage.wait(Duration::from_secs(1)).unwrap();
        assert_eq!(
            (homed.position, homed.state),
            (StagePos::default(), StageState::Idle)
        );
        // An alarm: at once, where it is.
        stage.jog(Axis::X, 100.0).unwrap();
        stage.inject(StageFault::Alarm);
        assert_eq!(stage.reading().unwrap().state, StageState::Alarm);
        assert!(events.try_iter().any(|e| e
            == StageEvent::Error {
                kind: ErrorKind::Hardware,
                message: "the stage hit a limit".into()
            }));
        stage.home().unwrap();
        // Disconnected: every call fails, until it comes back.
        stage.inject(StageFault::Disconnect);
        for failed in [
            stage.reading().map(drop),
            stage.move_to(StagePos::default(), None),
            stage.jog(Axis::X, 100.0),
            stage.stop_jog(),
            stage.stop(),
            stage.home(),
            stage.wait(Duration::from_secs(1)).map(drop),
        ] {
            assert_eq!(failed.unwrap_err().kind(), ErrorKind::Hardware);
        }
        stage.inject(StageFault::Reconnect);
        assert!(stage.reading().is_ok());
        assert!(format!("{stage:?}").contains("Tetera fake stage"));
    }

    #[test]
    fn an_alarm_holds_until_homing() {
        // On a fresh stage, and during a jog: the jog's end doesn't clear it.
        let stage = fake(Pace::Timed);
        stage.inject(StageFault::Alarm);
        assert_eq!(stage.reading().unwrap().state, StageState::Alarm);
        stage.home().unwrap();
        stage.wait(Duration::from_secs(1)).unwrap();
        let events = stage.subscribe();
        stage.jog(Axis::X, 10_000.0).unwrap();
        stage.inject(StageFault::Alarm);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(stage.reading().unwrap().state, StageState::Alarm);
        let states: Vec<StageState> = events
            .try_iter()
            .filter_map(|e| match e {
                StageEvent::State { state, .. } => Some(state),
                StageEvent::Error { .. } => None,
            })
            .collect();
        assert_eq!(states, [StageState::Jogging, StageState::Alarm]);
    }

    #[test]
    fn a_disconnection_ends_a_wait() {
        let stage = Arc::new(fake(Pace::Timed));
        stage.move_to(at(100_000.0, 0.0, 0.0), Some(100.0)).unwrap();
        let other = stage.clone();
        let waiting = std::thread::spawn(move || other.wait(Duration::from_secs(10)));
        std::thread::sleep(Duration::from_millis(20));
        stage.inject(StageFault::Disconnect);
        let start = Instant::now();
        assert_eq!(
            waiting.join().unwrap().unwrap_err().kind(),
            ErrorKind::Hardware
        );
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn dropping_the_stage_ends_its_motions_finisher() {
        let stage = fake(Pace::Timed);
        stage.move_to(at(100_000.0, 0.0, 0.0), Some(100.0)).unwrap();
        let shared = stage.shared.clone();
        drop(stage);
        let start = Instant::now();
        while Arc::strong_count(&shared) > 1 {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "the finisher lives on"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
