//! The virtual stage (P1.10b): a stage that moves the way a grblHAL one does, on the simulated
//! clock. Its moves are trapezoids within each axis's limits (P1.10a), its states are
//! grblHAL's, its soft limits refuse targets beyond its travel, and homing takes time. It
//! reports its position at a set rate, each report arriving a little late, and keeps the
//! reports in a track.
//!
//! What it does from each command on is a plan: legs, each part of a motion profile in one
//! state, worked out from the time alone. So the same commands at the same times give the
//! same reports, bit for bit.

use crate::clock::SimClock;
use crate::motion::{AxisLimits, Profile};
use std::collections::VecDeque;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};
use tetera_core::{Clock, Error, ErrorKind, Result, StagePos, Timestamp, Um};
use tetera_hal::{
    Axis, AxisCapabilities, Events, Span, Stage, StageCapabilities, StageEvent, StageReading,
    StageState,
};

/// grblHAL's states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrblState {
    /// Still, and ready.
    Idle,
    /// Jogging.
    Jog,
    /// Moving to a target.
    Run,
    /// A move slowing to rest after a feed hold.
    Hold,
    /// Homing.
    Home,
    /// Stopped by a fault: it must be homed before it moves again.
    Alarm,
}

impl GrblState {
    /// The state as the HAL knows it: a hold is a move slowing to rest.
    pub fn hal(self) -> StageState {
        match self {
            GrblState::Idle => StageState::Idle,
            GrblState::Jog => StageState::Jogging,
            GrblState::Run | GrblState::Hold => StageState::Moving,
            GrblState::Home => StageState::Homing,
            GrblState::Alarm => StageState::Alarm,
        }
    }
}

/// A fault the virtual stage shows when asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualStageFault {
    /// Into alarm now, stopping where it is, as a reset during a move does.
    Alarm,
    /// A hard limit switch, now: it stops where it is, in alarm.
    LimitHit,
    /// The next move stalls halfway: it stops there, in alarm.
    Stall,
}

/// One axis of the virtual stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VirtualAxis {
    /// Its travel, in µm: its soft limits.
    pub travel: Span,
    /// Its fastest speed and its acceleration.
    pub limits: AxisLimits,
}

/// What the virtual stage is made of.
#[derive(Debug, Clone, PartialEq)]
pub struct VirtualStageConfig {
    /// Its x, y and z axes.
    pub axes: [VirtualAxis; 3],
    /// Its smallest step, in µm.
    pub resolution: Um,
    /// Where homing takes it.
    pub home: StagePos,
    /// How fast it seeks its home switches, in µm/s.
    pub homing_speed: f64,
    /// How long it takes to locate its switches, once there.
    pub homing_locate: Duration,
    /// How often it reports its position.
    pub report_period: Duration,
    /// How long a report takes to arrive.
    pub report_latency: Duration,
    /// How many reports its track keeps.
    pub track_length: usize,
    /// Where it is when it starts.
    pub start: StagePos,
    /// Whether it starts in alarm, as a grblHAL stage that must be homed does.
    pub starts_in_alarm: bool,
}

impl Default for VirtualStageConfig {
    /// 100 × 75 mm in x and y at 20 mm/s and 200 mm/s²; 10 mm in z, the focus, at 2 mm/s and
    /// 20 mm/s²; reports 50 times a second, 5 ms late; homed at the origin.
    fn default() -> Self {
        let axis = |max, max_speed, acceleration| VirtualAxis {
            travel: Span { min: 0.0, max },
            limits: AxisLimits {
                max_speed,
                acceleration,
            },
        };
        Self {
            axes: [
                axis(100_000.0, 20_000.0, 200_000.0),
                axis(75_000.0, 20_000.0, 200_000.0),
                axis(10_000.0, 2_000.0, 20_000.0),
            ],
            resolution: Um(0.3125),
            home: StagePos::default(),
            homing_speed: 5_000.0,
            homing_locate: Duration::from_millis(250),
            report_period: Duration::from_millis(20),
            report_latency: Duration::from_millis(5),
            track_length: 3_000,
            start: StagePos::default(),
            starts_in_alarm: false,
        }
    }
}

impl VirtualStageConfig {
    fn limits(&self) -> [AxisLimits; 3] {
        self.axes.map(|a| a.limits)
    }

    /// What a stage made this way can do.
    pub fn capabilities(&self) -> StageCapabilities {
        StageCapabilities {
            model: "Tetera virtual stage (grblHAL-like)".into(),
            axes: Axis::ALL
                .iter()
                .zip(&self.axes)
                .map(|(&axis, a)| AxisCapabilities {
                    axis,
                    travel: a.travel,
                    max_speed: a.limits.max_speed,
                })
                .collect(),
            resolution: self.resolution,
            home: Some(self.home),
        }
    }

    /// Whether it makes a stage, saying what is wrong.
    pub fn check(&self) -> Result<()> {
        let bad = |what: String| Err(Error::invalid_argument(what));
        for (axis, a) in Axis::ALL.iter().zip(&self.axes) {
            let Span { min, max } = a.travel;
            if !(min.is_finite() && max.is_finite() && min < max) {
                return bad(format!("{axis:?} travels from {min} to {max} µm"));
            }
        }
        // A move along the diagonal checks every axis's limits.
        Profile::new(
            StagePos::default(),
            StagePos::new(Um(1.0), Um(1.0), Um(1.0)),
            None,
            &self.limits(),
        )?;
        let caps = self.capabilities();
        caps.check_target(&self.home)
            .map_err(|e| e.context("its home"))?;
        caps.check_target(&self.start)
            .map_err(|e| e.context("where it starts"))?;
        StageCapabilities::check_speed(self.homing_speed).map_err(|e| e.context("homing"))?;
        if self.resolution.um() <= 0.0 || self.report_period.is_zero() || self.track_length == 0 {
            return bad(format!(
                "a resolution of {} µm, reports every {:?}, a track of {}: each is above zero",
                self.resolution.um(),
                self.report_period,
                self.track_length
            ));
        }
        Ok(())
    }
}

/// A position report, as the stage sends it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Report {
    /// When it was taken, by the stage's clock.
    pub at: Timestamp,
    /// Where the stage was.
    pub position: StagePos,
    /// What it was doing.
    pub state: GrblState,
}

/// A stretch of a plan: part of a profile, in one state.
#[derive(Debug, Clone)]
struct Leg {
    start: Timestamp,
    profile: Arc<Profile>,
    /// How far into the profile the leg starts.
    into: Duration,
    takes: Duration,
    state: GrblState,
}

impl Leg {
    fn end(&self) -> Timestamp {
        self.start.saturating_add(self.takes)
    }

    fn position(&self, t: Timestamp) -> StagePos {
        let into = self.into + t.saturating_since(self.start).min(self.takes);
        self.profile.position(into)
    }
}

/// What the stage does from a command on: its legs, one after another, and its state once
/// they end.
#[derive(Debug, Clone)]
struct Plan {
    rest: StagePos,
    legs: Vec<Leg>,
    after: GrblState,
}

impl Plan {
    fn still(at: StagePos, state: GrblState) -> Self {
        Self {
            rest: at,
            legs: Vec::new(),
            after: state,
        }
    }

    /// Legs one after another from `start`, each a profile in `state`.
    fn legs(start: Timestamp, profiles: Vec<(Profile, Duration)>, state: GrblState) -> Self {
        let rest = profiles
            .first()
            .map_or(StagePos::default(), |(p, _)| p.from());
        let mut at = start;
        let mut legs = Vec::new();
        for (profile, takes) in profiles {
            legs.push(Leg {
                start: at,
                profile: Arc::new(profile),
                into: Duration::ZERO,
                takes,
                state,
            });
            at = at.saturating_add(takes);
        }
        Self {
            rest,
            legs,
            after: GrblState::Idle,
        }
    }

    fn end(&self) -> Option<Timestamp> {
        self.legs.last().map(Leg::end)
    }

    fn position(&self, t: Timestamp) -> StagePos {
        for leg in &self.legs {
            if t < leg.end() {
                return leg.position(t);
            }
        }
        self.legs
            .last()
            .map_or(self.rest, |leg| leg.position(leg.end()))
    }

    fn state(&self, t: Timestamp) -> GrblState {
        self.legs
            .iter()
            .find(|leg| t < leg.end())
            .map_or(self.after, |leg| leg.state)
    }

    /// The plan held at `t`: a move or a jog slows to rest from there, as a feed hold or a
    /// jog cancel does, a move in the hold state.
    fn hold(&self, t: Timestamp) -> Plan {
        let Some(leg) = self.legs.iter().find(|leg| t < leg.end()) else {
            return self.clone();
        };
        let into = leg.into + t.saturating_since(leg.start);
        let held = leg.profile.hold(into);
        let takes = held.duration().saturating_sub(into);
        let state = match leg.state {
            GrblState::Run => GrblState::Hold,
            other => other,
        };
        Plan {
            rest: self.position(t),
            legs: vec![Leg {
                start: t,
                profile: Arc::new(held),
                into,
                takes,
                state,
            }],
            after: GrblState::Idle,
        }
    }
}

struct Inner {
    plan: Plan,
    /// Reports that have arrived, the latest last.
    track: VecDeque<Report>,
    /// Reports taken, still on their way.
    sent: VecDeque<Report>,
    /// The next report's number: report `k` is taken at `k` report periods.
    next: u64,
    /// When the last command came: a wait waits for a report from then on.
    commanded: Timestamp,
    stall_next: bool,
    /// What put it in alarm.
    alarm: String,
}

struct Shared {
    config: VirtualStageConfig,
    clock: Arc<SimClock>,
    inner: Mutex<Inner>,
    events: Events<StageEvent>,
}

/// How long `stop` and `stop_jog` wait, on the host's clock, for the stage to come to rest.
const REST_WAIT: Duration = Duration::from_secs(10);

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn period_ns(&self) -> u64 {
        u64::try_from(self.config.report_period.as_nanos()).unwrap_or(u64::MAX)
    }

    /// When report `k` is taken.
    fn taken(&self, k: u64) -> Timestamp {
        Timestamp(self.period_ns().saturating_mul(k))
    }

    /// When the first report taken at or after `t` arrives.
    fn arrives(&self, t: Timestamp) -> Timestamp {
        self.taken(t.0.div_ceil(self.period_ns()))
            .saturating_add(self.config.report_latency)
    }

    /// Takes every report that has arrived by `now`, and says what changed.
    fn catch_up(&self, s: &mut Inner, now: Timestamp) {
        // Every report due by now is taken from the plan as it is, before a command can
        // change it.
        loop {
            let at = self.taken(s.next);
            if at > now {
                break;
            }
            let report = Report {
                at,
                position: s.plan.position(at),
                state: s.plan.state(at),
            };
            s.sent.push_back(report);
            s.next += 1;
        }
        while let Some(report) = s.sent.front().copied() {
            if report.at.saturating_add(self.config.report_latency) > now {
                return;
            }
            s.sent.pop_front();
            self.arrive(s, report);
        }
    }

    /// A report arrives: it says what changed, and joins the track.
    fn arrive(&self, s: &mut Inner, report: Report) {
        let before = s.track.back().map_or(report.state, |r| r.state);
        if report.state != before {
            if report.state == GrblState::Alarm {
                self.events.emit(StageEvent::Error {
                    kind: ErrorKind::Hardware,
                    message: s.alarm.clone(),
                });
            }
            if report.state.hal() != before.hal() {
                self.events.emit(StageEvent::State {
                    state: report.state.hal(),
                    position: report.position,
                });
            }
        }
        s.track.push_back(report);
        if s.track.len() > self.config.track_length {
            s.track.pop_front();
        }
    }

    /// The stage now, after catching up.
    fn now(&self) -> (MutexGuard<'_, Inner>, Timestamp) {
        let mut s = self.lock();
        let now = self.clock.now();
        self.catch_up(&mut s, now);
        (s, now)
    }

    /// Waits until a report taken after the last command, and after its plan ends, shows the
    /// stage still: that report. `None` if `timeout` passes first.
    fn settle(&self, timeout: Duration) -> Option<Report> {
        let deadline = Instant::now() + timeout;
        loop {
            let due = {
                let (s, _) = self.now();
                let last = *s.track.back().expect("a report");
                let settled = s.plan.end().map_or(s.commanded, |end| end.max(s.commanded));
                if last.at >= settled && matches!(last.state, GrblState::Idle | GrblState::Alarm) {
                    return Some(last);
                }
                self.arrives(settled.max(last.at.saturating_add(Duration::from_nanos(1))))
            };
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || !self.clock.wait_until(due, left) {
                return None;
            }
        }
    }

    /// Takes reports as they arrive, and says what changed, until the stage is dropped.
    fn report(shared: &Weak<Shared>) {
        loop {
            let Some(stage) = shared.upgrade() else {
                return;
            };
            let due = {
                let s = stage.lock();
                let next = s.sent.front().map_or(stage.taken(s.next), |r| r.at);
                next.saturating_add(stage.config.report_latency)
            };
            let clock = stage.clock.clone();
            drop(stage);
            clock.wait_until(due, Duration::from_millis(50));
            if let Some(stage) = shared.upgrade() {
                drop(stage.now());
            }
        }
    }
}

/// The virtual stage. Its methods that wait, `wait`, `stop` and `stop_jog`, wait on its
/// clock: on a stepped clock, another thread must step it. `feed_hold` doesn't wait.
pub struct VirtualStage {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for VirtualStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualStage")
            .field("rate", &self.shared.clock.rate())
            .finish_non_exhaustive()
    }
}

fn index(axis: Axis) -> usize {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

impl VirtualStage {
    /// A virtual stage made as `config` says, on `clock`.
    pub fn new(clock: Arc<SimClock>, config: VirtualStageConfig) -> Result<Self> {
        config.check()?;
        let state = if config.starts_in_alarm {
            GrblState::Alarm
        } else {
            GrblState::Idle
        };
        let period = u64::try_from(config.report_period.as_nanos()).unwrap_or(u64::MAX);
        let first = clock.now().0 / period;
        let at = Timestamp(first.saturating_mul(period));
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                plan: Plan::still(config.start, state),
                track: VecDeque::from([Report {
                    at,
                    position: config.start,
                    state,
                }]),
                sent: VecDeque::new(),
                next: first + 1,
                commanded: at,
                stall_next: false,
                alarm: "the stage starts in alarm, to be homed".into(),
            }),
            config,
            clock,
            events: Events::default(),
        });
        let weak = Arc::downgrade(&shared);
        std::thread::Builder::new()
            .name("virtual stage".into())
            .spawn(move || Shared::report(&weak))
            .map_err(|e| Error::internal(format!("couldn't start the virtual stage: {e}")))?;
        Ok(Self { shared })
    }

    /// Its clock.
    pub fn clock(&self) -> &Arc<SimClock> {
        &self.shared.clock
    }

    /// How it is made.
    pub fn config(&self) -> &VirtualStageConfig {
        &self.shared.config
    }

    /// Its state now, as the stage itself knows it, ahead of its reports.
    pub fn grbl_state(&self) -> GrblState {
        let (s, now) = self.shared.now();
        s.plan.state(now)
    }

    /// Where it is now, as the stage itself knows it, ahead of its reports: where a camera
    /// over it would look.
    pub fn position(&self) -> StagePos {
        let (s, now) = self.shared.now();
        s.plan.position(now)
    }

    /// The reports that have arrived, the oldest first, as many as the track keeps.
    pub fn track(&self) -> Vec<Report> {
        let (s, _) = self.shared.now();
        s.track.iter().copied().collect()
    }

    /// Shows `fault`.
    pub fn inject(&self, fault: VirtualStageFault) {
        let (mut s, now) = self.shared.now();
        let alarm = match fault {
            VirtualStageFault::Stall => {
                s.stall_next = true;
                return;
            }
            VirtualStageFault::Alarm => "an alarm stopped the stage",
            VirtualStageFault::LimitHit => "a hard limit stopped the stage",
        };
        s.plan = Plan::still(s.plan.position(now), GrblState::Alarm);
        s.alarm = alarm.into();
        s.commanded = now;
    }

    /// A feed hold, or a jog cancel: a move or a jog slows to rest from where it is. It
    /// doesn't wait; homing can't be held.
    pub fn feed_hold(&self) {
        let (mut s, now) = self.shared.now();
        if matches!(s.plan.state(now), GrblState::Run | GrblState::Jog) {
            s.plan = s.plan.hold(now);
            s.commanded = now;
        }
    }

    /// Ready to set off now: idle, or in alarm only for homing.
    fn ready(s: &Inner, now: Timestamp, homing: bool) -> Result<()> {
        match s.plan.state(now) {
            GrblState::Idle => Ok(()),
            GrblState::Alarm if homing => Ok(()),
            GrblState::Alarm => Err(Error::hardware(format!(
                "the stage is in alarm ({}); home it",
                s.alarm
            ))),
            other => Err(Error::busy(format!("the stage is busy: {other:?}"))),
        }
    }

    /// Waits for rest after a hold or a stop.
    fn rest(&self) -> Result<()> {
        self.shared.settle(REST_WAIT).map(drop).ok_or_else(|| {
            Error::timeout(format!(
                "the stage wasn't at rest after {REST_WAIT:?}; a stepped clock must be stepped"
            ))
        })
    }
}

impl Stage for VirtualStage {
    fn capabilities(&self) -> StageCapabilities {
        self.shared.config.capabilities()
    }

    fn reading(&self) -> Result<StageReading> {
        let (s, _) = self.shared.now();
        let last = s.track.back().expect("a report");
        Ok(StageReading {
            position: last.position,
            state: last.state.hal(),
            at: last.at,
        })
    }

    fn move_to(&self, target: StagePos, speed: Option<f64>) -> Result<()> {
        let config = &self.shared.config;
        config.capabilities().check_target(&target)?;
        if let Some(speed) = speed {
            StageCapabilities::check_speed(speed)?;
        }
        let (mut s, now) = self.shared.now();
        Self::ready(&s, now, false)?;
        let profile = Profile::new(s.plan.position(now), target, speed, &config.limits())?;
        let mut takes = profile.duration();
        let mut plan_after = GrblState::Idle;
        if std::mem::take(&mut s.stall_next) {
            takes /= 2;
            plan_after = GrblState::Alarm;
            s.alarm = "the stage stalled".into();
        }
        let mut plan = Plan::legs(now, vec![(profile, takes)], GrblState::Run);
        plan.after = plan_after;
        s.plan = plan;
        s.commanded = now;
        Ok(())
    }

    fn move_by(&self, delta: StagePos, speed: Option<f64>) -> Result<()> {
        let here = self.position();
        let target = StagePos::new(here.x + delta.x, here.y + delta.y, here.z + delta.z);
        self.move_to(target, speed)
    }

    fn jog(&self, axis: Axis, speed: f64) -> Result<()> {
        StageCapabilities::check_speed(speed.abs())?;
        let config = &self.shared.config;
        let (mut s, now) = self.shared.now();
        Self::ready(&s, now, false)?;
        let here = s.plan.position(now);
        let travel = config.axes[index(axis)].travel;
        let end = if speed > 0.0 { travel.max } else { travel.min };
        let profile = Profile::new(
            here,
            axis.with(here, Um(end)),
            Some(speed.abs()),
            &config.limits(),
        )?;
        let takes = profile.duration();
        s.plan = Plan::legs(now, vec![(profile, takes)], GrblState::Jog);
        s.commanded = now;
        Ok(())
    }

    fn stop_jog(&self) -> Result<()> {
        {
            let (mut s, now) = self.shared.now();
            if s.plan.state(now) != GrblState::Jog {
                return Ok(());
            }
            s.plan = s.plan.hold(now);
            s.commanded = now;
        }
        self.rest()
    }

    fn stop(&self) -> Result<()> {
        {
            let (mut s, now) = self.shared.now();
            match s.plan.state(now) {
                GrblState::Idle | GrblState::Alarm => return Ok(()),
                GrblState::Home => {
                    s.plan = Plan::still(s.plan.position(now), GrblState::Alarm);
                    s.alarm = "homing was stopped".into();
                }
                GrblState::Run | GrblState::Hold | GrblState::Jog => s.plan = s.plan.hold(now),
            }
            s.commanded = now;
        }
        self.rest()
    }

    fn home(&self) -> Result<()> {
        let config = &self.shared.config;
        let (mut s, now) = self.shared.now();
        Self::ready(&s, now, true)?;
        // z clears first, then x and y; then it locates the switches.
        let here = s.plan.position(now);
        let speed = Some(config.homing_speed);
        let raised = StagePos {
            z: config.home.z,
            ..here
        };
        let limits = config.limits();
        let z = Profile::new(here, raised, speed, &limits)?;
        let xy = Profile::new(raised, config.home, speed, &limits)?;
        let locate = Profile::new(config.home, config.home, None, &limits)?;
        let (tz, txy) = (z.duration(), xy.duration());
        s.plan = Plan::legs(
            now,
            vec![(z, tz), (xy, txy), (locate, config.homing_locate)],
            GrblState::Home,
        );
        s.commanded = now;
        Ok(())
    }

    fn wait(&self, timeout: Duration) -> Result<StageReading> {
        let last = self.shared.settle(timeout).ok_or_else(|| {
            Error::timeout(format!("the stage was still moving after {timeout:?}"))
        })?;
        if last.state == GrblState::Alarm {
            let s = self.shared.lock();
            return Err(Error::hardware(format!(
                "the stage is in alarm ({}); home it",
                s.alarm
            )));
        }
        Ok(StageReading {
            position: last.position,
            state: last.state.hal(),
            at: last.at,
        })
    }

    fn subscribe(&self) -> Receiver<StageEvent> {
        self.shared.events.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Rate;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tetera_hal::contract::stage as contract;

    fn at(x: f64, y: f64, z: f64) -> StagePos {
        StagePos::new(Um(x), Um(y), Um(z))
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn make(rate: Rate, config: VirtualStageConfig) -> VirtualStage {
        VirtualStage::new(Arc::new(SimClock::new(rate).unwrap()), config).unwrap()
    }

    fn stepped() -> VirtualStage {
        make(Rate::Stepped, VirtualStageConfig::default())
    }

    /// Steps `stage`'s clock `step` at a time, `n` times.
    fn step(stage: &VirtualStage, step: Duration, n: u32) {
        for _ in 0..n {
            stage.clock().step(step).unwrap();
        }
    }

    /// Steps the clock until the stage itself is in `state`, 1 ms at a time: how long that took.
    fn until(stage: &VirtualStage, state: GrblState) -> Duration {
        let start = stage.clock().now();
        while stage.grbl_state() != state {
            assert!(
                stage.clock().now().saturating_since(start) < Duration::from_secs(60),
                "never {state:?}"
            );
            stage.clock().step(ms(1)).unwrap();
        }
        stage.clock().now().saturating_since(start)
    }

    /// Runs `f`, which waits on the stage's stepped clock, while another thread steps it.
    fn stepping<T: Send>(stage: &VirtualStage, f: impl FnOnce() -> T + Send) -> T {
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| {
                let out = f();
                done.store(true, Ordering::Release);
                out
            });
            while !done.load(Ordering::Acquire) {
                stage.clock().step(ms(1)).unwrap();
                std::thread::sleep(Duration::from_micros(100));
            }
            worker.join().unwrap()
        })
    }

    /// The trapezoid's duration for a straight move, worked out here on its own.
    fn trapezoid(config: &VirtualStageConfig, from: StagePos, to: StagePos, speed: f64) -> f64 {
        let d = [
            to.x.um() - from.x.um(),
            to.y.um() - from.y.um(),
            to.z.um() - from.z.um(),
        ];
        let length = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        let (mut v, mut a) = (speed, f64::INFINITY);
        for (share, axis) in d.iter().zip(&config.axes) {
            let share = share.abs() / length;
            if share > 0.0 {
                v = v.min(axis.limits.max_speed / share);
                a = a.min(axis.limits.acceleration / share);
            }
        }
        if v * v / a >= length {
            2.0 * (length / a).sqrt()
        } else {
            length / v + v / a
        }
    }

    #[test]
    fn the_virtual_stage_keeps_the_contract_at_1x_at_10x_and_from_an_alarm() {
        contract::all(&|| Box::new(make(Rate::RealTime, VirtualStageConfig::default())));
        contract::all(&|| Box::new(make(Rate::Faster(10.0), VirtualStageConfig::default())));
        contract::all(&|| {
            Box::new(make(
                Rate::Faster(10.0),
                VirtualStageConfig {
                    start: at(20_000.0, 10_000.0, 5_000.0),
                    starts_in_alarm: true,
                    ..VirtualStageConfig::default()
                },
            ))
        });
    }

    #[test]
    fn in_stepped_time_a_moves_duration_is_the_trapezoids_within_1_ms() {
        let stage = stepped();
        let config = stage.config().clone();
        for (target, speed) in [
            (at(5_000.0, 2_000.0, 500.0), None),
            (at(5_100.0, 2_000.0, 500.0), None),
            (at(30_000.0, 50_000.0, 0.0), Some(7_000.0)),
        ] {
            let from = stage.position();
            stage.move_to(target, speed).unwrap();
            assert_eq!(stage.grbl_state(), GrblState::Run);
            let took = until(&stage, GrblState::Idle).as_secs_f64();
            let expected = trapezoid(&config, from, target, speed.unwrap_or(f64::INFINITY));
            assert!(
                took >= expected && took < expected + 1e-3,
                "{took} s against {expected} s"
            );
            assert_eq!(stage.position(), target);
        }
    }

    /// A run of commands at set times on a stepped clock: its track.
    fn a_run() -> Vec<Report> {
        let stage = stepped();
        stage.move_to(at(4_000.0, 3_000.0, 700.0), None).unwrap();
        step(&stage, ms(7), 80);
        stage.jog(Axis::Y, 3_000.0).unwrap();
        step(&stage, ms(7), 20);
        stage.feed_hold();
        step(&stage, ms(7), 10);
        stage.move_to(at(10_000.0, 1_000.0, 2_000.0), None).unwrap();
        step(&stage, ms(13), 15);
        stage.feed_hold();
        step(&stage, ms(5), 40);
        stage.home().unwrap();
        step(&stage, ms(11), 300);
        stage.inject(VirtualStageFault::LimitHit);
        step(&stage, ms(10), 5);
        stage.home().unwrap();
        step(&stage, ms(11), 100);
        stage.track()
    }

    #[test]
    fn the_same_steps_give_the_same_track_bit_for_bit() {
        let bits = |track: &[Report]| -> Vec<(u64, [u64; 3], GrblState)> {
            track
                .iter()
                .map(|r| {
                    let p = r.position;
                    (
                        r.at.0,
                        [p.x.0.to_bits(), p.y.0.to_bits(), p.z.0.to_bits()],
                        r.state,
                    )
                })
                .collect()
        };
        let (a, b) = (a_run(), a_run());
        assert!(a.len() > 250, "{} reports", a.len());
        assert_eq!(bits(&a), bits(&b));
        let states: std::collections::HashSet<GrblState> = a.iter().map(|r| r.state).collect();
        assert_eq!(states.len(), 6, "every state: {states:?}");
    }

    #[test]
    fn reports_come_at_their_rate_and_arrive_late() {
        let stage = stepped();
        assert_eq!(stage.track().len(), 1, "where it starts");
        stage.move_to(at(2_000.0, 0.0, 0.0), None).unwrap();
        let profile = Profile::new(
            StagePos::default(),
            at(2_000.0, 0.0, 0.0),
            None,
            &stage.config().limits(),
        )
        .unwrap();
        step(&stage, ms(1), 44);
        let reading = stage.reading().unwrap();
        assert_eq!(
            reading.at.as_duration(),
            ms(20),
            "taken at 20 ms, the 40 ms one on its way"
        );
        assert_eq!(reading.position, profile.position(ms(20)));
        step(&stage, ms(1), 1);
        assert_eq!(stage.reading().unwrap().at.as_duration(), ms(40));
        assert!(
            stage.position().x > reading.position.x,
            "the stage is ahead of its reports"
        );
        let track = stage.track();
        let times: Vec<Duration> = track.iter().map(|r| r.at.as_duration()).collect();
        assert_eq!(times, [ms(0), ms(20), ms(40)]);
        assert_eq!(track[0].state, GrblState::Idle, "taken before the move");
    }

    #[test]
    fn a_report_taken_before_a_command_shows_the_stage_before_it() {
        // The command at 22 ms comes after the report taken at 20 ms, which arrives at 25.
        let stage = stepped();
        step(&stage, ms(1), 22);
        stage.move_to(at(1_000.0, 0.0, 0.0), None).unwrap();
        step(&stage, ms(1), 3);
        let report = stage.track()[1];
        assert_eq!(report.at.as_duration(), ms(20));
        assert_eq!(
            (report.state, report.position),
            (GrblState::Idle, StagePos::default())
        );
    }

    #[test]
    fn a_feed_hold_slows_a_move_to_rest_in_the_hold_state() {
        let stage = stepped();
        stage.move_to(at(50_000.0, 0.0, 0.0), None).unwrap();
        step(&stage, ms(1), 500);
        let (x, v) = (stage.position().x.um(), 20_000.0);
        stage.feed_hold();
        assert_eq!(stage.grbl_state(), GrblState::Hold);
        let took = until(&stage, GrblState::Idle);
        // From 20 mm/s at 200 mm/s²: 0.1 s and 1 mm.
        assert!(took >= ms(100) && took < ms(101), "{took:?}");
        let rest = stage.position().x.um();
        assert!(
            (rest - (x + v * v / 400_000.0)).abs() < 1.0,
            "{rest} from {x}"
        );
        step(&stage, ms(10), 3);
        let states: Vec<GrblState> = stage.track().iter().map(|r| r.state).collect();
        let mut seen = states.clone();
        seen.dedup();
        assert_eq!(
            seen,
            [
                GrblState::Idle,
                GrblState::Run,
                GrblState::Hold,
                GrblState::Idle
            ]
        );
        assert!(stage.position().x.um() < 50_000.0);
    }

    #[test]
    fn a_jog_runs_to_the_end_of_its_travel_or_until_cancelled() {
        let stage = stepped();
        stage.jog(Axis::X, 5_000.0).unwrap();
        assert_eq!(stage.grbl_state(), GrblState::Jog);
        step(&stage, ms(1), 200);
        let x = stage.position().x.um();
        stage.feed_hold();
        assert_eq!(stage.grbl_state(), GrblState::Jog, "slowing, still a jog");
        let took = until(&stage, GrblState::Idle);
        assert!(took < ms(30), "from 5 mm/s at 200 mm/s²: {took:?}");
        let rest = stage.position().x.um();
        assert!(rest > x && rest < x + 100.0, "{rest} from {x}");
        // To the end of its travel, and no further.
        let near_the_end = make(
            Rate::Stepped,
            VirtualStageConfig {
                start: at(99_900.0, 0.0, 0.0),
                ..VirtualStageConfig::default()
            },
        );
        near_the_end.jog(Axis::X, 1e9).unwrap();
        until(&near_the_end, GrblState::Idle);
        assert_eq!(near_the_end.position(), at(100_000.0, 0.0, 0.0));
        near_the_end.jog(Axis::Z, -1_000.0).unwrap();
        assert_eq!(
            near_the_end.grbl_state(),
            GrblState::Idle,
            "at the end already"
        );
    }

    #[test]
    fn homing_clears_z_first_then_x_and_y_then_locates() {
        let stage = make(
            Rate::Stepped,
            VirtualStageConfig {
                start: at(20_000.0, 10_000.0, 5_000.0),
                starts_in_alarm: true,
                ..VirtualStageConfig::default()
            },
        );
        let config = stage.config().clone();
        assert_eq!(stage.grbl_state(), GrblState::Alarm);
        for refused in [
            stage.move_to(StagePos::default(), None),
            stage.jog(Axis::X, 100.0),
        ] {
            assert_eq!(refused.unwrap_err().kind(), ErrorKind::Hardware);
        }
        stage.home().unwrap();
        let raised = at(20_000.0, 10_000.0, 0.0);
        let tz = trapezoid(&config, at(20_000.0, 10_000.0, 5_000.0), raised, 5_000.0);
        let txy = trapezoid(&config, raised, StagePos::default(), 5_000.0);
        step(&stage, Duration::from_secs_f64(tz / 2.0), 1);
        let p = stage.position();
        assert_eq!(
            (p.x, p.y, stage.grbl_state()),
            (Um(20_000.0), Um(10_000.0), GrblState::Home)
        );
        assert!(p.z.um() < 5_000.0, "z first");
        let took = until(&stage, GrblState::Idle).as_secs_f64() + tz / 2.0;
        let expected = tz + txy + 0.25;
        assert!(
            (took - expected).abs() < 2e-3,
            "{took} s against {expected} s"
        );
        assert_eq!(stage.position(), StagePos::default());
    }

    #[test]
    fn it_is_busy_while_it_moves_and_a_stop_while_homing_is_an_alarm() {
        let stage = stepped();
        stage.move_to(at(50_000.0, 0.0, 0.0), None).unwrap();
        for refused in [
            stage.move_to(StagePos::default(), None),
            stage.jog(Axis::Y, 100.0),
            stage.home(),
        ] {
            assert_eq!(refused.unwrap_err().kind(), ErrorKind::Busy);
        }
        stepping(&stage, || stage.stop()).unwrap();
        assert_eq!(stage.grbl_state(), GrblState::Idle);
        stage.home().unwrap();
        step(&stage, ms(10), 5);
        assert_eq!(stage.grbl_state(), GrblState::Home);
        stage.feed_hold();
        assert_eq!(stage.grbl_state(), GrblState::Home, "homing can't be held");
        stepping(&stage, || stage.stop()).unwrap();
        assert_eq!(stage.grbl_state(), GrblState::Alarm);
        let e = stage.wait(Duration::from_secs(1)).unwrap_err();
        assert!(e.message().contains("homing was stopped"), "{e}");
        // Stopping a still stage, and a jog that isn't there, does nothing.
        stage.stop().unwrap();
        stage.stop_jog().unwrap();
        stage.feed_hold();
        assert_eq!(stage.grbl_state(), GrblState::Alarm);
    }

    #[test]
    fn faults_come_when_asked() {
        let stage = stepped();
        let events = stage.subscribe();
        for (fault, says) in [
            (VirtualStageFault::Alarm, "an alarm stopped the stage"),
            (
                VirtualStageFault::LimitHit,
                "a hard limit stopped the stage",
            ),
        ] {
            stage.jog(Axis::X, 1_000.0).unwrap();
            step(&stage, ms(50), 1);
            let x = stage.position().x;
            stage.inject(fault);
            assert_eq!(
                (stage.grbl_state(), stage.position().x),
                (GrblState::Alarm, x),
                "where it was"
            );
            step(&stage, ms(50), 1);
            stage.reading().unwrap();
            let seen: Vec<StageEvent> = events.try_iter().collect();
            assert!(
                seen.contains(&StageEvent::Error {
                    kind: ErrorKind::Hardware,
                    message: says.into()
                }),
                "{seen:?}"
            );
            assert!(matches!(
                seen.last(),
                Some(StageEvent::State {
                    state: StageState::Alarm,
                    ..
                })
            ));
            stage.home().unwrap();
            until(&stage, GrblState::Idle);
        }
        // A stall: halfway, in alarm.
        stage.inject(VirtualStageFault::Stall);
        stage.move_to(at(10_000.0, 0.0, 0.0), None).unwrap();
        let took = until(&stage, GrblState::Alarm).as_secs_f64();
        let whole = trapezoid(
            stage.config(),
            StagePos::default(),
            at(10_000.0, 0.0, 0.0),
            f64::INFINITY,
        );
        assert!((took - whole / 2.0).abs() < 2e-3, "{took} s of {whole}");
        assert!((stage.position().x.um() - 5_000.0).abs() < 1e-6);
        step(&stage, ms(50), 1);
        let e = stage.wait(Duration::from_secs(1)).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Hardware);
        assert!(e.message().contains("the stage stalled"), "{e}");
        stage.home().unwrap();
        until(&stage, GrblState::Idle);
        stage.move_to(at(10_000.0, 0.0, 0.0), None).unwrap();
        until(&stage, GrblState::Idle);
        assert_eq!(
            stage.position(),
            at(10_000.0, 0.0, 0.0),
            "the stall was once"
        );
    }

    #[test]
    fn its_reports_say_what_changed_without_being_asked() {
        let stage = stepped();
        let events = stage.subscribe();
        stage.move_to(at(1_000.0, 0.0, 0.0), None).unwrap();
        step(&stage, ms(5), 60);
        let first = events.recv_timeout(Duration::from_secs(5)).unwrap();
        let second = events.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            matches!(
                first,
                StageEvent::State {
                    state: StageState::Moving,
                    ..
                }
            ),
            "{first:?}"
        );
        assert_eq!(
            second,
            StageEvent::State {
                state: StageState::Idle,
                position: at(1_000.0, 0.0, 0.0)
            }
        );
    }

    #[test]
    fn the_track_keeps_as_many_reports_as_it_is_set_to() {
        let stage = make(
            Rate::Stepped,
            VirtualStageConfig {
                track_length: 5,
                ..VirtualStageConfig::default()
            },
        );
        step(&stage, ms(1), 205);
        let times: Vec<u64> = stage.track().iter().map(|r| r.at.0 / 1_000_000).collect();
        assert_eq!(times, [120, 140, 160, 180, 200]);
    }

    #[test]
    fn a_move_by_is_from_where_the_stage_itself_is() {
        let stage = stepped();
        stage.move_to(at(1_000.0, 0.0, 0.0), None).unwrap();
        until(&stage, GrblState::Idle);
        // Its reports lag, but a move by starts from where it is.
        stage.move_by(at(500.0, 200.0, 100.0), None).unwrap();
        until(&stage, GrblState::Idle);
        assert_eq!(stage.position(), at(1_500.0, 200.0, 100.0));
        assert!(format!("{stage:?}").contains("Stepped"));
    }

    #[test]
    fn each_thing_wrong_with_a_configuration_is_said() {
        let good = VirtualStageConfig::default();
        good.check().unwrap();
        let mut bad = Vec::new();
        let mut travel = good.clone();
        travel.axes[1].travel = Span { min: 5.0, max: 5.0 };
        bad.push((travel, "Y travels from 5 to 5 µm"));
        let mut nan = good.clone();
        nan.axes[0].travel.max = f64::NAN;
        bad.push((nan, "X travels from 0 to NaN µm"));
        let mut slow = good.clone();
        slow.axes[2].limits.acceleration = 0.0;
        bad.push((slow, "axis 2"));
        bad.push((
            VirtualStageConfig {
                home: at(-1.0, 0.0, 0.0),
                ..good.clone()
            },
            "its home",
        ));
        bad.push((
            VirtualStageConfig {
                start: at(0.0, 80_000.0, 0.0),
                ..good.clone()
            },
            "where it starts",
        ));
        bad.push((
            VirtualStageConfig {
                homing_speed: 0.0,
                ..good.clone()
            },
            "homing",
        ));
        for zero in [
            VirtualStageConfig {
                resolution: Um(0.0),
                ..good.clone()
            },
            VirtualStageConfig {
                report_period: Duration::ZERO,
                ..good.clone()
            },
            VirtualStageConfig {
                track_length: 0,
                ..good.clone()
            },
        ] {
            bad.push((zero, "each is above zero"));
        }
        for (config, says) in bad {
            let e = VirtualStage::new(Arc::new(SimClock::stepped()), config).unwrap_err();
            assert_eq!(e.kind(), ErrorKind::InvalidArgument);
            let all = format!("{e} {:?}", e.contexts().collect::<Vec<_>>());
            assert!(all.contains(says), "{all}");
        }
    }

    #[test]
    fn grblhal_states_as_the_hal_knows_them() {
        use GrblState::*;
        let hal: Vec<StageState> = [Idle, Jog, Run, Hold, Home, Alarm]
            .iter()
            .map(|s| s.hal())
            .collect();
        assert_eq!(
            hal,
            [
                StageState::Idle,
                StageState::Jogging,
                StageState::Moving,
                StageState::Moving,
                StageState::Homing,
                StageState::Alarm
            ]
        );
    }

    #[test]
    fn dropping_the_stage_lets_its_thread_go() {
        let stage = stepped();
        let shared = Arc::downgrade(&stage.shared);
        drop(stage);
        assert!(shared.upgrade().is_none());
    }
}
