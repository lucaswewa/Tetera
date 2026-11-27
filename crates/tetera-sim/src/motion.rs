//! Motion profiles: how a stage's move goes in time. A move is a straight line from rest to
//! rest; along it the stage speeds up at a constant acceleration, cruises, and slows down at
//! the same rate, the trapezoid, or only speeds up and slows down when the move is short, the
//! triangle. Each axis is kept to its own acceleration and fastest speed.
//!
//! A profile is worked out from the time alone, so the same times give the same positions,
//! bit for bit. Distances are in µm, speeds in µm/s, accelerations in µm/s² (plan D18).

use std::time::Duration;
use tetera_core::{Error, Result, StagePos, Um};

/// What one axis's motor can do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisLimits {
    /// Its fastest speed, in µm/s.
    pub max_speed: f64,
    /// Its acceleration, in µm/s².
    pub acceleration: f64,
}

impl AxisLimits {
    fn check(&self, axis: usize) -> Result<()> {
        let good = |v: f64| v.is_finite() && v > 0.0;
        if good(self.max_speed) && good(self.acceleration) {
            return Ok(());
        }
        Err(Error::invalid_argument(format!(
            "axis {axis}: a fastest speed of {} µm/s and an acceleration of {} µm/s²; each is above zero",
            self.max_speed, self.acceleration
        )))
    }
}

/// One part of a profile, of constant acceleration. Times are in seconds from the move's start.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Segment {
    /// Speeding up from rest at the start, until `end`.
    SpeedUp { end: f64 },
    /// At the profile's speed from `start`, `s0` along, until `end`.
    Cruise { start: f64, end: f64, s0: f64 },
    /// Slowing down from `start` to rest at `end`, `s_end` along.
    SlowDown { start: f64, end: f64, s_end: f64 },
}

/// A straight move from rest to rest, and how it goes in time.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    from: StagePos,
    /// Where it comes to rest: its target, unless it was held.
    to: StagePos,
    /// Each axis's share of the path, a unit vector.
    direction: [f64; 3],
    /// How far along the path it comes to rest, in µm.
    stop: f64,
    /// Its cruising speed, or its peak on a triangle, in µm/s.
    speed: f64,
    /// Its acceleration along the path, in µm/s².
    acceleration: f64,
    segments: Vec<Segment>,
}

fn coordinates(p: &StagePos) -> [f64; 3] {
    [p.x.um(), p.y.um(), p.z.um()]
}

impl Profile {
    /// A move from `from` to `to`, at `speed` µm/s or as fast as it goes, keeping each axis
    /// (x, y, z) to its `limits`.
    pub fn new(
        from: StagePos,
        to: StagePos,
        speed: Option<f64>,
        limits: &[AxisLimits; 3],
    ) -> Result<Self> {
        for (axis, l) in limits.iter().enumerate() {
            l.check(axis)?;
        }
        if let Some(v) = speed
            && !(v.is_finite() && v > 0.0)
        {
            return Err(Error::invalid_argument(format!(
                "a speed of {v} µm/s; a speed is above zero"
            )));
        }
        let (a, b) = (coordinates(&from), coordinates(&to));
        let delta = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        if delta.iter().any(|d| !d.is_finite()) {
            return Err(Error::invalid_argument(format!(
                "a move from {from:?} to {to:?}"
            )));
        }
        let length = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        if length == 0.0 {
            return Ok(Self {
                from,
                to,
                direction: [0.0; 3],
                stop: 0.0,
                speed: 0.0,
                acceleration: 0.0,
                segments: Vec::new(),
            });
        }
        let direction = delta.map(|d| d / length);
        // Along the path, as fast and as hard as the most limited axis allows.
        let mut v = speed.unwrap_or(f64::INFINITY);
        let mut acc = f64::INFINITY;
        for (share, l) in direction.iter().zip(limits) {
            let share = share.abs();
            if share > 0.0 {
                v = v.min(l.max_speed / share);
                acc = acc.min(l.acceleration / share);
            }
        }
        let ramp = v * v / acc;
        let segments = if ramp >= length {
            // A triangle: it never reaches `v`.
            v = (acc * length).sqrt();
            let t = v / acc;
            vec![
                Segment::SpeedUp { end: t },
                Segment::SlowDown {
                    start: t,
                    end: 2.0 * t,
                    s_end: length,
                },
            ]
        } else {
            let t = v / acc;
            let cruise = (length - ramp) / v;
            vec![
                Segment::SpeedUp { end: t },
                Segment::Cruise {
                    start: t,
                    end: t + cruise,
                    s0: ramp / 2.0,
                },
                Segment::SlowDown {
                    start: t + cruise,
                    end: 2.0 * t + cruise,
                    s_end: length,
                },
            ]
        };
        Ok(Self {
            from,
            to,
            direction,
            stop: length,
            speed: v,
            acceleration: acc,
            segments,
        })
    }

    /// Where it starts.
    pub fn from(&self) -> StagePos {
        self.from
    }

    /// Where it comes to rest.
    pub fn to(&self) -> StagePos {
        self.to
    }

    /// How far it goes, in µm.
    pub fn length(&self) -> f64 {
        self.stop
    }

    /// Its cruising speed, or its peak on a short move, in µm/s.
    pub fn peak_speed(&self) -> f64 {
        self.speed
    }

    /// Its acceleration along the path, in µm/s².
    pub fn acceleration(&self) -> f64 {
        self.acceleration
    }

    /// Each axis's share of the path, a unit vector; none, for a move that goes nowhere.
    pub fn direction(&self) -> [f64; 3] {
        self.direction
    }

    fn end(&self) -> f64 {
        match self.segments.last() {
            Some(Segment::SlowDown { end, .. }) => *end,
            _ => 0.0,
        }
    }

    /// How long it takes, to the nanosecond above: by then it is at rest.
    pub fn duration(&self) -> Duration {
        let d = Duration::from_secs_f64(self.end());
        if d.as_secs_f64() < self.end() {
            d + Duration::from_nanos(1)
        } else {
            d
        }
    }

    /// How far along it is, in µm, and how fast it goes, in µm/s, `t` after it starts.
    pub fn along(&self, t: Duration) -> (f64, f64) {
        let t = t.as_secs_f64();
        let (v, a) = (self.speed, self.acceleration);
        for segment in &self.segments {
            match *segment {
                Segment::SpeedUp { end } if t < end => return (a * t * t / 2.0, a * t),
                Segment::Cruise { end, start, s0 } if t < end => return (s0 + v * (t - start), v),
                Segment::SlowDown { end, s_end, .. } if t < end => {
                    let left = end - t;
                    return (s_end - a * left * left / 2.0, a * left);
                }
                _ => {}
            }
        }
        (self.stop, 0.0)
    }

    /// Where it is, `t` after it starts: exactly where it comes to rest, once it has.
    pub fn position(&self, t: Duration) -> StagePos {
        if t.as_secs_f64() >= self.end() {
            return self.to;
        }
        self.point(self.along(t).0)
    }

    /// The point `s` µm along its line.
    fn point(&self, s: f64) -> StagePos {
        let [x, y, z] = coordinates(&self.from);
        let [dx, dy, dz] = self.direction;
        StagePos::new(Um(x + dx * s), Um(y + dy * s), Um(z + dz * s))
    }

    /// The same move held `t` after it starts: from there it slows down to rest at the same
    /// acceleration, as a feed hold does. Held while it slows down already, or once it is at
    /// rest, it is the same move.
    pub fn hold(&self, t: Duration) -> Profile {
        let at = t.as_secs_f64();
        let (s, v) = self.along(t);
        let slowing = self
            .segments
            .iter()
            .any(|segment| matches!(*segment, Segment::SlowDown { start, .. } if at >= start));
        if slowing || at >= self.end() {
            return self.clone();
        }
        let a = self.acceleration;
        let stop = s + v * v / (2.0 * a);
        let mut segments: Vec<Segment> = Vec::new();
        for segment in &self.segments {
            match *segment {
                Segment::SpeedUp { end } => segments.push(Segment::SpeedUp { end: end.min(at) }),
                // Held before it cruises, the cruise ends at the hold, before it would start, and
                // is never reached.
                Segment::Cruise { start, s0, .. } => {
                    segments.push(Segment::Cruise { start, end: at, s0 });
                }
                _ => {}
            }
        }
        segments.push(Segment::SlowDown {
            start: at,
            end: at + v / a,
            s_end: stop,
        });
        Profile {
            to: self.point(stop),
            stop,
            // Its speed when held: its cruising speed, or less while it was speeding up.
            speed: v,
            segments,
            ..*self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The fake stage's axes: x and y at 10 mm/s and 100 mm/s², z at 2 mm/s and 50 mm/s².
    const LIMITS: [AxisLimits; 3] = [
        AxisLimits {
            max_speed: 10_000.0,
            acceleration: 100_000.0,
        },
        AxisLimits {
            max_speed: 10_000.0,
            acceleration: 100_000.0,
        },
        AxisLimits {
            max_speed: 2_000.0,
            acceleration: 50_000.0,
        },
    ];

    fn at(x: f64, y: f64, z: f64) -> StagePos {
        StagePos::new(Um(x), Um(y), Um(z))
    }

    fn secs(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
    }

    fn profile(to: StagePos, speed: Option<f64>) -> Profile {
        Profile::new(StagePos::default(), to, speed, &LIMITS).unwrap()
    }

    #[test]
    fn a_long_move_speeds_up_cruises_and_slows_down() {
        // 10 mm in x: 0.1 s to reach 10 mm/s, 0.5 mm; 0.9 s at speed; 0.1 s to stop.
        let p = profile(at(10_000.0, 0.0, 0.0), None);
        assert_eq!((p.peak_speed(), p.acceleration()), (10_000.0, 100_000.0));
        assert!(near(p.duration().as_secs_f64(), 1.1));
        for (t, s, v) in [
            (0.0, 0.0, 0.0),
            (0.05, 125.0, 5_000.0),
            (0.1, 500.0, 10_000.0),
            (0.6, 5_500.0, 10_000.0),
            (1.05, 9_875.0, 5_000.0),
            (1.1, 10_000.0, 0.0),
            (5.0, 10_000.0, 0.0),
        ] {
            let (got_s, got_v) = p.along(secs(t));
            assert!(
                near(got_s, s) && near(got_v, v),
                "at {t} s: {got_s}, {got_v}"
            );
        }
        assert_eq!(p.position(secs(0.6)).x, Um(p.along(secs(0.6)).0));
        assert_eq!(
            (p.from(), p.to(), p.length()),
            (StagePos::default(), at(10_000.0, 0.0, 0.0), 10_000.0)
        );
    }

    #[test]
    fn a_short_move_is_a_triangle() {
        // 100 µm in x: half way at 100 mm/s², it never reaches 10 mm/s.
        let p = profile(at(100.0, 0.0, 0.0), None);
        let peak = (100_000.0f64 * 100.0).sqrt();
        assert!(near(p.peak_speed(), peak));
        assert!(near(p.duration().as_secs_f64(), 2.0 * peak / 100_000.0));
        // At the peak, to the nanosecond the duration is rounded to.
        let (s, v) = p.along(p.duration() / 2);
        assert!(
            (s - 50.0).abs() < 1e-5 && (v - peak).abs() < 1e-3,
            "{s}, {v}"
        );
    }

    #[test]
    fn the_most_limited_axis_sets_the_pace() {
        // 3 mm in x and 4 mm in z: z, 0.8 of the path, keeps it to 2.5 mm/s and 62.5 mm/s².
        let p = profile(at(3_000.0, 0.0, 4_000.0), None);
        assert_eq!(p.direction(), [0.6, 0.0, 0.8]);
        assert!(near(p.peak_speed(), 2_500.0) && near(p.acceleration(), 62_500.0));
        assert!(near(
            profile(at(3_000.0, 0.0, 4_000.0), Some(1_000.0)).peak_speed(),
            1_000.0
        ));
        assert!(near(
            profile(at(10_000.0, 0.0, 0.0), Some(1e9)).peak_speed(),
            10_000.0
        ));
        let mid = p.position(p.duration() / 2);
        assert_eq!(mid.y, Um(0.0), "y stays put");
        assert!(near(mid.x.um() * 4.0, mid.z.um() * 3.0), "on the line");
    }

    #[test]
    fn a_move_to_where_it_is_takes_no_time() {
        let here = at(5.0, 6.0, 7.0);
        let p = Profile::new(here, here, None, &LIMITS).unwrap();
        assert_eq!(
            (p.duration(), p.length(), p.direction()),
            (Duration::ZERO, 0.0, [0.0; 3])
        );
        assert_eq!(
            (p.position(secs(1.0)), p.along(secs(1.0))),
            (here, (0.0, 0.0))
        );
        assert_eq!(p.hold(Duration::ZERO), p);
    }

    #[test]
    fn bad_limits_speeds_and_targets_are_refused() {
        let origin = StagePos::default();
        for (limit, says) in [
            (
                AxisLimits {
                    max_speed: 0.0,
                    acceleration: 1.0,
                },
                "axis 1: a fastest speed of 0",
            ),
            (
                AxisLimits {
                    max_speed: 1.0,
                    acceleration: f64::NAN,
                },
                "an acceleration of NaN",
            ),
            (
                AxisLimits {
                    max_speed: f64::INFINITY,
                    acceleration: 1.0,
                },
                "each is above zero",
            ),
            (
                AxisLimits {
                    max_speed: 1.0,
                    acceleration: -1.0,
                },
                "each is above zero",
            ),
        ] {
            let limits = [LIMITS[0], limit, LIMITS[2]];
            let e = Profile::new(origin, at(1.0, 0.0, 0.0), None, &limits).unwrap_err();
            assert!(e.message().contains(says), "{e}");
        }
        for speed in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let e = Profile::new(origin, at(1.0, 0.0, 0.0), Some(speed), &LIMITS).unwrap_err();
            assert!(e.message().contains("a speed is above zero"), "{e}");
        }
        for to in [at(f64::NAN, 0.0, 0.0), at(0.0, f64::INFINITY, 0.0)] {
            assert!(Profile::new(origin, to, None, &LIMITS).is_err());
        }
    }

    #[test]
    fn a_hold_slows_down_from_wherever_the_move_has_got_to() {
        let p = profile(at(10_000.0, 0.0, 0.0), None);
        // Cruising at 0.6 s, 5.5 mm along: 0.1 s and 0.5 mm more.
        let held = p.hold(secs(0.6));
        assert!(near(held.length(), 6_000.0) && near(held.duration().as_secs_f64(), 0.7));
        assert_eq!(held.peak_speed(), 10_000.0);
        assert!(near(held.to().x.um(), 6_000.0));
        assert_eq!(held.position(held.duration()), held.to());
        // Speeding up at 0.05 s, 125 µm along at 5 mm/s: 0.05 s and 125 µm more.
        let held = p.hold(secs(0.05));
        assert!(near(held.length(), 250.0) && near(held.duration().as_secs_f64(), 0.1));
        assert!(near(held.peak_speed(), 5_000.0));
        // Slowing down already, or done: the same move.
        assert_eq!(p.hold(secs(1.05)), p);
        assert_eq!(p.hold(secs(2.0)), p);
        // Before it starts: where it is.
        let held = p.hold(Duration::ZERO);
        assert_eq!(
            (held.duration(), held.to()),
            (Duration::ZERO, StagePos::default())
        );
    }

    #[test]
    fn the_same_times_give_the_same_positions_bit_for_bit() {
        let a = profile(at(1_234.5, -678.9, 321.0), Some(3_333.3));
        let b = profile(at(1_234.5, -678.9, 321.0), Some(3_333.3));
        for k in 0..1_000u32 {
            let t = a.duration().mul_f64(f64::from(k) / 999.0);
            let (p, q) = (a.position(t), b.position(t));
            assert_eq!(
                [p.x.0.to_bits(), p.y.0.to_bits(), p.z.0.to_bits()],
                [q.x.0.to_bits(), q.y.0.to_bits(), q.z.0.to_bits()]
            );
        }
    }

    fn point() -> impl Strategy<Value = StagePos> {
        (
            -100_000.0..100_000.0f64,
            -100_000.0..100_000.0f64,
            -10_000.0..10_000.0f64,
        )
            .prop_map(|(x, y, z)| at(x, y, z))
    }

    fn limits() -> impl Strategy<Value = [AxisLimits; 3]> {
        prop::array::uniform3((100.0..20_000.0f64, 1_000.0..1_000_000.0f64)).prop_map(|axes| {
            axes.map(|(max_speed, acceleration)| AxisLimits {
                max_speed,
                acceleration,
            })
        })
    }

    /// What the path allows, worked out here on its own: its speed, acceleration and the
    /// trapezoid's duration.
    fn expected(p: &Profile, speed: Option<f64>, limits: &[AxisLimits; 3]) -> (f64, f64, f64) {
        let mut v = speed.unwrap_or(f64::INFINITY);
        let mut a = f64::INFINITY;
        for (u, l) in p.direction().iter().zip(limits) {
            if *u != 0.0 {
                v = v.min(l.max_speed / u.abs());
                a = a.min(l.acceleration / u.abs());
            }
        }
        let length = p.length();
        let t = if v * v / a >= length {
            2.0 * (length / a).sqrt()
        } else {
            length / v + v / a
        };
        (v, a, t)
    }

    /// Sampled `n` times, it never goes back, and no axis goes faster or speeds up or slows
    /// down harder than it may.
    fn keeps_to(
        p: &Profile,
        v: f64,
        a: f64,
        limits: &[AxisLimits; 3],
    ) -> std::result::Result<(), TestCaseError> {
        let n = 200;
        let end = p.duration().as_secs_f64();
        let slack = 1e-9 * p.length().max(1.0);
        let mut last = p.along(Duration::ZERO);
        prop_assert_eq!(last, (0.0, 0.0), "from rest");
        for k in 1..=n {
            // Each sample is at a whole nanosecond, so a gap may be up to 2 ns longer.
            let dt = end / f64::from(n);
            let now = p.along(secs(end * f64::from(k) / f64::from(n)));
            prop_assert!(now.0 >= last.0 - slack, "back from {} to {}", last.0, now.0);
            prop_assert!(
                now.1 >= 0.0 && now.1 <= v * (1.0 + 1e-9),
                "a speed of {}",
                now.1
            );
            prop_assert!(
                (now.1 - last.1).abs() <= a * (dt + 2e-9) * (1.0 + 1e-9) + 1e-9,
                "a jump in speed at sample {}: from {} to {}, at most {} in {} s",
                k,
                last.1,
                now.1,
                a * dt,
                dt
            );
            prop_assert!(
                now.0 - last.0 <= v * (dt + 2e-9) * (1.0 + 1e-9) + slack,
                "faster than it may, on average"
            );
            last = now;
        }
        prop_assert_eq!(p.along(p.duration()).1, 0.0, "to rest");
        for (u, l) in p.direction().iter().zip(limits) {
            prop_assert!(u.abs() * p.peak_speed() <= l.max_speed * (1.0 + 1e-9));
            prop_assert!(u.abs() * p.acceleration() <= l.acceleration * (1.0 + 1e-9));
        }
        Ok(())
    }

    /// Whether `at` is `s` µm along the line from `from` in `direction`.
    fn on_line(at: &StagePos, from: &StagePos, direction: [f64; 3], s: f64) -> bool {
        let (a, b) = (coordinates(at), coordinates(from));
        (0..3).all(|i| (a[i] - (b[i] + direction[i] * s)).abs() <= 1e-6)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        #[test]
        fn any_move_keeps_to_its_limits_and_ends_where_it_should(
            from in point(),
            to in point(),
            speed in prop::option::of(10.0..50_000.0f64),
            limits in limits(),
        ) {
            let p = Profile::new(from, to, speed, &limits).unwrap();
            prop_assert_eq!((p.from(), p.position(Duration::ZERO)), (from, from));
            prop_assert_eq!(p.position(p.duration()), to);
            for k in 1..10u32 {
                let t = p.duration().mul_f64(f64::from(k) / 10.0);
                prop_assert!(on_line(&p.position(t), &from, p.direction(), p.along(t).0), "partway, on its line");
            }
            let (v, a, t) = expected(&p, speed, &limits);
            prop_assert!((p.duration().as_secs_f64() - t).abs() < 1e-6, "{:?} against {}", p.duration(), t);
            prop_assert!(near(p.acceleration(), a));
            prop_assert!(p.peak_speed() <= v * (1.0 + 1e-12));
            keeps_to(&p, v, a, &limits)?;
        }

        #[test]
        fn a_held_move_stops_within_its_slowing_distance(
            from in point(),
            to in point(),
            speed in prop::option::of(10.0..50_000.0f64),
            limits in limits(),
            when in 0.0..1.0f64,
        ) {
            let p = Profile::new(from, to, speed, &limits).unwrap();
            let t = p.duration().mul_f64(when);
            let held = p.hold(t);
            let (s, v) = p.along(t);
            let slack = 1e-9 * p.length().max(1.0);
            for k in 0..=20u32 {
                let before = t.mul_f64(f64::from(k) / 20.0);
                prop_assert!((held.along(before).0 - p.along(before).0).abs() <= slack, "the same until the hold");
            }
            prop_assert!(held.length() <= p.length() + slack);
            prop_assert!(held.length() <= s + v * v / (2.0 * p.acceleration()) + slack);
            prop_assert!(held.duration() <= p.duration() + Duration::from_nanos(1));
            prop_assert_eq!(held.position(held.duration()), held.to());
            prop_assert!(on_line(&held.to(), &from, p.direction(), held.length()), "at rest on its line");
            let (pv, pa, _) = expected(&p, speed, &limits);
            keeps_to(&held, pv, pa, &limits)?;
        }
    }
}
