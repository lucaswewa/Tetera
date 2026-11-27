//! The simulated clock (ADR-0014): one clock for every simulated device, which runs in step
//! with the host, faster, or only when stepped. Devices wait on it, so a test that steps it
//! decides when each thing happens.

use std::sync::{Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};
use tetera_core::{Clock, Error, Result, Timestamp, UtcTime};

/// How a simulated clock runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rate {
    /// In step with the host's clock.
    RealTime,
    /// This many times as fast as the host's clock.
    Faster(f64),
    /// Only when stepped.
    Stepped,
}

/// A simulated clock. Its UTC is its own: it maps its start to the wall clock's time then,
/// and its wall clock is that mapping, so it never drifts.
#[derive(Debug)]
pub struct SimClock {
    rate: Rate,
    started: Instant,
    origin: UtcTime,
    /// A stepped clock's time, in ns.
    stepped: Mutex<u64>,
    stepped_on: Condvar,
}

impl SimClock {
    /// A clock that runs at `rate`, from now. A rate of N× is above zero.
    pub fn new(rate: Rate) -> Result<Self> {
        if let Rate::Faster(n) = rate
            && !(n.is_finite() && n > 0.0)
        {
            return Err(Error::invalid_argument(format!(
                "a clock {n} times as fast; it runs faster than zero"
            )));
        }
        Ok(Self::from_valid(rate))
    }

    /// A clock in step with the host's.
    pub fn real_time() -> Self {
        Self::from_valid(Rate::RealTime)
    }

    /// A clock that moves only when stepped.
    pub fn stepped() -> Self {
        Self::from_valid(Rate::Stepped)
    }

    fn from_valid(rate: Rate) -> Self {
        Self {
            rate,
            started: Instant::now(),
            origin: UtcTime::now(),
            stepped: Mutex::new(0),
            stepped_on: Condvar::new(),
        }
    }

    /// How it runs.
    pub fn rate(&self) -> Rate {
        self.rate
    }

    /// Moves a stepped clock `d` forward, and wakes what waits on it. A clock that runs by
    /// itself refuses.
    pub fn step(&self, d: Duration) -> Result<()> {
        if self.rate != Rate::Stepped {
            return Err(Error::invalid_argument(format!(
                "the clock runs by itself ({:?}); only a stepped clock is stepped",
                self.rate
            )));
        }
        let mut now = self.stepped.lock().unwrap_or_else(PoisonError::into_inner);
        *now = now.saturating_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
        self.stepped_on.notify_all();
        Ok(())
    }

    /// How long `d` of this clock's time takes on the host's clock: `d` at 1×, `d / n` at N×,
    /// and nothing to say when it is stepped.
    pub fn host_time(&self, d: Duration) -> Option<Duration> {
        match self.rate {
            Rate::RealTime => Some(d),
            Rate::Faster(n) => Some(d.div_f64(n)),
            Rate::Stepped => None,
        }
    }

    /// Waits until the clock reaches `t`, or until `at_most` of the host's time has passed:
    /// whether it reached `t`.
    pub fn wait_until(&self, t: Timestamp, at_most: Duration) -> bool {
        let deadline = Instant::now() + at_most;
        if self.rate == Rate::Stepped {
            let mut now = self.stepped.lock().unwrap_or_else(PoisonError::into_inner);
            while *now < nanos(t) {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return false;
                }
                now = self
                    .stepped_on
                    .wait_timeout(now, left)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            return true;
        }
        loop {
            let now = self.now();
            if now >= t {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            let host = self.host_time(t.saturating_since(now)).unwrap_or(left);
            std::thread::sleep(host.min(left));
        }
    }
}

fn nanos(t: Timestamp) -> u64 {
    u64::try_from(t.as_duration().as_nanos()).unwrap_or(u64::MAX)
}

impl Clock for SimClock {
    fn now(&self) -> Timestamp {
        let elapsed = match self.rate {
            Rate::RealTime => self.started.elapsed(),
            Rate::Faster(n) => self.started.elapsed().mul_f64(n),
            Rate::Stepped => {
                Duration::from_nanos(*self.stepped.lock().unwrap_or_else(PoisonError::into_inner))
            }
        };
        Timestamp::from_duration(elapsed)
    }

    fn origin(&self) -> UtcTime {
        self.origin
    }

    fn wall_now(&self) -> UtcTime {
        self.utc(self.now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tetera_core::ErrorKind;

    fn at(ms: u64) -> Timestamp {
        Timestamp::from_duration(Duration::from_millis(ms))
    }

    #[test]
    fn a_stepped_clock_moves_only_when_stepped() {
        let clock = SimClock::stepped();
        assert_eq!(clock.rate(), Rate::Stepped);
        assert_eq!(clock.now(), at(0));
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(clock.now(), at(0), "still");
        clock.step(Duration::from_millis(5)).unwrap();
        clock.step(Duration::from_micros(250)).unwrap();
        assert_eq!(
            clock.now(),
            Timestamp::from_duration(Duration::from_micros(5_250))
        );
        assert_eq!(clock.host_time(Duration::from_secs(1)), None);
        assert_eq!(clock.drift(), 0, "its own UTC");
        assert_eq!(
            clock.utc(clock.now()),
            clock.origin().saturating_add(Duration::from_micros(5_250))
        );
    }

    #[test]
    fn a_running_clock_follows_the_host_at_its_rate_and_refuses_steps() {
        for (rate, n) in [(Rate::RealTime, 1.0), (Rate::Faster(10.0), 10.0)] {
            let clock = SimClock::new(rate).unwrap();
            let host = Instant::now();
            std::thread::sleep(Duration::from_millis(20));
            let (sim, host) = (clock.now().as_duration(), host.elapsed());
            assert!(
                sim >= Duration::from_millis(20).mul_f64(n),
                "{rate:?}: {sim:?}"
            );
            assert!(
                sim <= host.mul_f64(n) + Duration::from_millis(1),
                "{rate:?}: {sim:?}"
            );
            assert_eq!(
                clock.step(Duration::from_millis(1)).unwrap_err().kind(),
                ErrorKind::InvalidArgument
            );
            assert_eq!(
                clock.host_time(Duration::from_millis(100)),
                Some(Duration::from_millis(100).div_f64(n))
            );
            assert!(
                clock.drift().abs() < 1_000_000,
                "its own UTC: {}",
                clock.drift()
            );
        }
        assert_eq!(SimClock::real_time().rate(), Rate::RealTime);
    }

    #[test]
    fn a_rate_is_above_zero() {
        for n in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            let e = SimClock::new(Rate::Faster(n)).unwrap_err();
            assert_eq!(e.kind(), ErrorKind::InvalidArgument);
            assert!(e.message().contains("runs faster than zero"), "{e}");
        }
        assert!(SimClock::new(Rate::Faster(0.5)).is_ok());
    }

    #[test]
    fn a_clock_never_goes_back() {
        for clock in [
            SimClock::real_time(),
            SimClock::new(Rate::Faster(1_000.0)).unwrap(),
        ] {
            let mut last = clock.now();
            for _ in 0..10_000 {
                let now = clock.now();
                assert!(now >= last);
                last = now;
            }
        }
        let clock = SimClock::stepped();
        clock.step(Duration::MAX).unwrap();
        clock.step(Duration::from_secs(1)).unwrap();
        assert_eq!(
            clock.now().as_duration(),
            Duration::from_nanos(u64::MAX),
            "held at its end"
        );
    }

    #[test]
    fn a_step_wakes_what_waits_on_a_stepped_clock() {
        let clock = Arc::new(SimClock::stepped());
        let waiter = clock.clone();
        let waiting =
            std::thread::spawn(move || waiter.wait_until(at(10), Duration::from_secs(10)));
        std::thread::sleep(Duration::from_millis(20));
        clock.step(Duration::from_millis(5)).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert!(!waiting.is_finished(), "5 ms of the 10");
        clock.step(Duration::from_millis(5)).unwrap();
        assert!(waiting.join().unwrap(), "woken at 10 ms");
        assert!(clock.wait_until(at(3), Duration::ZERO), "already past");
        let host = Instant::now();
        assert!(
            !clock.wait_until(at(11), Duration::from_millis(30)),
            "never stepped"
        );
        assert!(host.elapsed() >= Duration::from_millis(30));
    }

    #[test]
    fn a_running_clock_waits_as_long_as_its_rate_says() {
        let clock = SimClock::new(Rate::Faster(100.0)).unwrap();
        let host = Instant::now();
        let t = clock.now().checked_add(Duration::from_secs(2)).unwrap();
        assert!(clock.wait_until(t, Duration::from_secs(5)));
        let took = host.elapsed();
        assert!(clock.now() >= t);
        assert!(took >= Duration::from_millis(19), "2 s at 100×: {took:?}");
        assert!(took < Duration::from_secs(1), "{took:?}");
        let host = Instant::now();
        let far = clock.now().checked_add(Duration::from_secs(3_600)).unwrap();
        assert!(
            !clock.wait_until(far, Duration::from_millis(20)),
            "an hour, at most 20 ms"
        );
        assert!(host.elapsed() < Duration::from_secs(1));
    }
}
