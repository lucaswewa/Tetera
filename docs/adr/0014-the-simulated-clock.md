# ADR-0014: The simulated clock

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.10a

## Context

- **M1 builds against a simulated instrument** (plan D9, §8): a virtual stage and camera imaging a simulated sample. A move must take the time its distance and speed imply, and the live image must move with it.
- **Tests must control time:** a move's timeline, a scan's or an autofocus run's steps, and a fault in the middle of a move. Waiting on the host's clock makes such tests slow and flaky, and can't give the same result twice.
- **ADR-0006** made `Clock` the one source of time: `now`, the UTC mapping made at the start, and the wall clock, for drift. It left open whether a simulated clock needs more.
- **A real stage's moves are trapezoids** (grblHAL plans each move with each axis's acceleration and fastest speed), and a feed hold slows a move to rest from wherever it has got to.

## Decision

1. **One `SimClock` for every simulated device,** implementing `Clock`, at one of three rates:
   - real time, in step with the host's clock;
   - N× as fast;
   - stepped, moving only when told.
2. **Simulated devices wait on the clock** (`wait_until`), never on the host's. A step wakes what waits on a stepped clock. Every wait has a limit in the host's time, so a device can't hang on a clock nobody steps.
3. **A simulated clock's UTC is its own:** its start maps to the wall clock's time then, and its wall clock is that mapping, so it never drifts. Timestamps from a simulated run read as times in that run.
4. **Motion is worked out from the time alone:**
   - a `Profile` is a straight move from rest to rest, trapezoidal or triangular, with each axis kept to its own acceleration and fastest speed;
   - its position at any time comes from closed forms, never from stepping a simulation;
   - so the same times give the same positions, bit for bit, at any rate, and a stepped test sees exactly what a 1× run would see at those times.
5. **A feed hold** is the same move, slowing to rest at the same acceleration from wherever it has got to.
6. **Durations round up to the nanosecond,** so a move is at rest by the time its duration says.

## Consequences

- **Gains:**
  - tests step through a move, a scan or a fault exactly, and get the same result each time;
  - the simulator runs faster than life when nothing needs to watch it;
  - a profile is cheap to sample, so the virtual camera can ask where the stage was at any instant of an exposure.
- **Costs:**
  - a stepped clock needs someone to step it: a test that waits on the stage from the thread that steps would hang, so waits have limits;
  - moves start from rest and end at rest; grblHAL blends one queued move into the next, and Tetera's stage doesn't queue moves.
- **To watch:**
  - whether the virtual camera's exposures and the stage's reports, on one clock, keep the same order at every rate;
  - whether any test needs more of a real stage's motion than a trapezoid, such as jerk.

## Alternatives considered

- **Each device on the host's clock:** simple, but slow and never the same twice.
- **A discrete-event simulation** stepping every device: exact, but every device must fit its loop, and the live server runs in real time anyway.
- **Stepping the motion numerically:** simpler to write, but errors grow with each step, and the same times would give different positions at different step sizes.
- **`ManualClock`** from ADR-0006: it steps, but nothing can wait on it, and it can't run in real time.
