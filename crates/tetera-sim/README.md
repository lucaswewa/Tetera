# tetera-sim

Tetera's simulated world, L2 (plan §8, [ADR-0014](../../docs/adr/0014-the-simulated-clock.md)): what stands in for the instrument when there is no hardware, on a clock that tests control.

- **`SimClock`** implements `Clock`, at one of three rates: real time, N× as fast, or stepped by hand. Simulated devices wait on it with `wait_until`, which a step wakes. Its UTC is its own, so it never drifts.
- **`Profile`:** a stage's move as a straight line from rest to rest, trapezoidal, or triangular when short, with each axis kept to its own acceleration and fastest speed (`AxisLimits`) and to the speed asked for. Its position and speed at any time come from closed forms, so the same times give the same positions, bit for bit. `hold` slows it to rest from wherever it has got to, as a feed hold does.

- **`VirtualStage`** implements `Stage` the way a grblHAL stage behaves, on a `SimClock`:
  - trapezoidal moves within each axis's limits;
  - grblHAL's states (Idle, Jog, Run, Hold, Home, Alarm);
  - soft limits that refuse targets beyond the travel;
  - homing that takes time: z first, then x and y, then locating the switches;
  - feed holds and jog cancels that slow to rest.

  It reports its position at a set rate (50 times a second by default), each report arriving late (5 ms), and keeps them in a track. `wait`, `stop` and `stop_jog` wait for a report taken after the last command, as a real grblHAL driver must. Faults on demand: an alarm, a hard limit, a stall. The same commands at the same times give the same track, bit for bit.

The sample, the optics and the virtual camera join in P1.11 and P1.12.

## How to try it

```bash
cargo run -p tetera-sim --example sim-clock
cargo run -p tetera-sim --example virtual-stage
```

`sim-clock` runs the clock for 100 ms of the host's time at 1× and 10×, wakes a waiting thread with five steps of 10 ms, then samples a move from the origin to (2000, 1000, 300) µm every 20 ms of stepped time, and holds it at 100 ms. `-- --smoke` waits 25 ms instead.

`virtual-stage` moves the virtual stage in stepped time, printing every 50 ms where the stage is and its latest report, which lags. Then, at 1× (10× with `-- --smoke`), through the `Stage` trait: a move, a jog, homing, a stall, a hard limit and a target beyond its soft limits, and every event.
