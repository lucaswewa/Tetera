# tetera-sim

Tetera's simulated world, L2 (plan §8, [ADR-0014](../../docs/adr/0014-the-simulated-clock.md)): what stands in for the instrument when there is no hardware, on a clock that tests control.

- **`SimClock`** implements `Clock`, at one of three rates: real time, N× as fast, or stepped by hand. Simulated devices wait on it with `wait_until`, which a step wakes. Its UTC is its own, so it never drifts.
- **`Profile`:** a stage's move as a straight line from rest to rest, trapezoidal, or triangular when short, with each axis kept to its own acceleration and fastest speed (`AxisLimits`) and to the speed asked for. Its position and speed at any time come from closed forms, so the same times give the same positions, bit for bit. `hold` slows it to rest from wherever it has got to, as a feed hold does.

The virtual stage joins in P1.10b; the sample, the optics and the virtual camera in P1.11 and P1.12.

## How to try it

```bash
cargo run -p tetera-sim --example sim-clock
```

`sim-clock` runs the clock for 100 ms of the host's time at 1× and 10×, wakes a waiting thread with five steps of 10 ms, then samples a move from the origin to (2000, 1000, 300) µm every 20 ms of stepped time, and holds it at 100 ms. `-- --smoke` waits 25 ms instead.
