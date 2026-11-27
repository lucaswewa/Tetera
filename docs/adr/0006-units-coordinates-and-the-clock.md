# ADR-0006: Units, coordinates and the clock

- Status: Accepted
- Date: 2026-11-26
- Phase: P1.02

## Context

- **Lengths come in several units:** the stage moves in micrometres, the sensor counts pixels of 2.4 µm, objectives are specified in millimetres and wavelengths in nanometres. A bare `f64` says none of that, and a pixel coordinate passed as a stage coordinate is off by a factor of 20 or more without an error.
- **The plan fixes the API's units** (D18): µm as `f64` and ms as `f64` everywhere.
- **Every frame and stage report needs a timestamp** on one clock, so a frame's position can be interpolated at its exposure's midpoint (plan §6.5). The host's wall clock can jump; under WSL2 it can drift after the host sleeps (plan §6.1).
- **Simulated devices run on a clock of their own** (P1.10a), which must stand in wherever the real one does.
- **Records and datasets need IDs** that are unique, and better still sort by creation.

## Decision

1. **Lengths are `Um`,** a micrometre newtype over `f64`, with conversions from and to mm and nm. Times are `std::time::Duration`; `duration_from_ms` and `duration_ms` convert at the API's edge, and refuse negative, infinite and NaN values.
2. **Points and vectors carry their unit in their type:** `Point2<U>` and `Vector2<U>`, with `U` `Micrometres` or `Pixels`, as `PointUm`, `PointPx`, `VectorUm` and `VectorPx`. A point minus a point is a vector; a point plus a vector is a point.
3. **`Affine2<From, To>` maps one unit to another,** composes with `then`, and inverts unless it flattens the plane (a determinant below 10⁻¹²). The camera-to-stage map is an `Affine2<Pixels, Micrometres>`, and `similarity` builds one from a scale, a rotation and an offset.
4. **`Rect` is whole pixels** (`u32`), with its edges computed in `u64` so a rectangle at the sensor's edge can't overflow. `StagePos` is x, y and z in `Um`.
5. **One clock:**
   - a `Timestamp` is nanoseconds since its clock started, monotonic;
   - a `Clock` maps its start to a `UtcTime` once;
   - `Clock::drift` is how far the wall clock has since moved from that mapping. Beyond `DRIFT_LIMIT`, one second either way, it is worth a warning, which `tetera-log` logs (ADR-0007);
   - `SystemClock` is the host's; `ManualClock` moves only when told, never backwards, and its wall clock can be offset to stand for a drifting host.
6. **`UtcTime` is nanoseconds since the Unix epoch** (`i64`, the years 1678 to 2261), read and written as RFC 3339 in UTC (`2026-11-26T22:53:20.25Z`), with Howard Hinnant's civil-date algorithms.
7. **`Id<T>` is a UUID v7 typed by what it names.** `uuid`'s `now_v7` guarantees that IDs from one process sort by creation.
8. **Every type that crosses the API serialises with serde and describes itself with `schemars`,** always on: the API needs the schemas from P1.13, and a feature no gate turns on would leave the derives unchecked.

## Consequences

- **Gains:** a pixel can't be used as a micrometre, nor a map applied the wrong way; tests can run time exactly with `ManualClock`; drift after a sleep is noticed, not silently baked into timestamps.
- **Costs:** generic points need hand-written schemas; `UtcTime`'s range stops in 2261.
- **To watch:** whether the simulated clock (P1.10a) needs more of `Clock` than `now`, `origin` and `wall_now`.

## Alternatives considered

- **Bare `f64` everywhere:** the bug this prevents is one of the commonest in instrument software.
- **A units crate** (`uom`): complete, but heavy for the few units Tetera uses, and its types leak into every signature.
- **`chrono` or `time` for UTC:** OScope's notes flag `chrono` as unmaintained; `time` would be a dependency for one format, which forty lines cover.
- **UUID v4:** unique, but in no useful order.
- **The wall clock for timestamps:** it can jump, and frames would be placed wrongly in time.
