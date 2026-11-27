# tetera-core

Tetera's foundation layer, L0: the types every other crate shares. It does no I/O and has no async code ([plan §6.2](../../docs/implementation-plan.md#62-layers-and-crates), [ADR-0006](../../docs/adr/0006-units-coordinates-and-the-clock.md)).

- **`units`:** `Um`, a length in micrometres; durations in milliseconds at the API's edge.
- **`geometry`:** `PointUm`, `PointPx` and their vectors, typed by unit; `Affine2<From, To>`, such as the camera-to-stage map; `Rect`, pixel rectangles; `StagePos`.
- **`time`:** `Timestamp`, nanoseconds on a monotonic clock; `UtcTime`, read and written as RFC 3339; the `Clock` trait, with `SystemClock` and `ManualClock`; the drift check.
- **`id`:** `Id<T>`, a UUID v7 typed by what it names, sorting by creation.
- **`error`:** `Error`, with an `ErrorKind`, a context chain and a source.
- **`log`:** `LogRecord`, `LogLevel` (with *critical*), and `LogRing`, which keeps the newest records.
- **`frame`** ([ADR-0008](../../docs/adr/0008-the-frame-model-the-buffer-pool-and-the-frame-bus.md)): `PixelFormat` with Bayer patterns that follow a crop; `FrameLayout` with aligned rows; `FrameMeta`; `FramePool`, which lends `FrameMut`s to fill and shares frozen `Frame`s, its buffers returning when the last share drops; safe views of rows, samples and Bayer sites.

## How to try it

```bash
cargo run -p tetera-core --example core_tour
cargo run -p tetera-core --example frame_pool
cargo run -p tetera-core --example hello -- --smoke
```

`core_tour` walks through each module:
- a length in sensor pixels;
- a pixel mapped to the stage at 50× and back;
- an ROI clipped to the sensor;
- the clock and its drift;
- two IDs in order;
- an error with its context.

With `-- --smoke`, it also prints `smoke: ok`.

`frame_pool` makes a pool of 16 full MIChrome 20 frames (about 640 MB) and feeds 48 frames through it to three consumer threads, each reading a row of every frame without a copy. It prints the pool's counts (48 taken, 16 allocated), then lends all 16 and shows a 17th take waiting 200 ms and failing with a clear message. `-- --smoke` checks the counts.

To try the debugger, open the repository in VS Code over Remote-WSL, set a breakpoint in an example, choose **Debug an example** in the Run and Debug view, enter the example's name when asked, and press F5.
