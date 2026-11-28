# tetera-workflows

Tetera's workflows, L4 (plan §6.2 and §6.10): work that drives the instrument from start to end, run as jobs ([ADR-0020](../../docs/adr/0020-the-job-service.md)). Autofocus, scans and calibration come in P1.16 to P1.18.

- **`capture(rig, request, step)`** takes one frame, N frames at an interval, or a burst at the camera's full rate, into a new dataset ([ADR-0021](../../docs/adr/0021-datasets-on-disk.md)):
  - **The request** names the project, the dataset, and what to take: `single`, `series` (`count`, `interval_ms`) or `burst` (`count`). The camera's `mode` is 0, full resolution, unless the request names one; its `bit_depth` is 12, and its `exposure_ms` is the live view's. A request that can't be taken, or a project that isn't there, is refused before the camera is touched.
  - **The mode switch (D5):** the camera is stopped if it streams, set to the capture's mode, and snapped once, which builds the mode's buffers. Afterwards it is set back as it was and, if it streamed, restarted with the live view's sink. Both switches are timed, `switch_ms` and `back_ms`.
  - **A single frame or a series** is snapped frame by frame into a dataset writer, waiting out each interval.
  - **A burst** records the camera's stream through the frame bus into a frame log on the scratch disk (`tetera-acquire`'s recorder), then ingests the first `count` frames into the dataset. The log is removed either way. `missing` counts the frames the camera dropped, each once.
  - **`step(done, message)`** is called as it goes, `done` from 0 to 1. An error from it cancels the capture: nothing is left, neither the dataset nor the frame log, and the camera is put back as it was.
- **`Rig`** is what a capture works with: the camera, the frame bus, the store, the scratch folder, the live view's sink, a sensor pixel's size at the sample, and the instrument as it is, for the dataset's metadata. The dataset's pixel size is the rig's, times the mode's binning.

The `capture` Thing (`tetera-api`) runs it as a hardware job, its step the job's progress and cancel check.

- **`focus::search(probe, depth_of_field, limits, options, step)`** (P1.16a) finds focus through z, as [P1.16a's behaviour spec](../../docs/m1/notes/P1.16a-focus-measures-and-the-search.md#behaviour-spec) has it:
  - **a coarse sweep** of ±8 depths of field in steps of one, then **a fine one** of ±2 in halves around its peak, and **a parabola** through the logarithm of the fine sweep's top half, whose vertex is the focus;
  - **against backlash,** every position approached from below: down by the take-up, 10 µm, before each sweep and the focus, then only up;
  - **no focus** on a flat curve, its peak less than 5 % above its median, or a peak at the range's end; the stage goes back to its start, and `Focus::found` gives a `not_found` error saying why;
  - **the result,** `Focus`: where it found focus, where it started, and the curve, each position as the stage reported it and its measure.
- **`autofocus(rig, request, step)`** (P1.16b) runs the search on the instrument. Each position is measured on **the first frame of the live stream exposed after the stage reported itself still,** through the frame bus, so a moving stage never blurs a measurement. The depth of field comes from the objective fitted and the sensor's pitch. The stream is started for it if it wasn't running, and stopped after. The request names the measure, normalised variance by default, and the search's options; the result is the search's, with the measure, the objective and the time.
- **`scan`** (P1.17a, [ADR-0022](../../docs/adr/0022-scan-planning-and-the-scan-engine.md)) plans a scan and runs it stop and go:
  - **planners:** `Pattern::Snake`, a rectangle of x and y covered with tiles of the field of view at an overlap, the grid centred on it, in serpentine order; `ZRange` and `ZAround`, planes rising;
  - **`preview`:** the plan, its positions and its estimated time, from the stage's speeds and accelerations, the settle, the frames and their writing; nothing moves;
  - **`scan`:** for each position, a move, the stage's report that it is still, an autofocus if `AutofocusAt` says so, the first frame of the full-resolution stream exposed a settle (50 ms) after, and its position from the stage, checked unchanged through the exposure; frames are written by a thread of their own, two deep, into one dataset of tiles;
  - **`ScanControl`** pauses and resumes it between positions; a cancel keeps the tiles taken as a valid, partial dataset;
  - **a z-scan's dataset is a z-stack** (P1.17b): its planes one 3D array along z, their step in its scale.
- **`FocusProbe`** is what the search drives: a z move that returns once the stage is still, and a measure of an image exposed since. P1.16b's is the instrument; **`focus::synthetic::Synthetic`** is a stage with backlash and steps over a known curve with noise, for tests and examples.

## How to try it

```bash
cargo run --release -p tetera-api --features cuda --example capture
```

The example lives in `tetera-api`, as it goes through the `capture`, `jobs` and `datasets` Things.

```bash
cargo run --release -p tetera-workflows --example focus-search
```

`tetera-api`'s example `autofocus` runs it on the virtual instrument, 100 times at 5× and at 50×, and its example `scan` takes a 5 × 5 snake at 10×, its time against the plan's estimate, and a 21-plane z-stack.

`focus-search` takes each focus measure through a stack of frames from the virtual camera, H&E at 20× on the CPU, and says where each peaks against the section's middle. Then it runs the search 100 times at each objective's depth of field, on noisy curves through a stage with backlash and steps, and prints its errors. `-- --smoke` takes a short stack from a tiny sensor and 20 searches. The tests run each kind of capture on a small virtual camera and the fake one, a cancel at each stage, and bad requests; and the search on synthetic curves, 100 times on narrow peaks and 100 on broad ones, through a stage with backlash: `cargo test -p tetera-workflows`.
