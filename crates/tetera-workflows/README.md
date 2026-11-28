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

## How to try it

```bash
cargo run --release -p tetera-api --features cuda --example capture
```

The example lives in `tetera-api`, as it goes through the `capture`, `jobs` and `datasets` Things. The tests run each kind of capture on a small virtual camera, a cancel at each stage, and bad requests: `cargo test -p tetera-workflows`.
