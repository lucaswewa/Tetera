# tetera-acquire

Tetera's acquisition services, L4 (plan §6.2, [ADR-0008](../../docs/adr/0008-the-frame-model-the-buffer-pool-and-the-frame-bus.md)): every frame from the camera to whatever needs it, and the live view's work, before any network is involved.

- **`FrameBus`** fans each frame out, shared, never copied. It implements the HAL's `FrameSink`, so a camera streams into it.
  - **`lossless(capacity)`:** every frame, in order, from a bounded queue. A full queue holds the publisher until it has room, so a slow reader slows the camera rather than losing frames. The recorder (P1.15b) is one.
  - **`latest()`:** the newest frame not yet taken, from a one-frame mailbox. A new frame replaces one waiting, and the mailbox counts the frames replaced. The preview and the statistics are such.
  - **`first_after(t, timeout)`:** the first frame whose exposure starts at `t` or later, for workflows.

  The bus keeps no frame itself: a frame's buffer goes back to the camera's pool once its readers let it go.
- **`Session`** is a preview session.
  - It takes the newest frame from the bus, cropped to its region, at no more than its maximum rate. The frames that come while it works, or waits, give way to the newest, and the last frame is never lost.
  - It bins the frame down to its viewport, and a live recipe demosaics, resizes and colours it: `[bin →] demosaic → resize → colour`, which the pipeline fuses into one kernel on the GPU. Then it encodes a JPEG.
  - `PreviewParams`: the viewport, the region, the quality, the maximum rate and the white balance. They change mid-stream.
  - `Link`, the hook for a slow link (P1.14b), caps the quality and scales the viewport on top of them.
  - Each preview carries its frame's metadata and each stage's time: the frame to a mosaic, the live recipe, the JPEG. `SessionStats` keeps the rate and the mean times over the last second.
- **`statistics`** works out a frame's live statistics from its raw mosaic:
  - a histogram for each colour;
  - the saturation mask at the preview's size, and the share of sites saturated;
  - profiles along a row and down a column, 2 × 2 cell by cell;
  - each colour's mean, deviation, least and greatest in a region;
  - a focus value, Brenner's measure on green, larger when sharper.

  **`LiveStats`** works them out on a thread of its own, from a latest-wins subscriber, at most 10 times a second.

- **`Recorder`** writes every frame of the camera's stream into a frame log on the scratch disk (`tetera-store`, P1.15c), from a lossless subscriber, on a thread of its own. A full queue holds the camera rather than losing a frame; `RecordStats` counts the frames, their bytes and rate, the times the camera was held, and any missing by their numbers or as the camera's own drops. `stop` writes what is queued and finishes the log.

## How to try it

```bash
cargo run --release -p tetera-acquire --features cuda --example preview-pipeline
```

It streams the virtual camera, in its binned live mode at 8 bits, into the bus for 3 s, the stage jogging. Two preview sessions take frames from the bus, 1920 × 1280 and 640 × 427, and so do the live statistics. It prints the camera's rate, each session's rate, frames given way and stage times, and the statistics, and writes a few previews to `target/preview-pipeline/`. On this PC, each session keeps up with the camera's 53 frames a second: the frame to a mosaic 1.7 ms, the fused recipe 8 ms and the JPEG 3.4 ms at 1920 × 1280. `-- --smoke` runs a small sensor on the CPU for a second.

```bash
cargo run --release -p tetera-acquire --features cuda --example recorder
```

It records the virtual camera at full resolution, 12 bits, at its full rate, 7.24 frames a second and 289 MB/s, for 4 s, then ingests the frame log into a dataset as a compute job and reads the first frame back against the log's. On this PC every frame was kept, at 284 MB/s, and the camera was never held. `-- --smoke` runs a small sensor on the CPU.
