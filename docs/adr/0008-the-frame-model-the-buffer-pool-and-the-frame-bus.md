# ADR-0008: The frame model, the buffer pool and the frame bus

- Status: Accepted
- Date: 2026-11-26
- Phase: P1.03

## Context

- **A full MIChrome 20 frame is about 40 MB** in 16-bit words: 5472 × 3648 sites. At 7.24 fps that is 289 MB/s. Allocating and freeing that memory for every frame would fragment the heap and stall the camera.
- **One frame has many readers:** the recorder, the live preview, statistics, autofocus, the live mosaic. Copying 40 MB to each is out of the question.
- **The raw frame is a Bayer mosaic.** Cropping it at an odd column or row changes which colour comes first. Get that wrong and every later step mixes red and green.
- **The camera's significant bits** (12 on the MIChrome 20) differ from the word size (16), and the plan keeps the raw frame reversible (D4).
- **Some memory is special:** CUDA's pinned host memory makes copies to the GPU fast (P1.04). DMA wants buffers on page boundaries.
- **Readers differ in what they can afford to miss:** the recorder must keep every frame; the preview must always show the newest and may skip the rest.

## Decision

1. **`PixelFormat`** names how pixels are stored: `Bayer8`, `Bayer16` with its pattern and significant bits, `Mono8`, `Mono16`, `Rgb8`, `Rgb16`, `Rgba8` and `PlanarF32`. A Bayer format carries its `BayerPattern`, and `cropped(x, y)` shifts it for an odd offset.
2. **`FrameLayout`** is a format, a width, a height and a stride. Every row starts on a multiple of 64 bytes. It works out the region of a crop and the size of a binning; a Bayer mosaic bins to an even size, so its tiles stay whole.
3. **`FrameMeta`** holds:
   - the sequence number;
   - the first row's exposure start and end, and the line time of a rolling shutter, so `row_midpoint` places any row in time;
   - exposure and gain, mode, binning, ROI, black level;
   - the objective, the light's state, the stage's position at the midpoint;
   - the frames dropped before this one.

   Durations serialise as milliseconds (ADR-0006).
4. **`FramePool`** holds a fixed number of buffers of one layout, all allocated when it is made, from an `Allocator`. Host memory is the first; pinned memory comes in P1.04. Buffers start on 4 KiB boundaries without unsafe code: each starts inside a slightly larger allocation.
5. **A frame's life:**
   - `take` lends a buffer as a `FrameMut`, waiting up to a timeout when all are lent, and failing then with a `Timeout` error that says so;
   - the camera fills it, and `freeze` makes it a `Frame`;
   - frames share their buffer by reference count;
   - when the last share drops, the buffer goes back to the pool, and a waiting `take` gets it;
   - a frame that outlives its pool frees its buffer.

   The pool counts its allocations, takes, waits and misses.
6. **Views are safe:** rows without padding, planes, samples read and written little-endian, Bayer sites. `to_full_range` and `from_full_range` scale between significant bits and 16 bits, and give back every value.
7. **The frame bus,** decided here and built in P1.14a, fans each `Frame` out to two kinds of subscriber:
   - **lossless,** with a bounded queue: a full queue slows the publisher. The recorder is one;
   - **latest-wins,** with a one-frame mailbox: a new frame replaces one not yet taken. The preview, statistics and the live mosaic are such.

   Workflows also ask for "the first frame exposed after t".

## Consequences

- **Gains:**
  - a running camera allocates nothing;
  - a frame reaches every reader without a copy;
  - a crop can't mislabel colours;
  - the pool's size bounds the memory frames use;
  - a slow preview can't hold up the recorder, and a stalled recorder shows as backpressure, not as lost frames.
- **Costs:**
  - a frame is read-only once shared: processing writes into a new frame;
  - the pool must be sized for the slowest reader's backlog;
  - views copy samples out of bytes rather than reinterpreting memory.
- **To watch:**
  - the pool's waits under the live view (P1.14a), which say whether 16 frames are enough;
  - the cost of the safe sample views in hot loops, where P1.05's kernels will read rows directly.

## Alternatives considered

- **A new allocation per frame:** simple, but 40 MB at camera rate fragments memory and stalls.
- **Copying frames to each reader:** 289 MB/s times the number of readers.
- **Reinterpreting bytes as `u16` with `unsafe` or `bytemuck`:** faster views, but unsafe code in L0, or a dependency for what `from_le_bytes` does.
- **A growable pool:** no wait when readers fall behind, but no bound on memory either.
- **One kind of subscriber:** either the preview stalls the recorder, or the recorder drops frames.
