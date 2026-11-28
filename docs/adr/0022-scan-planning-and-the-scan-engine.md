# ADR-0022: Scan planning and the scan engine

- Status: Accepted
- Date: 2026-11-28
- Phase: P1.17a

## Context

A scan takes many frames at many stage positions: a grid of tiles over an area of the slide, or a stack of planes through it. The user needs to see the plan before starting it, positions and time, and the scan must record where each frame was taken, survive a cancel with what it took, and pause without losing its place. M1's scans are stop and go: the stage stops for each frame. Raster scans, tiled z-stacks, the histo scan and stitching are M3's.

The facts that shape it:

- the camera's full-resolution frames come at 7.24 a second and are 40 MB each; writing one to OME-Zarr with zstd takes about as long as a tile's move and frame (P1.15c);
- the virtual stage, like grblHAL, moves along a straight line at the speed and acceleration of its most limited axis, and reports its position 50 times a second, 5 ms late;
- autofocus takes 26 positions, about 3 s (P1.16b).

## Decision

1. **A plan is a list of stage positions, made before the scan starts,** by a planner from the request, the field of view and the stage's capabilities. The plan carries its estimated time. Planners are pure functions, so the plan shown is the plan run.
2. **The XY snake** covers a rectangle with tiles of the field of view, each step the field less the overlap, the grid centred on the rectangle so its edges are covered. Rows run along x, alternately left to right and right to left, and advance along y from the rectangle's least y. Every tile is at the scan's z, or the last autofocus's.
3. **The z-scan** is a range and a step, or N planes a step apart centred on the current z, at the current x and y, rising so each plane is approached from below.
4. **Stop and go,** for each position:
   1. one move to it;
   2. the stage's report that it is still;
   3. autofocus, if the schedule says so;
   4. **the first frame of the full-resolution stream whose exposure starts a settle after the stage was still,** 50 ms unless asked otherwise. The camera is switched to full resolution once for the scan, and back after, as a capture does (D5), and streams meanwhile: a snap at full resolution took 280 to 385 ms on the virtual camera, its readout and its rendering one after the other, where the stream overlaps them at 7.24 frames a second;
   5. the position, recorded from the stage while still and checked unchanged after the frame: for stop and go, the stage's track at the exposure's midpoint;
   6. the frame handed to a writer thread, two frames deep, so writing overlaps the next move.
5. **The autofocus schedule:** none, the first tile, or every N tiles. A tile after an autofocus takes its z.
6. **The time estimate** follows the same model:
   - each move from the stage's capabilities, as its own planner would make it: a trapezoid or a triangle along the line, at the speed and acceleration of the most limited axis. The stage's capabilities now include each axis's acceleration;
   - per position: half the stage's report period and its latency; the settle; then half a frame interval on average to the stream's next exposure, the exposure, and its readout, a frame interval at full resolution. The interval is the mode's at the scan's bits: a camera mode now states its rate in 16-bit words as well as in 8 bits, half as fast on the MIChrome 20;
   - writing: the frame's bytes at a throughput, 120 MB/s unless asked otherwise, overlapped with the rest, so a tile takes the longer of the two;
   - autofocus: 26 positions, each a move of a depth of field, a report, and a live frame;
   - the mode switch there and back.
7. **Pause, resume and cancel** are heard between positions and during every wait. A paused scan keeps its place and the camera in its capture mode; resumed, it goes on. Cancelled, it finishes the dataset with the tiles taken, a valid, partial dataset, and ends cancelled; one cancelled before its first tile leaves none.
8. **A dataset of tiles** is one OME-Zarr array, a frame per tile along its first axis, each tile's recorded position in its frame's metadata, and the plan, the planned positions and the grid, in the dataset's metadata. The z-scan's 3D array is P1.17b's.

## Consequences

- **Gains:**
  - the plan shown is the plan run, and its time is a model the scan can be measured against;
  - a cancelled scan keeps what it took; a paused one, its place;
  - writing never holds the stage back while the disk keeps up.
- **Costs:**
  - the estimate needs the disk's throughput, which varies (P1.15c): a scan whose writes are slower than its tiles runs at the disk's pace, and the estimate errs by as much as the throughput assumed does;
  - stop and go spends most of a tile's time moving and settling.
- **To watch:**
  - the settle needed on the real stage, in M5;
  - the stream's next exposure is up to a frame interval after the settle: on average half of one, which the estimate assumes, so a short scan's time varies more than a long one's;
  - the real stage's repeatability against the positions recorded.

## Alternatives considered

- **Continuous scanning,** the stage moving through each exposure with a strobed light: faster, but the light and the stage's sync are M3's.
- **Planning tile by tile as the scan goes:** no plan to show, and no estimate.
- **Writing each frame before the next move:** simpler, but a full frame's write is as long as a tile's move and frame, so a scan would take nearly twice as long.
- **A dataset per tile:** thousands of folders for a large scan, and no single array to stitch from.
