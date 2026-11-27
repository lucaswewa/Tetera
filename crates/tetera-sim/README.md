# tetera-sim

Tetera's simulated world, L2 (plan §8, [ADR-0014](../../docs/adr/0014-the-simulated-clock.md), [ADR-0015](../../docs/adr/0015-simulated-samples-on-the-gpu.md)): what stands in for the instrument when there is no hardware, on a clock that tests control.

- **`SimClock`** implements `Clock`, at one of three rates: real time, N× as fast, or stepped by hand. Simulated devices wait on it with `wait_until`, which a step wakes. Its UTC is its own, so it never drifts.
- **`Profile`:** a stage's move as a straight line from rest to rest, trapezoidal, or triangular when short, with each axis kept to its own acceleration and fastest speed (`AxisLimits`) and to the speed asked for. Its position and speed at any time come from closed forms, so the same times give the same positions, bit for bit. `hold` slows it to rest from wherever it has got to, as a feed hold does.

- **`VirtualStage`** implements `Stage` the way a grblHAL stage behaves, on a `SimClock`:
  - trapezoidal moves within each axis's limits;
  - grblHAL's states (Idle, Jog, Run, Hold, Home, Alarm);
  - soft limits that refuse targets beyond the travel;
  - homing that takes time: z first, then x and y, then locating the switches;
  - feed holds and jog cancels that slow to rest.

  It reports its position at a set rate (50 times a second by default), each report arriving late (5 ms), and keeps them in a track. `wait`, `stop` and `stop_jog` wait for a report taken after the last command, as a real grblHAL driver must. Faults on demand: an alarm, a hard limit, a stall. The same commands at the same times give the same track, bit for bit.

- **`HeSample`:** a procedural section of H&E-stained tissue, from a seed. It gives each stain's density, in optical density per µm, at any position and depth:
  - a tissue mask, regions of stroma, glands and fat;
  - nuclei as noisy ellipsoids at depths, with chromatin;
  - collagen in bundles and fibrils;
  - glands with their epithelium and lumens;
  - vessels with red cells, and fat cells;
  - a section 3 to 5 µm thick, tilted or folded.

  It is built from improved noise and cellular noise on integer hashes (`noise`), so the same seed gives the same bits. `nuclei` lists its nuclei, the ground truth. Colour is by absorbance: Ruifrok and Johnston's stain vectors and Beer–Lambert.
- **`render`:** a region at any pixel size and range of depth, each pixel averaging points at most 0.25 µm apart and summing its depth in 0.5 µm layers; on all cores, the same bits whole or in tiles. This is the CPU twin the GPU matches.
- **On the GPU,** with the `cuda` feature: `render_gpu` and `render_on` run `tetera-compute`'s `he.cu`, the same functions, within 2e-3 OD of the CPU twin. In practice it is within 6e-4, half precision's rounding. A 512 × 512 region at 50× takes about 2 ms on the RTX 3070 Ti Laptop GPU.
- **Tiles:** 256 × 256 pixels, keyed by level of detail (0.04 × 2^ℓ µm a pixel), column, row and depth (`TileKey`). `TileCache` keeps the most recently used within a budget of bytes; `HeTiles` keeps them on the GPU in half precision, each holding a grant from a VRAM budget until it goes.
- **`optics`:** the field the sensor sees ([ADR-0016](../../docs/adr/0016-the-virtual-cameras-model.md)), each pixel's irradiance in red, green and blue.
  - `FieldOfView`: the magnification, the 2.4 µm pixels and the camera's mounting angle.
  - `depth_groups`: the section in groups as thick as the depth of field, each blurred by its defocus and the Airy disc.
  - Vignetting, and a `Light` with its intensity, its colour temperature by Planck's law, and its unevenness.

  `field` makes it on the CPU from a rendered region; `HeTiles::field_with` on the GPU, from the tile cache, within 4e-5 of the light. A full 5472 × 3648 field, still, takes 9 ms at 5×, 19 ms at 20× and 38 ms at 50× on the RTX 3070 Ti Laptop GPU.
- **`VirtualIllumination`** implements `Illumination`: a transmitted channel, dimmable, from 2,700 to 6,500 K, which becomes the field's `Light`.

The sensor and the virtual camera join in P1.12b.

## How to try it

```bash
cargo run -p tetera-sim --example sim-clock
cargo run -p tetera-sim --example virtual-stage
cargo run --release -p tetera-sim --example he-sample
cargo run --release -p tetera-sim --example virtual-optics
```

`sim-clock` runs the clock for 100 ms of the host's time at 1× and 10×, wakes a waiting thread with five steps of 10 ms, then samples a move from the origin to (2000, 1000, 300) µm every 20 ms of stepped time, and holds it at 100 ms. `-- --smoke` waits 25 ms instead.

`virtual-stage` moves the virtual stage in stepped time, printing every 50 ms where the stage is and its latest report, which lags. Then, at 1× (10× with `-- --smoke`), through the `Stage` trait: a move, a jog, homing, a stall, a hard limit and a target beyond its soft limits, and every event.

`he-sample` renders the sample from seed 7 (or the seed given) as PNGs in `target/he-sample/`: a 2 mm overview at 2 µm; fields at the sampling of the camera's 2.4 µm pixels through 5×, 20× and 50×; three depth slices through a nucleus at 50×; and a fold at 5×. Each with its time. `-- --smoke` renders them small. With `--features cuda` and a GPU, it renders each view on the GPU too, timed and compared with the CPU's, and writes it as `-gpu.png`. It then moves a field of 3 × 3 tiles across the slide through a cache of 12, saying what it made, found and let go.

`virtual-optics` writes the field the sensor sees as PNGs in `target/virtual-optics/`. On a 912 × 608 sensor on the CPU: at 5×, 20× and 50×; through focus at 50×; and under a cool, uneven light at 20×. With `--features cuda` and a GPU, it also times the full 5472 × 3648 field at each objective, cold, still and moving 100 pixels a frame, against the 40 ms budget. `-- --smoke` makes fewer and smaller fields.
