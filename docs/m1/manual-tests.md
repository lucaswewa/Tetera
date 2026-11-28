# Milestone 1: manual tests

How to try each phase by hand, and each checkpoint's demo. The agent runs each phase's entry and records the result in the phase's notes; the owner runs each checkpoint's (ADR-0001).

Every entry runs in WSL, in `~/repos/Tetera`, on the phase's branch.

## P1.01: Workspace and gates

- **Prerequisites:** the toolchain of [docs/setup/toolchain.md](../setup/toolchain.md); a fresh clone of the phase's branch.
- [ ] `cargo xtask doctor` — expect every line needed now to say `ok`, and the last line `Everything needed now is here.`
- [ ] `cargo xtask layering` — expect `The layer rules hold.`
- [ ] `cargo xtask gate quick` — expect five steps to pass, within 90 seconds once built, and a `summary:` line naming `target/gates/<time>-quick/summary.md`.
- [ ] `cargo xtask gate phase` — expect `fmt`, `layering`, `clippy`, `tests`, `smoke tetera-core/hello`, `docs`, `deny` and `stress` to pass, and `cuda` and `python` to be skipped with their reasons.
- [ ] Open the phase gate's `summary.md` — expect what the change reaches and why, and each step with its result, time and reason.
- [ ] Add `tetera-hal`, a crate with `[lints] workspace = true`, and make `tetera-core` depend on it; run `cargo xtask gate quick` — expect `layering` to fail with ``tetera-core` (L0) depends on `tetera-hal` (L2), a higher layer.`. Undo it.
- [ ] Remove `[lints]` and `workspace = true` from `crates/tetera-core/Cargo.toml`; run `cargo xtask layering` — expect ``tetera-core` must take the workspace's lints…`. Undo it.
- [ ] Commit, then `cargo xtask gate checkpoint` — expect one `fresh clone` step to pass, naming the clone's own summary.
- [ ] In VS Code over Remote-WSL, set a breakpoint in `crates/tetera-core/examples/hello.rs`, choose **Debug an example**, accept `tetera-core` and `hello`, and press F5 — expect it to stop at the breakpoint.
- **Result:** see the [notes](notes/P1.01-workspace-and-gates.md#manual-test).

## P1.02: Core types and logging

- **Prerequisites:** the phase's branch, built.
- [ ] `cargo run -p tetera-core --example core_tour` — expect:
  - 25,000 µm of travel as 10,417 sensor pixels;
  - pixel (5472, 3648) at about (1261.118, 2177.389) µm, and back to (5472, 3648) px;
  - the ROI clipped to 472 × 648;
  - a 5000.0 µm move;
  - the clock's UTC time, today, and its drift within the limit;
  - two IDs in order;
  - `capturing a burst: setting the exposure: the camera is streaming (kind busy)`.
- [ ] `cargo run -p tetera-log --example log_tour` — expect six coloured lines on standard error, then the ring's six records from `trace` to `critical`, and the JSON file's path. That file holds six JSON lines, the last with `"level":"critical"`.
- [ ] `cargo run -p tetera-log --example log_tour -- --journald`, then `journalctl -t tetera-log-tour -o json -n 6` — expect priorities 7, 7, 6, 4, 3 and **2** for the critical record, with `FREE_GB=0` and the other fields. Nothing on standard error, since journald has the records.
- [ ] `RUST_LOG=warn cargo run -p tetera-log --example log_tour` — expect only the warn, error and critical records in the ring (the smoke check would fail with this filter, so don't pass `--smoke`).
- **Result:** see the [notes](notes/P1.02-core-types-and-logging.md#manual-test).

## P1.03: The frame model

- **Prerequisites:** the phase's branch, built; about 1 GB of free memory.
- [ ] `cargo run -p tetera-core --example frame_pool` — expect:
  - `39.9 MB each, 639 MB in all`;
  - lines from three consumers reading frames 0, 12, 24 and 36, with row means of 0, 3084, 2056 and 1028 (the frame's level × 257);
  - `48 taken, 16 allocated`;
  - `With all 16 lent, a 17th: all 16 frames of the pool were still in use after 200 ms.`;
  - `Once they come back, a frame is lent at once.`
- [ ] Watch its memory with `/usr/bin/time -v cargo run -q -p tetera-core --example frame_pool` — expect a peak near 640 MB however many frames pass through: the pool doesn't grow.
- **Result:** see the [notes](notes/P1.03-the-frame-model.md#manual-test).

## P1.04a: Compute backends

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL ([toolchain.md](../setup/toolchain.md#the-cuda-toolkit-from-p104a)).
- [ ] `cargo run -p tetera-compute --example compute_info --features cuda` — expect:
  - `Backend: CUDA: CUDA on NVIDIA GeForce RTX 3070 Ti Laptop GPU, device 0`;
  - compute capability 8.6, 8.0 GiB, and the driver running CUDA up to 13.5;
  - `works` for pinned memory, cuFFT, NVRTC, nvJPEG and IPC memory handles.
- [ ] `cargo run -p tetera-compute --example compute_info` — expect `Backend: CPU: this build has no `cuda` feature`, and each feature not tried.
- [ ] `TETERA_FORCE_CPU=1 cargo run -p tetera-compute --example compute_info --features cuda` — expect `Backend: CPU: TETERA_FORCE_CPU is set`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-compute --features cuda -- --nocapture` — expect every test to pass, the feature table printed, and no test saying `skipped`.
- [ ] `cargo xtask doctor` — expect `nvcc` and `nvidia-smi` to say `ok`.
- **Result:** see the [notes](notes/P1.04a-compute-backends.md#manual-test).

## P1.04b: Kernels and the twin harness

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL ([toolchain.md](../setup/toolchain.md#the-cuda-toolkit-from-p104a)).
- [ ] `cargo run --release -p tetera-compute --example twin_scale --features cuda` — expect:
  - `Backend: CUDA: CUDA on NVIDIA GeForce RTX 3070 Ti Laptop GPU, device 0`;
  - a full frame, 19,961,856 samples, timed on the CPU and in the kernel, `from nvcc's fat binary`;
  - `Twins:   19961856 values, 0 beyond the tolerance; identical`.
- [ ] `NVCC=none cargo run --release -p tetera-compute --example twin_scale --features cuda` — expect a build warning that `nvcc wasn't found`, `from NVRTC's PTX`, and the twins still identical.
- [ ] `cuobjdump --list-elf --list-ptx $(ls -t target/release/build/tetera-compute-*/out/scale.fatbin | head -1)` — expect cubins for `sm_75`, `sm_80`, `sm_90`, `sm_100`, `sm_110` and `sm_120`, and PTX for `sm_120`.
- [ ] `cargo run -p tetera-compute --example twin_scale -- --smoke` — expect `Backend: CPU: this build has no `cuda` feature`, the CPU's time and mean, and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-compute --features cuda` — expect every test to pass, with none saying `skipped`.
- [ ] `cargo test -p tetera-compute budget` — expect the budget tests to pass, among them a request beyond a budget refused as `Busy`.
- **Result:** see the [notes](notes/P1.04b-kernels-and-the-twin-harness.md#manual-test).

## P1.05a: Bayer and demosaic

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL ([toolchain.md](../setup/toolchain.md#the-cuda-toolkit-from-p104a)).
- [ ] `cargo run --release -p tetera-imaging --example demosaic --features cuda` — expect:
  - `Backend: CUDA: CUDA on NVIDIA GeForce RTX 3070 Ti Laptop GPU, device 0`, and a 5472 × 3648 frame;
  - a time on the CPU and on CUDA for the superpixel, bilinear and Malvar-He-Cutler;
  - Malvar-He-Cutler's PSNR above bilinear's, and `twins identical` for both.
- [ ] Open `target/demosaic/pattern.ppm`, `bilinear.ppm` and `malvar-he-cutler.ppm` in an image viewer — expect the zone plate's rings in each, bilinear's the softer.
- [ ] `cargo run -p tetera-imaging --example demosaic -- --smoke --out target/demosaic-smoke` — expect `CPU: this build has no `cuda` feature`, `-` for CUDA, the PSNRs, and `smoke: ok`. Its files hold the whole 640 × 480 frame: hue ramps, the zone plate, and colour bars.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda` — expect every test to pass, with none saying `skipped`.
- **Result:** see the [notes](notes/P1.05a-bayer-and-demosaic.md#manual-test).

## P1.05b: Binning, resize, display transform and JPEG

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, `cmake` and `nasm` in WSL ([toolchain.md](../setup/toolchain.md#in-wsl-ubuntu-2404)).
- [ ] `cargo xtask doctor` — expect `cmake` and `nasm` to say `ok`.
- [ ] `cargo run --release -p tetera-imaging --example demosaic --features cuda` — expect, after P1.05a's table, the preview's:
  - a time on the CPU and on CUDA for binning, demosaicking, resizing and the display transform;
  - JPEG at 10 ms or less;
  - `Twins:   the 8-bit previews identical`.
- [ ] Open `target/demosaic/preview.jpg` — expect 1920 × 1280: the hue ramps, the zone plate, and the colour bars, brighter than linear (sRGB). The zone plate's finest rings fade to grey, without moiré.
- [ ] `cargo run -p tetera-imaging --example demosaic -- --smoke` — expect `-` for every CUDA time, a JPEG's size, and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda` — expect every test to pass, with none saying `skipped`.
- **Result:** see the [notes](notes/P1.05b-binning-resize-display-transform-and-jpeg.md#manual-test).

## P1.06a: Dark, flat and colour

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, `cmake` and `nasm` in WSL.
- [ ] `cargo run --release -p tetera-imaging --example isp --features cuda` — expect, for `full` (5472 × 3648) and `binned` (2736 × 1824):
  - each stage's time on the CPU and on CUDA, `-` for reducing a full frame;
  - `Twins:      the previews identical`.
- [ ] Open `target/isp/isp-full.jpg` and `isp-binned.jpg` — expect the test pattern: the white bar grey, not tinted, and the corners as bright as the centre.
- [ ] `cargo run -p tetera-imaging --example isp -- --smoke` — expect `-` for every CUDA time and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda` — expect every test to pass, with none saying `skipped`.
- **Result:** see the [notes](notes/P1.06a-dark-flat-and-colour.md#manual-test).

## P1.06b: The fused kernel

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, `cmake` and `nasm` in WSL.
- [ ] `cargo run --release -p tetera-imaging --example isp --features cuda` — expect, for `full` and `binned`, a `Fused` row with the fused kernel's time, 10 ms or less for `full`, and `Twins:      the previews identical, fused and unfused`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda fused` — expect the fused kernel to match the chain over 60 random cases and from full frames.
- [ ] `NVCC=none TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda fused` — expect the same through NVRTC, which compiles `fused.cu` with its header inlined.
- **Result:** see the [notes](notes/P1.06b-the-fused-kernel.md#manual-test).

## P1.07a: Denoise

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, `cmake` and `nasm` in WSL.
- [ ] `cargo run --release -p tetera-imaging --example denoise --features cuda` — expect:
  - the temporal mean of 8 raising the PSNR by about 9 dB;
  - the bilateral filter raising it too;
  - a time for each on the CPU and on CUDA, with no error, since each CUDA result is checked against the CPU's.
- [ ] `cargo run -p tetera-imaging --example denoise -- --smoke` — expect `-` for CUDA and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda denoise` — expect every test to pass.
- **Result:** see the [notes](notes/P1.07a-denoise.md#manual-test).

## P1.07b: PSFs and Richardson–Lucy

- **Prerequisites:** the phase's branch; `cmake` and `nasm` in WSL. No GPU is needed.
- [ ] `cargo run --release -p tetera-imaging --example deconvolve` — expect:
  - the RMS error against the specimen falling over the first iterations, plain and with total variation, then the plain one rising again;
  - a least error below the image's, for each, and for the 3D stack;
  - `specimen.jpg`, `image.jpg` and `deconvolved.jpg` in `target/deconvolve`, the last with sharper beads and discs than the image.
- [ ] `cargo run -p tetera-imaging --example deconvolve -- --smoke` — expect `smoke: ok`.
- [ ] `cargo test -p tetera-imaging -- psf:: deconvolve::` — expect every test to pass.
- **Result:** see the [notes](notes/P1.07b-psfs-and-richardson-lucy.md#manual-test).

## P1.07c: Richardson–Lucy on the GPU

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, with cuFFT, `cmake` and `nasm` in WSL.
- [ ] `cargo run --release -p tetera-imaging --example deconvolve --features cuda` — expect:
  - P1.07b's table on the CPU;
  - then the 2D and 3D cases on CUDA, each faster than on the CPU, with a relative RMS difference far below 1e-3, and no error, since the example checks the spec's tolerance;
  - then 20 iterations at 5,472 × 3,648 timed on CUDA.
- [ ] `cargo run -p tetera-imaging --example deconvolve -- --smoke` — expect `CUDA: not in this build` and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda deconvolve` and the same for `tetera-compute` — expect every test to pass.
- **Result:** see the [notes](notes/P1.07c-richardson-lucy-on-the-gpu.md#manual-test).

## P1.08a: The pipeline model

- **Prerequisites:** the phase's branch.
- [ ] `cargo run --release -p tetera-pipeline --example recipe` — expect:
  - the registry's two nodes, `tetera.crop` and `tetera.statistics`;
  - `broken on purpose` refused with six problems, each naming where it is: bad parameters, an unknown node, an unfed input, a cycle, a type mismatch, an output from nowhere;
  - `crop and measure` run, crop before measure, with each channel's statistics.
- [ ] `cargo run -p tetera-pipeline --example recipe -- crates/tetera-pipeline/recipes/broken.json` — expect the six problems and an error exit; with `crop-and-measure.json`, its order.
- [ ] `cargo test -p tetera-pipeline` — expect every test to pass.
- **Result:** see the [notes](notes/P1.08a-the-pipeline-model.md#manual-test).

## P1.08b: The raw path's nodes

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, `cmake` and `nasm` in WSL.
- [ ] `cargo run --release -p tetera-pipeline --example recipe --features cuda` — expect, after P1.08a's output:
  - the live chain's nodes in order, from `less_dark` to `jpeg`;
  - a time on the CPU and on CUDA, each saying the preview is the ISP's own, exactly, with no error;
  - `target/recipe/preview.jpg`: the test pattern, its corners as bright as its centre, its colours balanced.
- [ ] `cargo run -p tetera-pipeline --example recipe -- --smoke` — expect the live chain on the CPU alone and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-pipeline --features cuda` — expect every test to pass.
- **Result:** see the [notes](notes/P1.08b-the-raw-paths-nodes.md#manual-test).

## P1.08c: GPU residency and fusion

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, `cmake` and `nasm` in WSL.
- [ ] `cargo run --release -p tetera-pipeline --example recipe --features cuda` — expect, for the live chain:
  - the fused kernel taking the place of `less_dark`, `even`, `rgb`, `small` and `colour`;
  - three timed rows, the CPU, CUDA node by node and CUDA fused, CUDA well under the CPU and fused a little under node by node;
  - on CUDA, 1 up and 1 down a run, about 47 MB, and each preview the ISP's own, with no error.
- [ ] `cargo run -p tetera-pipeline --example recipe -- --smoke` — expect the CPU row alone and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-pipeline --features cuda` — expect every test to pass.
- **Result:** see the [notes](notes/P1.08c-gpu-residency-and-fusion.md#manual-test).

## P1.08d: Denoise and deconvolution nodes, and live runs

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, `cmake` and `nasm` in WSL.
- [ ] `cargo run --release -p tetera-pipeline --example recipe --features cuda` — expect, last, a live run of 29 frames at 14.5 fps on each backend: on CUDA, node by node and fused, every frame run at 14.5 fps; on the CPU, about half giving way to a later one; no errors.
- [ ] `cargo run -p tetera-pipeline --example recipe -- --smoke` — expect the live run on the CPU alone and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-pipeline --features cuda` — expect every test to pass.
- **Result:** see the [notes](notes/P1.08d-denoise-and-deconvolution-nodes-and-live-runs.md#manual-test).

## P1.08e: Batch runs and tiling

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3, `cmake` and `nasm` in WSL.
- [ ] `cargo run --release -p tetera-pipeline --example recipe --features cuda` — expect, last, four frames written to `target/recipe/frames` and read back, the `denoise` recipe's reach, and a time whole and in tiles of 1,024 × 1,024, the two the same exactly, with no error.
- [ ] `cargo run -p tetera-pipeline --example recipe -- --smoke` — expect two small frames, whole and tiled the same, and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-pipeline --features cuda tile::` — expect every test to pass.
- **Result:** see the [notes](notes/P1.08e-batch-runs-and-tiling.md#manual-test).

## P1.09a: The camera

- **Prerequisites:** the phase's branch.
- [ ] `cargo run -p tetera-hal --example hal-fakes` — expect the fake camera's two modes; its configuration as JSON; a snap of 640 × 480; about 28 frames streamed in a second, with 2 dropped on purpose; 3 frames for 3 software triggers; a snap failing once disconnected; and its events in order.
- [ ] `cargo run -p tetera-hal --example hal-fakes -- --smoke` — expect `smoke: ok`.
- [ ] `cargo test -p tetera-hal` — expect 20 tests to pass.
- **Result:** see the [notes](notes/P1.09a-the-camera.md#manual-test).

## P1.09b: The stage, the objectives and the light

- **Prerequisites:** the phase's branch.
- [ ] `cargo run -p tetera-hal --example hal-fakes` — after the camera's part, expect the fake stage's axes; a move of about 0.2 s, read partway along its line; a move by −1 mm; a jog of about 250 µm; a move beyond the travel refused; a stall halfway into alarm, then homed; its events in order; the four Mitutoyo objectives from `configs/objectives.toml`, the 20× fitted and a 40× refused; and the light as set, an intensity of 1.5 refused.
- [ ] `cargo run -p tetera-hal --example hal-fakes -- --smoke` — expect `smoke: ok`.
- [ ] `cargo test -p tetera-hal` — expect 52 tests to pass.
- **Result:** see the [notes](notes/P1.09b-the-stage-the-objectives-and-the-light.md#manual-test).

## P1.10a: The simulated clock and motion profiles

- **Prerequisites:** the phase's branch.
- [ ] `cargo run -p tetera-sim --example sim-clock` — expect about 100 ms on the clock for 100 ms of the host's at 1×, and about 1,000 ms at 10×; a thread woken at 50 ms by five steps of 10 ms; a move of 2,256 µm in 300 ms, sampled every 20 ms: speeding up, cruising at 11.28 mm/s, slowing, and at (2000, 1000, 300) µm at the end; held at 100 ms, at rest at (1000, 500, 150) µm.
- [ ] `cargo run -p tetera-sim --example sim-clock -- --smoke` — expect `smoke: ok`.
- [ ] `cargo test -p tetera-sim` — expect 15 tests to pass.
- **Result:** see the [notes](notes/P1.10a-the-simulated-clock-and-motion-profiles.md#manual-test).

## P1.10b: The virtual stage

- **Prerequisites:** the phase's branch.
- [ ] `cargo run -p tetera-sim --example virtual-stage` — expect, in stepped time, a move to (5000, 2000, 500) µm every 50 ms: Run until 350 ms, then Idle, its latest report up to 25 ms behind the stage itself. Then at 1×: the same move in about 365 ms; a jog stopped after 200 ms; homing in about 1.9 s; a stall halfway to x = 8 mm, in alarm; a hard limit while jogging; a target beyond the soft limits refused; and its events in order.
- [ ] `cargo run -p tetera-sim --example virtual-stage -- --smoke` — expect the second part at 10×, and `smoke: ok`.
- [ ] `cargo test -p tetera-sim` — expect 31 tests to pass.
- **Result:** see the [notes](notes/P1.10b-the-virtual-stage.md#manual-test).

## P1.11a: The H&E sample, on the CPU

- **Prerequisites:** the phase's branch.
- [ ] `cargo run --release -p tetera-sim --example he-sample` — expect seven PNGs in `target/he-sample/`, each with its time: a 2 mm overview, fields at 5×, 20× and 50×, three depth slices through a nucleus, the middle one darkest, and a fold at 5×. Look at each: tissue with bare glass between; stroma with nuclei; glands with lumens; vessels with red cells; fat; at 50×, collagen fibrils and nuclei with chromatin.
- [ ] `cargo run --release -p tetera-sim --example he-sample -- --smoke` — expect `smoke: ok`.
- [ ] `cargo test -p tetera-sim` — expect 53 tests to pass.
- **Result:** see the [notes](notes/P1.11a-the-he-sample-on-the-cpu.md#manual-test).

## P1.11b: The H&E sample on the GPU, and its tiles

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL, and the GPU.
- [ ] `cargo run --release -p tetera-sim --features cuda --example he-sample` — expect each view on the GPU too, about 9 to 19 times as fast as on the CPU, within about 5e-4 OD of it, each written as `-gpu.png` beside the CPU's; then a field of 3 × 3 tiles: 9 made, then 9 found, then moved a column at a time, 3 made each time and, once the cache of 12 is full, 3 let go. Compare a `-gpu.png` with its CPU twin: the same image.
- [ ] `cargo run --release -p tetera-sim --features cuda --example he-sample -- --smoke` — expect `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-sim --features cuda` — expect 63 tests to pass.
- **Result:** see the [notes](notes/P1.11b-the-he-sample-on-the-gpu-and-its-tiles.md#manual-test).

## P1.12a: Optics and light: the field the sensor sees

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL, and the GPU.
- [ ] `cargo run --release -p tetera-sim --features cuda --example virtual-optics` — expect the field at 5×, 20× and 50× with one, two and three depth groups; through focus at 50×, sharpest at 2 µm and a blur at 10; under a cool, uneven light; then the full 5472 × 3648 field on the GPU at each objective, cold, still and moving, still within 40 ms. Look at the PNGs in `target/virtual-optics/`: nuclei soft at 50×, as its NA allows; a warm cast from the uncorrected 3,200 K light.
- [ ] `cargo run --release -p tetera-sim --example virtual-optics -- --smoke` — expect `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-sim --features cuda` — expect 77 tests to pass.
- **Result:** see the [notes](notes/P1.12a-optics-and-light-the-field-the-sensor-sees.md#manual-test).

## P1.12b: The sensor and motion

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL, and the GPU.
- [ ] `cargo run --release -p tetera-sim --features cuda --example virtual-sensor` — expect raw frames at 20× still, in 8 bits, drifting, skewed by the rolling shutter (about 190 pixels top to bottom) and dim at a gain of 16; the noise on an even field within a few per cent of the model's mean and variance at each level of light; then the full 5472 × 3648 raw frame on the GPU, still and moving, within 40 ms, and the GPU's sensor differing from the CPU's in a handful of counts of 20 million. Look at the PNGs in `target/virtual-sensor/`: the moving frame smeared along x, the skewed one sheared, the dim one grainy.
- [ ] `cargo run --release -p tetera-sim --example virtual-sensor -- --smoke` — expect `smoke: ok`.
- [ ] `cargo run --release -p tetera-sim --features cuda --example virtual-optics` — expect the full field at 50× within 40 ms moving as well as still.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-sim -p tetera-compute --features tetera-sim/cuda` — expect 93 tests to pass in `tetera-sim` and 91 in `tetera-compute`.
- **Result:** see the [notes](notes/P1.12b-the-sensor-and-motion.md#manual-test).

## P1.12c: The virtual camera

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL, and the GPU.
- [ ] `cargo run --release -p tetera-sim --features cuda --example virtual-camera` — expect the virtual MIChrome 20 on the GPU, 5472 × 3648, live in mode 0 at 14.49 fps while the stage jogs at 1 mm/s, with none or few frames dropped; then jogging at 100 to 1,000 µm/s, each frame made within 40 ms up to 250 µm/s; then each mode at each bit depth at its rate (7.24 and 14.49; 26.57 and 53.14; 33.87 and 67.26 fps), none dropped, each still frame made within 40 ms. Look at the JPEGs in `target/virtual-camera/`: H&E at 20×, smeared along the jog and sheared by the rolling shutter.
- [ ] `cargo run --release -p tetera-sim --example virtual-camera -- --smoke` — expect each mode at its rate on a small sensor on the CPU, and `smoke: ok`.
- [ ] `TETERA_CUDA=1 cargo test -p tetera-sim --features cuda -- camera::` — expect the camera's tests to pass, the contract suite on the CPU and the GPU among them.
- **Result:** see the [notes](notes/P1.12c-the-virtual-camera.md#manual-test).

## Checkpoint A: the virtual instrument renders H&E live

The owner runs this at Checkpoint C, with B's and C's; the agent ran and recorded it at P1.12c, under the owner's standing approval.

- **Prerequisites:** `main` after P1.12c; the CUDA toolkit 13.3 in WSL, and the GPU.
- [ ] `cargo xtask gate checkpoint` — expect G2 to pass on a fresh clone: every suite, every example's smoke run, and the CUDA steps.
- [ ] `cargo run --release -p tetera-sim --features cuda --example he-sample` — expect the H&E sample's PNGs in `target/he-sample/`. Judge them plausible: tissue, stroma, glands, nuclei with chromatin, red cells in vessels, at 5×, 20× and 50×.
- [ ] `cargo run --release -p tetera-sim --features cuda --example virtual-optics` — expect the field through each objective and through focus, sharpest at 2 µm, with the full field on the GPU within 40 ms still and moving.
- [ ] `cargo run --release -p tetera-sim --features cuda --example virtual-sensor` — expect raw frames still, smeared, skewed and grainy, and the noise against the model.
- [ ] `cargo run --release -p tetera-sim --features cuda --example virtual-camera` — expect live H&E at the camera's rates while the stage jogs, as in P1.12c's entry. Look at the JPEG strip.
- **Result:** see the [notes](notes/P1.12c-the-virtual-camera.md#checkpoint-a).
## P1.13a: The server, configuration and systemd

- **Prerequisites:** the phase's branch; systemd in WSL (`/etc/wsl.conf`); the GPU for the full sensor.
- [ ] `cargo run --release -p tetera-server` — expect `serving url=http://127.0.0.1:8310 backend=cuda sensor=[5472, 3648]`, then teta-wot's `serving 2 Things`.
- [ ] `curl http://127.0.0.1:8310/system/platform`, then `system/compute`, `system/memory` and `system/disks` — expect the kernel, Ubuntu, WSL's version; the RTX 3070 Ti and the full sensor; the memory; the data and scratch roots' disk.
- [ ] `curl http://127.0.0.1:8310/simulation/sample`, then `POST simulation/load` with `{"seed": 8}`, `POST simulation/unload`, and `POST simulation/inject` with `{"imperfection": "stage_alarm"}` — expect seed 7, then 8, then `null`; each invocation `completed`.
- [ ] On Windows, open <http://localhost:8310/docs> — expect the interactive docs, with `system` and `simulation`.
- [ ] `cargo run -p tetera-server -- --check`, and again with a configuration that has `prot = 1` under `[server]` — expect the configuration printed, then `unknown field` and `prot`, exit code 2.
- [ ] Install the unit as a user unit, as `packaging/linux/README.md` says; `systemctl --user start tetera`; `systemctl --user kill -s KILL tetera` — expect it active again within a few seconds, with a new PID and one restart. `systemctl --user stop tetera` — expect it to stop cleanly, and stay down.
- [ ] `cargo test -p tetera-api -p tetera-server` — expect 15 and 6 tests to pass.
- **Result:** see the [notes](notes/P1.13a-the-server-configuration-and-systemd.md#manual-test).

## P1.13b: The device Things

- **Prerequisites:** the phase's branch; the GPU for the full sensor.
- [ ] `cargo run --release -p tetera-server`, then <http://localhost:8310/docs> on Windows — expect `camera`, `stage`, `objective` and `illumination` beside `system` and `simulation`.
- [ ] Follow `crates/tetera-api/README.md`'s `curl` walkthrough — expect the exposure and mode written and read back; a stream that counts frames and stops; the 50× fitted; the light dimmed and warmed.
- [ ] Its timed move: 10 mm along x at 20 mm/s, with `curl -N` watching `stage/position` — expect `completed` after about 0.6 s (`timeStarted` to `timeCompleted`), and positions streaming in as it goes.
- [ ] `POST simulation/inject` with `{"imperfection": "stage_alarm"}`, then a `stage/move_by` — expect the invocation to end `error`, led by `hardware`; `stage/home` clears it.
- [ ] `cargo test -p tetera-api -p tetera-server` — expect 25 and 10 tests to pass.
- **Result:** see the [notes](notes/P1.13b-the-device-things.md#manual-test).

## P1.14a: The frame bus and the preview pipeline

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL, and the GPU.
- [ ] `cargo run --release -p tetera-acquire --features cuda --example preview-pipeline` — expect the camera in mode 1 at 8 bits near its 53 fps; both sessions, 1920 × 1280 and 640 × 427, at the camera's rate, 30 a second or more, each with its stages' times (the frame to a mosaic, the live recipe, the JPEG); the statistics at about 8 to 10 a second. Look at the JPEGs in `target/preview-pipeline/`: H&E at 20×, glass white, the stage moving from one to the next.
- [ ] `cargo run --release -p tetera-acquire --example preview-pipeline -- --smoke` — expect `smoke: ok`.
- [ ] `cargo test -p tetera-acquire` — expect 16 tests to pass.
- **Result:** see the [notes](notes/P1.14a-the-frame-bus-and-the-preview-pipeline.md#manual-test).

## P1.14b: The preview stream

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL, and the GPU; port 8310 free.
- [ ] `cargo run --release -p tetera-api --features cuda --example preview-stream` — expect the camera on the GPU in its binned live mode; the quick client at the camera's rate, 30 frames a second or more, 100 ms or less from each frame's exposure to its receipt; the slow client at about five a second, its link lowered (quality 60 or 40).
- [ ] `cargo run --release -p tetera-api --example preview-stream -- --smoke` — expect `smoke: ok`.
- [ ] `cargo run --release -p tetera-server`, then from Windows: `curl.exe http://localhost:8310/preview/` — expect the Thing Description, with `defaults`, `sessions` and the `stream`'s link (`subprotocol` `websocket`); `curl.exe http://localhost:8310/preview/defaults` — the 1920 × 1280 viewport, quality 80, 60 a second.
- [ ] With the server running and the camera streaming (`POST /camera/start_streaming`), connect to `ws://localhost:8310/preview/stream` from Windows (the browser's console: `new WebSocket(...)`, `binaryType = "arraybuffer"`), send `{"width": 640, "height": 480}`, and expect two binary messages, each beginning `TPV1`, then none until an `ack`; send `{"quality": 0}` and expect `{"error": …}`.
- [ ] `cargo test -p tetera-acquire -p tetera-api` — expect 20 and 30 tests to pass.
- **Result:** see the [notes](notes/P1.14b-the-preview-stream.md#manual-test).

## P1.14c: MJPEG and the live-view page

- **Prerequisites:** the phase's branch; the CUDA toolkit 13.3 in WSL, and the GPU; port 8310 free; a browser on Windows.
- [ ] `cargo run --release -p tetera-server`, then `curl -s -o /dev/null -w '%{content_type}' 'http://127.0.0.1:8310/api/v1/preview/mjpeg?width=320&height=240' --max-time 2` — expect `multipart/x-mixed-replace; boundary=tetera-frame`.
- [ ] Open `http://localhost:8310/api/v1/preview/mjpeg?width=960&height=640` in the browser, the camera streaming (`POST /camera/start_streaming`) — expect it to play as a moving image.
- [ ] `curl http://127.0.0.1:8310/preview/statistics` while the camera streams — expect each colour's 256-bin histogram, the 48 × 32 mask, and the frame's number changing from one read to the next.
- [ ] `cargo test -p tetera-api` — expect 34 tests to pass.
- **Result:** see the [notes](notes/P1.14c-mjpeg-and-the-live-view-page.md#manual-test).

## Checkpoint B: live H&E in a browser page while jogging

The owner runs this at Checkpoint C, with A's and C's; the agent ran and recorded it at P1.14c.

- **Prerequisites:** the checkpoint's commit; the CUDA toolkit 13.3 in WSL, and the GPU; port 8310 free; a browser on Windows.
- [ ] `cargo xtask gate checkpoint` — expect G2 to pass on a fresh clone.
- [ ] `cargo run --release -p tetera-server` — expect it to serve on the GPU, the camera in its binned live mode.
- [ ] Open `http://localhost:8310/live-view` on Windows and press *Start* — expect live H&E at 20×, glass white, at the camera's rate (about 50 fps in mode 1 at 8 bits, 30 or more), and 100 ms or less from exposure to paint.
- [ ] Hold → for a few seconds, then ↓ — expect the tissue to slide left, then up, smoothly, and to stop when the button is let go; the histogram follows the field.
- [ ] Open the page's *MJPEG stream* link — expect it to play.
- [ ] Press *Stop* — expect the image to stop.
- **Result:** see [P1.14c's notes](notes/P1.14c-mjpeg-and-the-live-view-page.md#checkpoint-b).
