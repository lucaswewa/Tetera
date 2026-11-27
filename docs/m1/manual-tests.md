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
