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
