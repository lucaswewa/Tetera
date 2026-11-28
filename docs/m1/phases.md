# Milestone 1 phase specifications

| | |
|---|---|
| Status | **Draft for the owner's review**, 2026-11-26 |
| Covers | Every phase of Milestone 1, the backend core: P1.S0 and P1.01–P1.21 |
| Companion | [The implementation plan](../implementation-plan.md): its [§9](../implementation-plan.md#milestone-1-backend-core) gives the overview, its [§7](../implementation-plan.md#7-how-phases-work) the delivery rules, its [§8](../implementation-plan.md#8-local-ci-gates) the gates |

## How to read a spec

Each phase has:

- **a header table:** the crates it creates or changes, what it depends on, its ADRs, its gates, and its **first estimate**;
- **goal:** one or two sentences;
- **scope**, and what is **out of scope**;
- **try it:** the examples, and the manual test's outline;
- **done when:** what must be true before its PR opens;
- **tests:** what they cover, and what is **broken on purpose** to prove a test fails;
- **open questions** from plan §14 that the spec assumes an answer to. A different answer changes the spec.

**Estimates.**

- A first estimate is in five parts, in hand-written lines: code, tests, examples, docs-as-code (READMEs, schemas, configuration), and glue. Doc comments and blank lines count. Markdown docs don't.
- Before a phase's code, its second estimate goes into the plan's change log, in a commit of its own (plan §7).
- **The target is 1,500 lines a phase. An estimate over 2,000 goes to the owner,** who decides whether to split it. Two specs were over 2,000 (P1.14 and P1.15), and the owner split both on 2026-11-26: P1.14a and P1.14b, P1.15a and P1.15b. P1.14b's second estimate was over 2,000 again, and the agent split it on 2026-11-28 under the owner's standing approval: P1.14b and P1.14c. So was P1.15b's, split the same day into P1.15b, P1.15c and P1.15d, and P1.16's, into P1.16a and P1.16b; and P1.17's, into P1.17a and P1.17b.
- Specs between 1,500 and 2,000 list a split that is ready if the second estimate grows.

**What every phase does, so the specs don't repeat it** (plan §7):

- tests, examples with a README and a `--smoke` mode, implementation notes, a manual-test entry, new ADRs, the phase's status in the plan;
- **examples** live in the crate's `examples/` folder and run as `cargo run -p <crate> --example <name>`. With `--smoke`, an example finishes in under 10 seconds and exits 0;
- the tests follow plan §5.4: they wait for conditions, get their own data roots and ports, and are skipped without their switch (`TETERA_CUDA=1` and so on);
- **G1** before the PR. **G1c** when the change reaches CUDA code. **G2** at a checkpoint (P1.12c, P1.14c, P1.18). **G3** at P1.21;
- the notes name the ideas taken from OScope's documents (clean room, D1).

## Summary

| ID | Phase | Depends on | First estimate | Spec-first | Gates |
|---|---|---|---|---|---|
| [P1.S0](#p1s0-wsl2-hardware-spike) | WSL2 hardware spike (**deferred**) | — | 1,000 | — | G1 |
| [P1.01](#p101-workspace-and-gates) | Workspace and gates | — | 1,750 | — | G1, G2 |
| [P1.02](#p102-core-types-and-logging) | Core types and logging | P1.01 | 1,570 | — | G1 |
| [P1.03](#p103-the-frame-model) | The frame model | P1.02 | 1,220 | — | G1 |
| [P1.04a](#p104a-compute-backends) | Compute backends | P1.03 | 1,340 | — | G1, G1c |
| [P1.04b](#p104b-kernels-and-the-twin-harness) | Kernels and the twin harness | P1.04a | 1,310 | — | G1, G1c |
| [P1.05a](#p105a-bayer-and-demosaic) | Bayer and demosaic | P1.04b | 1,860 | — | G1, G1c |
| [P1.05b](#p105b-binning-resize-display-transform-and-jpeg) | Binning, resize, display transform and JPEG | P1.05a | 1,880 | — | G1, G1c |
| [P1.06a](#p106a-dark-flat-and-colour) | Dark, flat and colour | P1.05b | 1,730 | — | G1, G1c |
| [P1.06b](#p106b-the-fused-kernel) | The fused kernel | P1.06a | 1,270 | — | G1, G1c |
| [P1.07a](#p107a-denoise) | Denoise | P1.05b | 1,020 | Yes | G1, G1c |
| [P1.07b](#p107b-psfs-and-richardsonlucy) | PSFs and Richardson–Lucy | P1.07a | 1,395 | Yes | G1 |
| [P1.07c](#p107c-richardsonlucy-on-the-gpu) | Richardson–Lucy on the GPU | P1.07b | 1,430 | — | G1, G1c |
| [P1.08a](#p108a-the-pipeline-model) | The pipeline model | P1.07c | 1,510 | — | G1 |
| [P1.08b](#p108b-the-raw-paths-nodes) | The raw path's nodes | P1.08a | 1,390 | — | G1, G1c |
| [P1.08c](#p108c-gpu-residency-and-fusion) | GPU residency and fusion | P1.08b | 1,440 | — | G1, G1c |
| [P1.08d](#p108d-denoise-and-deconvolution-nodes-and-live-runs) | Denoise and deconvolution nodes, and live runs | P1.08c | 900 | — | G1, G1c |
| [P1.08e](#p108e-batch-runs-and-tiling) | Batch runs and tiling | P1.08d | 780 | — | G1, G1c |
| [P1.09a](#p109a-the-camera) | The camera | P1.03 | 1,150 | — | G1 |
| [P1.09b](#p109b-the-stage-the-objectives-and-the-light) | The stage, the objectives and the light | P1.09a | 1,490 | — | G1 |
| [P1.10a](#p110a-the-simulated-clock-and-motion-profiles) | The simulated clock and motion profiles | P1.09b | 1,080 | — | G1 |
| [P1.10b](#p110b-the-virtual-stage) | The virtual stage | P1.10a | 1,480 | — | G1 |
| [P1.11a](#p111a-the-he-sample-on-the-cpu) | The H&E sample, on the CPU | P1.04b | 1,490 | Yes | G1 |
| [P1.11b](#p111b-the-he-sample-on-the-gpu-and-its-tiles) | The H&E sample on the GPU, and its tiles | P1.11a | 1,340 | Yes | G1, G1c |
| [P1.12a](#p112a-optics-and-light-the-field-the-sensor-sees) | Optics and light: the field the sensor sees | P1.06b, P1.10b, P1.11b | 1,470 | — | G1, G1c |
| [P1.12b](#p112b-the-sensor-and-motion) | The sensor and motion | P1.12a | 1,150 | — | G1, G1c |
| [P1.12c](#p112c-the-virtual-camera-checkpoint-a) | The virtual camera (**Checkpoint A**) | P1.12b | 1,440 | — | G1, G1c, G2 |
| [P1.13a](#p113a-the-server-configuration-and-systemd) | The server, configuration and systemd | P1.12c | 1,320 | — | G1 |
| [P1.13b](#p113b-the-device-things) | The device Things | P1.13a | 1,350 | — | G1 |
| [P1.14a](#p114a-the-frame-bus-and-the-preview-pipeline) | The frame bus and the preview pipeline | P1.08d, P1.13b | 1,430 | — | G1, G1c |
| [P1.14b](#p114b-the-preview-stream) | The preview stream | P1.14a | 1,330 | — | G1 |
| [P1.14c](#p114c-mjpeg-and-the-live-view-page) | MJPEG and the live-view page (**Checkpoint B**) | P1.14b | 810 | — | G1, G2 |
| [P1.15a](#p115a-the-job-service) | The job service | P1.13b | 1,410 | — | G1 |
| [P1.15b](#p115b-the-dataset-store) | The dataset store | P1.15a | 1,190 | — | G1 |
| [P1.15c](#p115c-the-recorder-and-ingest) | The recorder and ingest | P1.14a, P1.15b | 980 | — | G1 |
| [P1.15d](#p115d-capture-and-its-things) | Capture and its Things | P1.15c | 1,100 | — | G1 |
| [P1.16a](#p116a-focus-measures-and-the-search) | Focus measures and the search | P1.15d | 1,090 | Yes | G1 |
| [P1.16b](#p116b-autofocus-on-the-instrument) | Autofocus on the instrument | P1.16a | 880 | — | G1 |
| [P1.17a](#p117a-scan-planning-and-the-stop-and-go-engine) | Scan planning and the stop-and-go engine | P1.16b | 1,370 | — | G1 |
| [P1.17b](#p117b-z-stacks-and-the-scan-thing) | z-stacks and the `scan` Thing | P1.17a | 860 | — | G1 |
| [P1.18](#p118-calibration) | Calibration (**Checkpoint C**) | P1.17 | 1,900 | Yes | G1, G1c, G2 |
| [P1.19](#p119-batch-processing) | Batch processing | P1.15b, P1.08e | 1,400 | — | G1, G1c |
| [P1.20](#p120-status-logs-power-and-settings) | Status, logs, power and settings | P1.15a | 1,780 | — | G1 |
| [P1.21](#p121-milestone-1-close) | Milestone 1 close | all | 950 | — | G3 |

**In all:** 43 phases since P1.04 to P1.13 were split on 2026-11-27, and P1.14b, P1.15b, P1.16 and P1.17 on 2026-11-28, about **34,800 lines** by first estimates, P1.S0 aside. The median phase is about 1,550.

- **Over 2,000:** none, since the owner split P1.14 and P1.15 on 2026-11-26.
- **Over the 1,500 target:** 15 phases. They're kept whole, because OScope's second estimates for later phases came in under their specs (M2's median was 0.79) as reuse grew. Twelve of them have a ready split; the other three (P1.04, P1.09, P1.17) are within 3 % of the target.
- **With every ready split taken,** M1 would have 35 phases of about 1,000 lines each.

**Checkpoints:**

- **A** (P1.12c): the virtual instrument renders H&E live, on the GPU, while the stage moves.
- **B** (P1.14c): live H&E in a browser page while jogging, from the server.
- **C** (P1.18): calibrate an objective, autofocus, and scan.

---

## P1.S0: WSL2 hardware spike

| | |
|---|---|
| Status | **Deferred** by the owner on 2026-11-26: the instrument isn't connected to this PC, and the TUCam Linux SDK isn't here yet. The USB 3 SSD stand-in and the link probe can run earlier |
| Crates | None. Standalone probes in `spikes/p1-s0/`, outside the workspace |
| Depends on | — |
| ADRs | Input to 0018 (the service in WSL2) and 0040 (devices in WSL2) |
| Gates | G1 (the probes build and their tests pass) |
| First estimate | code 650 · tests 100 · examples 0 · docs-as-code 150 · glue 100 = **1,000** |

**Goal.** Find out whether the real camera and stage work from inside WSL2, at what rates, and whether the Windows camera bridge is needed.

**Scope.**

- **A stand-in, before the instrument:** a USB 3 SSD's sequential read rate, natively on Windows and then attached to WSL2 through usbipd-win. The SSD moves data in bulk transfers, as the camera does.
- **`link-probe`:** sends 40 MB frames from a Windows process to a WSL2 process over TCP, under mirrored networking, and reports the rate. It answers whether the bridge could carry 289 MB/s.
- **`camera-probe`** (C, against the TUCam Linux SDK's own headers and documentation):
  - open the MIChrome 20 attached through usbipd-win;
  - list its modes;
  - capture raw 16-bit and 8-bit at modes 0, 1 and 2, ten minutes at mode 0;
  - report frame rate, MB/s, and dropped and broken frames.
- **`serial-probe`** (Rust): opens the grblHAL port, reads `$I` and `$$`, times 1,000 `?` status round trips, and reports the status rate while jogging.
- **`disk-probe`** (Rust): writes 40 MB frames sequentially to a preallocated file on ext4, at a target rate, and reports the sustained rate and the slowest writes.
- **usbipd-win's set-up:** bind, attach and auto-attach, and udev rules for the camera and the serial port.
- **`docs/hardware.md`:** OScope's measured facts about the camera, the stage and the disks, in our own words, plus what the spike measures in WSL2.
- **If USB forwarding falls short:** a sketch of the camera bridge (the transport, the frame format, how it implements the `Camera` trait) for P1.09 and P5.01.

**Out of scope.** Drivers (P5.02, P5.03); anything in the workspace.

**Try it.** Each probe's README: how to build it, attach the device, run it, and read its report.

**Done when.**

- `docs/hardware.md` records each measurement, with the date, the commands and the hardware.
- A decision is recorded in the notes: USB forwarding is enough, or the bridge is needed.

**Tests.** The probes' argument parsing and report formatting. Hardware runs are the manual test.

**Open questions.** Q14 (scratch disk), Q16 (raw bit depth: the probe records what the SDK delivers).

**Owner's steps.** Install usbipd-win and bind each device as an administrator; provide the TUCam Linux kit's path; be at the instrument.

---

## P1.01: Workspace and gates

| | |
|---|---|
| Crates | The workspace; `xtask` (new); `tetera-core` (new, empty, so the gates have something to run) |
| Depends on | — |
| ADRs | 0001 recording decisions and the delivery process · 0002 clean room and dependency licences · 0003 the runtime platform · 0004 teta-wot v0.2.0 · 0005 layered crates and the local gates |
| Gates | G1; G2 once, to prove it works |
| First estimate | code 1,050 · tests 400 · examples 20 · docs-as-code 200 · glue 80 = **1,750** |
| Ready split | P1.01a: workspace, layering, `doctor`, G0, docs skeleton (~900) · P1.01b: G1 and G2, with change reach, stress runs and summaries (~850) |

**Goal.** A complete, empty workspace in which every later phase lands, with the local gates, the docs skeleton and the working rules.

**Scope.**

- **The workspace:**
  - `Cargo.toml`: resolver 3, edition 2024, `rust-version = "1.98"`. Members `crates/*`, `examples/*` and `xtask`; `spikes/*` excluded;
  - workspace lints: `unsafe_code = "forbid"`, `missing_docs = "warn"`, clippy's `all` as warnings, which the gates deny;
  - `rust-toolchain.toml`: 1.98.1, with rustfmt and clippy;
  - `.cargo/config.toml`: clang with mold as the linker, `opt-level = 2` for dependencies in dev and test builds, and the `xtask` alias;
  - `rustfmt.toml`, `deny.toml` (the licences of plan §5.2; advisories; sources limited to crates.io and teta-wot's repository), `.gitattributes` (LF), `.gitignore`.
- **`cargo xtask layering`:** the layer table of plan §6.2, with every planned crate placed now, and the allowed edges within a layer. Checks every workspace member through `cargo metadata`:
  - no upward edge, and no edge within a layer that the table doesn't allow;
  - only `tetera-api` depends on `teta-wot`;
  - every crate takes the workspace's lints, except the crates allowed unsafe code, which repeat them with `unsafe_code` allowed and clippy's `undocumented_unsafe_blocks` denied.

  It is also an integration test, so `cargo test` runs it.
- **`cargo xtask gate quick | phase | checkpoint | milestone`**, with `--base <ref>` (default `main`), `--cuda`, `--python` and `--jobs N`:
  - **reach:** the files changed since the merge base, the packages holding them, and every package that depends on those. The workspace's manifest, the lock file, the toolchain, `.cargo/` and `xtask/` reach everything;
  - **steps:** each level's steps as plan §8 lists them. A step nothing reaches says `skip`;
  - **stress:** the reached test binaries 3 times side by side, each pinned to 2 CPUs with `taskset`;
  - **fresh clone:** `checkpoint` and `milestone` clone the commit into a temporary folder and run there, with a target folder of their own kept between runs so they start warm;
  - **summary:** `target/gates/<time>/summary.md`, with each step, why it ran, its time and its result, and each step's log beside it.
- **`cargo xtask doctor`:** reports the toolchain, mold, clang, cargo-deny, cargo-mutants, CUDA (optional until P1.04), Node (until M4) and uv (until M2); free disk (at least 40 GB); WSL's memory, and the `CARGO_BUILD_JOBS` it suggests; a warning when the repository is under `/mnt/`; git's identity and `core.autocrlf`.
- **`.vscode/`:** `launch.json` with CodeLLDB configurations for an example and for a crate's tests (the server's comes in P1.13a); `extensions.json` recommending rust-analyzer, CodeLLDB and Even Better TOML; `settings.json` with rust-analyzer checking through clippy and LF line endings.
- **`.github/pull_request_template.md`:** the Definition of Done of plan §7. No workflows.
- **Docs:**
  - `docs/README.md`, the index;
  - `docs/goal.md`, the spec, moved from `tetera.md` unchanged;
  - `docs/adr/` with its index and template, and ADRs 0001–0005;
  - `docs/m1/notes/` with its index and template; `docs/m1/manual-tests.md`;
  - `docs/setup/toolchain.md`, from plan §12, each step re-run on this PC;
  - `CLAUDE.md`, the working rules: the delivery process, the clean room, dependencies, unsafe code, tests, ports, the gates.
- **`tetera-core`:** a crate with its docs and one `hello` example, for the gates and the debugger to run.

**Out of scope.** Domain code (P1.02 on); CUDA (P1.04); the web checks (M4); performance and soak runs (P1.21).

**Try it.**

- `cargo xtask doctor` from a fresh clone in WSL.
- `cargo xtask gate quick`, then `cargo xtask gate phase --base main`.
- Add an upward dependency on purpose, and see the gate name the edge.
- In VS Code over Remote-WSL, debug the `hello` example and stop at a breakpoint.

**Done when.**

- From a fresh clone in WSL, `doctor` reports nothing required missing.
- `gate quick` passes in under 90 seconds warm, `gate phase` passes, and `gate checkpoint` passes on a fresh clone.
- The summary lists each step with why it ran.
- Debugging the example from VS Code stops at a breakpoint.

**Tests.**

- **Reach**, over synthetic dependency graphs: a leaf change, a core change, a manifest change, a docs-only change.
- **Layering**, over synthetic metadata: an upward edge, a sideways edge, an allowed edge within a layer, `teta-wot` outside `tetera-api`, a crate without the workspace's lints.
- **The file classifier,** the summary's rendering, and `doctor`'s checks with faked inputs.
- **Broken on purpose:** an upward edge, and a crate without `[lints] workspace = true`. Each must fail the gate. The mutants list covers reach and the layering rules.

**Open questions.** Q40 (who merges), Q41 (no workflows), Q42 (gate contents), Q46 (Rust 1.98.1).

---

## P1.02: Core types and logging

| | |
|---|---|
| Crates | `tetera-core`; `tetera-log` (new, L1). **A change to plan §6.2:** `tetera-core` does no I/O, so the log writers need a crate of their own |
| Depends on | P1.01 |
| ADRs | 0006 units, coordinates and the clock · 0007 logging |
| Gates | G1 |
| First estimate | code 900 · tests 450 · examples 120 · docs-as-code 60 · glue 40 = **1,570**. Second estimate 3,180; the owner kept it whole (2026-11-26) |
| Ready split | P1.02a: units, geometry, the clock, IDs, errors (~950) · P1.02b: logging (~620) |

**Goal.** The vocabulary every other crate uses, and logging that every example and test can turn on.

**Scope.**

- **Units:** `Um` (µm, `f64`) for lengths, `std::time::Duration` for times, and conversions at the edges. Serde, and JSON Schema through `schemars`, always on: every type that crosses the API needs it from P1.13a, and a feature no gate turns on would leave the derives unchecked.
- **Geometry:** points and vectors in µm and in pixels; `Rect` in pixels; `Affine2` (compose, invert, apply); `StagePos { x, y, z }`.
- **The clock:**
  - `Timestamp`: nanoseconds since the process started, from the monotonic clock;
  - a `Clock` trait, with `SystemClock`, and `ManualClock` for tests. P1.10a's simulated clock implements it too;
  - UTC mapped once at start-up, and re-checked: a drift over 1 s raises a warning (WSL2's clock can drift after the host sleeps).
- **IDs:** `Id<T>`, a UUID v7, so IDs sort by time.
- **Errors:** one `Error` with a kind (invalid argument, not found, busy, hardware, timeout, cancelled, internal), a context chain and an optional source.
- **Log records** (in `tetera-core`): `LogRecord` with time, level, target, message and fields. The levels are trace, debug, info, warn, error and **critical**. `LogRing`, bounded, drops the oldest record and counts the drops.
- **`tetera-log`:** the `tracing` subscriber's set-up:
  - a layer into the ring;
  - JSON files rotated by size, written in-house (the owner's choice, 2026-11-26: `tracing-appender` rotates only by time);
  - journald over its native protocol, written in-house, when running under systemd, so a critical record is priority 2 (CRIT). `tracing-journald` maps priority by level only;
  - readable output to stderr;
  - filters from the configuration or `RUST_LOG`;
  - a `critical!` macro: an error record with `critical = true`, reported as *critical*.

**Out of scope.** The `logs` Thing (P1.20); frames (P1.03).

**Try it.** `core-tour` (units, geometry, the clock, IDs); `log-tour` (writes records, then shows the ring's tail and the file).

**Done when.** The types round-trip through serde; the ring keeps the newest records under load; a critical record shows as critical in the ring, the file and journald.

**Tests.**

- Property tests (`proptest`): affine composition and inversion, rectangle intersections, unit conversions.
- The clock: monotonic, and the drift warning, with `ManualClock`.
- IDs sort by creation time.
- The ring overflowing under several threads.
- File rotation.
- **Broken on purpose:** the ring's drop of the oldest record, and the drift check.

---

## P1.03: The frame model

| | |
|---|---|
| Crates | `tetera-core` |
| Depends on | P1.02 |
| ADRs | 0008 the frame model, the buffer pool and the frame bus |
| Gates | G1 |
| First estimate | code 650 · tests 400 · examples 100 · docs-as-code 40 · glue 30 = **1,220** |

**Goal.** One way to hold a frame, from any camera, real or virtual, that every later stage shares without copying.

**Scope.**

- **`PixelFormat`:** `Bayer16 { pattern, significant_bits }`, `Bayer8`, `Mono8`, `Mono16`, `Rgb8`, `Rgb16`, `Rgba8`, planar `F32`. Sizes and strides for each, with ROI and binning. **A crop at an odd offset changes the Bayer phase,** and the format records it.
- **`FrameMeta`:** sequence number; exposure start, end and midpoint; line time (rolling shutter); exposure, gain, mode, binning, ROI, black level; objective; the light's state; the stage position at the midpoint, if known; frames dropped before this one.
- **`FramePool`:** a fixed number of buffers of one size, aligned to 4 KiB without unsafe code; `take()` waits up to a timeout when all are in use; a frame is shared by reference count, and its buffer goes back to the pool when the last reference drops; statistics (in use, waits, misses).
- **`Allocator` trait:** plain host memory now. P1.04a plugs in pinned memory for CUDA.
- **Safe views:** rows, planes, Bayer sites; scaling between significant bits and full range.
- **The frame bus's design,** decided here and recorded in ADR-0008: lossless subscribers with backpressure, latest-wins subscribers with a one-frame mailbox. P1.14a builds it.

**Out of scope.** Pinned memory (P1.04a); building the frame bus (P1.14a).

**Try it.** `frame-pool`: a pool of 16 full Bayer16 frames (about 640 MB), filled and shared by three consumers, with the statistics, then an exhausted pool.

**Done when.**

- A 5472 × 3648 Bayer16 frame takes 39.9 MB, and its metadata round-trips.
- After warming up, the pool allocates nothing (counted).
- An exhausted pool waits, then fails with a clear error.

**Tests.** Property tests of sizes and strides for every format, ROI and binning; Bayer phase after cropping; many threads taking and returning frames, also in the stress runs. **Broken on purpose:** returning a buffer to the pool, and the exhaustion timeout.

**Open questions.** Q16 (`significant_bits = 12`).

---

## P1.04a: Compute backends

| | |
|---|---|
| Crates | `tetera-compute` (new, L1, unsafe allowed) |
| Depends on | P1.03 |
| ADRs | 0009 compute backends: the CPU reference and CUDA |
| Gates | G1, G1c |
| Second estimate | code 760 · tests 400 · examples 100 · docs-as-code 60 · glue 20 = **1,340**. The owner split P1.04 (second estimate 2,670) on 2026-11-27 |

**Goal.** Choose between the CPU and CUDA at run time, hold data on either, and know what this machine's GPU and WSL2 offer.

**Scope.**

- **Backends:** `Cpu` (rayon) and `Cuda` (`cudarc`, the driver loaded at run time). The `cuda` feature builds the CUDA path; without a GPU, a driver, or the feature, everything runs on the CPU. `TETERA_FORCE_CPU=1` forces the CPU.
- **Detection:** the device, its compute capability, its memory and free VRAM, the driver's CUDA version.
- **Memory:** device buffers, copied to and from the host; streams; pinned host memory, through P1.03's `Allocator`.
- **What WSL2's CUDA offers,** checked and recorded in ADR-0009: pinned memory at 640 MB, cuFFT, NVRTC, nvJPEG, and inter-process memory handles.
- **The CUDA toolkit's version,** 13.3, pinned, with its exact install commands in `docs/setup/toolchain.md`.

**Out of scope.** Kernels and their build (P1.04b); imaging (P1.05).

**Try it.** `compute_info`: the device, the versions, and each WSL2 feature: works or not.

**Done when.**

- The workspace builds without the CUDA toolkit, and with it.
- With `TETERA_FORCE_CPU=1`, or no GPU, the CPU is chosen.
- With `TETERA_CUDA=1`, device buffers round-trip on this PC, and a frame pool lives in pinned memory.
- The WSL2 feature table is in ADR-0009.

**Tests.** Backend selection with faked detection; buffer round trips (CUDA, with `TETERA_CUDA=1`); pinned buffers' alignment. **Broken on purpose:** the fallback to the CPU when the driver is missing.

**Owner's steps.** Install the CUDA toolkit 13.3 for WSL-Ubuntu (it needs `sudo`), with the commands in `docs/setup/toolchain.md`.

---

## P1.04b: Kernels and the twin harness

| | |
|---|---|
| Crates | `tetera-compute` |
| Depends on | P1.04a |
| ADRs | 0010 building and loading CUDA kernels under WSL2 |
| Gates | G1, G1c |
| Estimate | code 575 · tests 600 · examples 90 · docs-as-code 25 · glue 20 = **1,310**, from P1.04's split; its own second estimate comes before its code |

**Goal.** Run the same operation on the CPU or on CUDA, and prove the two agree.

**Scope.**

- **Kernels:** CUDA C under `kernels/`, compiled to PTX by `build.rs` when the `cuda` feature is on and `nvcc` is found, and embedded in the binary; loaded as modules, and launched.
- **The twin harness:** runs an operation on both backends and compares the results within a stated tolerance; `cuda_or_skip!()` for tests that need `TETERA_CUDA=1`.
- **One small kernel pair** (scale and offset `u16` to `f32`) to prove the path.
- **VRAM budgets:** each user (simulator, preview, jobs) reserves a budget, and a request beyond it is refused.

**Out of scope.** Imaging kernels (P1.05a on); fusion (P1.06b).

**Try it.** `twin_scale`: both backends, with timings and their difference.

**Done when.**

- The CUDA twin tests pass with `TETERA_CUDA=1` on this PC.
- The workspace still builds without the CUDA toolkit.
- A budget refuses a request beyond it.

**Tests.** The harness's tolerance arithmetic; VRAM budget accounting; the CUDA twins (G1c). **Broken on purpose:** the harness's comparison, which a test must catch passing a result out of tolerance.
---

## P1.05a: Bayer and demosaic

| | |
|---|---|
| Crates | `tetera-imaging` (new, L1) |
| Depends on | P1.04b |
| ADRs | — |
| Gates | G1, G1c |
| Second estimate | code 1,020 · tests 650 · examples 120 · docs-as-code 30 · glue 40 = **1,860**. The owner split P1.05 (second estimate 3,140) on 2026-11-27, moving binning from the ready split's first half to its second so that both fit |

**Goal.** Turn a raw Bayer frame into full-colour RGB, on either backend.

**Scope.**

- **Images:** the mosaics and RGB images the operations take and give, on the host or the device.
- **Bayer split:** RGGB into R, G1, G2 and B planes; a half-resolution RGB superpixel.
- **Demosaic:** bilinear, and Malvar–He–Cutler (ICASSP 2004), to RGB in 16 bits or `f32`.
- **CUDA kernels** for each operation, each with its CPU twin.
- **ROIs at odd offsets** handled through the Bayer phase that P1.03 records.

**Out of scope.** Binning, resize, the display transform and JPEG (P1.05b); dark, flat and colour (P1.06a).

**Try it.** `demosaic`: a known RGB test pattern is mosaicked, demosaicked both ways on both backends, and written as PPM, with timings and each method's PSNR against the pattern.

**Done when.**

- Integer operations agree exactly between backends, and `f32` ones within 1 least significant bit.
- Malvar–He–Cutler's PSNR beats bilinear's on the test pattern; both numbers are recorded.

**Tests.** Flat colours survive mosaicking and demosaicking exactly; borders; every Bayer phase; the CUDA twins. **Broken on purpose:** the Bayer phase after an odd crop, and border handling.

**Open questions.** Q22 (demosaic methods).

---

## P1.05b: Binning, resize, display transform and JPEG

| | |
|---|---|
| Crates | `tetera-imaging` |
| Depends on | P1.05a |
| ADRs | — |
| Gates | G1, G1c |
| Second estimate | code 845 · tests 850 · examples 110 · docs-as-code 45 · glue 30 = **1,880**, against 1,370 from P1.05's split |

**Goal.** Turn a frame into a preview a browser can show, on either backend.

**Scope.**

- **Binning:** 2 × 2, 3 × 3 and 4 × 4, by sum or mean, keeping the Bayer layout.
- **Resize:** area averaging for shrinking, bilinear otherwise, to any preview size.
- **Display transform:** black and white levels, then a gamma table, to RGB8.
- **JPEG:** on the CPU, with SIMD, at a chosen quality. MJPEG is a sequence of these.
- **CUDA kernels** for each operation but JPEG, each with its CPU twin.

**Out of scope.** JPEG on the GPU (P5.06, if it pays); dark, flat and colour (P1.06a).

**Try it.** `demosaic` gains binning, a resize to preview size, the display transform and JPEG, with each step's time.

**Done when.**

- Integer operations agree exactly between backends, and `f32` ones within 1 least significant bit.
- A 1920 × 1280 JPEG at quality 80 takes 10 ms or less on the CPU, in a release build.

**Tests.** Property tests of binning sums; resizing flat images and ramps; the gamma table's ends; JPEG read back by a decoder; the CUDA twins. **Broken on purpose:** binning that mixes the Bayer layout's colours.
---

## P1.06a: Dark, flat and colour

| | |
|---|---|
| Crates | `tetera-imaging`, `tetera-compute` |
| Depends on | P1.05b |
| ADRs | — |
| Gates | G1, G1c |
| Second estimate | code 790 · tests 790 · examples 100 · docs-as-code 25 · glue 25 = **1,730**. The agent split P1.06 (second estimate 2,490) on 2026-11-27, under the owner's standing approval, along its ready split |

**Goal.** Correct the raw image for the sensor and the light, and turn it into colour a person sees, on either backend.

**Scope.**

- **Dark subtraction:** a master dark, matched by exposure and gain through the metadata, clamped at zero.
- **Flat-field correction:** a normalised, dark-subtracted master flat per Bayer channel, guarded against near-zero values.
- **Colour:** white-balance gains, a 3 × 3 matrix to linear sRGB, then the sRGB transfer curve, to 8 bits.
- **`IspParams`:** the calibration inputs in one struct.
- **The unfused chain:** crop or bin → dark → flat → demosaic → white balance and CCM → gamma → 8 bits → resize, stage by stage on either backend: P1.06b's reference. *Built resizing before colour, in linear light (see P1.06a's notes).*
- **CUDA kernels** for each new stage, each with its CPU twin.

**Out of scope.** The fused kernel (P1.06b); calibrating darks, flats and CCMs (P1.18); recipes (P1.08a).

**Try it.** `isp`: the unfused chain on each backend, from 20 MP and from a binned mode to 1920 × 1280, with each stage's time and the images.

**Done when.**

- Each stage's twins agree exactly.
- On a synthetic frame with an offset and vignetting, dark and flat correction leave a field flat within 1 %.

**Tests.** Each stage alone; property tests; the twins. **Broken on purpose:** the clamp at zero, and the guard against division by small values.

---

## P1.06b: The fused kernel

| | |
|---|---|
| Crates | `tetera-compute`, `tetera-imaging` |
| Depends on | P1.06a |
| ADRs | 0011 fused kernels |
| Gates | G1, G1c |
| Second estimate | code 660 · tests 500 · examples 80 · docs-as-code 20 · glue 10 = **1,270**, against 950 from P1.06's split |

**Goal.** Do the whole live chain in one GPU pass.

**Scope.**

- **The fused kernel** (D7): crop or bin → dark → flat → demosaic (either method) → resize → white balance and CCM → gamma → 8 bits, in one CUDA pass: P1.06a's order.
- **Its reference** is P1.06a's unfused chain on the CPU. The tolerance: within 1 level in 8 bits for 99.9 % of pixels, and 2 levels for all.

**Out of scope.** Recipes, and the pass that finds the chain in one (P1.08c).

**Try it.** `isp` gains the fused chain: fused against unfused, CPU against GPU, with timings and the images.

**Done when.**

- Fused on the GPU, 20 MP to 1920 × 1280 takes 10 ms or less on this PC (plan §6.15).
- The equivalence holds within the tolerance.

**Tests.** The fused kernel against the unfused chain over random parameters. **Broken on purpose:** a stage left out of the fused kernel, which the equivalence must catch.

**Open questions.** Q19 (a hand-written fused kernel).
---

## P1.07a: Denoise

| | |
|---|---|
| Crates | `tetera-imaging`, `tetera-compute` |
| Depends on | P1.05b |
| ADRs | — |
| Gates | G1, G1c |
| Spec-first | Yes: the notes open with the behaviour and its literature before any code |
| Second estimate | code 450 · tests 420 · examples 120 · docs-as-code 20 · glue 10 = **1,020** (1,330 at ×1.3). The agent split P1.07 (second estimate 2,850) on 2026-11-27, under the owner's standing approval, into P1.07a, P1.07b and P1.07c: the ready split's second half alone came to 2,380 at ×1.3 |

**Goal.** Reduce noise without blurring edges, on either backend.

**Scope.**

- **Temporal averaging:** a running mean over N frames, reset when the stage moves.
- **One edge-preserving spatial filter.** The behaviour spec chooses between bilateral (Tomasi and Manduchi, 1998) and non-local means (Buades, Coll and Morel, 2005), weighing quality against time at 20 MP.
- **CUDA kernels** for each, each with its CPU twin.

**Out of scope.** PSFs and deconvolution (P1.07b, P1.07c); BM3D and learned denoisers (extensions).

**Try it.** `denoise`: PSNR before and after, on a noisy synthetic image, for each method on each backend, with timings.

**Done when.**

- The behaviour spec, with its sources, was committed before the code.
- Each method raises the PSNR of a noisy synthetic image; the numbers are recorded.
- The backends agree exactly.

**Tests.** The running mean over N frames, and its reset; the filter's weights; edges kept while flat noise falls; the twins. **Broken on purpose:** the reset when the stage moves, and the filter's range weight.

**Open questions.** Q20 (denoise methods).

---

## P1.07b: PSFs and Richardson–Lucy

| | |
|---|---|
| Crates | `tetera-imaging` |
| Depends on | P1.07a |
| ADRs | — |
| Gates | G1 |
| Spec-first | Yes |
| Second estimate | code 610 · tests 610 · examples 130 · docs-as-code 25 · glue 20 = **1,395**, against 1,235 from P1.07's split |

**Goal.** Restore detail the optics blurred, on the CPU.

**Scope.**

- **PSFs:**
  - a theoretical widefield PSF from the objective's NA and the wavelength: 2D in focus, and 3D through focus (scalar diffraction, Born and Wolf);
  - a measured PSF, loaded from a file (measured in M5).
- **Richardson–Lucy** (Richardson, 1972; Lucy, 1974), FFT-based, 2D and 3D, with a set number of iterations and optional total-variation regularisation (Dey and others, 2006). FFTs through `rustfft`.

**Out of scope.** Richardson–Lucy on the GPU (P1.07c); blind deconvolution.

**Try it.** `deconvolve`: an image blurred by a known PSF, with noise, restored; the error against the truth by iteration.

**Done when.**

- The behaviour spec, with its sources, was committed before the code.
- Richardson–Lucy lowers the error against the ground truth on synthetic data; the numbers are recorded.

**Tests.** PSF normalisation and symmetry; Richardson–Lucy conserving flux; convergence on a point source. **Broken on purpose:** flux conservation, and the PSF's normalisation.

**Open questions.** Q21 (2D and 3D; theoretical PSF first).

---

## P1.07c: Richardson–Lucy on the GPU

| | |
|---|---|
| Crates | `tetera-compute`, `tetera-imaging` |
| Depends on | P1.07b |
| ADRs | — |
| Gates | G1, G1c |
| Second estimate | code 655 · tests 655 · examples 90 · docs-as-code 15 · glue 15 = **1,430** (1,860 at ×1.3), against 650 from P1.07's split |

**Goal.** Deconvolve on the GPU, matching the CPU.

**Scope.** Richardson–Lucy, 2D and 3D, with its regularisation, on cuFFT; its elementwise kernels, each with its CPU twin.

**Out of scope.** Blind deconvolution.

**Try it.** `deconvolve` gains the GPU: both backends, with timings.

**Done when.**

- The backends agree within the tolerance P1.07b's behaviour spec states.
- 20 iterations of 2D Richardson–Lucy at 20 MP on the GPU are timed and recorded.

**Tests.** The twins over random PSFs and images, 2D and 3D. **Broken on purpose:** the correction step's mirrored PSF.

---

## P1.08a: The pipeline model

| | |
|---|---|
| Crates | `tetera-pipeline` (new, L3) |
| Depends on | P1.07c |
| ADRs | 0012 the pipeline model |
| Gates | G1 |
| Spec-first | — |
| Second estimate | code 700 (ports, values, descriptors and the `Node` trait 180; recipes and their schema 110; validation and the order 230; the registry and two glue nodes 80; the CPU executor 100) · tests 700 · examples 60 · docs-as-code 30 · glue 20 = **1,510** (1,960 at ×1.3). The agent split P1.08 (second estimate about 5,070) on 2026-11-27, under the owner's standing approval, into P1.08a, P1.08b, P1.08c and P1.08d: the ready split's halves came to about 3,950 and 2,640 at ×1.3 |

**Goal.** Describe image processing as a graph of typed nodes, check a recipe precisely, and run it.

**Scope.**

- **`Node`:** a descriptor (id, version, input and output ports with types, parameters as JSON Schema, backends, fusable), and its run function.
- **Port types:** raw frame, image (its channels and sample type), scalar, table.
- **Recipes:** JSON, with a schema. Validation reports exactly what is wrong: an unknown node or version, an unknown port, mismatched port types, an input left unfed or fed twice, a cycle, a missing or invalid parameter.
- **The registry,** under stable ids, with the first glue nodes: crop, and statistics.
- **The executor,** on the CPU: topological order, each value handed from port to port.

**Out of scope.** The stock nodes (P1.08b, P1.08d); GPU residency and fusion (P1.08c); live runs (P1.08d); batch runs and tiling (P1.08e).

**Try it.** `recipe`: a JSON recipe validated, a broken one refused with its messages, and the good one run on a synthetic frame.

**Done when.**

- Every invalid recipe in the test set is refused with a precise message.
- The executor runs a recipe's nodes in topological order, and gives its outputs.
- ADR-0012 records the model.

**Tests.** Validation, a case for each error; topological order; the executor on test nodes. **Broken on purpose:** the validator's type check.

---

## P1.08b: The raw path's nodes

| | |
|---|---|
| Crates | `tetera-pipeline` |
| Depends on | P1.08a |
| ADRs | — |
| Gates | G1, G1c |
| Second estimate | code 620 (an encoded image as a port type 40; ten nodes of about 55 lines each: binning, the Bayer split, the superpixel, demosaicking, dark, flat, resizing, colour, the display transform, JPEG) · tests 620 · examples 120 · docs-as-code 20 · glue 10 = **1,390** (1,810 at ×1.3). P1.08b as first split, with every stock node and live runs, came to about 1,760, 2,290 at ×1.3: the agent re-split it on 2026-11-27, under the owner's standing approval, the denoise and deconvolution nodes joining live runs in P1.08d |

**Goal.** The raw path, from a raw frame to a JPEG, as nodes.

**Scope.**

- **An encoded image** as a port type: bytes in a format, JPEG first.
- **The raw path's nodes** in the registry, under stable ids, each on whichever backend `Compute` chose: binning, the Bayer split, the superpixel, demosaicking, dark subtraction, the flat field, resizing, colour, the display transform, and JPEG.
- **The live chain as a recipe:** raw → dark → flat → demosaic → resize → colour → JPEG.

**Out of scope.** The denoise and deconvolution nodes, and live runs (P1.08d); keeping data on the GPU between nodes, and fusion (P1.08c).

**Try it.** `recipe` runs the live chain's recipe on a synthetic raw frame, node by node, to a JPEG.

**Done when.**

- Each of the raw path's nodes is in the registry, with its descriptor and its parameters' schema, and matches the function it wraps.
- The live chain's recipe gives the same preview as `Isp::run`.

**Tests.** Each node against the function it wraps; the chain's recipe against the ISP. **Broken on purpose:** the demosaic node ignoring its method.

---

## P1.08c: GPU residency and fusion

| | |
|---|---|
| Crates | `tetera-pipeline` |
| Depends on | P1.08b |
| ADRs | — |
| Gates | G1, G1c |
| Second estimate | code 670 (values on the device, and the copies counted, 150; the raw path's device paths, eight short ones since `tetera-imaging` has each `_device` function, 120; the executor's residency, freeing each value after its last reader, 180; the fusion pass and its fused step 220) · tests 670 · examples 70 · docs-as-code 20 · glue 10 = **1,440** (1,870 at ×1.3), against 1,490 from P1.08's split |

**Goal.** Run recipes on the GPU without needless copies, and with the fused kernel where it fits.

**Scope.**

- **GPU residency:** data kept on the GPU between consecutive CUDA nodes, copied only at the boundaries; copies counted.
- **The fusion pass:** finds the live chain in a recipe, and runs P1.06b's fused kernel in its place, when the backend and the parameters allow.

**Out of scope.** The denoise and deconvolution nodes, and live runs (P1.08d); batch runs and tiling (P1.08e).

**Try it.** `recipe` on the GPU: the copies it made, and the live chain fused against unfused.

**Done when.**

- Fusion triggers, and its output matches the unfused recipe's.
- A chain of CUDA nodes makes only the boundary copies.

**Tests.** The fusion matcher; copy counts. **Broken on purpose:** the fusion matcher.

---

## P1.08d: Denoise and deconvolution nodes, and live runs

| | |
|---|---|
| Crates | `tetera-pipeline` |
| Depends on | P1.08c |
| ADRs | — |
| Gates | G1, G1c |
| Second estimate | code 400 (three nodes 210: temporal averaging, stateful, its window on either backend; the bilateral filter, on either; Richardson–Lucy, its PSF made once from the optics; live runs 190: a thread, one frame waiting at most, the latest kept) · tests 400 · examples 80 · docs-as-code 15 · glue 5 = **900** (1,170 at ×1.3), as P1.08b's re-split estimated |

**Goal.** The rest of the stock nodes, and recipes run on a live stream.

**Scope.**

- **The denoise and deconvolution nodes:** temporal averaging, the bilateral filter, and Richardson–Lucy, each on the backend chosen, and on the GPU without needless copies.
- **Live mode:** one frame at a time; a frame that arrives while one is running replaces the waiting one.

**Out of scope.** Batch runs and tiling (P1.08e).

**Try it.** `recipe` runs the live chain live at 14.5 fps.

**Done when.**

- Every stock node is in the registry, with its descriptor and its parameters' schema.
- Live mode never queues more than one frame.

**Tests.** Each node against the function it wraps; live mode under a source faster than the recipe. **Broken on purpose:** live mode keeping every frame.

---

## P1.08e: Batch runs and tiling


| | |
|---|---|
| Crates | `tetera-pipeline` |
| Depends on | P1.08d |
| ADRs | — |
| Gates | G1, G1c |
| Second estimate | code 340 (batch runs 40; each step's reach, from its parameters, 60; the halo over the recipe's paths 40; cutting values into tiles and stitching them back 120; the tiled run 80) · tests 340 · examples 80 (a folder of frames written and read back) · docs-as-code 15 · glue 5 = **780** (1,014 at ×1.3), against 740 from P1.08's split |

**Goal.** Run a recipe over many inputs, and over images too large to take whole.

**Scope.**

- **Batch mode:** over a list of inputs.
- **Tiling:** large images in tiles, with a halo for neighbourhood operations. Each node declares how far it reaches, or that it needs the whole image.

**Out of scope.** Batch over datasets as jobs (P1.19); extension nodes (M2).

**Try it.** `recipe` over a folder of frames, tiled and untiled.

**Done when.**

- Tiled and untiled results are the same.

**Tests.** Tiles with halos at edges and corners; a node that needs the whole image; batch order. **Broken on purpose:** a halo one pixel short.

---

## P1.09a: The camera

| | |
|---|---|
| Crates | `tetera-hal` (new, L2) |
| Depends on | P1.03 |
| ADRs | 0013 the hardware abstraction layer |
| Gates | G1 |
| Second estimate | code 550 (events and errors 100; the `Camera` trait, its capabilities and configuration 250; the fake camera 200) · tests 500 (the camera's contract suite 200; unit tests 300) · examples 60 · docs-as-code 20 · glue 20 = **1,150** (1,495 at ×1.3). The agent split P1.09 (second estimate about 2,190, 2,850 at ×1.3) on 2026-11-27, under the owner's standing approval, into P1.09a and P1.09b |

**Goal.** One interface for cameras, which a real driver, a virtual camera, a fake, or a remote bridge can each implement.

**Scope.**

- **`Camera`:** capabilities (modes with size, rates and bit depths; binning; ROI; exposure and gain ranges; triggers); configure; start and stop streaming into a `FrameSink`; snap; errors as events.
- **Ready for a bridge** (the owner's fallback, R1): no borrowed buffers cross the interface; frames come from a pool the driver owns; configurations serialise.
- **Blocking methods** on `Send + Sync` traits, since drivers block; the API layer calls them off its async threads.
- **The fake camera:** a test pattern at a set rate, with faults on demand.
- **The camera's contract suite:** tests any `Camera` must pass, run against the fake now, and against the virtual and real cameras later.

**Out of scope.** The stage, the objectives and the light (P1.09b); the simulator (P1.10–P1.12); drivers (M5).

**Try it.** `hal-fakes`: drives the fake camera, and prints its frames and events.

**Done when.**

- The fake camera passes the contract suite, and a fake broken on purpose fails it.
- ADR-0013 records the HAL.

**Tests.** The contract suite; faults; configurations round-tripping. **Broken on purpose:** a fake camera that streams on after it is stopped, which the suite must catch.

---

## P1.09b: The stage, the objectives and the light

| | |
|---|---|
| Crates | `tetera-hal` |
| Depends on | P1.09a |
| ADRs | — |
| Gates | G1 |
| Second estimate | code 600 (the `Stage` trait, its capabilities, readings and events 180; the fake stage, its timed moves, jogs, homing and faults 250; the objective catalogue and mount 100; illumination 70) · tests 750 (the stage's contract suite 250; a stage flawed on purpose for each check, 200; unit tests 300) · examples 80 · docs-as-code 20 · glue 40 (the `toml` dependency, `configs/objectives.toml`) = **1,490** (1,937 at ×1.3). P1.09a's contract suite needed tests of its own, so this one has them from the start |

**Goal.** One interface for the stage, and the objectives and the light described.

**Scope.**

- **`Stage`:** capabilities (axes, travel, speeds, homing); move to and by; jog and stop jogging; stop; home; position with its timestamp; state changes.
- **`ObjectiveMount`:** the catalogue from `configs/objectives.toml` (the four Mitutoyo objectives: 5× NA 0.14, 10× NA 0.28, 20× NA 0.42, 50× NA 0.55, with working distances); which one is fitted, set by the user.
- **`Illumination`:** channels, on and off, intensity, colour temperature; a manual implementation.
- **The fake stage:** instant or timed moves, with faults on demand.
- **The stage's contract suite,** run against the fake now, and the virtual and real stages later.

**Out of scope.** The simulator (P1.10–P1.12); drivers (M5).

**Try it.** `hal-fakes` drives the fake stage too, and prints its events.

**Done when.** The fake stage passes the contract suite, and a fake broken on purpose fails it.

**Tests.** The stage's contract suite; faults; the catalogue; configurations round-tripping. **Broken on purpose:** a fake that reports a move finished before it is, which the stage suite must catch.

**Open questions.** Q24 (the objectives), Q25 (the light).

---

## P1.10a: The simulated clock and motion profiles

| | |
|---|---|
| Crates | `tetera-sim` (new, L2) |
| Depends on | P1.09b |
| ADRs | 0014 the simulated clock |
| Gates | G1 |
| Second estimate | code 450 (the clock 200; motion profiles 250) · tests 500 (property tests of the profiles 250; the clock 150; unit tests 100) · examples 80 · docs-as-code 20 · glue 30 = **1,080** (1,404 at ×1.3). The agent split P1.10 (second estimate about 2,510, 3,263 at ×1.3) on 2026-11-27, under the owner's standing approval, into P1.10a and P1.10b |

**Goal.** Time that tests can control, and motion the way a real stage makes it.

**Scope.**

- **`SimClock`** implements `Clock`: real time at 1×, faster at N×, or stepped by hand. Simulated devices wait on it, and a step wakes them.
- **Motion profiles:**
  - a straight move from rest to rest, trapezoidal, or triangular when it is short, keeping each axis to its own acceleration and fastest speed, and to the speed asked for;
  - a feed hold, which decelerates to rest from wherever the move has got to;
  - position and speed at any time, worked out from the time alone, so the same times give the same positions, bit for bit.

**Out of scope.** The virtual stage (P1.10b); the sample and the camera (P1.11, P1.12).

**Try it.** `sim-clock`: the clock at 1×, faster and stepped, and a move's profile, sampled in stepped time.

**Done when.**

- Property tests hold for any move: it starts and ends at rest, where it should; no axis goes faster or accelerates harder than it may; it never goes back; its duration is the trapezoid's.
- A held move stops within its deceleration distance, and never goes back.
- The clock never goes back, at any pace; a step wakes what waits on it.

**Tests.** Property tests of the profiles and the feed hold; the clock at each pace; waiters woken by steps. **Broken on purpose:** a profile that starts decelerating late, which the property tests must catch.

---

## P1.10b: The virtual stage

| | |
|---|---|
| Crates | `tetera-sim` |
| Depends on | P1.10a |
| ADRs | — |
| Gates | G1 |
| Second estimate | code 670 (its configuration 80; the plan of legs and grblHAL's states 150; moves, jogs, holds, homing and faults 200; reports, the track and events 120; its thread 40; the `Stage` trait 80) · tests 650 (the contract suite at 1×, 10× and from an alarm; a move's duration in stepped time; the same track twice; each state's transitions; each fault; a jog cancelled mid-move; reports' latency) · examples 120 · docs-as-code 20 · glue 20 = **1,480** (1,924 at ×1.3) |

**Goal.** A stage that moves the way the real one does, in time that tests can control.

**Scope.**

- **`VirtualStage`** implements `Stage`:
  - trapezoidal motion, with acceleration and maximum speed per axis (P1.10a's profiles);
  - grblHAL's states: Idle, Jog, Run, Hold, Home, Alarm;
  - soft limits; a homing sequence that takes time;
  - position reports at a set rate, with latency, kept in a timestamped track;
  - faults on demand: an alarm, a limit hit, a stall.

**Out of scope.** The sample and the camera (P1.11, P1.12).

**Try it.** `virtual-stage`: a move's position against time, at 1× and stepped.

**Done when.**

- `VirtualStage` passes the stage contract suite.
- In stepped time, a move's duration matches the trapezoid's within 1 ms.
- The same steps give the same track, bit for bit.

**Tests.** The state machine; each fault; jogging cancelled mid-move. **Broken on purpose:** a soft limit, and a state transition.

---

## P1.11a: The H&E sample, on the CPU

| | |
|---|---|
| Crates | `tetera-sim` |
| Depends on | P1.04b |
| ADRs | 0015 simulated samples on the GPU |
| Gates | G1 |
| Spec-first | Yes |
| Second estimate | code 820 (hashing and noise 150; the section and its tissue 450; colour 60; rendering a region 160) · tests 450 · examples 160 (with a small PNG writer) · docs-as-code 40 · glue 20 = **1,490** (1,937 at ×1.3). The agent split P1.11 (second estimate about 2,830, 3,679 at ×1.3) on 2026-11-27, under the owner's standing approval, as its ready split proposed |

**Goal.** A slide of H&E tissue that looks right at every magnification and has depth, with its CPU twin the reference for the GPU.

**Scope.**

- **The model,** as a function of position and depth:
  - a tissue mask from large-scale noise;
  - nuclei placed by hashing a jittered grid, with noisy ellipsoids and chromatin texture;
  - stroma and cytoplasm as oriented fibres;
  - glands with lumens, vessels with red cells, and fat;
  - a section 3–5 µm thick, with nuclei at depths, an optional tilt, and folds.
- **Colour** through absorbance: haematoxylin and eosin optical densities and stain vectors (Ruifrok and Johnston, 2001), turned into transmitted light by Beer–Lambert.
- **Procedural texture** from the literature: improved noise (Perlin, 2002) and cellular noise (Worley, 1996).
- **Deterministic from a seed:** integer hashes, so the same region is the same bits.
- **Rendering a region** at a pixel size and a depth range, as stain densities and transmitted colour.
- **The tolerance the GPU must keep** (P1.11b), stated in the spec.

**Out of scope.** The GPU, tiles and their cache (P1.11b); optics and the camera (P1.12); calibration targets (P1.18).

**Try it.** `he-sample`: regions rendered at the sampling of 5×, 20× and 50×, as PNGs, with depth slices and timings.

**Done when.**

- The behaviour spec came first.
- The same seed gives the same bits; a seed one bit different gives a different image.
- Depth orders as the spec says: a nucleus is darkest at its own depth.
- The PNGs are written for the owner to judge at Checkpoint A.

**Tests.** Hashing and noise; the model's parts; depth ordering; determinism; colour. **Broken on purpose:** the seed's hashing (a changed bit must change the image).

**Open questions.** Q26 (H&E only, in 3D).

---

## P1.11b: The H&E sample on the GPU, and its tiles

| | |
|---|---|
| Crates | `tetera-sim`, `tetera-compute` |
| Depends on | P1.11a |
| ADRs | — (0015) |
| Gates | G1, G1c |
| Spec-first | Yes: P1.11a's spec states its tolerance and tiles |
| Second estimate | code 800 (the model's CUDA kernels, its cells and its pixels, 450; launching them, and half precision, 100; tiles, levels of detail and the cache within a VRAM budget 250) · tests 400 (the twins 150; the cache, testable without a GPU, 150; tiles and determinism 100) · examples 80 · docs-as-code 30 · glue 30 = **1,340** (1,742 at ×1.3). The kernels and their launch live in `tetera-compute`, the one crate allowed unsafe code; `tetera-sim` wraps them |

**Goal.** The same slide made on the GPU as fast as the camera needs it.

**Scope.**

- **The model on the GPU,** the same functions as the CPU twin, in a CUDA kernel.
- **Tiles on demand:** keyed by level of detail, position and depth layer; half precision on the GPU; an LRU cache within a VRAM budget.
- **The twins agree** within the tolerance P1.11a's spec states.

**Out of scope.** Optics and the camera (P1.12).

**Try it.** `he-sample` with `--features cuda`: the same regions on each backend, with timings.

**Done when.**

- The same seed gives the same tile bits on the GPU.
- The backends agree within tolerance.
- A 512 × 512 tile's time on the GPU is measured and recorded.
- The owner judges the PNGs plausible at Checkpoint A.

**Tests.** The twins; the cache's eviction within its budget; tiles stitching without seams. **Broken on purpose:** the cache's budget.

---

## P1.12a: Optics and light: the field the sensor sees

| | |
|---|---|
| Crates | `tetera-sim`, `tetera-compute` |
| Depends on | P1.06b, P1.10b, P1.11b |
| ADRs | 0016 the virtual camera's model |
| Gates | G1, G1c |
| Second estimate | code 850 (the field of view and depth groups 120; blur, vignetting and illumination 150; the field on the CPU 150; on the GPU, resampling, blurring and lighting kernels 250 and their launch 100; `VirtualIllumination` 80) · tests 450 · examples 100 · docs-as-code 40 · glue 30 = **1,470** (1,911 at ×1.3). The agent split P1.12 (second estimate about 2,960, 3,848 at ×1.3) on 2026-11-27, under the owner's standing approval, as its ready split proposed |

**Goal.** The light that reaches the sensor: the simulated slide through the fitted objective, its defocus and the illumination, on the GPU and on the CPU.

**Scope.**

- **The field of view** from the magnification, the 2.4 µm pixels and the camera's mounting angle; centred on where the stage itself is.
- **Optics:** the section in depth groups as thick as the depth of field allows; each group blurred by its defocus (a Gaussian approximation from the NA and the distance to focus), plus the in-focus diffraction blur; vignetting.
- **Simulated illumination:** intensity; colour temperature (blackbody weights per channel); a field uneven by a set amount. `VirtualIllumination` implements `Illumination`.
- **The field** on the GPU from the tile cache, and on the CPU from a rendered region, resampled into the sensor's grid the same way, so the twins agree.
- **Changing the objective** changes the field of view and the blur.

**Out of scope.** The sensor, its noise, modes and rates, motion and the rolling shutter, and `VirtualCamera` (P1.12b).

**Try it.** `virtual-optics`: the field at 5×, 20× and 50×, through focus, under two lights, as PNGs, with timings on each backend.

**Done when.**

- The field of view matches the magnification and the mounting angle.
- The blur grows with the distance to focus, as the spec's model says.
- The twins agree within the stated tolerance.
- A full 5472 × 3648 field's time on the GPU is measured, still and while the stage moves, against the 40 ms budget (§6.15).

**Tests.** The field of view against the magnification and angle; the blur against the distance to focus; vignetting and the uneven field; blackbody weights; the twins. **Broken on purpose:** the defocus model's sign (a field in focus must be the sharpest).

---

## P1.12b: The sensor and motion

| | |
|---|---|
| Crates | `tetera-sim`, `tetera-compute` |
| Depends on | P1.12a |
| ADRs | — (0016) |
| Gates | G1, G1c |
| Second estimate | code 600 (the sensor on the CPU 200; its kernel 120 and launch 60; motion and the rolling shutter in the field, both backends, 120; binned modes' fields 40; the stage's path 60) · tests 400 · examples 100 · docs-as-code 30 · glue 20 = **1,150** (1,495 at ×1.3). The agent split P1.12b (second estimate about 1,660, 2,158 at ×1.3) on 2026-11-27, under the owner's standing approval, into P1.12b and P1.12c |

**Goal.** Raw frames from the field: the MIChrome 20's sensor, with the stage moving during the exposure.

**Scope.**

- **The sensor:** RGGB sampling; quantum efficiency per channel; shot, read and dark noise, from a seed; full well; black level; gain; 12 significant bits in 16-bit words, or 8 bits.
- **Motion:** the stage's path during the exposure, as the field's average over the path; a rolling shutter, each row exposed a line time after the one above.
- **Binned modes:** the field at the binned pixel, its charge the sum of the pixels binned.
- **On the GPU and the CPU,** each frame the same bits from the same seed on the same backend, and the two backends the same statistics.

**Out of scope.** The camera's modes, rates and timing, `VirtualCamera`, and Checkpoint A (P1.12c).

**Try it.** `virtual-sensor`: raw frames still and moving, their noise against the model, and a rolling shutter's skew, as PNGs.

**Done when.**

- Noise statistics match the model: each channel's mean and variance against the photons, the read noise and the dark current.
- A moving stage smears the field along its path, and a rolling shutter skews it by the line time.
- With a seed, a frame repeats bit for bit on the same backend.

**Tests.** Noise statistics; full well and the black level; bit depths; motion blur's length; rolling-shutter skew; binning. **Broken on purpose:** the noise model's variance.

---

## P1.12c: The virtual camera (Checkpoint A)

| | |
|---|---|
| Crates | `tetera-sim` |
| Depends on | P1.12b |
| ADRs | — (0016) |
| Gates | G1, G1c, **G2 (Checkpoint A)** |
| Second estimate | code 700 (modes, rates and capabilities 100; the frame loop on the clock, with streaming, snaps, triggers and drops, 250; the scene, from the stage, the objective and the light, with the motion during each frame, 120; rendering on either backend into the frame pool 120; the SDK's processing 60; configuration 50) · tests 450 · examples 200 · docs-as-code 60 · glue 30 = **1,440** (1,872 at ×1.3). The first, 1,190, came from P1.12b's split |

**Goal.** A MIChrome 20 that images the simulated slide through the fitted objective, live, at its real modes and rates.

**Scope.**

- **Modes and rates** (OScope's measurements): mode 0, 5472 × 3648, 7.24 fps in 16 bits and 14.49 in 8; mode 1, 2736 × 1824, 26.57 and 53.14; mode 2, 1824 × 1216, 33.87 and 67.26. Frames are timed on the simulated clock.
- **Optionally, the SDK's processing** (gamma, white balance, rescaling with an offset near 2,000), to mimic the real raw.
- `VirtualCamera` implements `Camera`, from the stage, the objective, the light and the sample; changing the objective changes the field and the blur.
- **A CPU fallback,** with a scaled-down sensor for tests.

**Out of scope.** The server (P1.13a, P1.13b); the preview (P1.14a, P1.14b).

**Try it.** `virtual-camera`: live frames while the stage moves at 1×, on the GPU, written as a strip of JPEGs, with the rate each mode reaches. **Checkpoint A's demo script** is in the notes.

**Done when.**

- A full frame takes 40 ms or less on the GPU, still or while jogging (plan §6.15), or the notes say where it falls short and by how much.
- Each mode reaches its rate in real time on this PC, or the notes say which falls short and by how much.
- With a seed and stepped time, frames repeat bit for bit on the same backend.
- The camera passes the contract suite.

**Tests.** Mode timing; the contract; frames on a stepped clock. **Broken on purpose:** the mode's frame interval.

**Open questions.** Q27 (linear by default; the SDK's processing optional; rolling shutter modelled).

---

## P1.13a: The server, configuration and systemd

| | |
|---|---|
| Crates | `tetera-api` (new, L5), `tetera-server` (new, L6) |
| Depends on | P1.12c |
| ADRs | 0017 control plane and media plane · 0018 the service in WSL2: systemd, start-up, networking, devices |
| Gates | G1 |
| Second estimate | code 730 (the command line, configuration, data roots and shutdown 250; the instrument assembled from configuration 150; device errors as teta-wot's 60; the `system` Thing 150; the `simulation` Thing, and the camera's sample swapped while it runs, 120) · tests 350 · examples 0 · docs-as-code 200 (`configs/tetera.toml`, the unit, the READMEs, the Windows notes) · glue 40 = **1,320** (1,716 at ×1.3). The agent split P1.13 (first estimate 1,980, 2,574 at ×1.3) on 2026-11-27, under the owner's standing approval, into P1.13a and P1.13b, as its ready split proposed |

**Goal.** The virtual instrument served over HTTP by a service that runs under systemd in WSL: the server, its configuration, and the Things that describe the system and drive the simulation.

**Scope.**

- **`tetera-server`:**
  - a command line (`clap`), and `configs/tetera.toml`: address and port (127.0.0.1:8310), data root, scratch root, simulator, objectives, logging;
  - data roots: `./.tetera/` in development, `/var/lib/tetera/` for the service;
  - graceful shutdown.
- **`tetera-api`,** on teta-wot v0.2.0:
  - the instrument, assembled from the configuration: the simulated clock, the virtual stage, camera, objective mount and light, the sample, on the GPU when there is one;
  - device errors mapped to teta-wot's error responses;
  - `system`: version, WSL's version, the kernel and distribution, GPU, disks, memory;
  - `simulation`: the sample (load, unload, seed), imperfections on demand.
- **systemd:** `packaging/linux/tetera.service`, with restart on failure, and how to install it for development (the owner runs the `sudo` steps). The agent tests it as a user unit.
- **Windows:** how to reach the service from a browser; how a Windows task can start WSL at logon. Hardened in P5.07.
- **`.vscode/launch.json`** gains the server's debug configuration.

**Out of scope.** The camera, stage, objective and illumination Things (P1.13b); the preview (P1.14a, P1.14b); jobs (P1.15a); status, logs and power (P1.20).

**Try it.** `cargo run -p tetera-server`; teta-wot's `/docs`; `curl` for `system` and `simulation`.

**Done when.**

- `system` and `simulation` serve their Thing Descriptions, and the OpenAPI document lists them.
- The configuration's defaults hold, and a bad configuration is refused with what is wrong.
- Under systemd in WSL, the service restarts after it is killed.
- A browser on Windows reaches http://localhost:8310.

**Tests.** Each Thing through teta-wot's in-process test harness; configuration defaults and errors; the server on a port the system picks, shut down gracefully. **Broken on purpose:** the unit's restart on failure.

**Open questions.** Q11 (localhost only), Q12 (start-up), Q48 (port 8310).

---

## P1.13b: The device Things

| | |
|---|---|
| Crates | `tetera-api` |
| Depends on | P1.13a |
| ADRs | — (0017) |
| Gates | G1 |
| Second estimate | code 700 (`camera` 230: its configuration, capabilities, white balance and a stream to watch; `stage` 230: position, state and speed kept current, moves that wait and can be cancelled; `objective` 60; `illumination` 90; a stepped clock for tests, and the Things registered, 90) · tests 550 · docs-as-code 80 (a README with `curl` for each Thing) · glue 20 = **1,350** (1,755 at ×1.3). The first, 1,100, came from P1.13's split; P1.13a measured ×1.91 its estimate, and this one allows for its tests |

**Goal.** The virtual instrument's devices as Things, over HTTP.

**Scope.**

- `camera`: mode, exposure, gain, pixel format, binning, ROI, white balance, streaming;
- `stage`: position, moving, state, speed; move to, move by, jog, stop, home;
- `objective`: the catalogue, which is fitted;
- `illumination`: on, intensity, colour temperature;
- device errors mapped to teta-wot's error responses, the stage's alarm among them.

**Out of scope.** The preview (P1.14a, P1.14b); jobs (P1.15a).

**Try it.** `cargo run -p tetera-server`; a README with `curl` for each Thing, including a move and its timing.

**Done when.**

- Each Thing serves its Thing Description, and the OpenAPI document is complete.
- A move through HTTP takes the time the virtual stage says.
- Watching `position` over WebSocket shows it change during the move.

**Tests.** Each Thing through teta-wot's in-process test harness; a move timed on a stepped clock. **Broken on purpose:** the error mapping for a stage alarm.

---

## P1.14a: The frame bus and the preview pipeline

| | |
|---|---|
| Crates | `tetera-acquire` (new, L4) |
| Depends on | P1.08d, P1.13b |
| ADRs | — (the frame bus's design is in ADR-0008, from P1.03) |
| Gates | G1, G1c |
| Second estimate | code 710 (the frame bus, its two kinds of subscriber and the first frame after a time 200; the preview session, its parameters, the live recipe, JPEG, the slow link's hook and each stage's time 280; the live statistics 220; the binned live mode 10) · tests 450 · examples 200 · docs-as-code 40 · glue 30 = **1,430** (1,859 at ×1.3). The first, 1,050, from the owner's split of P1.14 on 2026-11-26 |

**Goal.** Get every frame from the camera to whatever needs it, and turn frames into preview JPEGs and live statistics, before any network is involved.

**Scope.**

- **The frame bus:**
  - *lossless* subscribers, with a bounded queue and backpressure (the recorder, in P1.15c);
  - *latest-wins* subscribers, with a one-frame mailbox (preview, statistics);
  - "the first frame exposed after t", for workflows.
- **The preview pipeline,** as a session object with no transport yet:
  - its parameters: viewport size, ROI, quality, maximum rate, changeable mid-stream;
  - crop or bin by the viewport, then P1.06b's fused kernel through a live recipe, then JPEG;
  - a hook to lower quality or size, for P1.14b's slow links;
  - each stage's time, measured per frame.
- **Live statistics,** at up to 10 Hz, from the raw data: a histogram per channel, the saturation mask at preview size, line profiles, ROI statistics, a focus value.
- **The camera's live mode** is binned by default (D5).

**Out of scope.** The WebSocket (P1.14b); MJPEG and the HTML page (P1.14c); recording (P1.15c).

**Try it.** `preview-pipeline`: the virtual camera into the bus, two preview sessions of different sizes and the statistics, with each session's rate and each stage's time, and a few JPEGs written out.

**Done when.**

- In mode 1 at 8 bits, each session keeps up with the camera's rate, at least 30 fps, and each stage's time is recorded.
- The statistics match a reference computed from the raw frame.
- A latest-wins subscriber drops frames when slow; a lossless one slows its publisher instead.

**Tests.** The bus: backpressure, and dropping; statistics against references; parameters changed mid-stream; the stress runs. **Broken on purpose:** the mailbox, and the backpressure.

**Open questions.** Q17 (binned live mode), Q18 (JPEG).

---

## P1.14b: The preview stream

| | |
|---|---|
| Crates | `tetera-acquire`, `tetera-api` |
| Depends on | P1.14a |
| ADRs | 0019 the preview protocol |
| Gates | G1 |
| Second estimate | code 560 (the protocol: each frame's header, the client's state, the acknowledgement window and the slow link's control 230; the `preview` Thing: its defaults, sessions and stream 280; the camera's stream into the bus, and the instrument's bus and compute 50) · tests 560 · examples 150 · docs-as-code 40 · glue 20 = **1,330** (1,729 at ×1.3). P1.14b's own, 1,150 at first, came to 2,080 (2,704 at ×1.3) with MJPEG and the page: the agent split it on 2026-11-28 under the owner's standing approval, and they became P1.14c |

**Goal.** Live images from the virtual camera over a WebSocket, at the camera's rate, with little delay, to any client.

**Scope.**

- **The binary WebSocket,** the `preview` Thing's stream:
  - each frame with a header: sequence number, exposure timestamps, size, format;
  - JSON messages from the client: viewport, ROI, quality, rate, acknowledgements.
- **Acknowledgements:** no more than two frames sent but not yet shown, so no socket buffer holds stale frames.
- **Slow links:** P1.14a's hook lowers quality or size.
- **The `preview` Thing's properties:** its defaults, and the sessions open.
- **The camera's stream** goes into the frame bus.

**Out of scope.** MJPEG and the HTML page (P1.14c); the Vue app's viewer (M4).

**Try it.** `preview-stream`: the server on a loopback port, a quick client and a slow one, with each one's rate, latency and link.

**Done when.**

- In mode 1 at 8 bits, a client that keeps up receives the camera's rate, at least 30 fps.
- From the exposure's end to a client's receipt takes 100 ms or less on localhost, measured from the timestamps in each frame's header.
- A slow client never receives a stale frame, and its link steps down.

**Tests.** A WebSocket client in the tests, also slowed down on purpose; the acknowledgement window; malformed client messages refused. **Broken on purpose:** the acknowledgement window, and the stale-frame check.

**Open questions.** None.

---

## P1.14c: MJPEG and the live-view page

| | |
|---|---|
| Crates | `tetera-api` |
| Depends on | P1.14b |
| ADRs | — (ADR-0019, from P1.14b) |
| Gates | G1, **G2 (Checkpoint B)** |
| Second estimate | code 230 (MJPEG 120, the live statistics on the `preview` Thing 50, the page's route and the server's wiring 60) · tests 230 · examples 300 (the page) · docs-as-code 40 · glue 10 = **810** (1,053 at ×1.3). The first, 750, from P1.14b's split on 2026-11-28 |

**Goal.** Live images from the virtual camera in a browser, at the camera's rate, with little delay.

**Scope.**

- **MJPEG,** for plain `<img>` pages and other tools.
- **The live statistics** on the `preview` Thing, for the page's histogram.
- **`live-view`:** one plain HTML page, served by the server.

**Out of scope.** The Vue app's viewer (M4).

**Try it.** `live-view`: one plain HTML page (no framework) with the live image, a histogram, and buttons that jog the stage. **Checkpoint B's demo script** is in the notes.

**Done when.**

- In mode 1 at 8 bits, the page shows the camera's rate, at least 30 fps.
- From the exposure's end to the page's paint takes 100 ms or less on localhost, measured from the timestamps in each frame's header.
- The MJPEG stream plays in an `<img>`.

**Tests.** MJPEG's parts and its end with the server; the statistics; the page served. **Broken on purpose:** MJPEG's part boundary.

**Open questions.** Q23 (MJPEG as a stream).

---

## P1.15a: The job service

| | |
|---|---|
| Crates | `tetera-jobs` (new, L3), `tetera-api` |
| Depends on | P1.13b |
| ADRs | 0020 the job service |
| Gates | G1 |
| Second estimate | code 620 (the job model 90; the service, its hardware queue, its compute pool on a VRAM budget, and cancel 260; the history in SQLite and its pruning 140; the `jobs` Thing 130) · tests 620 · examples 100 · docs-as-code 40 · glue 30 = **1,410** (1,833 at ×1.3). The first, 800, from the owner's split of P1.15 on 2026-11-26 |

**Goal.** Long work that can be followed, cancelled and looked up afterwards.

**Scope.**

- **A job:** id, kind (hardware or compute), state, progress and message, log lines, result.
- **Hardware jobs** one at a time from a queue; **compute jobs** on a pool that shares the GPU, through P1.04b's VRAM budgets.
- **Cancel,** through tokens that a job checks.
- **The history** in SQLite (`rusqlite`, bundled), pruned by age.
- **The `jobs` Thing:** list, get, cancel, with events.

**Out of scope.** The work itself: captures (P1.15d), autofocus (P1.16a, P1.16b), scans (P1.17a, P1.17b), batches (P1.19).

**Try it.** `jobs`: hardware and compute jobs queued, one cancelled, then the history read back after a restart.

**Done when.**

- Hardware jobs never overlap.
- A cancelled job stops within a second.
- The history survives a restart, and old entries are pruned.

**Tests.** State transitions; cancelling at each state; the queue's order; the pool's limit; the history across a restart. **Broken on purpose:** cancel, and the one-at-a-time rule.

---

## P1.15b: The dataset store

| | |
|---|---|
| Crates | `tetera-store` (new, L2) |
| Depends on | P1.15a |
| ADRs | 0021 datasets on disk |
| Gates | G1 |
| Second estimate | code 520 (the SQLite index of projects, datasets and files 150; each dataset's folder, its metadata and thumbnail 120; OME-Zarr v3 through `zarrs`, written and read back 200; the store 50) · tests 520 · examples 80 · docs-as-code 40 · glue 30 = **1,190** (1,547 at ×1.3). P1.15b's own second estimate, with the recorder, ingest and capture, was 3,030 (3,939 at ×1.3): the agent split it on 2026-11-28 under the owner's standing approval into P1.15b, P1.15c and P1.15d |

**Goal.** Datasets on disk, in projects, that read back exactly, and an index that never lists a dataset half written.

**Scope.**

- **Projects and datasets** (`tetera-store`):
  - the SQLite index, in `tetera.db`: projects, datasets, files;
  - each dataset's folder: OME-Zarr v3 arrays (through `zarrs`), compressed with zstd; `metadata.json` (instrument, camera settings, objective, calibration, positions, times); a JPEG thumbnail;
  - raw Bayer stored as one channel, with its pattern in the metadata;
  - a dataset is listed once it is complete; one abandoned while it is written leaves nothing behind.

**Out of scope.** The recorder and ingest (P1.15c); capture and the Things (P1.15d); exports (P1.19).

**Try it.** `datasets`: a project, a dataset of synthetic raw frames written and read back, its tree on disk, and one abandoned half way.

**Done when.**

- A dataset's OME-Zarr reads back exactly, and its metadata follows OME-NGFF 0.5.
- The index survives a restart, and lists no dataset abandoned while it was written.

**Tests.** OME-Zarr round trips, 8 and 16 bits; the NGFF metadata; the index across a restart; abandoning a dataset; deleting one. **Broken on purpose:** the index's clean-up after an abandoned dataset.

**Open questions.** Q14 (the scratch disk), Q36 and Q37 (answered: OME-Zarr, in projects).

---

## P1.15c: The recorder and ingest

| | |
|---|---|
| Crates | `tetera-store`, `tetera-acquire` |
| Depends on | P1.14a, P1.15b |
| ADRs | — (ADR-0021, from P1.15b) |
| Gates | G1 |
| Second estimate | code 420 (the frame log: preallocated segments and their index 200; the recorder, a lossless subscriber 120; ingest, cancellable, 100) · tests 420 · examples 100 · docs-as-code 30 · glue 10 = **980** (1,274 at ×1.3). The first, 880, from P1.15b's split on 2026-11-28 |

**Goal.** Every frame of a capture onto the scratch disk at the camera's full rate, then into a dataset.

**Scope.**

- **The scratch recorder:** preallocated segment files, raw frames appended in order, with an index; a lossless subscriber of the frame bus.
- **Ingest,** as a compute job: from the frame log into a P1.15b dataset.

**Out of scope.** Capture and its Things (P1.15d).

**Try it.** `recorder`: the virtual camera at its full rate into the frame log, then ingested into a dataset.

**Done when.**

- The recorder keeps up with the virtual camera at its full rate, 289 MB/s, on ext4, without a dropped frame.
- Ingest writes a dataset that reads back exactly.

**Tests.** The recorder under backpressure; the frame log read back; ingest, and cancelling it. **Broken on purpose:** the recorder's backpressure (a dropped frame must fail the test).

**Open questions.** Q14 (the scratch disk).

---

## P1.15d: Capture and its Things

| | |
|---|---|
| Crates | `tetera-workflows` (new, L4), `tetera-api` |
| Depends on | P1.15c |
| ADRs | — |
| Gates | G1 |
| Second estimate | code 480 (capture: one frame, N at an interval, or a burst through the recorder, the mode switch and back, and the clean-up after a cancel 200; the `capture` and `datasets` Things and the thumbnail's route 220; the store and the live stream's sink on the server 60) · tests 480 · examples 80 · docs-as-code 40 · glue 20 = **1,100** (1,430 at ×1.3). The first, 960, from P1.15b's split on 2026-11-28 |

**Goal.** Captures that land on disk as datasets, in projects, without losing a frame.

**Scope.**

- **Capture** (`tetera-workflows`): one frame, N frames at an interval, or a burst at the full rate, as a hardware job. **It switches the camera from its binned live mode to full resolution, captures, and switches back** (D5), and the switch's time is measured.
- **Things:** `capture`, and `datasets` (projects, datasets, metadata, thumbnails, delete).

**Out of scope.** Exports (P1.19); scans (P1.17a).

**Try it.** `capture`: a burst of 20 full frames into a new project, then the dataset's tree on disk, read back.

**Done when.**

- A capture switches the mode and back.
- A capture cancelled through the `jobs` Thing leaves no half-written dataset in the index.

**Tests.** Mode switching; a capture of each kind; cancelling a capture. **Broken on purpose:** the clean-up after a cancelled capture.

**Open questions.** None.

---

## P1.16a: Focus measures and the search

| | |
|---|---|
| Crates | `tetera-imaging`, `tetera-workflows` |
| Depends on | P1.15d |
| ADRs | — |
| Gates | G1 |
| Spec-first | Yes: the behaviour spec, in the notes, covers P1.16b too |
| Second estimate | code 480 (the green channel, binned 60; the four measures 120; the search: its range and steps from the depth of field 40, the sweeps and the take-up against backlash 80, the fit 60, no focus 40; the probe it drives and its result 80) · tests 480 · examples 80 · docs-as-code 40 · glue 10 = **1,090** (1,417 at ×1.3). The first, 1,300, was P1.16's whole, split by the agent on 2026-11-28 |

**Goal.** Measures that peak at focus, and a search that finds the peak quickly and reliably, against backlash, on any curve.

**Scope.**

- **Focus measures** (`tetera-imaging`) on the raw green channel, binned: Brenner's gradient, the variance of the Laplacian, normalised variance, Tenengrad. The spec picks the default from the literature (Groen, Young and Ligthart, 1985; Sun, Duthaler and Nelson, 2004).
- **The search** (`tetera-workflows`): a coarse sweep, then a fine one around the peak and a parabola through it, always approaching from one side, against backlash. The range and the steps come from the objective's depth of field.
- **The probe** it drives: a z move that ends when the stage is still, and a measure. The tests' probes are synthetic; P1.16b's is the instrument.
- **No focus:** on a flat curve, or a peak at the range's end, it reports that it found none, and moves back where it started.

**Out of scope.** The instrument's probe, the job and the `autofocus` Thing (P1.16b). Focus maps and tracking (M3's workflows).

**Try it.** `focus-search`: each measure through a synthetic stack, where it peaks; then the search, 100 times, on noisy known curves through a stage with backlash, with its errors.

**Done when.**

- The spec came first.
- Each measure peaks at a synthetic stack's focus.
- On known curves with noise, through a probe with backlash, the search's error is a quarter of the depth of field or less in at least 99 of 100 trials.
- A flat curve fails cleanly, the probe back where it started.

**Tests.** Each measure's peak on synthetic stacks; the search on known curves with noise; backlash; a flat curve, and a peak at the range's end. **Broken on purpose:** the approach direction, and the flat-curve check.

**Open questions.** None.

---

## P1.16b: Autofocus on the instrument

| | |
|---|---|
| Crates | `tetera-workflows`, `tetera-api` |
| Depends on | P1.16a |
| ADRs | — |
| Gates | G1 |
| Second estimate | code 360 (the instrument's probe: a z move until the stage is still, the first frame exposed after it, and its measure 150; the job and its result 50; the `autofocus` Thing 120; the depth of field from the objective fitted 40) · tests 360 · examples 120 · docs-as-code 30 · glue 10 = **880** (1,144 at ×1.3), from P1.16's split on 2026-11-28 |

**Goal.** Find focus quickly and reliably at every objective, on the instrument.

**Scope.**

- **Frames** from "the first frame exposed after t", so a moving stage never blurs a measurement: the live mode's stream, through the frame bus.
- **A job,** with the focus curve as its result; the `autofocus` Thing.
- **Failure:** on a blank field, it reports that it found no focus, and leaves the stage where it was.

**Out of scope.** Focus maps and tracking (M3's workflows).

**Try it.** `autofocus`: 100 starts at random distances from focus, at 5× and at 50×, with the error against the simulator's true focus; and a search's time at 10×.

**Done when.**

- In at least 95 of 100 trials at each objective, the error is a quarter of the depth of field or less.
- At 10×, it takes 5 s or less (plan §6.15).
- A blank field fails cleanly.

**Tests.** No frame exposed during a move is measured; autofocus on the virtual instrument; the blank field; the Thing over HTTP, and a cancel. **Broken on purpose:** the frame-after-t rule, a frame in flight measured.

**Open questions.** None.

---

## P1.17a: Scan planning and the stop-and-go engine

| | |
|---|---|
| Crates | `tetera-workflows`, `tetera-hal`, `tetera-sim` |
| Depends on | P1.16b |
| ADRs | 0022 scan planning and the scan engine |
| Gates | G1 |
| Second estimate | code 600 (the planners: an XY snake from a rectangle, the field of view and an overlap, in serpentine order, and a z-scan, by range and step or N planes around the focus 160; the plan's time, from the stage's speeds and accelerations, the settle, the frames and the writes 100; the stage's acceleration among its capabilities 20; the engine: move, settle, autofocus if asked, capture at full resolution, positions, pause, resume and cancel, and a dataset of tiles 320) · tests 600 · examples 120 · docs-as-code 40 · glue 10 = **1,370** (1,781 at ×1.3). The first, 1,530, was P1.17's whole, split by the agent on 2026-11-28 |

**Goal.** Acquire a grid of tiles, stop and go, with every frame's position recorded, in the time the plan says.

**Scope.**

- **Planners:**
  - **XY snake:** a rectangle covered with tiles at a set overlap, from the field of view, in serpentine order;
  - **z-scan:** a range and step, or N planes around the current focus.
- **The plan's preview:** positions, and the estimated time.
- **The engine:** move, wait until Idle and settled, optionally autofocus (none, first tile, every N tiles), capture at full resolution, next. **Pause, resume, cancel.**
- **Positions** from the stage's track at each exposure's midpoint.
- **A dataset of tiles,** each with its position.

**Out of scope.** z-stacks as 3D arrays and the `scan` Thing (P1.17b). Raster, position and tiled z-stack scans, and the histo scan (M3); stitching (M3).

**Try it.** `scan`: a 5 × 5 snake at 10×, with recorded positions against planned ones, and the time taken against the estimate.

**Done when.**

- Recorded positions agree with the simulator's truth within the stage's repeatability.
- A cancelled scan leaves a valid, partial dataset.
- The estimated time is within 10 % of the time taken at 1×.

**Tests.** The planners' coverage and order; overlap from the field of view; the engine's states; pause and cancel at every step. **Broken on purpose:** the serpentine order, and the settle wait.

**Open questions.** None.

---

## P1.17b: z-stacks and the `scan` Thing

| | |
|---|---|
| Crates | `tetera-workflows`, `tetera-store`, `tetera-api` |
| Depends on | P1.17a |
| ADRs | — (ADR-0022, from P1.17a) |
| Gates | G1 |
| Second estimate | code 360 (a dataset's z axis, with its step, written and read back as a 3D array 120; the z-scan on the engine 60; the `scan` Thing: the plan's preview, start, pause, resume, progress events, and the last scan's result 180) · tests 360 · examples 100 · docs-as-code 30 · glue 10 = **860** (1,118 at ×1.3), from P1.17's split on 2026-11-28 |

**Goal.** A stack of planes as one 3D array, and scans run, watched and steered over HTTP.

**Scope.**

- **Datasets:** a z-stack as one 3D array, its planes' step in OME-NGFF's scale.
- **The z-scan** run by P1.17a's engine.
- **The `scan` Thing,** with progress events: the plan's preview, start, pause, resume; cancel through the `jobs` Thing.

**Out of scope.** As P1.17a.

**Try it.** `scan`: a 21-plane z-scan, read back as a 3D array, with recorded positions against planned ones; and a snake run through the `scan` Thing, paused and resumed.

**Done when.**

- The z-scan reads back as a 3D array.
- A scan started through the `scan` Thing can be paused, resumed and cancelled, and says how far it has got.

**Tests.** The z axis written and read back; the Thing over HTTP, its events, pause, resume and cancel. **Broken on purpose:** the z step in the metadata.

**Open questions.** None.

---

## P1.18: Calibration

| | |
|---|---|
| Crates | `tetera-workflows`, `tetera-sim`, `tetera-store`, `tetera-imaging`, `tetera-api` |
| Depends on | P1.17b |
| ADRs | 0023 the calibration model |
| Gates | G1, G1c, **G2 (Checkpoint C)** |
| Spec-first | Yes |
| First estimate | code 1,300 · tests 500 · examples 100 = **1,900** |
| Ready split | P1.18a: targets, dark, flat, pixel size and rotation (~1,000) · P1.18b: colour, calibration sets, the `calibration` Thing, Checkpoint C (~900) |

**Goal.** Calibrate each objective from routines the user can run, and apply the results without being asked.

**Scope.**

- **Calibration targets in the simulator:** a blank field; a stage micrometer (a 10 µm grid); a 24-patch colour chart with known values.
- **Routines,** each a job:
  - **master dark:** N frames with the light off;
  - **master flat:** N frames on the blank field, slightly defocused (shading correction as in Peng and others, 2017);
  - **pixel size and the camera's rotation:** frames before and after known stage moves, registered by phase correlation (Kuglin and Hines, 1975), give the camera-to-stage affine;
  - **white balance and the CCM:** least squares from the chart's patches to their reference values; a root-polynomial fit (Finlayson, Mackiewicz and Hurlbert, 2015) as the spec's alternative.
- **Calibration sets,** per objective, with versions, in SQLite and files. The ISP uses the fitted objective's current set.
- **The `calibration` Thing:** sets, routines, and the steps a wizard follows (for M4).

**Out of scope.** Calibration on real optics (P5.04); measured PSFs (M5).

**Try it.** `calibrate`: every routine at 10×, with the recovered values against the simulator's truth. **Checkpoint C's demo script:** calibrate 10×, autofocus, and run a 5 × 5 scan.

**Done when.**

- The spec came first.
- Pixel size is recovered within 0.5 %, and rotation within 0.1°.
- After flat correction, the blank field is uniform within 1 %.
- The CCM's mean colour error over the chart is ΔE00 of 2 or less (Sharma, Wu and Dalal, 2005, for ΔE00).
- Changing the objective changes the calibration the preview uses.

**Tests.** Phase correlation on known shifts with noise; the affine fit; the CCM's fit on synthetic patches; sets and versions. **Broken on purpose:** the sign of the rotation, and the choice of set by objective.

**Open questions.** Q26 (calibration targets in the simulator).

---

## P1.19: Batch processing

| | |
|---|---|
| Crates | `tetera-workflows`, `tetera-pipeline`, `tetera-store`, `tetera-api` |
| Depends on | P1.15b, P1.08e |
| ADRs | — |
| Gates | G1, G1c |
| First estimate | code 900 · tests 400 · examples 80 · docs-as-code 20 = **1,400** |

**Goal.** Run recipes over datasets, keep the results with their provenance, and export them.

**Scope.**

- **Recipes over datasets,** as compute jobs: tiled, on the GPU.
- **Results** beside their dataset, with provenance: the recipe and its hash, each node's version, the input datasets, Tetera's version. The same inputs and recipe reuse an existing result.
- **The `pipeline` Thing:** the node catalogue; recipes saved, read, changed and deleted; runs on a dataset.
- **Exports:** OME-TIFF, with its OME-XML metadata; PNG and TIFF of images.

**Out of scope.** The Workspace's analyses and reports (M4); extension nodes (M2).

**Try it.** `batch`: denoising and colour correction over a scan's dataset, then an export to TIFF.

**Done when.**

- A tiled batch gives the same result as one untiled.
- The provenance is complete.
- An exported OME-TIFF reads back with its metadata; the manual test opens it in Fiji.

**Tests.** Provenance hashing; reuse of results; tiled against untiled; exports read back. **Broken on purpose:** the provenance hash (a changed parameter must change it).

---

## P1.20: Status, logs, power and settings

| | |
|---|---|
| Crates | `tetera-api`, `tetera-server`, `tetera-log` |
| Depends on | P1.15a |
| ADRs | 0024 power actions under WSL2 |
| Gates | G1 |
| First estimate | code 1,200 · tests 500 · examples 80 = **1,780** |
| Ready split | P1.20a: status and logs (~900) · P1.20b: power and settings (~880) |

**Goal.** Know the instrument's health, read its log, act on its power, and keep its settings.

**Scope.**

- **`status`:** each subsystem's health with its reasons: camera, stage, GPU (VRAM, temperature), disks (free space against thresholds), jobs, and WSL (clock drift, memory). A roll-up of OK, Degraded or Fault, with events when it changes.
- **`logs`:** the ring, searched by level, target, time and text; a live tail; the log files, listed and downloaded.
- **`power`,** four actions (D20):
  - restart the service: `systemctl restart`, allowed by a polkit rule for the service's account;
  - restart WSL: through a Windows task, started by interop;
  - reboot or shut down Windows: `shutdown.exe`, through interop;
  - each confirmed in two steps (a request, then a confirmation with its token within 30 s), logged, with a dry run that executes nothing.
- **`settings`:** data and scratch locations, log levels, display defaults; kept in TOML, with a version for migrations.

**Out of scope.** The Status Indicator and the Power tab (M4); extensions' health (M2).

**Try it.** `system`: the health, a log tail, and a dry-run reboot.

**Done when.**

- Faults injected into the virtual devices show in the health and its roll-up.
- A dry run executes nothing, and logs what it would have run.
- Restarting the service works under systemd in WSL.
- Settings survive a restart.
- The owner runs the real reboot and shutdown in the manual test.

**Tests.** The roll-up's rules; log queries; the two-step confirmation; the dry run; settings' migration. **Broken on purpose:** the dry-run switch (a test must catch a command that would run), and the confirmation's expiry.

**Open questions.** Q13 (answered: four actions).

---

## P1.21: Milestone 1 close

| | |
|---|---|
| Crates | `xtask` |
| Depends on | All of M1 |
| ADRs | — |
| Gates | **G3** |
| First estimate | code 700 · tests 200 · examples 50 = **950** |

**Goal.** Measure Milestone 1 against its budgets, run it for an hour, and look back.

**Scope.**

- **`cargo xtask perf`:** runs the budget examples in release builds, compares each with plan §6.15, and writes `docs/performance/m1-baseline.md`.
- **`cargo xtask soak --hours 1`:** the virtual instrument under a preview client, repeated scans, captures and batch jobs. It watches memory, VRAM, handles and disk, and fails on a growing trend.
- **G3** completed with both, and the licence report.
- **The manual tests:** the owner runs Checkpoints A, B and C, and the milestone's demo.
- **The retrospective:** `docs/m1/retrospective.md`, with estimates against measurements, phase by phase.
- **The plan:** M1's statuses, and anything M2 inherits.

**Try it.** `cargo xtask gate milestone`.

**Done when.**

- G3 passes, or each missed budget is explained in the baseline.
- The one-hour soak shows no growth.
- The retrospective is written, and the owner has run the checkpoints.

**Tests.** The soak's trend detection, on synthetic series; the perf comparison. **Broken on purpose:** a leak injected into a test build, which the soak must catch.
