# Tetera implementation plan

| | |
|---|---|
| Status | **Draft for the owner's review**, 2026-11-26. 16 questions answered the same day ([§14](#14-questions-for-the-owner)). Nothing is built until the owner approves it |
| Covers | Architecture for all five milestones, the phase overview of every milestone, the local gates, the toolchain set-up, risks, and the questions this draft needs answered |
| Inputs | [`goal.md`](goal.md) (the build specification, `tetera.md` until P1.01); `OScope` at `../OScope` (its plan, 62 ADRs, both retrospectives, `hardware.md`); `teta-wot` v0.2.0 at `../teta-wot`; this PC's toolchains, checked on 2026-11-26 |
| Next | Milestone 1's phase specs, drafted in [`m1/phases.md`](m1/phases.md) on 2026-11-26 for the owner's review; then phase P1.01 |

**How to review it.** Every proposal the owner hasn't confirmed yet is marked *(proposed)*. Every open question is in [§14](#14-questions-for-the-owner), with the answer this draft assumes. Answer there, and the plan changes to match, with a row in the [change log](#appendix-d-change-log).

## Contents

1. [Tetera in one page](#1-tetera-in-one-page)
2. [What Tetera takes from OScope](#2-what-tetera-takes-from-oscope)
3. [In and out of scope](#3-in-and-out-of-scope)
4. [Proposed decisions](#4-proposed-decisions)
5. [Ground rules](#5-ground-rules)
6. [Architecture](#6-architecture)
7. [How phases work](#7-how-phases-work)
8. [Local CI gates](#8-local-ci-gates)
9. [Milestones and phases](#9-milestones-and-phases)
10. [Planned ADRs](#10-planned-adrs)
11. [Documentation layout](#11-documentation-layout)
12. [Installing the toolchains](#12-installing-the-toolchains)
13. [Risks](#13-risks)
14. [Questions for the owner](#14-questions-for-the-owner)
15. [Appendices](#appendices)

---

## 1. Tetera in one page

Tetera is professional software for scientific lab instruments. Its first instrument is a motorised brightfield microscope:

- **Camera:** Tucsen MIChrome 20. Colour, 20 MP (5472 × 3648), 2.4 µm pixels, USB 3.1, 8- or 16-bit output.
  - OScope measured it in 2026 (its `hardware.md`): 14.49 fps at full resolution in 8 bits, 7.24 fps in 16 bits. That is about 289 MB/s either way.
  - Binned modes run faster: 53 fps at 2736 × 1824 and 67 fps at 1824 × 1216, both in 8 bits.
- **Stage:** XYZ motorised, driven by grblHAL over a serial port. XY moves for scanning, Z for focusing and z-scans.
- **Objectives:** a catalogue of objectives, each with its own calibration.
- **Compute module:** a Windows PC running WSL2 Ubuntu 24.04, with an NVIDIA GPU. The instrument service runs inside WSL2 under systemd. CUDA does the image processing.

**The software:**

- **Backend:** a layered Rust service. It serves the instrument over HTTP through `teta-wot` v0.2.0, as Web of Things *Things*.
- **The two streams:**
  - **Raw data** (the pristine 12-bit raw, in 16-bit words) stays in Rust memory, or goes straight to an NVMe scratch disk.
  - **The live view** gets small preview frames: binned, demosaicked and JPEG-encoded in Rust.
- **Processing:** a pipeline of nodes, on CUDA when a GPU is present and on the CPU otherwise. Stock nodes ship with Tetera. Third parties add nodes and workflows as extensions, written in Rust or Python.
- **Simulation:** when there is no real hardware, a virtual camera and a virtual stage image a simulated H&E tissue section. The section is generated on the GPU, with a CPU fallback.
- **Frontend:** a Vue 3 + TypeScript + Vite app in the browser, served by the backend. It uses reka-ui, pinia, Material Symbols and VTK.js, with OScope's themes and accent colours.

**The five milestones**

| Milestone | Delivers | Hardware |
|---|---|---|
| **M1: backend core** | Workspace and gates; core types; CPU and CUDA compute; the stock processing nodes and the pipeline engine; the HAL; the GPU simulator (H&E, optics, illumination, virtual camera and stage); the server and its Things; the live preview; jobs, capture and datasets; autofocus, snake scans, z-scans and calibration; status, logging and power | Virtual |
| **M2: extensions** | Extension packages; the Rust ABI and SDK; the Python runtime and SDK; extension points for nodes, workflows, artifacts and UI panels; the extension developer kit | Virtual |
| **M3: extension nodes and workflows** | First-party extensions, built only on the public SDK: XYZ→LAB, stitching, 2D deep zoom, depth from focus, 3D volume and object recognition as nodes; raster scan, position scan, tiled z-stack and histo scan as workflows | Virtual |
| **M4: frontend** | The Vue 3 app: every navRail tab in the spec, against the virtual instrument and the simulated samples | Virtual |
| **M5: real hardware and performance** | WSL2 device plumbing, the grblHAL and Tucsen drivers, calibration on real optics, real workflows, performance tuning, packaging, release 1.0 | Real |

## 2. What Tetera takes from OScope

OScope is the owner's previous build of the same instrument software. It ran three milestones from 2026-11-10 to 2026-11-25: about 134,000 lines in M1 and 78,000 in M2. Its M3, the real hardware, was never started. Tetera keeps what worked and changes what didn't.

**Tetera is a clean room** (D1, the owner's answer to Q1). It reuses OScope's ideas, lessons and look, described in its own words. None of OScope's code is copied or ported.

| OScope's lesson | Source | What Tetera does |
|---|---|---|
| Milestones were huge: 65 phases in M1, 52 in M2 | Plan §12, both retrospectives | **Five milestones** of 7 to 36 phases each. Extensions get their own two milestones |
| Line budgets (300/600/900) never fit. Estimates in parts, committed before the code, did fit: median 1.04 in M1 | M1 retrospective | **No budgets.** A five-part estimate, committed before the code, with a cap that triggers a question to the owner ([§7](#7-how-phases-work)) |
| The simulator on the CPU was the live view's bottleneck: about 160 ms a frame, 2 s while jogging. It moved to the GPU late | ADR-0062, performance notes | **The simulator is GPU-first from its first phase**, with the CPU path as reference and fallback |
| Full replays took about an hour per phase. Change-reach gates cut that | ADR-0038, ADR-0061 | **Tiered gates from day one:** quick, phase, checkpoint, milestone ([§8](#8-local-ci-gates)). Each phase runs only what its change reaches |
| GitHub Actions stopped for billing on 2026-11-14. The local replays carried every merge | ADR-0038 | **No GitHub Actions gate.** Every gate runs locally, in WSL |
| Two platforms, Windows and Linux, doubled every replay | ADR-0003, ADR-0038 | **One runtime platform:** Ubuntu 24.04 in WSL2. Windows only hosts WSL, the browser and, in M5, USB forwarding (Q8, confirmed) |
| Build output filled C: three times. Linking about 70 example crates at once ran out of memory | M1 retrospective | **Few example crates.** Examples live in each crate's `examples/` folder, with the `mold` linker, a capped number of build jobs, and `cargo xtask doctor` checking disk and memory |
| The `two-core.sh` stress runs found real races that 24 threads hid | M1 retrospective | **Stress repeats** of the reached suites on 2 CPUs, in the phase gate |
| Tests that waited a fixed time broke under load | Both retrospectives | Tests wait for a condition, never a fixed time. Each test gets its own data root and a port the system picks |
| Mutation checks, read against the tests first, found weak tests in nearly every phase | Both retrospectives | **A mutant list in each phase's notes** for code that detects, supervises or recovers. `cargo mutants` runs at checkpoints |
| Some plan numbers were assumptions: 580 MB/s assumed, 289 MB/s measured | M1 retrospective | **Numbers come from OScope's `hardware.md`,** and an early WSL2 hardware spike checks what WSL changes ([P1.S0](#stage-1a-groundwork)) |
| Design tokens, themes, nine accents, compact density, WCAG AA contrast tests | ADR-0041, `look.md` | **Written fresh** in M4 (P4.02) from `look.md` and ADR-0041: the same roles, values and accents |
| A Jobs tab | OScope's navRail | **Not in Tetera's navRail list.** Jobs show in the Status Indicator's panel and in each tab (Q6, confirmed) |
| The desktop app (Tauri) was dropped after W7 | ADR-0063 | **Browser only** from the start |
| OScope had no M3 | — | Tetera's M5 starts from OScope's M3 outline: H0–H7 |

## 3. In and out of scope

**In scope:** everything in [`goal.md`](goal.md), as read by the decisions in [§4](#4-proposed-decisions).

**Out of scope, or deferred** *(proposed)*

| Item | Where |
|---|---|
| Native Windows builds of the service | Not planned (Q8). The service runs in WSL2 only. Only the camera bridge, if needed, runs on Windows (R1) |
| Fluorescence, confocal and super-resolution samples | Not planned. The spec lists only H&E tissue |
| User accounts, roles, an audit trail, regulated-lab compliance | Later (Q5). v1 trusts the network it is on |
| Continuous, on-the-fly scanning with hardware triggers | Later. The MIChrome 20 has no trigger input. The frame model leaves room for it |
| Motorised or coded objective turrets | Later, if Q24 confirms. Objectives are swapped by hand |
| H.264 or WebP previews | M5, only if remote bandwidth needs them (Q18) |
| Frontend code loaded from extensions at run time | Not planned. Extensions describe panels as data, and the app renders them (Q35) |
| Languages other than English | Later. The app is i18n-ready from P4.01 |
| macOS, ARM64 | Not planned |

## 4. Proposed decisions

Each row is this draft's reading of the spec. Each becomes an ADR or a phase decision once the owner confirms it. The **Q** column links the question in §14 that can change it. Rows marked **Confirmed** are the owner's answers of 2026-11-26.

| # | Topic | Proposal | Q | Recorded in |
|---|---|---|---|---|
| D1 | Relation to OScope | **Confirmed: clean room.** All code is written fresh. OScope's ideas, designs, lessons and look may be reused, described in our own words. Nothing is copied from OScope, nor ported line by line | Q1, Q1b | ADR-0002 |
| D2 | Runtime platform | **Confirmed.** Ubuntu 24.04 in WSL2, x86_64, under systemd. No native Windows build | Q8 | ADR-0003 |
| D3 | Working copy | **Confirmed.** The repository moves into WSL's ext4 file system (`~/repos/Tetera`), and `D:\repos\Tetera` is retired. VS Code edits it over Remote-WSL. Building from `/mnt/d` is many times slower | Q10 | P1.01 |
| D4 | Raw frames | The Bayer mosaic (RGGB) in 16-bit words, with its significant bits (12) in the metadata | Q16 | ADR-0008 |
| D5 | Live preview | **Confirmed: binned live, full-resolution capture.** The live view defaults to a binned camera mode, for 30–60 fps. Captures and scans switch the camera to full resolution and back by themselves. The user can still pick any mode. JPEG over a binary WebSocket, sized to the client's viewport. Latest frame wins; never faster than the camera | Q17, Q18 | ADR-0019 |
| D6 | GPU | CUDA through `cudarc`, loaded at run time, with a CPU twin (rayon + SIMD) for every kernel. The twin is the reference in equivalence tests and the fallback without a GPU | — | ADR-0009 |
| D7 | Fused kernels | The stock live chain (dark → flat → demosaic → resize → white balance and CCM → gamma → 8-bit; resizing before colour since P1.06a) also exists as one hand-written fused CUDA kernel. The pipeline engine uses it when a recipe matches the chain, and tests it against the unfused nodes | Q19 | ADR-0011 |
| D8 | Simulator | GPU-first from P1.11. Deterministic from a seed. Tiles are made on demand and cached, in half precision on the GPU | Q26, Q27 | ADR-0015, ADR-0016 |
| D9 | Simulated clock | One clock for every simulated device. A move takes the time its distance and speed imply, and the live image moves with it. Tests can step the clock or run it faster | — | ADR-0014 |
| D10 | Rust extensions | **Confirmed.** In-process `cdylib`s behind a versioned C ABI, hidden by a safe SDK crate. Extensions are trusted code | Q29 | ADR-0026 |
| D11 | Python extensions | One worker process per extension, each with its own uv-managed virtual environment. Frames are shared as zero-copy NumPy arrays through shared memory | — | ADR-0028 |
| D12 | Extension nodes and workflows | **Confirmed.** M3's nodes and workflows are **first-party extensions** in this repository's `extensions/`, built only on the public SDK. They prove the SDK is enough | Q31 | ADR-0025 |
| D13 | Package format | A `.tetx` zip: a manifest (TOML), payloads, and the licences | Q30 | ADR-0025 |
| D14 | Datasets | **Confirmed.** OME-Zarr v3 (OME-NGFF 0.5) through `zarrs`. Every capture, multi-capture or scan is one dataset with its metadata. Datasets are grouped into projects, which the Gallery shows as cards. OME-TIFF and PNG/TIFF are export formats | Q36, Q37 | ADR-0021 |
| D15 | Jobs | Hardware jobs run one at a time from a queue. Compute jobs run in parallel on a pool that shares the GPU. Every job has progress, logs and cancel. The history is kept in SQLite | — | ADR-0020 |
| D16 | Configuration | TOML. The service's data is in `/var/lib/tetera/`; development uses `./.tetera/` | — | P1.13 |
| D17 | Default port | 8310, so Tetera and OScope can run side by side | Q48 | P1.13 |
| D18 | Units | µm as `f64`, and ms, at every API | — | ADR-0006 |
| D19 | Access | **Confirmed.** No accounts in v1. The service listens on localhost, and on the LAN only when configured to | Q5, Q11 | ADR-0017 |
| D20 | Power | **Confirmed.** Four actions: restart the service; restart WSL; reboot or shut down the Windows host, through WSL interop. Every action has a dry run for tests | Q13 | ADR-0024 |
| D21 | Gates | Local only, in WSL, tiered ([§8](#8-local-ci-gates)). No GitHub Actions workflow | Q41, Q42 | ADR-0005 |
| D22 | Toolchain | Rust pinned to 1.98.1, `teta-wot` v0.2.0's toolchain. Node 24 LTS. Python 3.12 with uv. A CUDA 13.x toolkit for WSL-Ubuntu | Q46, Q47 | P1.01 |
| D23 | `teta-wot` | Pinned to the `v0.2.0` tag. A gap is fixed upstream first, as a U-phase in `../teta-wot`, released as v0.3.0 | Q51 | ADR-0004 |

## 5. Ground rules

### 5.1 Clean room

- **All code is written fresh** (D1). Nothing is copied from OScope, `microscope` or any other codebase, nor ported line by line.
- **OScope's documents may be read and their ideas reused:** its plan, ADRs, notes, retrospectives, `look.md` and `hardware.md`. A phase's notes name the ideas it took, in our own words.
- **OScope's source code:** this draft assumes it is not opened while Tetera's code is written (Q1b).
- **Facts are not code.** The camera's measured modes and rates, and the look's colour values, may be carried over.
- **Algorithms** come from published methods, cited in the phase's notes.
- **Tucsen's sample images** are never committed. The Tucsen SDK is never redistributed unless its licence allows it.

### 5.2 Dependencies

- **Licences** must allow MIT distribution: MIT, Apache-2.0, BSD, ISC, Zlib, Unicode, or public domain. MPL-2.0 only as an unmodified development dependency. `cargo-deny` enforces this from P1.01, and an npm check from P4.01.
- **Only versions at least two weeks old.** For npm, `npm install --before=<date two weeks ago>`. For crates, check the release date.
- **A new class of dependency** goes to the owner first: a C library, a CUDA library, a model's weights, a Python package with native code.

### 5.3 Unsafe code

Forbidden through the workspace's lints, except in:

- `tetera-compute` (CUDA);
- `tetera-shm` (shared memory, M2);
- `tetera-ext` (loading native extensions, M2);
- `tetera-ext-sdk` (generated C ABI glue, M2);
- `tetera-tucsen` (the SDK's C API, M5).

Every `unsafe` block has a `SAFETY:` comment, which clippy's `undocumented_unsafe_blocks` enforces.

### 5.4 Tests

- **Wait for a condition, never a fixed time.**
- **Each test gets its own** data root, scratch space and port, chosen by the system, so suites run side by side.
- **Switches for what CI can't assume:**
  - `TETERA_CUDA=1`: CUDA tests;
  - `TETERA_PYTHON=1`: Python extension tests, which need uv;
  - `TETERA_GRBL_PORT=/dev/ttyACM0`: the real stage (M5);
  - `TETERA_TUCSEN=1`: the real camera (M5);
  - `TETERA_VISUAL=1`: the app's screenshot tests (M4).

  Without its switch, a test is skipped, and says so.
- **Every driver has a fake** at its protocol or SDK boundary.
- **Anything that supervises, recovers or detects** is broken on purpose once, to show its test fails. The notes say so.
- **Hardware tests run alone,** never side by side (OScope's M2 lesson 1).

### 5.5 Line endings and scripts

- LF everywhere, through `.gitattributes`.
- Gate tooling is Rust (`cargo xtask`), not Bash, so it is tested and can't write CRLF.

## 6. Architecture

### 6.1 Deployment

```text
┌──────────────────────────── Windows 11 host (the compute module) ─────────────────────────────┐
│  Browser (Edge/Chrome) ──▶ http://localhost:8310                                               │
│  NVIDIA Windows driver (gives WSL2 its CUDA)                                                   │
│  usbipd-win ──── forwards USB to WSL2: MIChrome 20 (USB 3.1), grblHAL (USB-serial)  (M5)       │
│  Task Scheduler: starts WSL at boot and keeps it alive  (Q12)                                  │
│                                                                                                │
│  ┌───────────────────────────── WSL2: Ubuntu 24.04 (systemd) ───────────────────────────────┐ │
│  │ tetera-server.service                                                                     │ │
│  │   HTTP/WS ─┬─ control plane: teta-wot Things (TDs, properties, actions, events)           │ │
│  │            └─ media plane: preview WebSocket, tiles, volume bricks, downloads             │ │
│  │   devices:  camera (virtual | Tucsen) · stage (virtual | grblHAL) · objective · light     │ │
│  │   compute:  CUDA (/usr/lib/wsl/lib) else CPU · Python extension workers (shared memory)   │ │
│  │   disks:    scratch (raw frame logs) · data (datasets in OME-Zarr) on ext4               │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
        ▲ LAN browsers, only with mirrored networking and the service configured to listen (Q11)
```

**What WSL2 changes, compared with OScope:**

- **USB devices** reach WSL2 only through `usbipd-win` (USB/IP). Whether a USB 3 camera keeps up at 289 MB/s that way is unknown. [P1.S0](#stage-1a-groundwork) measures it when the owner is at the instrument (deferred on 2026-11-26). If it falls short, the owner's chosen fallback is a small Windows-side camera bridge ([§13](#13-risks), R1).
- **Disks:** raw data must stay on ext4, inside the WSL virtual disk or on a disk attached with `wsl --mount`. `/mnt/c` and `/mnt/d` go through a network file system that is far too slow (Q14).
- **Memory:** WSL2 gets half of the host's RAM by default, about 7.8 GB on this PC. `.wslconfig` raises it ([§12](#12-installing-the-toolchains)).
- **Power:** `systemctl reboot` restarts only the WSL VM. Rebooting Windows goes through interop (`shutdown.exe`) (Q13).
- **Start-up:** WSL2 starts only when something on Windows starts it. An instrument that must come up without a login needs a Windows task (Q12).
- **Clock:** WSL2's wall clock can drift after the host sleeps. Tetera's own clock is monotonic, and UTC is mapped from it once at start-up, then re-checked.

### 6.2 Layers and crates

A crate depends only on crates in lower layers, plus the few edges within a layer that the table in `xtask` allows (OScope ADR-0005). A test reads `cargo metadata` and names any edge that breaks the rules.

| Layer | Crate | What it holds | From |
|---|---|---|---|
| L0 Foundation | `tetera-core` | Units, geometry (rects, 2D affines), the monotonic clock, IDs, frame and metadata types, the buffer pool's interface, errors. No I/O, no async | P1.02 |
| L1 Compute | `tetera-compute` | CPU and CUDA backends, device buffers, pinned memory, streams, the kernel build, every CUDA kernel and its launch, fused kernels *(unsafe)* | P1.04a |
| | `tetera-log` | The `tracing` set-up: the ring's layer, rolling JSON files, journald. Added by M1's specs: `tetera-core` does no I/O | P1.02 |
| | `tetera-shm` | Named shared memory and the frame-ring layout, for Python workers *(unsafe)* | P2.04 |
| L1 Algorithms | `tetera-imaging` | The stock algorithms, each with a CPU twin here and a CUDA kernel in `tetera-compute`: Bayer, demosaic, dark, flat, CCM, denoise, deconvolution, resize, encode, focus measures, registration | P1.05a |
| L2 Devices | `tetera-hal` | Device traits (`Camera`, `Stage`, `ObjectiveMount`, `Illumination`), capabilities and fakes | P1.09a |
| | `tetera-sim` | The simulated world: the H&E sample, optics, illumination, the virtual camera and the virtual stage | P1.10a |
| | `tetera-grbl` | The grblHAL driver and its protocol fake | P5.02 |
| | `tetera-tucsen` | The TUCam SDK binding and the camera driver *(unsafe)* | P5.03 |
| L2 Data | `tetera-store` | Datasets in OME-Zarr, the scratch frame log, the SQLite index | P1.15b |
| L3 Engines | `tetera-jobs` | The job service | P1.15a |
| | `tetera-pipeline` | The DAG engine, the node registry, the stock node wrappers, GPU residency and fusion | P1.08a |
| | `tetera-ext` | `.tetx` packages, the native loader and the Python worker supervisor *(unsafe)* | P2.01 |
| | `tetera-ext-sdk` | What Rust extension authors build against *(unsafe in generated glue)* | P2.02 |
| L4 Services | `tetera-acquire` | The frame bus, preview, live statistics and the recorder | P1.14a |
| | `tetera-workflows` | Capture, autofocus, scans, calibration, batch processing; the workflow trait that extensions implement | P1.15b |
| L5 Interface | `tetera-api` | teta-wot Things and the media-plane routes. **The only crate that depends on `teta-wot`** | P1.13 |
| L6 Applications | `tetera-server` | The binary: configuration, systemd integration, the web app's static files | P1.13 |

Beside the layers: `xtask/` (gates, the layering check, `doctor`, licences), `python/tetera-ext/` (the Python SDK, M2), `extensions/` (first-party extensions, M3) and `web/` (the app, M4).

### 6.3 Repository layout

```text
Tetera/
├─ Cargo.toml              workspace: resolver 3, edition 2024, shared lints and dependencies
├─ rust-toolchain.toml     1.98.1, as teta-wot v0.2.0
├─ .cargo/config.toml      mold as the linker; opt-level 2 for dependencies, sim, imaging and compute in dev
├─ .vscode/                launch.json (CodeLLDB), extensions.json, settings.json: shared, from P1.01
├─ crates/tetera-*/        the crates in §6.2; each crate's examples/ folder holds its examples
├─ examples/               a few workspace crates for whole-server demos; examples/vendor/ for vendored JS
├─ extensions/             M3: first-party extensions, each a Rust crate or a Python package, packed as .tetx
├─ python/tetera-ext/      M2: the Python SDK (package `tetera_ext`)
├─ web/                    M4: the Vue 3 app
├─ configs/                tetera.toml, objectives.toml, simulation.toml
├─ packaging/              M5: .deb, the systemd unit, the Windows bootstrap
├─ xtask/                  cargo xtask gate | doctor | ext | licences
└─ docs/                   §11
```

### 6.4 Control plane and media plane

**The control plane** is teta-wot Things. Their OpenAPI document becomes the app's TypeScript types in M4. An action that starts long work returns a job reference at once, and the job service runs it.

| Thing | Holds | From |
|---|---|---|
| `system` | Version, platform (WSL, kernel, distro), GPU, disks, memory, licences | P1.13 |
| `camera` | Mode, exposure, gain, pixel format, binning, ROI, white balance, streaming | P1.13 |
| `stage` | Position, moving, move, jog, stop, home, limits, speed | P1.13 |
| `objective` | The catalogue, which objective is fitted | P1.13 |
| `illumination` | On/off, intensity, colour temperature (simulated, or manual) | P1.13 |
| `simulation` | Samples (choose, load, unload), illumination model, imperfections on demand | P1.13 |
| `preview` | Preview defaults, sessions, and the binary stream | P1.14b |
| `jobs` | Queue, history, progress, cancel | P1.15a |
| `capture` | Single and multiple captures | P1.15b |
| `datasets` | The Gallery: datasets, thumbnails, metadata, delete, export | P1.15b |
| `autofocus` | Autofocus as a job | P1.16 |
| `scan` | Planners, plan preview, start, live progress | P1.17 |
| `calibration` | Per-objective calibration sets, routines, the wizard's steps | P1.18 |
| `pipeline` | The node catalogue, recipes, runs on datasets | P1.08a–e, P1.19 |
| `status` | Each subsystem's health, rolled up into one status | P1.20 |
| `logs` | Records, filters, live tail | P1.20 |
| `power` | Restart the service or WSL; reboot or shut down the host | P1.20 |
| `settings` | System, display defaults, data and scratch locations | P1.20 |
| `extensions` | Installed packages; install, enable, disable, remove | P2.01 |
| `workflows` | The workflow catalogue, stock and extension, and runs | P2.06 |
| `workspace` | Measurements, annotations, analysis results, reports | P4.10 |

**The media plane** is Tetera's own axum routes, mounted beside the Things through teta-wot's router composition, so they share its security and shutdown:

| Route | Carries |
|---|---|
| `GET /api/v1/preview/stream` (WebSocket) | Binary JPEG frames, and JSON from the client: viewport, ROI, quality, acknowledgements |
| `GET /api/v1/preview/mjpeg` | An MJPEG stream, for plain `<img>` pages and other tools |
| `GET /api/v1/media/dzi/{artifact}.dzi`, `…_files/{level}/{col}_{row}.jpg` | Deep-zoom tiles (M3) |
| `GET /api/v1/media/volume/{artifact}/meta`, `/slice`, `/brick` | Volume metadata, slices and bricks for VTK.js (M3) |
| `GET /api/v1/data/{dataset}/{path}` | Downloads, with Range support |
| `POST /api/v1/upload/{name}` | Uploads: extension packages, images to import |

### 6.5 Frames, the clock and the two streams

**The frame model:**

- `PixelFormat`: `Bayer16 { pattern, significant_bits }`, `Bayer8`, `Mono8/16`, `Rgb8/16`, `Rgba8`, planar `F32`.
- **A frame** is a pooled buffer plus its metadata:
  - its sequence number, and the host time of the exposure's start, end and midpoint;
  - the line time, for a rolling shutter;
  - exposure, gain, mode, binning, ROI and black level;
  - the objective, and the light's state;
  - the stage position at the exposure's midpoint;
  - frames dropped before it.
- **The pool:** pinned host memory when CUDA is present. One full Bayer16 frame is 39.9 MB, so 16 frames is about 640 MB. The size is configurable, since this PC's WSL has little memory.

**One clock.** Every timestamp is Tetera's monotonic clock in nanoseconds, mapped to UTC. Stage positions are kept in a timestamped track, and a frame's position is interpolated at its exposure midpoint.

**The frame bus** fans each frame out to two kinds of subscriber:

```text
camera ─▶ frame pool (Bayer16) ─▶ frame bus ─┬─▶ recorder (lossless) ─▶ scratch frame log ─▶ ingest job ─▶ OME-Zarr dataset
                                             ├─▶ preview (latest wins) ─▶ crop/bin ─▶ fused ISP (GPU) ─▶ JPEG ─▶ WS per client
                                             ├─▶ live statistics (≤ 10 Hz) ─▶ histogram, saturation, profile, focus value
                                             └─▶ workflows: "the first frame exposed after t" (autofocus, scans, calibration)
```

**The raw stream** never leaves Rust, except through an explicit download or export. During acquisition the recorder appends frames to a preallocated, uncompressed frame log on the scratch disk. An ingest job then writes them into the dataset's OME-Zarr store, compressed.

**The preview stream:**

- Each client asks for its viewport size, and an ROI when zoomed in.
- The server crops or bins the raw frame, runs the fused ISP kernel, resizes, encodes JPEG, and sends it.
- Each client has a one-frame mailbox, so a slow client always gets the newest frame.
- **The rate is the camera's, never more.** At full resolution that is 7.24 fps in 16 bits and 14.49 fps in 8 bits. The spec's 30–60 fps needs a binned mode: 53 fps at 2736 × 1824, or 67 fps at 1824 × 1216, in 8 bits.
- **So the live view runs binned, and captures run at full resolution** (D5). A capture or scan switches the camera's mode, takes its frames, and switches back. The switch's cost is measured on the virtual camera in P1.15b, and on the real one in P5.03. The user can still choose any mode by hand.

**Live statistics** come from the raw data, not from the 8-bit preview: per-channel histograms, the saturation mask, line profiles, ROI statistics, a focus value.

### 6.6 Compute

- **Two backends** in `tetera-compute`:
  - `Cpu`, with rayon, and SIMD where it pays;
  - `Cuda`, with `cudarc`. The driver, cuFFT and NVRTC are loaded at run time. Under WSL2 they come from `/usr/lib/wsl/lib`.
- **Kernels** are CUDA C, compiled by `build.rs` to a fat binary, machine code for each major GPU architecture and PTX for newer ones, when the `cuda` feature is on and `nvcc` is found; without `nvcc`, by NVRTC at run time ([ADR-0010](adr/0010-building-and-loading-cuda-kernels-under-wsl2.md)).
- **GPU first:** each node runs on CUDA when there is a GPU, and on the CPU otherwise, or when the build has no CUDA.
- **Every kernel has a CPU twin.** Equivalence tests compare them within a stated tolerance.
- **Data stays on the GPU** between consecutive CUDA nodes, and is copied back only at the boundaries.
- **Fused kernels** (D7) do the live chain in one pass. The unfused nodes stay the reference.
- **The GPU is shared** by the simulator, the preview and compute jobs. This PC's laptop GPU has 8 GB, so each user has a VRAM budget.

### 6.7 Processing pipeline and the stock nodes

- **A recipe** is a DAG of nodes with typed ports, written as JSON with a schema.
- **Two ways to run one:**
  - **per frame,** on the live stream, where a node must keep up with the camera;
  - **in batch,** on a dataset, as a compute job, tiled when the data is large.
- **The registry** holds the stock nodes in M1, then extension nodes from M2. Each node declares its ports, its parameters (as JSON Schema), its backends, and whether it can fuse.

**Stock nodes, in M1:**

| Node | Method *(proposed; spec-first phases cite the literature)* | Phase |
|---|---|---|
| Bayer split | RGGB into R, G1, G2 and B planes | P1.05a |
| Demosaic | Bilinear, and Malvar–He–Cutler | P1.05a |
| Raw → JPEG / MJPEG | SIMD JPEG encoding on the CPU; nvJPEG on the GPU if WSL2 has it (checked in P1.04a) | P1.05b |
| Dark-field subtraction | Subtract a master dark frame, matched by exposure and gain | P1.06a |
| Flat-field correction | Divide by a normalised master flat, dark-subtracted | P1.06a |
| Colour-correction matrix | White-balance gains, then a 3 × 3 matrix to linear sRGB, then gamma | P1.06a |
| Denoise | Temporal averaging of N frames; a spatial edge-preserving filter (bilateral, or non-local means) | P1.07a |
| Deconvolution with a PSF | Richardson–Lucy, FFT-based, with a theoretical widefield PSF from the objective's NA and wavelength, or a measured PSF | P1.07b, P1.07c |
| Glue | Crop, bin, resize, display transform (levels, gamma), statistics | P1.05b–P1.08a |

### 6.8 Hardware abstraction

| Trait | Real (M5) | Virtual (M1) |
|---|---|---|
| `Camera` | `tetera-tucsen`: MIChrome 20 over the TUCam Linux SDK | `VirtualCamera`: MIChrome 20's modes, rates, Bayer layout and noise, on the GPU |
| `Stage` | `tetera-grbl`: grblHAL over serial | `VirtualStage`: grblHAL-like motion, timing, limits, homing, faults on demand |
| `ObjectiveMount` | Manual: the user says which objective is fitted | The same, with the simulator's optics following it |
| `Illumination` | Manual, unless the owner names a controller (Q25) | `VirtualIllumination`: intensity, colour temperature, uneven field |

- **Capabilities are reported, not assumed:** modes, rates, triggers, binning, limits.
- **The camera trait doesn't assume the driver is in-process.** If USB forwarding into WSL2 can't keep up, a camera bridge on Windows implements it over shared memory or a socket. That is the owner's chosen fallback (R1), so P1.09 designs the trait for it from the start.

### 6.9 Simulation

GPU-first, with a bit-exact CPU reference on each backend, and the two backends agreeing within a stated tolerance (OScope ADR-0062).

- **The H&E sample** (P1.11):
  - procedural tissue: nuclei (haematoxylin) and stroma and cytoplasm (eosin), with glands, vessels and fat;
  - a section 3–5 µm thick, with optional tilt and folds, so z-scans and depth from focus have depth to find;
  - a coverslip-sized slide, made on demand as tiles, cached at several levels of detail;
  - deterministic from a seed.
- **Calibration targets** (P1.18, Q26): a blank field, for flats; a stage micrometer grid, for pixel size and rotation; a colour chart, for the CCM.
- **Optics** (P1.12a):
  - the objective's magnification and NA;
  - defocus blur from the distance to the sample's depth;
  - vignetting;
  - motion blur while the stage moves during an exposure;
  - the camera's mounting angle, which calibration then recovers.
- **Simulated illumination** (P1.12a): intensity, colour temperature, and a field that is uneven by a configurable amount, so flat-field correction has something to correct.
- **The virtual camera** (P1.12b):
  - the MIChrome 20's three modes and their rates;
  - RGGB Bayer;
  - shot, read and dark noise, black level, gain;
  - 12 significant bits in 16-bit words;
  - optionally, the SDK's own processing, to mimic the real raw (Q27).
- **The virtual stage** (P1.10b): grblHAL-like states (Idle, Jog, Run, Home, Alarm), acceleration, limits, report latency, and faults on demand.

### 6.10 Jobs and workflows

- **The job service** (P1.15a) queues hardware jobs one at a time, and runs compute jobs on a pool. Every job has progress, a log, cancel, and a history in SQLite.
- **Stock workflows (M1):**
  - **capture:** single and multiple;
  - **scanning:** an XY snake scan and a z-scan, stop and go;
  - **autofocus:** coarse to fine, on a focus measure;
  - **calibration:** dark, flat, pixel size and rotation, CCM, per objective.
- **The workflow trait** (P2.06) lets extensions add workflows. They reach the instrument only through an API that limits what they can do: move, capture, autofocus, run a pipeline, write a dataset.
- **Extension workflows (M3):** raster scan, position scan, tiled z-stack, histo scan.

### 6.11 Data on disk

```text
/var/lib/tetera/            (development: ./.tetera/)
├─ tetera.db                SQLite: datasets index, jobs history, calibration sets
├─ datasets/<id>/           one dataset: OME-Zarr arrays, metadata.json, thumbnails, results/
├─ calibration/<objective>/ master darks and flats, CCM, pixel size
├─ extensions/              installed packages and their Python environments
└─ logs/                    rolling JSON log files
/var/lib/tetera-scratch/    raw frame logs, caches; on ext4, ideally its own disk (Q14)
```

- **A dataset** is one capture, multi-capture or scan, with its positions, settings, objective and calibration recorded.
- **Results** derived from a dataset are kept beside it, with their provenance: the recipe, its version, and its inputs.
- **The Gallery** groups datasets into projects, or shows them flat with tags (Q37).

### 6.12 Extensions

- **A package** is a `.tetx` zip (Q30): `manifest.toml` (id, version, the SDK version it needs, kind, entry points, permissions), its payload (a Linux `.so`, or a Python package with a lock file), and its licences.
- **Rust extensions** (P2.02): in-process `cdylib`s behind a versioned C ABI. The SDK's macros generate the glue. A panic is caught at the boundary, and a version mismatch is refused at install.
- **GPU access for Rust extensions** (P2.03): the SDK gives an extension a handle to the host's CUDA context and stream. The extension can allocate device buffers, load its own PTX, and receive frames already on the GPU.
- **Python extensions** (P2.04): one worker process each, with its own uv environment. Messages go over a socket, and frames through shared memory as NumPy arrays. Workers are supervised and restarted, and calls time out.
- **Extension points:**

  | Point | What it adds | Phase |
  |---|---|---|
  | Processing node | A node in the registry, usable in any recipe, live or batch | P2.03, P2.05 |
  | Workflow | A workflow in the catalogue, run as a job | P2.06 |
  | Artifact | An output of a known kind (image, pyramid, volume, table, annotations, report section), which the host stores and serves | P2.07 |
  | UI panel | A panel described as data (JSON Schema forms, actions, results), which the app renders | P2.07 |

- **First-party extensions** (M3) live in `extensions/`. They are built and packed like a third party's, using only the public SDK.

### 6.13 The frontend (M4)

- **Stack:** Vue 3, TypeScript, Vite, pinia, vue-router (hash routing), reka-ui, `@material-symbols/svg-300`, vue-i18n, VTK.js, OpenSeadragon for deep zoom, `openapi-typescript` and `openapi-fetch` for the typed client.
- **The look:** OScope's look, with the tokens written fresh from its descriptions (ADR-0041, `look.md`):
  - light, dark and system themes;
  - nine accents, magenta (`#ce2283`) by default;
  - compact density;
  - the rail → control pane → view layout.
- **The navRail:** Control and Live View · Scan workflow · Gallery · Workspace, at the top. Settings · Logging · About · Power, then the Status Indicator, at the bottom.
- **The live view:** frames decoded in a worker, drawn with WebGL or a canvas. The overlays (crosshair, ROI, histogram, saturation, profile) come from the server's live statistics.
- **Tests:**
  - Vitest for units;
  - Playwright for end-to-end, each spec with its own server and a `// Covers:` line naming the features it covers;
  - axe for accessibility;
  - a small set of screenshot baselines (Q49).

### 6.14 Status, logging and power

- **Logging:** `tracing`, into three places:
  - an in-memory ring, which the `logs` Thing serves and tails;
  - rolling JSON files;
  - journald, under systemd.

  `tracing` has no *critical* level. A `critical = true` field marks a critical record, and the API reports it as `critical`.
- **Status:** each subsystem reports its health: camera, stage, GPU, disk space, scratch, jobs, extensions and WSL. The Status Indicator shows the roll-up: OK, Degraded or Fault.
- **Power** (Q13):
  - *restart the service:* `systemctl restart tetera`, allowed by a polkit rule for the service's account;
  - *restart WSL:* through the Windows task;
  - *reboot* and *shut down:* the Windows host, through interop.

  Each action is confirmed in the app, logged, and has a dry run that tests use.

### 6.15 Performance budgets *(proposed; measured in M1, tuned in M5)*

| What | Budget |
|---|---|
| Virtual camera, full frame, GPU | ≤ 40 ms a frame, still or jogging |
| Virtual camera, full frame, CPU fallback | ≤ 400 ms a frame; tests use a scaled-down sensor |
| Fused ISP, 20 MP to a 1920 × 1280 preview, GPU | ≤ 10 ms |
| JPEG, 1920 × 1280, quality 80 | ≤ 10 ms |
| Preview, exposure end to the browser's paint, localhost | ≤ 100 ms at the camera's rate |
| Live view rate | The camera's rate: 14.49 fps at full resolution in 8 bits, ≥ 30 fps in a binned mode |
| Recorder, sustained to scratch | ≥ 289 MB/s (the camera), aiming for 580 MB/s of headroom |
| Autofocus at 10×, simulated | ≤ 5 s |

## 7. How phases work

**IDs and branches.** Phase P1.07 is milestone 1's seventh phase:

- the branch is `m1/p1.07-denoise-deconvolution`, and the PR title `P1.07: Denoise and deconvolution`;
- an upstream `teta-wot` phase is `U01` on `u/u01-short-name`, in `../teta-wot`.

**Order.** One phase at a time, in the order of [§9](#9-milestones-and-phases). Each starts from an up-to-date `main`, after the previous one is merged.

**Size: estimates, not budgets.**

- Before the code, an estimate in five parts goes into the change log, in a commit of its own: **code, tests, examples, docs-as-code (READMEs, schemas), and glue**. Doc comments and blank lines count.
- **The target is at most 1,500 hand-written lines a phase** (Q4, confirmed). An estimate over 2,000 goes to the owner first, who decides whether to split it.
- The notes report the measured size against the estimate.

**Every pull request contains:**

- the code and its tests;
- **examples**, each with a README saying how to try it, and a `--smoke` mode that the gates run;
- **implementation notes**, `docs/mN/notes/PN.NN-short-name.md`, from the template;
- **new ADRs**, marked *Accepted*, since approving the PR accepts them, with their rows in the ADR index;
- **its manual test**: a checklist entry in `docs/mN/manual-tests.md`, saying how to try the phase by hand, with what to expect. The agent runs it and records the result in the notes. The owner runs each checkpoint's demo script (Q43);
- the phase marked *In review* in §9's table. The next phase's PR marks it *Done*;
- the gate's summary, pasted into the PR's description.

**Approval.** The owner reviews and approves every PR before it is merged *(proposed, Q40)*. Nothing is pushed to `main`, and no tag is created, without the owner's word.

**Spec-first phases** open their notes with a behaviour spec and its literature before any code: P1.07a, P1.07b, P1.11, P1.16, P1.18, P3.02, P3.06, P3.08 and P3.09.

**Checkpoints.** Each stage that ends in a checkpoint has a demo script in its last phase's notes, and runs the checkpoint gate (G2).

**Definition of Done** (the PR template, from P1.01):

- [ ] The scope matches the phase spec; any deviation is explained in the notes.
- [ ] The phase gate (G1) passed on the commit merged. G1c and G1p passed if the change reaches CUDA or Python.
- [ ] Tests cover the new behaviour. Anything that detects or supervises was broken once on purpose.
- [ ] Each example runs from a clean checkout as its README says.
- [ ] The notes, ADRs and manual test are written, and the indexes updated.
- [ ] Clean room: nothing copied or ported from OScope or any other codebase. The ideas taken are named in the notes.
- [ ] New dependencies' licences are checked, and each version is at least two weeks old.
- [ ] The measured size is reported against the estimate.

## 8. Local CI gates

Every gate runs on the local machine, in WSL, through `cargo xtask gate <level>`. No gate runs on GitHub. Each run writes `target/gates/<time>/summary.md`: what ran, why the change reached it, and how long it took.

| Gate | When | Target time (warm) | Runs |
|---|---|---|---|
| **G0 quick** | Before each commit, by hand or by an optional pre-commit hook | ≤ 90 s | `cargo fmt --check`; `clippy -D warnings` on the changed packages; the changed packages' unit tests; the layering check. From M4, ESLint and the type check on changed web files |
| **G1 phase** | Before opening the PR, and after review fixes | ≤ 10 min | G0, plus everything the change reaches: the tests of the changed packages and those that depend on them, with doc tests; the examples' `--smoke` runs among them; `rustdoc -D warnings`; `cargo deny`. **Stress:** the reached suites 3 times, side by side, 2 CPUs each. From M4: the related Vitest tests, the end-to-end specs whose `Covers:` the change reaches, the app's build, and the npm licence and advisory checks |
| **G1c CUDA** | When the change reaches a crate with CUDA code | ≤ 5 min | `TETERA_CUDA=1`: the CPU/CUDA twin tests and the CUDA examples of the reached packages |
| **G1p Python** | When the change reaches `python/` or the Python runtime (M2 on) | ≤ 5 min | `TETERA_PYTHON=1`: the Python SDK's tests and the Python extension tests |
| **G2 checkpoint** | At a stage's checkpoint phase, and when the toolchain, lock file, `.cargo/` or the gate's own code changes | ≤ 30 min | Everything, on a **fresh clone** of the commit: every suite, every example's smoke run, CUDA and Python. `cargo mutants` on the stage's detection and supervision code. From M4, every end-to-end spec |
| **G3 milestone** | At each milestone's last phase | ≤ 60 min, plus the soak | G2, plus: the performance budgets (§6.15) measured and recorded; a one-hour soak; the licence report; a release build. From M4: the workflow tests, axe, and the screenshot baselines |

**How "reached" is computed:** the files changed since the merge base with `main`, the workspace packages holding them, and every package that depends on those (from `cargo metadata`). A change to the workspace manifest, the lock file, the toolchain, `.cargo/` or `xtask/` reaches everything.

**Keeping the gates quick on this PC:**

- the `mold` linker;
- `opt-level = 2` for dependencies, the simulator, imaging and compute in dev builds;
- few example binaries;
- `CARGO_BUILD_JOBS` capped by `doctor` from the memory WSL has;
- `cargo xtask doctor` checks free disk (≥ 40 GB) and memory before G2 and G3;
- `cargo clean` between milestones, with the owner's word.

## 9. Milestones and phases

The tables give each phase's scope. The detailed specs (goal, scope, example, done when, tests, first estimate) are written per milestone: M1's right after this plan is approved, and each later milestone's as the previous one nears its end.

### Milestone 1: backend core

#### Stage 1A: Groundwork

| ID | Phase | Scope | Try it | ADRs |
|---|---|---|---|---|
| P1.S0 | WSL2 hardware spike (**deferred**) | Deferred by the owner on 2026-11-26, until the instrument and the TUCam Linux SDK are at hand; a USB 3 SSD stand-in and a Windows-to-WSL link probe can run earlier. Docs and probe programs: attach the MIChrome 20 and the grblHAL controller to WSL2 with usbipd-win; the TUCam Linux SDK under Ubuntu 24.04; sustained frame rate and MB/s; serial round trips; ext4 write rate. Writes `docs/hardware.md`, with OScope's measured facts carried over in our own words. If USB forwarding falls short, it sketches the Windows camera bridge for P1.09 to design for | The probes' READMEs | 0018 (input) |
| P1.01 | Workspace and gates | Cargo workspace, toolchain pin, lints, the layering check, `cargo xtask gate` G0/G1/G2 and `doctor`, `cargo-deny`, `.gitattributes`, the PR template, shared VS Code settings in `.vscode/` (CodeLLDB debug configurations for the server, an example and a crate's tests; the recommended extensions for Remote-WSL; rust-analyzer running clippy), docs skeleton and templates, `CLAUDE.md`, the toolchain guide (§12), `tetera.md` moved to `docs/goal.md` | `cargo xtask doctor`; `cargo xtask gate quick`; an example debugged from VS Code over Remote-WSL | 0001–0005 |
| P1.02 | Core types and logging | Units, geometry, the clock, IDs, errors; logging set-up (ring, files, journald, the critical marker) | `ex-core-tour` | 0006, 0007 |
| P1.03 | The frame model | `PixelFormat`, `FrameMeta`, the buffer pool (pinned when CUDA is present), shared frames | `ex-frame-pool` | 0008 |

#### Stage 1B: Compute and the stock nodes

| ID | Phase | Scope | Try it | ADRs |
|---|---|---|---|---|
| P1.04a | Compute backends | CPU and CUDA backends, run-time detection and the CPU fallback, device buffers and streams, pinned memory; what WSL2's CUDA offers: pinned memory, cuFFT, NVRTC, nvJPEG | `ex-compute-info` | 0009 |
| P1.04b | Kernels and the twin harness | The PTX build and module loading, the twin-test harness, a scale-and-offset kernel pair, VRAM budgets | `ex-twin-scale` | 0010 |
| P1.05a | Bayer and demosaic | Bayer split, bilinear and Malvar–He–Cutler demosaic; CPU and CUDA | `ex-demosaic`: a synthetic Bayer frame demosaicked, on each backend | — |
| P1.05b | Binning, resize, display transform and JPEG | Binning, resize, levels and gamma, raw → JPEG/MJPEG; CPU and CUDA | `ex-demosaic`, to JPEG | — |
| P1.06a | Dark, flat and colour | Dark subtraction, flat field, white balance and CCM, gamma; the unfused chain on each backend | `ex-isp`: each stage's time, CPU against GPU | — |
| P1.06b | The fused kernel | The fused ISP kernel and its equivalence with the unfused chain | `ex-isp`: fused against unfused | 0011 |
| P1.07a | Denoise *(spec-first)* | Temporal averaging, an edge-preserving spatial filter; CPU and CUDA | `ex-denoise`: PSNR before and after | — |
| P1.07b | PSFs and Richardson–Lucy *(spec-first)* | Theoretical and measured PSFs; Richardson–Lucy, 2D and 3D, on the CPU | `ex-deconvolve`, against a blurred ground truth | — |
| P1.07c | Richardson–Lucy on the GPU | Richardson–Lucy on cuFFT, matching the CPU | `ex-deconvolve` on the GPU | — |
| P1.08a | The pipeline model | DAG, typed ports, JSON recipes and their schema, validation, the registry, the CPU executor | `ex-recipe`: a recipe validated and run on a frame | 0012 |
| P1.08b | The raw path's nodes | An encoded image as a port type; the raw path's nodes, from binning to JPEG; the live chain as a recipe | `ex-recipe`: the live chain, node by node | — |
| P1.08c | GPU residency and fusion | Data kept on the GPU between CUDA nodes, copies counted; the fusion pass | `ex-recipe` on the GPU, fused | — |
| P1.08d | Denoise and deconvolution nodes, and live runs | Temporal averaging, the bilateral filter and Richardson–Lucy as nodes; live runs, the latest frame kept | `ex-recipe`, live | — |
| P1.08e | Batch runs and tiling | Runs over a list of inputs; tiles with halos | `ex-recipe` over a folder, tiled | — |

#### Stage 1C: Hardware abstraction and simulation (ends with Checkpoint A)

| ID | Phase | Scope | Try it | ADRs |
|---|---|---|---|---|
| P1.09a | The camera | `Camera`, its capabilities and configuration, the fake camera, the camera's contract suite | `ex-hal-fakes`: the fake camera | 0013 |
| P1.09b | The stage, the objectives and the light | `Stage`, `ObjectiveMount`, `Illumination`, the fake stage, the stage's contract suite; the objective catalogue | `ex-hal-fakes`: the fake stage too | — |
| P1.10a | The simulated clock and motion profiles | The simulated clock (real, stepped, faster); trapezoidal motion profiles within each axis's limits, and feed holds | `ex-sim-clock`: the clock, and a move's profile | 0014 |
| P1.10b | The virtual stage | `VirtualStage` with grblHAL-like motion, states, limits, homing, reports with latency, faults | `ex-virtual-stage`: a move's timeline | — |
| P1.11a | The H&E sample, on the CPU *(spec-first)* | Procedural, 3D, deterministic H&E: the section, its tissue and its stains; colour by absorbance; regions rendered at any sampling; the CPU twin | `ex-he-sample`: regions at several magnifications, as PNGs | 0015 |
| P1.11b | The H&E sample on the GPU, and its tiles *(spec-first)* | The model in a CUDA kernel; tiles on demand, levels of detail, half precision, the cache | `ex-he-sample` on each backend, timed | — |
| P1.12a | Optics and light: the field the sensor sees | The field of view and mounting angle; defocus by depth group, diffraction, vignetting; simulated illumination; the field from the tile cache on the GPU, and its CPU twin | `ex-virtual-optics`: the field through focus, under two lights | 0016 |
| P1.12b | The virtual camera | Motion and the rolling shutter; `VirtualCamera` with MIChrome 20's modes, Bayer and noise, timed on the clock. **Checkpoint A:** the virtual instrument renders H&E live, on the GPU | `ex-virtual-camera`: frames while the stage moves | — |

#### Stage 1D: The live instrument over HTTP (ends with Checkpoint B)

| ID | Phase | Scope | Try it | ADRs |
|---|---|---|---|---|
| P1.13 | The server and the device Things | `tetera-server` on teta-wot v0.2.0; configuration; data roots; the systemd unit in WSL; Things for system, camera, stage, objective, illumination and simulation | `curl` and teta-wot's `/docs` | 0017, 0018 |
| P1.14a | The frame bus and the preview pipeline | Frame bus (lossless and latest-wins subscribers); preview sessions: crop or bin, the fused ISP on the GPU, JPEG; live statistics; no network yet | `ex-preview-pipeline` | — |
| P1.14b | The preview protocol | The binary WebSocket with acknowledgements, and MJPEG; the `preview` Thing. **Checkpoint B:** live H&E in a browser while jogging | `ex-live-view`: one plain HTML page | 0019 |
| P1.15a | The job service | Jobs, the hardware queue and the compute pool, cancel, the SQLite history; the `jobs` Thing | `ex-jobs` | 0020 |
| P1.15b | Datasets and capture | The scratch recorder; projects and datasets in OME-Zarr with the SQLite index; single and multiple capture, switching the camera from its binned live mode to full resolution and back; the `capture` and `datasets` Things | `ex-capture`: a burst, then the dataset on disk | 0021 |

#### Stage 1E: Workflows (ends with Checkpoint C)

| ID | Phase | Scope | Try it | ADRs |
|---|---|---|---|---|
| P1.16 | Autofocus *(spec-first)* | Focus measures; coarse-to-fine search; a job | `ex-autofocus` at 5× and 50× | — |
| P1.17 | Snake scan and z-scan | Planners (XY snake, z-scan), the stop-and-go engine, positions in the metadata, scan datasets | `ex-scan`: a 5 × 5 snake and a 21-plane z-scan | 0022 |
| P1.18 | Calibration *(spec-first)* | Calibration targets in the simulator; master dark and flat; pixel size and rotation from moves; CCM from a colour chart; per-objective sets, applied by the ISP. **Checkpoint C:** calibrate, autofocus, scan | `ex-calibrate` | 0023 |

#### Stage 1F: Running an instrument (ends with Milestone 1)

| ID | Phase | Scope | Try it | ADRs |
|---|---|---|---|---|
| P1.19 | Batch processing | Recipes on datasets as compute jobs, tiled; results with their provenance; the `pipeline` Thing's runs | `ex-batch`: denoise and CCM over a scan | — |
| P1.20 | Status, logs, power and settings | `status`, `logs` (ring, tail, files), `power` (dry run; WSL semantics), `settings` with persistence | `ex-system`: health, a log tail, a dry-run reboot | 0024 |
| P1.21 | Milestone 1 close | Performance baseline against §6.15; G3; a one-hour soak; the manual-test run; the retrospective | The demo script | — |

### Milestone 2: extensions

| ID | Phase | Scope | Try it | ADRs |
|---|---|---|---|---|
| P2.01 | Packages and the registry | `.tetx` format, manifest, versions and compatibility, install, enable, disable, remove; the `extensions` Thing | Install a package with `curl` | 0025 |
| P2.02 | The Rust ABI and SDK | `tetera-ext-sdk`: a versioned C ABI, safe traits and macros, panics caught, ABI compatibility tests | `ext-invert`, a Rust node extension | 0026 |
| P2.03 | Rust nodes, and the GPU | Extension nodes in the registry and in recipes; GPU access through the SDK (context, stream, device buffers, the extension's PTX) | `ext-invert` on the GPU, in a live recipe | 0027 |
| P2.04 | The Python runtime | `tetera-shm`; a worker per extension; uv environments; the IPC protocol; supervision, restart, timeouts; the `tetera_ext` package | `ext-py-threshold`, a Python node | 0028 |
| P2.05 | Python nodes in recipes | Python nodes in live and batch recipes, mixed with Rust; transfer costs measured | A mixed recipe, live | — |
| P2.06 | Workflow extensions | The workflow trait and its limited instrument API, in Rust and Python; runs as jobs; cancel; the `workflows` Thing | `ext-spiral-scan` | 0029 |
| P2.07 | Artifacts and UI panels | Artifact kinds (image, pyramid, volume, table, annotations, report section), stored and served; panels described as data, for M4 | An extension that writes a table and declares a panel | 0030 |
| P2.08 | The developer kit; Milestone 2 close | `cargo xtask ext new / pack / install`; Rust and Python templates; the extension guide; G3; the retrospective | Make, pack and install an extension from the template | — |

### Milestone 3: extension nodes and workflows

All are first-party extensions in `extensions/`, built only on the public SDK.

| ID | Phase | Language | Scope | ADRs |
|---|---|---|---|---|
| P3.01 | Colour spaces | Rust (CPU + CUDA) | sRGB → XYZ → CIELAB, with ΔE; the smallest extension end to end | — |
| P3.02 | Stitching *(spec-first)* | Rust | Phase correlation between neighbours, a global solve, blending; checked against the simulator's ground truth | 0031 |
| P3.03 | 2D deep zoom | Rust | DZI pyramids from stitched images, as artifacts; the host's tile route | 0032 |
| P3.04 | Raster and position scans | Rust | Raster (one direction) and position-list scans; positions from a file or picked by the user | — |
| P3.05 | Tiled z-stack | Rust | A z-stack at each tile, with focus tracking | — |
| P3.06 | Depth from focus *(spec-first)* | Rust (CUDA) | Focus-measure stacks, a depth map, an all-in-focus image; checked against the sample's ground truth | — |
| P3.07 | 3D volume | Rust | A stack to a volume in OME-Zarr; bricks and slices for VTK.js | 0033 |
| P3.08 | Object recognition *(spec-first)* | Python | Nuclei and cell detection on H&E; annotations and a table as artifacts (Q33) | — |
| P3.09 | Histo scan *(spec-first)* | Rust | Overview, tissue detection, regions, focus map, tiled scan, stitch, deep zoom. **Checkpoint D** | — |
| P3.10 | Milestone 3 close | — | The extension paths' performance; G3; the retrospective | — |

### Milestone 4: frontend

Each phase is tried in the app, against the virtual instrument. Its notes have screenshots in both themes.

| ID | Phase | Scope | ADRs |
|---|---|---|---|
| P4.01 | The app's workspace | Vite, Vue 3, TypeScript, pinia, router, vue-i18n; ESLint, Prettier, Vitest, Playwright; the npm licence and advisory checks; served by `tetera-server`; G0–G3 learn the web checks | 0034 |
| P4.02 | Tokens, themes and components | Tokens, themes and nine accents from OScope's look, written fresh; the contrast tests; the Material Symbols registry; the base components on reka-ui; the dev gallery | 0035, 0036 |
| P4.03 | The typed client and live properties | OpenAPI → TypeScript; the client; observing properties and events over WebSocket; the connection's state | 0037 |
| P4.04 | The shell | The navRail with the spec's tabs; the Status Indicator and its panel, jobs included; the rail → pane → view layout; notifications | — |
| P4.05 | The live view | Preview client in a worker; WebGL or canvas; zoom and pan as an ROI; overlays (crosshair, ROI, histogram, saturation, profile) | 0038 |
| P4.06 | Camera and stage control | Camera panel; stage panel (position, move, jog from keys and buttons, home, stop); objective and illumination. **Checkpoint E:** control and live view | — |
| P4.07 | The Scan workflow tab | XY snake, z-scan and autofocus; forms from schemas for extension workflows; progress | — |
| P4.08 | The Gallery | Dataset cards; details and metadata; a viewer (deep zoom, stacks); import, export, delete. **Checkpoint F** | — |
| P4.09 | Workspace I: processing | Recipes on datasets, edited as a list or a graph (Q39); runs as jobs; results with provenance | — |
| P4.10 | Workspace II: analysis and reports | Measurements and annotations; object-recognition results; 3D in VTK.js; reports, with their server part | 0039 |
| P4.11 | Settings, Logging, About, Power | Settings (system, display, camera, stage, calibration and its wizard, simulated sample settings and load/unload, extensions); Logging; About; Power. **Checkpoint G** | — |
| P4.12 | Milestone 4 close | Workflow tests of each checkpoint; axe and keyboard paths; screenshot baselines; the app's budgets; the user's guide; G3; the retrospective | — |

### Milestone 5: real hardware and performance

Each driver phase is *Done* only after the owner has run its manual test on the instrument.

| ID | Phase | Scope | ADRs |
|---|---|---|---|
| P5.01 | WSL2 device plumbing | usbipd-win auto-attach, udev rules, the service account's device permissions, the Windows boot task; `docs/hardware.md` brought up to date. If the spike failed: the Windows camera bridge | 0040 |
| P5.02 | The grblHAL driver | `tetera-grbl`: the line protocol and real-time commands, `$J=` jog, `$H` homing, alarms, settings, position reports and their latency; a protocol fake and recorded sessions | — |
| P5.03 | The Tucsen driver | `tetera-tucsen`: the TUCam Linux SDK, loaded at run time; modes and real rates; raw Bayer with the SDK's processing set explicitly; host timestamps; broken frames; reconnecting; a recorded-session fake | 0041 |
| P5.04 | Objectives and calibration on real optics | Per-objective pixel size, darks and flats, CCM from a real chart, a measured PSF | — |
| P5.05 | Workflows on the instrument | Autofocus tuning; snake and z-scans of real slides; M3's extension workflows on real hardware | — |
| P5.06 | Performance | The raw path at the camera's rate to scratch; preview latency and rate; GPU tuning; memory; several clients; H.264 only if needed | — |
| P5.07 | Packaging and release 1.0 | A `.deb` and the systemd unit for WSL; a Windows bootstrap (`.wslconfig`, the task, usbipd); a 24-hour soak; the user's guide; the release; the retrospective | 0042 |

**In all:** 73 phases, plus the spike: M1 36, M2 8, M3 10, M4 12, M5 7.

**Each phase's status** is in [`m1/notes/README.md`](m1/notes/README.md): *Not started*, *In review* or *Done*.

**Milestone 1's detailed specs** are in [`m1/phases.md`](m1/phases.md): about 34,800 lines by first estimates. P1.14 and P1.15 were over 2,000 lines, and the owner split each in two on 2026-11-26.

## 10. Planned ADRs

Numbered across the project. Each is written in the phase named.

| ADR | Title | Phase |
|---|---|---|
| 0001 | Recording decisions, and the delivery process | P1.01 |
| 0002 | Clean room and dependency licences | P1.01 |
| 0003 | The runtime platform: Ubuntu 24.04 in WSL2 | P1.01 |
| 0004 | teta-wot v0.2.0, extended upstream first | P1.01 |
| 0005 | Layered crates, and the local gates | P1.01 |
| 0006 | Units, coordinates and the clock | P1.02 |
| 0007 | Logging | P1.02 |
| 0008 | The frame model, the buffer pool and the frame bus | P1.03 |
| 0009 | Compute backends: the CPU reference and CUDA, and what WSL2 offers | P1.04a |
| 0010 | Building and loading CUDA kernels under WSL2 | P1.04b |
| 0011 | Fused kernels | P1.06b |
| 0012 | The pipeline model | P1.08a |
| 0013 | The hardware abstraction layer | P1.09a |
| 0014 | The simulated clock | P1.10a |
| 0015 | Simulated samples on the GPU | P1.11a |
| 0016 | The virtual camera's model | P1.12a |
| 0017 | Control plane and media plane | P1.13 |
| 0018 | The service in WSL2: systemd, start-up, networking, devices | P1.13 |
| 0019 | The preview protocol | P1.14b |
| 0020 | The job service | P1.15a |
| 0021 | Datasets on disk | P1.15b |
| 0022 | Scan planning and the scan engine | P1.17 |
| 0023 | The calibration model | P1.18 |
| 0024 | Power actions under WSL2 | P1.20 |
| 0025 | Extension packages | P2.01 |
| 0026 | The Rust extension ABI | P2.02 |
| 0027 | GPU access for extensions | P2.03 |
| 0028 | The Python extension runtime | P2.04 |
| 0029 | Workflow extensions | P2.06 |
| 0030 | Artifacts and UI panels from extensions | P2.07 |
| 0031 | Stitching | P3.02 |
| 0032 | Deep-zoom serving | P3.03 |
| 0033 | Volumes on the media plane | P3.07 |
| 0034 | The web app: layers, state and routing | P4.01 |
| 0035 | Design tokens and themes | P4.02 |
| 0036 | Components on reka-ui | P4.02 |
| 0037 | The typed client and live properties | P4.03 |
| 0038 | The preview client | P4.05 |
| 0039 | Reports | P4.10 |
| 0040 | Devices in WSL2 | P5.01 |
| 0041 | The Tucsen driver | P5.03 |
| 0042 | Packaging and release | P5.07 |

## 11. Documentation layout

```text
docs/
├─ README.md                  the index
├─ goal.md                    the spec, moved from tetera.md in P1.01
├─ implementation-plan.md     this plan; updated as phases land
├─ setup/toolchain.md         §12, kept current from P1.01
├─ hardware.md                the real instrument; written in P1.S0 from OScope's facts and the spike, updated in M5
├─ adr/                       NNNN-title.md, README.md (the index), template.md
├─ m1/ … m5/
│  ├─ phases.md               the milestone's phase specs
│  ├─ notes/                  one file per phase, from notes/template.md; README.md indexes them
│  ├─ manual-tests.md         one checklist entry per phase, and one per checkpoint
│  └─ retrospective.md        written in the milestone's last phase
├─ performance/               baselines and measurements
└─ user-guide/                M4: one page per tab
```

## 12. Installing the toolchains

A first draft. **From P1.01, [`setup/toolchain.md`](setup/toolchain.md) is the guide to follow:** each step re-run on this PC, and kept current. This section stays as the plan's record.

**Checked on 2026-11-26:**

| | This PC (Windows) | WSL2 Ubuntu-24.04 |
|---|---|---|
| CPU, RAM | i7-12700H, 20 threads; 15.7 GB | 20 threads; 7.8 GB (the default half), no `.wslconfig` |
| GPU | RTX 3070 Ti Laptop, 8 GB; driver 621.24 | Visible to `nvidia-smi` |
| Disks | C: 307 GB free; D: 823 GB free | `/` 952 GB free (a virtual disk) |
| Rust | 1.100.0 | **Not installed** |
| CUDA toolkit | 13.5 | **Not installed** |
| Node | 26.13.1 | 18.19.1 (too old) |
| Python, uv | uv 0.13.10; no Python | Python 3.12.3; **no uv** |
| systemd | — | On (`/etc/wsl.conf`) |
| usbipd-win | **Not installed** (needed from P1.S0 or M5) | — |

### 12.1 On Windows

1. **WSL2 with Ubuntu 24.04:** already installed. To install it on another PC:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

2. **The NVIDIA Windows driver.** It gives WSL2 its CUDA. Never install a Linux NVIDIA driver inside WSL.

3. **`%UserProfile%\.wslconfig`**, to give WSL enough memory and keep it running. Each key is as [Microsoft's WSL documentation](https://learn.microsoft.com/windows/wsl/wsl-config) gives it, checked on 2026-11-26 with WSL 3.0.5. Comments go on lines of their own: WSL silently ignores a file it can't parse.

   ```ini
   [wsl2]
   # Of the host's 15.7 GB; builds and a 20 MP pipeline need it.
   memory=12GB
   swap=16GB
   processors=16
   # Lets LAN browsers reach the service, when it listens there (Q11).
   networkingMode=mirrored

   [general]
   # Never shut the distro down for being idle.
   instanceIdleTimeout=-1

   [experimental]
   autoMemoryReclaim=dropCache
   sparseVhd=true
   ```

   Then restart WSL:

   ```powershell
   wsl --shutdown
   ```

4. **Optional: move the distro to D:**, which has more free space than C: for build output:

   ```powershell
   wsl --manage Ubuntu-24.04 --move D:\WSL\Ubuntu-24.04
   ```

5. **VS Code**, with the *WSL* extension (`ms-vscode-remote.remote-wsl`). Open the repository with `code --remote wsl+Ubuntu-24.04 /home/lucas/repos/Tetera`, never as a `\\wsl.localhost` folder. Extensions are installed on the WSL side: rust-analyzer, CodeLLDB (the debugger) and Even Better TOML from P1.01; Python from M2; Vue (Official), ESLint and Prettier from M4; Nsight from P1.04. The repository's `.vscode/extensions.json` recommends them.

6. **usbipd-win** (from P1.S0, or M5):

   ```powershell
   winget install --exact dorssel.usbipd-win
   ```

   Then list the devices, bind each one once as an administrator, and attach it to WSL:

   ```powershell
   usbipd list
   usbipd bind --busid <BUSID>
   usbipd attach --wsl --busid <BUSID> --auto-attach
   ```

7. **A browser:** Edge or Chrome, to use the app.

### 12.2 In WSL2 (Ubuntu 24.04)

1. **System packages:**

   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo apt install -y build-essential pkg-config cmake clang lld mold git curl unzip \
     ca-certificates libudev-dev jq sqlite3 python3-venv gh
   git config --global core.autocrlf input
   ```

   `libudev-dev` is for the serial port crate (M5). `clang` is for bindgen, if the Tucsen binding uses it (M5).

2. **Rust**, through rustup. The repository's `rust-toolchain.toml` picks the version on the first `cargo` command:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
   . "$HOME/.cargo/env"
   cargo install --locked cargo-deny cargo-mutants
   ```

3. **The CUDA toolkit for WSL-Ubuntu.** Only the `cuda-toolkit-*` package: the `cuda` and `cuda-drivers` packages would install a Linux driver. P1.04 pins the version:

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt install -y cuda-toolkit-13-0
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   . ~/.bashrc
   nvidia-smi && nvcc --version
   ```

4. **Node.js 24 LTS**, through nvm (from M4, or earlier for the example pages' tooling):

   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
   . ~/.nvm/nvm.sh
   nvm install 24
   ```

   At P4.01, Playwright's Chromium: `npx playwright install --with-deps chromium`.

5. **uv**, for Python extensions (from M2):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

6. **The repositories**, on WSL's ext4 file system (D3), with OScope and teta-wot beside Tetera for reference:

   ```bash
   mkdir -p ~/repos && cd ~/repos
   git clone https://github.com/lucaswewa/Tetera.git
   git clone https://github.com/lucaswewa/teta-wot.git
   git clone https://github.com/lucaswewa/OScope.git
   cd Tetera && code .
   ```

7. **Check it all** (from P1.01):

   ```bash
   cargo xtask doctor
   ```

   It reports the toolchains, CUDA, Node, uv, free disk and WSL's memory, and says what is missing.

## 13. Risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | **The USB 3 camera through usbipd into WSL2** may not sustain 289 MB/s, or may drop frames | The real camera can't feed the service at full rate | P1.S0 measures it when the owner is at the instrument; until then, M1 assumes forwarding works. A USB 3 SSD stand-in gives an early reading. **The owner's chosen fallback:** a small Windows camera bridge using the Windows TUCam SDK, sending frames into WSL over shared memory or a socket. The `Camera` trait allows a remote driver from P1.09 (§6.8). The service itself stays in WSL |
| R2 | **The TUCam Linux SDK** may not support Ubuntu 24.04, or USB/IP devices | No camera in WSL | P1.S0 checks it; the same fallback as R1 |
| R3 | **The development PC has 16 GB of RAM** (WSL gets 7.8 GB by default) and an 8 GB GPU, against OScope's 32–64 GB target. The instrument PC is a different machine (Q9), not yet specified | Builds run out of memory; the simulator, preview and jobs fight over VRAM during development | `.wslconfig` at 12 GB plus swap; `mold`; capped build jobs; few example binaries; VRAM budgets per user; a configurable frame pool. The budgets in §6.15 are set on this PC in M1 and re-measured on the instrument PC in M5 |
| R4 | **The camera's raw isn't the ADC's 12 bits.** OScope found that the SDK applies gamma and white balance, and rescales to about 58,000 levels with a black offset near 2,000 | Dark, flat and CCM nodes built for linear 12-bit data misbehave on real frames | Bayer16 with `significant_bits` and the black level in the metadata; the SDK set to its linear settings (gamma 128, contrast 32); the virtual camera can mimic the SDK (Q16, Q27) |
| R5 | **CUDA under WSL2 has limits** that native Linux doesn't. NVIDIA's WSL guide lists them | A feature Tetera plans to use may not work, such as pinned memory at size, or IPC to Python workers | P1.04 checks each feature Tetera uses. Python workers get frames through host shared memory, not CUDA IPC |
| R6 | **Disk:** OScope's phases built 60–70 GB each. WSL's virtual disk grows and doesn't shrink by itself | Full disks break WSL | `doctor` checks before G2 and G3; `sparseVhd`; the distro on D: (§12.1); cleaning between milestones |
| R7 | **WSL doesn't start without a login,** and stops when idle | The instrument isn't there after a reboot | `instanceIdleTimeout=-1` in `.wslconfig`; a Windows task at boot (P1.13 for development, P5.07 hardened) (Q12) |
| R8 | **Scope:** 73 phases | A long project | Checkpoints give usable increments. Estimates before the code. Cut lines: P3.08 (object recognition), P3.05 (tiled z-stack) and P4.10's reports can move to a later release |
| R9 | **The extension ABI** must stay stable across SDK versions | Old extensions break, or crash the server | A versioned C ABI checked at install; panics caught; tests against the previous SDK version; Python in separate processes |
| R10 | **Extensions as the only home** of stitching, deep zoom, depth from focus and volumes | The SDK must be rich enough for heavy, GPU-bound nodes | P2.03 gives Rust extensions the GPU; P3.01 is a small extension that proves the path before P3.02 |
| R11 | **The host's sleep, and WSL's clock drift** | Wrong wall-clock times in metadata | A monotonic clock; UTC mapped at start-up and re-checked; a status warning on drift |
| R12 | **The browser at 30–60 fps** at high resolution | Jank in the live view | Viewport-sized frames, decoding in a worker, WebGL |
| R13 | **Upstream teta-wot gaps** | A phase waits on an upstream release | Small U-phases; or a local route, recorded in an ADR, until the release (Q51) |

## 14. Questions for the owner

Each question has this draft's assumption in **bold**. Answer any of them, and the plan changes to match. The ones that change the plan most come first in each group.

**Answered on 2026-11-26:**

| Q | Answer | Effect on the plan |
|---|---|---|
| Q1 | **Clean room, ideas only** (not this draft's assumption, which was to port) | D1, §2, §5.1, the DoD, the notes template; Q1b added |
| Q4 | ≤ 1,500 lines a phase | §7 |
| Q5 | No accounts; trust the network | D19 |
| Q6 | Jobs in the Status Indicator's panel | §2, P4.04 |
| Q8 | Linux (WSL2) only | D2, §8 |
| Q9 | This laptop is a development machine only | R3; the instrument PC's spec is still open (Q9b) |
| Q10 | Move the working copy into WSL's ext4 | D3 |
| Q13 | Restart the service; restart WSL; reboot and shut down Windows | D20 |
| Q15 | The WSL2 hardware spike early in M1; a Windows camera bridge as the fallback | P1.S0, §6.1, §6.8, R1 |
| Q17 | Binned live view; captures and scans switch to full resolution | D5, §6.5, P1.15b (not quite this draft's assumption) |
| Q29 | In-process `cdylib`s behind a C ABI | D10 |
| Q31 | M3's extensions in this repository's `extensions/` | D12 |
| Q33 | Classical first; an ONNX model optional | P3.08 |
| Q36, Q37 | OME-Zarr, grouped into projects | D14 |
| Q43 | The agent runs each phase's manual test; the owner runs each checkpoint's | §7 |

### A. Scope and the relation to OScope

- **Q1. Reusing OScope's code.** ~~Port and adapt, clean room, or fork?~~ **Answered: clean room, ideas only (D1).**
- **Q1b. Reading OScope's source.** In a clean room, may a phase read OScope's source code for reference, without copying it? Or only its documents (plan, ADRs, notes, retrospectives, `look.md`, `hardware.md`), as OScope took `microscope`'s look from the running app and never its source? **Documents only; the source isn't opened while Tetera's code is written.**
- **Q2. OScope from now on.** Is OScope frozen, with Tetera superseding it? Should Tetera's milestone-close phases compare against OScope's numbers? **Frozen; compared where it helps.**
- **Q3. `microscope`** (the sibling folder `../microscope`). Is it relevant to Tetera at all? **No: not read, not used.**
- **Q4. Phase size.** OScope's median phase was about 2,250 lines in M1 and 1,380 in M2. Your spec asks for smaller phases. **Target ≤ 1,500 hand-written lines; ask the owner over 2,000.** *Answered: see the table above.*
- **Q5. Accounts, login and a control lease** (OScope had all three; the spec doesn't mention them). **None in v1; the service trusts its network.** *Answered: see the table above.*
- **Q6. Jobs.** The spec's navRail has no Jobs tab, unlike OScope's. **Jobs show in the Status Indicator's panel and in each tab that starts them.** *Answered: see the table above.*
- **Q7. The navRail's order.** **Exactly as the spec lists it,** with Settings, Logging, About and Power at the bottom, above the Status Indicator.

### B. Platform and deployment

- **Q8. Platforms.** Build and test on Linux (WSL2) only, or also natively on Windows, as OScope did? **Linux only (D2).** It halves every gate. *Answered: see the table above.*
- **Q9. The compute module.** **Answered: this laptop is a development machine only.**
- **Q9b. The instrument PC.** What will it have: CPU, RAM, GPU and its VRAM, and a separate NVMe for scratch? Is it at hand during M1–M4? **At least this laptop's CPU and GPU, 32 GB of RAM or more, and a separate NVMe; at hand from M5.**
- **Q10. Where the working copy lives.** Move it into WSL's ext4 (`~/repos/Tetera`) and edit through VS Code's Remote-WSL, or keep it on `D:`? **Move it (D3).** *Answered: see the table above.*
- **Q11. Who opens the app.** Only a browser on the same PC, or also browsers on the lab's network (WSL's mirrored networking, firewall rules)? **Same PC in M1–M4; the LAN from M5, configured.**
- **Q12. Start-up.** Must the instrument come up at Windows boot, without anyone logging in? **Yes, from M5; development starts it by hand.**
- **Q13. Power actions.** What should *reboot* and *shut down* act on: the Windows PC, the WSL VM, or both as separate actions? **Restart the service; restart WSL; reboot and shut down the Windows PC.** *Answered: see the table above.*
- **Q14. The scratch and data disks.** Inside WSL's virtual disk, on a separate physical disk attached with `wsl --mount`, or somewhere else? Does the instrument PC have a separate NVMe for scratch? **The virtual disk in M1–M4; decided for the instrument PC in M5.**
- **Q15. The WSL2 hardware spike (P1.S0).** Is the instrument at hand to run it early in M1, or does it wait until M5? If USB forwarding falls short, do you prefer a Windows camera bridge, or the service native on Windows? **Early, whenever you are at the instrument; the bridge if needed.** *Answered: see the table above.*

### C. Imaging

- **Q16. Raw bit depth.** The spec says 12-bit raw; the camera delivers 16-bit words that its SDK has already processed (gamma, white balance, rescaling). Store `Bayer16` with `significant_bits = 12`, and set the SDK to its linear settings? **Yes (D4).**
- **Q17. Live view rate.** The camera gives 7.24 fps (16-bit) or 14.49 fps (8-bit) at full resolution, and 27–67 fps binned. Should the live view switch the camera to a binned mode automatically, and capture at full resolution? Or follow whatever mode the user sets? **Follow the user's mode, with a binned mode as the live view's default; captures switch to full resolution.** *Answered: see the table above.*
- **Q18. Preview codec.** **JPEG first; WebP never (too slow to encode at these rates); H.264 in M5 only if remote bandwidth needs it.** Is any other format needed, such as for recording?
- **Q19. Fused CUDA kernels.** One hand-written fused kernel for the standard live chain, or fusion generated at run time (NVRTC) for any chain of fusable nodes? **Hand-written, for the standard chain (D7).**
- **Q20. Denoise.** Which methods should be stock? **Temporal averaging, plus one edge-preserving spatial filter (bilateral, or non-local means).** BM3D or a learned denoiser could be extensions.
- **Q21. Deconvolution.** 2D on single frames, or 3D on z-stacks too? A theoretical PSF, or one measured from beads? **2D and 3D Richardson–Lucy; a theoretical PSF in M1, a measured one in M5.**
- **Q22. Demosaicing.** **Bilinear for speed, Malvar–He–Cutler for quality.** Anything else?
- **Q23. MJPEG.** Is it an HTTP stream for other tools, a recording format (MJPEG in AVI or MKV), or both? **An HTTP stream in M1; recording later, if wanted.**

### D. Hardware and simulation

- **Q24. Objectives.** The same four Mitutoyo Plan Apo long-working-distance objectives as OScope (5×, 10×, 20×, 50×), swapped by hand? Any plan for a turret? **Yes, swapped by hand.**
- **Q25. Illumination.** Is there a real light controller, or is the light manual on the real instrument? What should "simulation illumination" cover? **Manual on the real instrument; the simulator models intensity, colour temperature and an uneven field.**
- **Q26. Simulated samples.** H&E only, plus calibration targets (a blank field, a stage micrometer, a colour chart)? Should H&E be 3D (section thickness, tilt, folds) for z-scans and depth from focus? **Yes to both.**
- **Q27. Virtual camera fidelity.** An ideal linear sensor, or one that also mimics the SDK's processing (gamma, white balance, rescaling)? Rolling shutter? **Linear by default, with the SDK's processing as an option; rolling shutter modelled.**
- **Q28. The stage.** Do OScope's findings still hold: homing switches, linear encoders (how they are read is unknown), travel ranges? **Yes, as in OScope's `hardware.md`.**

### E. Extensions

- **Q29. How Rust extensions load.** In-process `cdylib`s behind a C ABI (fast, GPU-friendly, a crash takes the server down), out-of-process workers like Python's (isolated, slower for large frames), or WebAssembly (sandboxed, no CUDA)? **In-process `cdylib`s (D10).** *Answered: see the table above.*
- **Q30. The package's file extension.** **`.tetx`.**
- **Q31. Where M3's extensions live.** In this repository's `extensions/`, built only on the public SDK? Or in separate repositories, as a third party's would be? **In `extensions/` (D12).** *Answered: see the table above.*
- **Q32. The GPU from extensions.** Must Rust extensions have the GPU in M2? Must Python extensions (with CuPy or PyTorch)? **Rust: yes, in P2.03. Python: CPU only through M3, since CUDA IPC under WSL2 is uncertain.**
- **Q33. Object recognition.** Classical image processing (scikit-image, for example), a model run with ONNX Runtime, or PyTorch? Where would a model's weights come from, under which licence? **Classical first; an ONNX model optional, if its weights' licence allows.** *Answered: see the table above.*
- **Q34. Trust.** Are extensions trusted code, or should packages be signed and checked? **Trusted, and unsigned, in v1.**
- **Q35. UI from extensions.** Panels described as data only, or real frontend code loaded at run time? **Data only.**

### F. Data and the Workspace

- **Q36. Dataset format.** OME-Zarr v3, OME-TIFF, or both (one stored, the other exported)? **OME-Zarr stored; OME-TIFF and PNG/TIFF exported.** *Answered: see the table above.*
- **Q37. The Gallery's organisation.** Projects holding datasets, or a flat list of datasets with tags and search? **Projects holding datasets.** *Answered: see the table above.*
- **Q38. Reports.** PDF (Typst, in process), HTML, or both? What goes in one? **Both, from templates: images, measurements, tables, metadata.**
- **Q39. The Workspace.** Which measurements (distance, area, intensity, counts)? Is a node-graph editor for recipes wanted, as OScope had, or is a list editor enough? **Distance, area, intensity and counts; a list editor first, a graph editor later.**

### G. Process and gates

- **Q40. Pull requests.** One branch and one PR per phase, each approved by you before it is merged? Who merges? **Yes; you merge.**
- **Q41. GitHub Actions.** No workflow files at all, or one that runs only when started by hand? **None.**
- **Q42. The gates.** Are the target times and contents in §8 right? Should the phase gate repeat the reached suites three times on two CPUs? **Yes, as written.**
- **Q43. Manual tests.** Run by you from each phase's checklist, or run by the agent and recorded, with you running each checkpoint's? **The agent runs and records each phase's; you run each checkpoint's.** *Answered: see the table above.*
- **Q44. Examples in M1.** `cargo run` examples, some serving one plain HTML page (no Vue before M4), as OScope did? **Yes.**
- **Q45. Dependencies.** OScope's rules: permissive licences only, each version at least two weeks old, `cargo-deny`, and new classes of dependency to you? **Yes.**
- **Q46. The Rust toolchain.** Pin 1.98.1 (teta-wot's), or something newer, such as 1.100 (installed on Windows)? **1.98.1.**
- **Q47. Node.** 24 LTS (as OScope), or 26 (installed on Windows)? **24 LTS.**
- **Q48. Default port.** **8310,** so Tetera and OScope (8210) can run side by side.
- **Q49. Browsers and screenshots.** Test in Chromium only, or Firefox and WebKit too? Screenshot baselines, which cost OScope much upkeep, or none? **Chromium for every test; Firefox for the main paths at milestone close; a small set of baselines, one per tab in each theme.**
- **Q50. Languages.** **English, i18n-ready.**
- **Q51. teta-wot.** If Tetera needs something teta-wot v0.2.0 lacks, upstream first as U-phases, released as v0.3.0, or a local workaround first? **Upstream first (D23).**
- **Q52. The detailed phase specs.** Write all of M1's before P1.01 starts, and each later milestone's as the previous one nears its end? **Yes.**

---

## Appendices

### Appendix A: ADR template

```markdown
# ADR-NNNN: <Title>

- Status: Proposed | Accepted | Superseded by ADR-NNNN
- Date: YYYY-MM-DD
- Phase: PN.NN

## Context
## Decision
## Consequences
## Alternatives considered
```

### Appendix B: implementation notes template

```markdown
# PN.NN: <Title>

- Status: In review | Done
- Branch / PR:
- Estimate (code / tests / examples / docs-as-code / glue) → measured:

## Summary
## Behaviour spec (spec-first phases only, with literature sources)
## What was built
## Ideas taken from OScope (in our own words; no code)
## How to try it (the examples)
## Manual test (the entry in manual-tests.md)
## Performance (against §6.15, where relevant)
## Design notes and deviations from the plan
## Tests, and the mutants checked
## Gate summary (G1, plus G1c / G1p where reached)
## Follow-ups and known gaps
```

### Appendix C: manual test entry template

```markdown
### PN.NN: <Title>

- Prerequisites: (build, configuration, sample loaded, …)
- [ ] Step 1 — expect: …
- [ ] Step 2 — expect: …
- Result: pass / fail, by whom, on what (commit, date)
```

### Appendix D: change log

| Date | Change |
|---|---|
| 2026-11-26 | First draft, for the owner's review |
| 2026-11-26 | The owner's answers to 16 questions (§14). Clean room instead of porting (D1); binned live view with full-resolution captures (D5); Q1b and Q9b added |
| 2026-11-26 | §12.1 and R7: `instanceIdleTimeout=-1` under `[general]` replaces `vmIdleTimeout=-1`, which the WSL documentation doesn't give as a way to stop idle shutdown; no comments after values |
| 2026-11-26 | P1.01 also ships shared VS Code settings in `.vscode/`: debug configurations, recommended extensions, rust-analyzer's check. §6.3 and §12.1 to match |
| 2026-11-26 | P1.S0 deferred by the owner until the instrument and the TUCam Linux SDK are at hand (§6.1, §9, R1) |
| 2026-11-26 | Milestone 1's phase specs drafted in `m1/phases.md`. `tetera-log` added to §6.2 for P1.02's log writers |
| 2026-11-26 | The owner split P1.14 (frame bus and preview pipeline; preview protocol) and P1.15 (job service; datasets and capture): 60 phases, M1 23. ADR-0008 also records the frame bus's design |
| 2026-11-26 | **P1.01's second estimate,** before its code: code 1,200 (xtask: layering 230, reach 170, the gate's steps and runner 330, stress 120, fresh clone 70, summary 90, doctor 190; `tetera-core` 30) · tests 420 · examples 20 · docs-as-code 230 (manifests, `deny.toml`, `.cargo/`, `.vscode/`, the PR template, READMEs) · glue 40 = **1,910**, against a first estimate of 1,750. Under 2,000, so the phase stays whole |
| 2026-11-26 | P1.01 in review. `examples/*` left out of the workspace's members until its first crate (Cargo refuses a pattern with no folder); dependency optimisation set in `Cargo.toml`'s profile rather than `.cargo/config.toml` |
| 2026-11-26 | **P1.01 measured 3,867 lines against its estimate of 1,910 (×2.02):** code 2,099, tests 1,476. The tests grew with the end-to-end checks that the checkpoint's mutants called for; the code with rustfmt's line breaks and doc comments. Later estimates count both ([notes](m1/notes/P1.01-workspace-and-gates.md#size)) |
| 2026-11-26 | **P1.02's second estimate,** before its code, counting doc comments, rustfmt's line breaks, and tests at least the size of what decides or detects: code 1,575 (`tetera-core`: units 110, geometry 230, time and the clock 280, IDs 90, errors 120, log records and the ring 170; `tetera-log`: configuration 80, the layer 130, the rotating file 125, journald 130, set-up 110) · tests 1,250 · examples 170 · docs-as-code 145 · glue 40 = **3,180**, against a first estimate of 1,570. Over 2,000: **the owner kept it whole.** **The owner also chose in-house log writers:** a size-based rotating file, and journald over its native protocol, since `tracing-appender` rotates only by time and `tracing-journald` maps priority by level only, so it can't send a critical record as CRIT (2) |
| 2026-11-26 | **P1.02 measured 3,674 lines against its second estimate of 3,180 (×1.16):** code 2,050, tests 1,349. Ten transitive crates pinned back to releases at least two weeks old ([notes](m1/notes/P1.02-core-types-and-logging.md#size)) |
| 2026-11-26 | **P1.03's second estimate,** before its code: code 860 (formats and layout 220, binning 40, metadata 120, durations as ms in serde 30, the pool and frames 260, views and bit scaling 130, the allocator 60) · tests 800 · examples 110 · docs-as-code 25 · glue 20 = **1,815**, against a first estimate of 1,220. Under 2,000, so the phase stays whole |
| 2026-11-27 | **P1.03 measured 1,741 lines against its second estimate of 1,815 (×0.96):** code 988, tests 630 ([notes](m1/notes/P1.03-the-frame-model.md#size)) |
| 2026-11-27 | **P1.04's second estimate,** before its code: about 2,670 (code 1,355 · tests 1,000 · examples 190 · docs-as-code 85 · glue 40). Over 2,000: **the owner split it** into P1.04a (backends, detection, buffers, pinned memory, the WSL2 feature check) and P1.04b (the kernel build, the twin harness, VRAM budgets). 61 phases, M1 24 |
| 2026-11-27 | **P1.04a's second estimate:** code 760 (backends and selection 180, detection 150, buffers and streams 200, pinned memory 110, the WSL2 feature check 120) · tests 400 · examples 100 · docs-as-code 60 · glue 20 = **1,340**. CUDA toolkit 13.3.1 (2026-06-23), the newest in NVIDIA's WSL-Ubuntu repository; `cudarc` 0.19.10 (2026-09-24); `rayon` 1.12.0 (2026-04-14) |
| 2026-11-27 | **P1.04a measured 1,349 lines against its second estimate of 1,340 (×1.01).** All five WSL2 CUDA features work on this PC; the gate's G1c now lints, tests and smokes CUDA code with its feature on ([notes](m1/notes/P1.04a-compute-backends.md)) |
| 2026-11-27 | **P1.04b's second estimate,** before its code: code 625 (`build.rs` 90, loading and launching kernels 150, the scale pair 90, the twin harness 120, VRAM budgets 160, the kernel 15) · tests 650 · examples 100 · docs-as-code 25 · glue 20 = **1,420**, against 1,310 from P1.04's split. Under 2,000, so the phase stays whole |
| 2026-11-27 | **P1.04b measured 1,405 lines against its second estimate of 1,420 (×0.99).** Kernels are built as fat binaries (`-arch=all-major`), not PTX, which a driver older than the toolkit refuses (§6.6 updated, ADR-0010); copies with pinned memory now wait until they are done, a fix to P1.04a ([notes](m1/notes/P1.04b-kernels-and-the-twin-harness.md)) |
| 2026-11-27 | **P1.05's second estimate,** before its code: about 3,140 (code 1,790 · tests 1,100 · examples 150 · docs-as-code 50 · glue 50). Over 2,000: **the owner split it** into P1.05a (the Bayer split and both demosaics) and P1.05b (binning, resize, the display transform and JPEG). Binning moved from the ready split's first half to its second, so that both fit. 62 phases, M1 25 |
| 2026-11-27 | **P1.05a's second estimate:** code 1,020 (the crate and its images 150, the Bayer split and superpixel 170, bilinear 260, Malvar–He–Cutler 350, dispatch and the Bayer phase 90) · tests 650 · examples 120 · docs-as-code 30 · glue 40 = **1,860** |
| 2026-11-27 | **P1.05a measured 2,552 lines against its second estimate of 1,860 (×1.37),** over the 2,000 cap: code 1,270, tests 1,043. The tests grew with the CUDA twins and the mutants' findings; estimates for phases with twins now count tests at about the code's size. The imaging kernels live in `tetera-compute`, since launching one is unsafe (§6.2 updated) ([notes](m1/notes/P1.05a-bayer-and-demosaic.md#size)) |
| 2026-11-27 | **The owner's standing approval to the end of P1.18:** each phase is merged once its gate passes; estimates over 2,000 are split by the agent; Checkpoints A and B are run and recorded by the agent, and the owner runs all three demos at C; teta-wot gaps go upstream as U-phases under the same approval; P1.13's systemd unit is tested as a user service, and the owner installs the system one at C |
| 2026-11-27 | **P1.05b's second estimate,** before its code, counting tests at about the code's size (P1.05a's lesson): code 845 (binning 200, resize and its weight tables 285, the display transform 180, JPEG 70, device images of any element and dispatch 110) · tests 850 · examples 110 · docs-as-code 45 · glue 30 = **1,880**, against 1,370 from P1.05's split. Under 2,000, so the phase stays whole. **JPEG through libjpeg-turbo** (`turbojpeg` 1.5.1), the owner's choice: the pure-Rust `jpeg-encoder` 0.7.1 took 13.5 ms for 1920 × 1280 at quality 80, against the 10 ms budget. It needs `nasm`, which the owner installs |
| 2026-11-27 | **P1.05b measured 2,238 lines against its second estimate of 1,880 (×1.19),** over the 2,000 cap: code 1,022, tests 1,002. The kernels' safe launches, each checking what its `SAFETY` comment relies on, took about 80 lines each; later estimates count them. JPEG at 1920 × 1280, quality 80: 4.8 ms ([notes](m1/notes/P1.05b-binning-resize-display-transform-and-jpeg.md#size)) |
| 2026-11-27 | **P1.06's second estimate,** before its code, counting tests at the code's size and about 80 lines a safe launch: about 2,490 (code 1,160 · tests 1,100 · examples 160 · docs-as-code 40 · glue 30). Over 2,000: **split by the agent** under the owner's standing approval, along its ready split, into P1.06a (dark, flat, white balance and CCM, `IspParams`, the unfused chain) and P1.06b (the fused kernel, its equivalence, ADR-0011). 63 phases, M1 26 |
| 2026-11-27 | **P1.06a's second estimate:** code 790 (dark subtraction and its matching 205, the flat field and its gains 215, white balance, CCM and the sRGB curve 220, `IspParams` and the unfused chain 150) · tests 790 · examples 100 · docs-as-code 25 · glue 25 = **1,730** |
| 2026-11-27 | **P1.06a measured 2,315 lines against its second estimate of 1,730 (×1.34),** over the 2,000 cap: code 1,038, tests 974, examples 280. GPU phases keep landing at 1.2–1.4 times their estimates; later estimates multiply by 1.3 before comparing with the cap. **The chain resizes before colour,** in linear light: plan D7 and P1.06b's spec give the new order ([notes](m1/notes/P1.06a-dark-flat-and-colour.md#size)) |
| 2026-11-27 | **P1.06b's second estimate,** before its code: code 660 (the demosaic's device code shared through a header, which `build.rs` inlines for NVRTC, 40; the fused kernel, 350, tiled through shared memory if a direct one misses 10 ms; its launch and parameters 150; the fused path and the equivalence check in `tetera-imaging` 120) · tests 500 · examples 80 · docs-as-code 20 · glue 10 = **1,270**, about 1,650 at P1.06a's ×1.3. Under 2,000, so the phase stays whole |
| 2026-11-27 | **P1.06b measured 1,113 lines against its second estimate of 1,270 (×0.88):** the direct fused kernel met the 10 ms budget (4–5 ms for 20 MP to 1920 × 1280), so no tiling. It matches the unfused chain exactly (ADR-0011; [notes](m1/notes/P1.06b-the-fused-kernel.md#size)) |
| 2026-11-27 | **P1.07's second estimate,** before its code: about 2,850 (code 1,290 · tests 1,200 · examples 270 · docs-as-code 50 · glue 40). Over 2,000, and its ready split's second half still 2,380 at ×1.3: **split by the agent** under the owner's standing approval into P1.07a (denoise), P1.07b (PSFs and Richardson–Lucy on the CPU) and P1.07c (Richardson–Lucy on the GPU). 65 phases, M1 28 |
| 2026-11-27 | **P1.07a's second estimate:** code 450 (temporal averaging 160, the bilateral filter and its weight tables 290) · tests 420 · examples 120 · docs-as-code 20 · glue 10 = **1,020**, 1,330 at ×1.3. Its behaviour spec, committed with this estimate, chooses the bilateral filter over non-local means |
| 2026-11-27 | **P1.07a measured 1,430 lines against its second estimate of 1,020 (×1.40):** code 683, tests 584. Averaging 8 frames gains 9.0 dB, √8's; the bilateral filter keeps edges and smooths flat regions. Both are exact twins on CUDA ([notes](m1/notes/P1.07a-denoise.md#size)) |
| 2026-11-27 | **P1.07b's second estimate,** before its code: code 610 (PSFs 310: Bessel functions, Airy, scalar diffraction through focus, an NPY loader; Richardson–Lucy 300: N-dimensional FFTs over `rustfft`, padding, the iteration, total variation) · tests 610 · examples 130 · docs-as-code 25 · glue 20 = **1,395**, 1,810 at ×1.3. Under 2,000, so the phase stays whole. Its behaviour spec, committed with this estimate, states the tolerance P1.07c's GPU twin must meet. `rustfft` 6.4.1 (MIT or Apache-2.0) |
| 2026-11-27 | **P1.07b measured 1,835 lines against its second estimate of 1,395 (×1.32):** code 923, tests 674. Richardson–Lucy lowers the error against the truth: from 63.2 to 59.0 on the example's specimen, 57.9 with total variation. It is invariant to the PSF's scale, so the flux was broken on purpose by dropping the correlation step instead ([notes](m1/notes/P1.07b-psfs-and-richardson-lucy.md#size)) |
| 2026-11-27 | **P1.07c's second estimate,** before its code: code 655 (in `tetera-compute`: seven element-wise kernels, about 110 lines of CUDA, their safe launches about 45 lines each, and a cuFFT plan 60; in `tetera-imaging`, `DeviceDeconvolution` and the dispatch 170; the padding's geometry moves to `tetera-compute` for both backends to share) · tests 655 · examples 90 · docs-as-code 15 · glue 15 = **1,430**, 1,860 at ×1.3. Under 2,000, so the phase stays whole, against 650 from P1.07's split: each kernel is a launch of its own, so each can be twinned with its CPU step |
| 2026-11-27 | **P1.07c measured 1,404 lines against its second estimate of 1,430 (×0.98):** code 715, tests 576. Each element-wise step matches its CPU twin bit for bit; the whole iteration agrees with the CPU to about 1e-6 relative, against the spec's 1e-3. 20 iterations at 20 MP take about 350 ms on the GPU ([notes](m1/notes/P1.07c-richardson-lucy-on-the-gpu.md#size)) |
| 2026-11-27 | **P1.08's second estimate,** before its code, counting tests at the code's size: about 5,070 (code 2,330 · tests 2,330 · examples 280 · docs-as-code 85 · glue 45), against 1,880 at first. The stock nodes need a wrapper each, and GPU residency a device path for each. Over 2,000, and its ready split's halves still about 3,950 and 2,640 at ×1.3: **split by the agent** under the owner's standing approval into P1.08a (the model, recipes, validation and the CPU executor), P1.08b (the stock nodes and live runs), P1.08c (GPU residency and fusion) and P1.08d (batch runs and tiling). P1.14a now depends on P1.08c, and P1.19 on P1.08d. 68 phases, M1 31 |
| 2026-11-27 | **P1.08a's second estimate:** code 700 (ports, values, descriptors and the `Node` trait 180; recipes and their schema 110; validation and the order 230; the registry and two glue nodes 80; the CPU executor 100) · tests 700 · examples 60 · docs-as-code 30 · glue 20 = **1,510**, 1,960 at ×1.3 |
| 2026-11-27 | **P1.08a measured 2,447 lines against its second estimate of 1,510 (×1.62),** over the 2,000 cap: code 1,245, tests 945. Validation that says every problem once, precisely, took 459 lines alone. P1.08b–d's second estimates count messages and checks at this rate ([notes](m1/notes/P1.08a-the-pipeline-model.md#size)) |
| 2026-11-27 | **P1.08b's second estimate,** before its code, at P1.08a's rate: every stock node and live runs came to about 1,760 (code 785 · tests 785 · examples 150 · docs-as-code 25 · glue 15), 2,290 at ×1.3. Over 2,000: **re-split by the agent** under the owner's standing approval. P1.08b keeps the raw path's ten nodes, with an encoded image as a port type: code 620 · tests 620 · examples 120 · docs-as-code 20 · glue 10 = **1,390**, 1,810 at ×1.3. The denoise and deconvolution nodes join live runs in a new P1.08d (900), after P1.08c; batch runs and tiling become P1.08e. P1.14a now depends on P1.08d, and P1.19 on P1.08e. 69 phases, M1 32 |
| 2026-11-27 | **P1.08b measured 1,346 lines against its second estimate of 1,390 (×0.97):** code 577, tests 591. The live chain's recipe gives exactly the preview `Isp::run` gives, on either backend; on CUDA it is slower than the CPU until P1.08c keeps data on the GPU ([notes](m1/notes/P1.08b-the-raw-paths-nodes.md#size)) |
| 2026-11-27 | **P1.08c's second estimate,** before its code: code 670 (values on the device, and the copies counted, 150; eight device paths, short since `tetera-imaging` has each `_device` function, 120; the executor's residency 180; the fusion pass 220) · tests 670 · examples 70 · docs-as-code 20 · glue 10 = **1,440**, 1,870 at ×1.3. Under 2,000, so the phase stays whole |
| 2026-11-27 | **P1.08c measured 1,528 lines against its second estimate of 1,440 (×1.06):** code 738, tests 744. On the GPU the live chain copies only the frame up and the preview down; the fused kernel replaces it and gives the same preview. Held inputs, added to the scope, make constant masters cost nothing after the first run ([notes](m1/notes/P1.08c-gpu-residency-and-fusion.md#size)) |
| 2026-11-27 | **P1.08d's second estimate,** before its code: code 400 (three nodes 210; live runs 190) · tests 400 · examples 80 · docs-as-code 15 · glue 5 = **900**, 1,170 at ×1.3, as P1.08b's re-split estimated |
| 2026-11-27 | **P1.08d measured 1,054 lines against its second estimate of 900 (×1.17):** code 394, tests 550. Every stock node is in the registry. Live runs keep one frame waiting at most, the latest; the live chain runs live at 14.5 fps on CUDA, every frame ([notes](m1/notes/P1.08d-denoise-and-deconvolution-nodes-and-live-runs.md#size)) |
| 2026-11-27 | **P1.08e's second estimate,** before its code: code 340 (batch runs 40; each step's reach 60; the halo 40; tiles cut and stitched 120; the tiled run 80) · tests 340 · examples 80 · docs-as-code 15 · glue 5 = **780**, 1,014 at ×1.3 |
| 2026-11-27 | **P1.08e measured 861 lines against its second estimate of 780 (×1.10):** code 326, tests 379. Tiled runs give exactly the whole frame's outputs. P1.08's five phases came to 7,236 lines, against 5,070 at its second estimate and 1,880 at its first: the model's validation and the GPU's residency were most of the difference ([notes](m1/notes/P1.08e-batch-runs-and-tiling.md#size)) |
| 2026-11-27 | **P1.09's second estimate,** before its code, counting tests at the code's size: about 2,190 (code 1,120 · tests 900 · examples 100 · docs-as-code 40 · glue 30), 2,850 at ×1.3. Over 2,000: **split by the agent** under the owner's standing approval into P1.09a (the camera, its fake and contract suite, ADR-0013) and P1.09b (the stage, the objectives and the light, the fake stage and its suite). P1.10 now depends on P1.09b. 70 phases, M1 33 |
| 2026-11-27 | **P1.09a's second estimate:** code 550 (events and errors 100; the `Camera` trait, capabilities and configuration 250; the fake camera 200) · tests 500 · examples 60 · docs-as-code 20 · glue 20 = **1,150**, 1,495 at ×1.3 |
| 2026-11-27 | **P1.09a measured 1,775 lines against its second estimate of 1,150 (×1.54):** code 670, tests 938. The contract suite counts as tests, and has tests of its own, written for the mutants: a camera flawed on purpose fails each check ([notes](m1/notes/P1.09a-the-camera.md#size)) |
| 2026-11-27 | **P1.09b's second estimate:** code 600 (the `Stage` trait, its capabilities, readings and events 180; the fake stage 250; the objective catalogue and mount 100; illumination 70) · tests 750 (the contract suite 250, a flawed stage for each check 200, unit tests 300) · examples 80 · docs-as-code 20 · glue 40 = **1,490**, 1,937 at ×1.3, up from 1,170 at the split: P1.09a's suite needed tests of its own. New dependency: `toml` 1.2.1, for the objective catalogue |
| 2026-11-27 | **P1.09b measured 2,311 lines against its second estimate of 1,490 (×1.55):** code 969, tests 1,115. The fake stage's timed motions and faults, and the catalogue's and the light's checks, were most of the difference. P1.09's two phases came to 4,086 lines, against about 2,190 at its second estimate ([notes](m1/notes/P1.09b-the-stage-the-objectives-and-the-light.md#size)) |
| 2026-11-27 | **P1.10's second estimate,** before its code: about 2,510 (code 1,100 · tests 1,150 · examples 180 · docs-as-code 40 · glue 40), 3,263 at ×1.3. Over 2,000: **split by the agent** under the owner's standing approval into P1.10a (the simulated clock and motion profiles, ADR-0014) and P1.10b (the virtual stage). P1.12 now depends on P1.10b. 71 phases, M1 34 |
| 2026-11-27 | **P1.10a's second estimate:** code 450 (the clock 200; motion profiles 250) · tests 500 · examples 80 · docs-as-code 20 · glue 30 = **1,080**, 1,404 at ×1.3 |
| 2026-11-27 | **P1.10a measured 1,077 lines against its second estimate of 1,080 (×1.00):** code 431, tests 490. The property tests found two rounding faults before the code was committed ([notes](m1/notes/P1.10a-the-simulated-clock-and-motion-profiles.md#size)) |
| 2026-11-27 | **P1.10b's second estimate:** code 670 (configuration 80; the plan and grblHAL's states 150; moves, jogs, holds, homing and faults 200; reports, the track and events 120; its thread 40; the `Stage` trait 80) · tests 650 · examples 120 · docs-as-code 20 · glue 20 = **1,480**, 1,924 at ×1.3 |
| 2026-11-27 | **P1.10b measured 1,500 lines against its second estimate of 1,480 (×1.01):** code 738, tests 577. P1.10's two phases came to 2,577 lines, against about 2,510 at its second estimate ([notes](m1/notes/P1.10b-the-virtual-stage.md#size)) |
| 2026-11-27 | **P1.11's second estimate,** before its code: about 2,830 (code 1,570 · tests 850 · examples 280 · docs-as-code 80 · glue 50), 3,679 at ×1.3. Over 2,000: **split by the agent** under the owner's standing approval as its ready split proposed, into P1.11a (the model on the CPU, the behaviour spec, ADR-0015) and P1.11b (the GPU kernel and the tile cache). P1.12 now depends on P1.11b. 72 phases, M1 35 |
| 2026-11-27 | **P1.11a's second estimate:** code 820 (hashing and noise 150; the section and its tissue 450; colour 60; rendering 160) · tests 450 · examples 160 · docs-as-code 40 · glue 20 = **1,490**, 1,937 at ×1.3 |
| 2026-11-27 | **P1.11a measured 1,728 lines against its second estimate of 1,490 (×1.16):** code 776, tests 697. A golden test pins the procedural model, which property tests alone couldn't: the first mutants run missed 339 of its constants ([notes](m1/notes/P1.11a-the-he-sample-on-the-cpu.md#size)) |
| 2026-11-27 | **P1.11b's second estimate:** code 800 (the CUDA kernels 450; launching them and half precision 100; tiles and the cache 250) · tests 400 · examples 80 · docs-as-code 30 · glue 30 = **1,340**, 1,742 at ×1.3, as at the split |
| 2026-11-27 | **P1.11b measured 1,524 lines against its second estimate of 1,340 (×1.14):** code 977, tests 389. The GPU twin agrees with the CPU within 6e-4 OD, and renders a 512 × 512 region at 50× in 1.87 ms. P1.11's two phases came to 3,252 lines, against about 2,830 at its second estimate ([notes](m1/notes/P1.11b-the-he-sample-on-the-gpu-and-its-tiles.md#size)) |
| 2026-11-27 | **P1.12's second estimate,** before its code: about 2,960 (code 1,650 · tests 900 · examples 250 · docs-as-code 100 · glue 60), 3,848 at ×1.3. Over 2,000: **split by the agent** under the owner's standing approval as its ready split proposed, into P1.12a (optics and light, the field the sensor sees, ADR-0016) and P1.12b (the virtual camera, Checkpoint A). Later phases that depended on P1.12 now depend on P1.12b. 73 phases, M1 36 |
| 2026-11-27 | **P1.12a's second estimate:** code 850 (the field of view and depth groups 120; blur, vignetting and illumination 150; the CPU field 150; the GPU's kernels 250 and their launch 100; `VirtualIllumination` 80) · tests 450 · examples 100 · docs-as-code 40 · glue 30 = **1,470**, 1,911 at ×1.3 |
| 2026-11-27 | **P1.12a measured 2,290 lines against its second estimate of 1,470 (×1.56):** code 1,149, tests 833. A full field still takes 9, 19 and 39 ms at 5×, 20× and 50× on the GPU, within the 40 ms budget; moving, 50× runs 1 to 6 ms over ([notes](m1/notes/P1.12a-optics-and-light-the-field-the-sensor-sees.md#size)) |
