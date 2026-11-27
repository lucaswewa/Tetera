# ADR-0010: Building and loading CUDA kernels under WSL2

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.04b

## Context

- **Tetera's imaging runs on CUDA, with the CPU as the reference and the fallback** (ADR-0009). Every kernel from P1.05 on needs a way to be built, loaded and launched, and a way to prove it computes what the CPU computes.
- **The workspace must build without the CUDA toolkit** (ADR-0009), and its gates run on machines without a GPU.
- **CUDA kernels can reach the driver in three forms:**
  - a cubin, machine code for one GPU architecture, which runs on that architecture's later minor versions too;
  - PTX, a virtual instruction set the driver compiles for whichever GPU it finds;
  - CUDA C source, compiled to PTX at run time by NVRTC.

  A fat binary bundles cubins and PTX, and the driver picks what fits its GPU.
- **A driver refuses PTX from a toolkit newer than itself:** OScope's plan built PTX alone, and its driver, which ran CUDA 13.2, refused PTX from CUDA 13.4 with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (its P07 notes). Cubins from any CUDA 13 toolkit run on any CUDA 13 driver.
- **On this PC,** under WSL2, CUDA 13.3's `nvcc` and NVRTC (`libnvrtc.so.13`) both work, and the driver runs CUDA up to 13.5. The GPU, an RTX 3070 Ti Laptop GPU, is compute capability 8.6. The instrument PC's GPU and driver aren't known yet.
- **Floating point differs between backends** unless both sides round the same way. A multiply and an add rounded once, fused, can differ in the last bit from the two rounded separately. CUDA's compilers fuse `a × b + c` on their own by default; Rust never does.
- **`cudarc` doesn't wait for a copy from or to a plain slice.** From pageable memory the driver waits anyway; from pinned memory, the copy runs on after the call returns. Frame pools live in pinned memory (ADR-0009). OScope found the same.
- **The GPU's memory is shared:** the simulator's tile cache (P1.11), the preview (P1.14a) and compute jobs (P1.15a) all want some of 8 GiB, and none may starve the others.

## Decision

1. **Kernels are CUDA C files under `crates/tetera-compute/kernels/`,** one module per file.
2. **`build.rs` compiles each to a fat binary** when the `cuda` feature is on and `nvcc` is found:
   - `nvcc --fatbin -arch=all-major -O3`: with CUDA 13.3, cubins for compute capabilities 7.5, 8.0, 9.0, 10.0, 11.0 and 12.0, and PTX for 12.0, for GPUs newer than the toolkit;
   - `nvcc` is the one `NVCC` names, when it is set; otherwise `nvcc` on the path, or `/usr/local/cuda/bin/nvcc`;
   - a kernel `nvcc` refuses fails the build, with its errors.
3. **Each module is embedded in the binary** twice over: its fat binary, when `nvcc` made one, and always its CUDA C source.
4. **Without `nvcc`,** the build warns, and the module's source is compiled by NVRTC when it is first loaded. `NVCC=none` forces this. Without the `cuda` feature nothing is compiled, so the workspace builds without the toolkit.
5. **A backend loads each module once,** on first use, and keeps it; clones of the backend share it. A module that fails to load isn't kept, so the next use tries again.
6. **Launches** are one-dimensional for now, in blocks of 256 threads:
   - lengths are checked before the launch, and an empty input launches nothing;
   - more than `u32::MAX` elements is refused;
   - each launch is an `unsafe` block whose `SAFETY:` comment matches the kernel's parameters to its arguments, in order and type, and says which indices it touches.

   A launch returns without waiting; a download, `synchronize` or `time_ms` waits.
7. **Copies between host and device return once they are done** (this refines ADR-0009's buffers): `upload` returns once the copy has read the host's memory, and `download_into` once it has written it. A pinned frame can go back to its pool as soon as its upload returns. Overlapping copies with work, on streams of their own, comes with the pipelines that need it.
8. **Every kernel has a CPU twin** in `ops`, computing the same thing in the same order:
   - `twin::compare` compares the two results value by value within a `Tolerance`, `|actual − expected| ≤ absolute + relative × |expected|`. NaN matches only NaN, and an infinity only the same infinity;
   - it reports how many values are beyond the tolerance, and where the two are furthest apart;
   - each operation states its tolerance in its test. A kernel that rounds the same as its twin is held to `Tolerance::EXACT`.
9. **Fused multiply-adds are written out on both sides:** `fmaf` in the kernel, `mul_add` in the twin. Neither side depends on what its compiler chooses to fuse, so scale and offset agree bit for bit.
10. **CUDA tests take their GPU from `cuda_or_skip!()`:** the backend with `TETERA_CUDA=1`, or a note and a return without it. With it set, a device that won't start fails the test.
11. **Kernel time is measured by events,** `CudaBackend::time_ms`, either side of the work on the default stream.
12. **VRAM budgets** share out the GPU's memory:
    - `VramBudgets` holds a capacity, by default the free memory less a share kept for the driver and other programs;
    - each user, by name, reserves one budget; a reservation beyond the unreserved capacity is refused;
    - users take grants from their budget before they allocate; a grant beyond what is left of it is refused;
    - both refusals are `Busy` errors, which say how much was asked for and how much is left;
    - a grant gives its bytes back when it drops. A budget dropped while its grants are held stays reserved until the last one drops.

## Consequences

- **Gains:**
  - one build runs on any GPU from compute capability 7.5 with any CUDA 13 driver, so the instrument PC's needn't be known now;
  - a kernel's compile errors appear at build time, where `nvcc` is installed;
  - a machine without the toolkit still runs the kernels, through NVRTC;
  - every kernel is proved against the CPU, on this PC, by the gate's G1c;
  - a pinned frame can't be reused while the GPU still reads it;
  - no user of the GPU can take another's memory.
- **Costs:**
  - six cubins per kernel: 29 KiB for scale and offset, and a slower build of the kernels;
  - each operation is written twice, once per backend;
  - copies wait, so they don't yet overlap with work;
  - budgets are bookkeeping: a user that allocates without a grant isn't stopped.
- **To watch:**
  - NVRTC's PTX has the same limit as `nvcc`'s: a driver older than the NVRTC that made it refuses it. The fallback is for machines without `nvcc`, where the toolkit and the driver usually match;
  - kernels whose twins can't agree exactly, such as sums whose order differs across threads: each states its tolerance and why;
  - tying device allocations to grants, once the pipeline engine (P1.08) allocates on its users' behalf.

## Alternatives considered

- **PTX alone,** as the plan said: one small file, but refused by a driver older than the toolkit, which OScope hit.
- **A cubin for this PC's GPU alone** (`sm_86`): the smallest build, but it wouldn't run on the instrument PC unless its GPU were the same generation.
- **NVRTC alone, always at run time:** no `nvcc` needed to build, but every compile error would wait until a kernel's first use, and PTX's limit applies.
- **Requiring `nvcc` for the `cuda` feature:** a build without the toolkit is an aim of ADR-0009.
- **A tolerance for every kernel, even scale and offset:** a wrong kernel can hide inside a loose tolerance. Where the two can agree exactly, they must.
- **Copies that hold the frame until an event says they are done,** instead of waiting: they overlap with work, but need the frame shared with the copy. That waits for the preview pipeline (P1.14a), which will measure whether it matters.
- **Budgets enforced by an allocator that refuses:** a wrapper around every allocation, before any user exists. Grants come first, and the allocator can check them later.
