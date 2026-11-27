# ADR-0009: Compute backends: the CPU reference and CUDA, and what WSL2 offers

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.04a

## Context

- **The spec puts image processing on the GPU, with the CPU as its fallback:** "GPU first, then fallback to CPU if GPU is not available".
- **Tetera's gates run on machines without a GPU too,** and its builds must work without the CUDA toolkit installed.
- **CUDA under WSL2** reaches the GPU through the Windows driver (`/usr/lib/wsl/lib/libcuda.so.1`). NVIDIA's WSL guide lists features that differ from native Linux, so what Tetera plans to use had to be tried, not assumed (plan R5).
- **`cudarc`** binds CUDA's driver API, NVRTC and cuFFT, and can load their libraries at run time. Asked for a library that won't load, it panics rather than failing.
- **Frames live in a pool of fixed buffers** (ADR-0008). Copies to the GPU from pinned, page-locked memory run at the bus's full speed.

## Decision

1. **Two backends:**
   - `CpuBackend`, a rayon pool of its own: the reference for every CUDA kernel (P1.04b), and the fallback;
   - `CudaBackend`: a device's context and default stream, through `cudarc` 0.19.10 with `dynamic-loading` and CUDA 13.3's bindings.
2. **The `cuda` feature** builds the CUDA path. With its libraries loaded at run time, it builds without the toolkit.
3. **The choice** is `select` over what a `Probe` reports, so tests fake any machine:
   - CPU, when `TETERA_FORCE_CPU` is set;
   - CPU, when the build has no `cuda` feature;
   - CPU, when the driver's library won't load or finds no device;
   - otherwise CUDA, on the device with the most memory, the lowest number among equals.

   A device chosen that won't start leaves the CPU in use. Either way, `Compute::why` says why.
4. **Every library is checked to load** (`library_loads`) before `cudarc` touches it, so a missing driver is an error, never a panic.
5. **Device buffers** are uploaded, zeroed and downloaded on the default stream; more streams can be made for overlapping work.
6. **`PinnedAllocator`** gives frame pools page-locked memory: portable, cached for the CPU, not write-combined, since the CPU reads frames too. It is zeroed at allocation. Its buffer reads through the pointer taken when it was allocated: frames are uploaded only once frozen, so no copy writes to a buffer that is being read.
7. **The CUDA toolkit is pinned at 13.3,** the newest NVIDIA's WSL-Ubuntu repository offers (13.3.1). The driver, 621.24, runs CUDA up to 13.5. `docs/setup/toolchain.md` gives the install.
8. **What WSL2's CUDA offers,** tried by `features::check_all` on 2026-11-27, on an RTX 3070 Ti Laptop GPU (compute capability 8.6, 8 GiB) under WSL 3.0.5:

   | Feature | What was tried | Result |
   |---|---|---|
   | Pinned memory | 640 MiB of page-locked host memory, a pool of 16 full frames | Works |
   | cuFFT | A 1,024-point complex plan, `libcufft.so.12` | Works |
   | NVRTC | A kernel compiled at run time, `libnvrtc.so.13` | Works |
   | nvJPEG | A library handle made and released, `libnvjpeg.so.13` | Works |
   | IPC memory handles | A handle for memory from `cuMemAlloc` | Works: the handle is made. Opening it from another process wasn't tried |

   **IPC needs memory from `cuMemAlloc`.** `cudarc` allocates from a stream-ordered pool, whose memory never has an IPC handle, on any system. The first probe used `cudarc`'s allocation and failed with `CUDA_ERROR_INVALID_VALUE`, which looked like a WSL2 limit and wasn't. M2's Python workers, if they share frames on the GPU, must allocate those with `cuMemAlloc`.

## Consequences

- **Gains:**
  - the same build runs on any machine, using the GPU when there is one;
  - the reason for the choice is always known;
  - frame pools can live in pinned memory;
  - the features P1.05 to P1.07 and M2 rely on are confirmed on this machine.
- **Costs:**
  - CUDA code compiles only with the `cuda` feature, so the gate's G1c runs clippy, the tests and the examples' smoke runs with it on (ADR-0005's gate, extended in this phase);
  - unsafe code in this crate, each block with its `SAFETY:` comment.
- **To watch:**
  - the instrument PC's own feature table, at M5;
  - opening an IPC handle across processes, when M2 needs it;
  - `cudarc`'s deprecations, such as `memcpy_stod` becoming `clone_htod` in 0.19.

## Alternatives considered

- **Linking CUDA at build time:** every build would need the toolkit, and a machine without a driver couldn't start.
- **Catching `cudarc`'s panics** instead of checking libraries first: a panic as control flow, and some panics abort.
- **Write-combined pinned memory,** `cudarc`'s default: faster uploads, but CPU reads of frames many times slower.
- **wgpu or Vulkan compute:** portable, but without cuFFT, NVRTC or nvJPEG, and the spec names CUDA.
