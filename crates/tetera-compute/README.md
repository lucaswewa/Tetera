# tetera-compute

Tetera's compute backends, L1 ([ADR-0009](../../docs/adr/0009-compute-backends-the-cpu-reference-and-cuda.md)): the CPU, with rayon, which is the reference and the fallback, and CUDA, through `cudarc` with its libraries loaded at run time. It may use unsafe code, each block with its `SAFETY:` comment.

- **`Compute::detect`** chooses CUDA when the build has the `cuda` feature, the driver loads and a device is there. Otherwise, or with `TETERA_FORCE_CPU=1`, it chooses the CPU, and says why.
- **`CudaBackend`** holds device buffers (upload, zeros, download) and streams. Its `PinnedAllocator` puts a frame pool in page-locked memory.
- **`features::check_all`** tries what CUDA offers on this machine: pinned memory, cuFFT, NVRTC, nvJPEG, IPC memory handles.

P1.04b adds kernels, the harness that proves the backends agree, and VRAM budgets.

The `cuda` feature needs no toolkit to build. To run on the GPU, it needs the Windows NVIDIA driver, and for NVRTC, cuFFT and nvJPEG the CUDA toolkit 13.3 in WSL ([docs/setup/toolchain.md](../../docs/setup/toolchain.md)).

## How to try it

```bash
cargo run -p tetera-compute --example compute_info --features cuda
cargo run -p tetera-compute --example compute_info
TETERA_FORCE_CPU=1 cargo run -p tetera-compute --example compute_info --features cuda
```

- **With the feature,** on this PC: CUDA on the RTX 3070 Ti Laptop GPU (compute capability 8.6, 8.0 GiB), a driver that runs CUDA up to 13.5, and five features that work.
- **Without the feature,** or forced: the CPU, with the reason, and nothing tried.

`-- --smoke` adds `smoke: ok`. The CUDA tests run with `TETERA_CUDA=1 cargo test -p tetera-compute --features cuda`.
