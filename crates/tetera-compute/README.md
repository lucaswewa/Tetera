# tetera-compute

Tetera's compute backends, L1 ([ADR-0009](../../docs/adr/0009-compute-backends-the-cpu-reference-and-cuda.md)): the CPU, with rayon, which is the reference and the fallback, and CUDA, through `cudarc` with its libraries loaded at run time. It may use unsafe code, each block with its `SAFETY:` comment.

- **`Compute::detect`** chooses CUDA when the build has the `cuda` feature, the driver loads and a device is there. Otherwise, or with `TETERA_FORCE_CPU=1`, it chooses the CPU, and says why.
- **`CudaBackend`** holds device buffers (upload, zeros, download) and streams. Copies return once they are done, even from pinned memory. Its `PinnedAllocator` puts a frame pool in page-locked memory.
- **`features::check_all`** tries what CUDA offers on this machine: pinned memory, cuFFT, NVRTC, nvJPEG, IPC memory handles.
- **Kernels** ([ADR-0010](../../docs/adr/0010-building-and-loading-cuda-kernels-under-wsl2.md)) are CUDA C files in `kernels/`. With the `cuda` feature, `build.rs` compiles each to a fat binary when it finds `nvcc` (`NVCC` names it; `NVCC=none` skips it): machine code for every major GPU architecture from compute capability 7.5, and PTX for newer ones. It embeds both the fat binary and the source. Without `nvcc`, NVRTC compiles the source when the module first loads. `CudaBackend::module` loads each module once.
- **`ops`** holds each operation twice: a CPU twin, and a CUDA kernel. The first is `scale_offset`, 16-bit samples to floats, fused on both sides so the two agree bit for bit.
- **`twin::compare`** compares a kernel's result with its twin's within a `Tolerance`, and says where they differ most. `cuda_or_skip!()` gives a test the GPU when `TETERA_CUDA=1` is set.
- **`VramBudgets`** shares out the GPU's memory: each user reserves a budget, and takes grants from it; a request beyond either is refused as `Busy`.
- **`imaging`** holds the imaging kernels (`bayer.cu`, `demosaic.cu`, `resize.cu`, `display.cu`) and their launches: the Bayer split, the superpixel, binning, bilinear and Malvar-He-Cutler demosaicking, resizing by weight tables, and the display transform by lookup tables. Their CPU twins, and the images they work on, are in [`tetera-imaging`](../tetera-imaging/README.md): every kernel lives here, since launching one is unsafe.
- **`isp`** holds the ISP's corrections (`isp.cu`): dark subtraction, the flat field, and colour, a fixed-point matrix then a table per channel.
- **`fused`** launches the fused ISP kernel (`fused.cu`, [ADR-0011](../../docs/adr/0011-fused-kernels.md)): crop or bin, dark, flat, demosaic, resize and colour in one pass, matching the unfused chain exactly. The demosaic's device code is shared with `demosaic.cu` through `demosaic.cuh`, which `build.rs` inlines for NVRTC.
- **`denoise`** holds temporal averaging's window on the device and the bilateral filter's tables and launch (`denoise.cu`).

The `cuda` feature needs no toolkit to build. To run on the GPU, it needs the Windows NVIDIA driver, and for NVRTC, cuFFT and nvJPEG the CUDA toolkit 13.3 in WSL ([docs/setup/toolchain.md](../../docs/setup/toolchain.md)).

## How to try it

```bash
cargo run -p tetera-compute --example compute_info --features cuda
cargo run -p tetera-compute --example compute_info
TETERA_FORCE_CPU=1 cargo run -p tetera-compute --example compute_info --features cuda
```

- **With the feature,** on this PC: CUDA on the RTX 3070 Ti Laptop GPU (compute capability 8.6, 8.0 GiB), a driver that runs CUDA up to 13.5, and five features that work.
- **Without the feature,** or forced: the CPU, with the reason, and nothing tried.

```bash
cargo run --release -p tetera-compute --example twin_scale --features cuda
NVCC=none cargo run --release -p tetera-compute --example twin_scale --features cuda
cargo run --release -p tetera-compute --example twin_scale
```

`twin_scale` scales and offsets a full frame, 5,472 × 3,648, on the CPU and, with the feature, in the kernel, and compares the two. On this PC: about 6 ms on 16 CPU threads, 0.35 ms in the kernel, and the twins identical, from `nvcc`'s fat binary or NVRTC's PTX.

`-- --smoke` adds `smoke: ok` to either example. The CUDA tests run with `TETERA_CUDA=1 cargo test -p tetera-compute --features cuda`.
