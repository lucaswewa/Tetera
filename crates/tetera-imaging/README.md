# tetera-imaging

Tetera's imaging algorithms, L1 (plan §6.6). Each has a CPU twin here, and, with the `cuda` feature, a CUDA kernel in [`tetera-compute`](../tetera-compute/README.md), the crate allowed to launch one. The two agree bit for bit wherever the arithmetic allows it; so far, everywhere. This crate has no unsafe code.

- **`Mosaic`:** a Bayer mosaic of 16-bit sites with its `MosaicLayout`: size, pattern and significant bits. It comes from a raw frame (`Mosaic::from_frame`), and a crop at an odd column or row moves its pattern.
- **`Image<T>`:** interleaved channels in 16 bits or floats. `DeviceMosaic` and `DeviceImage` are the same on a GPU.
- **`bayer`:** the four colour planes, the half-resolution superpixel, and RGB sampled into a mosaic.
- **`demosaic`:** bilinear, and Malvar-He-Cutler (ICASSP 2004), to RGB in 16 bits, rounded and clamped, or floats, exact. Both are filters of whole-number weights over a power of two, and mirror the mosaic at its edges without repeating them, which keeps every site's colour.
- **`pattern`:** a test pattern with smooth colour, a zone plate and colour bars, and the PSNR of an image against it.

Each operation runs on whichever backend a `Compute` chose. `_cpu` runs its twin, and `_device` leaves data on the GPU.

## How to try it

```bash
cargo run --release -p tetera-imaging --example demosaic --features cuda
cargo run --release -p tetera-imaging --example demosaic
```

`demosaic` makes a full-frame test pattern, 5,472 × 3,648 in 12 bits, and mosaics it as RGGB. It then takes it back to RGB three ways, on each backend. For each it prints the time, the PSNR against the pattern, and whether the CPU and CUDA agree. On this PC:

| | CPU, 16 threads | CUDA kernel | PSNR |
|---|---|---|---|
| Superpixel | 5.7 ms | 0.20 ms | |
| Bilinear | 47 ms | 0.56 ms | 31.0 dB |
| Malvar-He-Cutler | 94 ms | 0.75 ms | 36.1 dB |

It writes the pattern and both results to `target/demosaic` as 16-bit PPM files, each cropped to at most 960 × 640 from its centre. `-- --out DIR` writes them elsewhere, and `-- --smoke` takes a 640 × 480 frame, whole, once.

The CUDA tests run with `TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda`.
