# tetera-imaging

Tetera's imaging algorithms, L1 (plan §6.6). Each has a CPU twin here, and, with the `cuda` feature, a CUDA kernel in [`tetera-compute`](../tetera-compute/README.md), the crate allowed to launch one. The two agree bit for bit wherever the arithmetic allows it; so far, everywhere. This crate has no unsafe code.

- **`Mosaic`:** a Bayer mosaic of 16-bit sites with its `MosaicLayout`: size, pattern and significant bits. It comes from a raw frame (`Mosaic::from_frame`), and a crop at an odd column or row moves its pattern.
- **`Image<T>`:** interleaved channels in 16 bits or floats. `DeviceMosaic` and `DeviceImage` are the same on a GPU.
- **`bayer`:** the four colour planes, the half-resolution superpixel, and RGB sampled into a mosaic.
- **`demosaic`:** bilinear, and Malvar-He-Cutler (ICASSP 2004), to RGB in 16 bits, rounded and clamped, or floats, exact. Both are filters of whole-number weights over a power of two, and mirror the mosaic at its edges without repeating them, which keeps every site's colour.
- **`pattern`:** a test pattern with smooth colour, a zone plate and colour bars, and the PSNR of an image against it.
- **`bayer::bin`:** binning by 2, 3 or 4, by sum or mean, keeping the Bayer layout. A sum adds the bits it needs, up to 16.
- **`resize`:** area averaging to shrink, bilinear to enlarge. The weights are integers in tables made once for a pair of sizes and shared with the GPU, so the twins agree exactly.
- **`display`:** black and white levels for each channel, then a tone curve (linear, a gamma, or sRGB), to 8 bits, as one lookup table per channel.
- **`jpeg`:** JPEG through libjpeg-turbo, RGB in 4:2:0 or grey. A `JpegEncoder` is kept for a stream of frames, MJPEG. Building libjpeg-turbo needs `cmake` and `nasm`; its licences are in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
- **`correct`:** dark subtraction, clamped at zero, with a `DarkLibrary` that picks the master dark matching a frame's mode, binning, gain and exposure; and the `FlatField`, a gain per site in 1/4096ths, each channel evened out, near-zero sites left alone.
- **`colour`:** white balance and a colour matrix as one fixed-point matrix, then levels and the transfer curve to 8 bits.
- **`isp`:** the whole chain, `IspParams` to `Isp`, from a raw frame to a preview: crop or bin → dark → flat → demosaic → resize → colour → 8 bits. It resizes before colour, in linear light. On either backend, and the two agree exactly. `Isp::run_fused` runs it as one fused kernel on the GPU (P1.06b), which matches it exactly too; `compare_previews` measures two previews against the fused kernel's tolerance.
- **`pattern::Sensor`:** a synthetic sensor with an offset, vignetting and unequal channels, for testing the ISP.
- **`denoise`:** temporal averaging, the exact mean of the last N frames, reset when the stage moves; and the bilateral filter, weights from fixed-point tables, one range weight for all three channels. `pattern::noisy` adds seeded Gaussian noise for tests.
- **`psf`:** point spread functions, each centred and summing to 1. The widefield PSF comes from the NA, the wavelength and the pixel size, in focus (the Airy pattern) or through focus (scalar diffraction), each pixel averaged over 5 × 5 points; a measured one comes from a NumPy `.npy` file.
- **`deconvolve`:** Richardson–Lucy in 2D and 3D on the CPU, with total-variation regularisation if asked for. Its convolutions are FFTs (`rustfft`) over the image mirrored at its edges, padded to sizes of small primes. `convolve_cpu` blurs an image by a PSF the same way. On the GPU (P1.07c), `DeviceDeconvolution` runs the same steps through cuFFT, each element-wise step a kernel matching its CPU twin bit for bit. The two FFTs round differently, so the backends agree within the behaviour spec's tolerance: about 1e-6 relative after 20 iterations, against the 1e-3 allowed. `richardson_lucy` runs on whichever backend was chosen.

- **`focus`** (P1.16a): focus measures, each higher when sharper, on a raw frame's green channel: each 2 × 2 cell's two greens averaged, then binned to at most 512 across. Brenner's gradient, the variance of the Laplacian, normalised variance (the default) and Tenengrad, as P1.16a's behaviour spec chose them. On the CPU only: the binned image is small.

Each operation runs on whichever backend a `Compute` chose. `_cpu` runs its twin, and `_device` leaves data on the GPU.

## How to try it

```bash
cargo run --release -p tetera-imaging --example demosaic --features cuda
cargo run --release -p tetera-imaging --example demosaic
```

`demosaic` makes a full-frame test pattern, 5,472 × 3,648 in 12 bits, and mosaics it as RGGB. It then takes it back to RGB three ways, on each backend. For each it prints the time, the PSNR against the pattern, and whether the CPU and CUDA agree. On this PC:

| | CPU, 16 threads | CUDA kernel | PSNR |
|---|---|---|---|
| Superpixel | 5.0 ms | 0.20 ms | |
| Bilinear | 44 ms | 0.55 ms | 31.0 dB |
| Malvar-He-Cutler | 93 ms | 0.74 ms | 36.1 dB |

Then it runs the live view's chain, timed on each backend, and checks that the two 8-bit previews are identical:

| | CPU, 16 threads | CUDA |
|---|---|---|
| Bin 2 × 2, by the mean | 2.8 ms | 0.21 ms |
| Malvar-He-Cutler, 2,736 × 1,824 | 23.4 ms | 2.1 ms |
| Resize to 1,920 × 1,280 | 9.3 ms | 0.45 ms |
| The display transform, sRGB | 2.5 ms | 0.16 ms |
| JPEG at quality 80 | 4.8 ms | on the CPU |

The CUDA column includes allocating each step's output.

It writes the pattern and both results to `target/demosaic` as 16-bit PPM files, each cropped to at most 960 × 640 from its centre, and the preview as `preview.jpg`. `-- --out DIR` writes them elsewhere, and `-- --smoke` takes a 640 × 480 frame, whole, once.

```bash
cargo run --release -p tetera-imaging --example isp --features cuda
```

`isp` puts the test pattern through the synthetic sensor, then through the ISP with the sensor's own dark and flat field and white balance, from a full frame and from a 2 × 2 binned one, to 1920 × 1280. It times each stage on each backend, checks that the previews agree, and writes them to `target/isp`. On this PC, the binned chain takes 38 ms on the CPU and 2.4 ms with CUDA; the full frame, 107 ms and 9.6 ms, most of either in demosaicking. The fused kernel takes 5.3 ms from the full frame, inside the plan's 10 ms.

```bash
cargo run --release -p tetera-imaging --example denoise --features cuda
```

`denoise` adds noise of σ 80 to a full 12-bit frame. Averaging 8 frames raises its PSNR by 9.0 dB, as √8 predicts, in 1.0 ms a frame on the GPU. The bilateral filter on one demosaicked frame raises it by 1.3 dB, in 18 ms on the GPU and 290 ms on the CPU.

```bash
cargo run --release -p tetera-imaging --example deconvolve --features cuda
```

`deconvolve` blurs a synthetic specimen of beads and discs, 1,024 × 1,024, by a widefield PSF (NA 0.5, 550 nm, 0.12 µm a pixel), adds noise of σ 15, and deconvolves it. It prints the RMS error against the specimen as the iterations go, plain and with total variation. The error falls, then rises again as Richardson–Lucy fits the noise: plain, from 63.2 to 59.0 at iteration 28; with total variation, to 57.9 at iteration 53. An iteration takes 23 to 29 ms on 16 threads. It does the same for a 192 × 192 × 48 stack. With CUDA, it runs both again on the GPU and checks the two backends agree: 1.2 ms an iteration in 2D and 3.0 ms in 3D, against 25 ms and 94 ms on the CPU. Then it times 20 iterations on a full 5,472 × 3,648 frame: about 350 ms, 17.6 ms each. It writes the specimen, the image and the best estimate to `target/deconvolve`.

The CUDA tests run with `TETERA_CUDA=1 cargo test -p tetera-imaging --features cuda`.
