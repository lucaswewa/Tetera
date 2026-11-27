//! Richardson–Lucy deconvolution (P1.07b; the behaviour spec is in its notes), in 2D and 3D:
//! convolutions by FFTs over each axis in turn, the image padded by mirroring, and
//! total-variation regularisation if asked for. On the CPU through `rustfft`; on CUDA (P1.07c)
//! through cuFFT, with each element-wise step a kernel whose twin is the CPU's step here. The
//! two FFTs round differently, so the backends agree within the behaviour spec's tolerance,
//! not exactly.

use crate::image::Image;
use crate::psf::Psf;
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use tetera_compute::deconvolve::{Padding, reflect};
use tetera_compute::{Compute, CpuBackend};
use tetera_core::{Error, Result};

/// The values a parallel task takes at least: below it, waking threads costs more than they
/// save.
const GRAIN: usize = 32_768;

/// How many items of `len` values a parallel task takes: `GRAIN` values' worth, at least one.
fn per_task(len: usize) -> usize {
    (GRAIN / len).max(1)
}

/// A volume of values, `width × height × depth`, x fastest, each finite and none negative. An
/// image is a volume of depth 1.
#[derive(Debug, Clone, PartialEq)]
pub struct Volume {
    size: [usize; 3],
    values: Vec<f32>,
}

impl Volume {
    /// A volume of `values`, which must fill it.
    pub fn new(width: u32, height: u32, depth: u32, values: Vec<f32>) -> Result<Self> {
        let n = width as usize * height as usize * depth as usize;
        if n == 0 || values.len() != n {
            return Err(Error::invalid_argument(format!(
                "a {width} × {height} × {depth} volume holds {n} values, not {}",
                values.len()
            )));
        }
        if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::invalid_argument(
                "a volume's values must be finite and none negative",
            ));
        }
        Ok(Self {
            size: [width as usize, height as usize, depth as usize],
            values,
        })
    }

    /// A one-channel image, as a volume of depth 1.
    pub fn from_image(image: &Image<u16>) -> Result<Self> {
        if image.channels() != 1 {
            return Err(Error::invalid_argument(format!(
                "deconvolution takes one channel at a time, not {}",
                image.channels()
            )));
        }
        let values = image.data().iter().map(|&v| f32::from(v)).collect();
        Self::new(image.width(), image.height(), 1, values)
    }

    /// A volume of depth 1 as a one-channel image, each value rounded and held to `0..=max`.
    pub fn to_image(&self, max: u16) -> Result<Image<u16>> {
        let [w, h, d] = self.size;
        if d != 1 {
            return Err(Error::invalid_argument(format!(
                "a volume {d} deep isn't an image"
            )));
        }
        let top = f32::from(max);
        let data = self
            .values
            .iter()
            .map(|v| v.round().min(top) as u16)
            .collect();
        Image::new(w as u32, h as u32, 1, data)
    }

    /// Its width, height and depth.
    pub fn size(&self) -> (u32, u32, u32) {
        let [w, h, d] = self.size;
        (w as u32, h as u32, d as u32)
    }

    /// Its values, x fastest, then y, then z.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The sum of its values: its flux.
    pub fn sum(&self) -> f64 {
        self.values.iter().map(|&v| f64::from(v)).sum()
    }
}

/// Richardson–Lucy's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RichardsonLucy {
    /// How many iterations to run.
    pub iterations: u32,
    /// Total variation's weight λ, about 0.002; 0 for plain Richardson–Lucy.
    pub total_variation: f32,
}

impl RichardsonLucy {
    /// Plain Richardson–Lucy, for `iterations`.
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations,
            total_variation: 0.0,
        }
    }

    /// The same, regularised by total variation of weight `lambda`.
    pub fn with_total_variation(self, lambda: f32) -> Self {
        Self {
            total_variation: lambda,
            ..self
        }
    }
}

/// `image` deconvolved by `psf` by Richardson–Lucy, on whichever backend `compute` chose.
pub fn richardson_lucy(
    compute: &Compute,
    image: &Volume,
    psf: &Psf,
    settings: &RichardsonLucy,
) -> Result<Volume> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let mut run = DeviceDeconvolution::new(gpu, image, psf, settings.total_variation)?;
        for _ in 0..settings.iterations {
            run.step()?;
        }
        return run.estimate();
    }
    richardson_lucy_cpu(compute.cpu(), image, psf, settings)
}

/// `image` deconvolved by `psf` on the CPU, by Richardson–Lucy.
pub fn richardson_lucy_cpu(
    cpu: &CpuBackend,
    image: &Volume,
    psf: &Psf,
    settings: &RichardsonLucy,
) -> Result<Volume> {
    let mut run = Deconvolution::new(cpu, image, psf, settings.total_variation)?;
    for _ in 0..settings.iterations {
        run.step();
    }
    Ok(run.estimate())
}

/// `image` convolved with `psf` on the CPU, padded by mirroring as for deconvolution: the
/// image an objective of that PSF would make of it.
pub fn convolve_cpu(cpu: &CpuBackend, image: &Volume, psf: &Psf) -> Result<Volume> {
    cpu.install(|| {
        let mut convolver = Convolver::new(image, psf)?;
        let mut values = convolver.pad(image);
        convolver.blur(&mut values, false);
        // FFTs round a little either side of zero.
        values
            .par_iter_mut()
            .with_min_len(GRAIN)
            .for_each(|v| *v = v.max(0.0));
        Ok(convolver.crop(&values))
    })
}

/// A Richardson–Lucy deconvolution under way, one iteration at a time, on the padded image.
pub struct Deconvolution<'a> {
    cpu: &'a CpuBackend,
    convolver: Convolver,
    /// The image, padded.
    data: Vec<f32>,
    /// The estimate uₖ, padded.
    estimate: Vec<f32>,
    /// Each iteration's blurred estimate, ratio and correction in turn.
    work: Vec<f32>,
    /// Total variation's denominators, when it is on.
    denominators: Vec<f32>,
    /// Below this, a blurred value takes a ratio of 1: 1e-12 of the image's largest value.
    floor: f32,
    lambda: f32,
    /// The least gradient total variation divides by: 1e-6 of the image's largest value.
    epsilon: f32,
    iterations: u32,
}

impl<'a> Deconvolution<'a> {
    /// A deconvolution of `image` by `psf`, its estimate the image itself, regularised by total
    /// variation of weight `lambda` (0 for none).
    pub fn new(cpu: &'a CpuBackend, image: &Volume, psf: &Psf, lambda: f32) -> Result<Self> {
        check_lambda(lambda)?;
        let (convolver, data) = cpu.install(|| {
            let convolver = Convolver::new(image, psf)?;
            let data = convolver.pad(image);
            Ok::<_, Error>((convolver, data))
        })?;
        let (floor, epsilon) = limits(image);
        let n = data.len();
        Ok(Self {
            cpu,
            convolver,
            estimate: data.clone(),
            data,
            work: vec![0.0; n],
            denominators: if lambda > 0.0 {
                vec![1.0; n]
            } else {
                Vec::new()
            },
            floor,
            lambda,
            epsilon,
            iterations: 0,
        })
    }

    /// One iteration: `uₖ₊₁ = uₖ · [(d / (uₖ ⊛ h)) ⊛ h*]`, over total variation's denominator
    /// when it is on, and held at 0 or more.
    pub fn step(&mut self) {
        let cpu = self.cpu;
        cpu.install(|| {
            self.work.copy_from_slice(&self.estimate);
            self.convolver.blur(&mut self.work, false);
            ratio(&mut self.work, &self.data, self.floor);
            self.convolver.blur(&mut self.work, true);
            let denominators = if self.lambda > 0.0 {
                total_variation(
                    &self.estimate,
                    self.convolver.padding.padded,
                    self.lambda,
                    self.epsilon,
                    &mut self.denominators,
                );
                Some(&self.denominators[..])
            } else {
                None
            };
            update(&mut self.estimate, &self.work, denominators);
        });
        self.iterations += 1;
    }

    /// The iterations run so far.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// The estimate, cropped to the image.
    pub fn estimate(&self) -> Volume {
        self.convolver.crop(&self.estimate)
    }

    /// The estimate's flux over the padded domain, which each iteration keeps.
    pub fn flux(&self) -> f64 {
        self.estimate.iter().map(|&v| f64::from(v)).sum()
    }

    /// The padded image's flux, which the estimate's starts at.
    pub fn image_flux(&self) -> f64 {
        self.data.iter().map(|&v| f64::from(v)).sum()
    }
}

/// Total variation's weight, checked: finite, and 0 or more.
fn check_lambda(lambda: f32) -> Result<()> {
    if lambda.is_finite() && lambda >= 0.0 {
        Ok(())
    } else {
        Err(Error::invalid_argument(format!(
            "total variation's weight is 0 or more, not {lambda}"
        )))
    }
}

/// The ratio's floor, 1e-12 of `image`'s largest value, and the least gradient total
/// variation divides by, 1e-6 of it.
fn limits(image: &Volume) -> (f32, f32) {
    let largest = image.values.iter().fold(0.0f32, |a, &v| a.max(v));
    (1e-12 * largest, (1e-6 * largest).max(f32::MIN_POSITIVE))
}

/// The ratio step, in place: each blurred value b becomes d / b where b is above `floor`, and
/// 1 elsewhere, so nothing divides by zero. The GPU's twin is `CudaBackend::rl_ratio`.
fn ratio(work: &mut [f32], data: &[f32], floor: f32) {
    work.par_iter_mut()
        .zip(data)
        .with_min_len(GRAIN)
        .for_each(|(w, &d)| *w = if *w > floor { d / *w } else { 1.0 });
}

/// The update step: each estimate value times its correction, over its denominator when total
/// variation is on, held at 0 or more. The GPU's twin is `CudaBackend::rl_update`.
fn update(estimate: &mut [f32], correction: &[f32], denominators: Option<&[f32]>) {
    match denominators {
        Some(t) => estimate
            .par_iter_mut()
            .zip(correction)
            .zip(t)
            .with_min_len(GRAIN)
            .for_each(|((u, &c), &t)| *u = (*u * c / t).max(0.0)),
        None => estimate
            .par_iter_mut()
            .zip(correction)
            .with_min_len(GRAIN)
            .for_each(|(u, &c)| *u = (*u * c).max(0.0)),
    }
}

/// `buffer` times the PSF's transform `otf`, or its conjugate for `mirrored`, in place. The
/// GPU's twin is `CudaBackend::rl_multiply`, which scales its transform as it goes.
fn multiply(buffer: &mut [Complex32], otf: &[Complex32], mirrored: bool) {
    buffer
        .par_iter_mut()
        .zip(otf)
        .with_min_len(GRAIN)
        .for_each(|(b, h)| *b *= if mirrored { h.conj() } else { *h });
}

/// The PSF's values with their places in the padded volume, its centre at the origin.
fn placed<'p>(psf: &'p Psf, padding: &'p Padding) -> impl Iterator<Item = (usize, f32)> + 'p {
    let (w, h, _) = psf.size();
    let (w, h) = (w as usize, h as usize);
    psf.values()
        .iter()
        .enumerate()
        .map(move |(i, &v)| (padding.centred([i % w, i / w % h, i / (w * h)]), v))
}

/// The padding of `image` for `psf`.
fn padding(image: &Volume, psf: &Psf) -> Result<Padding> {
    let (w, h, d) = psf.size();
    Padding::new(image.size, [w as usize, h as usize, d as usize])
}

/// `1 − λ div(∇u / |∇u|)` at each point of `u`, held at 0.1 or more, into `out`: the gradient
/// by forward differences, the divergence by backward ones, both 0 across the volume's edges.
/// The GPU's twin is `CudaBackend::rl_total_variation`.
fn total_variation(u: &[f32], size: [usize; 3], lambda: f32, epsilon: f32, out: &mut [f32]) {
    let [nx, ny, nz] = size;
    let at = |x: usize, y: usize, z: usize| u[(z * ny + y) * nx + x];
    let unit = |x: usize, y: usize, z: usize| {
        let c = at(x, y, z);
        let gx = if x + 1 < nx { at(x + 1, y, z) - c } else { 0.0 };
        let gy = if y + 1 < ny { at(x, y + 1, z) - c } else { 0.0 };
        let gz = if z + 1 < nz { at(x, y, z + 1) - c } else { 0.0 };
        let norm = (gx * gx + gy * gy + gz * gz).sqrt().max(epsilon);
        [gx / norm, gy / norm, gz / norm]
    };
    out.par_chunks_mut(nx)
        .enumerate()
        .with_min_len(per_task(nx))
        .for_each(|(row, line)| {
            let (y, z) = (row % ny, row / ny);
            for (x, o) in line.iter_mut().enumerate() {
                let n = unit(x, y, z);
                let mut divergence = n[0] + n[1] + n[2];
                if x > 0 {
                    divergence -= unit(x - 1, y, z)[0];
                }
                if y > 0 {
                    divergence -= unit(x, y - 1, z)[1];
                }
                if z > 0 {
                    divergence -= unit(x, y, z - 1)[2];
                }
                *o = (1.0 - lambda * divergence).max(0.1);
            }
        });
}

/// Convolution with one PSF by FFT, over an image padded on every side by the PSF's half-size
/// or a little more, to a size the FFTs factor well.
struct Convolver {
    padding: Padding,
    /// The PSF's transform, centred on the origin, over the padded count to undo the inverse
    /// FFT's scale.
    otf: Vec<Complex32>,
    transform: Transform,
    buffer: Vec<Complex32>,
}

impl Convolver {
    fn new(image: &Volume, psf: &Psf) -> Result<Self> {
        let padding = padding(image, psf)?;
        let mut transform = Transform::new(padding.padded);
        let n = padding.count();
        let mut otf = vec![Complex32::default(); n];
        for (i, v) in placed(psf, &padding) {
            otf[i] = Complex32::new(v, 0.0);
        }
        transform.run(&mut otf, false);
        let scale = 1.0 / n as f32;
        otf.iter_mut().for_each(|h| *h *= scale);
        Ok(Self {
            padding,
            otf,
            transform,
            buffer: vec![Complex32::default(); n],
        })
    }

    /// `image`, padded by mirroring it about its edges. The GPU's twin is
    /// `CudaBackend::rl_pad`.
    fn pad(&self, image: &Volume) -> Vec<f32> {
        let p = &self.padding;
        let ([w, h, d], [pw, ph, _], [ox, oy, oz]) = (p.size, p.padded, p.offset);
        let mut out = vec![0.0; p.count()];
        out.par_chunks_mut(pw)
            .enumerate()
            .with_min_len(per_task(pw))
            .for_each(|(row, line)| {
                let y = reflect((row % ph) as i64 - oy as i64, h);
                let z = reflect((row / ph) as i64 - oz as i64, d);
                let source = &image.values[(z * h + y) * w..][..w];
                for (px, v) in line.iter_mut().enumerate() {
                    *v = source[reflect(px as i64 - ox as i64, w)];
                }
            });
        out
    }

    /// The image's part of padded `values`. The GPU's twin is `CudaBackend::rl_crop`.
    fn crop(&self, values: &[f32]) -> Volume {
        let p = &self.padding;
        let ([w, h, d], [pw, ph, _], [ox, oy, oz]) = (p.size, p.padded, p.offset);
        let mut out = Vec::with_capacity(w * h * d);
        for z in 0..d {
            for y in 0..h {
                out.extend_from_slice(&values[((z + oz) * ph + y + oy) * pw + ox..][..w]);
            }
        }
        Volume {
            size: p.size,
            values: out,
        }
    }

    /// Padded `values`, convolved with the PSF in place, or correlated with it (convolved with
    /// it mirrored) for `mirrored`.
    fn blur(&mut self, values: &mut [f32], mirrored: bool) {
        self.buffer
            .par_iter_mut()
            .zip(&*values)
            .with_min_len(GRAIN)
            .for_each(|(b, &v)| *b = Complex32::new(v, 0.0));
        self.transform.run(&mut self.buffer, false);
        multiply(&mut self.buffer, &self.otf, mirrored);
        self.transform.run(&mut self.buffer, true);
        values
            .par_iter_mut()
            .zip(&self.buffer)
            .with_min_len(GRAIN)
            .for_each(|(v, b)| *v = b.re);
    }
}

/// Complex FFTs over each axis of a volume in turn, unnormalised as cuFFT's are.
struct Transform {
    size: [usize; 3],
    forward: [Arc<dyn Fft<f32>>; 3],
    inverse: [Arc<dyn Fft<f32>>; 3],
    /// Columns gathered for the y and z axes.
    columns: Vec<Complex32>,
}

impl Transform {
    fn new(size: [usize; 3]) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            size,
            forward: size.map(|n| planner.plan_fft_forward(n)),
            inverse: size.map(|n| planner.plan_fft_inverse(n)),
            columns: vec![Complex32::default(); size.iter().product()],
        }
    }

    /// `data` transformed over every axis, forward or back.
    fn run(&mut self, data: &mut [Complex32], inverse: bool) {
        let [nx, ny, nz] = self.size;
        let plans = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        rows(data, &plans[0]);
        // y: within each plane, columns of ny values; z: across the planes, of nz values.
        along(data, &mut self.columns, (nz, ny, nx), &plans[1]);
        along(data, &mut self.columns, (1, nz, nx * ny), &plans[2]);
    }
}

/// Each run of `fft.len()` values in `data` transformed, in parallel.
fn rows(data: &mut [Complex32], fft: &Arc<dyn Fft<f32>>) {
    let n = fft.len();
    if n == 1 {
        return;
    }
    data.par_chunks_mut(per_task(n) * n).for_each_init(
        || vec![Complex32::default(); fft.get_inplace_scratch_len()],
        |scratch, chunk| fft.process_with_scratch(chunk, scratch),
    );
}

/// Each column of `data` transformed: `data` is `blocks` blocks of `rows × cols` values, and
/// every block's columns are gathered into `columns` as rows, transformed, and put back, each
/// in one parallel pass.
fn along(
    data: &mut [Complex32],
    columns: &mut [Complex32],
    (blocks, rows_in_block, cols): (usize, usize, usize),
    fft: &Arc<dyn Fft<f32>>,
) {
    if rows_in_block == 1 {
        return;
    }
    let size = rows_in_block * cols;
    let source: &[Complex32] = data;
    // Column i is block i / cols's column i % cols.
    columns[..blocks * size]
        .par_chunks_mut(rows_in_block)
        .enumerate()
        .with_min_len(per_task(rows_in_block))
        .for_each(|(i, column)| {
            let start = i / cols * size + i % cols;
            for (r, v) in column.iter_mut().enumerate() {
                *v = source[start + r * cols];
            }
        });
    rows(&mut columns[..blocks * size], fft);
    let gathered: &[Complex32] = columns;
    // Row j is block j / rows_in_block's row j % rows_in_block.
    data.par_chunks_mut(cols)
        .enumerate()
        .with_min_len(per_task(cols))
        .for_each(|(j, row)| {
            let start = j / rows_in_block * size + j % rows_in_block;
            for (c, v) in row.iter_mut().enumerate() {
                *v = gathered[start + c * rows_in_block];
            }
        });
}

#[cfg(feature = "cuda")]
pub use device::DeviceDeconvolution;

#[cfg(feature = "cuda")]
mod device {
    use super::{Padding, Psf, Volume, check_lambda, limits, padding, placed};
    use tetera_compute::deconvolve::{Complex, DeviceFft};
    use tetera_compute::{CudaBackend, DeviceBuffer};
    use tetera_core::Result;

    /// A Richardson–Lucy deconvolution under way on the GPU, one iteration at a time: the
    /// CPU's `Deconvolution`, step for step, with cuFFT's transforms.
    #[derive(Debug)]
    pub struct DeviceDeconvolution<'a> {
        gpu: &'a CudaBackend,
        padding: Padding,
        fft: DeviceFft,
        /// The PSF's transform, centred on the origin, unscaled: each product scales it.
        otf: DeviceBuffer<Complex>,
        scale: f32,
        data: DeviceBuffer<f32>,
        estimate: DeviceBuffer<f32>,
        /// The transforms' two buffers, in and out by turns.
        a: DeviceBuffer<Complex>,
        b: DeviceBuffer<Complex>,
        denominators: Option<DeviceBuffer<f32>>,
        floor: f32,
        lambda: f32,
        epsilon: f32,
        iterations: u32,
    }

    impl<'a> DeviceDeconvolution<'a> {
        /// A deconvolution of `image` by `psf` on `gpu`, its estimate the image itself,
        /// regularised by total variation of weight `lambda` (0 for none).
        pub fn new(gpu: &'a CudaBackend, image: &Volume, psf: &Psf, lambda: f32) -> Result<Self> {
            check_lambda(lambda)?;
            let padding = padding(image, psf)?;
            let n = padding.count();
            let fft = gpu.fft_plan(padding.padded)?;
            let mut host = vec![Complex { x: 0.0, y: 0.0 }; n];
            for (i, v) in placed(psf, &padding) {
                host[i].x = v;
            }
            let mut a = gpu.upload(&host)?;
            let mut otf = gpu.zeros(n)?;
            fft.run(&mut a, &mut otf, false)?;
            let on_device = gpu.upload(image.values())?;
            let (mut data, mut estimate) = (gpu.zeros(n)?, gpu.zeros(n)?);
            gpu.rl_pad(&on_device, &padding, &mut data)?;
            gpu.rl_pad(&on_device, &padding, &mut estimate)?;
            let (floor, epsilon) = limits(image);
            Ok(Self {
                gpu,
                padding,
                fft,
                otf,
                scale: 1.0 / n as f32,
                data,
                estimate,
                a,
                b: gpu.zeros(n)?,
                denominators: if lambda > 0.0 {
                    Some(gpu.zeros(n)?)
                } else {
                    None
                },
                floor,
                lambda,
                epsilon,
                iterations: 0,
            })
        }

        /// `a` convolved with the PSF, or correlated with it for `mirrored`, back into `a`.
        fn blur(&mut self, mirrored: bool) -> Result<()> {
            self.fft.run(&mut self.a, &mut self.b, false)?;
            self.gpu
                .rl_multiply(&mut self.b, &self.otf, self.scale, mirrored)?;
            self.fft.run(&mut self.b, &mut self.a, true)
        }

        /// One iteration, as the CPU's `Deconvolution::step`.
        pub fn step(&mut self) -> Result<()> {
            let gpu = self.gpu;
            gpu.rl_to_complex(&self.estimate, &mut self.a)?;
            self.blur(false)?;
            gpu.rl_ratio(&mut self.a, &self.data, self.floor)?;
            self.blur(true)?;
            if let Some(t) = &mut self.denominators {
                gpu.rl_total_variation(
                    &self.estimate,
                    self.padding.padded,
                    self.lambda,
                    self.epsilon,
                    t,
                )?;
            }
            gpu.rl_update(&mut self.estimate, &self.a, self.denominators.as_ref())?;
            self.iterations += 1;
            Ok(())
        }

        /// The iterations run so far.
        pub fn iterations(&self) -> u32 {
            self.iterations
        }

        /// The estimate, cropped to the image, copied back.
        pub fn estimate(&self) -> Result<Volume> {
            let mut out = self.gpu.zeros(self.padding.image_count())?;
            self.gpu.rl_crop(&self.estimate, &self.padding, &mut out)?;
            Ok(Volume {
                size: self.padding.size,
                values: self.gpu.download(&out)?,
            })
        }

        /// The estimate's flux over the padded domain, which each iteration keeps.
        pub fn flux(&self) -> Result<f64> {
            let values = self.gpu.download(&self.estimate)?;
            Ok(values.iter().map(|&v| f64::from(v)).sum())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psf::Optics;

    fn cpu() -> CpuBackend {
        CpuBackend::new(4).unwrap()
    }

    fn psf_2d() -> Psf {
        // NA 0.5 at 0.2 µm a pixel: 25 × 25.
        Psf::widefield_2d(&Optics {
            na: 0.5,
            wavelength_um: 0.55,
            pixel_um: 0.2,
            refractive_index: 1.0,
        })
        .unwrap()
    }

    fn volume(w: u32, h: u32, d: u32, f: impl Fn(u32, u32, u32) -> f32) -> Volume {
        let values = (0..d)
            .flat_map(|z| (0..h).flat_map(move |y| (0..w).map(move |x| (x, y, z))))
            .map(|(x, y, z)| f(x, y, z))
            .collect();
        Volume::new(w, h, d, values).unwrap()
    }

    fn rms(a: &Volume, b: &Volume) -> f64 {
        let squares: f64 = a
            .values()
            .iter()
            .zip(b.values())
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        (squares / a.values().len() as f64).sqrt()
    }

    #[test]
    fn the_transform_matches_a_plain_dft_and_inverts() {
        let size = [5, 3, 4];
        let n: usize = size.iter().product();
        let input: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i * 7 % 11) as f32, (i * 3 % 5) as f32 - 2.0))
            .collect();
        let mut data = input.clone();
        let mut transform = Transform::new(size);
        transform.run(&mut data, false);
        for (k, got) in data.iter().enumerate() {
            let kk = [k % 5, k / 5 % 3, k / 15];
            let mut sum = rustfft::num_complex::Complex64::default();
            for (j, v) in input.iter().enumerate() {
                let jj = [j % 5, j / 5 % 3, j / 15];
                let phase: f64 = (0..3)
                    .map(|a| (kk[a] * jj[a]) as f64 / size[a] as f64)
                    .sum::<f64>()
                    * -std::f64::consts::TAU;
                sum += rustfft::num_complex::Complex64::new(f64::from(v.re), f64::from(v.im))
                    * rustfft::num_complex::Complex64::from_polar(1.0, phase);
            }
            assert!(
                (f64::from(got.re) - sum.re).abs() < 1e-3
                    && (f64::from(got.im) - sum.im).abs() < 1e-3,
                "bin {kk:?}: {got} against {sum}"
            );
        }
        transform.run(&mut data, true);
        for (a, b) in data.iter().zip(&input) {
            assert!((a / n as f32 - b).norm() < 1e-4);
        }
    }

    #[test]
    fn a_point_convolves_to_the_psf_and_a_flat_image_stays_flat() {
        let psf = psf_2d();
        let point = volume(
            41,
            37,
            1,
            |x, y, _| if (x, y) == (20, 18) { 1000.0 } else { 0.0 },
        );
        let blurred = convolve_cpu(&cpu(), &point, &psf).unwrap();
        for y in 0..25 {
            for x in 0..25 {
                let got = blurred.values()[((y + 6) * 41 + x + 8) as usize];
                assert!((got - 1000.0 * psf.at(x, y, 0)).abs() < 1e-3, "({x}, {y})");
            }
        }
        assert!((blurred.sum() - 1000.0).abs() < 1e-2, "{}", blurred.sum());
        let flat = volume(30, 20, 1, |_, _, _| 500.0);
        let same = convolve_cpu(&cpu(), &flat, &psf).unwrap();
        assert!(same.values().iter().all(|v| (v - 500.0).abs() < 1e-2));
        // At an edge, the mirror's light makes up for the light lost beyond it.
        let ramp = volume(30, 20, 1, |x, _, _| 10.0 * x as f32);
        let blurred_ramp = convolve_cpu(&cpu(), &ramp, &psf).unwrap();
        assert!(
            (blurred_ramp.values()[15] - 150.0).abs() < 1e-2,
            "a ramp's middle stays"
        );
    }

    #[test]
    fn richardson_lucy_keeps_the_flux_over_its_domain() {
        let psf = psf_2d();
        let scene = volume(48, 40, 1, |x, y, _| {
            ((x * 37 + y * 11) % 23) as f32 * 40.0 + 10.0
        });
        let image = convolve_cpu(&cpu(), &scene, &psf).unwrap();
        let backend = cpu();
        let mut run = Deconvolution::new(&backend, &image, &psf, 0.0).unwrap();
        let start = run.image_flux();
        assert_eq!(run.flux(), start);
        for _ in 0..10 {
            run.step();
            let drift = (run.flux() - start).abs() / start;
            // Measured: 3.4e-8 after 10 iterations.
            assert!(
                drift < 1e-6,
                "iteration {}: drift {drift}",
                run.iterations()
            );
        }
        assert_eq!(run.iterations(), 10);
    }

    #[test]
    fn a_blurred_point_sharpens_each_iteration() {
        let psf = psf_2d();
        let point = volume(
            40,
            40,
            1,
            |x, y, _| if (x, y) == (20, 20) { 4000.0 } else { 0.0 },
        );
        let image = convolve_cpu(&cpu(), &point, &psf).unwrap();
        let backend = cpu();
        let mut run = Deconvolution::new(&backend, &image, &psf, 0.0).unwrap();
        let peak = |run: &Deconvolution| run.estimate().values()[20 * 40 + 20];
        let mut last = peak(&run);
        for _ in 0..15 {
            run.step();
            let now = peak(&run);
            assert!(
                now > last,
                "iteration {}: {now} after {last}",
                run.iterations()
            );
            last = now;
        }
        // Measured: the centre from 411 of the point's 4,000 to 2,397 after 15 iterations.
        assert!(last > 5.0 * image.values()[20 * 40 + 20], "{last}");
    }

    #[test]
    fn richardson_lucy_brings_a_blurred_noisy_image_closer_to_the_truth() {
        let psf = psf_2d();
        // Discs and points on a dim background.
        let truth = volume(64, 64, 1, |x, y, _| {
            let (fx, fy) = (x as f32, y as f32);
            let disc = |cx: f32, cy: f32, r: f32| (fx - cx).hypot(fy - cy) <= r;
            if disc(20.0, 22.0, 6.0) || (x, y) == (45, 15) || (x, y) == (48, 18) {
                3000.0
            } else if disc(40.0, 44.0, 9.0) {
                1500.0
            } else {
                100.0
            }
        });
        let blurred = convolve_cpu(&cpu(), &truth, &psf).unwrap();
        let rounded: Vec<u16> = blurred.values().iter().map(|v| v.round() as u16).collect();
        let noisy = crate::pattern::noisy(&rounded, 20.0, 4_095, 7);
        let image = Volume::new(64, 64, 1, noisy.iter().map(|&v| f32::from(v)).collect()).unwrap();
        let before = rms(&image, &truth);
        let after = rms(
            &richardson_lucy_cpu(&cpu(), &image, &psf, &RichardsonLucy::new(20)).unwrap(),
            &truth,
        );
        // Measured: an RMS error of 192.3 before, and 143.0 after 20 iterations.
        assert!(after < 0.8 * before, "{before} to {after}");
    }

    #[test]
    fn a_flat_image_stays_flat() {
        let flat = volume(32, 24, 1, |_, _, _| 800.0);
        let out = richardson_lucy_cpu(&cpu(), &flat, &psf_2d(), &RichardsonLucy::new(5)).unwrap();
        assert!(
            out.values().iter().all(|v| (v - 800.0).abs() < 0.05),
            "{:?}",
            &out.values()[..4]
        );
        let dark = volume(16, 16, 1, |_, _, _| 0.0);
        let backend = cpu();
        let mut run = Deconvolution::new(&backend, &dark, &psf_2d(), 0.0).unwrap();
        run.step();
        assert!(
            run.work.iter().all(|v| v.is_finite()),
            "where nothing is blurred in, the ratio is 1, not 0 / 0"
        );
        let out = richardson_lucy_cpu(&cpu(), &dark, &psf_2d(), &RichardsonLucy::new(3)).unwrap();
        assert!(
            out.values().iter().all(|&v| v == 0.0),
            "an empty image stays empty"
        );
    }

    #[test]
    fn three_dimensional_deconvolution_sharpens_a_point_in_every_axis() {
        let optics = Optics {
            na: 0.5,
            wavelength_um: 0.55,
            pixel_um: 0.2,
            refractive_index: 1.0,
        };
        let psf = Psf::widefield_3d(&optics, 0.4, 9).unwrap();
        let point = volume(32, 32, 16, |x, y, z| {
            if (x, y, z) == (16, 16, 8) {
                10_000.0
            } else {
                0.0
            }
        });
        let image = convolve_cpu(&cpu(), &point, &psf).unwrap();
        assert!((image.sum() - 10_000.0).abs() < 0.5, "{}", image.sum());
        let at = |v: &Volume, x: usize, y: usize, z: usize| v.values()[(z * 32 + y) * 32 + x];
        let backend = cpu();
        let mut run = Deconvolution::new(&backend, &image, &psf, 0.0).unwrap();
        let start = run.image_flux();
        for _ in 0..10 {
            run.step();
        }
        assert!((run.flux() - start).abs() / start < 1e-6);
        let out = run.estimate();
        // Measured: the centre from 114 to 2,331, and its ratio to two planes off from 1.12
        // to 19.9.
        assert!(at(&out, 16, 16, 8) > 5.0 * at(&image, 16, 16, 8));
        // Sharper along z as well: the centre gains on the planes either side.
        let ratio = |v: &Volume| at(v, 16, 16, 8) / at(v, 16, 16, 10);
        assert!(
            ratio(&out) > 2.0 * ratio(&image),
            "{} against {}",
            ratio(&out),
            ratio(&image)
        );
        // A 2D PSF on a volume deconvolves each plane alone.
        assert!(richardson_lucy_cpu(&cpu(), &image, &psf_2d(), &RichardsonLucy::new(1)).is_ok());
    }

    #[test]
    fn total_variation_smooths_noise_in_a_flat_region() {
        let rounded = vec![1000u16; 48 * 48];
        let noisy = crate::pattern::noisy(&rounded, 30.0, 4_095, 3);
        let image = Volume::new(48, 48, 1, noisy.iter().map(|&v| f32::from(v)).collect()).unwrap();
        let spread = |v: &Volume| {
            let mean = v.sum() / v.values().len() as f64;
            let flat = volume(48, 48, 1, |_, _, _| mean as f32);
            rms(v, &flat)
        };
        let plain =
            richardson_lucy_cpu(&cpu(), &image, &psf_2d(), &RichardsonLucy::new(15)).unwrap();
        let settings = RichardsonLucy::new(15).with_total_variation(0.01);
        let smoothed = richardson_lucy_cpu(&cpu(), &image, &psf_2d(), &settings).unwrap();
        // Measured: noise of 30.2 became 47.4 plain, and 19.6 with total variation.
        assert!(
            spread(&smoothed) < 0.6 * spread(&plain),
            "{} against {}",
            spread(&smoothed),
            spread(&plain)
        );
        // And it keeps the region's level, which total variation needn't do exactly.
        let mean = |v: &Volume| v.sum() / v.values().len() as f64;
        assert!(
            (mean(&smoothed) / mean(&image) - 1.0).abs() < 0.01,
            "{} against {}",
            mean(&smoothed),
            mean(&image)
        );
    }

    #[test]
    fn the_floor_and_epsilon_scale_with_the_largest_value() {
        let image = volume(3, 1, 1, |x, _, _| [10.0, 4_000.0, 0.0][x as usize]);
        assert_eq!(limits(&image), (1e-12 * 4_000.0, 1e-6 * 4_000.0));
        let dark = volume(2, 2, 1, |_, _, _| 0.0);
        assert_eq!(
            limits(&dark),
            (0.0, f32::MIN_POSITIVE),
            "epsilon stays above 0"
        );
    }

    #[test]
    fn total_variation_is_one_where_the_estimate_is_flat_and_held_at_a_tenth() {
        let flat = vec![5.0; 4 * 3 * 2];
        let mut out = vec![0.0; flat.len()];
        total_variation(&flat, [4, 3, 2], 0.5, 1e-6, &mut out);
        assert!(out.iter().all(|&v| v == 1.0));
        // A single bright point: its unit gradient is −1/√3 on each axis, and each neighbour
        // before it points at it with 1, so its divergence is −(3 + √3).
        let mut point = vec![0.0; 27];
        point[13] = 1.0;
        let mut out = vec![0.0; 27];
        total_variation(&point, [3, 3, 3], 0.1, 1e-6, &mut out);
        let expected = 1.0 + 0.1 * (3.0 + 3f32.sqrt());
        assert!((out[13] - expected).abs() < 1e-6, "{}", out[13]);
        total_variation(&point, [3, 3, 3], 10.0, 1e-6, &mut out);
        assert!(out.iter().all(|&v| v >= 0.1) && out.contains(&0.1));
    }

    #[test]
    fn bad_volumes_and_settings_are_refused() {
        assert!(Volume::new(2, 2, 1, vec![0.0; 3]).is_err());
        assert!(Volume::new(0, 2, 1, vec![]).is_err());
        assert!(Volume::new(1, 1, 1, vec![-1.0]).is_err());
        assert!(Volume::new(1, 1, 1, vec![f32::INFINITY]).is_err());
        let rgb = Image::new(2, 2, 3, vec![0u16; 12]).unwrap();
        assert!(Volume::from_image(&rgb).is_err());
        let grey = Image::new(3, 2, 1, vec![0u16, 1, 2, 4_000, 5_000, 6]).unwrap();
        let v = Volume::from_image(&grey).unwrap();
        assert_eq!(v.size(), (3, 2, 1));
        assert_eq!(
            v.to_image(4_095).unwrap().data(),
            [0, 1, 2, 4_000, 4_095, 6]
        );
        assert!(volume(2, 2, 3, |_, _, _| 0.0).to_image(255).is_err());
        let backend = cpu();
        let image = volume(16, 16, 1, |_, _, _| 1.0);
        assert!(Deconvolution::new(&backend, &image, &psf_2d(), -0.1).is_err());
        assert!(Deconvolution::new(&backend, &image, &psf_2d(), f32::NAN).is_err());
        let psf_3d = Psf::new(3, 3, 3, vec![1.0; 27]).unwrap();
        assert_eq!(
            richardson_lucy_cpu(&backend, &image, &psf_3d, &RichardsonLucy::new(1))
                .unwrap_err()
                .to_string(),
            "a 3D PSF deconvolves a volume, not an image"
        );
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use tetera_compute::deconvolve::Complex;

    /// Seeded values in 0..1, by xorshift.
    fn uniform(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    fn scaled(n: usize, seed: u64, scale: f32, offset: f32) -> Vec<f32> {
        uniform(n, seed)
            .iter()
            .map(|v| v * scale + offset)
            .collect()
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn complex(values: &[Complex32]) -> Vec<Complex> {
        values
            .iter()
            .map(|c| Complex { x: c.re, y: c.im })
            .collect()
    }

    /// A PSF of random values, so no symmetry hides a mirror image.
    fn random_psf(size: [u32; 3], seed: u64) -> Psf {
        let n = (size[0] * size[1] * size[2]) as usize;
        Psf::new(size[0], size[1], size[2], scaled(n, seed, 1.0, 0.01)).unwrap()
    }

    #[test]
    fn padding_and_cropping_match_the_cpu_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        for (seed, (size, psf)) in [
            ([7, 5, 1], [9, 3, 1]),
            ([13, 11, 1], [25, 25, 1]),
            ([6, 5, 4], [3, 5, 7]),
            ([1, 1, 1], [1, 1, 1]),
            ([1, 8, 3], [5, 3, 3]),
        ]
        .into_iter()
        .enumerate()
        {
            let n = (size[0] * size[1] * size[2]) as usize;
            let image = Volume::new(
                size[0],
                size[1],
                size[2],
                scaled(n, seed as u64, 4_095.0, 0.0),
            )
            .unwrap();
            let convolver = Convolver::new(&image, &random_psf(psf, 9)).unwrap();
            let padded = convolver.pad(&image);
            let padding = convolver.padding;
            let mut out = gpu.zeros(padding.count()).unwrap();
            gpu.rl_pad(&gpu.upload(image.values()).unwrap(), &padding, &mut out)
                .unwrap();
            assert!(
                same_bits(&gpu.download(&out).unwrap(), &padded),
                "{size:?}, {psf:?}"
            );
            let mut cropped = gpu.zeros(n).unwrap();
            gpu.rl_crop(&out, &padding, &mut cropped).unwrap();
            assert_eq!(
                gpu.download(&cropped).unwrap(),
                convolver.crop(&padded).values
            );
            assert_eq!(convolver.crop(&padded), image);
        }
    }

    #[test]
    fn the_element_wise_kernels_match_their_cpu_twins_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        let n = 10_007;
        // The ratio, some blurred values at the floor or below it.
        let floor = 1e-3;
        let data = scaled(n, 1, 4_095.0, 0.0);
        let mut blurred = scaled(n, 2, 4_000.0, -2.0);
        blurred[..4].copy_from_slice(&[floor, 0.0, -0.5, 2.0 * floor]);
        let mut on_device = gpu
            .upload(
                &blurred
                    .iter()
                    .map(|&x| Complex { x, y: 0.7 })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        gpu.rl_ratio(&mut on_device, &gpu.upload(&data).unwrap(), floor)
            .unwrap();
        ratio(&mut blurred, &data, floor);
        let got = gpu.download(&on_device).unwrap();
        assert!(same_bits(
            &got.iter().map(|c| c.x).collect::<Vec<_>>(),
            &blurred
        ));
        assert!(got.iter().all(|c| c.y == 0.0), "imaginary parts cleared");
        assert_eq!(&blurred[..3], [1.0, 1.0, 1.0], "at the floor or below, 1");
        // The update, plain and over denominators, some corrections negative.
        let estimate = scaled(n, 3, 1_000.0, 0.0);
        let correction = scaled(n, 4, 2.0, -0.1);
        let denominators = scaled(n, 5, 1.0, 0.1);
        let as_complex: Vec<Complex> = correction.iter().map(|&x| Complex { x, y: 3.0 }).collect();
        for tv in [None, Some(&denominators[..])] {
            let mut cpu = estimate.clone();
            update(&mut cpu, &correction, tv);
            let mut on_device = gpu.upload(&estimate).unwrap();
            let t = tv.map(|t| gpu.upload(t).unwrap());
            gpu.rl_update(
                &mut on_device,
                &gpu.upload(&as_complex).unwrap(),
                t.as_ref(),
            )
            .unwrap();
            assert!(
                same_bits(&gpu.download(&on_device).unwrap(), &cpu),
                "{}",
                tv.is_some()
            );
            assert!(cpu.contains(&0.0), "some held at 0");
        }
        // The product with the PSF's transform, scaled, and with its conjugate.
        let buffer: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(data[i] - 2_000.0, estimate[i] - 500.0))
            .collect();
        let otf: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(correction[i], denominators[i] - 0.5))
            .collect();
        let scale = 1.0 / 12_345.0;
        let scaled_otf: Vec<Complex32> = otf.iter().map(|h| h * scale).collect();
        for mirrored in [false, true] {
            let mut cpu = buffer.clone();
            multiply(&mut cpu, &scaled_otf, mirrored);
            let mut on_device = gpu.upload(&complex(&buffer)).unwrap();
            gpu.rl_multiply(
                &mut on_device,
                &gpu.upload(&complex(&otf)).unwrap(),
                scale,
                mirrored,
            )
            .unwrap();
            let got = gpu.download(&on_device).unwrap();
            let parts = |v: &[Complex]| v.iter().flat_map(|c| [c.x, c.y]).collect::<Vec<_>>();
            assert!(
                same_bits(&parts(&got), &parts(&complex(&cpu))),
                "mirrored: {mirrored}"
            );
        }
        // Real values as complex ones.
        let mut out = gpu.zeros(n).unwrap();
        gpu.rl_to_complex(&gpu.upload(&data).unwrap(), &mut out)
            .unwrap();
        let got = gpu.download(&out).unwrap();
        assert!(got.iter().zip(&data).all(|(c, &d)| c.x == d && c.y == 0.0));
    }

    #[test]
    fn total_variation_matches_its_cpu_twin_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        for (seed, (size, lambda)) in [
            ([17, 9, 1], 0.002),
            ([8, 7, 5], 0.05),
            ([1, 1, 1], 0.002),
            ([33, 1, 2], 5.0),
        ]
        .into_iter()
        .enumerate()
        {
            let n: usize = size.iter().product();
            let mut u = scaled(n, seed as u64 + 20, 1_000.0, 0.0);
            // A flat patch, where the gradient is 0 and its length is held at epsilon.
            for v in u.iter_mut().take(n / 3) {
                *v = 250.0;
            }
            let epsilon = 1e-3;
            let mut cpu = vec![0.0; n];
            total_variation(&u, size, lambda, epsilon, &mut cpu);
            let mut out = gpu.zeros(n).unwrap();
            gpu.rl_total_variation(&gpu.upload(&u).unwrap(), size, lambda, epsilon, &mut out)
                .unwrap();
            assert!(same_bits(&gpu.download(&out).unwrap(), &cpu), "{size:?}");
            if lambda > 1.0 {
                assert!(cpu.contains(&0.1), "held at 0.1");
            }
        }
    }

    #[test]
    fn cuffts_transforms_match_the_cpus() {
        let gpu = tetera_compute::cuda_or_skip!();
        for (seed, size) in [
            [16, 12, 1],
            [30, 7, 1],
            [10, 6, 4],
            [1, 1, 1],
            [1, 5, 1],
            [9, 1, 3],
        ]
        .into_iter()
        .enumerate()
        {
            let n: usize = size.iter().product();
            let input: Vec<Complex32> = scaled(n, seed as u64, 2.0, -1.0)
                .iter()
                .zip(scaled(n, seed as u64 + 50, 2.0, -1.0))
                .map(|(&re, im)| Complex32::new(re, im))
                .collect();
            let fft = gpu.fft_plan(size).unwrap();
            assert_eq!(fft.count(), n);
            let mut cpu = input.clone();
            let mut transform = Transform::new(size);
            let mut a = gpu.upload(&complex(&input)).unwrap();
            let mut b = gpu.zeros(n).unwrap();
            for inverse in [false, true] {
                transform.run(&mut cpu, inverse);
                if inverse {
                    fft.run(&mut b, &mut a, true).unwrap();
                } else {
                    fft.run(&mut a, &mut b, false).unwrap();
                }
                let got = gpu.download(if inverse { &a } else { &b }).unwrap();
                let largest = cpu.iter().map(|c| c.norm()).fold(0.0, f32::max);
                for (g, c) in got.iter().zip(&cpu) {
                    let error = (Complex32::new(g.x, g.y) - c).norm();
                    assert!(
                        error <= 1e-5 * largest,
                        "{size:?}, inverse {inverse}: {error}"
                    );
                }
            }
        }
    }

    /// The relative RMS difference of `a` from `b`, and the largest difference.
    fn differences(a: &Volume, b: &Volume) -> (f64, f64) {
        let (mut squares, mut norm, mut largest) = (0.0, 0.0, 0.0f64);
        for (&x, &y) in a.values().iter().zip(b.values()) {
            let d = f64::from(x) - f64::from(y);
            squares += d * d;
            norm += f64::from(y) * f64::from(y);
            largest = largest.max(d.abs());
        }
        ((squares / norm).sqrt(), largest)
    }

    #[test]
    fn richardson_lucy_on_the_gpu_meets_the_specs_tolerance() {
        let gpu = tetera_compute::cuda_or_skip!();
        let cpu = CpuBackend::new(0).unwrap();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        for (seed, (size, psf_size, lambda)) in [
            ([64, 48, 1], [7, 9, 1], 0.0),
            ([50, 41, 1], [15, 11, 1], 0.002),
            ([24, 20, 10], [5, 7, 5], 0.0),
            ([18, 22, 8], [7, 5, 3], 0.002),
        ]
        .into_iter()
        .enumerate()
        {
            let seed = seed as u64;
            let psf = random_psf(psf_size, seed + 100);
            // A 12-bit specimen of bright points on a dim background, blurred, with noise.
            let n: usize = size.iter().product();
            let points: Vec<f32> = uniform(n, seed + 200)
                .iter()
                .map(|&v| if v > 0.97 { 3_000.0 } else { 80.0 })
                .collect();
            let [w, h, d] = size.map(|v| v as u32);
            let specimen = Volume::new(w, h, d, points).unwrap();
            let blurred = convolve_cpu(&cpu, &specimen, &psf).unwrap();
            let rounded: Vec<u16> = blurred.values().iter().map(|v| v.round() as u16).collect();
            let noisy = crate::pattern::noisy(&rounded, 10.0, 4_095, seed);
            let image =
                Volume::new(w, h, d, noisy.iter().map(|&v| f32::from(v)).collect()).unwrap();
            let largest = image.values().iter().fold(0.0f32, |a, &v| a.max(v));
            let settings = RichardsonLucy::new(20).with_total_variation(lambda);
            let on_cpu = richardson_lucy_cpu(&cpu, &image, &psf, &settings).unwrap();
            let mut run = DeviceDeconvolution::new(&gpu, &image, &psf, lambda).unwrap();
            let start = run.flux().unwrap();
            let cpu_start = Deconvolution::new(&cpu, &image, &psf, lambda)
                .unwrap()
                .image_flux();
            assert_eq!(start, cpu_start, "the same padded image, summed alike");
            for _ in 0..20 {
                run.step().unwrap();
            }
            assert_eq!(run.iterations(), 20);
            if lambda == 0.0 {
                let drift = (run.flux().unwrap() - start).abs() / start;
                assert!(drift < 1e-5, "{size:?}: the flux drifted by {drift}");
            }
            let on_gpu = run.estimate().unwrap();
            let (relative, worst) = differences(&on_gpu, &on_cpu);
            assert!(
                relative < 1e-3 && worst < 0.01 * f64::from(largest),
                "{size:?}, λ {lambda}: relative RMS {relative:e}, worst {worst} of {largest}"
            );
            // The dispatch takes the GPU when there is one.
            assert_eq!(
                richardson_lucy(&compute, &image, &psf, &settings).unwrap(),
                on_gpu
            );
        }
    }

    #[test]
    fn the_device_refuses_what_the_cpu_refuses() {
        let gpu = tetera_compute::cuda_or_skip!();
        let image = Volume::new(4, 4, 1, vec![1.0; 16]).unwrap();
        let psf_3d = random_psf([3, 3, 3], 1);
        assert_eq!(
            DeviceDeconvolution::new(&gpu, &image, &psf_3d, 0.0)
                .unwrap_err()
                .to_string(),
            "a 3D PSF deconvolves a volume, not an image"
        );
        assert!(DeviceDeconvolution::new(&gpu, &image, &random_psf([3, 3, 1], 1), -1.0).is_err());
    }
}
