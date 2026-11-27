//! Richardson–Lucy deconvolution on the CPU (P1.07b; the behaviour spec is in its notes), in
//! 2D and 3D: convolutions by FFTs over each axis in turn, the image padded by mirroring, and
//! total-variation regularisation if asked for. P1.07c adds the GPU's twin.

use crate::image::Image;
use crate::psf::Psf;
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
use tetera_compute::CpuBackend;
use tetera_core::{Error, Result};

/// The values a parallel task takes at least: below it, waking threads costs more than they
/// save.
const GRAIN: usize = 1 << 15;

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
        let mut convolver = Convolver::new(image.size, psf)?;
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
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(Error::invalid_argument(format!(
                "total variation's weight is 0 or more, not {lambda}"
            )));
        }
        let (convolver, data) = cpu.install(|| {
            let convolver = Convolver::new(image.size, psf)?;
            let data = convolver.pad(image);
            Ok::<_, Error>((convolver, data))
        })?;
        let largest = image.values.iter().fold(0.0f32, |a, &v| a.max(v));
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
            floor: 1e-12 * largest,
            lambda,
            epsilon: (1e-6 * largest).max(f32::MIN_POSITIVE),
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
            let floor = self.floor;
            self.work
                .par_iter_mut()
                .zip(&self.data)
                .with_min_len(GRAIN)
                .for_each(|(w, &d)| *w = if *w > floor { d / *w } else { 1.0 });
            self.convolver.blur(&mut self.work, true);
            if self.lambda > 0.0 {
                total_variation(
                    &self.estimate,
                    self.convolver.padded,
                    self.lambda,
                    self.epsilon,
                    &mut self.denominators,
                );
                self.estimate
                    .par_iter_mut()
                    .zip(&self.work)
                    .zip(&self.denominators)
                    .with_min_len(GRAIN)
                    .for_each(|((u, &c), &t)| *u = (*u * c / t).max(0.0));
            } else {
                self.estimate
                    .par_iter_mut()
                    .zip(&self.work)
                    .with_min_len(GRAIN)
                    .for_each(|(u, &c)| *u = (*u * c).max(0.0));
            }
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

/// `1 − λ div(∇u / |∇u|)` at each point of `u`, held at 0.1 or more, into `out`: the gradient
/// by forward differences, the divergence by backward ones, both 0 across the volume's edges.
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
    let rows = (GRAIN / nx).max(1);
    out.par_chunks_mut(nx)
        .enumerate()
        .with_min_len(rows)
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
    /// The image's size.
    size: [usize; 3],
    /// The padded size.
    padded: [usize; 3],
    /// Where the image's first pixel lies in the padded volume: the PSF's half-size.
    offset: [usize; 3],
    /// The PSF's transform, centred on the origin, over the padded count to undo the inverse
    /// FFT's scale.
    otf: Vec<Complex32>,
    transform: Transform,
    buffer: Vec<Complex32>,
}

impl Convolver {
    fn new(size: [usize; 3], psf: &Psf) -> Result<Self> {
        let (pw, ph, pd) = psf.size();
        let psf_size = [pw as usize, ph as usize, pd as usize];
        if size[2] == 1 && psf_size[2] > 1 {
            return Err(Error::invalid_argument(
                "a 3D PSF deconvolves a volume, not an image",
            ));
        }
        let offset = psf_size.map(|n| n / 2);
        let padded = [0, 1, 2].map(|a| smooth(size[a] + 2 * offset[a]));
        let mut transform = Transform::new(padded);
        let n: usize = padded.iter().product();
        let mut otf = vec![Complex32::default(); n];
        for (i, &v) in psf.values().iter().enumerate() {
            let at = [
                i % psf_size[0],
                i / psf_size[0] % psf_size[1],
                i / (psf_size[0] * psf_size[1]),
            ];
            let [x, y, z] = [0, 1, 2].map(|a| (at[a] + padded[a] - offset[a]) % padded[a]);
            otf[(z * padded[1] + y) * padded[0] + x] = Complex32::new(v, 0.0);
        }
        transform.run(&mut otf, false);
        let scale = 1.0 / n as f32;
        otf.iter_mut().for_each(|h| *h *= scale);
        Ok(Self {
            size,
            padded,
            offset,
            otf,
            transform,
            buffer: vec![Complex32::default(); n],
        })
    }

    /// `image`, padded by mirroring it about its edges.
    fn pad(&self, image: &Volume) -> Vec<f32> {
        let ([w, h, d], [pw, ph, _], [ox, oy, oz]) = (self.size, self.padded, self.offset);
        let mut out = vec![0.0; self.padded.iter().product()];
        let rows = (GRAIN / pw).max(1);
        out.par_chunks_mut(pw)
            .enumerate()
            .with_min_len(rows)
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

    /// The image's part of padded `values`.
    fn crop(&self, values: &[f32]) -> Volume {
        let ([w, h, d], [pw, ph, _], [ox, oy, oz]) = (self.size, self.padded, self.offset);
        let mut out = Vec::with_capacity(w * h * d);
        for z in 0..d {
            for y in 0..h {
                out.extend_from_slice(&values[((z + oz) * ph + y + oy) * pw + ox..][..w]);
            }
        }
        Volume {
            size: self.size,
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
        self.buffer
            .par_iter_mut()
            .zip(&self.otf)
            .with_min_len(GRAIN)
            .for_each(|(b, h)| *b *= if mirrored { h.conj() } else { *h });
        self.transform.run(&mut self.buffer, true);
        values
            .par_iter_mut()
            .zip(&self.buffer)
            .with_min_len(GRAIN)
            .for_each(|(v, b)| *v = b.re);
    }
}

/// Index `i` reflected into `0..n` about its ends without repeating them, as often as it takes:
/// …, 2, 1, 0, 1, 2, …, n − 2, n − 1, n − 2, ….
fn reflect(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as i64 - 1);
    let i = i.rem_euclid(period);
    (if i < n as i64 { i } else { period - i }) as usize
}

/// The least size from `n` up whose only prime factors are 2, 3, 5 and 7, which FFTs, cuFFT's
/// among them, transform fastest.
fn smooth(n: usize) -> usize {
    let mut m = n;
    loop {
        let mut rest = m;
        for p in [2, 3, 5, 7] {
            while rest.is_multiple_of(p) {
                rest /= p;
            }
        }
        if rest == 1 {
            return m;
        }
        m += 1;
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
    let per_task = (GRAIN / n).max(1) * n;
    data.par_chunks_mut(per_task).for_each_init(
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
        .with_min_len((GRAIN / rows_in_block).max(1))
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
        .with_min_len((GRAIN / cols).max(1))
        .for_each(|(j, row)| {
            let start = j / rows_in_block * size + j % rows_in_block;
            for (c, v) in row.iter_mut().enumerate() {
                *v = gathered[start + c * rows_in_block];
            }
        });
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
    fn reflection_mirrors_without_repeating_the_edge() {
        let seen: Vec<usize> = (-5..9).map(|i| reflect(i, 4)).collect();
        assert_eq!(seen, [1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2]);
        assert_eq!(reflect(-7, 1), 0);
        assert_eq!(reflect(3, 1), 0);
        assert_eq!(reflect(-1, 2), 1);
    }

    #[test]
    fn padded_sizes_factor_into_2_3_5_and_7() {
        let sizes: Vec<usize> = [1, 11, 13, 17, 23, 29, 31, 97, 121].map(smooth).to_vec();
        assert_eq!(sizes, [1, 12, 14, 18, 24, 30, 32, 98, 125]);
        assert_eq!(smooth(5_472 + 48), 5_600);
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
