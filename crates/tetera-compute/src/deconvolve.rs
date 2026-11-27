//! Deconvolution's geometry, which both backends share, and, with the `cuda` feature, its
//! kernels and FFTs (P1.07c): the element-wise steps of Richardson–Lucy, each a launch of its
//! own, and cuFFT's plans. The CPU twins are in `tetera-imaging`'s `deconvolve` module, which
//! composes these into the iteration.

use tetera_core::{Error, Result};

/// An image's place in the volume it is padded to for FFTs: mirrored on every side by the
/// PSF's half-size, and further at the far ends, up to sizes whose only prime factors are 2,
/// 3, 5 and 7, which FFTs transform fastest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    /// The image's width, height and depth.
    pub size: [usize; 3],
    /// The padded volume's.
    pub padded: [usize; 3],
    /// Where the image's first pixel lies in the padded volume: the PSF's half-size.
    pub offset: [usize; 3],
}

impl Padding {
    /// The padding for an image of `size` and a PSF of `psf`, each odd. A 3D PSF deconvolves
    /// a volume only.
    pub fn new(size: [usize; 3], psf: [usize; 3]) -> Result<Self> {
        if size.contains(&0) || psf.iter().any(|n| n.is_multiple_of(2)) {
            return Err(Error::invalid_argument(format!(
                "padding takes an image of some size and a PSF odd each way, not {size:?} and {psf:?}"
            )));
        }
        if size[2] == 1 && psf[2] > 1 {
            return Err(Error::invalid_argument(
                "a 3D PSF deconvolves a volume, not an image",
            ));
        }
        let offset = psf.map(|n| n / 2);
        let padded = [0, 1, 2].map(|a| smooth(size[a] + 2 * offset[a]));
        Ok(Self {
            size,
            padded,
            offset,
        })
    }

    /// The padded volume's count of values.
    pub fn count(&self) -> usize {
        self.padded.iter().product()
    }

    /// The image's count of values.
    pub fn image_count(&self) -> usize {
        self.size.iter().product()
    }

    /// Where the PSF's value at `at` goes in the padded volume when its centre is moved to the
    /// origin, wrapping around, as its transform needs.
    pub fn centred(&self, at: [usize; 3]) -> usize {
        let [x, y, z] =
            [0, 1, 2].map(|a| (at[a] + self.padded[a] - self.offset[a]) % self.padded[a]);
        (z * self.padded[1] + y) * self.padded[0] + x
    }
}

/// Index `i` reflected into `0..n` about its ends without repeating them, as often as it takes:
/// …, 2, 1, 0, 1, 2, …, n − 2, n − 1, n − 2, ….
pub fn reflect(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as i64 - 1);
    let i = i.rem_euclid(period);
    (if i < n as i64 { i } else { period - i }) as usize
}

/// The least size from `n` up whose only prime factors are 2, 3, 5 and 7.
pub fn smooth(n: usize) -> usize {
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

#[cfg(feature = "cuda")]
pub use device::{Complex, DeviceFft};

#[cfg(feature = "cuda")]
mod device {
    use super::Padding;
    use crate::cuda::library_loads;
    use crate::kernels::DECONVOLVE;
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::cufft::sys::cufftType;
    use cudarc::cufft::{CudaFft, FftDirection};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use tetera_core::{Error, Result};

    /// A complex value on the device, as cuFFT takes it: `x` its real part, `y` its imaginary.
    pub type Complex = cudarc::cufft::sys::float2;

    fn launch_failed(what: &str, e: cudarc::driver::DriverError) -> Error {
        Error::hardware(format!("the {what} kernel wouldn't launch: {e:?}"))
    }

    /// One thread a value, 256 a block.
    fn per_value(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u64).div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Whether a volume's sizes each fit an `i32`, as the kernels take them.
    fn sizes(values: &[usize]) -> Result<Vec<i32>> {
        values
            .iter()
            .map(|&n| i32::try_from(n))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| Error::invalid_argument(format!("{values:?} is too large for one launch")))
    }

    fn lengths(what: &str, got: &[usize], expected: &[usize]) -> Result<()> {
        if got == expected {
            Ok(())
        } else {
            Err(Error::invalid_argument(format!(
                "{what} takes buffers of {expected:?} values, not {got:?}"
            )))
        }
    }

    /// A plan for complex FFTs over a volume, unnormalised, on the backend's stream.
    pub struct DeviceFft {
        plan: CudaFft,
        count: usize,
    }

    impl std::fmt::Debug for DeviceFft {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "DeviceFft {{ count: {} }}", self.count)
        }
    }

    impl DeviceFft {
        /// `input` transformed into `output`, forward or back. cuFFT may overwrite `input`.
        pub fn run(
            &self,
            input: &mut DeviceBuffer<Complex>,
            output: &mut DeviceBuffer<Complex>,
            inverse: bool,
        ) -> Result<()> {
            lengths("an FFT", &[input.len(), output.len()], &[self.count; 2])?;
            let direction = if inverse {
                FftDirection::Inverse
            } else {
                FftDirection::Forward
            };
            self.plan
                .exec_c2c(input.slice_mut(), output.slice_mut(), direction)
                .map_err(|e| Error::hardware(format!("cuFFT: {e:?}")))
        }

        /// The count of values it transforms.
        pub fn count(&self) -> usize {
            self.count
        }
    }

    impl CudaBackend {
        /// A plan for complex FFTs over a volume of `size`, x fastest. A 3D plan serves every
        /// volume: cuFFT takes axes of 1, and an image planned as 1 deep runs as fast as with
        /// a 2D plan.
        pub fn fft_plan(&self, size: [usize; 3]) -> Result<DeviceFft> {
            library_loads(&["libcufft.so.12", "libcufft.so"])
                .map_err(|e| Error::hardware(format!("cuFFT: {e}")))?;
            let count = size.iter().product();
            let [nx, ny, nz] = sizes(&size)?[..] else {
                unreachable!("three sizes")
            };
            // cuFFT takes the slowest axis first.
            let plan = CudaFft::plan_3d(nz, ny, nx, cufftType::CUFFT_C2C, self.stream().clone())
                .map_err(|e| Error::hardware(format!("cuFFT's plan for {size:?}: {e:?}")))?;
            Ok(DeviceFft { plan, count })
        }

        /// `image`, `padding.size` values, mirrored into `out`, the padded volume.
        pub fn rl_pad(
            &self,
            image: &DeviceBuffer<f32>,
            padding: &Padding,
            out: &mut DeviceBuffer<f32>,
        ) -> Result<()> {
            let (n, image_n) = (padding.count(), padding.image_count());
            lengths("padding", &[image.len(), out.len()], &[image_n, n])?;
            let [w, h, d, pw, ph, pd, ox, oy, oz] =
                sizes(&[padding.size, padding.padded, padding.offset].concat())?[..]
            else {
                unreachable!("nine sizes")
            };
            let kernel = self.module(&DECONVOLVE)?.function("rl_pad_f32")?;
            let mut launch = self.stream().launch_builder(&kernel);
            launch.arg(out.slice_mut()).arg(image.slice());
            for v in [&w, &h, &d, &pw, &ph, &pd, &ox, &oy, &oz] {
                launch.arg(v);
            }
            // SAFETY: the kernel takes (float* out, const float* image, int w, int h, int d,
            // int pw, int ph, int pd, int ox, int oy, int oz), matched here in order and type.
            // Thread i < pw × ph × pd writes out[i], which holds that many (checked); it reads
            // image at a point reflected into 0..w, 0..h and 0..d, which holds w × h × d
            // (checked). The sizes are positive, as `Padding::new` made them.
            unsafe { launch.launch(per_value(n)) }.map_err(|e| launch_failed("rl_pad_f32", e))?;
            Ok(())
        }

        /// The image's part of `padded` into `out`.
        pub fn rl_crop(
            &self,
            padded: &DeviceBuffer<f32>,
            padding: &Padding,
            out: &mut DeviceBuffer<f32>,
        ) -> Result<()> {
            let (n, image_n) = (padding.count(), padding.image_count());
            lengths("cropping", &[padded.len(), out.len()], &[n, image_n])?;
            let [w, h, d, pw, ph, _, ox, oy, oz] =
                sizes(&[padding.size, padding.padded, padding.offset].concat())?[..]
            else {
                unreachable!("nine sizes")
            };
            let kernel = self.module(&DECONVOLVE)?.function("rl_crop_f32")?;
            let mut launch = self.stream().launch_builder(&kernel);
            launch.arg(out.slice_mut()).arg(padded.slice());
            for v in [&w, &h, &d, &pw, &ph, &ox, &oy, &oz] {
                launch.arg(v);
            }
            // SAFETY: the kernel takes (float* out, const float* padded, int w, int h, int d,
            // int pw, int ph, int ox, int oy, int oz), matched here in order and type. Thread
            // i < w × h × d writes out[i], which holds that many (checked), and reads padded at
            // (x + ox, y + oy, z + oz), inside the padded volume since each padded size is at
            // least the image's plus twice its offset (as `Padding::new` made it); padded
            // holds pw × ph × pd (checked).
            unsafe { launch.launch(per_value(image_n)) }
                .map_err(|e| launch_failed("rl_crop_f32", e))?;
            Ok(())
        }

        /// `values` as complex numbers, their imaginary parts 0, into `out`.
        pub fn rl_to_complex(
            &self,
            values: &DeviceBuffer<f32>,
            out: &mut DeviceBuffer<Complex>,
        ) -> Result<()> {
            lengths("rl_to_complex", &[out.len()], &[values.len()])?;
            let n = values.len() as i64;
            let kernel = self.module(&DECONVOLVE)?.function("rl_to_complex")?;
            let mut launch = self.stream().launch_builder(&kernel);
            launch.arg(out.slice_mut()).arg(values.slice()).arg(&n);
            // SAFETY: the kernel takes (float2* out, const float* values, long long n), matched
            // here in order and type; `Complex` is cuFFT's float2. Thread i < n reads values[i]
            // and writes out[i], both n long (checked).
            unsafe { launch.launch(per_value(values.len())) }
                .map_err(|e| launch_failed("rl_to_complex", e))?;
            Ok(())
        }

        /// `buffer` times `otf` scaled by `scale`, or by its conjugate, in place.
        pub fn rl_multiply(
            &self,
            buffer: &mut DeviceBuffer<Complex>,
            otf: &DeviceBuffer<Complex>,
            scale: f32,
            conjugate: bool,
        ) -> Result<()> {
            lengths("rl_multiply", &[otf.len()], &[buffer.len()])?;
            let (n, conjugate) = (buffer.len() as i64, i32::from(conjugate));
            let kernel = self.module(&DECONVOLVE)?.function("rl_multiply")?;
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(buffer.slice_mut())
                .arg(otf.slice())
                .arg(&scale)
                .arg(&conjugate)
                .arg(&n);
            // SAFETY: the kernel takes (float2* buffer, const float2* otf, float scale, int
            // conjugate, long long n), matched here in order and type. Thread i < n reads
            // otf[i] and reads and writes buffer[i], both n long (checked).
            unsafe { launch.launch(per_value(n as usize)) }
                .map_err(|e| launch_failed("rl_multiply", e))?;
            Ok(())
        }

        /// Richardson–Lucy's ratio, in place: each value's real part b becomes d / b, with d
        /// from `data`, where b is above `floor`, and 1 elsewhere; its imaginary part 0.
        pub fn rl_ratio(
            &self,
            blurred: &mut DeviceBuffer<Complex>,
            data: &DeviceBuffer<f32>,
            floor: f32,
        ) -> Result<()> {
            lengths("rl_ratio", &[data.len()], &[blurred.len()])?;
            let n = blurred.len() as i64;
            let kernel = self.module(&DECONVOLVE)?.function("rl_ratio")?;
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(blurred.slice_mut())
                .arg(data.slice())
                .arg(&floor)
                .arg(&n);
            // SAFETY: the kernel takes (float2* blurred, const float* data, float floor, long
            // long n), matched here in order and type. Thread i < n reads data[i] and reads
            // and writes blurred[i], both n long (checked).
            unsafe { launch.launch(per_value(n as usize)) }
                .map_err(|e| launch_failed("rl_ratio", e))?;
            Ok(())
        }

        /// Richardson–Lucy's update: each value of `estimate` times the real part of
        /// `correction`, over `denominators` when total variation is on, held at 0 or more.
        pub fn rl_update(
            &self,
            estimate: &mut DeviceBuffer<f32>,
            correction: &DeviceBuffer<Complex>,
            denominators: Option<&DeviceBuffer<f32>>,
        ) -> Result<()> {
            let n = estimate.len();
            let given = denominators.map_or(n, DeviceBuffer::len);
            lengths("rl_update", &[correction.len(), given], &[n, n])?;
            let name = if denominators.is_some() {
                "rl_update_tv"
            } else {
                "rl_update"
            };
            let kernel = self.module(&DECONVOLVE)?.function(name)?;
            let count = n as i64;
            let mut launch = self.stream().launch_builder(&kernel);
            launch.arg(estimate.slice_mut()).arg(correction.slice());
            if let Some(t) = denominators {
                launch.arg(t.slice());
            }
            launch.arg(&count);
            // SAFETY: `rl_update` takes (float* estimate, const float2* correction, long long
            // n) and `rl_update_tv` (float* estimate, const float2* correction, const float*
            // denominators, long long n): the denominators are passed exactly when the second
            // is launched, matched in order and type. Thread i < n reads correction[i] and
            // denominators[i] and reads and writes estimate[i], each n long (checked).
            unsafe { launch.launch(per_value(n)) }.map_err(|e| launch_failed(name, e))?;
            Ok(())
        }

        /// Total variation's denominators of `u`, a volume of `size`, x fastest, into `out`:
        /// `1 − λ div(∇u / |∇u|)`, held at 0.1 or more, `|∇u|` at `epsilon` or more.
        pub fn rl_total_variation(
            &self,
            u: &DeviceBuffer<f32>,
            size: [usize; 3],
            lambda: f32,
            epsilon: f32,
            out: &mut DeviceBuffer<f32>,
        ) -> Result<()> {
            let n: usize = size.iter().product();
            lengths("rl_total_variation", &[u.len(), out.len()], &[n, n])?;
            let [nx, ny, nz] = sizes(&size)?[..] else {
                unreachable!("three sizes")
            };
            if n == 0 || nz > 65_535 {
                return Err(Error::invalid_argument(format!(
                    "total variation takes a volume of some size, at most 65,535 deep, not {size:?}"
                )));
            }
            let kernel = self.module(&DECONVOLVE)?.function("rl_total_variation")?;
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(u.slice())
                .arg(&nx)
                .arg(&ny)
                .arg(&nz)
                .arg(&lambda)
                .arg(&epsilon);
            let config = LaunchConfig {
                grid_dim: (
                    (nx as u32).div_ceil(16),
                    (ny as u32).div_ceil(16),
                    nz as u32,
                ),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the kernel takes (float* out, const float* u, int nx, int ny, int nz,
            // float lambda, float epsilon), matched here in order and type. Thread (x, y, z),
            // for x < nx, y < ny and z < nz, reads u at its point and at most one step past it
            // or before it on each axis, only where that stays inside the volume, and writes
            // out at its point; both hold nx × ny × nz (checked). The gradient's length is at
            // least `epsilon`, so its division is defined for any epsilon above 0, and gives
            // infinities or NaNs otherwise, never a fault.
            unsafe { launch.launch(config) }.map_err(|e| launch_failed("rl_total_variation", e))?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn padding_mirrors_by_the_psfs_half_size_and_centres_it_at_the_origin() {
        let p = Padding::new([10, 6, 1], [5, 3, 1]).unwrap();
        assert_eq!((p.padded, p.offset), ([14, 8, 1], [2, 1, 0]));
        assert_eq!((p.count(), p.image_count()), (112, 60));
        // The centre goes to the origin; the PSF's first value wraps to the far corner.
        assert_eq!(p.centred([2, 1, 0]), 0);
        assert_eq!(p.centred([3, 1, 0]), 1);
        assert_eq!(p.centred([2, 2, 0]), 14);
        assert_eq!(p.centred([0, 0, 0]), 7 * 14 + 12);
        let v = Padding::new([4, 4, 6], [3, 3, 5]).unwrap();
        assert_eq!((v.padded, v.offset), ([6, 6, 10], [1, 1, 2]));
        assert_eq!(v.centred([1, 1, 0]), 8 * 36);
        assert_eq!(
            Padding::new([4, 4, 1], [3, 3, 3]).unwrap_err().to_string(),
            "a 3D PSF deconvolves a volume, not an image"
        );
        assert!(Padding::new([4, 0, 1], [3, 3, 1]).is_err());
        assert!(Padding::new([4, 4, 1], [3, 2, 1]).is_err());
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use crate::DeviceBuffer;

    fn zero() -> Complex {
        Complex { x: 0.0, y: 0.0 }
    }

    #[test]
    fn a_plan_transforms_a_point_to_a_flat_spectrum_and_back() {
        let gpu = crate::cuda_or_skip!();
        for size in [[8, 1, 1], [6, 5, 1], [4, 3, 5], [1, 1, 1]] {
            let fft = gpu.fft_plan(size).unwrap();
            let n: usize = size.iter().product();
            let mut point = vec![zero(); n];
            point[0] = Complex { x: 2.0, y: 0.0 };
            let mut a = gpu.upload(&point).unwrap();
            let mut b: DeviceBuffer<Complex> = gpu.zeros(n).unwrap();
            fft.run(&mut a, &mut b, false).unwrap();
            assert!(
                gpu.download(&b)
                    .unwrap()
                    .iter()
                    .all(|c| *c == Complex { x: 2.0, y: 0.0 })
            );
            fft.run(&mut b, &mut a, true).unwrap();
            let back = gpu.download(&a).unwrap();
            assert_eq!(
                back[0],
                Complex {
                    x: 2.0 * n as f32,
                    y: 0.0
                },
                "unnormalised"
            );
            assert!(
                back[1..]
                    .iter()
                    .all(|c| c.x.abs() < 1e-5 && c.y.abs() < 1e-5)
            );
        }
        // A wave of one cycle along y lands in the bin of frequency 1 along y.
        let fft = gpu.fft_plan([4, 4, 1]).unwrap();
        assert_eq!(fft.count(), 16);
        assert_eq!(format!("{fft:?}"), "DeviceFft { count: 16 }");
        let wave: Vec<Complex> = (0..16)
            .map(|i| {
                let phase = std::f32::consts::TAU * (i / 4) as f32 / 4.0;
                Complex {
                    x: phase.cos(),
                    y: phase.sin(),
                }
            })
            .collect();
        let (mut a, mut b) = (gpu.upload(&wave).unwrap(), gpu.zeros(16).unwrap());
        fft.run(&mut a, &mut b, false).unwrap();
        let spectrum = gpu.download(&b).unwrap();
        assert!((spectrum[4].x - 16.0).abs() < 1e-4, "{:?}", spectrum[4]);
        assert!(
            spectrum
                .iter()
                .enumerate()
                .all(|(i, c)| i == 4 || c.x.abs() + c.y.abs() < 1e-4)
        );
        let err = fft
            .run(
                &mut gpu.zeros(15).unwrap(),
                &mut gpu.zeros(16).unwrap(),
                false,
            )
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "an FFT takes buffers of [16, 16] values, not [15, 16]"
        );
    }

    #[test]
    fn the_kernels_work_out_small_cases_by_hand() {
        let gpu = crate::cuda_or_skip!();
        // Padding [1, 2, 3] by one each side mirrors it: 2 1 2 3 2.
        let padding = Padding::new([3, 1, 1], [3, 1, 1]).unwrap();
        assert_eq!(padding.padded, [5, 1, 1]);
        let image = gpu.upload(&[1.0f32, 2.0, 3.0]).unwrap();
        let mut padded = gpu.zeros(5).unwrap();
        gpu.rl_pad(&image, &padding, &mut padded).unwrap();
        assert_eq!(gpu.download(&padded).unwrap(), [2.0, 1.0, 2.0, 3.0, 2.0]);
        let mut cropped = gpu.zeros(3).unwrap();
        gpu.rl_crop(&padded, &padding, &mut cropped).unwrap();
        assert_eq!(gpu.download(&cropped).unwrap(), [1.0, 2.0, 3.0]);
        // (1 + 2i) × (3 + 4i) / 2 = −2.5 + 5i; by its conjugate, (11 + 2i) / 2.
        let mut buffer = gpu.upload(&[Complex { x: 1.0, y: 2.0 }]).unwrap();
        let otf = gpu.upload(&[Complex { x: 3.0, y: 4.0 }]).unwrap();
        gpu.rl_multiply(&mut buffer, &otf, 0.5, false).unwrap();
        assert_eq!(
            gpu.download(&buffer).unwrap(),
            [Complex { x: -2.5, y: 5.0 }]
        );
        let mut buffer = gpu.upload(&[Complex { x: 1.0, y: 2.0 }]).unwrap();
        gpu.rl_multiply(&mut buffer, &otf, 0.5, true).unwrap();
        assert_eq!(gpu.download(&buffer).unwrap(), [Complex { x: 5.5, y: 1.0 }]);
        // The ratio: 6 / 3, and 1 at or below the floor of 0.5.
        let blurred = [3.0, 0.5, -1.0].map(|x| Complex { x, y: 9.0 });
        let mut buffer = gpu.upload(&blurred).unwrap();
        gpu.rl_ratio(&mut buffer, &gpu.upload(&[6.0f32, 6.0, 6.0]).unwrap(), 0.5)
            .unwrap();
        let ratios: Vec<(f32, f32)> = gpu
            .download(&buffer)
            .unwrap()
            .iter()
            .map(|c| (c.x, c.y))
            .collect();
        assert_eq!(ratios, [(2.0, 0.0), (1.0, 0.0), (1.0, 0.0)]);
        // The update: 4 × 1.5 = 6, 4 × −1 held at 0; over a denominator of 2, 3.
        let correction = gpu
            .upload(&[1.5, -1.0].map(|x| Complex { x, y: 7.0 }))
            .unwrap();
        let mut estimate = gpu.upload(&[4.0f32, 4.0]).unwrap();
        gpu.rl_update(&mut estimate, &correction, None).unwrap();
        assert_eq!(gpu.download(&estimate).unwrap(), [6.0, 0.0]);
        let mut estimate = gpu.upload(&[4.0f32, 4.0]).unwrap();
        let halves = gpu.upload(&[2.0f32, 2.0]).unwrap();
        gpu.rl_update(&mut estimate, &correction, Some(&halves))
            .unwrap();
        assert_eq!(gpu.download(&estimate).unwrap(), [3.0, 0.0]);
        // Real to complex.
        let mut out = gpu.zeros(2).unwrap();
        gpu.rl_to_complex(&gpu.upload(&[5.0f32, -1.0]).unwrap(), &mut out)
            .unwrap();
        assert_eq!(
            gpu.download(&out).unwrap(),
            [Complex { x: 5.0, y: 0.0 }, Complex { x: -1.0, y: 0.0 }]
        );
        // Total variation on 0 1 0 along x: the point's divergence is −2, its neighbours'
        // +1 each, by forward and backward differences; λ 0.25.
        let mut tv = gpu.zeros(3).unwrap();
        gpu.rl_total_variation(
            &gpu.upload(&[0.0f32, 1.0, 0.0]).unwrap(),
            [3, 1, 1],
            0.25,
            1e-6,
            &mut tv,
        )
        .unwrap();
        assert_eq!(gpu.download(&tv).unwrap(), [0.75, 1.5, 0.75]);
    }

    #[test]
    fn each_launch_refuses_buffers_of_the_wrong_length() {
        let gpu = crate::cuda_or_skip!();
        let padding = Padding::new([3, 2, 1], [3, 3, 1]).unwrap();
        let n = padding.count();
        let f = |len: usize| gpu.zeros::<f32>(len).unwrap();
        let c = |len: usize| gpu.zeros::<Complex>(len).unwrap();
        let message = |e: tetera_core::Error| e.to_string();
        assert_eq!(
            message(gpu.rl_pad(&f(5), &padding, &mut f(n)).unwrap_err()),
            format!("padding takes buffers of [6, {n}] values, not [5, {n}]")
        );
        assert!(gpu.rl_pad(&f(6), &padding, &mut f(n - 1)).is_err());
        assert!(gpu.rl_crop(&f(n - 1), &padding, &mut f(6)).is_err());
        assert!(gpu.rl_crop(&f(n), &padding, &mut f(7)).is_err());
        assert!(gpu.rl_to_complex(&f(4), &mut c(5)).is_err());
        assert!(gpu.rl_multiply(&mut c(4), &c(5), 1.0, false).is_err());
        assert!(gpu.rl_ratio(&mut c(4), &f(5), 0.0).is_err());
        assert!(gpu.rl_update(&mut f(4), &c(5), None).is_err());
        assert!(gpu.rl_update(&mut f(4), &c(4), Some(&f(3))).is_err());
        assert!(
            gpu.rl_total_variation(&f(6), [3, 2, 1], 0.1, 1.0, &mut f(5))
                .is_err()
        );
        assert!(
            gpu.rl_total_variation(&f(5), [3, 2, 1], 0.1, 1.0, &mut f(6))
                .is_err()
        );
        assert_eq!(
            message(
                gpu.rl_total_variation(&f(0), [0, 2, 1], 0.1, 1.0, &mut f(0))
                    .unwrap_err()
            ),
            "total variation takes a volume of some size, at most 65,535 deep, not [0, 2, 1]"
        );
        assert_eq!(
            message(
                gpu.rl_total_variation(&f(65_536), [1, 1, 65_536], 0.1, 1.0, &mut f(65_536))
                    .unwrap_err()
            ),
            "total variation takes a volume of some size, at most 65,535 deep, not [1, 1, 65536]"
        );
        assert!(
            gpu.rl_total_variation(&f(65_535), [1, 1, 65_535], 0.1, 1.0, &mut f(65_535))
                .is_ok()
        );
        // Each length alone is checked, and the right ones pass.
        assert!(gpu.rl_pad(&f(6), &padding, &mut f(n)).is_ok());
        assert!(gpu.rl_crop(&f(n), &padding, &mut f(6)).is_ok());
        assert!(gpu.rl_update(&mut f(4), &c(4), Some(&f(4))).is_ok());
        assert!(
            gpu.rl_total_variation(&f(6), [3, 2, 1], 0.1, 1.0, &mut f(6))
                .is_ok()
        );
    }
}
