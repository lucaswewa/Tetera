//! The sensor's kernel (P1.12b, ADR-0016): `kernels/sensor.cu`, the GPU twin of
//! `tetera-sim`'s `sensor::pixel`. The model lives in `tetera-sim`; this module launches the
//! kernel from plain numbers, since only this crate may hold `unsafe` code.

/// What the kernel needs of a sensor and an exposure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorParams {
    /// The seed, folded to 32 bits as `tetera-sim` folds it.
    pub seed: u32,
    /// The frame's number.
    pub frame: u64,
    /// Quantum efficiency in red, green and blue.
    pub quantum_efficiency: [f32; 3],
    /// The full well, electrons.
    pub full_well: f32,
    /// Read noise, electrons rms.
    pub read_noise: f32,
    /// Dark current, electrons a second.
    pub dark_current: f32,
    /// The black level, 12-bit counts.
    pub black_level: f32,
    /// Electrons a count at a gain of 1.
    pub electrons_per_count: f32,
    /// The exposure, ms.
    pub ms: f32,
    /// The gain.
    pub gain: f32,
    /// Significant bits, 8 or 12.
    pub bits: u8,
    /// Pixels binned each way.
    pub binning: u32,
}

#[cfg(feature = "cuda")]
mod device {
    use super::SensorParams;
    use crate::kernels::SENSOR;
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use tetera_core::{Error, Result};

    impl CudaBackend {
        /// A raw frame from `rgb`, a `width` × `height` field's irradiance three to a pixel,
        /// into `raw`, a count to a pixel, its `height` rows `raw.len() / height` counts
        /// apart, as a frame's stride may set them.
        pub fn sensor_expose(
            &self,
            rgb: &DeviceBuffer<f32>,
            width: u32,
            height: u32,
            params: &SensorParams,
            raw: &mut DeviceBuffer<u16>,
        ) -> Result<()> {
            let n = width as usize * height as usize;
            let pitch = raw.len() / (height.max(1) as usize);
            if rgb.len() != 3 * n
                || height == 0
                || !raw.len().is_multiple_of(height as usize)
                || pitch < width as usize
                || !matches!(params.bits, 8 | 12)
                || params.binning == 0
            {
                return Err(Error::invalid_argument(format!(
                    "a {width} × {height} frame from {} values into {}, {} bits, binned {}",
                    rgb.len(),
                    raw.len(),
                    params.bits,
                    params.binning
                )));
            }
            let function = self.module(&SENSOR)?.function("sensor_expose")?;
            let size = |v: u32| {
                i32::try_from(v)
                    .map_err(|_| Error::invalid_argument(format!("a frame {v} pixels across")))
            };
            let (w, h) = (size(width)?, size(height)?);
            let pitch = size(u32::try_from(pitch).unwrap_or(u32::MAX))?;
            let (lo, hi) = (params.frame as u32, (params.frame >> 32) as u32);
            let [r, g, b] = params.quantum_efficiency;
            let bits = i32::from(params.bits);
            let binning = params.binning as i32;
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(raw.slice_mut())
                .arg(&pitch)
                .arg(rgb.slice())
                .arg(&w)
                .arg(&h)
                .arg(&params.seed)
                .arg(&lo)
                .arg(&hi)
                .arg(&r)
                .arg(&g)
                .arg(&b)
                .arg(&params.full_well)
                .arg(&params.read_noise)
                .arg(&params.dark_current)
                .arg(&params.black_level)
                .arg(&params.electrons_per_count)
                .arg(&params.ms)
                .arg(&params.gain)
                .arg(&bits)
                .arg(&binning);
            let config = LaunchConfig {
                grid_dim: (width.div_ceil(16), height.div_ceil(16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `sensor_expose` takes (unsigned short* raw, int pitch, const float* rgb, int width,
            // int height, unsigned int seed, unsigned int frame_lo, unsigned int frame_hi,
            // float qe_r, float qe_g, float qe_b, float full_well, float read_noise,
            // float dark_current, float black_level, float electrons_per_count, float ms,
            // float gain, int bits, int binning), matched here in order and type. Thread
            // (col, row) reads three values at 3 · (row · width + col), inside `rgb` of
            // 3 · width · height, and writes one at row · pitch + col, inside `raw` of
            // pitch · height with width <= pitch, both checked above.
            unsafe { launch.launch(config) }.map_err(|e| {
                Error::hardware(format!("the sensor_expose kernel wouldn't launch: {e:?}"))
            })?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn gpu() -> Option<crate::CudaBackend> {
        if !crate::select::cuda_tests_enabled() {
            eprintln!("skipped: set TETERA_CUDA=1 to run CUDA tests");
            return None;
        }
        Some(crate::CudaBackend::new(0).expect("TETERA_CUDA=1 promises a working device 0"))
    }

    fn params() -> SensorParams {
        SensorParams {
            seed: 7,
            frame: 3,
            quantum_efficiency: [0.45, 0.62, 0.42],
            full_well: 15_000.0,
            read_noise: 2.2,
            dark_current: 0.5,
            black_level: 100.0,
            electrons_per_count: 3.76,
            ms: 10.0,
            gain: 1.0,
            bits: 12,
            binning: 1,
        }
    }

    #[test]
    fn a_frame_is_made_and_its_shapes_are_checked() {
        let Some(gpu) = gpu() else { return };
        let rgb = gpu.upload(&[100.0f32; 3 * 64]).unwrap();
        let mut raw = gpu.zeros::<u16>(64).unwrap();
        gpu.sensor_expose(&rgb, 8, 8, &params(), &mut raw).unwrap();
        let counts = gpu.download(&raw).unwrap();
        assert!(
            counts.iter().all(|c| (100..4_096).contains(c)),
            "{counts:?}"
        );
        let again = {
            gpu.sensor_expose(&rgb, 8, 8, &params(), &mut raw).unwrap();
            gpu.download(&raw).unwrap()
        };
        assert_eq!(counts, again, "the same bits again");
        let later = SensorParams {
            frame: 3 + (1 << 40),
            ..params()
        };
        gpu.sensor_expose(&rgb, 8, 8, &later, &mut raw).unwrap();
        assert_ne!(
            gpu.download(&raw).unwrap(),
            counts,
            "a frame number's high bits"
        );
        let mut short = gpu.zeros::<u16>(63).unwrap();
        assert!(
            gpu.sensor_expose(&rgb, 8, 8, &params(), &mut short)
                .is_err(),
            "a short frame"
        );
        for (width, p) in [
            (9, params()),
            (
                8,
                SensorParams {
                    bits: 10,
                    ..params()
                },
            ),
            (
                8,
                SensorParams {
                    binning: 0,
                    ..params()
                },
            ),
        ] {
            assert!(
                gpu.sensor_expose(&rgb, width, 8, &p, &mut raw).is_err(),
                "{width} {p:?}"
            );
        }
        // Rows 8 counts apart for a frame 6 wide: the ends of the rows are left alone.
        let narrow = gpu.upload(&[100.0f32; 3 * 24]).unwrap();
        let mut pitched = gpu.upload(&[7u16; 32]).unwrap();
        gpu.sensor_expose(&narrow, 6, 4, &params(), &mut pitched)
            .unwrap();
        let rows = gpu.download(&pitched).unwrap();
        for row in rows.chunks(8) {
            assert!(row[..6].iter().all(|c| *c >= 100), "{row:?}");
            assert_eq!(&row[6..], &[7, 7], "{row:?}");
        }
        let eight = SensorParams {
            bits: 8,
            gain: 16.0,
            ..params()
        };
        let bright = gpu.upload(&[1e9f32; 3 * 64]).unwrap();
        gpu.sensor_expose(&bright, 8, 8, &eight, &mut raw).unwrap();
        assert_eq!(gpu.download(&raw).unwrap(), vec![255; 64], "8 bits");
    }
}
