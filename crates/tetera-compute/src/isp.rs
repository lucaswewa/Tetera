//! The ISP's correction kernels (P1.06a): dark subtraction and the flat field on a mosaic, and
//! colour, a 3 × 3 matrix then a lookup table per channel, on RGB. Their CPU twins, and the
//! tables they take, are in `tetera-imaging`'s `correct` and `colour` modules.

/// A flat-field gain's unit: a gain of 1.
pub const GAIN_ONE: u16 = 4_096;

/// A colour matrix entry's unit: 1.
pub const MATRIX_ONE: i32 = 16_384;

#[cfg(feature = "cuda")]
mod device {
    use crate::imaging::TABLE_SIZE;
    use crate::kernels::ISP;
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use tetera_core::{Error, Result};

    fn linear(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u64).div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn same_lengths(what: &str, lengths: &[usize]) -> Result<()> {
        if lengths.windows(2).any(|w| w[0] != w[1]) {
            return Err(Error::invalid_argument(format!(
                "{what} needs buffers of one length, not {lengths:?}"
            )));
        }
        Ok(())
    }

    fn launch_failed(what: &str, e: cudarc::driver::DriverError) -> Error {
        Error::hardware(format!("the {what} kernel wouldn't launch: {e:?}"))
    }

    impl CudaBackend {
        /// `input − dark`, site by site, clamped at zero, into `out`.
        pub fn subtract_dark(
            &self,
            input: &DeviceBuffer<u16>,
            dark: &DeviceBuffer<u16>,
            out: &mut DeviceBuffer<u16>,
        ) -> Result<()> {
            same_lengths("dark subtraction", &[input.len(), dark.len(), out.len()])?;
            if input.is_empty() {
                return Ok(());
            }
            let kernel = self.module(&ISP)?.function("subtract_dark_u16")?;
            let sites = input.len() as i64;
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(dark.slice())
                .arg(&sites);
            // SAFETY: the kernel takes (unsigned short* out, const unsigned short* in, const
            // unsigned short* dark, long long sites), matched here in order and type. Thread
            // i < sites reads in[i] and dark[i] and writes out[i]; all three hold `sites`
            // values, checked above.
            unsafe { launch.launch(linear(input.len())) }
                .map_err(|e| launch_failed("subtract_dark_u16", e))?;
            Ok(())
        }

        /// `input` times `gains`, site by site, each gain in 1/[`GAIN_ONE`](super::GAIN_ONE)ths,
        /// rounded and clamped to `max`, into `out`.
        pub fn apply_flat(
            &self,
            input: &DeviceBuffer<u16>,
            gains: &DeviceBuffer<u16>,
            max: u16,
            out: &mut DeviceBuffer<u16>,
        ) -> Result<()> {
            same_lengths("the flat field", &[input.len(), gains.len(), out.len()])?;
            if input.is_empty() {
                return Ok(());
            }
            let kernel = self.module(&ISP)?.function("apply_flat_u16")?;
            let (sites, maxv) = (input.len() as i64, i32::from(max));
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(gains.slice())
                .arg(&sites)
                .arg(&maxv);
            // SAFETY: the kernel takes (unsigned short* out, const unsigned short* in, const
            // unsigned short* gains, long long sites, int maxv), matched here in order and
            // type. Thread i < sites reads in[i] and gains[i] and writes out[i]; all three hold
            // `sites` values, checked above.
            unsafe { launch.launch(linear(input.len())) }
                .map_err(|e| launch_failed("apply_flat_u16", e))?;
            Ok(())
        }

        /// Colour: `input`, interleaved RGB, each pixel times `matrix` (9 entries, row by
        /// row, in 1/[`MATRIX_ONE`](super::MATRIX_ONE)ths), rounded and clamped to 16 bits,
        /// then each channel looked up in its table of `tables`, into `out`.
        pub fn colour(
            &self,
            input: &DeviceBuffer<u16>,
            matrix: &DeviceBuffer<i32>,
            tables: &DeviceBuffer<u8>,
            out: &mut DeviceBuffer<u8>,
        ) -> Result<()> {
            if !input.len().is_multiple_of(3)
                || out.len() != input.len()
                || matrix.len() != 9
                || tables.len() != 3 * TABLE_SIZE
            {
                return Err(Error::invalid_argument(format!(
                    "colour needs whole RGB pixels, an output as long, 9 matrix entries and {} table entries; it has {} values in, {} out, {} entries and {}",
                    3 * TABLE_SIZE,
                    input.len(),
                    out.len(),
                    matrix.len(),
                    tables.len()
                )));
            }
            if input.is_empty() {
                return Ok(());
            }
            let kernel = self.module(&ISP)?.function("colour_u16_u8")?;
            let values = input.len() as i64;
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(&values)
                .arg(matrix.slice())
                .arg(tables.slice());
            // SAFETY: the kernel takes (unsigned char* out, const unsigned short* in, long long
            // values, const int* matrix, const unsigned char* tables), matched here in order
            // and type. Thread p < values / 3 reads in[3p..3p+3] and writes out[3p..3p+3],
            // inside both, which hold `values`, a multiple of 3 (checked). It reads matrix[0..9], checked to hold 9,
            // and tables[c × 65536 + v] for c < 3 and v clamped to 0..=65535, inside the
            // tables' checked 3 × 65536.
            unsafe { launch.launch(linear(input.len() / 3)) }
                .map_err(|e| launch_failed("colour_u16_u8", e))?;
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::imaging::TABLE_SIZE;

    #[test]
    fn dark_subtraction_clamps_at_zero() {
        let gpu = crate::cuda_or_skip!();
        let input = gpu.upload(&[100u16, 50, 0, 4_095, 7]).unwrap();
        let dark = gpu.upload(&[40u16, 60, 0, 95, 7]).unwrap();
        let mut out = gpu.zeros::<u16>(5).unwrap();
        gpu.subtract_dark(&input, &dark, &mut out).unwrap();
        assert_eq!(gpu.download(&out).unwrap(), [60, 0, 0, 4_000, 0]);
        let mut short = gpu.zeros::<u16>(4).unwrap();
        let err = gpu.subtract_dark(&input, &dark, &mut short).unwrap_err();
        assert_eq!(
            err.to_string(),
            "dark subtraction needs buffers of one length, not [5, 5, 4]"
        );
        let empty = gpu.upload::<u16>(&[]).unwrap();
        let mut none = gpu.zeros::<u16>(0).unwrap();
        gpu.subtract_dark(&empty, &empty, &mut none).unwrap();
    }

    #[test]
    fn the_flat_field_rounds_and_clamps() {
        let gpu = crate::cuda_or_skip!();
        let input = gpu.upload(&[1_000u16, 1_000, 3, 4_000, 1]).unwrap();
        // 1, 1.5, 0.5 (3 × 0.5 = 1.5 rounds up), 2 (clamped), 0.5 (0.5 rounds up).
        let gains = gpu.upload(&[GAIN_ONE, 6_144, 2_048, 8_192, 2_048]).unwrap();
        let mut out = gpu.zeros::<u16>(5).unwrap();
        gpu.apply_flat(&input, &gains, 4_095, &mut out).unwrap();
        assert_eq!(gpu.download(&out).unwrap(), [1_000, 1_500, 2, 4_095, 1]);
        let long = gpu.upload(&[GAIN_ONE; 6]).unwrap();
        assert!(gpu.apply_flat(&input, &long, 4_095, &mut out).is_err());
        let empty = gpu.upload::<u16>(&[]).unwrap();
        let mut none = gpu.zeros::<u16>(0).unwrap();
        gpu.apply_flat(&empty, &empty, 4_095, &mut none).unwrap();
    }

    #[test]
    fn colour_mixes_channels_then_looks_each_up() {
        let gpu = crate::cuda_or_skip!();
        // Swap red and blue, double green; tables that halve each value, kept in a byte.
        let m = MATRIX_ONE;
        let matrix = gpu.upload(&[0, 0, m, 0, 2 * m, 0, m, 0, 0]).unwrap();
        let tables: Vec<u8> = (0..3 * TABLE_SIZE)
            .map(|i| ((i % TABLE_SIZE) / 2).min(255) as u8)
            .collect();
        let tables = gpu.upload(&tables).unwrap();
        let input = gpu.upload(&[10u16, 20, 30, 100, 200, 300]).unwrap();
        let mut out = gpu.zeros::<u8>(6).unwrap();
        gpu.colour(&input, &matrix, &tables, &mut out).unwrap();
        // (30, 40, 10) and (300, 400, 100), halved.
        assert_eq!(gpu.download(&out).unwrap(), [15, 20, 5, 150, 200, 50]);
        // A negative mix clamps at 0, and one beyond 16 bits at 65,535.
        let wild = gpu.upload(&[-m, 0, 0, 4 * m, 0, 0, 0, 0, m]).unwrap();
        let bright = gpu.upload(&[30_000u16, 0, 7]).unwrap();
        let mut three = gpu.zeros::<u8>(3).unwrap();
        gpu.colour(&bright, &wild, &tables, &mut three).unwrap();
        assert_eq!(gpu.download(&three).unwrap(), [0, 255, 3]);
        let err = gpu.colour(&bright, &matrix, &tables, &mut out).unwrap_err();
        assert!(
            err.to_string().starts_with("colour needs whole RGB pixels"),
            "{err}"
        );
        let short_matrix = gpu.upload(&[m; 8]).unwrap();
        assert!(
            gpu.colour(&bright, &short_matrix, &tables, &mut three)
                .is_err()
        );
        let two = gpu.upload(&[1u16, 2]).unwrap();
        let mut two_out = gpu.zeros::<u8>(2).unwrap();
        assert!(
            gpu.colour(&two, &matrix, &tables, &mut two_out).is_err(),
            "not whole pixels"
        );
        let few_tables = gpu.upload(&vec![0u8; TABLE_SIZE]).unwrap();
        assert!(
            gpu.colour(&bright, &matrix, &few_tables, &mut three)
                .is_err()
        );
        let empty = gpu.upload::<u16>(&[]).unwrap();
        let mut none = gpu.zeros::<u8>(0).unwrap();
        gpu.colour(&empty, &matrix, &tables, &mut none).unwrap();
    }
}
