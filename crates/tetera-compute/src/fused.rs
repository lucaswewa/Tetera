//! The fused ISP kernel's launch (P1.06b, plan D7, ADR-0011): a raw frame to an 8-bit preview
//! in one pass, crop or bin, dark, flat, demosaic, resize and colour together. Its reference is
//! the unfused chain in `tetera-imaging`'s `isp`, which it matches exactly.

#[cfg(feature = "cuda")]
pub use device::FusedInputs;

#[cfg(feature = "cuda")]
mod device {
    use crate::imaging::{
        Demosaic, DeviceAxis, MosaicLayout, Reduce, TABLE_SIZE, TILE, pattern_offset,
    };
    use crate::kernels::FUSED;
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
    use tetera_core::{Error, Result};

    /// The kernel's parameters, field for field as `fused.cu`'s `FusedParams`.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct FusedParams {
        raw_width: i32,
        width: i32,
        height: i32,
        crop_x: i32,
        crop_y: i32,
        factor: i32,
        ox: i32,
        oy: i32,
        maxv: i32,
        mhc: i32,
        has_dark: i32,
        has_flat: i32,
        out_width: i32,
        out_height: i32,
        x_taps: i32,
        y_taps: i32,
    }

    // SAFETY: sixteen i32s with no padding, in the order and layout of the kernel's struct of
    // sixteen ints; the kernel receives it by value.
    unsafe impl DeviceRepr for FusedParams {}

    /// What the fused kernel reads.
    #[derive(Debug)]
    pub struct FusedInputs<'a> {
        /// The raw frame, whole, even for a crop.
        pub raw: &'a DeviceBuffer<u16>,
        /// Its layout.
        pub layout: MosaicLayout,
        /// How it is reduced first.
        pub reduce: Reduce,
        /// The master dark at the reduced layout, or none.
        pub dark: Option<&'a DeviceBuffer<u16>>,
        /// The flat field's gains at the reduced layout, or none.
        pub gains: Option<&'a DeviceBuffer<u16>>,
        /// The demosaicking method.
        pub demosaic: Demosaic,
        /// The resize across, from the reduced width.
        pub x: &'a DeviceAxis,
        /// The resize down, from the reduced height.
        pub y: &'a DeviceAxis,
        /// The colour matrix, 9 entries in 1/16384ths.
        pub matrix: &'a DeviceBuffer<i32>,
        /// Red's, green's and blue's tables.
        pub tables: &'a DeviceBuffer<u8>,
    }

    fn int(value: u32, what: &str) -> Result<i32> {
        i32::try_from(value).map_err(|_| {
            Error::invalid_argument(format!(
                "{what} of {value} is too large for the fused kernel"
            ))
        })
    }

    impl CudaBackend {
        /// The fused ISP: `inputs.raw` to an interleaved 8-bit RGB preview of `x.output() ×
        /// y.output()` in `out`. Every buffer is checked against the layouts first.
        pub fn fused_isp(
            &self,
            inputs: &FusedInputs<'_>,
            out: &mut DeviceBuffer<u8>,
        ) -> Result<()> {
            let raw = inputs.layout;
            if inputs.raw.len() != raw.sites() {
                return Err(Error::invalid_argument(format!(
                    "a {} × {} frame has {} sites, not {}",
                    raw.width,
                    raw.height,
                    raw.sites(),
                    inputs.raw.len()
                )));
            }
            let reduced = raw.reduced(inputs.reduce)?;
            let min = inputs.demosaic.min_size();
            if reduced.width < min || reduced.height < min {
                return Err(Error::invalid_argument(format!(
                    "{:?} needs at least {min} × {min}, not {} × {}",
                    inputs.demosaic, reduced.width, reduced.height
                )));
            }
            for (what, buffer) in [("dark", inputs.dark), ("flat field", inputs.gains)] {
                if let Some(b) = buffer
                    && b.len() != reduced.sites()
                {
                    return Err(Error::invalid_argument(format!(
                        "the {what} holds {} sites, not the reduced frame's {}",
                        b.len(),
                        reduced.sites()
                    )));
                }
            }
            let (x, y) = (inputs.x, inputs.y);
            if (x.input(), y.input()) != (reduced.width, reduced.height) {
                return Err(Error::invalid_argument(format!(
                    "the resize takes {} × {}, but the reduced frame is {} × {}",
                    x.input(),
                    y.input(),
                    reduced.width,
                    reduced.height
                )));
            }
            let pixels = x.output() as usize * y.output() as usize;
            if inputs.matrix.len() != 9
                || inputs.tables.len() != 3 * TABLE_SIZE
                || out.len() != 3 * pixels
            {
                return Err(Error::invalid_argument(format!(
                    "colour needs 9 matrix entries and {} table entries, and the preview {} values; they have {}, {} and {}",
                    3 * TABLE_SIZE,
                    3 * pixels,
                    inputs.matrix.len(),
                    inputs.tables.len(),
                    out.len()
                )));
            }
            if pixels == 0 {
                return Ok(());
            }
            let (crop_x, crop_y, factor) = match inputs.reduce {
                Reduce::Crop(roi) => (roi.x, roi.y, 1),
                Reduce::Bin(factor) => (0, 0, factor),
                Reduce::Full => (0, 0, 1),
            };
            let (ox, oy) = pattern_offset(reduced.pattern);
            let params = FusedParams {
                raw_width: int(raw.width, "a raw width")?,
                width: int(reduced.width, "a width")?,
                height: int(reduced.height, "a height")?,
                crop_x: crop_x as i32,
                crop_y: crop_y as i32,
                factor: factor as i32,
                ox,
                oy,
                maxv: i32::from(reduced.max()),
                mhc: i32::from(inputs.demosaic == Demosaic::MalvarHeCutler),
                has_dark: i32::from(inputs.dark.is_some()),
                has_flat: i32::from(inputs.gains.is_some()),
                out_width: int(x.output(), "a preview width")?,
                out_height: int(y.output(), "a preview height")?,
                x_taps: x.taps() as i32,
                y_taps: y.taps() as i32,
            };
            int(raw.height, "a raw height")?;
            let kernel = self.module(&FUSED)?.function("fused_isp")?;
            // Absent calibrations are never read; the raw frame stands in for their pointers.
            let dark = inputs.dark.unwrap_or(inputs.raw);
            let gains = inputs.gains.unwrap_or(inputs.raw);
            let config = LaunchConfig {
                grid_dim: (x.output().div_ceil(TILE), y.output().div_ceil(TILE), 1),
                block_dim: (TILE, TILE, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(inputs.raw.slice())
                .arg(dark.slice())
                .arg(gains.slice())
                .arg(x.start.slice())
                .arg(x.weights.slice())
                .arg(y.start.slice())
                .arg(y.weights.slice())
                .arg(inputs.matrix.slice())
                .arg(inputs.tables.slice())
                .arg(&params);
            // SAFETY: the kernel takes (unsigned char* out, const unsigned short* raw, dark,
            // gains, const int* x_start, const unsigned short* x_weights, const int* y_start,
            // const unsigned short* y_weights, const int* matrix, const unsigned char* tables,
            // FusedParams p), matched here in order and type, `FusedParams` laid out as its
            // struct. Thread (x, y), for x < out_width and y < out_height:
            // - writes out[(y × out_width + x) × 3 ..+ 3], inside `out`, checked to hold
            //   out_width × out_height × 3;
            // - reads x_start[x], y_start[y], and x_taps (y_taps) weights from x × x_taps
            //   (y × y_taps), inside the axes' tables, whose lengths `upload_axis` checked;
            // - demosaics pixels xs + b, ys + a, which `upload_axis` checked lie inside the axes'
            //   inputs, the reduced width and height (checked above), reading sites up to two
            //   beyond, mirrored inside them, since both are at least the method's minimum;
            // - reads a reduced site (sx, sy) from the raw frame: for a crop, at (sx + roi.x,
            //   sy + roi.y), inside it since the roi lies inside the frame; for a bin, at most
            //   column 2 × (reduced width / 2) × factor − 1, below the raw width (rows likewise),
            //   as `MosaicLayout::binned` makes the reduced size; `raw` holds the frame's sites;
            // - reads dark and gains at sy × width + sx, inside them, checked to hold the
            //   reduced sites, and only when present;
            // - reads matrix[0..9] and tables[c × 65536 + v] for c < 3 and v clamped to
            //   0..=65535, inside both, as checked.
            unsafe { launch.launch(config) }.map_err(|e| {
                Error::hardware(format!("the fused_isp kernel wouldn't launch: {e:?}"))
            })?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use crate::imaging::{Demosaic, MosaicLayout, Reduce, TABLE_SIZE};
    use crate::isp::MATRIX_ONE;
    use crate::{CudaBackend, DeviceBuffer};
    use tetera_core::Rect;
    use tetera_core::frame::format::BayerPattern;

    /// Tables that keep each value up to 255 and clip above.
    fn identity_tables(gpu: &CudaBackend) -> DeviceBuffer<u8> {
        let tables: Vec<u8> = (0..3 * TABLE_SIZE)
            .map(|i| (i % TABLE_SIZE).min(255) as u8)
            .collect();
        gpu.upload(&tables).unwrap()
    }

    fn identity_matrix(gpu: &CudaBackend) -> DeviceBuffer<i32> {
        let m = MATRIX_ONE;
        gpu.upload(&[m, 0, 0, 0, m, 0, 0, 0, m]).unwrap()
    }

    /// An axis that copies `n` inputs to `n` outputs.
    fn copy_axis(gpu: &CudaBackend, n: u32) -> crate::imaging::DeviceAxis {
        let start: Vec<i32> = (0..n as i32).collect();
        gpu.upload_axis(n, n, 1, &start, &vec![16_384; n as usize])
            .unwrap()
    }

    fn layout(width: u32, height: u32) -> MosaicLayout {
        MosaicLayout {
            width,
            height,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        }
    }

    #[test]
    fn a_flat_grey_frame_comes_through_each_correction_as_it_should() {
        let gpu = crate::cuda_or_skip!();
        let (tables, matrix) = (identity_tables(&gpu), identity_matrix(&gpu));
        let raw = gpu.upload(&[200u16; 8 * 8]).unwrap();
        let (x, y) = (copy_axis(&gpu, 8), copy_axis(&gpu, 8));
        let mut out = gpu.zeros::<u8>(8 * 8 * 3).unwrap();
        let mut run =
            |dark: Option<&DeviceBuffer<u16>>, gains: Option<&DeviceBuffer<u16>>, method| {
                let inputs = super::FusedInputs {
                    raw: &raw,
                    layout: layout(8, 8),
                    reduce: Reduce::Full,
                    dark,
                    gains,
                    demosaic: method,
                    x: &x,
                    y: &y,
                    matrix: &matrix,
                    tables: &tables,
                };
                gpu.fused_isp(&inputs, &mut out).unwrap();
                gpu.download(&out).unwrap()
            };
        assert!(
            run(None, None, Demosaic::MalvarHeCutler)
                .iter()
                .all(|&v| v == 200)
        );
        assert!(
            run(None, None, Demosaic::Bilinear)
                .iter()
                .all(|&v| v == 200)
        );
        let dark = gpu.upload(&[150u16; 64]).unwrap();
        assert!(
            run(Some(&dark), None, Demosaic::Bilinear)
                .iter()
                .all(|&v| v == 50),
            "200 − 150"
        );
        let more = gpu.upload(&[250u16; 64]).unwrap();
        assert!(
            run(Some(&more), None, Demosaic::Bilinear)
                .iter()
                .all(|&v| v == 0),
            "clamped at zero"
        );
        let half = gpu.upload(&[2_048u16; 64]).unwrap();
        assert!(
            run(None, Some(&half), Demosaic::Bilinear)
                .iter()
                .all(|&v| v == 100),
            "× 0.5"
        );
        assert!(
            run(Some(&dark), Some(&half), Demosaic::Bilinear)
                .iter()
                .all(|&v| v == 25),
            "dark, then flat"
        );
    }

    #[test]
    fn the_method_asked_for_is_the_method_used() {
        let gpu = crate::cuda_or_skip!();
        // Tables that divide by 16, so 12-bit values fit a byte.
        let tables: Vec<u8> = (0..3 * TABLE_SIZE)
            .map(|i| ((i % TABLE_SIZE) / 16).min(255) as u8)
            .collect();
        let (tables, matrix) = (gpu.upload(&tables).unwrap(), identity_matrix(&gpu));
        // A bright red site at (4, 4) of 9 × 9, the rest dark.
        let mut sites = vec![0u16; 81];
        sites[4 * 9 + 4] = 4_000;
        let raw = gpu.upload(&sites).unwrap();
        let (x, y) = (copy_axis(&gpu, 9), copy_axis(&gpu, 9));
        let mut out = gpu.zeros::<u8>(81 * 3).unwrap();
        let mut at_site = |demosaic| {
            let inputs = super::FusedInputs {
                raw: &raw,
                layout: layout(9, 9),
                reduce: Reduce::Full,
                dark: None,
                gains: None,
                demosaic,
                x: &x,
                y: &y,
                matrix: &matrix,
                tables: &tables,
            };
            gpu.fused_isp(&inputs, &mut out).unwrap();
            gpu.download(&out).unwrap()[(4 * 9 + 4) * 3..(4 * 9 + 5) * 3].to_vec()
        };
        // Malvar-He-Cutler takes green and blue from the bright site's own gradient: 2,000 and
        // 3,000; bilinear, from the dark neighbours alone. Then each over 16.
        assert_eq!(at_site(Demosaic::MalvarHeCutler), [250, 125, 187]);
        assert_eq!(at_site(Demosaic::Bilinear), [250, 0, 0]);
    }

    #[test]
    fn crops_and_bins_read_the_right_sites() {
        let gpu = crate::cuda_or_skip!();
        let (tables, matrix) = (identity_tables(&gpu), identity_matrix(&gpu));
        // Left half 40, right half 120: a crop of the right half reads 120 alone.
        let sites: Vec<u16> = (0..16 * 8)
            .map(|i| if i % 16 < 8 { 40 } else { 120 })
            .collect();
        let raw = gpu.upload(&sites).unwrap();
        let (x4, y4) = (copy_axis(&gpu, 8), copy_axis(&gpu, 8));
        let mut out = gpu.zeros::<u8>(8 * 8 * 3).unwrap();
        let crop = super::FusedInputs {
            raw: &raw,
            layout: layout(16, 8),
            reduce: Reduce::Crop(Rect::new(8, 0, 8, 8)),
            dark: None,
            gains: None,
            demosaic: Demosaic::Bilinear,
            x: &x4,
            y: &y4,
            matrix: &matrix,
            tables: &tables,
        };
        gpu.fused_isp(&crop, &mut out).unwrap();
        assert!(gpu.download(&out).unwrap().iter().all(|&v| v == 120));
        // Binned by 2, 16 × 8 becomes 8 × 4: the left half stays 40, the right 120.
        let (x8, y4b) = (copy_axis(&gpu, 8), copy_axis(&gpu, 4));
        let mut binned = gpu.zeros::<u8>(8 * 4 * 3).unwrap();
        let bin = super::FusedInputs {
            reduce: Reduce::Bin(2),
            x: &x8,
            y: &y4b,
            ..crop
        };
        gpu.fused_isp(&bin, &mut binned).unwrap();
        let values = gpu.download(&binned).unwrap();
        assert_eq!((values[0], values[3 * 7]), (40, 120), "{values:?}");
    }

    #[test]
    fn inputs_that_dont_fit_are_refused() {
        let gpu = crate::cuda_or_skip!();
        let (tables, matrix) = (identity_tables(&gpu), identity_matrix(&gpu));
        let raw = gpu.upload(&[0u16; 64]).unwrap();
        let (x, y) = (copy_axis(&gpu, 8), copy_axis(&gpu, 8));
        let good = super::FusedInputs {
            raw: &raw,
            layout: layout(8, 8),
            reduce: Reduce::Full,
            dark: None,
            gains: None,
            demosaic: Demosaic::MalvarHeCutler,
            x: &x,
            y: &y,
            matrix: &matrix,
            tables: &tables,
        };
        let mut out = gpu.zeros::<u8>(192).unwrap();
        assert!(gpu.fused_isp(&good, &mut out).is_ok());
        let refused = |inputs: &super::FusedInputs, out: &mut DeviceBuffer<u8>| {
            gpu.fused_isp(inputs, out).unwrap_err().to_string()
        };
        assert_eq!(
            refused(
                &super::FusedInputs {
                    layout: layout(8, 7),
                    ..good
                },
                &mut out
            ),
            "a 8 × 7 frame has 56 sites, not 64"
        );
        assert!(
            refused(
                &super::FusedInputs {
                    reduce: Reduce::Crop(Rect::new(4, 0, 8, 8)),
                    ..good
                },
                &mut out
            )
            .ends_with("isn't inside the 8 × 8 frame")
        );
        assert_eq!(
            refused(
                &super::FusedInputs {
                    reduce: Reduce::Bin(4),
                    ..good
                },
                &mut out
            ),
            "MalvarHeCutler needs at least 3 × 3, not 2 × 2"
        );
        // Too short, though wide enough: 16 × 4 binned by 2 is 8 × 2.
        let low = gpu.upload(&[0u16; 64]).unwrap();
        let short_frame = super::FusedInputs {
            raw: &low,
            layout: layout(16, 4),
            reduce: Reduce::Bin(2),
            ..good
        };
        assert_eq!(
            refused(&short_frame, &mut out),
            "MalvarHeCutler needs at least 3 × 3, not 8 × 2"
        );
        assert_eq!(
            refused(
                &super::FusedInputs {
                    reduce: Reduce::Bin(4),
                    demosaic: Demosaic::Bilinear,
                    ..good
                },
                &mut out
            ),
            "the resize takes 8 × 8, but the reduced frame is 2 × 2"
        );
        let short = gpu.upload(&[0u16; 63]).unwrap();
        assert_eq!(
            refused(
                &super::FusedInputs {
                    dark: Some(&short),
                    ..good
                },
                &mut out
            ),
            "the dark holds 63 sites, not the reduced frame's 64"
        );
        assert_eq!(
            refused(
                &super::FusedInputs {
                    gains: Some(&short),
                    ..good
                },
                &mut out
            ),
            "the flat field holds 63 sites, not the reduced frame's 64"
        );
        let few = gpu.upload(&[MATRIX_ONE; 8]).unwrap();
        assert!(
            refused(
                &super::FusedInputs {
                    matrix: &few,
                    ..good
                },
                &mut out
            )
            .starts_with("colour needs 9 matrix entries")
        );
        let small_tables = gpu.upload(&vec![0u8; TABLE_SIZE]).unwrap();
        assert!(
            refused(
                &super::FusedInputs {
                    tables: &small_tables,
                    ..good
                },
                &mut out
            )
            .starts_with("colour needs")
        );
        let mut small = gpu.zeros::<u8>(191).unwrap();
        assert!(gpu.fused_isp(&good, &mut small).is_err());
        let wide = layout(u32::MAX, 1);
        assert!(
            gpu.fused_isp(
                &super::FusedInputs {
                    layout: wide,
                    ..good
                },
                &mut out
            )
            .is_err()
        );
        let empty_axis = gpu.upload_axis(8, 0, 1, &[], &[]).unwrap();
        let mut none = gpu.zeros::<u8>(0).unwrap();
        assert!(
            gpu.fused_isp(
                &super::FusedInputs {
                    x: &empty_axis,
                    ..good
                },
                &mut none
            )
            .is_ok(),
            "nothing to draw"
        );
    }
}
