//! Colour: white-balance gains and a colour-correction matrix, from the camera's RGB to
//! linear sRGB, then levels and a transfer curve to 8 bits.
//!
//! White balance and the matrix are linear, so they are one matrix, `CCM × diag(gains)`, in
//! fixed point, 1/16384ths. Each pixel is that matrix times its RGB, rounded once and clamped
//! to 16 bits, then each channel looked up in its table (P1.05b's display transform). The CPU
//! twin and the CUDA kernel (`tetera-compute`'s `isp.cu`) compute the same integers.

use crate::display::{Curve, DisplayTransform, Levels};
use crate::image::Image;
use rayon::prelude::*;
use tetera_compute::imaging::TABLE_SIZE;
use tetera_compute::isp::MATRIX_ONE;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::{Error, Result};

/// White balance and a colour-correction matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColourCorrection {
    /// Red's, green's and blue's gains, applied first.
    pub white_balance: [f64; 3],
    /// The matrix to linear sRGB, row by row: output red is the first row times the balanced
    /// input.
    pub matrix: [[f64; 3]; 3],
}

impl ColourCorrection {
    /// Nothing changed.
    pub const IDENTITY: Self = Self {
        white_balance: [1.0; 3],
        matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    /// The largest an entry of the combined matrix may be, either way.
    pub const LIMIT: f64 = 64.0;

    /// White balance then the matrix, as one matrix in 1/16384ths, row by row.
    pub fn fixed_point(&self) -> Result<[i32; 9]> {
        let mut out = [0i32; 9];
        for (r, row) in self.matrix.iter().enumerate() {
            for (c, &m) in row.iter().enumerate() {
                let entry = m * self.white_balance[c];
                if !entry.is_finite() || entry.abs() > Self::LIMIT {
                    return Err(Error::invalid_argument(format!(
                        "a colour matrix entry, times its white balance, must be finite and within ±{}, not {entry} (row {r}, column {c})",
                        Self::LIMIT
                    )));
                }
                out[r * 3 + c] = (entry * f64::from(MATRIX_ONE)).round() as i32;
            }
        }
        Ok(out)
    }
}

impl Default for ColourCorrection {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Colour, ready to apply: the matrix in fixed point, and the tables.
#[derive(Debug, Clone, PartialEq)]
pub struct Colour {
    matrix: [i32; 9],
    display: DisplayTransform,
}

impl Colour {
    /// `correction`, then `levels` for red, green and blue, then `curve`, to 8 bits.
    pub fn new(correction: &ColourCorrection, levels: [Levels; 3], curve: Curve) -> Result<Self> {
        Ok(Self {
            matrix: correction.fixed_point()?,
            display: DisplayTransform::new(&levels, curve)?,
        })
    }

    /// The matrix, in 1/16384ths, row by row.
    pub fn matrix(&self) -> &[i32; 9] {
        &self.matrix
    }

    /// The levels and curve, as tables.
    pub fn display(&self) -> &DisplayTransform {
        &self.display
    }
}

fn check(image: &Image<u16>) -> Result<()> {
    if image.channels() != 3 {
        return Err(Error::invalid_argument(format!(
            "colour takes RGB, not {} channels",
            image.channels()
        )));
    }
    Ok(())
}

/// The CPU twin: each pixel through the matrix, then each channel through its table.
pub fn colour_cpu(cpu: &CpuBackend, image: &Image<u16>, colour: &Colour) -> Result<Image<u8>> {
    check(image)?;
    let (m, tables) = (&colour.matrix, colour.display.tables());
    let mut data = vec![0u8; image.data().len()];
    cpu.install(|| {
        data.par_chunks_mut(3 * 1_024)
            .zip(image.data().par_chunks(3 * 1_024))
            .for_each(|(out, rgb)| {
                for (o, p) in out
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(rgb.as_chunks::<3>().0)
                {
                    for c in 0..3 {
                        let sum: i64 = (0..3)
                            .map(|k| i64::from(m[c * 3 + k]) * i64::from(p[k]))
                            .sum();
                        let v = ((sum + 8_192) >> 14).clamp(0, 65_535) as usize;
                        o[c] = tables[c * TABLE_SIZE + v];
                    }
                }
            });
    });
    Image::new(image.width(), image.height(), 3, data)
}

/// Colour, on whichever backend `compute` chose.
pub fn colour(compute: &Compute, image: &Image<u16>, colour: &Colour) -> Result<Image<u8>> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        check(image)?;
        let on_device = crate::DeviceImage::upload(gpu, image)?;
        let tables = DeviceColour::upload(gpu, colour)?;
        return colour_device(gpu, &on_device, &tables)?.download(gpu);
    }
    colour_cpu(compute.cpu(), image, colour)
}

/// Colour's matrix and tables on the device, uploaded once.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeviceColour {
    pub(crate) matrix: tetera_compute::DeviceBuffer<i32>,
    pub(crate) tables: tetera_compute::DeviceBuffer<u8>,
}

#[cfg(feature = "cuda")]
impl DeviceColour {
    /// `colour`'s matrix and tables, copied to `gpu`.
    pub fn upload(gpu: &tetera_compute::CudaBackend, colour: &Colour) -> Result<Self> {
        Ok(Self {
            matrix: gpu.upload(&colour.matrix)?,
            tables: gpu.upload(colour.display.tables())?,
        })
    }
}

/// Colour on the device, the result left there.
#[cfg(feature = "cuda")]
pub fn colour_device(
    gpu: &tetera_compute::CudaBackend,
    image: &crate::DeviceImage<u16>,
    colour: &DeviceColour,
) -> Result<crate::DeviceImage<u8>> {
    if image.channels() != 3 {
        return Err(Error::invalid_argument(format!(
            "colour takes RGB, not {} channels",
            image.channels()
        )));
    }
    let mut out = crate::DeviceImage::zeros(gpu, image.width(), image.height(), 3)?;
    gpu.colour(
        image.buffer(),
        &colour.matrix,
        &colour.tables,
        out.buffer_mut(),
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: Levels = Levels {
        black: 0,
        white: 4_095,
    };

    fn cpu() -> CpuBackend {
        CpuBackend::new(2).unwrap()
    }

    fn linear(levels: [Levels; 3], correction: ColourCorrection) -> Colour {
        Colour::new(&correction, levels, Curve::Linear).unwrap()
    }

    #[test]
    fn white_balance_and_the_matrix_become_one_fixed_point_matrix() {
        assert_eq!(
            ColourCorrection::IDENTITY.fixed_point().unwrap(),
            [16_384, 0, 0, 0, 16_384, 0, 0, 0, 16_384]
        );
        assert_eq!(ColourCorrection::default(), ColourCorrection::IDENTITY);
        let c = ColourCorrection {
            white_balance: [2.0, 1.0, 0.5],
            matrix: [
                [1.5, -0.25, -0.25],
                [-0.125, 1.25, -0.125],
                [0.0, -0.5, 1.5],
            ],
        };
        // Each column scaled by its channel's gain.
        assert_eq!(
            c.fixed_point().unwrap(),
            [
                49_152, -4_096, -2_048, -4_096, 20_480, -1_024, 0, -8_192, 12_288
            ]
        );
        let bad = |m: f64, wb: f64| ColourCorrection {
            white_balance: [wb, 1.0, 1.0],
            matrix: [[m, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let err = bad(65.0, 1.0).fixed_point().unwrap_err();
        assert_eq!(
            err.to_string(),
            "a colour matrix entry, times its white balance, must be finite and within ±64, not 65 (row 0, column 0)"
        );
        assert!(bad(32.0, 3.0).fixed_point().is_err(), "the product counts");
        assert!(bad(-64.0, 1.0).fixed_point().is_ok());
        assert!(bad(f64::NAN, 1.0).fixed_point().is_err());
        assert!(bad(1.0, f64::INFINITY).fixed_point().is_err());
    }

    #[test]
    fn the_identity_is_the_display_transform_alone() {
        let image = crate::pattern::test_pattern(40, 30, 12);
        let colour = Colour::new(&ColourCorrection::IDENTITY, [FULL; 3], Curve::Srgb).unwrap();
        let display = crate::display::display_cpu(&cpu(), &image, colour.display()).unwrap();
        assert_eq!(colour_cpu(&cpu(), &image, &colour).unwrap(), display);
        assert_eq!(colour.matrix()[4], MATRIX_ONE);
    }

    #[test]
    fn pixels_are_mixed_rounded_once_and_clamped() {
        let levels = [Levels {
            black: 0,
            white: 255,
        }; 3];
        // Swap red and blue; green at 1.5, with a quarter of red taken away.
        let swap = ColourCorrection {
            white_balance: [1.0; 3],
            matrix: [[0.0, 0.0, 1.0], [-0.25, 1.5, 0.0], [1.0, 0.0, 0.0]],
        };
        let image = Image::new(2, 1, 3, vec![10u16, 20, 30, 200, 100, 7]).unwrap();
        let out = colour_cpu(&cpu(), &image, &linear(levels, swap)).unwrap();
        // (30, 27.5 → 28, 10) and (7, 100, 200); 255 for anything at or above 255.
        assert_eq!(out.data(), [30, 28, 10, 7, 100, 200]);
        // Below zero clamps to black; beyond 16 bits, to the brightest.
        let wild = ColourCorrection {
            white_balance: [1.0; 3],
            matrix: [[-1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let bright = Image::new(1, 1, 3, vec![30_000u16, 0, 7]).unwrap();
        assert_eq!(
            colour_cpu(&cpu(), &bright, &linear(levels, wild))
                .unwrap()
                .data(),
            [0, 255, 7]
        );
        let grey = Image::new(1, 1, 1, vec![0u16]).unwrap();
        let err =
            colour_cpu(&cpu(), &grey, &linear(levels, ColourCorrection::IDENTITY)).unwrap_err();
        assert_eq!(err.to_string(), "colour takes RGB, not 1 channels");
    }

    #[test]
    fn each_channel_goes_through_its_own_table() {
        let levels = [
            Levels {
                black: 0,
                white: 1_000,
            },
            Levels {
                black: 0,
                white: 2_000,
            },
            Levels {
                black: 0,
                white: 4_000,
            },
        ];
        let image = Image::new(1, 1, 3, vec![500u16, 500, 500]).unwrap();
        let out = colour_cpu(&cpu(), &image, &linear(levels, ColourCorrection::IDENTITY)).unwrap();
        // Half, a quarter and an eighth of white: 127.5 → 128, 63.75 → 64, 31.875 → 32.
        assert_eq!(out.data(), [128, 64, 32]);
    }

    #[test]
    fn white_balance_evens_out_a_grey_seen_through_unequal_channels() {
        // A grey that the camera sees as (2,000, 4,000, 1,000): gains of 2, 1 and 4 make it grey.
        let seen = Image::new(1, 1, 3, vec![2_000u16, 4_000, 1_000]).unwrap();
        let balanced = ColourCorrection {
            white_balance: [2.0, 1.0, 4.0],
            ..ColourCorrection::IDENTITY
        };
        let out = colour_cpu(
            &cpu(),
            &seen,
            &linear(
                [Levels {
                    black: 0,
                    white: 8_000,
                }; 3],
                balanced,
            ),
        )
        .unwrap();
        assert_eq!(out.data(), [128, 128, 128]);
    }

    #[test]
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let image = crate::pattern::test_pattern(20, 12, 12);
        let c = Colour::new(
            &ColourCorrection {
                white_balance: [1.2, 1.0, 0.9],
                ..ColourCorrection::IDENTITY
            },
            [FULL; 3],
            Curve::Srgb,
        )
        .unwrap();
        assert_eq!(
            colour(&compute, &image, &c).unwrap(),
            colour_cpu(compute.cpu(), &image, &c).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernel_and_its_twin_agree_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let c = Colour::new(
            &ColourCorrection {
                white_balance: [1.8, 1.0, 1.4],
                matrix: [[1.6, -0.4, -0.2], [-0.3, 1.5, -0.2], [-0.1, -0.5, 1.6]],
            },
            [
                Levels {
                    black: 64,
                    white: 4_000,
                },
                FULL,
                Levels {
                    black: 10,
                    white: 3_900,
                },
            ],
            Curve::Srgb,
        )
        .unwrap();
        for (w, h) in [(1, 1), (33, 17), (1_920, 1_280)] {
            let image = crate::pattern::test_pattern(w, h, 12);
            assert_eq!(
                colour(&compute, &image, &c).unwrap(),
                colour_cpu(&CpuBackend::new(0).unwrap(), &image, &c).unwrap(),
                "{w} × {h}"
            );
        }
        let grey =
            crate::DeviceImage::upload(&gpu, &Image::new(1, 1, 1, vec![0u16]).unwrap()).unwrap();
        assert!(colour_device(&gpu, &grey, &DeviceColour::upload(&gpu, &c).unwrap()).is_err());
    }
}
