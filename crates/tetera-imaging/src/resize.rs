//! Resizing an image of 16-bit values to any size: area averaging to shrink, bilinear to
//! enlarge, axis by axis.
//!
//! Each axis is a table, made here once for a pair of sizes: for each output column (or row),
//! the first input it reads and a window of weights in 1/16384ths that sum to 16384. A pixel is
//! the sum of its window's values times both axes' weights, which sum to 2^28, rounded once.
//! Integer weights, made on the host and shared with the GPU, make the CPU twin and the CUDA
//! kernel (`tetera-compute`'s `resize.cu`) agree exactly.
//!
//! **Area averaging:** an output covers `input / output` inputs, and each input weighs the
//! share of it that falls inside. **Bilinear:** an output's centre, mapped onto the input,
//! falls between two inputs, which it weighs by nearness; at the edges, it takes the edge.
//! Equal sizes copy.

use crate::image::Image;
use rayon::prelude::*;
use tetera_compute::imaging::WEIGHT_ONE;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::{Error, Result};

/// One axis of a resize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResizeAxis {
    input: u32,
    output: u32,
    taps: u32,
    start: Vec<i32>,
    weights: Vec<u16>,
}

/// An output's weights before they are placed in a window: the first input, and a weight for
/// each input from it.
type Weights = (u32, Vec<u32>);

/// Weights in 1/16384ths that sum to 16384 exactly: the rounding's excess or shortfall goes to
/// the largest, the first of equals.
fn normalise(weights: &mut [u32]) {
    let total: u32 = weights.iter().sum();
    let one = u32::from(WEIGHT_ONE);
    let largest = (0..weights.len()).fold(
        0,
        |best, i| if weights[i] > weights[best] { i } else { best },
    );
    weights[largest] = (weights[largest] + one).saturating_sub(total);
}

/// Area averaging: output `j` covers inputs `j × input / output` to `(j + 1) × input /
/// output`. In units of 1/output of an input, input `i` spans `i × output` to `(i + 1) ×
/// output`, and output `j` spans `j × input` to `(j + 1) × input`: their overlap, over
/// `input`, is the weight.
fn area(input: u32, output: u32, j: u32) -> Weights {
    let (input, output, j) = (u64::from(input), u64::from(output), u64::from(j));
    let (lo, hi) = (j * input, (j + 1) * input);
    let (first, last) = (lo / output, (hi - 1) / output);
    let mut weights: Vec<u32> = (first..=last)
        .map(|i| {
            let overlap = hi.min((i + 1) * output) - lo.max(i * output);
            ((overlap * u64::from(WEIGHT_ONE) + input / 2) / input) as u32
        })
        .collect();
    normalise(&mut weights);
    (first as u32, weights)
}

/// Bilinear: output `j`'s centre falls at input `(2j + 1) × input / (2 × output) − ½`,
/// between two inputs, each weighed by its nearness. Before the first input's centre, or after
/// the last's, it takes that input alone.
fn bilinear(input: u32, output: u32, j: u32) -> Weights {
    let (input, output, j) = (i64::from(input), i64::from(output), i64::from(j));
    let (numerator, denominator) = ((2 * j + 1) * input - output, 2 * output);
    if numerator <= 0 {
        return (0, vec![u32::from(WEIGHT_ONE)]);
    }
    let (i0, remainder) = (numerator / denominator, numerator % denominator);
    if i0 >= input - 1 {
        return ((input - 1) as u32, vec![u32::from(WEIGHT_ONE)]);
    }
    let far = ((remainder * i64::from(WEIGHT_ONE) + denominator / 2) / denominator) as u32;
    if far == 0 {
        return (i0 as u32, vec![u32::from(WEIGHT_ONE)]);
    }
    (i0 as u32, vec![u32::from(WEIGHT_ONE) - far, far])
}

impl ResizeAxis {
    /// The axis from `input` to `output` columns (or rows), both at least 1: area averaging
    /// when it shrinks, bilinear otherwise.
    pub fn new(input: u32, output: u32) -> Result<Self> {
        if input == 0 || output == 0 {
            return Err(Error::invalid_argument(format!(
                "a resize needs at least one pixel each way, not {input} to {output}"
            )));
        }
        let each: Vec<Weights> = (0..output)
            .map(|j| {
                if output < input {
                    area(input, output, j)
                } else {
                    bilinear(input, output, j)
                }
            })
            .collect();
        let taps = each
            .iter()
            .map(|(_, w)| w.len() as u32)
            .max()
            .unwrap_or(1)
            .min(input);
        let mut start = Vec::with_capacity(output as usize);
        let mut weights = vec![0u16; output as usize * taps as usize];
        for (j, (first, own)) in each.iter().enumerate() {
            // The window holds the output's own weights, moved left at the right edge.
            let window = (*first).min(input - taps);
            start.push(window as i32);
            let offset = (first - window) as usize;
            for (k, &w) in own.iter().enumerate() {
                weights[j * taps as usize + offset + k] = w as u16;
            }
        }
        Ok(Self {
            input,
            output,
            taps,
            start,
            weights,
        })
    }

    /// Columns or rows read.
    pub fn input(&self) -> u32 {
        self.input
    }

    /// Columns or rows written.
    pub fn output(&self) -> u32 {
        self.output
    }

    /// Weights for each output.
    pub fn taps(&self) -> u32 {
        self.taps
    }

    /// Each output's first input.
    pub fn start(&self) -> &[i32] {
        &self.start
    }

    /// Each output's `taps` weights, one output after another.
    pub fn weights(&self) -> &[u16] {
        &self.weights
    }

    /// Output `j`'s first input and weights.
    fn window(&self, j: usize) -> (usize, &[u16]) {
        let taps = self.taps as usize;
        (
            self.start[j] as usize,
            &self.weights[j * taps..(j + 1) * taps],
        )
    }
}

/// A resize from one size to another: an axis across and one down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resize {
    /// Across.
    pub x: ResizeAxis,
    /// Down.
    pub y: ResizeAxis,
}

impl Resize {
    /// The resize from `width × height` to `out_width × out_height`.
    pub fn new(width: u32, height: u32, out_width: u32, out_height: u32) -> Result<Self> {
        Ok(Self {
            x: ResizeAxis::new(width, out_width)?,
            y: ResizeAxis::new(height, out_height)?,
        })
    }

    fn check(&self, width: u32, height: u32) -> Result<()> {
        if (width, height) != (self.x.input, self.y.input) {
            return Err(Error::invalid_argument(format!(
                "a resize from {} × {} can't take a {width} × {height} image",
                self.x.input, self.y.input
            )));
        }
        Ok(())
    }
}

/// The CPU twin: `image` resized by `plan`.
pub fn resize_cpu(cpu: &CpuBackend, image: &Image<u16>, plan: &Resize) -> Result<Image<u16>> {
    plan.check(image.width(), image.height())?;
    let (c, in_width, out_width) = (
        usize::from(image.channels()),
        image.width() as usize,
        plan.x.output as usize,
    );
    let mut data = vec![0u16; out_width * plan.y.output as usize * c];
    cpu.install(|| {
        data.par_chunks_mut(out_width * c)
            .enumerate()
            .for_each(|(y, row)| {
                let (ys, wy) = plan.y.window(y);
                for (x, pixel) in row.chunks_mut(c).enumerate() {
                    let (xs, wx) = plan.x.window(x);
                    for (channel, value) in pixel.iter_mut().enumerate() {
                        let mut sum = 0u64;
                        for (a, &weight_y) in wy.iter().enumerate() {
                            let row = &image.data()[((ys + a) * in_width + xs) * c..];
                            let across: u64 = wx
                                .iter()
                                .enumerate()
                                .map(|(b, &weight_x)| {
                                    u64::from(weight_x) * u64::from(row[b * c + channel])
                                })
                                .sum();
                            sum += across * u64::from(weight_y);
                        }
                        *value = ((sum + (1 << 27)) >> 28) as u16;
                    }
                }
            });
    });
    Image::new(plan.x.output, plan.y.output, image.channels(), data)
}

/// `image` resized to `width × height`, on whichever backend `compute` chose.
pub fn resize(
    compute: &Compute,
    image: &Image<u16>,
    width: u32,
    height: u32,
) -> Result<Image<u16>> {
    let plan = Resize::new(image.width(), image.height(), width, height)?;
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let on_device = crate::DeviceImage::upload(gpu, image)?;
        let plan = DeviceResize::upload(gpu, &plan)?;
        return resize_device(gpu, &on_device, &plan)?.download(gpu);
    }
    resize_cpu(compute.cpu(), image, &plan)
}

/// A resize's tables on the device, uploaded once for a pair of sizes.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeviceResize {
    pub(crate) x: tetera_compute::imaging::DeviceAxis,
    pub(crate) y: tetera_compute::imaging::DeviceAxis,
}

#[cfg(feature = "cuda")]
impl DeviceResize {
    /// `plan`'s tables, copied to `gpu`.
    pub fn upload(gpu: &tetera_compute::CudaBackend, plan: &Resize) -> Result<Self> {
        let axis =
            |a: &ResizeAxis| gpu.upload_axis(a.input, a.output, a.taps, &a.start, &a.weights);
        Ok(Self {
            x: axis(&plan.x)?,
            y: axis(&plan.y)?,
        })
    }
}

/// Resizing an image on the device by `plan`, the result left there.
#[cfg(feature = "cuda")]
pub fn resize_device(
    gpu: &tetera_compute::CudaBackend,
    image: &crate::DeviceImage<u16>,
    plan: &DeviceResize,
) -> Result<crate::DeviceImage<u16>> {
    let mut out =
        crate::DeviceImage::zeros(gpu, plan.x.output(), plan.y.output(), image.channels())?;
    gpu.resize(
        image.buffer(),
        image.channels(),
        &plan.x,
        &plan.y,
        out.buffer_mut(),
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> CpuBackend {
        CpuBackend::new(3).unwrap()
    }

    /// Each output's weights, as they apply to each input: a dense matrix, output by input.
    fn dense(axis: &ResizeAxis) -> Vec<Vec<u32>> {
        (0..axis.output as usize)
            .map(|j| {
                let (start, weights) = axis.window(j);
                let mut row = vec![0u32; axis.input as usize];
                for (k, &w) in weights.iter().enumerate() {
                    row[start + k] += u32::from(w);
                }
                row
            })
            .collect()
    }

    #[test]
    fn every_outputs_weights_sum_to_one_and_stay_inside_the_input() {
        for (input, output) in [
            (1, 1),
            (1, 7),
            (7, 1),
            (5_472, 1_920),
            (1_920, 5_472),
            (100, 33),
            (33, 100),
            (2, 3),
            (3, 2),
            (640, 640),
        ] {
            let axis = ResizeAxis::new(input, output).unwrap();
            assert_eq!(axis.start().len(), output as usize);
            assert_eq!(axis.weights().len(), (output * axis.taps()) as usize);
            assert!(axis.taps() <= input);
            for j in 0..output as usize {
                let (start, weights) = axis.window(j);
                assert!(
                    start + weights.len() <= input as usize,
                    "{input} → {output}, output {j}"
                );
                let total: u32 = weights.iter().map(|&w| u32::from(w)).sum();
                assert_eq!(
                    total,
                    u32::from(WEIGHT_ONE),
                    "{input} → {output}, output {j}"
                );
            }
        }
    }

    #[test]
    fn shrinking_averages_the_area_each_output_covers() {
        // 6 to 2: each output covers three inputs, a third each.
        let axis = ResizeAxis::new(6, 2).unwrap();
        // 16,383 in all after rounding: the first of the equal weights takes the last 1/16384.
        assert_eq!(
            dense(&axis),
            [
                vec![5_462, 5_461, 5_461, 0, 0, 0],
                vec![0, 0, 0, 5_462, 5_461, 5_461]
            ]
        );
        // 3 to 2: each output covers an input and a half.
        let axis = ResizeAxis::new(3, 2).unwrap();
        assert_eq!(
            dense(&axis),
            [vec![10_923, 5_461, 0], vec![0, 5_461, 10_923]]
        );
        // 7 to 3: each output covers 2⅓ inputs. Rounded, the first output's weights sum to
        // 16,385: the largest, the first of the two equal ones, gives the 1/16384 back.
        let axis = ResizeAxis::new(7, 3).unwrap();
        assert_eq!((axis.input(), axis.output(), axis.taps()), (7, 3, 3));
        assert_eq!(
            dense(&axis),
            [
                vec![7_021, 7_022, 2_341, 0, 0, 0, 0],
                vec![0, 0, 4_681, 7_022, 4_681, 0, 0],
                vec![0, 0, 0, 0, 2_341, 7_021, 7_022],
            ]
        );
        // 4 to 1: all four, a quarter each; the window is the whole input.
        assert_eq!(dense(&ResizeAxis::new(4, 1).unwrap()), [vec![4_096; 4]]);
    }

    #[test]
    fn enlarging_weighs_the_two_nearest_by_nearness() {
        // 2 to 4: centres at input −¼ (the edge), ¼, ¾ and 1¼ (the edge).
        let axis = ResizeAxis::new(2, 4).unwrap();
        assert_eq!(
            dense(&axis),
            [
                vec![16_384, 0],
                vec![12_288, 4_096],
                vec![4_096, 12_288],
                vec![0, 16_384]
            ]
        );
        // 3 to 5: the second output's centre falls 0.4 past the first input: 6,553.6 of
        // 16,384 for the second, which rounds to the nearest, 6,554.
        assert_eq!(
            dense(&ResizeAxis::new(3, 5).unwrap()),
            [
                vec![16_384, 0, 0],
                vec![9_830, 6_554, 0],
                vec![0, 16_384, 0],
                vec![0, 6_554, 9_830],
                vec![0, 0, 16_384],
            ]
        );
        // Equal sizes copy.
        let same = ResizeAxis::new(5, 5).unwrap();
        for (j, row) in dense(&same).iter().enumerate() {
            assert_eq!(row[j], 16_384, "output {j}");
        }
        assert_eq!(same.taps(), 1, "every output falls on an input");
        // One input fills every output.
        assert_eq!(
            dense(&ResizeAxis::new(1, 3).unwrap()),
            vec![vec![16_384]; 3]
        );
    }

    #[test]
    fn a_flat_image_stays_flat_and_a_ramp_stays_a_ramp() {
        let flat = Image::new(37, 23, 3, [1_000u16, 2_000, 3_000].repeat(37 * 23)).unwrap();
        for (w, h) in [(10, 7), (80, 50), (37, 23), (1, 1)] {
            let out = resize_cpu(&cpu(), &flat, &Resize::new(37, 23, w, h).unwrap()).unwrap();
            assert!(
                out.data().chunks(3).all(|p| p == [1_000, 2_000, 3_000]),
                "{w} × {h}"
            );
        }
        // A ramp along x: halving averages neighbouring pairs, exactly.
        let ramp = Image::new(8, 1, 1, (0..8u16).map(|x| 100 * x).collect()).unwrap();
        let half = resize_cpu(&cpu(), &ramp, &Resize::new(8, 1, 4, 1).unwrap()).unwrap();
        assert_eq!(half.data(), [50, 250, 450, 650]);
        let double = resize_cpu(&cpu(), &ramp, &Resize::new(8, 1, 16, 1).unwrap()).unwrap();
        // Between the edges, each output falls a quarter from an input: 25 on either side.
        assert_eq!(
            &double.data()[1..15],
            [
                25, 75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575, 625, 675
            ]
        );
        assert_eq!(
            (double.data()[0], double.data()[15]),
            (0, 700),
            "the edges are kept"
        );
        let same = resize_cpu(&cpu(), &ramp, &Resize::new(8, 1, 8, 1).unwrap()).unwrap();
        assert_eq!(same, ramp);
        // A quarter each of 1, 2, 3 and 4 is 2.5, which rounds up.
        let four = Image::new(4, 1, 1, vec![1u16, 2, 3, 4]).unwrap();
        assert_eq!(
            resize_cpu(&cpu(), &four, &Resize::new(4, 1, 1, 1).unwrap())
                .unwrap()
                .data(),
            [3]
        );
    }

    #[test]
    fn each_channel_is_resized_from_its_own_values_on_every_row() {
        // 4 × 2 of two channels, the second ten times the first, to 2 × 1: each output
        // averages a 2 × 2 block of its own channel.
        let values: Vec<u16> = (0..8u16).flat_map(|v| [v, 10 * v]).collect();
        let image = Image::new(4, 2, 2, values).unwrap();
        let out = resize_cpu(&cpu(), &image, &Resize::new(4, 2, 2, 1).unwrap()).unwrap();
        // The first block holds 0, 1, 4 and 5: their mean, 2.5, rounds up to 3; the second's
        // channel, 25, is exact.
        assert_eq!(out.data(), [3, 25, 5, 45]);
        assert_eq!((out.width(), out.height(), out.channels()), (2, 1, 2));
    }

    #[test]
    fn the_brightest_value_survives_rounding() {
        let white = Image::new(5, 3, 1, vec![u16::MAX; 15]).unwrap();
        for (w, h) in [(2, 2), (9, 4), (5, 3)] {
            let out = resize_cpu(&cpu(), &white, &Resize::new(5, 3, w, h).unwrap()).unwrap();
            assert!(out.data().iter().all(|&v| v == u16::MAX), "{w} × {h}");
        }
    }

    #[test]
    fn sizes_that_dont_fit_are_refused() {
        let err = ResizeAxis::new(0, 4).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a resize needs at least one pixel each way, not 0 to 4"
        );
        assert!(ResizeAxis::new(4, 0).is_err());
        let image = Image::new(4, 4, 1, vec![0u16; 16]).unwrap();
        let err = resize_cpu(&cpu(), &image, &Resize::new(4, 5, 2, 2).unwrap()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a resize from 4 × 5 can't take a 4 × 4 image"
        );
        assert!(resize_cpu(&cpu(), &image, &Resize::new(5, 4, 2, 2).unwrap()).is_err());
    }

    #[test]
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let image = crate::pattern::test_pattern(30, 20, 12);
        let plan = Resize::new(30, 20, 12, 9).unwrap();
        assert_eq!(
            resize(&compute, &image, 12, 9).unwrap(),
            resize_cpu(compute.cpu(), &image, &plan).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernel_and_its_twin_agree_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = CpuBackend::new(0).unwrap();
        for ((w, h), (ow, oh)) in [
            ((33, 17), (16, 9)),
            ((33, 17), (70, 40)),
            ((5, 3), (5, 3)),
            ((1, 1), (3, 2)),
            ((2_736, 1_824), (1_920, 1_280)),
        ] {
            let image = crate::pattern::test_pattern(w, h, 16);
            let plan = Resize::new(w, h, ow, oh).unwrap();
            let on_cpu = resize_cpu(&cpu, &image, &plan).unwrap();
            assert_eq!(
                resize(&compute, &image, ow, oh).unwrap(),
                on_cpu,
                "{w} × {h} → {ow} × {oh}"
            );
            let on_device = crate::DeviceImage::upload(&gpu, &image).unwrap();
            let tables = DeviceResize::upload(&gpu, &plan).unwrap();
            assert_eq!(
                resize_device(&gpu, &on_device, &tables)
                    .unwrap()
                    .download(&gpu)
                    .unwrap(),
                on_cpu
            );
        }
    }
}
