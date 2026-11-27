//! Denoising (P1.07a; the behaviour spec is in its notes): temporal averaging over a window
//! of frames, reset when the stage moves; and the bilateral filter (Tomasi and Manduchi, 1998),
//! the stock edge-preserving spatial filter.
//!
//! Both are integer arithmetic: a 32-bit sum over the window, and bilateral weights from
//! fixed-point tables made here once. So the CPU twins and the CUDA kernels
//! (`tetera-compute`'s `denoise.cu`) agree exactly.

use crate::image::{Image, Mosaic, MosaicLayout, mirror};
use rayon::prelude::*;
use std::collections::VecDeque;
use tetera_compute::denoise::{BILATERAL_ONE, MAX_RADIUS, MAX_WINDOW};
use tetera_compute::imaging::TABLE_SIZE;
use tetera_compute::{Compute, CpuBackend};
use tetera_core::{Error, Result};

/// The mean of the last frames, on the CPU: the exact mean of the most recent `min(n,
/// window)` frames since the window was last reset.
#[derive(Debug, Clone)]
pub struct TemporalMean {
    window: usize,
    layout: Option<MosaicLayout>,
    frames: VecDeque<Vec<u16>>,
    sum: Vec<u32>,
}

fn check_window(window: u32) -> Result<()> {
    if !(1..=MAX_WINDOW).contains(&window) {
        return Err(Error::invalid_argument(format!(
            "a temporal window holds 1 to {MAX_WINDOW} frames, not {window}"
        )));
    }
    Ok(())
}

impl TemporalMean {
    /// An empty window of up to `window` frames, 1 to 64.
    pub fn new(window: u32) -> Result<Self> {
        check_window(window)?;
        Ok(Self {
            window: window as usize,
            layout: None,
            frames: VecDeque::new(),
            sum: Vec::new(),
        })
    }

    /// The frames it holds now.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Empties it: the next frame starts a new window.
    pub fn reset(&mut self) {
        self.frames.clear();
        self.sum.clear();
        self.layout = None;
    }

    /// `frame` into the window, and the window's mean. A move of the stage, or a frame of
    /// another layout, resets the window first, so the mean never blends two scenes.
    pub fn push_cpu(&mut self, cpu: &CpuBackend, frame: &Mosaic, moved: bool) -> Result<Mosaic> {
        if moved || self.layout != Some(*frame.layout()) {
            self.reset();
            self.layout = Some(*frame.layout());
            self.sum = vec![0; frame.sites().len()];
        }
        let oldest = if self.frames.len() == self.window {
            self.frames.pop_front()
        } else {
            None
        };
        let n = (self.frames.len() + 1) as u32;
        let mut mean = vec![0u16; frame.sites().len()];
        cpu.install(|| {
            let oldest = oldest.as_deref();
            mean.par_iter_mut()
                .zip(self.sum.par_iter_mut())
                .zip(frame.sites().par_iter())
                .enumerate()
                .for_each(|(i, ((m, s), &v))| {
                    *s = *s + u32::from(v) - oldest.map_or(0, |o| u32::from(o[i]));
                    *m = ((*s + n / 2) / n) as u16;
                });
        });
        self.frames.push_back(frame.sites().to_vec());
        Mosaic::new(*frame.layout(), mean)
    }
}

/// The mean of the last frames on the device, as [`TemporalMean`] keeps it on the CPU.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeviceTemporalMean {
    window: u32,
    state: Option<(MosaicLayout, tetera_compute::denoise::DeviceTemporal)>,
}

#[cfg(feature = "cuda")]
impl DeviceTemporalMean {
    /// An empty window of up to `window` frames, 1 to 64.
    pub fn new(window: u32) -> Result<Self> {
        check_window(window)?;
        Ok(Self {
            window,
            state: None,
        })
    }

    /// The frames it holds now.
    pub fn len(&self) -> usize {
        self.state.as_ref().map_or(0, |(_, t)| t.len() as usize)
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `frame` into the window, and the window's mean, left on the device; reset first by a
    /// move or a frame of another layout.
    pub fn push(
        &mut self,
        gpu: &tetera_compute::CudaBackend,
        frame: &crate::DeviceMosaic,
        moved: bool,
    ) -> Result<crate::DeviceMosaic> {
        let layout = *frame.layout();
        let fresh = match &mut self.state {
            Some((l, temporal)) if *l == layout && !moved => temporal,
            state => {
                *state = Some((
                    layout,
                    tetera_compute::denoise::DeviceTemporal::new(gpu, self.window, layout.sites())?,
                ));
                &mut state.as_mut().expect("just set").1
            }
        };
        let mut mean = gpu.zeros::<u16>(layout.sites())?;
        fresh.push(gpu, frame.buffer(), &mut mean)?;
        Ok(crate::DeviceMosaic::from_parts(layout, mean))
    }
}

/// The bilateral filter's parameters and tables.
#[derive(Debug, Clone, PartialEq)]
pub struct Bilateral {
    sigma_spatial: f64,
    sigma_range: f64,
    radius: u32,
    spatial: Vec<u16>,
    range: Vec<u16>,
}

impl Bilateral {
    /// The filter with spatial σ `sigma_spatial`, in pixels, above 0 and at most 3.5 (so its
    /// radius, ⌈2σ⌉, is at most 7); and range σ `sigma_range`, in the image's units, above 0.
    pub fn new(sigma_spatial: f64, sigma_range: f64) -> Result<Self> {
        if !(sigma_spatial > 0.0 && sigma_spatial <= f64::from(MAX_RADIUS) / 2.0)
            || !(sigma_range > 0.0 && sigma_range.is_finite())
        {
            return Err(Error::invalid_argument(format!(
                "a bilateral filter takes a spatial σ above 0 and at most {}, and a range σ above 0, not {sigma_spatial} and {sigma_range}",
                f64::from(MAX_RADIUS) / 2.0
            )));
        }
        let radius = (2.0 * sigma_spatial).ceil() as u32;
        let one = f64::from(BILATERAL_ONE);
        let r = radius as i32;
        let spatial = (-r..=r)
            .flat_map(|dy| (-r..=r).map(move |dx| f64::from(dx * dx + dy * dy)))
            .map(|d2| (one * (-d2 / (2.0 * sigma_spatial * sigma_spatial)).exp()).round() as u16)
            .collect();
        let range = (0..TABLE_SIZE)
            .map(|d| {
                (one * (-(d as f64).powi(2) / (2.0 * sigma_range * sigma_range)).exp()).round()
                    as u16
            })
            .collect();
        Ok(Self {
            sigma_spatial,
            sigma_range,
            radius,
            spatial,
            range,
        })
    }

    /// Its radius, ⌈2 σ_s⌉.
    pub fn radius(&self) -> u32 {
        self.radius
    }

    /// Its spatial weights, `(2 × radius + 1)²` in 1/32768ths, row by row.
    pub fn spatial(&self) -> &[u16] {
        &self.spatial
    }

    /// Its range weights, one for each difference from 0 to 65,535, in 1/32768ths.
    pub fn range(&self) -> &[u16] {
        &self.range
    }

    /// Its σs, spatial and range.
    pub fn sigmas(&self) -> (f64, f64) {
        (self.sigma_spatial, self.sigma_range)
    }

    fn check(&self, image: &Image<u16>) -> Result<()> {
        if !matches!(image.channels(), 1 | 3) {
            return Err(Error::invalid_argument(format!(
                "the bilateral filter takes grey or RGB, not {} channels",
                image.channels()
            )));
        }
        if image.width() <= self.radius || image.height() <= self.radius {
            return Err(Error::invalid_argument(format!(
                "a bilateral filter of radius {} needs an image larger than that, not {} × {}",
                self.radius,
                image.width(),
                image.height()
            )));
        }
        Ok(())
    }
}

/// The CPU twin of the bilateral filter.
pub fn bilateral_cpu(
    cpu: &CpuBackend,
    image: &Image<u16>,
    filter: &Bilateral,
) -> Result<Image<u16>> {
    filter.check(image)?;
    let (w, h, c) = (
        image.width() as i32,
        image.height() as i32,
        usize::from(image.channels()),
    );
    let r = filter.radius as i32;
    let side = (2 * r + 1) as usize;
    let data = image.data();
    let mut out = vec![0u16; data.len()];
    cpu.install(|| {
        out.par_chunks_mut(w as usize * c)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as i32;
                for x in 0..w {
                    let at = |px: i32, py: i32| {
                        &data[((mirror(py, h) * w + mirror(px, w)) as usize) * c..][..c]
                    };
                    let centre = at(x, y);
                    let (mut acc, mut total) = ([0u64; 3], 0u64);
                    for ny in y - r..=y + r {
                        for nx in x - r..=x + r {
                            let p = at(nx, ny);
                            let d = if c == 3 {
                                (p[0].abs_diff(centre[0]) as u32
                                    + 2 * p[1].abs_diff(centre[1]) as u32
                                    + p[2].abs_diff(centre[2]) as u32)
                                    / 4
                            } else {
                                u32::from(p[0].abs_diff(centre[0]))
                            };
                            let ws = filter.spatial
                                [(ny - y + r) as usize * side + (nx - x + r) as usize];
                            let w = u64::from(ws) * u64::from(filter.range[d as usize]);
                            total += w;
                            for (a, &v) in acc.iter_mut().zip(p) {
                                *a += w * u64::from(v);
                            }
                        }
                    }
                    for (k, &a) in acc.iter().take(c).enumerate() {
                        row[x as usize * c + k] = ((a + total / 2) / total) as u16;
                    }
                }
            });
    });
    Image::new(image.width(), image.height(), image.channels(), out)
}

/// The bilateral filter, on whichever backend `compute` chose.
pub fn bilateral(compute: &Compute, image: &Image<u16>, filter: &Bilateral) -> Result<Image<u16>> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        filter.check(image)?;
        let on_device = crate::DeviceImage::upload(gpu, image)?;
        let tables = DeviceBilateral::upload(gpu, filter)?;
        return bilateral_device(gpu, &on_device, &tables)?.download(gpu);
    }
    bilateral_cpu(compute.cpu(), image, filter)
}

/// The bilateral filter's tables on the device, uploaded once.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DeviceBilateral(tetera_compute::denoise::BilateralTables);

#[cfg(feature = "cuda")]
impl DeviceBilateral {
    /// `filter`'s tables, copied to `gpu`.
    pub fn upload(gpu: &tetera_compute::CudaBackend, filter: &Bilateral) -> Result<Self> {
        Ok(Self(gpu.upload_bilateral(
            filter.radius,
            &filter.spatial,
            &filter.range,
        )?))
    }
}

/// The bilateral filter on the device, the result left there.
#[cfg(feature = "cuda")]
pub fn bilateral_device(
    gpu: &tetera_compute::CudaBackend,
    image: &crate::DeviceImage<u16>,
    filter: &DeviceBilateral,
) -> Result<crate::DeviceImage<u16>> {
    let mut out = crate::DeviceImage::zeros(gpu, image.width(), image.height(), image.channels())?;
    gpu.bilateral(
        image.buffer(),
        (image.width(), image.height(), image.channels()),
        &filter.0,
        out.buffer_mut(),
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::{noisy, psnr, test_pattern};
    use tetera_core::frame::format::BayerPattern;

    fn cpu() -> CpuBackend {
        CpuBackend::new(3).unwrap()
    }

    fn frame(values: &[u16]) -> Mosaic {
        let layout = MosaicLayout {
            width: values.len() as u32,
            height: 1,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        };
        Mosaic::new(layout, values.to_vec()).unwrap()
    }

    #[test]
    fn the_window_is_the_exact_mean_of_its_last_frames() {
        let mut t = TemporalMean::new(3).unwrap();
        assert!(t.is_empty());
        let push = |t: &mut TemporalMean, v: [u16; 2]| {
            t.push_cpu(&cpu(), &frame(&v), false)
                .unwrap()
                .sites()
                .to_vec()
        };
        assert_eq!(push(&mut t, [10, 100]), [10, 100]);
        assert_eq!(push(&mut t, [20, 101]), [15, 101], "100.5 rounds up");
        assert_eq!(push(&mut t, [30, 102]), [20, 101]);
        assert_eq!(push(&mut t, [40, 103]), [30, 102], "10 has left the window");
        assert_eq!(push(&mut t, [50, 0]), [40, 68]);
        assert_eq!((t.len(), t.is_empty()), (3, false));
    }

    #[test]
    fn a_move_or_a_new_layout_starts_the_window_again() {
        let mut t = TemporalMean::new(4).unwrap();
        t.push_cpu(&cpu(), &frame(&[100, 100]), false).unwrap();
        t.push_cpu(&cpu(), &frame(&[200, 200]), false).unwrap();
        let after_move = t.push_cpu(&cpu(), &frame(&[7, 9]), true).unwrap();
        assert_eq!(after_move.sites(), [7, 9], "the new scene alone");
        assert_eq!(t.len(), 1);
        let wider = t.push_cpu(&cpu(), &frame(&[1, 2, 3]), false).unwrap();
        assert_eq!((wider.sites(), t.len()), ([1u16, 2, 3].as_slice(), 1));
        t.reset();
        assert!(t.is_empty());
        assert!(TemporalMean::new(0).is_err() && TemporalMean::new(65).is_err());
        assert!(TemporalMean::new(64).is_ok());
    }

    #[test]
    fn averaging_lowers_the_noise_as_the_square_root_of_the_frames() {
        let clean = vec![2_000u16; 64 * 64];
        let mut t = TemporalMean::new(16).unwrap();
        let mut last = None;
        for seed in 0..16 {
            last = Some(
                t.push_cpu(&cpu(), &frame(&noisy(&clean, 40.0, 4_095, seed)), false)
                    .unwrap(),
            );
        }
        let sd = |v: &[u16]| {
            (v.iter()
                .map(|&x| (f64::from(x) - 2_000.0).powi(2))
                .sum::<f64>()
                / v.len() as f64)
                .sqrt()
        };
        let (one, sixteen) = (
            sd(&noisy(&clean, 40.0, 4_095, 0)),
            sd(last.unwrap().sites()),
        );
        assert!(
            (one / sixteen - 4.0).abs() < 0.6,
            "σ {one:.1} to {sixteen:.1}: about 4 times less"
        );
    }

    #[test]
    fn the_tables_fall_away_from_the_centre_and_with_the_difference() {
        let f = Bilateral::new(1.5, 100.0).unwrap();
        assert_eq!((f.radius(), f.sigmas()), (3, (1.5, 100.0)));
        let side = 7;
        assert_eq!(f.spatial().len(), side * side);
        assert_eq!(
            f.spatial()[side * side / 2],
            BILATERAL_ONE,
            "the centre weighs 1"
        );
        for dy in 0..side {
            for dx in 0..side {
                assert_eq!(
                    f.spatial()[dy * side + dx],
                    f.spatial()[(side - 1 - dy) * side + (side - 1 - dx)],
                    "symmetric"
                );
            }
        }
        assert!(f.spatial()[0] < f.spatial()[1] && f.spatial()[1] < f.spatial()[side * side / 2]);
        assert_eq!(f.range()[0], BILATERAL_ONE);
        assert!(f.range().windows(2).all(|w| w[0] >= w[1]), "never rises");
        // exp(−½) of 32,768 at one σ.
        assert_eq!(f.range()[100], (32_768.0 * (-0.5f64).exp()).round() as u16);
        assert_eq!(f.range()[TABLE_SIZE - 1], 0);
        assert_eq!(Bilateral::new(0.4, 1.0).unwrap().radius(), 1);
        assert_eq!(Bilateral::new(3.5, 1.0).unwrap().radius(), 7);
        for (s, r) in [
            (0.0, 10.0),
            (3.6, 10.0),
            (1.0, 0.0),
            (1.0, f64::NAN),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert!(Bilateral::new(s, r).is_err(), "{s}, {r}");
        }
    }

    /// The filter at one pixel, written out plainly from its definition and the tables.
    fn by_hand(image: &Image<u16>, f: &Bilateral, x: i32, y: i32) -> Vec<u16> {
        let (w, h, c) = (
            image.width() as i32,
            image.height() as i32,
            usize::from(image.channels()),
        );
        let r = f.radius() as i32;
        let pixel = |px: i32, py: i32| {
            image
                .pixel(mirror(px, w) as u32, mirror(py, h) as u32)
                .to_vec()
        };
        let centre = pixel(x, y);
        let (mut sums, mut total) = (vec![0u64; c], 0u64);
        for (j, dy) in (-r..=r).enumerate() {
            for (i, dx) in (-r..=r).enumerate() {
                let p = pixel(x + dx, y + dy);
                let diff = |k: usize| u32::from(p[k].abs_diff(centre[k]));
                let d = if c == 3 {
                    (diff(0) + 2 * diff(1) + diff(2)) / 4
                } else {
                    diff(0)
                };
                let weight = u64::from(f.spatial()[j * (2 * r as usize + 1) + i])
                    * u64::from(f.range()[d as usize]);
                total += weight;
                for k in 0..c {
                    sums[k] += weight * u64::from(p[k]);
                }
            }
        }
        sums.iter()
            .map(|&s| ((s + total / 2) / total) as u16)
            .collect()
    }

    #[test]
    fn the_filter_is_its_definition_at_every_pixel() {
        for (w, h, c, s, r) in [
            (9, 7, 1, 1.5, 60.0),
            (8, 6, 3, 1.0, 120.0),
            (11, 9, 3, 2.5, 300.0),
        ] {
            let base = test_pattern(w, h, 12);
            let values: Vec<u16> = if c == 3 {
                base.data().to_vec()
            } else {
                base.data().iter().step_by(3).copied().collect()
            };
            let image = Image::new(w, h, c, noisy(&values, 70.0, 4_095, u64::from(w))).unwrap();
            let f = Bilateral::new(s, r).unwrap();
            let out = bilateral_cpu(&cpu(), &image, &f).unwrap();
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(
                        out.pixel(x, y),
                        by_hand(&image, &f, x as i32, y as i32),
                        "{w} × {h} × {c}, ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn flat_stays_flat_and_an_edge_stays_an_edge() {
        let f = Bilateral::new(1.5, 50.0).unwrap();
        let flat = Image::new(16, 12, 3, [500u16, 1_000, 1_500].repeat(16 * 12)).unwrap();
        assert_eq!(bilateral_cpu(&cpu(), &flat, &f).unwrap(), flat);
        // A step of 1,000 between two halves: far beyond σ_r, so it stays sharp.
        let step: Vec<u16> = (0..12)
            .flat_map(|_| (0..16).map(|x| if x < 8 { 1_000 } else { 2_000 }))
            .collect();
        let edge = Image::new(16, 12, 1, step).unwrap();
        assert_eq!(bilateral_cpu(&cpu(), &edge, &f).unwrap(), edge);
    }

    #[test]
    fn noise_falls_in_flat_regions_and_the_psnr_rises() {
        // Noise of σ 80 in 12 bits, filtered with σ_s 1 and σ_r 2σ. On the pattern's smooth top
        // band, the PSNR rises by about 7 dB; on the whole pattern, whose zone plate holds
        // detail finer than the filter's window, by about 3.
        let f = Bilateral::new(1.0, 160.0).unwrap();
        let full = test_pattern(96, 96, 12);
        let smooth = Image::new(96, 32, 3, full.data()[..96 * 32 * 3].to_vec()).unwrap();
        for (clean, gain) in [(smooth, 5.0), (full, 2.5)] {
            let n = Image::new(
                clean.width(),
                clean.height(),
                3,
                noisy(clean.data(), 80.0, 4_095, 7),
            )
            .unwrap();
            let out = bilateral_cpu(&cpu(), &n, &f).unwrap();
            let (before, after) = (
                psnr(&clean, &n, 4_095).unwrap(),
                psnr(&clean, &out, 4_095).unwrap(),
            );
            assert!(
                after > before + gain,
                "{} rows: {before:.2} dB to {after:.2} dB",
                clean.height()
            );
        }
    }

    #[test]
    fn images_the_filter_cant_take_are_refused() {
        let f = Bilateral::new(1.5, 50.0).unwrap();
        let rgba = Image::new(8, 8, 4, vec![0u16; 256]).unwrap();
        assert_eq!(
            bilateral_cpu(&cpu(), &rgba, &f).unwrap_err().to_string(),
            "the bilateral filter takes grey or RGB, not 4 channels"
        );
        let small = Image::new(3, 8, 1, vec![0u16; 24]).unwrap();
        assert_eq!(
            bilateral_cpu(&cpu(), &small, &f).unwrap_err().to_string(),
            "a bilateral filter of radius 3 needs an image larger than that, not 3 × 8"
        );
        let four = Image::new(4, 4, 1, vec![0u16; 16]).unwrap();
        assert!(
            bilateral_cpu(&cpu(), &four, &f).is_ok(),
            "a radius below the size"
        );
    }

    #[test]
    fn on_the_cpu_the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        let image = Image::new(
            24,
            16,
            3,
            noisy(test_pattern(24, 16, 12).data(), 30.0, 4_095, 3),
        )
        .unwrap();
        let f = Bilateral::new(1.0, 80.0).unwrap();
        assert_eq!(
            bilateral(&compute, &image, &f).unwrap(),
            bilateral_cpu(compute.cpu(), &image, &f).unwrap()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernels_and_their_twins_agree_exactly() {
        let gpu = tetera_compute::cuda_or_skip!();
        let compute = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = CpuBackend::new(0).unwrap();
        for (w, h, c, s) in [
            (8, 8, 1, 1.0),
            (33, 17, 3, 1.5),
            (40, 30, 1, 3.5),
            (1_920, 1_280, 3, 1.5),
        ] {
            let clean = if c == 3 {
                test_pattern(w, h, 12)
            } else {
                Image::new(
                    w,
                    h,
                    1,
                    test_pattern(w, h, 12)
                        .data()
                        .iter()
                        .step_by(3)
                        .copied()
                        .collect(),
                )
                .unwrap()
            };
            let image =
                Image::new(w, h, c, noisy(clean.data(), 40.0, 4_095, u64::from(w))).unwrap();
            let f = Bilateral::new(s, 90.0).unwrap();
            assert_eq!(
                bilateral(&compute, &image, &f).unwrap(),
                bilateral_cpu(&cpu, &image, &f).unwrap(),
                "{w} × {h} × {c}"
            );
        }
        let layout = MosaicLayout {
            width: 64,
            height: 48,
            pattern: BayerPattern::Grbg,
            significant_bits: 12,
        };
        let (mut on_cpu, mut on_gpu) = (
            TemporalMean::new(4).unwrap(),
            DeviceTemporalMean::new(4).unwrap(),
        );
        for k in 0..9u64 {
            let sites = noisy(&vec![1_500; layout.sites()], 50.0, 4_095, k);
            let frame = Mosaic::new(layout, sites).unwrap();
            let moved = k == 6;
            let expected = on_cpu.push_cpu(&cpu, &frame, moved).unwrap();
            let got = on_gpu
                .push(
                    &gpu,
                    &crate::DeviceMosaic::upload(&gpu, &frame).unwrap(),
                    moved,
                )
                .unwrap();
            assert_eq!(got.download(&gpu).unwrap(), expected, "frame {k}");
            assert_eq!(on_gpu.len(), on_cpu.len());
        }
        assert!(!on_gpu.is_empty() && DeviceTemporalMean::new(0).is_err());
        assert!(DeviceTemporalMean::new(2).unwrap().is_empty());
    }
}
