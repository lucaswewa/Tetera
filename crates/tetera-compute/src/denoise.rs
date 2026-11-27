//! Denoising's kernels (P1.07a): temporal averaging over a window of frames, and the bilateral
//! filter. Their CPU twins, and the tables the filter takes, are in `tetera-imaging`'s
//! `denoise` module.

/// A bilateral weight's unit: a weight of 1.
pub const BILATERAL_ONE: u16 = 32_768;

/// The largest radius the bilateral filter takes.
pub const MAX_RADIUS: u32 = 7;

/// The most frames a temporal window holds: their sum must fit 32 bits.
pub const MAX_WINDOW: u32 = 64;

#[cfg(feature = "cuda")]
pub use device::{BilateralTables, DeviceTemporal};

#[cfg(feature = "cuda")]
mod device {
    use super::{MAX_RADIUS, MAX_WINDOW};
    use crate::imaging::TABLE_SIZE;
    use crate::kernels::DENOISE;
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use tetera_core::{Error, Result};

    fn launch_failed(what: &str, e: cudarc::driver::DriverError) -> Error {
        Error::hardware(format!("the {what} kernel wouldn't launch: {e:?}"))
    }

    /// A temporal window on the device: the last frames, a ring, and their sum.
    #[derive(Debug)]
    pub struct DeviceTemporal {
        window: u32,
        sites: usize,
        sum: DeviceBuffer<u32>,
        ring: Vec<DeviceBuffer<u16>>,
        oldest: usize,
    }

    impl DeviceTemporal {
        /// An empty window of up to `window` frames of `sites` each.
        pub fn new(gpu: &CudaBackend, window: u32, sites: usize) -> Result<Self> {
            if !(1..=MAX_WINDOW).contains(&window) {
                return Err(Error::invalid_argument(format!(
                    "a temporal window holds 1 to {MAX_WINDOW} frames, not {window}"
                )));
            }
            Ok(Self {
                window,
                sites,
                sum: gpu.zeros(sites)?,
                ring: Vec::new(),
                oldest: 0,
            })
        }

        /// The most frames it holds.
        pub fn window(&self) -> u32 {
            self.window
        }

        /// The frames it holds now.
        pub fn len(&self) -> u32 {
            self.ring.len() as u32
        }

        /// Whether it holds none.
        pub fn is_empty(&self) -> bool {
            self.ring.is_empty()
        }

        /// Its frames' size.
        pub fn sites(&self) -> usize {
            self.sites
        }

        /// Empties it: the next frame starts a new window.
        pub fn reset(&mut self, gpu: &CudaBackend) -> Result<()> {
            self.sum = gpu.zeros(self.sites)?;
            self.ring.clear();
            self.oldest = 0;
            Ok(())
        }

        /// `frame` into the window, the oldest frame out if it was full; the window's mean into
        /// `mean`. It gives how many frames the mean is over.
        pub fn push(
            &mut self,
            gpu: &CudaBackend,
            frame: &DeviceBuffer<u16>,
            mean: &mut DeviceBuffer<u16>,
        ) -> Result<u32> {
            if frame.len() != self.sites || mean.len() != self.sites {
                return Err(Error::invalid_argument(format!(
                    "a window of {} sites takes frames and means of as many, not {} and {}",
                    self.sites,
                    frame.len(),
                    mean.len()
                )));
            }
            let full = self.ring.len() == self.window as usize;
            let index = if full {
                let i = self.oldest;
                self.oldest = (self.oldest + 1) % self.window as usize;
                i
            } else {
                self.ring.push(gpu.zeros(self.sites)?);
                self.ring.len() - 1
            };
            let n = self.ring.len() as u32;
            if self.sites == 0 {
                return Ok(n);
            }
            let kernel = gpu.module(&DENOISE)?.function("temporal_push_u16")?;
            let (sites, full, count) = (self.sites as i64, i32::from(full), n as i32);
            let slot = &mut self.ring[index];
            let mut launch = gpu.stream().launch_builder(&kernel);
            launch
                .arg(mean.slice_mut())
                .arg(self.sum.slice_mut())
                .arg(slot.slice_mut())
                .arg(frame.slice())
                .arg(&sites)
                .arg(&full)
                .arg(&count);
            let config = LaunchConfig {
                grid_dim: ((self.sites as u64).div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the kernel takes (unsigned short* mean, unsigned int* sum, unsigned short*
            // slot, const unsigned short* frame, long long sites, int full, int n), matched here
            // in order and type. Thread i < sites reads and writes sum[i] and slot[i], reads
            // frame[i] and writes mean[i]; the sum and every slot hold `sites` (made so), and
            // frame and mean were checked to. n is at least 1, so the division is defined.
            unsafe { launch.launch(config) }.map_err(|e| launch_failed("temporal_push_u16", e))?;
            Ok(n)
        }
    }

    /// The bilateral filter's tables on the device, checked when uploaded.
    #[derive(Debug)]
    pub struct BilateralTables {
        radius: u32,
        spatial: DeviceBuffer<u16>,
        range: DeviceBuffer<u16>,
    }

    impl BilateralTables {
        /// Its radius.
        pub fn radius(&self) -> u32 {
            self.radius
        }
    }

    impl CudaBackend {
        /// The bilateral filter's tables: `spatial`, `(2 × radius + 1)²` weights row by row,
        /// and `range`, one weight for each difference from 0 to 65,535. The centre's spatial
        /// weight and the range weight of no difference must be above zero, so no pixel's
        /// weights sum to zero.
        pub fn upload_bilateral(
            &self,
            radius: u32,
            spatial: &[u16],
            range: &[u16],
        ) -> Result<BilateralTables> {
            let side = (2 * radius + 1) as usize;
            if radius > MAX_RADIUS || spatial.len() != side * side || range.len() != TABLE_SIZE {
                return Err(Error::invalid_argument(format!(
                    "a bilateral filter of radius {radius} (at most {MAX_RADIUS}) takes {} spatial weights and {TABLE_SIZE} range weights, not {} and {}",
                    side * side,
                    spatial.len(),
                    range.len()
                )));
            }
            if spatial[side * side / 2] == 0 || range[0] == 0 {
                return Err(Error::invalid_argument(
                    "a bilateral filter's centre must weigh more than nothing",
                ));
            }
            Ok(BilateralTables {
                radius,
                spatial: self.upload(spatial)?,
                range: self.upload(range)?,
            })
        }

        /// The bilateral filter on `input`, `width × height` pixels of `channels` (1 or 3)
        /// interleaved values, into `out`.
        pub fn bilateral(
            &self,
            input: &DeviceBuffer<u16>,
            (width, height, channels): (u32, u32, u8),
            tables: &BilateralTables,
            out: &mut DeviceBuffer<u16>,
        ) -> Result<()> {
            let values = width as usize * height as usize * usize::from(channels);
            if !matches!(channels, 1 | 3) || input.len() != values || out.len() != values {
                return Err(Error::invalid_argument(format!(
                    "the bilateral filter takes 1 or 3 channels, and {values} values in and out for {width} × {height} × {channels}; not {} and {}",
                    input.len(),
                    out.len()
                )));
            }
            if values == 0 {
                return Ok(());
            }
            if width <= tables.radius || height <= tables.radius {
                return Err(Error::invalid_argument(format!(
                    "a bilateral filter of radius {} needs an image larger than that, not {width} × {height}",
                    tables.radius
                )));
            }
            let (Ok(w), Ok(h), Ok(_)) = (
                i32::try_from(width),
                i32::try_from(height),
                i32::try_from(values),
            ) else {
                return Err(Error::invalid_argument(format!(
                    "a {width} × {height} image is too large for one launch"
                )));
            };
            let kernel = self.module(&DENOISE)?.function("bilateral_u16")?;
            let (c, r) = (i32::from(channels), tables.radius as i32);
            let mut launch = self.stream().launch_builder(&kernel);
            launch
                .arg(out.slice_mut())
                .arg(input.slice())
                .arg(&w)
                .arg(&h)
                .arg(&c)
                .arg(&r)
                .arg(tables.spatial.slice())
                .arg(tables.range.slice());
            let tile = crate::imaging::TILE;
            let config = LaunchConfig {
                grid_dim: (width.div_ceil(tile), height.div_ceil(tile), 1),
                block_dim: (tile, tile, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the kernel takes (unsigned short* out, const unsigned short* in, int
            // width, int height, int channels, int radius, const unsigned short* spatial, const
            // unsigned short* range), matched here in order and type. Thread (x, y), for x <
            // width and y < height, reads neighbours mirrored into 0..width and 0..height,
            // which holds since both exceed the radius (checked), so it reads and writes inside
            // `input` and `out`, checked to hold width × height × channels. It reads spatial[0
            // .. (2r + 1)²] and range[d] for d at most 65,535, inside the tables, checked when
            // uploaded. The weights sum to at least the centre's, checked above zero, so the
            // division is defined.
            unsafe { launch.launch(config) }.map_err(|e| launch_failed("bilateral_u16", e))?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::imaging::TABLE_SIZE;

    #[test]
    fn a_window_keeps_the_mean_of_its_last_frames_and_resets() {
        let gpu = crate::cuda_or_skip!();
        let mut window = DeviceTemporal::new(&gpu, 3, 2).unwrap();
        assert_eq!(
            (
                window.window(),
                window.len(),
                window.is_empty(),
                window.sites()
            ),
            (3, 0, true, 2)
        );
        let mut mean = gpu.zeros::<u16>(2).unwrap();
        let mut push = |w: &mut DeviceTemporal, a: u16, b: u16| {
            let n = w
                .push(&gpu, &gpu.upload(&[a, b]).unwrap(), &mut mean)
                .unwrap();
            (n, gpu.download(&mean).unwrap())
        };
        assert_eq!(push(&mut window, 10, 100), (1, vec![10, 100]));
        assert_eq!((window.len(), window.is_empty()), (1, false));
        assert_eq!(
            push(&mut window, 20, 101),
            (2, vec![15, 101]),
            "100.5 rounds up"
        );
        assert_eq!(push(&mut window, 30, 102), (3, vec![20, 101]));
        // Full: 10 leaves, 40 comes, and then 20 leaves.
        assert_eq!(push(&mut window, 40, 103), (3, vec![30, 102]));
        assert_eq!(
            push(&mut window, 50, 0),
            (3, vec![40, 68]),
            "(102 + 103 + 0) / 3 = 68.3"
        );
        window.reset(&gpu).unwrap();
        assert!(window.is_empty());
        assert_eq!(push(&mut window, 7, 9), (1, vec![7, 9]), "a fresh start");
        let err = window
            .push(
                &gpu,
                &gpu.upload(&[1u16]).unwrap(),
                &mut gpu.zeros(2).unwrap(),
            )
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "a window of 2 sites takes frames and means of as many, not 1 and 2"
        );
        assert!(
            DeviceTemporal::new(&gpu, 0, 2).is_err()
                && DeviceTemporal::new(&gpu, MAX_WINDOW + 1, 2).is_err()
        );
        let mut empty = DeviceTemporal::new(&gpu, 2, 0).unwrap();
        assert_eq!(
            empty
                .push(
                    &gpu,
                    &gpu.upload::<u16>(&[]).unwrap(),
                    &mut gpu.zeros(0).unwrap()
                )
                .unwrap(),
            1
        );
    }

    /// Tables that weigh every neighbour alike, whatever its difference.
    fn box_tables(gpu: &crate::CudaBackend, radius: u32) -> BilateralTables {
        let side = (2 * radius + 1) as usize;
        gpu.upload_bilateral(
            radius,
            &vec![BILATERAL_ONE; side * side],
            &vec![BILATERAL_ONE; TABLE_SIZE],
        )
        .unwrap()
    }

    #[test]
    fn the_filter_weighs_neighbours_by_place_and_likeness() {
        let gpu = crate::cuda_or_skip!();
        // 3 × 3 grey, a box of radius 1: every pixel the mean of its mirrored neighbourhood.
        let input = gpu.upload(&[0u16, 0, 0, 0, 90, 0, 0, 0, 0]).unwrap();
        let mut out = gpu.zeros::<u16>(9).unwrap();
        gpu.bilateral(&input, (3, 3, 1), &box_tables(&gpu, 1), &mut out)
            .unwrap();
        // The centre: 90 / 9 = 10. A corner mirrors to read the centre 4 times: 360 / 9 = 40.
        assert_eq!(
            gpu.download(&out).unwrap(),
            [40, 20, 40, 20, 10, 20, 40, 20, 40]
        );
        // A range table that ignores any difference: nothing is averaged.
        let mut range = vec![0u16; TABLE_SIZE];
        range[0] = BILATERAL_ONE;
        let strict = gpu
            .upload_bilateral(1, &[BILATERAL_ONE; 9], &range)
            .unwrap();
        assert_eq!(strict.radius(), 1);
        gpu.bilateral(&input, (3, 3, 1), &strict, &mut out).unwrap();
        assert_eq!(gpu.download(&out).unwrap(), [0, 0, 0, 0, 90, 0, 0, 0, 0]);
        // RGB: green counts twice in the difference.
        let rgb = gpu
            .upload(&[100u16, 0, 0, 100, 0, 0, 100, 4, 0, 100, 0, 0])
            .unwrap();
        let mut range = vec![0u16; TABLE_SIZE];
        range[0] = BILATERAL_ONE;
        range[1] = BILATERAL_ONE;
        let near = gpu
            .upload_bilateral(1, &[BILATERAL_ONE; 9], &range)
            .unwrap();
        let mut out = gpu.zeros::<u16>(12).unwrap();
        gpu.bilateral(&rgb, (2, 2, 3), &near, &mut out).unwrap();
        // (0, 0)'s difference from (0, 1), green 4: 2 × 4 / 4 = 2, so it isn't averaged in.
        assert_eq!(gpu.download(&out).unwrap()[1], 0);
    }

    #[test]
    fn tables_and_images_that_dont_fit_are_refused() {
        let gpu = crate::cuda_or_skip!();
        assert!(
            gpu.upload_bilateral(8, &vec![1; 17 * 17], &vec![1; TABLE_SIZE])
                .is_err(),
            "past the largest radius"
        );
        assert!(
            gpu.upload_bilateral(1, &[1; 8], &vec![1; TABLE_SIZE])
                .is_err()
        );
        assert!(gpu.upload_bilateral(1, &[1; 9], &[1; 10]).is_err());
        let mut zero_centre = [1u16; 9];
        zero_centre[4] = 0;
        let err = gpu
            .upload_bilateral(1, &zero_centre, &vec![1; TABLE_SIZE])
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "a bilateral filter's centre must weigh more than nothing"
        );
        let mut zero_range = vec![1u16; TABLE_SIZE];
        zero_range[0] = 0;
        assert!(gpu.upload_bilateral(1, &[1; 9], &zero_range).is_err());
        // Radius 2 fits 3 × 3, mirrored; radius 3 doesn't.
        assert!(
            gpu.bilateral(
                &gpu.upload(&[0u16; 9]).unwrap(),
                (3, 3, 1),
                &box_tables(&gpu, 2),
                &mut gpu.zeros(9).unwrap()
            )
            .is_ok()
        );
        let tables = box_tables(&gpu, 3);
        assert_eq!(box_tables(&gpu, 2).radius(), 2);
        assert!(
            gpu.upload_bilateral(MAX_RADIUS, &vec![1; 15 * 15], &vec![1; TABLE_SIZE])
                .is_ok(),
            "the largest radius"
        );
        let mut zero_centre_5 = vec![1u16; 25];
        zero_centre_5[12] = 0;
        assert!(
            gpu.upload_bilateral(2, &zero_centre_5, &vec![1; TABLE_SIZE])
                .is_err(),
            "the centre of 5 × 5"
        );
        // Each refusal alone: two channels with lengths that fit them; an input, or an
        // output, that doesn't fit; too few rows though enough columns.
        let six = gpu.upload(&[0u16; 18]).unwrap();
        let mut six_out = gpu.zeros::<u16>(18).unwrap();
        let one = box_tables(&gpu, 1);
        assert!(
            gpu.bilateral(&six, (3, 3, 2), &one, &mut six_out).is_err(),
            "two channels"
        );
        let nine = gpu.upload(&[0u16; 9]).unwrap();
        assert!(
            gpu.bilateral(
                &gpu.upload(&[0u16; 8]).unwrap(),
                (3, 3, 1),
                &one,
                &mut gpu.zeros(9).unwrap()
            )
            .is_err(),
            "the input"
        );
        assert!(
            gpu.bilateral(&nine, (3, 3, 1), &one, &mut gpu.zeros(8).unwrap())
                .is_err(),
            "the output"
        );
        let wide = gpu.upload(&[0u16; 24]).unwrap();
        assert!(
            gpu.bilateral(&wide, (8, 3, 1), &tables, &mut gpu.zeros(24).unwrap())
                .is_err(),
            "3 rows for radius 3"
        );
        let input = gpu.upload(&[0u16; 9]).unwrap();
        let mut out = gpu.zeros::<u16>(9).unwrap();
        let err = gpu
            .bilateral(&input, (3, 3, 1), &tables, &mut out)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "a bilateral filter of radius 3 needs an image larger than that, not 3 × 3"
        );
        assert!(
            gpu.bilateral(&input, (3, 3, 2), &box_tables(&gpu, 1), &mut out)
                .is_err(),
            "2 channels"
        );
        assert!(
            gpu.bilateral(&input, (3, 2, 1), &box_tables(&gpu, 1), &mut out)
                .is_err(),
            "a size that doesn't fit"
        );
        let empty = gpu.upload::<u16>(&[]).unwrap();
        gpu.bilateral(&empty, (0, 0, 1), &tables, &mut gpu.zeros(0).unwrap())
            .unwrap();
    }
}
