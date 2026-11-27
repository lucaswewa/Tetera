//! The field's kernels (P1.12a, ADR-0016): `kernels/field.cu`, the GPU twin of `tetera-sim`'s
//! `optics`. A depth group's tiles resampled into the sensor's pixels, blurred and summed with
//! the other groups, then lit.

/// Where the sensor looks: its middle on the slide, its pixel there, its angle and its size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorView {
    /// Where its middle looks, µm.
    pub centre: (f64, f64),
    /// A sensor pixel on the slide, µm.
    pub pixel: f64,
    /// Its rows against the stage's x axis, radians.
    pub angle: f64,
    /// Its width, pixels.
    pub width: u32,
    /// Its height, pixels.
    pub height: u32,
    /// The stage's velocity during the frame, µm a second.
    pub velocity: (f64, f64),
    /// Each row's exposure, s.
    pub exposure: f64,
    /// The rolling shutter's line time, s.
    pub line_time: f64,
    /// Samples a pixel averages along its path, 1 or more.
    pub samples: u32,
}

/// Where each sensor pixel's path starts on a grid, in the grid's pixels, as the kernel takes
/// it: pixel (col, row) at `x.0 + col · x.1 + row · x.2` across and the same of `y` down,
/// less half a pixel, as bilinear sampling takes it; and the step between its samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridMap {
    /// Across: the first pixel's place, then a column's step and a row's.
    pub x: (f64, f64, f64),
    /// Down.
    pub y: (f64, f64, f64),
    /// From one of a pixel's samples to the next, grid pixels.
    pub step: (f32, f32),
}

impl SensorView {
    /// The map from its pixels to `grid`'s: the view's place and angle, the rolling shutter's
    /// delay from row to row, and the motion during each row's exposure.
    pub fn grid_map(&self, grid: &TileGrid) -> GridMap {
        let inv = 1.0 / grid.pixel;
        let (sin, cos) = self.angle.sin_cos();
        let u0 = (0.5 - f64::from(self.width) / 2.0) * self.pixel;
        let v0 = (0.5 - f64::from(self.height) / 2.0) * self.pixel;
        let per_sample = self.exposure * inv / f64::from(self.samples.max(1));
        GridMap {
            x: (
                (self.centre.0 + u0 * cos - v0 * sin) * inv - grid.first.0 as f64 - 0.5,
                self.pixel * cos * inv,
                (-self.pixel * sin + self.velocity.0 * self.line_time) * inv,
            ),
            y: (
                (self.centre.1 + u0 * sin + v0 * cos) * inv - grid.first.1 as f64 - 0.5,
                self.pixel * sin * inv,
                (self.pixel * cos + self.velocity.1 * self.line_time) * inv,
            ),
            step: (
                (self.velocity.0 * per_sample) as f32,
                (self.velocity.1 * per_sample) as f32,
            ),
        }
    }
}

/// The grid of level pixels a field resamples from, and the tiles that hold it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileGrid {
    /// The grid's first level pixel, each way.
    pub first: (i64, i64),
    /// How many level pixels, each way.
    pub size: (u32, u32),
    /// A level pixel, µm.
    pub pixel: f64,
    /// The first tile's column and row.
    pub tile: (i32, i32),
    /// How many tiles each way.
    pub tiles: (u32, u32),
}

/// The light as the kernel takes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldLight {
    /// Through bare glass at the middle, photoelectrons a ms.
    pub intensity: f32,
    /// The field's unevenness.
    pub unevenness: f32,
    /// The vignetting.
    pub vignetting: f32,
    /// Red's, green's and blue's weights, by the light's colour temperature.
    pub warmth: [f32; 3],
}

#[cfg(feature = "cuda")]
pub use device::FieldPlanes;

#[cfg(feature = "cuda")]
mod device {
    use super::{FieldLight, SensorView, TileGrid};
    use crate::kernels::FIELD;
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    use tetera_core::{Error, Result};

    fn launch_failed(what: &str, e: cudarc::driver::DriverError) -> Error {
        Error::hardware(format!("the {what} kernel wouldn't launch: {e:?}"))
    }

    /// The most shared memory a blur's block takes before it reads global memory instead.
    const SHARED_LIMIT: u32 = 40_960;

    fn per_pixel(width: u32, height: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (width.div_ceil(16), height.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        }
    }

    /// The planes a field is made in, each the sensor's size, kept from frame to frame so no
    /// frame allocates: a group's stains, their blur along rows, the groups' sum, and the
    /// light, three to a pixel.
    #[derive(Debug)]
    pub struct FieldPlanes {
        /// The sensor's width, pixels.
        pub width: u32,
        /// Its height.
        pub height: u32,
        pub(crate) h: DeviceBuffer<f32>,
        pub(crate) e: DeviceBuffer<f32>,
        pub(crate) rows_h: DeviceBuffer<f32>,
        pub(crate) rows_e: DeviceBuffer<f32>,
        /// The groups' haematoxylin, summed.
        pub sum_h: DeviceBuffer<f32>,
        /// Their eosin.
        pub sum_e: DeviceBuffer<f32>,
        /// The field: irradiance in red, green and blue, three to a pixel.
        pub rgb: DeviceBuffer<f32>,
    }

    impl FieldPlanes {
        /// Whether each plane is the sensor's size, as the kernels take it.
        fn check(&self) -> Result<()> {
            let n = self.width as usize * self.height as usize;
            let planes = [
                &self.h,
                &self.e,
                &self.rows_h,
                &self.rows_e,
                &self.sum_h,
                &self.sum_e,
            ];
            if planes.iter().any(|p| p.len() != n) || self.rgb.len() != 3 * n {
                return Err(Error::internal(format!(
                    "field planes not {} × {}",
                    self.width, self.height
                )));
            }
            Ok(())
        }
    }

    impl CudaBackend {
        /// Planes for a `width` × `height` sensor.
        pub fn field_planes(&self, width: u32, height: u32) -> Result<FieldPlanes> {
            let n = width as usize * height as usize;
            Ok(FieldPlanes {
                width,
                height,
                h: self.zeros(n)?,
                e: self.zeros(n)?,
                rows_h: self.zeros(n)?,
                rows_e: self.zeros(n)?,
                sum_h: self.zeros(n)?,
                sum_e: self.zeros(n)?,
                rgb: self.zeros(3 * n)?,
            })
        }

        /// Each sensor pixel's stains for one depth group, bilinear from `tiles`, the grid's
        /// tiles row by row, each 256 × 256 pixels of two halves, into `planes`.
        pub fn field_resample(
            &self,
            tiles: &[&DeviceBuffer<u16>],
            grid: &TileGrid,
            view: &SensorView,
            planes: &mut FieldPlanes,
        ) -> Result<()> {
            planes.check()?;
            let expected = grid.tiles.0 as usize * grid.tiles.1 as usize;
            if tiles.len() != expected
                || view.samples == 0
                || !(0..256).contains(&(grid.first.0 - i64::from(grid.tile.0) * 256))
                || !(0..256).contains(&(grid.first.1 - i64::from(grid.tile.1) * 256))
                || tiles.iter().any(|t| t.len() != 256 * 256 * 2)
                || (view.width, view.height) != (planes.width, planes.height)
            {
                return Err(Error::invalid_argument(format!(
                    "{} tiles for a grid of {expected}, each 256 × 256 pairs, for a {} × {} view on {} × {} planes",
                    tiles.len(),
                    view.width,
                    view.height,
                    planes.width,
                    planes.height
                )));
            }
            let stream = self.stream();
            let pointers: Vec<_> = tiles.iter().map(|t| t.slice().device_ptr(stream)).collect();
            let table = self.upload(&pointers.iter().map(|(p, _)| *p).collect::<Vec<u64>>())?;
            let function = self.module(&FIELD)?.function("field_resample")?;
            let (columns, gw, gh) = (grid.tiles.0 as i32, grid.size.0 as i32, grid.size.1 as i32);
            let (width, height) = (view.width as i32, view.height as i32);
            let samples = view.samples as i32;
            // The grid's first pixel within its first tile.
            let off_x = (grid.first.0 - i64::from(grid.tile.0) * 256) as i32;
            let off_y = (grid.first.1 - i64::from(grid.tile.1) * 256) as i32;
            let map = view.grid_map(grid);
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(planes.h.slice_mut())
                .arg(planes.e.slice_mut())
                .arg(table.slice())
                .arg(&columns)
                .arg(&off_x)
                .arg(&off_y)
                .arg(&gw)
                .arg(&gh)
                .arg(&map.x.0)
                .arg(&map.x.1)
                .arg(&map.x.2)
                .arg(&map.y.0)
                .arg(&map.y.1)
                .arg(&map.y.2)
                .arg(&map.step.0)
                .arg(&map.step.1)
                .arg(&width)
                .arg(&height)
                .arg(&samples);
            // SAFETY: `field_resample` takes (float* out_h, float* out_e, const unsigned long
            // long* tiles, int columns, int off_x, int off_y, int gw, int gh, double ax,
            // double bx, double cx, double ay, double by, double cy, float step_x, float
            // step_y, int width, int height, int samples), matched here in order and type.
            // `texel` clamps each level pixel into the grid's gw × gh; off_x and off_y are
            // within 0..256, as the grid's first tile holds its first pixel.
            // Thread (col, row) < (width, height) writes out_h and out_e at row · width + col,
            // inside the planes, which are width · height (checked above). It reads level
            // pixels held within the grid's gw × gh, which the grid's tiles cover (checked
            // above: one 256 × 256 tile of pairs for each of its columns × rows), through
            // `table`, whose pointers are live: `tiles` borrows each buffer, and its guard in
            // `pointers` lives past the launch.
            unsafe { launch.launch(per_pixel(view.width, view.height)) }
                .map_err(|e| launch_failed("field_resample", e))?;
            drop(pointers);
            Ok(())
        }

        /// The group's stains in `planes` blurred by `weights` along rows, then columns, and
        /// put into the groups' sum: the `first` group's replaces it, each later one adds.
        pub fn field_blur(
            &self,
            planes: &mut FieldPlanes,
            weights: &[f32],
            first: bool,
        ) -> Result<()> {
            if weights.len() % 2 != 1 {
                return Err(Error::invalid_argument(format!(
                    "{} weights; a blur takes an odd number",
                    weights.len()
                )));
            }
            planes.check()?;
            let table = self.upload(weights)?;
            let radius = (weights.len() / 2) as i32;
            let (w, h) = (planes.width as i32, planes.height as i32);
            let module = self.module(&FIELD)?;
            // Shared memory for a halo that fits; the global kernels for one that doesn't.
            let row_bytes = 2 * (256 + 2 * radius as u32) * 4;
            let col_bytes = 2 * 32 * (8 + 2 * radius as u32) * 4;
            let (rows_name, rows_config) = if row_bytes <= SHARED_LIMIT {
                (
                    "field_blur_rows_shared",
                    LaunchConfig {
                        grid_dim: (planes.width.div_ceil(256), planes.height, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: row_bytes,
                    },
                )
            } else {
                ("field_blur_rows", per_pixel(planes.width, planes.height))
            };
            let (cols_name, cols_config) = if col_bytes <= SHARED_LIMIT {
                (
                    "field_blur_cols_shared",
                    LaunchConfig {
                        grid_dim: (planes.width.div_ceil(32), planes.height.div_ceil(8), 1),
                        block_dim: (32, 8, 1),
                        shared_mem_bytes: col_bytes,
                    },
                )
            } else {
                ("field_blur_cols", per_pixel(planes.width, planes.height))
            };
            let rows = module.function(rows_name)?;
            let mut launch = self.stream().launch_builder(&rows);
            launch
                .arg(planes.rows_h.slice_mut())
                .arg(planes.rows_e.slice_mut())
                .arg(planes.h.slice())
                .arg(planes.e.slice())
                .arg(&w)
                .arg(&h)
                .arg(table.slice())
                .arg(&radius);
            // SAFETY: both row kernels take (float* out_h, float* out_e, const float* in_h,
            // const float* in_e, int width, int height, const float* weights, int radius),
            // matched here in order and type. Every plane is width · height, made so by
            // `field_planes`; a thread reads its row at columns held within 0..width and
            // writes its own value, for col < width and row < height; `weights` holds
            // 2 · radius + 1. The shared kernel's block of 256 reads 2 · (256 + 2 · radius)
            // floats into the shared memory it is given, `row_bytes`.
            unsafe { launch.launch(rows_config) }.map_err(|e| launch_failed(rows_name, e))?;
            let cols = module.function(cols_name)?;
            let add = i32::from(!first);
            let mut launch = self.stream().launch_builder(&cols);
            launch
                .arg(planes.sum_h.slice_mut())
                .arg(planes.sum_e.slice_mut())
                .arg(planes.rows_h.slice())
                .arg(planes.rows_e.slice())
                .arg(&w)
                .arg(&h)
                .arg(table.slice())
                .arg(&radius)
                .arg(&add);
            // SAFETY: both column kernels take the same arguments and int accumulate, writing or
            // adding into the sums; the same bounds hold, along columns. The shared kernel's
            // block of 32 × 8 reads 2 · 32 · (8 + 2 · radius) floats, `col_bytes`.
            unsafe { launch.launch(cols_config) }.map_err(|e| launch_failed(cols_name, e))?;
            Ok(())
        }

        /// The groups' summed stains in `planes` lit, into its `rgb`.
        pub fn field_light(&self, planes: &mut FieldPlanes, light: &FieldLight) -> Result<()> {
            planes.check()?;
            let function = self.module(&FIELD)?.function("field_light")?;
            let (w, h) = (planes.width as i32, planes.height as i32);
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(planes.rgb.slice_mut())
                .arg(planes.sum_h.slice())
                .arg(planes.sum_e.slice())
                .arg(&w)
                .arg(&h)
                .arg(&light.intensity)
                .arg(&light.unevenness)
                .arg(&light.vignetting)
                .arg(&light.warmth[0])
                .arg(&light.warmth[1])
                .arg(&light.warmth[2]);
            // SAFETY: `field_light` takes (float* rgb, const float* sum_h, const float* sum_e,
            // int width, int height, float intensity, float unevenness, float vignetting,
            // float wr, float wg, float wb), matched here in order and type. Thread (col, row)
            // reads its value of the sums, width · height, and writes three at
            // 3 · (row · width + col), inside `rgb` of 3 · width · height, as `field_planes`
            // made them.
            unsafe { launch.launch(per_pixel(planes.width, planes.height)) }
                .map_err(|e| launch_failed("field_light", e))?;
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::half;

    /// A tile whose haematoxylin is its column / 256 and eosin its row / 256.
    fn ramp(gpu: &crate::CudaBackend) -> crate::DeviceBuffer<u16> {
        let mut host = Vec::with_capacity(256 * 256 * 2);
        for row in 0..256 {
            for col in 0..256 {
                host.push(half::from_f32(col as f32 / 256.0));
                host.push(half::from_f32(row as f32 / 256.0));
            }
        }
        gpu.upload(&host).unwrap()
    }

    fn one_tile() -> TileGrid {
        TileGrid {
            first: (0, 0),
            size: (256, 256),
            pixel: 1.0,
            tile: (0, 0),
            tiles: (1, 1),
        }
    }

    fn view(centre: (f64, f64), angle: f64) -> SensorView {
        SensorView {
            centre,
            pixel: 1.0,
            angle,
            width: 8,
            height: 4,
            velocity: (0.0, 0.0),
            exposure: 0.0,
            line_time: 0.0,
            samples: 1,
        }
    }

    #[test]
    fn a_grid_may_start_anywhere_in_its_first_tile() {
        let gpu = crate::cuda_or_skip!();
        let (tile, zeros) = (ramp(&gpu), gpu.zeros::<u16>(256 * 256 * 2).unwrap());
        let mut planes = gpu.field_planes(8, 4).unwrap();
        // Four tiles from (-1, -1), the ramp the last, at (0, 0); the grid from (-56, -30).
        let grid = TileGrid {
            first: (-56, -30),
            size: (200, 200),
            pixel: 1.0,
            tile: (-1, -1),
            tiles: (2, 2),
        };
        gpu.field_resample(
            &[&zeros, &zeros, &zeros, &tile],
            &grid,
            &view((128.0, 128.0), 0.0),
            &mut planes,
        )
        .unwrap();
        let (h, e) = (
            gpu.download(&planes.h).unwrap(),
            gpu.download(&planes.e).unwrap(),
        );
        for k in 0..32 {
            let (col, row) = ((k % 8) as f32, (k / 8) as f32);
            assert_eq!(
                (h[k], e[k]),
                ((124.0 + col) / 256.0, (126.0 + row) / 256.0),
                "{k}"
            );
        }
    }

    #[test]
    fn a_field_resamples_its_tiles_bilinearly_held_at_their_edge() {
        let gpu = crate::cuda_or_skip!();
        let tile = ramp(&gpu);
        let mut planes = gpu.field_planes(8, 4).unwrap();
        // The view's pixels fall on the tile's pixels' middles: column 124 + col, row 126 + row.
        gpu.field_resample(
            &[&tile],
            &one_tile(),
            &view((128.0, 128.0), 0.0),
            &mut planes,
        )
        .unwrap();
        let (h, e) = (
            gpu.download(&planes.h).unwrap(),
            gpu.download(&planes.e).unwrap(),
        );
        for k in 0..32 {
            let (col, row) = ((k % 8) as f32, (k / 8) as f32);
            assert_eq!(
                (h[k], e[k]),
                ((124.0 + col) / 256.0, (126.0 + row) / 256.0),
                "{k}"
            );
        }
        // A quarter pixel along: a quarter of the way to the next.
        gpu.field_resample(
            &[&tile],
            &one_tile(),
            &view((128.25, 128.0), 0.0),
            &mut planes,
        )
        .unwrap();
        assert!((gpu.download(&planes.h).unwrap()[0] - 124.25 / 256.0).abs() < 1e-6);
        // Turned a quarter: the sensor's rows run along the tile's columns.
        gpu.field_resample(
            &[&tile],
            &one_tile(),
            &view((128.0, 128.0), std::f64::consts::FRAC_PI_2),
            &mut planes,
        )
        .unwrap();
        let (h, e) = (
            gpu.download(&planes.h).unwrap(),
            gpu.download(&planes.e).unwrap(),
        );
        assert!(
            (e[1] - e[0] - 1.0 / 256.0).abs() < 1e-6 && (h[1] - h[0]).abs() < 1e-6,
            "{:?}",
            &e[..2]
        );
        // Beyond the grid, held at its edge.
        gpu.field_resample(
            &[&tile],
            &one_tile(),
            &view((-50.0, 400.0), 0.0),
            &mut planes,
        )
        .unwrap();
        assert_eq!(
            (
                gpu.download(&planes.h).unwrap()[0],
                gpu.download(&planes.e).unwrap()[0]
            ),
            (0.0, 255.0 / 256.0)
        );
    }

    #[test]
    fn a_field_refuses_tiles_and_planes_that_dont_fit() {
        let gpu = crate::cuda_or_skip!();
        let tile = ramp(&gpu);
        let short = gpu.zeros::<u16>(256).unwrap();
        let mut planes = gpu.field_planes(8, 4).unwrap();
        let two = TileGrid {
            tiles: (2, 1),
            ..one_tile()
        };
        for (tiles, grid, v) in [
            (vec![&tile], two, view((128.0, 128.0), 0.0)),
            (vec![&short], one_tile(), view((128.0, 128.0), 0.0)),
            (
                vec![&tile],
                one_tile(),
                SensorView {
                    width: 9,
                    ..view((128.0, 128.0), 0.0)
                },
            ),
            (
                vec![&tile],
                one_tile(),
                SensorView {
                    height: 3,
                    ..view((128.0, 128.0), 0.0)
                },
            ),
            (
                vec![&tile],
                one_tile(),
                SensorView {
                    samples: 0,
                    ..view((128.0, 128.0), 0.0)
                },
            ),
            // A grid whose first pixel isn't in its first tile, either way.
            (
                vec![&tile],
                TileGrid {
                    first: (one_tile().first.0 + 256, one_tile().first.1),
                    ..one_tile()
                },
                view((128.0, 128.0), 0.0),
            ),
            (
                vec![&tile],
                TileGrid {
                    first: (one_tile().first.0, one_tile().first.1 - 1),
                    ..one_tile()
                },
                view((128.0, 128.0), 0.0),
            ),
        ] {
            assert!(
                gpu.field_resample(&tiles, &grid, &v, &mut planes).is_err(),
                "{grid:?} {v:?}"
            );
        }
        assert!(
            gpu.field_blur(&mut planes, &[0.5, 0.5], true).is_err(),
            "an even blur"
        );
        assert!(gpu.field_blur(&mut planes, &[], true).is_err());
        // Planes of the wrong size are refused before any kernel runs.
        for broken in 0..3 {
            let mut planes = gpu.field_planes(8, 4).unwrap();
            match broken {
                0 => planes.h = gpu.zeros(31).unwrap(),
                1 => planes.sum_e = gpu.zeros(33).unwrap(),
                _ => planes.rgb = gpu.zeros(95).unwrap(),
            }
            assert!(
                gpu.field_resample(
                    &[&tile],
                    &one_tile(),
                    &view((128.0, 128.0), 0.0),
                    &mut planes
                )
                .is_err()
            );
            assert!(gpu.field_blur(&mut planes, &[1.0], true).is_err());
            let light = FieldLight {
                intensity: 1.0,
                unevenness: 0.0,
                vignetting: 0.0,
                warmth: [1.0; 3],
            };
            assert!(gpu.field_light(&mut planes, &light).is_err(), "{broken}");
        }
    }

    /// `plane` (`w` × `h`) blurred by `weights` along rows then columns, held at its edges.
    fn host_blur(plane: &[f32], w: usize, h: usize, weights: &[f32]) -> Vec<f32> {
        let r = (weights.len() / 2) as i64;
        let pass = |src: &[f32], rows: bool| -> Vec<f32> {
            (0..w * h)
                .map(|k| {
                    let (row, col) = ((k / w) as i64, (k % w) as i64);
                    let mut s = 0.0f32;
                    for (i, weight) in weights.iter().enumerate() {
                        let d = i as i64 - r;
                        let (c, rr) = if rows {
                            ((col + d).clamp(0, w as i64 - 1), row)
                        } else {
                            (col, (row + d).clamp(0, h as i64 - 1))
                        };
                        s += weight * src[(rr * w as i64 + c) as usize];
                    }
                    s
                })
                .collect()
        };
        pass(&pass(plane, true), false)
    }

    #[test]
    fn a_blur_in_shared_memory_or_not_is_the_same_and_adds_up() {
        let gpu = crate::cuda_or_skip!();
        let (w, h) = (300usize, 40usize);
        let input: Vec<f32> = (0..w * h)
            .map(|k| ((k * 37) % 101) as f32 / 101.0)
            .collect();
        for radius in [1usize, 60, 100, 3_000] {
            // Rows and columns in shared memory, rows only, then neither.
            let weights: Vec<f32> = (0..=2 * radius)
                .map(|i| 1.0 / (1.0 + (i as f32 - radius as f32).abs()))
                .collect();
            let total: f32 = weights.iter().sum();
            let weights: Vec<f32> = weights.iter().map(|x| x / total).collect();
            let mut planes = gpu.field_planes(w as u32, h as u32).unwrap();
            planes.h = gpu.upload(&input).unwrap();
            planes.e = gpu
                .upload(&input.iter().map(|x| 1.0 - x).collect::<Vec<_>>())
                .unwrap();
            gpu.field_blur(&mut planes, &weights, true).unwrap();
            let once = gpu.download(&planes.sum_h).unwrap();
            let expected = host_blur(&input, w, h, &weights);
            assert_eq!((once.len(), expected.len()), (w * h, w * h));
            let worst = once
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(worst < 1e-5, "radius {radius}: {worst}");
            gpu.field_blur(&mut planes, &weights, false).unwrap();
            let twice = gpu.download(&planes.sum_h).unwrap();
            assert!(
                twice
                    .iter()
                    .zip(&once)
                    .all(|(a, b)| (a - 2.0 * b).abs() < 1e-5),
                "added, radius {radius}"
            );
            let e = gpu.download(&planes.sum_e).unwrap();
            assert!(
                e.iter()
                    .zip(&twice)
                    .all(|(e, h)| (e + h - 2.0).abs() < 1e-4),
                "eosin too, radius {radius}"
            );
        }
    }

    #[test]
    fn a_field_is_lit_by_beer_lambert_and_its_light() {
        let gpu = crate::cuda_or_skip!();
        let (w, h) = (4u32, 2u32);
        let mut planes = gpu.field_planes(w, h).unwrap();
        let stains: Vec<f32> = (0..8).map(|k| k as f32 * 0.1).collect();
        planes.sum_h = gpu.upload(&stains).unwrap();
        planes.sum_e = gpu
            .upload(&stains.iter().map(|x| 0.5 * x).collect::<Vec<_>>())
            .unwrap();
        let light = FieldLight {
            intensity: 1_000.0,
            unevenness: 0.1,
            vignetting: 0.2,
            warmth: [1.4, 1.0, 0.6],
        };
        gpu.field_light(&mut planes, &light).unwrap();
        let rgb = gpu.download(&planes.rgb).unwrap();
        let (hv, ev) = ([0.650f32, 0.704, 0.286], [0.072f32, 0.990, 0.105]);
        for k in 0..8 {
            let (col, row) = ((k % 4) as f32, (k / 4) as f32);
            let u = (col + 0.5) / 4.0 * 2.0 - 1.0;
            let v = (row + 0.5) / 2.0 * 2.0 - 1.0;
            let lit = 1_000.0
                * (1.0 + 0.1 * (0.6 * u - 0.3 * v + 0.1 * u * v))
                * (1.0 - 0.2 * (u * u + v * v) / 2.0);
            for c in 0..3 {
                let od = stains[k] * hv[c] + 0.5 * stains[k] * ev[c];
                let expected = lit * light.warmth[c] * 10f32.powf(-od);
                assert!(
                    (rgb[3 * k + c] - expected).abs() < 1e-3 * expected,
                    "{k} {c}: {} against {expected}",
                    rgb[3 * k + c]
                );
            }
        }
    }
}
