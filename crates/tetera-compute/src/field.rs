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
    const SHARED_LIMIT: u32 = 40 * 1024;

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
        h: DeviceBuffer<f32>,
        e: DeviceBuffer<f32>,
        rows_h: DeviceBuffer<f32>,
        rows_e: DeviceBuffer<f32>,
        /// The groups' haematoxylin, summed.
        pub sum_h: DeviceBuffer<f32>,
        /// Their eosin.
        pub sum_e: DeviceBuffer<f32>,
        /// The field: irradiance in red, green and blue, three to a pixel.
        pub rgb: DeviceBuffer<f32>,
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
            let expected = grid.tiles.0 as usize * grid.tiles.1 as usize;
            if tiles.len() != expected
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
            let (sin, cos) = view.angle.sin_cos();
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(planes.h.slice_mut())
                .arg(planes.e.slice_mut())
                .arg(table.slice())
                .arg(&grid.tile.0)
                .arg(&grid.tile.1)
                .arg(&columns)
                .arg(&grid.first.0)
                .arg(&grid.first.1)
                .arg(&gw)
                .arg(&gh)
                .arg(&grid.pixel)
                .arg(&view.centre.0)
                .arg(&view.centre.1)
                .arg(&view.pixel)
                .arg(&cos)
                .arg(&sin)
                .arg(&width)
                .arg(&height);
            // SAFETY: `field_resample` takes (float* out_h, float* out_e, const unsigned long
            // long* tiles, int c0, int r0, int columns, long long gi0, long long gj0, int gw,
            // int gh, double level_pixel, double centre_x, double centre_y, double pixel,
            // double cos, double sin, int width, int height), matched here in order and type.
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
