//! The H&E sample's kernels (P1.11b, ADR-0015): `kernels/he.cu`, the GPU twin of
//! `tetera-sim`'s procedural sample. The model lives in `tetera-sim`; this module launches its
//! kernels from plain numbers, since only this crate may hold `unsafe` code.

/// The nuclei's grid, µm, as `tetera-sim` places them.
pub const NUCLEUS_CELL: f64 = 10.0;
/// The red cells' grid, µm.
pub const RED_CELL_CELL: f64 = 7.0;

/// What the kernels need of a sample: its folded seed and its section.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeParams {
    /// The seed, folded to 32 bits as `tetera-sim` folds it.
    pub seed: u32,
    /// The section's thickness, µm.
    pub thickness: f32,
    /// Whether it lies folded in places.
    pub folds: bool,
}

/// A region for the kernels, as `tetera-sim`'s `Region` is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeRegion {
    /// Its first pixel's corner on the slide, µm.
    pub origin: (f64, f64),
    /// Its width, pixels.
    pub width: u32,
    /// Its height, pixels.
    pub height: u32,
    /// Its pixel size, µm.
    pub pixel: f64,
    /// How many points a side each pixel averages.
    pub points: u32,
    /// The depths it sums, µm above the section's lower face; `None` for the whole section.
    pub depth: Option<(f32, f32)>,
}

/// The grid cells a region's nuclei or red cells come from: its first cell, and how many
/// columns and rows, one more around than the region touches, as the CPU twin takes them.
pub fn grid(cell: f64, region: &HeRegion) -> ((i32, i32), i32, i32) {
    let (x0, y0) = region.origin;
    let x1 = x0 + f64::from(region.width) * region.pixel;
    let y1 = y0 + f64::from(region.height) * region.pixel;
    let first = (
        (x0 / cell).floor() as i32 - 1,
        (y0 / cell).floor() as i32 - 1,
    );
    let columns = (x1 / cell).floor() as i32 + 2 - first.0;
    let rows = (y1 / cell).floor() as i32 + 2 - first.1;
    (first, columns, rows)
}

#[cfg(feature = "cuda")]
mod device {
    use super::{HeParams, HeRegion, NUCLEUS_CELL, RED_CELL_CELL, grid};
    use crate::kernels::HE;
    use crate::{CudaBackend, DeviceBuffer};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use tetera_core::{Error, Result};

    fn launch_failed(what: &str, e: cudarc::driver::DriverError) -> Error {
        Error::hardware(format!("the {what} kernel wouldn't launch: {e:?}"))
    }

    /// A grid's cells on the device: each one's centre, and its other values.
    struct Cells {
        centres: DeviceBuffer<f64>,
        params: DeviceBuffer<f32>,
        first: (i32, i32),
        columns: i32,
    }

    impl CudaBackend {
        /// Decides each cell's nucleus (`per` 8) or red cell (`per` 2) by `kernel`.
        fn he_cells(
            &self,
            kernel: &str,
            cell: f64,
            per: usize,
            params: &HeParams,
            region: &HeRegion,
        ) -> Result<Cells> {
            let (first, columns, rows) = grid(cell, region);
            let n = (columns * rows) as usize;
            let mut centres = self.zeros::<f64>(2 * n)?;
            let mut values = self.zeros::<f32>(per * n)?;
            let function = self.module(&HE)?.function(kernel)?;
            let (count, folds) = (n as i32, u32::from(params.folds));
            let config = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(centres.slice_mut())
                .arg(values.slice_mut())
                .arg(&first.0)
                .arg(&first.1)
                .arg(&columns)
                .arg(&count)
                .arg(&params.seed)
                .arg(&params.thickness)
                .arg(&folds);
            // SAFETY: both kernels take (double* centres, float* params, int i0, int j0, int
            // columns, int n, unsigned seed, float thickness, unsigned folds), matched here in
            // order and type. Thread k < n writes centres[2k..2k+2] and params[per·k..per·k+per],
            // inside buffers of 2n and per·n, `per` being 8 for nuclei and 2 for red cells.
            unsafe { launch.launch(config) }.map_err(|e| launch_failed(kernel, e))?;
            Ok(Cells {
                centres,
                params: values,
                first,
                columns,
            })
        }

        /// Renders `region` of the sample `params` describe: each pixel's haematoxylin and
        /// eosin, summed over its depths, in half precision, two to a pixel, row by row.
        pub fn he_render(&self, params: &HeParams, region: &HeRegion) -> Result<DeviceBuffer<u16>> {
            let nuclei = self.he_cells("he_nuclei", NUCLEUS_CELL, 8, params, region)?;
            let red = self.he_cells("he_red_cells", RED_CELL_CELL, 2, params, region)?;
            let (width, height) = (region.width as i32, region.height as i32);
            let mut out = self.zeros::<u16>(2 * region.width as usize * region.height as usize)?;
            let function = self.module(&HE)?.function("he_render")?;
            let (whole, (from, to)) = match region.depth {
                None => (1i32, (0.0f32, 0.0f32)),
                Some(range) => (0, range),
            };
            let (points, folds) = (region.points as i32, u32::from(params.folds));
            let config = LaunchConfig {
                grid_dim: (region.width.div_ceil(16), region.height.div_ceil(16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(out.slice_mut())
                .arg(nuclei.centres.slice())
                .arg(nuclei.params.slice())
                .arg(&nuclei.first.0)
                .arg(&nuclei.first.1)
                .arg(&nuclei.columns)
                .arg(red.centres.slice())
                .arg(red.params.slice())
                .arg(&red.first.0)
                .arg(&red.first.1)
                .arg(&red.columns)
                .arg(&region.origin.0)
                .arg(&region.origin.1)
                .arg(&width)
                .arg(&height)
                .arg(&region.pixel)
                .arg(&points)
                .arg(&whole)
                .arg(&from)
                .arg(&to)
                .arg(&params.seed)
                .arg(&params.thickness)
                .arg(&folds);
            // SAFETY: `he_render` takes (unsigned short* out, const double* n_centres, const
            // float* n_params, int n_i0, int n_j0, int n_columns, the same three and three for
            // red cells, double x0, double y0, int width, int height, double pixel, int points,
            // int whole, float from, float to, unsigned seed, float thickness, unsigned folds),
            // matched here in order and type. Thread (col, row) < (width, height) writes
            // out[2(row·width + col)..+2], inside a buffer of 2·width·height. Its points lie in
            // the region, and the grids hold every cell from one before the region's first to
            // one after its last (`grid`), so the 3 × 3 cells around each point it reads are
            // inside them.
            unsafe { launch.launch(config) }.map_err(|e| launch_failed("he_render", e))?;
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_holds_one_more_cell_around_its_region() {
        let region = HeRegion {
            origin: (95.0, -12.0),
            width: 10,
            height: 4,
            pixel: 3.0,
            points: 1,
            depth: None,
        };
        // x from 95 to 125: cells 9 to 12, and one more each side; y from -12 to 0: -2 to 0.
        assert_eq!(grid(10.0, &region), ((8, -3), 6, 5));
        assert_eq!(grid(7.0, &region), ((12, -3), 7, 5));
    }
}
