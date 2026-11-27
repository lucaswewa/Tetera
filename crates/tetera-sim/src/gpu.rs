//! The H&E sample on the GPU (P1.11b, ADR-0015): `tetera-compute`'s kernels, the twin of
//! `render`, and the GPU's tiles in their cache, within a VRAM budget.

use crate::he::{HeSample, Stains};
use crate::optics::{Field, FieldOfView, Grid, Light, blackbody, depth_groups, gaussian};
use crate::render::{Depth, Region, Rendered};
use crate::tiles::{CacheStats, TILE, TileCache, TileDepth, TileKey};
use std::sync::{Arc, Mutex, PoisonError};
use tetera_compute::budget::{Budget, Grant};
use tetera_compute::field::{FieldLight, FieldPlanes, SensorView, TileGrid};
use tetera_compute::half;
use tetera_compute::he::{HeParams, HeRegion};
use tetera_compute::{CudaBackend, DeviceBuffer};
use tetera_core::Result;

/// A tile's bytes on the GPU: two half-precision stains a pixel.
pub const TILE_BYTES: u64 = TILE as u64 * TILE as u64 * 2 * 2;

impl HeSample {
    fn kernel_params(&self) -> HeParams {
        HeParams {
            seed: self.seed.0,
            thickness: self.section.thickness,
            folds: self.section.folds,
        }
    }

    /// `region` rendered on the GPU, left there: each pixel's stains in half precision, two
    /// to a pixel.
    pub fn render_on(&self, gpu: &CudaBackend, region: &Region) -> Result<DeviceBuffer<u16>> {
        region.check()?;
        let depth = match region.depth {
            Depth::Whole => None,
            Depth::Range(a, b) => Some((a, b)),
        };
        let kernel_region = HeRegion {
            origin: region.origin,
            width: region.width,
            height: region.height,
            pixel: region.pixel,
            points: region.points(),
            depth,
        };
        gpu.he_render(&self.kernel_params(), &kernel_region)
    }

    /// `region` rendered on the GPU and brought back: the CPU twin's result within the
    /// behaviour spec's tolerance.
    pub fn render_gpu(&self, gpu: &CudaBackend, region: &Region) -> Result<Rendered> {
        let halves = gpu.download(&self.render_on(gpu, region)?)?;
        Ok(Rendered {
            width: region.width,
            height: region.height,
            stains: stains(&halves),
        })
    }
}

/// Half-precision pairs as stains.
pub fn stains(halves: &[u16]) -> Vec<Stains> {
    halves
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| Stains {
            haematoxylin: half::to_f32(p[0]),
            eosin: half::to_f32(p[1]),
        })
        .collect()
}

/// A tile on the GPU, and the part of the VRAM budget it holds until it goes.
#[derive(Debug)]
pub struct DeviceTile {
    /// Its stains, in half precision, two to a pixel, row by row.
    pub buffer: DeviceBuffer<u16>,
    _grant: Grant,
}

/// The sample's tiles on the GPU, made on demand and kept in an LRU cache within a VRAM
/// budget.
pub struct HeTiles {
    gpu: Arc<CudaBackend>,
    sample: HeSample,
    budget: Budget,
    cache: Mutex<TileCache<Arc<DeviceTile>>>,
    planes: Mutex<Option<FieldPlanes>>,
}

impl std::fmt::Debug for HeTiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeTiles")
            .field("budget", &self.budget.limit())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl HeTiles {
    /// `sample`'s tiles on `gpu`, within `budget`.
    pub fn new(gpu: Arc<CudaBackend>, sample: HeSample, budget: Budget) -> Result<Self> {
        let cache = TileCache::new(budget.limit(), TILE_BYTES)?;
        Ok(Self {
            gpu,
            sample,
            budget,
            cache: Mutex::new(cache),
            planes: Mutex::new(None),
        })
    }

    /// The tile for `key`, made if it isn't held.
    pub fn tile(&self, key: TileKey) -> Result<Arc<DeviceTile>> {
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        cache.get_or_make(key, |key| {
            let grant = self.budget.allocate(TILE_BYTES)?;
            let buffer = self.sample.render_on(&self.gpu, &key.region())?;
            Ok(Arc::new(DeviceTile {
                buffer,
                _grant: grant,
            }))
        })
    }

    /// The tile for `key`, brought back to the host.
    pub fn tile_stains(&self, key: TileKey) -> Result<Vec<Stains>> {
        let tile = self.tile(key)?;
        Ok(stains(&self.gpu.download(&tile.buffer)?))
    }

    /// What its cache has done, and holds.
    pub fn stats(&self) -> CacheStats {
        self.cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stats()
    }

    /// Its VRAM budget.
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// The field `fov` sees, focused `focus` µm above the section's lower face, under `light`,
    /// made on the GPU in its planes, kept for the next frame, and handed to `with`. The twin
    /// of `optics::field`; each depth group's tiles come from the cache, made if they aren't
    /// held.
    pub fn field_with<R>(
        &self,
        fov: &FieldOfView,
        light: &Light,
        focus: f64,
        with: impl FnOnce(&FieldPlanes) -> Result<R>,
    ) -> Result<R> {
        let s = fov.optics.sensor;
        let groups = depth_groups(&fov.optics, focus, self.sample.section().thickness);
        let widest = groups.iter().map(|g| g.sigma).fold(0.0, f32::max);
        let grid = Grid::new(fov, widest);
        let (tile, columns, rows) = grid.tiles();
        let tiles = TileGrid {
            first: grid.first,
            size: grid.size,
            pixel: grid.pixel(),
            tile,
            tiles: (columns, rows),
        };
        let view = SensorView {
            centre: fov.centre,
            pixel: fov.optics.pixel(),
            angle: fov.optics.mounting_angle,
            width: s.width,
            height: s.height,
        };
        let mut planes = self.planes.lock().unwrap_or_else(PoisonError::into_inner);
        if planes
            .as_ref()
            .is_none_or(|p| (p.width, p.height) != (s.width, s.height))
        {
            *planes = None;
            *planes = Some(self.gpu.field_planes(s.width, s.height)?);
        }
        let planes = planes.as_mut().expect("planes made");
        for (k, group) in groups.iter().enumerate() {
            let depth = TileDepth::Layers {
                first: group.first,
                count: group.count,
            };
            let mut held = Vec::with_capacity(columns as usize * rows as usize);
            for r in 0..rows as i32 {
                for c in 0..columns as i32 {
                    held.push(self.tile(TileKey {
                        level: grid.level,
                        column: tile.0 + c,
                        row: tile.1 + r,
                        depth,
                    })?);
                }
            }
            let buffers: Vec<&DeviceBuffer<u16>> = held.iter().map(|t| &t.buffer).collect();
            self.gpu.field_resample(&buffers, &tiles, &view, planes)?;
            self.gpu
                .field_blur(planes, &gaussian(group.sigma), k == 0)?;
        }
        let lit = FieldLight {
            intensity: light.intensity,
            unevenness: light.unevenness,
            vignetting: fov.optics.vignetting,
            warmth: blackbody(light.colour_temperature),
        };
        self.gpu.field_light(planes, &lit)?;
        with(planes)
    }

    /// The field, brought back to the host.
    pub fn field(&self, fov: &FieldOfView, light: &Light, focus: f64) -> Result<Field> {
        let rgb = self.field_with(fov, light, focus, |planes| self.gpu.download(&planes.rgb))?;
        Ok(Field {
            width: fov.optics.sensor.width,
            height: fov.optics.sensor.height,
            rgb: rgb.as_chunks::<3>().0.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::he::SectionConfig;
    use crate::render::render;
    use crate::tiles::TileDepth;
    use tetera_compute::budget::VramBudgets;

    fn sample(folds: bool) -> HeSample {
        HeSample::new(
            7,
            SectionConfig {
                folds,
                ..SectionConfig::default()
            },
        )
        .unwrap()
    }

    fn region(origin: (f64, f64), size: u32, pixel: f64, depth: Depth) -> Region {
        Region {
            origin,
            width: size,
            height: size,
            pixel,
            depth,
        }
    }

    /// Whether `gpu` is within the behaviour spec's tolerance of `cpu`: 2e-3 OD plus 1e-3 of
    /// the value on 99.99 % of values, and none beyond 0.05 OD. The worst difference, and
    /// the count beyond the first bound.
    fn within(cpu: &Rendered, gpu: &Rendered) -> (f32, usize, bool) {
        assert_eq!(
            (cpu.width, cpu.height, cpu.stains.len()),
            (gpu.width, gpu.height, gpu.stains.len())
        );
        let mut worst = 0.0f32;
        let mut beyond = 0;
        for (c, g) in cpu.stains.iter().zip(&gpu.stains) {
            for (a, b) in [(c.haematoxylin, g.haematoxylin), (c.eosin, g.eosin)] {
                let d = (a - b).abs();
                worst = worst.max(d);
                if d > 2e-3 + 1e-3 * a.abs() {
                    beyond += 1;
                }
            }
        }
        let values = cpu.stains.len() * 2;
        (worst, beyond, worst <= 0.05 && beyond * 10_000 <= values)
    }

    #[test]
    fn the_gpu_renders_within_the_tolerance_of_its_cpu_twin() {
        let gpu = tetera_compute::cuda_or_skip!();
        for (folds, region) in [
            (false, region((0.0, 0.0), 128, 2.0, Depth::Whole)),
            (false, region((1_095.0, -15.0), 128, 0.48, Depth::Whole)),
            (true, region((-100.0, -100.0), 96, 1.0, Depth::Whole)),
            (false, region((1_850.0, 1_850.0), 128, 0.048, Depth::Whole)),
            (
                false,
                region((1_850.0, 1_850.0), 128, 0.048, Depth::Range(1.5, 2.5)),
            ),
            (
                false,
                region((2_440.0, -20.0), 128, 0.48, Depth::Range(0.3, 3.8)),
            ),
        ] {
            let s = sample(folds);
            let (cpu, on_gpu) = (
                render(&s, &region).unwrap(),
                s.render_gpu(&gpu, &region).unwrap(),
            );
            let (worst, beyond, ok) = within(&cpu, &on_gpu);
            eprintln!(
                "{:?} at {} µm: worst {worst:.2e} OD, {beyond} beyond the first bound",
                region.depth, region.pixel
            );
            assert!(ok, "{region:?}: worst {worst}, {beyond} beyond 2e-3");
            assert!(
                cpu.stains.iter().any(|p| p.eosin > 0.0),
                "tissue in {region:?}"
            );
        }
    }

    #[test]
    fn a_tile_is_256_kib_on_the_gpu() {
        assert_eq!(TILE_BYTES, 262_144);
        assert_eq!(
            stains(&[0x3c00, 0x3800, 0, 0x4000]),
            [
                Stains {
                    haematoxylin: 1.0,
                    eosin: 0.5
                },
                Stains {
                    haematoxylin: 0.0,
                    eosin: 2.0
                }
            ]
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_bits_on_the_gpu() {
        let gpu = tetera_compute::cuda_or_skip!();
        let s = sample(false);
        let r = region((1_000.0, 1_000.0), 64, 0.5, Depth::Whole);
        let (a, b) = (
            gpu.download(&s.render_on(&gpu, &r).unwrap()).unwrap(),
            gpu.download(&s.render_on(&gpu, &r).unwrap()).unwrap(),
        );
        assert_eq!(a, b);
        let other = HeSample::new(7 ^ (1 << 40), SectionConfig::default()).unwrap();
        let c = gpu.download(&other.render_on(&gpu, &r).unwrap()).unwrap();
        assert!(
            a.iter().zip(&c).filter(|(x, y)| x != y).count() > a.len() / 2,
            "a seed one bit different"
        );
        assert!(s.render_on(&gpu, &Region { width: 0, ..r }).is_err());
    }

    #[test]
    fn tiles_meet_without_seams_and_the_cache_keeps_to_its_budget() {
        let gpu = Arc::new(tetera_compute::cuda_or_skip!());
        let budgets = VramBudgets::new(64 * TILE_BYTES);
        let budget = budgets.reserve("tiles", 3 * TILE_BYTES).unwrap();
        let tiles = HeTiles::new(gpu.clone(), sample(false), budget).unwrap();
        let key = |column, row| TileKey {
            level: 3,
            column,
            row,
            depth: TileDepth::Whole,
        };
        // Four tiles side by side match the region they make, rendered whole on the CPU.
        let whole = render(
            &sample(false),
            &region((0.0, 0.0), 512, key(0, 0).pixel(), Depth::Whole),
        )
        .unwrap();
        for (column, row) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let tile = tiles.tile_stains(key(column, row)).unwrap();
            let part = Rendered {
                width: TILE,
                height: TILE,
                stains: (0..TILE * TILE)
                    .map(|k| {
                        whole.stains[((row as u32 * TILE + k / TILE) * 512
                            + column as u32 * TILE
                            + k % TILE) as usize]
                    })
                    .collect(),
            };
            let (worst, beyond, ok) = within(
                &part,
                &Rendered {
                    width: TILE,
                    height: TILE,
                    stains: tile,
                },
            );
            assert!(ok, "tile ({column}, {row}): worst {worst}, {beyond} beyond");
            assert!(tiles.budget().used() <= 3 * TILE_BYTES, "within its budget");
        }
        let stats = tiles.stats();
        assert_eq!((stats.misses, stats.evictions, stats.tiles), (4, 1, 3));
        assert_eq!(
            tiles.budget().used(),
            3 * TILE_BYTES,
            "each tile held is granted, each let go is returned"
        );
        tiles.tile(key(1, 1)).unwrap();
        assert_eq!(tiles.stats().hits, 1);
        assert!(format!("{tiles:?}").contains("HeTiles"));
    }

    #[test]
    fn a_512_tile_on_the_gpu_takes_its_time() {
        let gpu = tetera_compute::cuda_or_skip!();
        let s = sample(false);
        let r = region((1_000.0, 1_000.0), 512, 0.048, Depth::Whole);
        s.render_on(&gpu, &r).unwrap();
        let ms = gpu.time_ms(|| s.render_on(&gpu, &r).map(drop)).unwrap();
        eprintln!("a 512 × 512 region at 50× on the GPU: {ms:.2} ms");
        assert!(ms > 0.0 && ms < 1_000.0, "{ms} ms");
    }

    fn tiles_on(gpu: CudaBackend, sample: HeSample, tiles: u64) -> HeTiles {
        let budgets = VramBudgets::new(4 * 1024 * TILE_BYTES);
        let budget = budgets.reserve("field", tiles * TILE_BYTES).unwrap();
        HeTiles::new(Arc::new(gpu), sample, budget).unwrap()
    }

    #[test]
    fn the_gpu_field_is_the_cpu_fields_twin() {
        use crate::optics::{FieldOfView, Light, Optics, Sensor, field};
        let gpu = tetera_compute::cuda_or_skip!();
        let tiles = tiles_on(gpu, sample(false), 256);
        let light = Light::default();
        for (m, na, angle, focus) in [
            (5.0, 0.14, 0.0, 2.0),
            (20.0, 0.42, 0.3, 1.0),
            (50.0, 0.55, -0.2, 2.5),
            (50.0, 0.55, 0.0, 8.0),
        ] {
            let fov = FieldOfView {
                centre: (1_000.0, 1_000.0),
                optics: Optics {
                    sensor: Sensor {
                        width: 160,
                        height: 120,
                        pixel: 2.4,
                    },
                    ..Optics::new(m, na, angle).unwrap()
                },
            };
            let (cpu, on_gpu) = (
                field(&sample(false), &fov, &light, focus).unwrap(),
                tiles.field(&fov, &light, focus).unwrap(),
            );
            assert_eq!(
                (on_gpu.width, on_gpu.height, on_gpu.rgb.len()),
                (cpu.width, cpu.height, cpu.rgb.len())
            );
            let worst = cpu
                .rgb
                .iter()
                .zip(&on_gpu.rgb)
                .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
                .fold(0.0, f32::max);
            eprintln!(
                "{m}× at {focus} µm: within {:.2e} of the light",
                worst / light.intensity
            );
            assert!(
                worst <= 3e-3 * light.intensity,
                "{m}×, focus {focus}: {worst}"
            );
        }
    }

    #[test]
    fn a_full_field_on_the_gpu_takes_its_time() {
        use crate::optics::{FieldOfView, Light, Optics};
        use std::time::Instant;
        let gpu = tetera_compute::cuda_or_skip!();
        let tiles = tiles_on(gpu.clone(), sample(false), 4_000);
        let light = Light::default();
        for (m, na) in [(5.0, 0.14), (50.0, 0.55)] {
            let optics = Optics::new(m, na, 0.0).unwrap();
            let mut times = Vec::new();
            for (k, dx) in [0.0, 0.0, 20.0].into_iter().enumerate() {
                let fov = FieldOfView {
                    centre: (2_000.0 + dx * optics.pixel() * 10.0, 2_000.0),
                    optics,
                };
                let start = Instant::now();
                tiles.field_with(&fov, &light, 2.0, |_| Ok(())).unwrap();
                gpu.synchronize().unwrap();
                times.push(start.elapsed().as_secs_f64() * 1e3);
                let _ = k;
            }
            eprintln!(
                "a full field at {m}×: {:.0} ms cold, {:.1} ms still, {:.1} ms moved 200 pixels; {:?}",
                times[0],
                times[1],
                times[2],
                tiles.stats()
            );
            assert!(times[1] < times[0], "still is cheaper than cold");
        }
    }
}
