//! The H&E sample on the GPU (P1.11b, ADR-0015): `tetera-compute`'s kernels, the twin of
//! `render`, and the GPU's tiles in their cache, within a VRAM budget.

use crate::he::{HeSample, Stains};
use crate::noise::Seed;
use crate::optics::{Field, FieldOfView, Grid, Light, Motion, blackbody, depth_groups, gaussian};
use crate::render::{Depth, Region, Rendered};
use crate::sensor::{Exposure, SensorModel};
use crate::tiles::{CacheStats, TILE, TileCache, TileDepth, TileKey};
use std::sync::{Arc, Mutex, PoisonError};
use tetera_compute::budget::{Budget, Grant};
use tetera_compute::field::{FieldLight, FieldPlanes, SensorView, TileGrid};
use tetera_compute::half;
use tetera_compute::he::{HeParams, HeRegion};
use tetera_compute::sensor::SensorParams;
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

/// What the sensor's kernel takes of `model` and `exposure`, for frame `frame` from `seed`.
pub fn sensor_params(
    model: &SensorModel,
    exposure: &Exposure,
    seed: u64,
    frame: u64,
) -> SensorParams {
    SensorParams {
        seed: Seed::new(seed).0,
        frame,
        quantum_efficiency: model.quantum_efficiency,
        full_well: model.full_well,
        read_noise: model.read_noise,
        dark_current: model.dark_current,
        black_level: f32::from(model.black_level),
        electrons_per_count: model.electrons_per_count,
        ms: exposure.time.as_secs_f32() * 1e3,
        gain: exposure.gain,
        bits: exposure.bits,
        binning: exposure.binning,
    }
}

/// A raw frame on the GPU from a `width` × `height` field there, into `raw`, its rows
/// `raw.len() / height` counts apart: the twin of `sensor::expose`.
#[allow(clippy::too_many_arguments)]
pub fn expose_on(
    gpu: &CudaBackend,
    rgb: &DeviceBuffer<f32>,
    width: u32,
    height: u32,
    model: &SensorModel,
    exposure: &Exposure,
    seed: u64,
    frame: u64,
    raw: &mut DeviceBuffer<u16>,
) -> Result<()> {
    exposure.check()?;
    gpu.sensor_expose(
        rgb,
        width,
        height,
        &sensor_params(model, exposure, seed, frame),
        raw,
    )
}

/// What a raw frame is made of besides its field: the sensor, the exposure, the seed and the
/// frame's number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shot {
    /// The sensor.
    pub model: SensorModel,
    /// How the frame is exposed.
    pub exposure: Exposure,
    /// The noise's seed.
    pub seed: u64,
    /// The frame's number.
    pub frame: u64,
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
    raw: Mutex<Option<DeviceBuffer<u16>>>,
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
            raw: Mutex::new(None),
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
        self.field_moving_with(fov, &Motion::STILL, light, focus, with)
    }

    /// The same, the stage moving as `motion` says: the twin of `optics::field_moving`.
    pub fn field_moving_with<R>(
        &self,
        fov: &FieldOfView,
        motion: &Motion,
        light: &Light,
        focus: f64,
        with: impl FnOnce(&FieldPlanes) -> Result<R>,
    ) -> Result<R> {
        let s = fov.optics.sensor;
        let groups = depth_groups(&fov.optics, focus, self.sample.section().thickness);
        let widest = groups.iter().map(|g| g.sigma).fold(0.0, f32::max);
        let grid = Grid::new(fov, widest, motion.travel(s.height));
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
            velocity: motion.velocity,
            exposure: motion.exposure,
            line_time: motion.line_time,
            samples: motion.samples(fov.optics.pixel()),
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

    /// A raw frame, row by row: the field made on the GPU, the stage moving as `motion` says,
    /// and exposed there as `shot` says. The twin of `optics::field_moving` then
    /// `sensor::expose`.
    pub fn frame(
        &self,
        fov: &FieldOfView,
        motion: &Motion,
        light: &Light,
        focus: f64,
        shot: &Shot,
    ) -> Result<Vec<u16>> {
        let s = fov.optics.sensor;
        let samples = s.width as usize * s.height as usize;
        self.exposed(fov, motion, light, focus, shot, samples, |raw| {
            self.gpu.download(raw)
        })
    }

    /// The same, straight into `out`: 16-bit little-endian counts, its sensor's rows evenly
    /// spaced through it, as a frame from a pool holds them; pinned memory takes it fastest.
    /// The sensor's launch refuses rows that don't fit, and the download a buffer of an odd
    /// length.
    pub fn frame_into(
        &self,
        fov: &FieldOfView,
        motion: &Motion,
        light: &Light,
        focus: f64,
        shot: &Shot,
        out: &mut [u8],
    ) -> Result<()> {
        let samples = out.len() / 2;
        self.exposed(fov, motion, light, focus, shot, samples, |raw| {
            self.gpu.download_into_bytes(raw, out)
        })
    }

    /// The raw frame on the GPU, in a buffer of `samples` counts kept for the next frame,
    /// handed to `then`.
    #[allow(clippy::too_many_arguments)]
    fn exposed<R>(
        &self,
        fov: &FieldOfView,
        motion: &Motion,
        light: &Light,
        focus: f64,
        shot: &Shot,
        samples: usize,
        then: impl FnOnce(&DeviceBuffer<u16>) -> Result<R>,
    ) -> Result<R> {
        shot.exposure.check()?;
        self.field_moving_with(fov, motion, light, focus, |planes| {
            let mut kept = self.raw.lock().unwrap_or_else(PoisonError::into_inner);
            if kept.as_ref().is_none_or(|r| r.len() != samples) {
                *kept = None;
                *kept = Some(self.gpu.zeros(samples)?);
            }
            let buffer = kept.as_mut().expect("raw made");
            expose_on(
                &self.gpu,
                &planes.rgb,
                planes.width,
                planes.height,
                &shot.model,
                &shot.exposure,
                shot.seed,
                shot.frame,
                buffer,
            )?;
            then(buffer)
        })
    }

    /// The field, brought back to the host.
    pub fn field(&self, fov: &FieldOfView, light: &Light, focus: f64) -> Result<Field> {
        self.field_moving(fov, &Motion::STILL, light, focus)
    }

    /// The field, the stage moving as `motion` says, brought back to the host.
    pub fn field_moving(
        &self,
        fov: &FieldOfView,
        motion: &Motion,
        light: &Light,
        focus: f64,
    ) -> Result<Field> {
        let rgb = self.field_moving_with(fov, motion, light, focus, |planes| {
            self.gpu.download(&planes.rgb)
        })?;
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
        // Two sizes of sensor through one cache, its planes made again for the second.
        for (m, na, angle, focus, width) in [
            (5.0, 0.14, 0.0, 2.0, 160),
            (20.0, 0.42, 0.3, 1.0, 96),
            (50.0, 0.55, -0.2, 2.5, 160),
            (50.0, 0.55, 0.0, 8.0, 160),
        ] {
            let fov = FieldOfView {
                centre: (1_000.0, 1_000.0),
                optics: Optics {
                    sensor: Sensor {
                        width,
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
    fn the_moving_gpu_field_is_the_cpu_fields_twin() {
        use crate::optics::{FieldOfView, Light, Optics, Sensor, field_moving};
        let gpu = tetera_compute::cuda_or_skip!();
        let tiles = tiles_on(gpu, sample(false), 256);
        let light = Light::default();
        for (m, angle, velocity, exposure, line_time) in [
            (20.0, 0.3, (2_000.0, 500.0), 0.004, 0.0),
            (50.0, 0.0, (-800.0, 300.0), 0.002, 2e-5),
            (5.0, -0.2, (5_000.0, 0.0), 0.0, 5e-5),
        ] {
            let fov = FieldOfView {
                centre: (1_000.0, 1_000.0),
                optics: Optics {
                    sensor: Sensor {
                        width: 128,
                        height: 96,
                        pixel: 2.4,
                    },
                    ..Optics::new(m, 0.42, angle).unwrap()
                },
            };
            let motion = Motion {
                velocity,
                exposure,
                line_time,
            };
            let cpu = field_moving(&sample(false), &fov, &motion, &light, 2.0).unwrap();
            let on_gpu = tiles.field_moving(&fov, &motion, &light, 2.0).unwrap();
            let worst = cpu
                .rgb
                .iter()
                .zip(&on_gpu.rgb)
                .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
                .fold(0.0, f32::max);
            eprintln!(
                "{m}×, {} samples: within {:.2e} of the light",
                motion.samples(fov.optics.pixel()),
                worst / light.intensity
            );
            assert!(worst <= 3e-3 * light.intensity, "{m}×: {worst}");
            assert_ne!(on_gpu, tiles.field(&fov, &light, 2.0).unwrap(), "it moved");
        }
    }

    /// The mean count of `colour` in a raw frame `width` wide.
    fn mean_of(raw: &[u16], width: u32, colour: usize) -> f64 {
        let v: Vec<f64> = (0..raw.len() as u32)
            .filter(|k| crate::sensor::bayer(k % width, k / width) == colour)
            .map(|k| f64::from(raw[k as usize]))
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    }

    /// An exposure of `ms` ms at `gain`, in `bits`, binned `binning`.
    fn exposure(ms: u64, gain: f32, bits: u8, binning: u32) -> Exposure {
        Exposure {
            time: std::time::Duration::from_millis(ms),
            gain,
            bits,
            binning,
        }
    }

    #[test]
    fn the_gpu_sensor_is_the_cpu_sensors_twin() {
        use crate::sensor::expose;
        let gpu = tetera_compute::cuda_or_skip!();
        let (w, h) = (256u32, 192u32);
        // A ramp from dark to bright, unlike in each colour.
        let field: Vec<[f32; 3]> = (0..w * h)
            .map(|k| {
                let t = (k % w) as f32 / w as f32;
                [300.0 * t, 500.0 * t * t, 2.0 + 100.0 * (1.0 - t)]
            })
            .collect();
        let flat: Vec<f32> = field.iter().flatten().copied().collect();
        let rgb = gpu.upload(&flat).unwrap();
        let mut raw = gpu.zeros::<u16>((w * h) as usize).unwrap();
        let model = SensorModel::default();
        for e in [
            exposure(10, 1.0, 12, 1),
            exposure(1, 1.0, 12, 1),
            exposure(4, 2.0, 12, 2),
            exposure(10, 1.0, 8, 1),
        ] {
            let cpu = expose(&field, w, h, &model, &e, 21, 9).unwrap();
            expose_on(&gpu, &rgb, w, h, &model, &e, 21, 9, &mut raw).unwrap();
            let on_gpu = gpu.download(&raw).unwrap();
            let differ = cpu.iter().zip(&on_gpu).filter(|(a, b)| a != b).count();
            let worst = cpu
                .iter()
                .zip(&on_gpu)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            eprintln!(
                "{e:?}: {differ} of {} differ, by {worst} at most",
                cpu.len()
            );
            // Single precision's functions round a draw across a count's edge now and then.
            assert!(
                worst <= 1 && differ * 200 < cpu.len(),
                "{e:?}: {differ}, {worst}"
            );
            expose_on(&gpu, &rgb, w, h, &model, &e, 21, 9, &mut raw).unwrap();
            assert_eq!(gpu.download(&raw).unwrap(), on_gpu, "the same bits again");
            expose_on(&gpu, &rgb, w, h, &model, &e, 21, 10, &mut raw).unwrap();
            assert_ne!(
                gpu.download(&raw).unwrap(),
                on_gpu,
                "the next frame's noise"
            );
        }
        assert!(
            expose_on(
                &gpu,
                &rgb,
                w,
                h,
                &model,
                &exposure(1, 0.5, 12, 1),
                1,
                0,
                &mut raw
            )
            .is_err()
        );
    }

    #[test]
    fn a_raw_frame_is_made_on_the_gpu() {
        use crate::optics::{FieldOfView, Light, Optics, Sensor, field_moving};
        use crate::sensor::{bayer, expose};
        let gpu = tetera_compute::cuda_or_skip!();
        let tiles = tiles_on(gpu, sample(false), 256);
        let light = Light::default();
        let fov = FieldOfView {
            centre: (1_000.0, 1_000.0),
            optics: Optics {
                sensor: Sensor {
                    width: 160,
                    height: 120,
                    pixel: 2.4,
                },
                ..Optics::new(20.0, 0.42, 0.0).unwrap()
            },
        };
        let motion = Motion {
            velocity: (1_000.0, 0.0),
            exposure: 0.005,
            line_time: 1e-5,
        };
        let shot = Shot {
            model: SensorModel::default(),
            exposure: exposure(5, 1.0, 12, 1),
            seed: 3,
            frame: 1,
        };
        let raw = tiles.frame(&fov, &motion, &light, 2.0, &shot).unwrap();
        assert_eq!(raw, tiles.frame(&fov, &motion, &light, 2.0, &shot).unwrap());
        let field = field_moving(&sample(false), &fov, &motion, &light, 2.0).unwrap();
        let cpu = expose(&field.rgb, 160, 120, &shot.model, &shot.exposure, 3, 1).unwrap();
        for colour in 0..3 {
            let mean = |r: &[u16]| {
                let v: Vec<f64> = (0..160 * 120)
                    .filter(|k| bayer(k % 160, k / 160) == colour)
                    .map(|k| f64::from(r[k as usize]))
                    .collect();
                v.iter().sum::<f64>() / v.len() as f64
            };
            let (a, b) = (mean(&cpu), mean(&raw));
            assert!(
                (a - b).abs() < 0.01 * a,
                "colour {colour}: {a} on the CPU, {b} on the GPU"
            );
        }
        // Another size of sensor: its buffer made again.
        let smaller = FieldOfView {
            optics: Optics {
                sensor: fov.optics.sensor.binned(2),
                ..fov.optics
            },
            ..fov
        };
        let binned = Shot {
            exposure: exposure(5, 1.0, 12, 2),
            ..shot
        };
        let half = tiles
            .frame(&smaller, &motion, &light, 2.0, &binned)
            .unwrap();
        assert_eq!(half.len(), 80 * 60);
        let (full, quarter) = (mean_of(&raw, 160, 1), mean_of(&half, 80, 1));
        let black = f64::from(shot.model.black_level);
        assert!(
            ((quarter - black) / (full - black) - 4.0).abs() < 0.2,
            "binned 2 × 2, four pixels' light: {full} then {quarter}"
        );
        // Straight into a frame's bytes, its rows padded: the same counts.
        let stride = 2 * 160 + 64;
        let mut bytes = vec![0u8; stride * 120];
        tiles
            .frame_into(&fov, &motion, &light, 2.0, &shot, &mut bytes)
            .unwrap();
        for (row, line) in bytes.chunks(stride).enumerate() {
            let counts: Vec<u16> = line[..320]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| u16::from_le_bytes(*p))
                .collect();
            assert_eq!(counts, raw[row * 160..][..160], "row {row}");
        }
        // Rows a byte apart from whole counts, too narrow, not whole rows, a byte over.
        for len in [321 * 120, 318 * 120, 320 * 119, 320 * 120 + 1] {
            let mut out = vec![0u8; len];
            assert!(
                tiles
                    .frame_into(&fov, &motion, &light, 2.0, &shot, &mut out)
                    .is_err(),
                "{len} bytes"
            );
        }
        let bad = Shot {
            exposure: exposure(5, 1.0, 10, 1),
            ..shot
        };
        assert!(tiles.frame(&fov, &motion, &light, 2.0, &bad).is_err());
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
