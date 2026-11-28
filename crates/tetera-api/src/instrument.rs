//! The virtual instrument, assembled from configuration: the simulated clock, the virtual
//! stage, camera, objective mount and light, and the sample on the stage, rendered on the
//! GPU when there is one. The server registers it as a service, and the Things reach it
//! through the server.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use tetera_core::frame::meta::LightState;
use tetera_core::{Error, Result, StagePos, Um};
use tetera_hal::illumination::Illumination;
use tetera_hal::objective::{ManualObjectiveMount, ObjectiveCatalogue, ObjectiveMount};
use tetera_sim::camera::{Backend, Scene, VirtualCamera, VirtualCameraConfig};
use tetera_sim::light::CHANNEL;
use tetera_sim::optics::{MICHROME_20, Sensor};
use tetera_sim::{
    HeSample, Rate, SectionConfig, SimClock, VirtualIllumination, VirtualStage, VirtualStageConfig,
};

/// The objectives Tetera ships, `configs/objectives.toml`, used when no catalogue is named.
pub const SHIPPED_OBJECTIVES: &str = include_str!("../../../configs/objectives.toml");

/// The sample on the stage: H&E from a seed, a section of a thickness, folded or not.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    /// The seed the sample grows from.
    pub seed: u64,
    /// The section's thickness, µm: 3 to 5.
    pub thickness: f32,
    /// Whether the section lies folded in places.
    pub folds: bool,
}

impl Default for Sample {
    fn default() -> Self {
        Self {
            seed: 7,
            thickness: 4.0,
            folds: false,
        }
    }
}

impl Sample {
    /// Whether it is one the simulator makes.
    pub fn check(&self) -> Result<()> {
        self.he().map(drop).map_err(|e| {
            Error::invalid_argument(format!("a sample from seed {}: {}", self.seed, e.message()))
        })
    }

    fn he(&self) -> Result<HeSample> {
        HeSample::new(
            self.seed,
            SectionConfig {
                thickness: self.thickness,
                folds: self.folds,
                ..SectionConfig::default()
            },
        )
    }
}

/// The camera's sensor in the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SensorSize {
    /// The whole MIChrome 20 on the GPU, a 912 × 608 sensor on the CPU.
    Auto,
    /// The whole MIChrome 20, 5472 × 3648.
    Full,
    /// 912 × 608, a sixth each way: fast enough on the CPU.
    Small,
    /// 96 × 72, for tests.
    Tiny,
}

impl SensorSize {
    fn sensor(self, gpu: bool) -> Sensor {
        let small = |w, h| Sensor {
            width: w,
            height: h,
            pixel: 2.4,
        };
        match (self, gpu) {
            (Self::Full, _) | (Self::Auto, true) => MICHROME_20,
            (Self::Small, _) | (Self::Auto, false) => small(912, 608),
            (Self::Tiny, _) => small(96, 72),
        }
    }
}

/// How the virtual instrument is made: `configs/tetera.toml`'s `[simulator]` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SimulatorConfig {
    /// The sample on the stage at the start, if any.
    pub sample: Option<Sample>,
    /// How fast the simulated clock runs: 1 is real time.
    pub clock_rate: f64,
    /// Whether the clock moves only when stepped, as tests step it; `clock_rate` is then
    /// ignored.
    pub stepped: bool,
    /// Whether to render on the GPU when there is one.
    pub gpu: bool,
    /// The camera's sensor.
    pub sensor: SensorSize,
    /// The light at full intensity through bare glass, photoelectrons a ms.
    pub light: f32,
    /// How uneven the light's field is.
    pub unevenness: f32,
    /// The stage's z, µm, at which the section's lower face is in focus.
    pub section_z: f64,
    /// Where the stage starts, µm.
    pub start: [f64; 3],
    /// The GPU's memory the sample's tiles may take, MiB.
    pub vram_mib: u64,
    /// The camera noise's seed.
    pub noise_seed: u64,
}

impl Default for SimulatorConfig {
    /// The H&E sample from seed 7, in real time, on the GPU when there is one, the stage over
    /// tissue and in focus.
    fn default() -> Self {
        Self {
            sample: Some(Sample::default()),
            clock_rate: 1.0,
            stepped: false,
            gpu: true,
            sensor: SensorSize::Auto,
            light: 1_000.0,
            unevenness: 0.05,
            section_z: 1_000.0,
            start: [2_000.0, 2_000.0, 1_002.0],
            vram_mib: 3_072,
            noise_seed: 1,
        }
    }
}

/// The objectives: `configs/tetera.toml`'s `[objectives]` table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObjectivesConfig {
    /// The catalogue's file; the shipped catalogue when `None`.
    pub catalogue: Option<PathBuf>,
    /// The objective fitted at the start, by name, if any.
    pub fitted: Option<String>,
}

impl Default for ObjectivesConfig {
    /// The shipped catalogue, 20× fitted.
    fn default() -> Self {
        Self {
            catalogue: None,
            fitted: Some("20x".into()),
        }
    }
}

/// Where the camera renders, as `system` reports it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Compute {
    /// `cuda` or `cpu`.
    pub backend: String,
    /// The GPU's name.
    pub device: Option<String>,
    /// The GPU's memory, bytes.
    pub memory: Option<u64>,
    /// Its compute capability, such as `8.6`.
    pub compute_capability: Option<String>,
    /// The newest CUDA its driver runs, such as `13.5`.
    pub driver_cuda: Option<String>,
    /// Why not the GPU, on the CPU.
    pub reason: Option<String>,
    /// The camera's sensor, pixels across and down.
    pub sensor: [u32; 2],
}

/// The virtual instrument.
pub struct Instrument {
    /// The simulated clock everything runs on.
    pub clock: Arc<SimClock>,
    /// The stage.
    pub stage: Arc<VirtualStage>,
    /// The camera.
    pub camera: Arc<VirtualCamera>,
    /// The objective mount.
    pub objectives: Arc<ManualObjectiveMount>,
    /// The light.
    pub light: Arc<VirtualIllumination>,
    sample: Mutex<Option<Sample>>,
    compute: Compute,
}

impl std::fmt::Debug for Instrument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instrument")
            .field("compute", &self.compute)
            .field("sample", &self.sample())
            .finish_non_exhaustive()
    }
}

impl Instrument {
    /// The instrument `simulator` and `objectives` describe, its light on.
    pub fn new(simulator: &SimulatorConfig, objectives: &ObjectivesConfig) -> Result<Self> {
        let rate = if simulator.stepped {
            Rate::Stepped
        } else if simulator.clock_rate == 1.0 {
            Rate::RealTime
        } else {
            Rate::Faster(simulator.clock_rate)
        };
        let clock = Arc::new(SimClock::new(rate)?);
        let [x, y, z] = simulator.start;
        let stage = Arc::new(VirtualStage::new(
            clock,
            VirtualStageConfig {
                start: StagePos::new(Um(x), Um(y), Um(z)),
                ..VirtualStageConfig::default()
            },
        )?);
        let catalogue = match &objectives.catalogue {
            Some(path) => ObjectiveCatalogue::load(path)?,
            None => ObjectiveCatalogue::from_toml(SHIPPED_OBJECTIVES)?,
        };
        let mount = Arc::new(ManualObjectiveMount::new(catalogue));
        mount.fit(objectives.fitted.as_deref())?;
        let light = Arc::new(VirtualIllumination::new(
            simulator.light,
            simulator.unevenness,
        ));
        light.set(
            CHANNEL,
            LightState {
                on: true,
                intensity: 1.0,
                colour_temperature_k: Some(tetera_sim::light::DEFAULT_KELVIN),
            },
        )?;
        let first = simulator.sample.unwrap_or_default();
        let (backend, compute) = backend(simulator)?;
        let sensor = simulator.sensor.sensor(compute.backend == "cuda");
        let scene = Scene {
            stage: stage.clone(),
            objectives: mount.clone(),
            light: light.clone(),
            sample: first.he()?,
        };
        let camera = Arc::new(VirtualCamera::new(
            scene,
            VirtualCameraConfig {
                sensor,
                seed: simulator.noise_seed,
                section_z: simulator.section_z,
                ..VirtualCameraConfig::default()
            },
            backend,
        )?);
        if simulator.sample.is_none() {
            camera.set_sample(None)?;
        }
        Ok(Self {
            clock: stage.clock().clone(),
            stage,
            camera,
            objectives: mount,
            light,
            sample: Mutex::new(simulator.sample),
            compute: Compute {
                sensor: [sensor.width, sensor.height],
                ..compute
            },
        })
    }

    /// Where the camera renders.
    pub fn compute(&self) -> &Compute {
        &self.compute
    }

    /// The sample on the stage, if any.
    pub fn sample(&self) -> Option<Sample> {
        *self.sample.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Puts `sample` on the stage in place of what was there.
    pub fn load(&self, sample: Sample) -> Result<()> {
        self.camera.set_sample(Some(sample.he()?))?;
        *self.sample.lock().unwrap_or_else(PoisonError::into_inner) = Some(sample);
        Ok(())
    }

    /// Takes the slide off the stage: the camera sees the light alone.
    pub fn unload(&self) -> Result<()> {
        self.camera.set_sample(None)?;
        *self.sample.lock().unwrap_or_else(PoisonError::into_inner) = None;
        Ok(())
    }
}

/// The camera's backend, and what `system` says of it.
fn backend(simulator: &SimulatorConfig) -> Result<(Backend, Compute)> {
    let cpu = |reason: String| Compute {
        backend: "cpu".into(),
        device: None,
        memory: None,
        compute_capability: None,
        driver_cuda: None,
        reason: Some(reason),
        sensor: [0, 0],
    };
    if !simulator.gpu {
        return Ok((
            Backend::Cpu,
            cpu("the configuration asks for the CPU".into()),
        ));
    }
    #[cfg(feature = "cuda")]
    {
        use tetera_compute::select::{Selection, SystemProbe, select};
        match select(&SystemProbe) {
            Selection::Cuda(device) => {
                let gpu = Arc::new(tetera_compute::CudaBackend::new(device.ordinal)?);
                let mib = simulator.vram_mib << 20;
                let budgets = tetera_compute::budget::VramBudgets::new(mib);
                let budget = budgets.reserve("camera", mib)?;
                let compute = Compute {
                    backend: "cuda".into(),
                    device: Some(device.name.clone()),
                    memory: Some(device.total_memory),
                    compute_capability: Some(format!(
                        "{}.{}",
                        device.compute_capability.0, device.compute_capability.1
                    )),
                    driver_cuda: Some(format!("{}.{}", device.driver_cuda.0, device.driver_cuda.1)),
                    reason: None,
                    sensor: [0, 0],
                };
                Ok((Backend::Gpu { gpu, budget }, compute))
            }
            Selection::Cpu { reason } => Ok((Backend::Cpu, cpu(reason))),
        }
    }
    #[cfg(not(feature = "cuda"))]
    Ok((Backend::Cpu, cpu("this build has no `cuda` feature".into())))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An instrument for tests: on the CPU, a tiny sensor, ten times as fast as real time.
    pub(crate) fn instrument() -> Instrument {
        Instrument::new(
            &SimulatorConfig {
                gpu: false,
                sensor: SensorSize::Tiny,
                clock_rate: 10.0,
                ..SimulatorConfig::default()
            },
            &ObjectivesConfig::default(),
        )
        .unwrap()
    }

    /// The same, on a clock that moves only when the test steps it.
    pub(crate) fn stepped() -> Instrument {
        Instrument::new(
            &SimulatorConfig {
                gpu: false,
                sensor: SensorSize::Tiny,
                stepped: true,
                ..SimulatorConfig::default()
            },
            &ObjectivesConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn the_instrument_is_made_as_its_configuration_says() {
        let i = instrument();
        assert_eq!(i.compute().backend, "cpu");
        assert_eq!(i.compute().sensor, [96, 72]);
        assert_eq!(i.sample(), Some(Sample::default()));
        assert_eq!(
            i.objectives.fitted().map(|o| o.name).as_deref(),
            Some("20x")
        );
        assert!(i.light.light().intensity > 0.0, "the light on");
        let p = i.stage.position();
        assert_eq!((p.x.um(), p.y.um(), p.z.um()), (2_000.0, 2_000.0, 1_002.0));
        assert!(format!("{i:?}").contains("cpu"));
    }

    #[test]
    fn a_sample_is_loaded_and_unloaded() {
        let i = instrument();
        let other = Sample {
            seed: 9,
            thickness: 3.5,
            folds: true,
        };
        i.load(other).unwrap();
        assert_eq!(i.sample(), Some(other));
        assert!(i.camera.sample().is_some());
        i.unload().unwrap();
        assert_eq!(i.sample(), None);
        assert_eq!(i.camera.sample(), None);
        let thick = Sample {
            thickness: 9.0,
            ..other
        };
        assert!(thick.check().is_err());
        assert!(i.load(thick).is_err(), "a section 9 µm thick");
        assert_eq!(i.sample(), None, "nothing changed");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn on_the_gpu_the_camera_takes_the_whole_sensor_within_its_budget() {
        use tetera_hal::camera::Camera;
        if !tetera_compute::select::cuda_tests_enabled() {
            eprintln!("skipped: set TETERA_CUDA=1 to run CUDA tests");
            return;
        }
        let i = Instrument::new(
            &SimulatorConfig {
                clock_rate: 10.0,
                vram_mib: 512,
                ..SimulatorConfig::default()
            },
            &ObjectivesConfig::default(),
        )
        .unwrap();
        let c = i.compute();
        assert_eq!(c.backend, "cuda");
        assert_eq!(c.sensor, [5_472, 3_648]);
        assert!(
            c.device.is_some() && c.memory.is_some() && c.reason.is_none(),
            "{c:?}"
        );
        // A frame on the GPU, its tiles taken from the budget.
        let frame = i.camera.snap(std::time::Duration::from_secs(10)).unwrap();
        assert_eq!(frame.layout().width, 5_472);
    }

    #[test]
    fn a_configuration_it_cannot_make_is_refused() {
        let none_fitted = Instrument::new(
            &SimulatorConfig {
                gpu: false,
                sensor: SensorSize::Tiny,
                sample: None,
                ..SimulatorConfig::default()
            },
            &ObjectivesConfig {
                fitted: None,
                ..ObjectivesConfig::default()
            },
        )
        .unwrap();
        assert_eq!(none_fitted.sample(), None);
        assert_eq!(none_fitted.camera.sample(), None);
        assert!(none_fitted.objectives.fitted().is_none());
        for (simulator, objectives) in [
            (
                SimulatorConfig {
                    clock_rate: 0.0,
                    ..SimulatorConfig::default()
                },
                ObjectivesConfig::default(),
            ),
            (
                SimulatorConfig::default(),
                ObjectivesConfig {
                    fitted: Some("7x".into()),
                    ..ObjectivesConfig::default()
                },
            ),
            (
                SimulatorConfig::default(),
                ObjectivesConfig {
                    catalogue: Some("/nowhere/objectives.toml".into()),
                    ..ObjectivesConfig::default()
                },
            ),
        ] {
            let simulator = SimulatorConfig {
                gpu: false,
                sensor: SensorSize::Tiny,
                ..simulator
            };
            assert!(
                Instrument::new(&simulator, &objectives).is_err(),
                "{simulator:?} {objectives:?}"
            );
        }
        for (size, gpu, expected) in [
            (SensorSize::Auto, true, (5_472, 3_648)),
            (SensorSize::Auto, false, (912, 608)),
            (SensorSize::Full, false, (5_472, 3_648)),
            (SensorSize::Small, true, (912, 608)),
            (SensorSize::Tiny, true, (96, 72)),
        ] {
            let s = size.sensor(gpu);
            assert_eq!((s.width, s.height), expected, "{size:?} {gpu}");
        }
    }
}
