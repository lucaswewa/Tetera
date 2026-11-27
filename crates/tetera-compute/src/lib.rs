//! Tetera's compute backends (ADR-0009): the CPU, with rayon, which is the reference and the
//! fallback, and CUDA, through `cudarc` with its libraries loaded at run time.
//!
//! [`Compute::detect`] chooses: CUDA when the build has the `cuda` feature, a driver loads
//! and a device is there; the CPU otherwise, or when `TETERA_FORCE_CPU=1`. Either way it says
//! why. [`features::check_all`] tries what CUDA offers on this machine under WSL2.
//!
//! Every operation in [`ops`] has a CPU twin and, with the `cuda` feature, a CUDA kernel
//! (ADR-0010): [`twin`] proves the two agree, and [`budget`] shares out the GPU's memory.
//! [`imaging`] holds the imaging kernels, whose CPU twins are in `tetera-imaging`.

pub mod budget;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod features;
pub mod fused;
pub mod imaging;
pub mod isp;
#[cfg(feature = "cuda")]
pub mod kernels;
pub mod ops;
pub mod select;
pub mod twin;

pub use budget::{Budget, Grant, VramBudgets};
pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::{CudaBackend, DeviceBuffer, PinnedAllocator};
pub use select::{BackendKind, DeviceInfo, Probe, Selection, SystemProbe, select};
pub use twin::{Comparison, Tolerance, compare};

use std::fmt;
use tetera_core::Result;

/// The backends this process computes with: always the CPU, and CUDA when it was chosen.
pub struct Compute {
    cpu: CpuBackend,
    #[cfg(feature = "cuda")]
    cuda: Option<CudaBackend>,
    why: String,
}

impl Compute {
    /// Looks at this machine and this build, and chooses.
    pub fn detect() -> Result<Self> {
        Self::from_probe(&SystemProbe)
    }

    /// Chooses from what `probe` reports. A device chosen that then won't start leaves the
    /// CPU in use, with the reason.
    pub fn from_probe(probe: &dyn Probe) -> Result<Self> {
        let cpu = CpuBackend::new(0)?;
        Ok(match select(probe) {
            Selection::Cpu { reason } => Self::on_cpu(cpu, reason),
            Selection::Cuda(device) => Self::start_cuda(cpu, &device),
        })
    }

    /// The CPU alone.
    pub fn cpu_only() -> Result<Self> {
        Ok(Self::on_cpu(
            CpuBackend::new(0)?,
            "asked for the CPU".to_string(),
        ))
    }

    fn on_cpu(cpu: CpuBackend, why: String) -> Self {
        Self {
            cpu,
            #[cfg(feature = "cuda")]
            cuda: None,
            why,
        }
    }

    #[cfg(feature = "cuda")]
    fn start_cuda(cpu: CpuBackend, device: &DeviceInfo) -> Self {
        match CudaBackend::new(device.ordinal) {
            Ok(gpu) => Self {
                cpu,
                cuda: Some(gpu),
                why: format!("CUDA on {}, device {}", device.name, device.ordinal),
            },
            Err(e) => Self::on_cpu(cpu, format!("CUDA on {} wouldn't start: {e}", device.name)),
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn start_cuda(cpu: CpuBackend, _device: &DeviceInfo) -> Self {
        Self::on_cpu(cpu, "this build has no `cuda` feature".to_string())
    }

    /// The backend that does the work.
    pub fn kind(&self) -> BackendKind {
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            return BackendKind::Cuda;
        }
        BackendKind::Cpu
    }

    /// Why this backend: the device, or why not CUDA.
    pub fn why(&self) -> &str {
        &self.why
    }

    /// The CPU, always there.
    pub fn cpu(&self) -> &CpuBackend {
        &self.cpu
    }

    /// The CUDA device, when it was chosen.
    #[cfg(feature = "cuda")]
    pub fn cuda(&self) -> Option<&CudaBackend> {
        self.cuda.as_ref()
    }
}

impl fmt::Debug for Compute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Compute")
            .field("kind", &self.kind())
            .field("why", &self.why)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::{FakeProbe, fake_device};

    #[test]
    fn forcing_the_cpu_is_obeyed_and_explained() {
        let probe = FakeProbe {
            force_cpu: true,
            cuda_built: true,
            devices: Ok(vec![fake_device(0, 8)]),
        };
        let compute = Compute::from_probe(&probe).unwrap();
        assert_eq!(compute.kind(), BackendKind::Cpu);
        assert_eq!(compute.why(), "TETERA_FORCE_CPU is set");
        assert!(compute.cpu().threads() >= 1);
        assert!(format!("{compute:?}").contains("Cpu"));
    }

    #[test]
    fn a_missing_driver_falls_back_to_the_cpu() {
        let probe = FakeProbe {
            force_cpu: false,
            cuda_built: true,
            devices: Err("libcuda.so.1 doesn't load".into()),
        };
        let compute = Compute::from_probe(&probe).unwrap();
        assert_eq!(compute.kind(), BackendKind::Cpu);
        assert_eq!(compute.why(), "no CUDA driver: libcuda.so.1 doesn't load");
    }

    #[test]
    fn a_chosen_device_that_wont_start_leaves_the_cpu() {
        let probe = FakeProbe {
            force_cpu: false,
            cuda_built: true,
            devices: Ok(vec![fake_device(99, 8)]),
        };
        let compute = Compute::from_probe(&probe).unwrap();
        assert_eq!(compute.kind(), BackendKind::Cpu);
        let expected = if cfg!(feature = "cuda") {
            "CUDA on GPU 99 wouldn't start"
        } else {
            "this build has no `cuda` feature"
        };
        assert!(compute.why().starts_with(expected), "{}", compute.why());
    }

    #[test]
    fn asking_for_the_cpu_gets_it() {
        let compute = Compute::cpu_only().unwrap();
        assert_eq!(
            (compute.kind(), compute.why()),
            (BackendKind::Cpu, "asked for the CPU")
        );
    }

    #[test]
    fn detection_follows_the_build_and_the_switches() {
        let compute = Compute::detect().unwrap();
        if !cfg!(feature = "cuda") {
            assert_eq!(compute.kind(), BackendKind::Cpu);
        } else if crate::select::cuda_tests_enabled() && !SystemProbe.force_cpu() {
            assert_eq!(compute.kind(), BackendKind::Cuda, "{}", compute.why());
            assert!(compute.why().starts_with("CUDA on "), "{}", compute.why());
        }
    }
}
