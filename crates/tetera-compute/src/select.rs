//! Choosing a backend (ADR-0009): CUDA when the build has it, a driver loads and a device is
//! there; the CPU otherwise, or when `TETERA_FORCE_CPU` asks. The choice is a function of
//! what a [`Probe`] reports, so tests can fake any machine.

use std::fmt;

/// Which backend does the work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// The CPU, with rayon: the reference, and the fallback.
    Cpu,
    /// An NVIDIA GPU, through CUDA.
    Cuda,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
        })
    }
}

/// A CUDA device, as detection finds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Its number, from 0.
    pub ordinal: usize,
    /// Its name, such as `NVIDIA GeForce RTX 3070 Ti Laptop GPU`.
    pub name: String,
    /// Its compute capability, major and minor.
    pub compute_capability: (u32, u32),
    /// Its memory, in bytes.
    pub total_memory: u64,
    /// Its memory free when detected, in bytes.
    pub free_memory: u64,
    /// The newest CUDA version the driver runs, major and minor.
    pub driver_cuda: (u32, u32),
}

/// What a machine offers, as far as choosing a backend goes.
pub trait Probe {
    /// Whether the CPU is asked for, whatever else is there.
    fn force_cpu(&self) -> bool;
    /// Whether this build has the CUDA path.
    fn cuda_built(&self) -> bool;
    /// The CUDA devices, or why there are none to ask.
    fn devices(&self) -> Result<Vec<DeviceInfo>, String>;
}

/// A backend chosen, with the device or the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// The CPU, and why.
    Cpu {
        /// Why not CUDA.
        reason: String,
    },
    /// CUDA, on this device.
    Cuda(DeviceInfo),
}

/// Chooses a backend from what `probe` reports. Among several devices it takes the one with
/// the most memory, the lowest number first among equals.
pub fn select(probe: &dyn Probe) -> Selection {
    let cpu = |reason: &str| Selection::Cpu {
        reason: reason.to_string(),
    };
    if probe.force_cpu() {
        return cpu("TETERA_FORCE_CPU is set");
    }
    if !probe.cuda_built() {
        return cpu("this build has no `cuda` feature");
    }
    match probe.devices() {
        Err(why) => cpu(&format!("no CUDA driver: {why}")),
        Ok(devices) => devices
            .into_iter()
            .max_by(|a, b| {
                a.total_memory
                    .cmp(&b.total_memory)
                    .then(b.ordinal.cmp(&a.ordinal))
            })
            .map_or_else(|| cpu("the driver finds no CUDA device"), Selection::Cuda),
    }
}

/// Whether an environment variable's value turns a switch on: `1`, `true`, `yes` or `on`, in
/// any case.
pub fn env_flag(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        ["1", "true", "yes", "on"]
            .iter()
            .any(|on| v.trim().eq_ignore_ascii_case(on))
    })
}

/// Whether tests that need a GPU should run: `TETERA_CUDA=1` (plan §5.4).
pub fn cuda_tests_enabled() -> bool {
    env_flag(std::env::var("TETERA_CUDA").ok().as_deref())
}

/// This machine and this build, as they are.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemProbe;

impl Probe for SystemProbe {
    fn force_cpu(&self) -> bool {
        env_flag(std::env::var("TETERA_FORCE_CPU").ok().as_deref())
    }

    fn cuda_built(&self) -> bool {
        cfg!(feature = "cuda")
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>, String> {
        #[cfg(feature = "cuda")]
        return crate::cuda::devices();
        #[cfg(not(feature = "cuda"))]
        Err("this build has no `cuda` feature".to_string())
    }
}

#[cfg(test)]
pub(crate) use tests::{FakeProbe, device as fake_device};

#[cfg(test)]
mod tests {
    use super::*;

    /// A machine described in a test.
    pub(crate) struct FakeProbe {
        pub force_cpu: bool,
        pub cuda_built: bool,
        pub devices: Result<Vec<DeviceInfo>, String>,
    }

    impl Probe for FakeProbe {
        fn force_cpu(&self) -> bool {
            self.force_cpu
        }
        fn cuda_built(&self) -> bool {
            self.cuda_built
        }
        fn devices(&self) -> Result<Vec<DeviceInfo>, String> {
            self.devices.clone()
        }
    }

    pub(crate) fn device(ordinal: usize, total_gb: u64) -> DeviceInfo {
        DeviceInfo {
            ordinal,
            name: format!("GPU {ordinal}"),
            compute_capability: (8, 6),
            total_memory: total_gb << 30,
            free_memory: total_gb << 29,
            driver_cuda: (13, 5),
        }
    }

    fn machine(devices: Result<Vec<DeviceInfo>, String>) -> FakeProbe {
        FakeProbe {
            force_cpu: false,
            cuda_built: true,
            devices,
        }
    }

    fn reason(selection: Selection) -> String {
        match selection {
            Selection::Cpu { reason } => reason,
            Selection::Cuda(d) => panic!("chose CUDA on {}", d.name),
        }
    }

    #[test]
    fn a_device_with_the_driver_and_the_feature_gets_cuda() {
        assert_eq!(
            select(&machine(Ok(vec![device(0, 8)]))),
            Selection::Cuda(device(0, 8))
        );
    }

    #[test]
    fn the_cpu_is_chosen_and_says_why() {
        let forced = FakeProbe {
            force_cpu: true,
            ..machine(Ok(vec![device(0, 8)]))
        };
        assert_eq!(reason(select(&forced)), "TETERA_FORCE_CPU is set");
        let unbuilt = FakeProbe {
            cuda_built: false,
            ..machine(Ok(vec![device(0, 8)]))
        };
        assert_eq!(reason(select(&unbuilt)), "this build has no `cuda` feature");
        let no_driver = machine(Err("libcuda.so.1 isn't there".into()));
        assert_eq!(
            reason(select(&no_driver)),
            "no CUDA driver: libcuda.so.1 isn't there"
        );
        assert_eq!(
            reason(select(&machine(Ok(vec![])))),
            "the driver finds no CUDA device"
        );
    }

    #[test]
    fn among_several_devices_the_largest_wins_then_the_first() {
        let three = machine(Ok(vec![device(0, 8), device(1, 24), device(2, 24)]));
        assert_eq!(select(&three), Selection::Cuda(device(1, 24)));
        let equal = machine(Ok(vec![device(3, 8), device(1, 8)]));
        assert_eq!(select(&equal), Selection::Cuda(device(1, 8)));
    }

    #[test]
    fn switches_turn_on_with_the_usual_words() {
        for on in ["1", "true", "TRUE", "yes", "On", " 1 "] {
            assert!(env_flag(Some(on)), "{on:?}");
        }
        for off in ["0", "false", "no", "", "2", "enabled"] {
            assert!(!env_flag(Some(off)), "{off:?}");
        }
        assert!(!env_flag(None));
    }

    #[test]
    fn backends_print_their_names() {
        assert_eq!(format!("[{:<5}]", BackendKind::Cuda), "[CUDA ]");
        assert_eq!(BackendKind::Cpu.to_string(), "CPU");
    }

    #[test]
    fn the_system_probe_reports_this_build() {
        assert_eq!(SystemProbe.cuda_built(), cfg!(feature = "cuda"));
        #[cfg(not(feature = "cuda"))]
        assert!(SystemProbe.devices().is_err());
    }
}
