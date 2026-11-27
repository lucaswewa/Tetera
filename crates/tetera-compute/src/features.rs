//! What CUDA offers on this machine (ADR-0009): each feature Tetera plans to use, tried for
//! real. Under WSL2 some features of native Linux are missing or limited; this says which.

use crate::Compute;

/// How a feature fared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// It works, with what was seen.
    Works(String),
    /// It was tried and failed, with why.
    Fails(String),
    /// Not tried: the CPU is in use, or the build has no `cuda` feature.
    NotTried(String),
}

/// One feature, and how it fared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// The feature.
    pub name: &'static str,
    /// What was tried.
    pub what: &'static str,
    /// How it fared.
    pub outcome: Outcome,
}

/// The features, and what is tried for each.
pub const FEATURES: [(&str, &str); 5] = [
    (
        "pinned memory",
        "640 MiB of page-locked host memory, a pool of 16 full frames",
    ),
    ("cuFFT", "a 1,024-point complex plan"),
    ("NVRTC", "a kernel compiled at run time"),
    ("nvJPEG", "a library handle"),
    (
        "IPC memory handles",
        "a device buffer's handle, for another process",
    ),
];

/// Tries every feature on `compute`'s CUDA device.
pub fn check_all(compute: &Compute) -> Vec<Check> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let outcomes = [
            tried(|| tries::pinned(gpu)),
            tried(|| tries::cufft(gpu)),
            tried(tries::nvrtc),
            tried(tries::nvjpeg),
            tried(|| tries::ipc(gpu)),
        ];
        return FEATURES
            .iter()
            .zip(outcomes)
            .map(|(&(name, what), outcome)| Check {
                name,
                what,
                outcome,
            })
            .collect();
    }
    let reason = format!("not tried: {}", compute.why());
    FEATURES
        .iter()
        .map(|&(name, what)| Check {
            name,
            what,
            outcome: Outcome::NotTried(reason.clone()),
        })
        .collect()
}

/// Runs one try, turning a panic into a failure: a probe must report, never crash.
#[cfg(feature = "cuda")]
fn tried(attempt: impl FnOnce() -> Result<String, String>) -> Outcome {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(attempt)) {
        Ok(Ok(seen)) => Outcome::Works(seen),
        Ok(Err(why)) => Outcome::Fails(why),
        Err(panic) => Outcome::Fails(format!(
            "panicked: {}",
            panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or(panic.downcast_ref::<&str>().copied())
                .unwrap_or("?")
        )),
    }
}

#[cfg(feature = "cuda")]
mod tries {
    use crate::cuda::{CudaBackend, library_loads};
    use cudarc::driver::sys;
    use std::ffi::{c_int, c_void};
    use tetera_core::frame::Allocator;

    pub(super) fn pinned(gpu: &CudaBackend) -> Result<String, String> {
        let buffer = gpu
            .pinned_allocator()
            .allocate(640 << 20)
            .map_err(|e| e.to_string())?;
        Ok(format!("{} MiB allocated", buffer.bytes().len() >> 20))
    }

    pub(super) fn cufft(gpu: &CudaBackend) -> Result<String, String> {
        let library = library_loads(&["libcufft.so.12", "libcufft.so"])?;
        cudarc::cufft::CudaFft::plan_1d(
            1_024,
            cudarc::cufft::sys::cufftType::CUFFT_C2C,
            1,
            gpu.stream().clone(),
        )
        .map(|_| format!("a plan made, with {library}"))
        .map_err(|e| format!("{e:?}"))
    }

    pub(super) fn nvrtc() -> Result<String, String> {
        let library = library_loads(&["libnvrtc.so.13", "libnvrtc.so"])?;
        let source = r#"extern "C" __global__ void probe(float* x) { x[threadIdx.x] = 1.0f; }"#;
        cudarc::nvrtc::compile_ptx(source)
            .map(|ptx| format!("{} bytes of PTX, with {library}", ptx.to_src().len()))
            .map_err(|e| format!("{e:?}"))
    }

    pub(super) fn nvjpeg() -> Result<String, String> {
        let name = library_loads(&["libnvjpeg.so.13", "libnvjpeg.so"])?;
        type Create = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
        type Destroy = unsafe extern "C" fn(*mut c_void) -> c_int;
        // SAFETY: as in `library_loads`: NVIDIA's library, made to be loaded.
        let library = unsafe { libloading::Library::new(&name) }.map_err(|e| e.to_string())?;
        // SAFETY: nvJPEG's header declares these two functions with exactly these signatures:
        // `nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t*)` and
        // `nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t)`, a handle being a pointer.
        let (create, destroy) = unsafe {
            (
                library
                    .get::<Create>(b"nvjpegCreateSimple\0")
                    .map_err(|e| e.to_string())?,
                library
                    .get::<Destroy>(b"nvjpegDestroy\0")
                    .map_err(|e| e.to_string())?,
            )
        };
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: `create` writes a handle through the pointer it is given, which points at
        // `handle`; `destroy` then takes that handle, once, while the library is loaded.
        let status = unsafe { create(&mut handle) };
        if status != 0 {
            return Err(format!("nvjpegCreateSimple returned {status}"));
        }
        // SAFETY: as above.
        let status = unsafe { destroy(handle) };
        if status != 0 {
            return Err(format!("nvjpegDestroy returned {status}"));
        }
        Ok(format!("a handle made and released, with {name}"))
    }

    /// IPC handles exist only for memory from `cuMemAlloc`: `cudarc` allocates from a
    /// stream-ordered pool, whose memory never has one, so this allocates the classic way.
    pub(super) fn ipc(gpu: &CudaBackend) -> Result<String, String> {
        gpu.context()
            .bind_to_thread()
            .map_err(|e| format!("{e:?}"))?;
        let mut pointer: sys::CUdeviceptr = 0;
        // SAFETY: `cuMemAlloc_v2` writes one device pointer through the pointer it is given,
        // which points at `pointer`; this context is current on this thread.
        let status = unsafe { sys::cuMemAlloc_v2(&mut pointer, 4_096) };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemAlloc: {status:?}"));
        }
        let mut handle = std::mem::MaybeUninit::<sys::CUipcMemHandle>::zeroed();
        // SAFETY: `cuIpcGetMemHandle` writes one handle through the pointer it is given, which
        // points at `handle`, alive for the call; `pointer` is the live allocation just made.
        let status = unsafe { sys::cuIpcGetMemHandle(handle.as_mut_ptr(), pointer) };
        // SAFETY: `pointer` came from `cuMemAlloc_v2` above and is freed once, here.
        let freed = unsafe { sys::cuMemFree_v2(pointer) };
        match (status, freed) {
            (sys::CUresult::CUDA_SUCCESS, sys::CUresult::CUDA_SUCCESS) => {
                Ok("a handle made for memory from cuMemAlloc".to_string())
            }
            (sys::CUresult::CUDA_SUCCESS, freed) => Err(format!("cuMemFree: {freed:?}")),
            (status, _) => Err(format!(
                "cuIpcGetMemHandle on memory from cuMemAlloc: {status:?}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::{FakeProbe, fake_device};

    #[test]
    fn on_the_cpu_nothing_is_tried_and_each_feature_says_why() {
        let probe = FakeProbe {
            force_cpu: true,
            cuda_built: true,
            devices: Ok(vec![fake_device(0, 8)]),
        };
        let checks = check_all(&Compute::from_probe(&probe).unwrap());
        assert_eq!(checks.len(), FEATURES.len());
        for check in &checks {
            assert_eq!(
                check.outcome,
                Outcome::NotTried("not tried: TETERA_FORCE_CPU is set".into()),
                "{}",
                check.name
            );
        }
        let names: Vec<&str> = checks.iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            [
                "pinned memory",
                "cuFFT",
                "NVRTC",
                "nvJPEG",
                "IPC memory handles"
            ]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_panic_in_a_try_is_reported_as_a_failure() {
        assert_eq!(tried(|| Ok("fine".into())), Outcome::Works("fine".into()));
        assert_eq!(tried(|| Err("no".into())), Outcome::Fails("no".into()));
        assert_eq!(
            tried(|| panic!("boom")),
            Outcome::Fails("panicked: boom".into())
        );
        let dynamic = tried(|| panic!("{}", String::from("formatted")));
        assert_eq!(dynamic, Outcome::Fails("panicked: formatted".into()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn on_this_gpu_the_features_are_tried() {
        if !crate::select::cuda_tests_enabled() {
            eprintln!("skipped: set TETERA_CUDA=1 to run CUDA tests");
            return;
        }
        let compute = Compute::detect().unwrap();
        let checks = check_all(&compute);
        for check in &checks {
            eprintln!("{:<20} {:?}", check.name, check.outcome);
        }
        let seen = |name: &str| match &checks.iter().find(|c| c.name == name).unwrap().outcome {
            Outcome::Works(seen) => seen.clone(),
            other => panic!("{name}: {other:?}"),
        };
        assert_eq!(seen("pinned memory"), "640 MiB allocated");
        assert!(seen("cuFFT").starts_with("a plan made, with libcufft"));
        assert!(seen("NVRTC").ends_with("bytes of PTX, with libnvrtc.so.13"));
        let ptx_bytes: usize = seen("NVRTC").split(' ').next().unwrap().parse().unwrap();
        assert!(ptx_bytes > 100, "{ptx_bytes} bytes of PTX");
        assert!(seen("nvJPEG").starts_with("a handle made and released, with libnvjpeg"));
        assert_eq!(
            seen("IPC memory handles"),
            "a handle made for memory from cuMemAlloc"
        );
    }
}
