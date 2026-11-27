//! Loading kernels (ADR-0010): each `.cu` file in `kernels/` is one module, embedded at build
//! time as a fat binary when `nvcc` compiled it, and always as CUDA C source, which NVRTC
//! compiles at run time when there is no fat binary. A backend loads each module once, on
//! first use.

use crate::cuda::{CudaBackend, library_loads};
use cudarc::driver::{CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;
use std::fmt;
use std::sync::Arc;
use tetera_core::{Error, Result};

mod generated {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}

/// One kernel file, as the build embedded it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSource {
    /// The file's name, without `.cu`.
    pub name: &'static str,
    /// Its CUDA C source.
    pub source: &'static str,
    /// Its fat binary, when `nvcc` compiled it at build time: machine code for each major
    /// architecture, and PTX for newer ones.
    pub fatbin: Option<&'static [u8]>,
}

/// `kernels/scale.cu`: scale and offset 16-bit samples to floats.
pub const SCALE: KernelSource = KernelSource {
    name: "scale",
    source: generated::SCALE_SOURCE,
    fatbin: generated::SCALE_FATBIN,
};

/// Where a loaded module's code came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// `nvcc`'s fat binary, made when the crate was built.
    FatBinary,
    /// NVRTC's PTX, compiled when the module was loaded.
    Nvrtc,
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FatBinary => "nvcc's fat binary",
            Self::Nvrtc => "NVRTC's PTX",
        })
    }
}

/// A loaded module.
pub struct Module {
    module: Arc<CudaModule>,
    origin: Origin,
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Module")
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

impl Module {
    /// Loads `kernel` on `gpu`'s context: its fat binary, or its source through NVRTC.
    pub fn load(gpu: &CudaBackend, kernel: &KernelSource) -> Result<Self> {
        let (code, origin) = match kernel.fatbin {
            Some(bytes) => (Ptx::from_binary(bytes.to_vec()), Origin::FatBinary),
            None => {
                library_loads(&["libnvrtc.so.13", "libnvrtc.so"]).map_err(|why| {
                    Error::hardware(format!(
                        "the {} kernels have no fat binary, and NVRTC {why}",
                        kernel.name
                    ))
                })?;
                let ptx = cudarc::nvrtc::compile_ptx(kernel.source).map_err(|e| {
                    Error::internal(format!(
                        "NVRTC couldn't compile the {} kernels: {e:?}",
                        kernel.name
                    ))
                })?;
                (ptx, Origin::Nvrtc)
            }
        };
        let module = gpu.context().load_module(code).map_err(|e| {
            Error::hardware(format!(
                "CUDA wouldn't load the {} kernels: {e:?}",
                kernel.name
            ))
        })?;
        Ok(Self { module, origin })
    }

    /// Where its code came from.
    pub fn origin(&self) -> Origin {
        self.origin
    }

    /// The kernel called `name` in it.
    pub fn function(&self, name: &str) -> Result<CudaFunction> {
        self.module
            .load_function(name)
            .map_err(|e| Error::internal(format!("no kernel `{name}` in the module: {e:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first four bytes of a fat binary, little-endian.
    const FATBIN_MAGIC: u32 = 0xBA55_ED50;

    #[test]
    fn the_kernels_are_embedded_with_their_source() {
        assert_eq!(SCALE.name, "scale");
        assert!(
            SCALE
                .source
                .contains("__global__ void scale_offset_u16_f32")
        );
        if let Some(fatbin) = SCALE.fatbin {
            assert_eq!(fatbin[..4], FATBIN_MAGIC.to_le_bytes(), "a fat binary");
        }
        assert_eq!(Origin::FatBinary.to_string(), "nvcc's fat binary");
        assert_eq!(Origin::Nvrtc.to_string(), "NVRTC's PTX");
    }

    #[test]
    fn a_module_loads_and_names_its_kernels() {
        let gpu = crate::cuda_or_skip!();
        let module = Module::load(&gpu, &SCALE).unwrap();
        let expected = if SCALE.fatbin.is_some() {
            Origin::FatBinary
        } else {
            Origin::Nvrtc
        };
        assert_eq!(module.origin(), expected);
        assert!(module.function("scale_offset_u16_f32").is_ok());
        let err = module.function("no_such_kernel").unwrap_err();
        assert!(
            err.to_string().contains("no kernel `no_such_kernel`"),
            "{err}"
        );
        assert!(format!("{module:?}").contains("origin"));
    }

    #[test]
    fn without_a_fat_binary_the_source_is_compiled_with_nvrtc() {
        let gpu = crate::cuda_or_skip!();
        let from_source = KernelSource {
            fatbin: None,
            ..SCALE
        };
        let module = Module::load(&gpu, &from_source).unwrap();
        assert_eq!(module.origin(), Origin::Nvrtc);
        assert!(module.function("scale_offset_u16_f32").is_ok());
    }

    #[test]
    fn a_kernel_that_doesnt_compile_or_load_is_an_error() {
        let gpu = crate::cuda_or_skip!();
        let broken = KernelSource {
            name: "broken",
            source: "this isn't CUDA",
            fatbin: None,
        };
        let err = Module::load(&gpu, &broken).unwrap_err();
        assert!(
            err.to_string()
                .contains("couldn't compile the broken kernels"),
            "{err}"
        );
        let garbled = KernelSource {
            name: "garbled",
            source: "",
            // Not a module; NUL-terminated, since the driver reads anything else as PTX text.
            fatbin: Some(b"not a module\0"),
        };
        let err = Module::load(&gpu, &garbled).unwrap_err();
        assert_eq!(err.kind(), tetera_core::ErrorKind::Hardware);
        assert!(
            err.to_string()
                .contains("CUDA wouldn't load the garbled kernels"),
            "{err}"
        );
    }
}
