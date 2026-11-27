//! The CUDA backend (ADR-0009), through `cudarc` with its libraries loaded at run time: the
//! build needs no toolkit, and a machine without a driver falls back to the CPU.
//!
//! `cudarc` panics when it can't load a library, so every path here first checks that the
//! library loads, with [`library_loads`], and reports an error instead.

use crate::kernels::{KernelSource, Module};
use crate::select::DeviceInfo;
use cudarc::driver::sys::{self, CUdevice_attribute};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DeviceRepr, DriverError, PinnedHostSlice, ValidAsZeroBits,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use tetera_core::frame::{Allocator, BufferStorage};
use tetera_core::{Error, Result};

/// The names the CUDA driver's library goes by.
pub const DRIVER_LIBRARIES: &[&str] = &["libcuda.so.1", "libcuda.so"];

/// The first of `names` that loads, or why none did.
pub fn library_loads(names: &[&str]) -> std::result::Result<String, String> {
    let mut last = String::new();
    for &name in names {
        // SAFETY: loading a shared library runs its initialisers. These are NVIDIA's own
        // libraries, made to be loaded this way, and `cudarc` loads them the same way.
        match unsafe { libloading::Library::new(name) } {
            Ok(_) => return Ok(name.to_string()),
            Err(e) => last = e.to_string(),
        }
    }
    Err(format!("none of {} loads ({last})", names.join(", ")))
}

fn hardware(what: &str, e: DriverError) -> Error {
    Error::hardware(format!("CUDA {what}: {e:?}"))
}

/// The newest CUDA version the driver runs, major and minor.
pub fn driver_cuda_version() -> std::result::Result<(u32, u32), String> {
    library_loads(DRIVER_LIBRARIES)?;
    let mut version: std::ffi::c_int = 0;
    // SAFETY: `cuDriverGetVersion` writes one int through the pointer it is given, which
    // points at `version`, alive for the whole call.
    let status = unsafe { sys::cuDriverGetVersion(&mut version) };
    if status != sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuDriverGetVersion: {status:?}"));
    }
    Ok(cuda_version(version))
}

/// A CUDA version as the driver API gives it, `1000 × major + 10 × minor`, as major and minor.
pub fn cuda_version(encoded: std::ffi::c_int) -> (u32, u32) {
    let v = u32::try_from(encoded).unwrap_or(0);
    (v / 1_000, v % 1_000 / 10)
}

/// Every CUDA device, or why there is none to ask.
pub fn devices() -> std::result::Result<Vec<DeviceInfo>, String> {
    let driver = driver_cuda_version()?;
    let count = CudaContext::device_count().map_err(|e| format!("counting devices: {e:?}"))?;
    (0..usize::try_from(count).unwrap_or(0))
        .map(|ordinal| {
            let ctx = CudaContext::new(ordinal).map_err(|e| format!("device {ordinal}: {e:?}"))?;
            device_info(&ctx, driver).map_err(|e| format!("device {ordinal}: {e:?}"))
        })
        .collect()
}

fn device_info(
    ctx: &Arc<CudaContext>,
    driver_cuda: (u32, u32),
) -> std::result::Result<DeviceInfo, DriverError> {
    let attribute = |a| ctx.attribute(a).map(|v| u32::try_from(v).unwrap_or(0));
    let (free, total) = ctx.mem_get_info()?;
    Ok(DeviceInfo {
        ordinal: ctx.ordinal(),
        name: ctx.name()?,
        compute_capability: (
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        ),
        total_memory: total as u64,
        free_memory: free as u64,
        driver_cuda,
    })
}

/// A CUDA device, with its context, its default stream and the kernel modules loaded on it.
/// Clones share all three.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    info: DeviceInfo,
    modules: Arc<Mutex<HashMap<&'static str, Arc<Module>>>>,
}

impl CudaBackend {
    /// The device numbered `ordinal`.
    pub fn new(ordinal: usize) -> Result<Self> {
        let driver = driver_cuda_version().map_err(Error::hardware)?;
        let ctx =
            CudaContext::new(ordinal).map_err(|e| hardware(&format!("device {ordinal}"), e))?;
        let info = device_info(&ctx, driver).map_err(|e| hardware("device info", e))?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            info,
            modules: Arc::default(),
        })
    }

    /// The device, as it was when the backend started.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Its memory now, free and in all, in bytes.
    pub fn memory(&self) -> Result<(u64, u64)> {
        let (free, total) = self.ctx.mem_get_info().map_err(|e| hardware("memory", e))?;
        Ok((free as u64, total as u64))
    }

    /// Its context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The module of `kernel`, loaded on first use and kept after.
    pub fn module(&self, kernel: &KernelSource) -> Result<Arc<Module>> {
        let mut modules = self.modules.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(module) = modules.get(kernel.name) {
            return Ok(module.clone());
        }
        let module = Arc::new(Module::load(self, kernel)?);
        modules.insert(kernel.name, module.clone());
        Ok(module)
    }

    /// Its default stream: work on it runs in the order it was given.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// A new stream, whose work may overlap the default stream's.
    pub fn new_stream(&self) -> Result<Arc<CudaStream>> {
        self.ctx.new_stream().map_err(|e| hardware("stream", e))
    }

    /// `host` copied to the device. It returns once the copy has read `host`, so the caller
    /// may change or reuse it: from pinned memory, the copy runs on after the call returns,
    /// and is waited for.
    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> Result<DeviceBuffer<T>> {
        let slice = self
            .stream
            .clone_htod(host)
            .map_err(|e| hardware("upload", e))?;
        self.synchronize()?;
        Ok(DeviceBuffer { slice })
    }

    /// `len` zeroed elements on the device.
    pub fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        let slice = self
            .stream
            .alloc_zeros(len)
            .map_err(|e| hardware("allocation", e))?;
        Ok(DeviceBuffer { slice })
    }

    /// The buffer copied back to the host.
    pub fn download<T: DeviceRepr>(&self, buffer: &DeviceBuffer<T>) -> Result<Vec<T>> {
        self.stream
            .clone_dtoh(&buffer.slice)
            .map_err(|e| hardware("download", e))
    }

    /// The buffer copied into `host`, which must be its length. It returns once the copy has
    /// written `host`.
    pub fn download_into<T: DeviceRepr>(
        &self,
        buffer: &DeviceBuffer<T>,
        host: &mut [T],
    ) -> Result<()> {
        if host.len() != buffer.len() {
            return Err(Error::invalid_argument(format!(
                "a device buffer of {} can't be copied into {} elements",
                buffer.len(),
                host.len()
            )));
        }
        self.stream
            .memcpy_dtoh(&buffer.slice, host)
            .map_err(|e| hardware("download", e))?;
        self.synchronize()
    }

    /// `buffer`'s 16-bit samples copied into `host`'s bytes, little-endian, as a frame holds
    /// them; `host` must be twice its length and start on an even address. It returns once
    /// the copy has written `host`.
    #[cfg(target_endian = "little")]
    pub fn download_into_bytes(&self, buffer: &DeviceBuffer<u16>, host: &mut [u8]) -> Result<()> {
        // SAFETY: any two bytes are a valid u16, and `align_to_mut` gives an aligned middle;
        // the length and the empty ends are checked before it is used.
        let (head, samples, tail) = unsafe { host.align_to_mut::<u16>() };
        if !head.is_empty() || !tail.is_empty() || samples.len() != buffer.len() {
            return Err(Error::invalid_argument(format!(
                "a device buffer of {} samples can't be copied into {} bytes{}",
                buffer.len(),
                head.len() + 2 * samples.len() + tail.len(),
                if head.is_empty() {
                    ""
                } else {
                    " at an odd address"
                }
            )));
        }
        self.download_into(buffer, samples)
    }

    /// Waits until the default stream's work is done.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| hardware("synchronize", e))
    }

    /// How long the device takes over what `work` queues on the default stream, in
    /// milliseconds, timed by events either side of it. It waits for the work to finish.
    pub fn time_ms(&self, work: impl FnOnce() -> Result<()>) -> Result<f32> {
        let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = self
            .stream
            .record_event(flags)
            .map_err(|e| hardware("event", e))?;
        work()?;
        let end = self
            .stream
            .record_event(flags)
            .map_err(|e| hardware("event", e))?;
        start.elapsed_ms(&end).map_err(|e| hardware("timing", e))
    }

    /// An allocator of pinned host memory on this device's context, for frame pools: copies
    /// between it and the device run at the bus's full speed.
    pub fn pinned_allocator(&self) -> PinnedAllocator {
        PinnedAllocator {
            ctx: self.ctx.clone(),
        }
    }
}

/// Elements on the device.
#[derive(Debug)]
pub struct DeviceBuffer<T> {
    slice: CudaSlice<T>,
}

impl<T> DeviceBuffer<T> {
    /// How many elements it holds.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.slice.len() == 0
    }

    /// The `cudarc` slice, for kernels.
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// The `cudarc` slice, to write, for kernels.
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

/// Page-locked host memory, for frame pools (ADR-0008). Its buffers are cached for the CPU,
/// not write-combined, since the CPU reads frames too.
#[derive(Debug, Clone)]
pub struct PinnedAllocator {
    ctx: Arc<CudaContext>,
}

struct PinnedBuffer {
    /// Owns the memory; dropping it frees the memory.
    _slice: PinnedHostSlice<u8>,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the memory is page-locked host memory owned by `_slice`, which `cudarc` marks Send
// and Sync. `ptr` only points into it, and is reached through `&self` or `&mut self` alone,
// so Rust's borrows keep reads and writes apart as for any owned buffer.
unsafe impl Send for PinnedBuffer {}
// SAFETY: as for Send; a shared `PinnedBuffer` only ever hands out `&[u8]`.
unsafe impl Sync for PinnedBuffer {}

impl BufferStorage for PinnedBuffer {
    fn bytes(&self) -> &[u8] {
        // SAFETY: `ptr` and `len` describe memory that `_slice` keeps allocated while `self`
        // lives, zeroed when it was made, so every byte is initialised; `&self` rules out a
        // `&mut` to it for the slice's lifetime. Frames are uploaded to the GPU only once
        // frozen, so no copy writes to it meanwhile (ADR-0009).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`, and `&mut self` makes this the only reference.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Allocator for PinnedAllocator {
    fn allocate(&self, len: usize) -> Result<Box<dyn BufferStorage>> {
        // SAFETY: the memory comes uninitialised; it is zeroed below, before anything reads it.
        let mut slice = unsafe {
            self.ctx
                .alloc_pinned_with_flags::<u8>(len.max(1), sys::CU_MEMHOSTALLOC_PORTABLE)
        }
        .map_err(|e| hardware(&format!("{len} bytes of pinned memory"), e))?;
        let ptr = slice
            .as_mut_ptr()
            .map_err(|e| hardware("pinned memory", e))?;
        // SAFETY: `ptr` points at `len.max(1)` bytes that `slice` owns and nothing else uses yet.
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
        Ok(Box::new(PinnedBuffer {
            _slice: slice,
            ptr,
            len,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::cuda_tests_enabled;
    use std::time::Duration;
    use tetera_core::frame::{BayerPattern, FramePool, PixelView};
    use tetera_core::{FrameLayout, PixelFormat};

    /// The backend, or `None` and a note when `TETERA_CUDA=1` isn't set.
    fn gpu() -> Option<CudaBackend> {
        if !cuda_tests_enabled() {
            eprintln!("skipped: set TETERA_CUDA=1 to run CUDA tests");
            return None;
        }
        Some(CudaBackend::new(0).expect("TETERA_CUDA=1 promises a working device 0"))
    }

    #[test]
    fn samples_download_into_a_frames_bytes() {
        let Some(gpu) = gpu() else { return };
        let buffer = gpu.upload(&[0x0102u16, 0xa0b0, 7]).unwrap();
        let mut bytes = [0u8; 10];
        // Where in `bytes` an even address starts, and an odd one.
        let even = usize::from(!(bytes.as_ptr() as usize).is_multiple_of(2));
        let odd = 1 - even;
        gpu.download_into_bytes(&buffer, &mut bytes[even..even + 6])
            .unwrap();
        assert_eq!(
            &bytes[even..even + 6],
            &[0x02, 0x01, 0xb0, 0xa0, 7, 0],
            "little-endian"
        );
        for (from, len, why) in [
            (even, 4, "too short"),
            (even, 8, "too long"),
            (odd, 6, "odd"),
        ] {
            assert!(
                gpu.download_into_bytes(&buffer, &mut bytes[from..from + len])
                    .is_err(),
                "{why}"
            );
        }
    }

    #[test]
    fn driver_versions_decode_as_cuda_writes_them() {
        assert_eq!(cuda_version(13_050), (13, 5));
        assert_eq!(cuda_version(12_040), (12, 4));
        assert_eq!(cuda_version(9_020), (9, 2));
        assert_eq!(cuda_version(13_000), (13, 0));
        assert_eq!(cuda_version(-1), (0, 0), "nonsense reads as no version");
    }

    #[test]
    fn a_missing_library_is_an_error_not_a_panic() {
        let err = library_loads(&["libtetera-no-such-library.so"]).unwrap_err();
        assert!(err.contains("libtetera-no-such-library.so"), "{err}");
    }

    #[test]
    fn a_device_that_isnt_there_is_an_error() {
        // Without a driver this fails at the library; with one, at the ordinal.
        let err = CudaBackend::new(99).unwrap_err();
        assert_eq!(err.kind(), tetera_core::ErrorKind::Hardware, "{err}");
    }

    #[test]
    fn the_device_describes_itself() {
        let Some(gpu) = gpu() else { return };
        let info = gpu.info();
        assert!(!info.name.is_empty());
        assert!(info.compute_capability.0 >= 5, "{info:?}");
        assert!(
            info.total_memory > 1 << 30 && info.free_memory <= info.total_memory,
            "{info:?}"
        );
        assert!(
            info.driver_cuda >= (13, 3),
            "the driver runs the toolkit's CUDA: {info:?}"
        );
        let (free, total) = gpu.memory().unwrap();
        assert_eq!(total, info.total_memory);
        assert!(free <= total);
        let listed = devices().unwrap();
        assert_eq!(listed[0].name, info.name);
    }

    #[test]
    fn buffers_round_trip_through_the_device() {
        let Some(gpu) = gpu() else { return };
        let samples: Vec<u16> = (0..1_000_000u32).map(|i| (i % 4_096) as u16).collect();
        let on_device = gpu.upload(&samples).unwrap();
        assert_eq!(on_device.len(), samples.len());
        assert_eq!(gpu.download(&on_device).unwrap(), samples);
        let floats: Vec<f32> = (0..1_000).map(|i| i as f32 * 0.5).collect();
        let mut back = vec![0.0f32; 1_000];
        gpu.download_into(&gpu.upload(&floats).unwrap(), &mut back)
            .unwrap();
        assert_eq!(back, floats);
        let zeros = gpu.zeros::<f32>(4_096).unwrap();
        assert!(gpu.download(&zeros).unwrap().iter().all(|&v| v == 0.0));
        assert!(!zeros.is_empty());
        let empty = gpu.upload::<u8>(&[]).unwrap();
        assert!(empty.is_empty());
        assert!(gpu.download(&empty).unwrap().is_empty());
        let mut short = vec![0.0f32; 10];
        assert!(gpu.download_into(&zeros, &mut short).is_err());
        let other = gpu.new_stream().unwrap();
        other.synchronize().unwrap();
        gpu.synchronize().unwrap();
    }

    #[test]
    fn work_is_timed_on_the_device_and_its_errors_pass_through() {
        let Some(gpu) = gpu() else { return };
        let samples = vec![7u16; 1 << 24];
        let mut copies = 0;
        let ms = gpu
            .time_ms(|| {
                copies += 1;
                gpu.upload(&samples).map(drop)
            })
            .unwrap();
        assert_eq!(copies, 1);
        assert!(ms > 0.0 && ms < 10_000.0, "32 MiB uploaded in {ms} ms");
        let err = gpu
            .time_ms(|| Err(Error::invalid_argument("no work")))
            .unwrap_err();
        assert_eq!(err.to_string(), "no work");
    }

    #[test]
    fn a_module_is_loaded_once_and_shared_by_clones() {
        let Some(gpu) = gpu() else { return };
        let first = gpu.module(&crate::kernels::SCALE).unwrap();
        let again = gpu.clone().module(&crate::kernels::SCALE).unwrap();
        assert!(
            Arc::ptr_eq(&first, &again),
            "the clone found the loaded module"
        );
        let broken = KernelSource {
            name: "broken",
            source: "not CUDA",
            fatbin: None,
        };
        assert!(gpu.module(&broken).is_err());
        assert!(gpu.module(&broken).is_err(), "a failed load isn't kept");
    }

    #[test]
    fn pinned_buffers_are_aligned_zeroed_and_writable() {
        let Some(gpu) = gpu() else { return };
        let allocator = gpu.pinned_allocator();
        let mut buffer = allocator.allocate(1 << 20).unwrap();
        assert_eq!(buffer.bytes().len(), 1 << 20);
        assert_eq!(
            buffer.bytes().as_ptr() as usize % tetera_core::frame::pool::BUFFER_ALIGN,
            0
        );
        assert!(buffer.bytes().iter().all(|&b| b == 0));
        buffer.bytes_mut()[12_345] = 7;
        assert_eq!(buffer.bytes()[12_345], 7);
    }

    #[test]
    fn copies_with_pinned_memory_are_done_when_they_return() {
        let Some(gpu) = gpu() else { return };
        let len = 64 << 20;
        let allocator = gpu.pinned_allocator();
        let mut source = allocator.allocate(len).unwrap();
        source.bytes_mut().fill(1);
        let on_device = gpu.upload(source.bytes()).unwrap();
        // A copy reads from the start, so the last byte is read last: change it at once.
        source.bytes_mut()[len - 1] = 2;
        let mut back = allocator.allocate(len).unwrap();
        gpu.download_into(&on_device, back.bytes_mut()).unwrap();
        let last = back.bytes()[len - 1];
        assert_ne!(last, 0, "the download returned before it wrote");
        assert_eq!(last, 1, "the upload returned before it read");
    }

    #[test]
    fn a_frame_pool_lives_in_pinned_memory() {
        let Some(gpu) = gpu() else { return };
        let layout = FrameLayout::new(
            PixelFormat::bayer16(BayerPattern::Rggb, 12).unwrap(),
            5_472,
            3_648,
        )
        .unwrap();
        let pool = FramePool::new(layout, 4, &gpu.pinned_allocator()).unwrap();
        let mut frame = pool.take(Duration::ZERO).unwrap();
        assert!(frame.set_sample_u16(5_471, 3_647, 4_095));
        let frame = frame.freeze();
        assert_eq!(frame.sample_u16(5_471, 3_647), Some(4_095));
        let on_device = gpu.upload(frame.data()).unwrap();
        assert_eq!(on_device.len(), layout.bytes());
        assert_eq!(gpu.download(&on_device).unwrap(), frame.data());
    }
}
