//! Operations with a twin on each backend. The first is scale and offset, 16-bit samples to
//! floats, which proves the path every later kernel takes (ADR-0010): the CUDA kernel fuses
//! the multiply and the add, and so does its CPU twin, so the two agree bit for bit.

use crate::{Compute, CpuBackend};
use rayon::prelude::*;
use tetera_core::{Error, Result};

/// Samples a CPU worker takes at a time.
const CHUNK: usize = 65_536;

fn same_length(input: usize, output: usize) -> Result<()> {
    if input == output {
        Ok(())
    } else {
        Err(Error::invalid_argument(format!(
            "{input} samples can't fill {output} outputs"
        )))
    }
}

/// The CPU twin: `out[i] = input[i] × scale + offset`, fused.
pub fn scale_offset_cpu(
    cpu: &CpuBackend,
    input: &[u16],
    scale: f32,
    offset: f32,
    out: &mut [f32],
) -> Result<()> {
    same_length(input.len(), out.len())?;
    cpu.install(|| {
        out.par_chunks_mut(CHUNK)
            .zip(input.par_chunks(CHUNK))
            .for_each(|(out, input)| {
                for (o, &i) in out.iter_mut().zip(input) {
                    *o = f32::from(i).mul_add(scale, offset);
                }
            });
    });
    Ok(())
}

/// Scale and offset on whichever backend `compute` chose, from and to host memory.
pub fn scale_offset(compute: &Compute, input: &[u16], scale: f32, offset: f32) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let on_device = gpu.upload(input)?;
        let mut out = gpu.zeros::<f32>(input.len())?;
        gpu.scale_offset(&on_device, scale, offset, &mut out)?;
        return gpu.download(&out);
    }
    let mut out = vec![0.0; input.len()];
    scale_offset_cpu(compute.cpu(), input, scale, offset, &mut out)?;
    Ok(out)
}

/// Threads in a block for one-dimensional kernels.
#[cfg(feature = "cuda")]
pub const BLOCK: u32 = 256;

#[cfg(feature = "cuda")]
impl crate::CudaBackend {
    /// The CUDA kernel: `out[i] = input[i] × scale + offset`, fused, queued on the default
    /// stream. It returns without waiting; a download, or [`synchronize`](Self::synchronize),
    /// waits.
    pub fn scale_offset(
        &self,
        input: &crate::DeviceBuffer<u16>,
        scale: f32,
        offset: f32,
        out: &mut crate::DeviceBuffer<f32>,
    ) -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        same_length(input.len(), out.len())?;
        if input.is_empty() {
            return Ok(());
        }
        let n = u32::try_from(input.len()).map_err(|_| {
            Error::invalid_argument(format!(
                "{} samples are more than one launch takes",
                input.len()
            ))
        })?;
        let kernel = self
            .module(&crate::kernels::SCALE)?
            .function("scale_offset_u16_f32")?;
        let config = LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream().launch_builder(&kernel);
        launch
            .arg(out.slice_mut())
            .arg(input.slice())
            .arg(&scale)
            .arg(&offset)
            .arg(&n);
        // SAFETY: the kernel's parameters are (float*, const unsigned short*, float, float,
        // unsigned int), and the arguments match them in order and type. It writes out[i] and
        // reads in[i] only for i < n, and both buffers hold n elements, as checked above.
        unsafe { launch.launch(config) }
            .map_err(|e| Error::hardware(format!("the scale kernel wouldn't launch: {e:?}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use crate::twin::{Tolerance, compare};

    fn samples(n: usize) -> Vec<u16> {
        (0..n)
            .map(|i| ((i * 2_654_435_761) % 65_536) as u16)
            .collect()
    }

    #[test]
    fn the_cpu_scales_and_offsets_every_sample() {
        let cpu = CpuBackend::new(4).unwrap();
        let input = [0u16, 1, 4_095, 65_535];
        let mut out = [f32::NAN; 4];
        scale_offset_cpu(&cpu, &input, 0.5, -2.0, &mut out).unwrap();
        assert_eq!(out, [-2.0, -1.5, 2_045.5, 32_765.5]);
        let mut short = [0.0f32; 3];
        let err = scale_offset_cpu(&cpu, &input, 1.0, 0.0, &mut short).unwrap_err();
        assert_eq!(err.to_string(), "4 samples can't fill 3 outputs");
    }

    #[test]
    fn the_cpu_fuses_the_multiply_and_the_add() {
        let cpu = CpuBackend::new(1).unwrap();
        // 3 × the float nearest 1/3 is just above 1: fused, that excess survives the −1;
        // unfused, the product rounds to 1 first and the excess is lost.
        let (scale, offset) = (1.0 / 3.0f32, -1.0f32);
        let mut out = [0.0f32];
        scale_offset_cpu(&cpu, &[3], scale, offset, &mut out).unwrap();
        assert_eq!(out[0].to_bits(), 3.0f32.mul_add(scale, offset).to_bits());
        assert!(out[0] > 0.0 && out[0] < 1e-7, "{}", out[0]);
        assert_eq!(3.0f32 * scale + offset, 0.0, "unfused rounds twice");
    }

    #[test]
    fn work_larger_than_a_chunk_is_shared_out_whole() {
        let cpu = CpuBackend::new(3).unwrap();
        let input = samples(3 * CHUNK + 17);
        let mut out = vec![0.0f32; input.len()];
        scale_offset_cpu(&cpu, &input, 2.0, 1.0, &mut out).unwrap();
        assert!(
            out.iter()
                .zip(&input)
                .all(|(&o, &i)| o == f32::from(i) * 2.0 + 1.0)
        );
    }

    #[test]
    fn the_chosen_backend_does_the_work() {
        let compute = Compute::cpu_only().unwrap();
        assert_eq!(
            scale_offset(&compute, &[10, 20], 0.25, 1.0).unwrap(),
            [3.5, 6.0]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernel_and_its_twin_agree_bit_for_bit() {
        let gpu = crate::cuda_or_skip!();
        let cpu = CpuBackend::new(0).unwrap();
        for n in [1, 255, 256, 257, 1_000_003, 5_472 * 3_648] {
            let input = samples(n);
            let (scale, offset) = (1.0 / 4_095.0, -0.031_25);
            let mut on_cpu = vec![0.0f32; n];
            scale_offset_cpu(&cpu, &input, scale, offset, &mut on_cpu).unwrap();
            let mut out = gpu.zeros::<f32>(n).unwrap();
            gpu.scale_offset(&gpu.upload(&input).unwrap(), scale, offset, &mut out)
                .unwrap();
            let on_gpu = gpu.download(&out).unwrap();
            let comparison = compare(&on_cpu, &on_gpu, Tolerance::EXACT).unwrap();
            assert!(comparison.passes(), "{n} samples: {comparison}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_kernel_refuses_mismatched_buffers_and_skips_empty_ones() {
        let gpu = crate::cuda_or_skip!();
        let input = gpu.upload(&[1u16, 2, 3]).unwrap();
        let mut short = gpu.zeros::<f32>(2).unwrap();
        assert!(gpu.scale_offset(&input, 1.0, 0.0, &mut short).is_err());
        let empty = gpu.upload::<u16>(&[]).unwrap();
        let mut none = gpu.upload::<f32>(&[]).unwrap();
        gpu.scale_offset(&empty, 1.0, 0.0, &mut none).unwrap();
        let compute = Compute::from_probe(&crate::SystemProbe).unwrap();
        assert_eq!(scale_offset(&compute, &[8], 0.5, 0.0).unwrap(), [4.0]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn synchronize_waits_for_every_queued_kernel() {
        use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let gpu = crate::cuda_or_skip!();
        let n = 5_472 * 3_648;
        let input = gpu.upload(&samples(n)).unwrap();
        let mut out = gpu.zeros::<f32>(n).unwrap();
        gpu.synchronize().unwrap();
        let start = gpu.stream().record_event(Some(CU_EVENT_DEFAULT)).unwrap();
        // Some milliseconds of work, queued in microseconds: launches don't wait.
        for _ in 0..50 {
            gpu.scale_offset(&input, 1.0, 0.0, &mut out).unwrap();
        }
        let end = gpu.stream().record_event(Some(CU_EVENT_DEFAULT)).unwrap();
        gpu.synchronize().unwrap();
        assert!(
            end.is_complete(),
            "synchronize returned before the queued kernels were done"
        );
        assert!(start.elapsed_ms(&end).unwrap() > 0.0);
    }
}
