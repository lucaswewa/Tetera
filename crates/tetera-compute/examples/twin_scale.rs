//! A full frame through scale and offset on both backends, timed, and the two results
//! compared bit for bit (ADR-0010): the path every later kernel takes.
//!
//! Run it with `cargo run --release -p tetera-compute --example twin_scale --features cuda`
//! for the GPU and its twin, or without `--features cuda` for the CPU alone. `-- --smoke`
//! does one pass instead of the best of several, and says so.

use std::time::Instant;
use tetera_compute::Compute;
use tetera_compute::ops::scale_offset_cpu;

/// The MIChrome 20's full frame.
const WIDTH: usize = 5_472;
const HEIGHT: usize = 3_648;

fn main() -> tetera_core::Result<()> {
    let smoke = std::env::args().any(|arg| arg == "--smoke");
    let passes = if smoke { 1 } else { 10 };
    let n = WIDTH * HEIGHT;
    // 12-bit samples spread over the whole range, normalised to 0..1 less a black level.
    let input: Vec<u16> = (0..n)
        .map(|i| ((i * 2_654_435_761) % 4_096) as u16)
        .collect();
    let (scale, offset) = (1.0 / 4_095.0f32, -64.0 / 4_095.0f32);

    let compute = Compute::detect()?;
    println!("Backend: {}: {}", compute.kind(), compute.why());
    println!("Frame:   {WIDTH} × {HEIGHT}, {n} samples, best of {passes}");

    let mut on_cpu = vec![0.0f32; n];
    let mut best = f64::INFINITY;
    for _ in 0..passes {
        let start = Instant::now();
        scale_offset_cpu(compute.cpu(), &input, scale, offset, &mut on_cpu)?;
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "CPU:     {best:7.2} ms on {} threads",
        compute.cpu().threads()
    );

    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        use tetera_compute::{Tolerance, VramBudgets, compare};
        let (free, _) = gpu.memory()?;
        let budgets = VramBudgets::from_free(free, 0.1);
        let budget = budgets.reserve("twin_scale", (n * (2 + 4)) as u64)?;
        let _input_grant = budget.allocate((n * 2) as u64)?;
        let _output_grant = budget.allocate((n * 4) as u64)?;
        println!(
            "VRAM:    {:.0} MiB of a {:.0} MiB budget, out of {:.0} MiB the budgets share",
            budget.used() as f64 / 1_048_576.0,
            budget.limit() as f64 / 1_048_576.0,
            budgets.capacity() as f64 / 1_048_576.0
        );

        let mut on_device = None;
        let upload = gpu.time_ms(|| {
            on_device = Some(gpu.upload(&input)?);
            Ok(())
        })?;
        let on_device = on_device.expect("the upload ran");
        let mut out = gpu.zeros::<f32>(n)?;
        gpu.scale_offset(&on_device, scale, offset, &mut out)?; // loads the module
        let mut kernel = f32::INFINITY;
        for _ in 0..passes {
            kernel =
                kernel.min(gpu.time_ms(|| gpu.scale_offset(&on_device, scale, offset, &mut out))?);
        }
        let start = Instant::now();
        let on_gpu = gpu.download(&out)?;
        let download = start.elapsed().as_secs_f64() * 1e3;
        let module = gpu.module(&tetera_compute::kernels::SCALE)?;
        println!(
            "CUDA:    {kernel:7.2} ms in the kernel, from {}",
            module.origin()
        );
        println!("         {upload:7.2} ms up, {download:7.2} ms down, from pageable memory");

        let comparison = compare(&on_cpu, &on_gpu, Tolerance::EXACT)?;
        println!("Twins:   {comparison}");
        if !comparison.passes() {
            return Err(tetera_core::Error::internal(format!(
                "the twins disagree: {comparison}"
            )));
        }
    }

    let checksum: f64 = on_cpu.iter().map(|&v| f64::from(v)).sum();
    println!("Mean:    {:.6}", checksum / n as f64);
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
