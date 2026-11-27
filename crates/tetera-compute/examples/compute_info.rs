//! What computes here: the backend chosen and why, the CPU's workers, the GPU and its driver,
//! and each CUDA feature Tetera plans to use, tried for real.
//!
//! Run it with `cargo run -p tetera-compute --example compute_info --features cuda` for the
//! GPU, or without `--features cuda` to see the CPU's fallback. `-- --smoke` does the same
//! and says so.

use tetera_compute::Compute;
use tetera_compute::features::{Outcome, check_all};

fn main() -> tetera_core::Result<()> {
    let smoke = std::env::args().any(|arg| arg == "--smoke");
    let compute = Compute::detect()?;
    println!("Backend: {}: {}", compute.kind(), compute.why());
    println!("CPU:     {} worker threads", compute.cpu().threads());

    #[cfg(feature = "cuda")]
    if let Some(gpu) = compute.cuda() {
        let info = gpu.info();
        let (free, total) = gpu.memory()?;
        println!(
            "GPU:     {}, compute capability {}.{}",
            info.name, info.compute_capability.0, info.compute_capability.1
        );
        println!(
            "         {:.1} GiB, {:.1} GiB free",
            total as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64
        );
        println!(
            "Driver:  runs CUDA up to {}.{}",
            info.driver_cuda.0, info.driver_cuda.1
        );
    }

    println!("\nWhat CUDA offers here:");
    for check in check_all(&compute) {
        let (mark, detail) = match &check.outcome {
            Outcome::Works(seen) => ("works", seen.as_str()),
            Outcome::Fails(why) => ("FAILS", why.as_str()),
            Outcome::NotTried(why) => ("-", why.as_str()),
        };
        println!("  {:<19} {mark:<6} {detail}", check.name);
        println!("  {:<19} {:<6} ({})", "", "", check.what);
    }

    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
