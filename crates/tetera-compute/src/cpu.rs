//! The CPU backend: a rayon pool of its own (ADR-0009). It is the reference every CUDA
//! kernel is tested against (P1.04b), and the fallback without a GPU.

use std::sync::Arc;
use tetera_core::{Error, Result};

/// The CPU, as a pool of worker threads. Cloning it shares the pool.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    pool: Arc<rayon::ThreadPool>,
}

impl CpuBackend {
    /// A pool of `threads` workers, or one per CPU if `threads` is 0.
    pub fn new(threads: usize) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("tetera-cpu-{i}"))
            .build()
            .map_err(|e| {
                Error::internal(format!("the CPU's worker threads wouldn't start: {e}"))
            })?;
        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    /// How many workers it has.
    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Runs `work` on the pool: rayon's parallel iterators inside it use the pool's workers.
    pub fn install<R: Send>(&self, work: impl FnOnce() -> R + Send) -> R {
        self.pool.install(work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn a_pool_has_the_workers_asked_for() {
        assert_eq!(CpuBackend::new(3).unwrap().threads(), 3);
        let all = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(CpuBackend::new(0).unwrap().threads(), all);
    }

    #[test]
    fn work_runs_on_the_pools_own_workers() {
        let cpu = CpuBackend::new(2).unwrap();
        assert_eq!(
            rayon::current_thread_index(),
            None,
            "the test's own thread isn't a worker"
        );
        let (inside, sum) = cpu.install(|| {
            let name = std::thread::current().name().map(str::to_string);
            (name, (1..=1_000_000u64).into_par_iter().sum::<u64>())
        });
        assert!(inside.unwrap().starts_with("tetera-cpu-"));
        assert_eq!(sum, 500_000_500_000);
        let shared = cpu.clone();
        assert_eq!(shared.threads(), 2);
    }
}
