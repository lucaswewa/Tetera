//! The `jobs` Thing (ADR-0020): the jobs queued and running, the history, and cancel, with an
//! event for every change; and the job service's options, the server's `[jobs]`.

use crate::error;
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;
use teta_wot::prelude::*;
use tetera_jobs::{Job, JobService, JobsConfig};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The job service's options: `[jobs]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct JobsOptions {
    /// Compute jobs that run at once.
    pub compute_workers: usize,
    /// Days the history keeps a job after it ends.
    pub keep_days: u32,
    /// The VRAM compute jobs share, MiB, on the GPU.
    pub vram_mib: u64,
}

impl Default for JobsOptions {
    /// Two compute jobs at once, a history of 30 days, and 1 GiB of VRAM.
    fn default() -> Self {
        Self {
            compute_workers: 2,
            keep_days: 30,
            vram_mib: 1_024,
        }
    }
}

impl JobsOptions {
    /// The service's configuration.
    pub fn config(&self) -> JobsConfig {
        JobsConfig {
            compute_workers: self.compute_workers,
            keep: Duration::from_secs(u64::from(self.keep_days) * 24 * 3_600),
            ..JobsConfig::default()
        }
    }
}

/// The jobs.
#[derive(Thing)]
pub struct JobsThing {
    /// The jobs queued and running, oldest first.
    #[property(readonly, default = Vec::new())]
    active: Prop<Vec<Job>>,
    /// Every change to a job: its state, progress, message or log.
    #[event]
    changed: Event<Job>,
    service: OnceLock<Arc<JobService>>,
    forwarder: Mutex<Option<(Arc<AtomicBool>, JoinHandle<()>)>>,
}

impl JobsThing {
    fn service(&self) -> Result<&Arc<JobService>, ActionError> {
        self.service
            .get()
            .ok_or_else(|| ActionError::handled("the jobs haven't started"))
    }
}

#[thing_impl]
impl JobsThing {
    /// Finds the job service, and follows its jobs.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let server = ctx.server();
        let service = server
            .service::<JobService>()
            .context("the server has no job service")?;
        let me = server
            .thing_ref::<JobsThing>(ctx.name())
            .context("the jobs Thing isn't on the server")?;
        let me = Arc::downgrade(ThingRef::thing(&me));
        let changes = service.subscribe();
        let _ = self.active.set(service.active());
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let jobs = service.clone();
        let forwarder = std::thread::Builder::new()
            .name("jobs Thing".into())
            .spawn(move || {
                while !stopping.load(Ordering::Acquire) {
                    match changes.recv_timeout(Duration::from_millis(100)) {
                        Ok(job) => {
                            let Some(me) = me.upgrade() else { break };
                            me.changed.emit(job);
                            let _ = me.active.set(jobs.active());
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .context("the jobs Thing's thread won't start")?;
        *lock(&self.forwarder) = Some((stop, forwarder));
        let _ = self.service.set(service);
        Ok(())
    }

    /// Stops following the jobs.
    #[on_stop]
    async fn stop_following(&self) {
        let forwarder = lock(&self.forwarder).take();
        if let Some((stop, thread)) = forwarder {
            stop.store(true, Ordering::Release);
            let _ = tokio::task::spawn_blocking(move || thread.join()).await;
        }
    }

    /// Job `id`, as it stands, or as it ended.
    #[action(global_lock = false)]
    async fn get(&self, id: String) -> Result<Job, ActionError> {
        self.service()?.get(&id).map_err(error::action)
    }

    /// The newest jobs, newest first: `limit` of them, 50 when not given.
    #[action(global_lock = false)]
    async fn list(&self, #[param(default)] limit: Option<usize>) -> Result<Vec<Job>, ActionError> {
        let service = self.service()?.clone();
        tokio::task::spawn_blocking(move || service.recent(limit.unwrap_or(50)))
            .await
            .map_err(anyhow::Error::from)?
            .map_err(error::action)
    }

    /// Cancels job `id`: one queued ends at once, one running when it next checks.
    #[action(global_lock = false)]
    async fn cancel(&self, id: String) -> Result<Job, ActionError> {
        self.service()?.cancel(&id).map_err(error::action)
    }
}

#[cfg(test)]
impl JobsThing {
    /// Whether it follows the jobs.
    pub(crate) fn following(&self) -> bool {
        lock(&self.forwarder).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_options_make_the_services_configuration() {
        let c = JobsOptions {
            compute_workers: 3,
            keep_days: 2,
            vram_mib: 64,
        }
        .config();
        assert_eq!(
            (c.compute_workers, c.keep),
            (3, Duration::from_secs(2 * 86_400))
        );
        assert_eq!(
            JobsOptions::default().config().keep,
            Duration::from_secs(30 * 86_400)
        );
    }
}
