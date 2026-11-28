//! The job service (ADR-0020): hardware jobs one at a time from a queue, compute jobs on a
//! pool of workers that share the GPU through a VRAM budget, each job followed, cancelled
//! through a token it checks, and kept in the history.

use crate::history::History;
use crate::job::{Job, JobKind, JobState, LogLine};
use std::collections::{BTreeMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetera_compute::budget::{Budget, Grant};
use tetera_core::{Error, ErrorKind, Result, UtcTime};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The time `d` ago.
fn ago(d: Duration) -> UtcTime {
    let nanos = i64::try_from(d.as_nanos()).unwrap_or(i64::MAX);
    UtcTime::from_unix_nanos(UtcTime::now().unix_nanos().saturating_sub(nanos))
}

/// How the service runs.
#[derive(Debug, Clone, PartialEq)]
pub struct JobsConfig {
    /// Compute jobs that run at once.
    pub compute_workers: usize,
    /// How long the history keeps a job after it ends.
    pub keep: Duration,
    /// The log lines each job keeps, the latest.
    pub log_lines: usize,
}

impl Default for JobsConfig {
    /// Two compute jobs at once; a history of 30 days; 200 log lines.
    fn default() -> Self {
        Self {
            compute_workers: 2,
            keep: Duration::from_secs(30 * 24 * 3_600),
            log_lines: 200,
        }
    }
}

/// A job's work: it reports through its context and returns its result.
type Work = Box<dyn FnOnce(&JobContext) -> Result<serde_json::Value> + Send>;

struct Pending {
    id: String,
    vram: u64,
    work: Work,
}

/// A job in memory, with its cancel token.
struct Entry {
    job: Job,
    cancel: Arc<AtomicBool>,
}

struct Shared {
    config: JobsConfig,
    jobs: Mutex<BTreeMap<String, Entry>>,
    hardware: Mutex<VecDeque<Pending>>,
    compute: Mutex<VecDeque<Pending>>,
    /// Signalled, with `hardware`, when a hardware job is queued or the service stops.
    hardware_ready: Condvar,
    /// Signalled, with `compute`, when a compute job is queued, one frees its VRAM, or the
    /// service stops.
    compute_ready: Condvar,
    history: Mutex<History>,
    listeners: Mutex<Vec<Sender<Job>>>,
    budget: Option<Budget>,
    stopping: AtomicBool,
}

/// The job service. Dropping it cancels what is queued and running, and waits for the
/// running jobs to stop.
pub struct JobService {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for JobService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobService")
            .field("config", &self.shared.config)
            .finish_non_exhaustive()
    }
}

/// What a running job reports through, and checks for cancel.
pub struct JobContext {
    id: String,
    cancel: Arc<AtomicBool>,
    shared: Arc<Shared>,
}

impl JobContext {
    /// The job's id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// How far it has got, 0 to 1, and what it is doing.
    pub fn progress(&self, fraction: f64, message: impl Into<String>) {
        let message = message.into();
        self.shared.update(&self.id, |j| {
            j.progress = fraction.clamp(0.0, 1.0);
            j.message = message;
        });
    }

    /// A line for its log.
    pub fn log(&self, text: impl Into<String>) {
        let line = LogLine {
            time: UtcTime::now(),
            text: text.into(),
        };
        let keep = self.shared.config.log_lines;
        self.shared.update(&self.id, |j| {
            j.log.push(line);
            let over = j.log.len().saturating_sub(keep);
            j.log.drain(..over);
        });
    }

    /// Whether it has been asked to stop.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    /// An error once it has been asked to stop: `?` it often, at least every 100 ms.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(Error::cancelled(format!("job {} was cancelled", self.id)));
        }
        Ok(())
    }

    /// Waits `d`, or less if it is asked to stop: then the cancel error.
    pub fn sleep(&self, d: Duration) -> Result<()> {
        let until = Instant::now() + d;
        loop {
            self.check()?;
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(());
            }
            std::thread::sleep(left.min(Duration::from_millis(10)));
        }
    }
}

impl Shared {
    /// Changes job `id` in memory, and tells the listeners.
    fn update(&self, id: &str, change: impl FnOnce(&mut Job)) {
        let job = {
            let mut jobs = lock(&self.jobs);
            let Some(entry) = jobs.get_mut(id) else {
                return;
            };
            change(&mut entry.job);
            entry.job.clone()
        };
        self.tell(&job);
    }

    fn tell(&self, job: &Job) {
        lock(&self.listeners).retain(|l| l.send(job.clone()).is_ok());
    }

    /// Writes job `id` to the history, if it is still in memory: a failure is logged, as the
    /// job goes on. The jobs' lock is held while it writes, so it can't write over the record
    /// of a job that ended meanwhile: the jobs' lock, then the history's, always.
    fn save(&self, id: &str) {
        let jobs = lock(&self.jobs);
        if let Some(e) = jobs.get(id)
            && let Err(err) = lock(&self.history).save(&e.job)
        {
            tracing::warn!("job {id} not saved: {err}");
        }
    }

    /// Ends job `id` in `state`: saves it, lets the ended jobs past the history's age go,
    /// tells the listeners, and lets it leave memory, all under the jobs' lock, so whoever sees
    /// it ended finds it in the history, and has heard its end.
    fn end(&self, id: &str, state: JobState, change: impl FnOnce(&mut Job)) {
        let mut jobs = lock(&self.jobs);
        let Some(entry) = jobs.get(id) else {
            return;
        };
        let mut job = entry.job.clone();
        change(&mut job);
        job.finish(state);
        {
            let history = lock(&self.history);
            if let Err(e) = history.save(&job) {
                tracing::warn!("job {id} not saved: {e}");
            }
            if let Err(e) = history.prune(ago(self.config.keep)) {
                tracing::warn!("the jobs' history not pruned: {e}");
            }
        }
        self.tell(&job);
        jobs.remove(id);
        drop(jobs);
        self.wake();
    }

    /// Wakes the workers waiting for a job or for VRAM. Each queue's lock is taken first, so
    /// a worker between its look and its wait can't miss the call.
    fn wake(&self) {
        {
            let _queue = lock(&self.hardware);
            self.hardware_ready.notify_all();
        }
        let _queue = lock(&self.compute);
        self.compute_ready.notify_all();
    }

    /// The next job for a worker of `kind`, waiting for one; `None` once the service stops.
    fn next(&self, kind: JobKind) -> Option<Pending> {
        let (queue, ready) = match kind {
            JobKind::Hardware => (&self.hardware, &self.hardware_ready),
            JobKind::Compute => (&self.compute, &self.compute_ready),
        };
        let mut q = lock(queue);
        loop {
            if self.stopping.load(Ordering::Acquire) {
                return None;
            }
            if let Some(p) = q.pop_front() {
                return Some(p);
            }
            q = ready.wait(q).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Runs `pending` on this thread, holding its VRAM while it runs.
    fn run(self: &Arc<Self>, pending: Pending) {
        let Pending { id, vram, work } = pending;
        let Some(cancel) = lock(&self.jobs).get(&id).map(|e| e.cancel.clone()) else {
            return;
        };
        let grant = match self.vram(&id, vram, &cancel) {
            Ok(grant) => grant,
            Err(e) => {
                let state = if e.kind() == ErrorKind::Cancelled {
                    JobState::Cancelled
                } else {
                    JobState::Failed
                };
                self.end(&id, state, |j| j.error = Some(e.to_string()));
                return;
            }
        };
        if cancel.load(Ordering::Acquire) {
            self.end(&id, JobState::Cancelled, |j| {
                j.error = Some("cancelled before it ran".into());
            });
            return;
        }
        self.update(&id, |j| {
            j.state = JobState::Running;
            j.started = Some(UtcTime::now());
        });
        self.save(&id);
        let ctx = JobContext {
            id: id.clone(),
            cancel: cancel.clone(),
            shared: self.clone(),
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| work(&ctx)));
        drop(grant);
        let cancelled = cancel.load(Ordering::Acquire);
        match outcome {
            Ok(Ok(result)) => self.end(&id, JobState::Succeeded, |j| j.result = Some(result)),
            Ok(Err(e)) if cancelled || e.kind() == ErrorKind::Cancelled => {
                self.end(&id, JobState::Cancelled, |j| j.error = Some(e.to_string()));
            }
            Ok(Err(e)) => self.end(&id, JobState::Failed, |j| j.error = Some(e.to_string())),
            Err(panic) => {
                let why = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                self.end(&id, JobState::Failed, |j| {
                    j.error = Some(format!("the job panicked: {why}"));
                });
            }
        }
    }

    /// The VRAM a compute job asked for, from the budget, waiting while others hold it.
    fn vram(&self, id: &str, bytes: u64, cancel: &AtomicBool) -> Result<Option<Grant>> {
        let Some(budget) = self.budget.as_ref() else {
            return Ok(None);
        };
        if bytes == 0 {
            return Ok(None);
        }
        if bytes > budget.limit() {
            return Err(Error::invalid_argument(format!(
                "job {id} asks for {bytes} bytes of VRAM, more than the compute jobs' {}",
                budget.limit()
            )));
        }
        let mut waiting = lock(&self.compute);
        loop {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::cancelled(format!(
                    "job {id} was cancelled waiting for VRAM"
                )));
            }
            if self.stopping.load(Ordering::Acquire) {
                return Err(Error::cancelled("the job service stopped"));
            }
            if let Ok(grant) = budget.allocate(bytes) {
                return Ok(Some(grant));
            }
            waiting = self
                .compute_ready
                .wait(waiting)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl JobService {
    /// The service, its history in the SQLite file `db`: the jobs interrupted by its last stop
    /// marked failed, and those past `config.keep` pruned. Compute jobs share `budget`'s VRAM,
    /// or none is counted.
    pub fn open(db: &Path, config: JobsConfig, budget: Option<Budget>) -> Result<Self> {
        if config.compute_workers == 0 || config.log_lines == 0 {
            return Err(Error::invalid_argument(format!(
                "a job service runs a compute job or more, and keeps a log line or more: {config:?}"
            )));
        }
        let history = History::open(db)?;
        let interrupted = history.interrupted()?;
        let pruned = history.prune(ago(config.keep))?;
        tracing::info!(interrupted, pruned, "the jobs' history");
        let shared = Arc::new(Shared {
            jobs: Mutex::default(),
            hardware: Mutex::default(),
            compute: Mutex::default(),
            hardware_ready: Condvar::new(),
            compute_ready: Condvar::new(),
            history: Mutex::new(history),
            listeners: Mutex::default(),
            budget,
            stopping: AtomicBool::new(false),
            config,
        });
        let mut workers = Vec::new();
        let kinds = std::iter::once(JobKind::Hardware).chain(std::iter::repeat_n(
            JobKind::Compute,
            shared.config.compute_workers,
        ));
        for (n, kind) in kinds.enumerate() {
            let s = shared.clone();
            let worker = std::thread::Builder::new()
                .name(format!("jobs {n}"))
                .spawn(move || {
                    while let Some(p) = s.next(kind) {
                        s.run(p);
                    }
                })
                .map_err(|e| Error::internal(format!("a job worker won't start: {e}")))?;
            workers.push(worker);
        }
        Ok(Self { shared, workers })
    }

    /// Queues `work` as a job of `kind` named `name`, asking a compute job's `vram` bytes of
    /// the budget: its id.
    pub fn submit(
        &self,
        kind: JobKind,
        name: &str,
        vram: u64,
        work: impl FnOnce(&JobContext) -> Result<serde_json::Value> + Send + 'static,
    ) -> Result<String> {
        if self.shared.stopping.load(Ordering::Acquire) {
            return Err(Error::busy("the job service is stopping"));
        }
        let id = uuid::Uuid::now_v7().to_string();
        let vram = if kind == JobKind::Compute { vram } else { 0 };
        let job = Job::new(id.clone(), kind, name.to_string(), vram);
        lock(&self.shared.jobs).insert(
            id.clone(),
            Entry {
                job: job.clone(),
                cancel: Arc::default(),
            },
        );
        self.shared.save(&id);
        self.shared.tell(&job);
        let pending = Pending {
            id: id.clone(),
            vram,
            work: Box::new(work),
        };
        match kind {
            JobKind::Hardware => lock(&self.shared.hardware).push_back(pending),
            JobKind::Compute => lock(&self.shared.compute).push_back(pending),
        }
        self.shared.wake();
        Ok(id)
    }

    /// Job `id`, from memory while it is queued or running, else from the history.
    pub fn get(&self, id: &str) -> Result<Job> {
        if let Some(e) = lock(&self.shared.jobs).get(id) {
            return Ok(e.job.clone());
        }
        lock(&self.shared.history)
            .get(id)?
            .ok_or_else(|| Error::not_found(format!("no job {id}")))
    }

    /// The jobs queued and running, oldest first.
    pub fn active(&self) -> Vec<Job> {
        let mut jobs: Vec<Job> = lock(&self.shared.jobs)
            .values()
            .filter(|e| !e.job.state.is_finished())
            .map(|e| e.job.clone())
            .collect();
        jobs.sort_by(|a, b| a.id.cmp(&b.id));
        jobs
    }

    /// The `limit` newest jobs in the history, newest first, as they last stood.
    pub fn recent(&self, limit: usize) -> Result<Vec<Job>> {
        lock(&self.shared.history).recent(limit)
    }

    /// Cancels job `id`: one queued ends at once; one running is asked to stop, and ends when
    /// its work next checks. One that has ended is refused.
    pub fn cancel(&self, id: &str) -> Result<Job> {
        {
            let jobs = lock(&self.shared.jobs);
            let Some(entry) = jobs.get(id).filter(|e| !e.job.state.is_finished()) else {
                drop(jobs);
                let job = self.get(id)?;
                return Err(Error::invalid_argument(format!(
                    "job {id} has ended: {}",
                    job.state.name()
                )));
            };
            entry.cancel.store(true, Ordering::Release);
        }
        // Still in a queue: no worker has it, so it ends here.
        let taken = [&self.shared.hardware, &self.shared.compute]
            .into_iter()
            .any(|q| {
                let mut q = lock(q);
                q.iter()
                    .position(|p| p.id == id)
                    .and_then(|i| q.remove(i))
                    .is_some()
            });
        if taken {
            self.shared.end(id, JobState::Cancelled, |j| {
                j.cancel_requested = true;
                j.error = Some("cancelled before it ran".into());
            });
        } else {
            self.shared.update(id, |j| j.cancel_requested = true);
            self.shared.save(id);
            // A compute job waiting for VRAM wakes to see it.
            self.shared.wake();
        }
        self.get(id)
    }

    /// Waits up to `timeout` for job `id` to end: the job as it ended.
    pub fn wait(&self, id: &str, timeout: Duration) -> Result<Job> {
        let until = Instant::now() + timeout;
        loop {
            let job = self.get(id)?;
            if job.state.is_finished() {
                return Ok(job);
            }
            if Instant::now() >= until {
                return Err(Error::timeout(format!(
                    "job {id} still {} after {timeout:?}",
                    job.state.name()
                )));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Every change to every job from now: its state, progress, message and log.
    pub fn subscribe(&self) -> Receiver<Job> {
        let (tx, rx) = channel();
        lock(&self.shared.listeners).push(tx);
        rx
    }

    /// Stops: the queued jobs are cancelled, the running ones asked to stop and waited for.
    pub fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        let s = &self.shared;
        s.stopping.store(true, Ordering::Release);
        let queued: Vec<String> = [&s.hardware, &s.compute]
            .iter()
            .flat_map(|q| lock(q).drain(..).map(|p| p.id).collect::<Vec<_>>())
            .collect();
        for id in queued {
            s.end(&id, JobState::Cancelled, |j| {
                j.error = Some("cancelled: the job service stopped".into());
            });
        }
        for e in lock(&s.jobs).values() {
            e.cancel.store(true, Ordering::Release);
        }
        s.wake();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
impl JobService {
    /// Stops it, keeping it, so a test can see what it does stopped.
    pub(crate) fn halt_for_tests(&mut self) {
        self.halt();
    }
}

impl Drop for JobService {
    fn drop(&mut self) {
        self.halt();
    }
}
