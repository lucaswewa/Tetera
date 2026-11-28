//! The job service: states, cancel at each state, the hardware queue's order and its one at a
//! time, the compute pool's limit and its VRAM, events, and the history across a restart.

use crate::{History, Job, JobKind, JobService, JobState, JobsConfig};
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tetera_compute::budget::VramBudgets;
use tetera_core::{Error, ErrorKind, UtcTime};

const MIB: u64 = 1 << 20;

const LONG: Duration = Duration::from_secs(10);

/// A folder of the test's own, removed when it drops.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("tetera-jobs-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn db(&self) -> PathBuf {
        self.0.join("tetera.db")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn service(scratch: &Scratch) -> JobService {
    JobService::open(&scratch.db(), JobsConfig::default(), None).unwrap()
}

/// Waits for job `id` to run, or to have run.
fn until_running(jobs: &JobService, id: &str) {
    let started = Instant::now();
    while jobs.get(id).unwrap().state == JobState::Queued {
        assert!(started.elapsed() < LONG, "job {id} never ran");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_job_goes_from_queued_to_running_to_its_end_with_its_progress_log_and_result() {
    let scratch = Scratch::new();
    let jobs = service(&scratch);
    let events = jobs.subscribe();
    let (go, gate) = std::sync::mpsc::channel::<()>();
    let id = jobs
        .submit(JobKind::Hardware, "capture", 0, move |ctx| {
            gate.recv().unwrap();
            ctx.progress(0.5, "half way");
            ctx.log("a line");
            ctx.progress(7.0, "past the end");
            Ok(json!({"frames": 3, "id": ctx.id()}))
        })
        .unwrap();
    until_running(&jobs, &id);
    let running = jobs.get(&id).unwrap();
    assert!(running.started.is_some() && running.finished.is_none());
    assert_eq!(jobs.active().len(), 1);
    go.send(()).unwrap();
    let done = jobs.wait(&id, LONG).unwrap();
    assert_eq!(done.state, JobState::Succeeded);
    assert_eq!(
        (done.progress, done.message.as_str()),
        (1.0, "past the end")
    );
    assert_eq!(
        done.log.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(),
        ["a line"]
    );
    assert_eq!(done.result, Some(json!({"frames": 3, "id": id})));
    assert!(done.finished >= done.started && done.started >= Some(done.created));
    assert_eq!(
        (done.kind, done.name.as_str()),
        (JobKind::Hardware, "capture")
    );
    assert!(jobs.active().is_empty());
    // Each change, in order.
    let seen: Vec<Job> = events.try_iter().collect();
    let states: Vec<JobState> = seen.iter().map(|j| j.state).collect();
    assert_eq!(states.first(), Some(&JobState::Queued));
    assert_eq!(states.last(), Some(&JobState::Succeeded));
    assert!(states.contains(&JobState::Running));
    let half = seen.iter().find(|j| j.message == "half way").unwrap();
    assert_eq!((half.state, half.progress), (JobState::Running, 0.5));
    let past = seen.iter().find(|j| j.message == "past the end").unwrap();
    assert_eq!(past.progress, 1.0, "at most 1");
    assert!(format!("{jobs:?}").contains("JobService"));
}

#[test]
fn a_failing_or_panicking_job_fails_with_its_reason_and_the_next_one_runs() {
    let scratch = Scratch::new();
    let jobs = service(&scratch);
    let failing = jobs
        .submit(JobKind::Hardware, "move", 0, |ctx| {
            ctx.progress(0.3, "moving");
            Err(Error::hardware("the stage stalled"))
        })
        .unwrap();
    let ends_well = jobs
        .submit(JobKind::Hardware, "move", 0, |ctx| {
            ctx.progress(0.3, "moving");
            Ok(json!(null))
        })
        .unwrap();
    let panicking = jobs
        .submit(JobKind::Compute, "sum", 0, |_| panic!("out of range"))
        .unwrap();
    let panicking_owned = jobs
        .submit(JobKind::Compute, "sum", 0, |_| {
            panic!("{}", String::from("owned"))
        })
        .unwrap();
    let after = jobs
        .submit(JobKind::Hardware, "move", 0, |_| Ok(json!(null)))
        .unwrap();
    let f = jobs.wait(&failing, LONG).unwrap();
    assert_eq!(f.state, JobState::Failed);
    assert_eq!(f.progress, 0.3, "where it got to");
    assert_eq!(
        jobs.wait(&ends_well, LONG).unwrap().progress,
        1.0,
        "all the way"
    );
    assert!(
        f.error
            .as_deref()
            .is_some_and(|e| e.contains("the stage stalled")),
        "{f:?}"
    );
    let p = jobs.wait(&panicking, LONG).unwrap();
    assert_eq!(p.state, JobState::Failed);
    assert_eq!(p.error.as_deref(), Some("the job panicked: out of range"));
    let o = jobs.wait(&panicking_owned, LONG).unwrap();
    assert_eq!(o.error.as_deref(), Some("the job panicked: owned"));
    assert_eq!(jobs.wait(&after, LONG).unwrap().state, JobState::Succeeded);
    assert_eq!(
        jobs.get("no such job").unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

#[test]
fn hardware_jobs_run_one_at_a_time_in_the_order_they_came() {
    let scratch = Scratch::new();
    let jobs = service(&scratch);
    let running = Arc::new(AtomicUsize::new(0));
    let most = Arc::new(AtomicUsize::new(0));
    let order = Arc::new(Mutex::new(Vec::new()));
    let ids: Vec<String> = (0..6)
        .map(|n| {
            let (running, most, order) = (running.clone(), most.clone(), order.clone());
            jobs.submit(JobKind::Hardware, "move", 0, move |ctx| {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(now, Ordering::SeqCst);
                order.lock().unwrap().push(n);
                ctx.sleep(Duration::from_millis(20))?;
                running.fetch_sub(1, Ordering::SeqCst);
                Ok(json!(n))
            })
            .unwrap()
        })
        .collect();
    for id in &ids {
        assert_eq!(jobs.wait(id, LONG).unwrap().state, JobState::Succeeded);
    }
    assert_eq!(most.load(Ordering::SeqCst), 1, "never two at once");
    assert_eq!(
        *order.lock().unwrap(),
        (0..6).collect::<Vec<_>>(),
        "in order"
    );
}

#[test]
fn compute_jobs_run_side_by_side_up_to_the_pool_and_their_vram() {
    let scratch = Scratch::new();
    let budgets = VramBudgets::new(1_024 * MIB);
    let config = JobsConfig {
        compute_workers: 3,
        ..JobsConfig::default()
    };
    let jobs = JobService::open(
        &scratch.db(),
        config,
        Some(budgets.reserve("compute jobs", 512 * MIB).unwrap()),
    )
    .unwrap();
    let run = |vram: u64, n: usize| {
        let running = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let ids: Vec<String> = (0..n)
            .map(|_| {
                let (running, most) = (running.clone(), most.clone());
                jobs.submit(JobKind::Compute, "denoise", vram, move |ctx| {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    most.fetch_max(now, Ordering::SeqCst);
                    ctx.sleep(Duration::from_millis(60))?;
                    running.fetch_sub(1, Ordering::SeqCst);
                    Ok(json!(null))
                })
                .unwrap()
            })
            .collect();
        for id in &ids {
            assert_eq!(jobs.wait(id, LONG).unwrap().state, JobState::Succeeded);
        }
        most.load(Ordering::SeqCst)
    };
    assert_eq!(run(0, 6), 3, "the pool's three, asking no VRAM");
    assert_eq!(run(200 * MIB, 6), 2, "200 MiB each of 512: two at once");
    assert_eq!(budgets.reserved(), 512 * MIB, "the budget's grants gone");
    // More than the whole budget: refused when its turn comes.
    let big = jobs
        .submit(JobKind::Compute, "deconvolve", 600 * MIB, |_| {
            Ok(json!(null))
        })
        .unwrap();
    let b = jobs.wait(&big, LONG).unwrap();
    assert_eq!(b.state, JobState::Failed);
    assert!(b.error.unwrap().contains("VRAM"));
    assert_eq!(b.vram, 600 * MIB);
    // Exactly the budget: it runs.
    let all = jobs
        .submit(JobKind::Compute, "deconvolve", 512 * MIB, |_| {
            Ok(json!(null))
        })
        .unwrap();
    assert_eq!(jobs.wait(&all, LONG).unwrap().state, JobState::Succeeded);
    // A hardware job asks no VRAM.
    let h = jobs
        .submit(JobKind::Hardware, "move", 600 * MIB, |_| Ok(json!(null)))
        .unwrap();
    assert_eq!(jobs.wait(&h, LONG).unwrap().vram, 0);
}

#[test]
fn a_job_cancelled_while_queued_never_runs_and_one_running_stops_within_a_second() {
    let scratch = Scratch::new();
    let jobs = service(&scratch);
    let ran = Arc::new(AtomicUsize::new(0));
    let busy = jobs
        .submit(JobKind::Hardware, "scan", 0, |ctx| {
            ctx.sleep(Duration::from_secs(60))?;
            Ok(json!(null))
        })
        .unwrap();
    let queued = {
        let ran = ran.clone();
        jobs.submit(JobKind::Hardware, "capture", 0, move |_| {
            ran.fetch_add(1, Ordering::SeqCst);
            Ok(json!(null))
        })
        .unwrap()
    };
    until_running(&jobs, &busy);
    let q = jobs.cancel(&queued).unwrap();
    assert_eq!(q.state, JobState::Cancelled, "at once");
    assert!(q.cancel_requested && q.started.is_none());
    let asked = Instant::now();
    assert!(jobs.cancel(&busy).unwrap().cancel_requested);
    let b = jobs.wait(&busy, LONG).unwrap();
    assert_eq!(b.state, JobState::Cancelled);
    assert!(
        asked.elapsed() < Duration::from_secs(1),
        "{:?}",
        asked.elapsed()
    );
    assert!(b.error.is_some_and(|e| e.contains("cancelled")));
    // Ended: refused.
    assert_eq!(
        jobs.cancel(&busy).unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        jobs.cancel("no such job").unwrap_err().kind(),
        ErrorKind::NotFound
    );
    // The next job runs; the cancelled one never did.
    let next = jobs
        .submit(JobKind::Hardware, "capture", 0, |_| Ok(json!(1)))
        .unwrap();
    assert_eq!(jobs.wait(&next, LONG).unwrap().state, JobState::Succeeded);
    assert_eq!(ran.load(Ordering::SeqCst), 0);
    // A job that misses its token but ends in an error is cancelled all the same; one that
    // ends well despite it succeeds.
    for (ok, state) in [(false, JobState::Cancelled), (true, JobState::Succeeded)] {
        let (go, gate) = std::sync::mpsc::channel::<()>();
        let id = jobs
            .submit(JobKind::Hardware, "deaf", 0, move |_| {
                gate.recv().unwrap();
                if ok {
                    Ok(json!(null))
                } else {
                    Err(Error::internal("it ended its own way"))
                }
            })
            .unwrap();
        until_running(&jobs, &id);
        let asked = jobs.cancel(&id).unwrap();
        assert_eq!(
            (asked.state, asked.cancel_requested),
            (JobState::Running, true),
            "running until its work ends"
        );
        go.send(()).unwrap();
        assert_eq!(jobs.wait(&id, LONG).unwrap().state, state);
    }
}

#[test]
fn a_compute_job_waiting_for_vram_can_be_cancelled() {
    let scratch = Scratch::new();
    let budgets = VramBudgets::new(1_024 * MIB);
    let jobs = JobService::open(
        &scratch.db(),
        JobsConfig::default(),
        Some(budgets.reserve("compute jobs", 512 * MIB).unwrap()),
    )
    .unwrap();
    let holder = jobs
        .submit(JobKind::Compute, "hold", 400 * MIB, |ctx| {
            ctx.sleep(Duration::from_secs(60))?;
            Ok(json!(null))
        })
        .unwrap();
    until_running(&jobs, &holder);
    // The second worker takes it, and waits for VRAM.
    let waiting = jobs
        .submit(JobKind::Compute, "wait", 400 * MIB, |_| Ok(json!(null)))
        .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(jobs.get(&waiting).unwrap().state, JobState::Queued);
    let asked = Instant::now();
    jobs.cancel(&waiting).unwrap();
    let w = jobs.wait(&waiting, LONG).unwrap();
    assert_eq!(w.state, JobState::Cancelled);
    assert!(asked.elapsed() < Duration::from_secs(1));
    jobs.cancel(&holder).unwrap();
    assert_eq!(jobs.wait(&holder, LONG).unwrap().state, JobState::Cancelled);
}

#[test]
fn the_history_survives_a_restart_marks_the_interrupted_and_prunes_the_old() {
    let scratch = Scratch::new();
    let (done, cut_short) = {
        let jobs = service(&scratch);
        let done = jobs
            .submit(JobKind::Compute, "sum", 0, |ctx| {
                ctx.log("summing");
                Ok(json!(42))
            })
            .unwrap();
        jobs.wait(&done, LONG).unwrap();
        let cut_short = jobs
            .submit(JobKind::Hardware, "scan", 0, |ctx| {
                ctx.sleep(Duration::from_secs(60))?;
                Ok(json!(null))
            })
            .unwrap();
        let queued = jobs
            .submit(JobKind::Hardware, "capture", 0, |_| Ok(json!(null)))
            .unwrap();
        until_running(&jobs, &cut_short);
        // Stopping: the running job is cancelled, and the queued one.
        let asked = Instant::now();
        jobs.stop();
        assert!(asked.elapsed() < Duration::from_secs(1));
        let h = History::open(&scratch.db()).unwrap();
        assert_eq!(h.get(&queued).unwrap().unwrap().state, JobState::Cancelled);
        assert_eq!(
            h.get(&cut_short).unwrap().unwrap().state,
            JobState::Cancelled
        );
        (done, cut_short)
    };
    // A job left running, as by a crash: the next start marks it failed, interrupted.
    {
        let h = History::open(&scratch.db()).unwrap();
        let mut left = h.get(&cut_short).unwrap().unwrap();
        left.id = "left-running".into();
        left.state = JobState::Running;
        left.finished = None;
        h.save(&left).unwrap();
    }
    let jobs = service(&scratch);
    let d = jobs.get(&done).unwrap();
    assert_eq!(
        (d.state, d.result.clone()),
        (JobState::Succeeded, Some(json!(42)))
    );
    assert_eq!(d.log[0].text, "summing");
    let left = jobs.get("left-running").unwrap();
    assert_eq!(left.state, JobState::Failed);
    assert!(left.error.unwrap().contains("interrupted"));
    let recent = jobs.recent(10).unwrap();
    assert_eq!(recent.len(), 4);
    assert!(
        recent.windows(2).all(|w| w[0].created >= w[1].created),
        "newest first"
    );
    assert_eq!(jobs.recent(1).unwrap().len(), 1);
    assert!(
        jobs.submit(JobKind::Hardware, "late", 0, |_| Ok(json!(null)))
            .is_ok()
    );
    drop(jobs);
    // Kept 50 ms: the ended jobs are pruned when the service opens, and as jobs end.
    let brief = JobsConfig {
        keep: Duration::from_millis(50),
        ..JobsConfig::default()
    };
    std::thread::sleep(Duration::from_millis(100));
    let jobs = JobService::open(&scratch.db(), brief, None).unwrap();
    assert!(jobs.recent(10).unwrap().is_empty(), "all pruned");
    let first = jobs
        .submit(JobKind::Hardware, "move", 0, |_| Ok(json!(null)))
        .unwrap();
    jobs.wait(&first, LONG).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let later = jobs
        .submit(JobKind::Hardware, "move", 0, |_| Ok(json!(null)))
        .unwrap();
    let l = jobs.wait(&later, LONG).unwrap();
    let kept: Vec<String> = jobs.recent(10).unwrap().into_iter().map(|j| j.id).collect();
    assert_eq!(kept, [later], "the first pruned as the later one ended");
    // An unfinished job is never pruned.
    let h = History::open(&scratch.db()).unwrap();
    let mut open = l;
    open.id = "still-queued".into();
    open.state = JobState::Queued;
    open.finished = None;
    h.save(&open).unwrap();
    assert_eq!(h.prune(UtcTime::now()).unwrap(), 1, "the ended one alone");
    assert!(h.get("still-queued").unwrap().is_some());
}

#[test]
fn a_stopped_service_takes_no_more_jobs() {
    let scratch = Scratch::new();
    let mut jobs = service(&scratch);
    jobs.halt_for_tests();
    assert_eq!(
        jobs.submit(JobKind::Hardware, "move", 0, |_| Ok(json!(null)))
            .unwrap_err()
            .kind(),
        ErrorKind::Busy
    );
}

#[test]
fn a_jobs_log_keeps_its_latest_lines_and_bad_configurations_are_refused() {
    let scratch = Scratch::new();
    let config = JobsConfig {
        log_lines: 3,
        ..JobsConfig::default()
    };
    let jobs = JobService::open(&scratch.db(), config, None).unwrap();
    let id = jobs
        .submit(JobKind::Compute, "chatty", 0, |ctx| {
            for n in 0..5 {
                ctx.log(format!("line {n}"));
            }
            Ok(json!(null))
        })
        .unwrap();
    let j = jobs.wait(&id, LONG).unwrap();
    let lines: Vec<&str> = j.log.iter().map(|l| l.text.as_str()).collect();
    assert_eq!(lines, ["line 2", "line 3", "line 4"]);
    for bad in [
        JobsConfig {
            compute_workers: 0,
            ..JobsConfig::default()
        },
        JobsConfig {
            log_lines: 0,
            ..JobsConfig::default()
        },
    ] {
        assert!(JobService::open(&scratch.db(), bad, None).is_err());
    }
    let slow = jobs
        .submit(JobKind::Hardware, "slow", 0, |ctx| {
            ctx.sleep(Duration::from_millis(500))?;
            Ok(json!(null))
        })
        .unwrap();
    let e = jobs.wait(&slow, Duration::from_millis(20)).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Timeout);
    assert!(JobState::Cancelled.is_finished() && !JobState::Running.is_finished());
}

#[test]
fn a_job_cancelled_as_it_ends_is_recorded_as_it_ended() {
    // Cancel and the job's own end race: whichever writes the history last must write the
    // job as it ended. Many short jobs, each cancelled as it runs.
    let scratch = Scratch::new();
    let jobs = service(&scratch);
    for _ in 0..40 {
        let id = jobs
            .submit(JobKind::Hardware, "short", 0, |ctx| {
                ctx.sleep(Duration::from_millis(3))?;
                Ok(json!(null))
            })
            .unwrap();
        until_running(&jobs, &id);
        let _ = jobs.cancel(&id);
        let ended = jobs.wait(&id, Duration::from_secs(2)).unwrap();
        assert!(ended.state.is_finished(), "{ended:?}");
    }
}

#[test]
fn the_history_follows_a_job_as_it_goes_and_dropping_the_service_stops_it() {
    let scratch = Scratch::new();
    let jobs = service(&scratch);
    let (go, gate) = std::sync::mpsc::channel::<()>();
    let blocker = jobs
        .submit(JobKind::Hardware, "first", 0, move |_| {
            gate.recv().unwrap();
            Ok(json!(null))
        })
        .unwrap();
    let id = jobs
        .submit(JobKind::Hardware, "scan", 0, |ctx| {
            ctx.sleep(Duration::from_secs(60))?;
            Ok(json!(null))
        })
        .unwrap();
    // A second reader of the same history, as a crash would leave it.
    let h = History::open(&scratch.db()).unwrap();
    assert_eq!(h.get(&id).unwrap().unwrap().state, JobState::Queued);
    go.send(()).unwrap();
    until_running(&jobs, &id);
    assert_eq!(h.get(&id).unwrap().unwrap().state, JobState::Running);
    assert_eq!(h.get(&blocker).unwrap().unwrap().state, JobState::Succeeded);
    // Dropped while the job runs: it is asked to stop, and waited for.
    let asked = Instant::now();
    drop(jobs);
    assert!(asked.elapsed() < Duration::from_secs(1));
    assert_eq!(h.get(&id).unwrap().unwrap().state, JobState::Cancelled);
}
