//! The job service (P1.15a, ADR-0020): hardware jobs queued one at a time, compute jobs side
//! by side within a VRAM budget, one of each cancelled, then the service stopped and opened
//! again, and the history read back.
//!
//! Run it with `cargo run --release -p tetera-jobs --example jobs`; `-- --smoke` runs it
//! briefly.

use serde_json::json;
use std::time::{Duration, Instant};
use tetera_compute::budget::VramBudgets;
use tetera_core::Result;
use tetera_jobs::{Job, JobKind, JobService, JobState, JobsConfig};

const MIB: u64 = 1 << 20;

fn main() -> Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let step = Duration::from_millis(if smoke { 20 } else { 100 });
    let dir = std::env::temp_dir().join(format!("tetera-jobs-example-{}", std::process::id()));
    std::fs::create_dir_all(&dir)
        .map_err(|e| tetera_core::Error::internal(format!("{}: {e}", dir.display())))?;
    let db = dir.join("tetera.db");
    let budgets = VramBudgets::new(1_024 * MIB);
    let started = Instant::now();
    let at = move || started.elapsed().as_secs_f64();

    let (ids, seen) = {
        let jobs = JobService::open(
            &db,
            JobsConfig::default(),
            Some(budgets.reserve("compute jobs", 512 * MIB)?),
        )?;
        let changes = jobs.subscribe();
        println!("Two compute workers, 512 MiB of VRAM between them.\n");
        let mut ids = Vec::new();
        for (n, name) in ["move", "capture", "move", "capture"].iter().enumerate() {
            ids.push(jobs.submit(JobKind::Hardware, name, 0, move |ctx| {
                for k in 1..=4 {
                    ctx.sleep(step)?;
                    ctx.progress(f64::from(k) / 4.0, format!("{name} {n}: step {k} of 4"));
                }
                Ok(json!({"hardware": n}))
            })?);
        }
        for (n, mib) in [(0, 200), (1, 200), (2, 200), (3, 300)] {
            ids.push(
                jobs.submit(JobKind::Compute, "denoise", mib * MIB, move |ctx| {
                    ctx.log(format!("{mib} MiB granted"));
                    ctx.sleep(step * 3)?;
                    Ok(json!({"compute": n}))
                })?,
            );
        }
        // Cancel the third hardware job while it waits, and the last compute job once it runs.
        let queued = jobs.cancel(&ids[2])?;
        println!(
            "cancelled while queued: {} ({})",
            queued.name,
            queued.state.name()
        );
        while jobs.get(&ids[7])?.state == JobState::Queued {
            std::thread::sleep(Duration::from_millis(2));
        }
        let asked = Instant::now();
        jobs.cancel(&ids[7])?;
        let stopped = jobs.wait(&ids[7], Duration::from_secs(10))?;
        println!(
            "cancelled while running: {} ({}, {:.0} ms after asking)\n",
            stopped.name,
            stopped.state.name(),
            asked.elapsed().as_secs_f64() * 1e3
        );
        for id in &ids {
            jobs.wait(id, Duration::from_secs(30))?;
        }
        let seen: Vec<(f64, Job)> = changes.try_iter().map(|j| (at(), j)).collect();
        jobs.stop();
        (ids, seen)
    };

    // Each job's run, from its events: when it started and ended.
    println!("Each job, as it ran (s from the start):");
    let jobs = JobService::open(&db, JobsConfig::default(), None)?;
    for id in &ids {
        let j = jobs.get(id)?;
        let s = |t: Option<tetera_core::UtcTime>, from: tetera_core::UtcTime| {
            t.map_or("—".to_string(), |t| {
                format!("{:.2}", t.nanos_since(from) as f64 / 1e9)
            })
        };
        println!(
            "  {:8} {:7} {:9} started {:>5}, ended {:>5}",
            format!("{:?}", j.kind).to_lowercase(),
            j.name,
            j.state.name(),
            s(j.started, j.created),
            s(j.finished, j.created)
        );
    }
    let hardware: Vec<Job> = ids[..4]
        .iter()
        .map(|id| jobs.get(id))
        .collect::<Result<_>>()?;
    let ran: Vec<&Job> = hardware.iter().filter(|j| j.started.is_some()).collect();
    let overlap = ran.windows(2).any(|w| w[1].started < w[0].finished);
    println!(
        "\nThe hardware jobs overlapped: {}. {} changes were heard.",
        if overlap { "yes" } else { "never" },
        seen.len()
    );
    println!(
        "After a restart, the history holds {} jobs; the last: {} ({}).",
        jobs.recent(100)?.len(),
        jobs.recent(1)?[0].name,
        jobs.recent(1)?[0].state.name()
    );
    drop(jobs);
    // Kept a moment: the history prunes the ended jobs when it opens.
    let brief = JobsConfig {
        keep: Duration::from_millis(1),
        ..JobsConfig::default()
    };
    std::thread::sleep(Duration::from_millis(5));
    let jobs = JobService::open(&db, brief, None)?;
    println!(
        "Opened again, keeping jobs 1 ms: {} left.",
        jobs.recent(100)?.len()
    );
    drop(jobs);
    let _ = std::fs::remove_dir_all(&dir);
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}
