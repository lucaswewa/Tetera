# tetera-jobs

Tetera's job service, L3 (plan §6.2, D15, [ADR-0020](../../docs/adr/0020-the-job-service.md)): long work that can be followed, cancelled, and looked up afterwards.

- **A `Job`** has an id (a UUID v7, so ids sort by age), a kind, a name, a state, progress and a message, its latest log lines, and its result or its error.
- **`JobService::submit`** queues work, a closure given a `JobContext`, as one of two kinds:
  - **hardware:** one at a time, in the order they came, as they drive the instrument;
  - **compute:** side by side on a pool of workers, each taking the VRAM it asked for from the compute jobs' budget (P1.04b), and waiting while others hold it.
- **The work reports through its `JobContext`:** `progress`, `log`, and `check` or `sleep`, which return the cancel error once the job is asked to stop.
- **`cancel`:** a queued job ends at once and never runs; a running one is asked to stop, and ends cancelled when it next checks.
- **`get`, `active`, `recent`, `wait`,** and `subscribe` for every change to every job.
- **The history,** in SQLite: each job as it last stood, kept across restarts.
  - A job queued or running when the service last stopped is marked failed, interrupted, when it opens again.
  - Jobs that ended longer ago than `keep` are pruned when it opens, and as jobs end.
- **Stopping** cancels what is queued, asks what runs to stop, and waits for it.

## How to try it

```bash
cargo run --release -p tetera-jobs --example jobs
```

It queues four hardware jobs and four compute jobs, two compute workers sharing 512 MiB of VRAM, and cancels one hardware job while it waits and one compute job while it runs. Then it stops the service, opens it again, and reads the history back: each job's start and end, the hardware jobs never overlapping, the compute jobs two at a time. Last, it opens the history keeping jobs 1 ms, and they are pruned. `-- --smoke` runs it briefly.
