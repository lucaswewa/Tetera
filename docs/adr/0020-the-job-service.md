# ADR-0020: The job service

- Status: Accepted
- Date: 2026-11-28
- Phase: P1.15a

## Context

- **Captures, autofocus, scans and batches run for seconds to hours** (P1.15b to P1.19). A client must follow them, cancel them, and look them up afterwards, also after a restart (D15).
- **The instrument does one thing at a time:** two jobs that both move the stage or switch the camera's mode must not interleave.
- **Compute jobs share the GPU** with the simulator and the preview, through P1.04b's VRAM budgets: each user reserves a budget, and an allocation beyond it is refused (ADR-0010).
- **Rust can't stop a thread from outside.** Work stops when it decides to.
- **The plan keeps one SQLite file,** `tetera.db` in the data root, for the jobs' history, the datasets' index and the calibration sets (plan §6.9).

## Decision

1. **Two kinds of job.**
   - **Hardware jobs run one at a time,** in the order they came, on one worker.
   - **Compute jobs run on a pool** of workers, two by default. Each asks for the VRAM it needs, and takes it from the compute jobs' own budget before it runs. When the budget is held, it waits. A job asking more than the whole budget fails.
2. **A job is a closure given a context.** It reports progress, a message and log lines through the context, and returns its result as JSON or an error.
3. **Cancel is a token the work checks,** through the context's `check`, or its `sleep`, which wakes for it.
   - A queued job ends at once and never runs.
   - A running job ends cancelled when its work returns an error after it was asked.
   - Work that ends well despite the ask has succeeded.
   - Work is expected to check every 100 ms or so, so a cancel takes effect within a second.
4. **Every change to a job is an event** to subscribers; the `jobs` Thing turns each into its `changed` event, and keeps `active` current.
5. **The history is SQLite,** through `rusqlite` with SQLite built in: a table of jobs, each as JSON beside its state and times.
   - A job is written when it is queued, when it starts, when its cancel is asked, and when it ends; progress stays in memory.
   - The jobs' lock is held while a job is written, so a write can't fall behind the job's end and overwrite it.
   - A job still queued or running when the service last stopped is marked failed, interrupted, when it opens again.
   - Ended jobs older than `keep`, 30 days by default, are pruned when it opens and as jobs end.
6. **The server owns one service,** its history in the data root's `tetera.db`, its options in `[jobs]`. The instrument shares out the GPU's memory, less a quarter for the driver and the desktop. The camera reserves its budget, and the compute jobs reserve theirs, 1 GiB by default.

## Consequences

- **Gains:**
  - two hardware jobs can't fight over the stage or the camera;
  - compute jobs can't starve the camera or the preview of VRAM;
  - a job's end, its result or its error, outlives a restart.
- **Costs:**
  - work that never checks its token can't be cancelled;
  - a job's progress isn't in the history until it ends;
  - the hardware queue is first come, first served: a long scan holds up a quick capture behind it.
- **To watch:**
  - whether hardware jobs need priorities, or a capture that can cut in, when the scan engine comes (P1.17);
  - the history's size, at one row a job: pruning keeps it to 30 days.

## Alternatives considered

- **Tokio tasks for jobs:** cancelling by dropping the future would stop work at any await, mid-move, and most work here blocks anyway, on devices and on the GPU.
- **A process for each job:** isolated, but heavy, and the work needs the instrument in memory.
- **A history in JSON files:** no transactions, and the plan already keeps `tetera.db` for the datasets' index.
- **One pool for both kinds, with a hardware lock:** simpler, but a hardware job waiting for the lock would hold a worker that a compute job could use.
