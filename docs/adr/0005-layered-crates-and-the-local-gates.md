# ADR-0005: Layered crates, and the local gates

- Status: Accepted
- Date: 2026-11-26
- Phase: P1.01

## Context

- **The plan layers the backend** (§6.2): algorithms testable without devices, devices without a server, the web framework in one crate. Cargo refuses only cycles, so without a check a convenient upward dependency creeps in.
- **The spec asks for gates run locally, not on GitHub,** in smaller pieces for each phase and in full at each milestone's end. OScope's full replays took about an hour a phase until they ran only what a change reached.
- **Races hide on a machine with many threads.** OScope's runs of the touched suites on two CPUs each found faults the full machine never showed.
- **This PC has 16 GB of RAM,** and OScope's builds filled a disk and ran out of memory linking many examples at once.

## Decision

**1. The layers.** `xtask/src/layering.rs` holds the table of plan §6.2, with every planned crate placed before it exists, and the few edges allowed within a layer. `cargo xtask layering` checks every member through `cargo metadata`, and an integration test runs it with `cargo test`:

- a `tetera-*` crate depends only on lower layers, or on an allowed crate in its own layer, for every kind of dependency;
- it depends on no tool or example crate;
- only `tetera-api` depends on `teta-wot` (ADR-0004);
- every crate takes the workspace's lints, which forbid unsafe code. The crates of plan §5.3 set their own instead, with `unsafe_code = "allow"` and clippy's `undocumented_unsafe_blocks = "deny"`.

A new crate, or a new edge within a layer, changes the table in the pull request that needs it. Moving a crate to another layer needs an ADR.

**2. The gates.** `cargo xtask gate <level>`, in WSL:

| Gate | When | Runs |
|---|---|---|
| G0 `quick` | Before each commit | fmt, layering, clippy and unit tests of the **changed** packages |
| G1 `phase` | Before a pull request | Everything the change **reaches**: the changed packages and every package that depends on them. Clippy, all tests, each example's `--smoke` run, the docs, `cargo deny`, and the **stress** step; the CUDA tests when a reached package has a `cuda` feature |
| G2 `checkpoint` | At a checkpoint | Everything, on a **fresh clone** of the commit; `cargo mutants` on the change with `--mutants` |
| G3 `milestone` | At a milestone's end | G2, the licence report and a release build; the performance budgets and the soak from P1.21 |

- **Reach:** the files changed since the merge base with `--base` (default `main`), committed or not, and untracked files. The workspace's manifest and lock file, the toolchain, `rustfmt.toml`, `deny.toml`, `.cargo/` and `xtask/` reach everything. Docs reach no package.
- **Stress:** the reached test binaries, 3 times each, side by side, each run pinned to two CPUs with `taskset`.
- **Fresh clone:** `target/gate-clone/src`, fetched and cleaned for each run, with its own target folder kept between runs so it starts warm. Uncommitted changes aren't included, and the summary says so.
- **Summary:** `target/gates/<time>-<level>/summary.md`, with what the change reaches and why, and each step with its result, time and reason, and each step's log beside it. A pull request carries its phase gate's summary.
- **GitHub runs no workflow.**

**3. The build,** for quick gates on this PC: mold as the linker through clang; dependencies optimised (`opt-level = 2`) in dev and test builds; few example binaries; `cargo xtask doctor` checks the tools, the disk and the memory, and suggests `CARGO_BUILD_JOBS`.

## Consequences

- **Gains:** a phase's gate costs what its change reaches: G0 runs in seconds, G1 in minutes. The layers are checked on every gate, not only in review. Each run leaves a summary to paste and logs to read.
- **Costs:** a change can break something it doesn't reach, such as a test in a crate that uses another's output without depending on it; the checkpoint's full run finds it. Every new crate edits the layer table.
- **To watch:** the gates' times as the workspace grows; whether 3 stress repeats are enough; the clone's target folder on disk.

## Alternatives considered

- **GitHub Actions as the gate:** the spec rules it out, and OScope's stopped for billing.
- **Shell scripts:** quicker to write, but untested, and OScope's wrote CRLF and mangled escapes on Windows.
- **cargo-nextest for the stress runs:** another tool to install, for what `taskset` and the test binaries do.
- **Finer layers instead of allowed edges:** more layers than the plan has, and a crate would change layer whenever it gained an edge.
- **Everything on every gate:** simple, but OScope's hour-long gates were why it stopped.
