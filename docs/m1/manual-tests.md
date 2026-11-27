# Milestone 1: manual tests

How to try each phase by hand, and each checkpoint's demo. The agent runs each phase's entry and records the result in the phase's notes; the owner runs each checkpoint's (ADR-0001).

Every entry runs in WSL, in `~/repos/Tetera`, on the phase's branch.

## P1.01: Workspace and gates

- **Prerequisites:** the toolchain of [docs/setup/toolchain.md](../setup/toolchain.md); a fresh clone of the phase's branch.
- [ ] `cargo xtask doctor` — expect every line needed now to say `ok`, and the last line `Everything needed now is here.`
- [ ] `cargo xtask layering` — expect `The layer rules hold.`
- [ ] `cargo xtask gate quick` — expect five steps to pass, within 90 seconds once built, and a `summary:` line naming `target/gates/<time>-quick/summary.md`.
- [ ] `cargo xtask gate phase` — expect `fmt`, `layering`, `clippy`, `tests`, `smoke tetera-core/hello`, `docs`, `deny` and `stress` to pass, and `cuda` and `python` to be skipped with their reasons.
- [ ] Open the phase gate's `summary.md` — expect what the change reaches and why, and each step with its result, time and reason.
- [ ] Add `tetera-hal`, a crate with `[lints] workspace = true`, and make `tetera-core` depend on it; run `cargo xtask gate quick` — expect `layering` to fail with ``tetera-core` (L0) depends on `tetera-hal` (L2), a higher layer.`. Undo it.
- [ ] Remove `[lints]` and `workspace = true` from `crates/tetera-core/Cargo.toml`; run `cargo xtask layering` — expect ``tetera-core` must take the workspace's lints…`. Undo it.
- [ ] Commit, then `cargo xtask gate checkpoint` — expect one `fresh clone` step to pass, naming the clone's own summary.
- [ ] In VS Code over Remote-WSL, set a breakpoint in `crates/tetera-core/examples/hello.rs`, choose **Debug an example**, accept `tetera-core` and `hello`, and press F5 — expect it to stop at the breakpoint.
- **Result:** see the [notes](notes/P1.01-workspace-and-gates.md#manual-test).

## P1.02: Core types and logging

- **Prerequisites:** the phase's branch, built.
- [ ] `cargo run -p tetera-core --example core_tour` — expect:
  - 25,000 µm of travel as 10,417 sensor pixels;
  - pixel (5472, 3648) at about (1261.118, 2177.389) µm, and back to (5472, 3648) px;
  - the ROI clipped to 472 × 648;
  - a 5000.0 µm move;
  - the clock's UTC time, today, and its drift within the limit;
  - two IDs in order;
  - `capturing a burst: setting the exposure: the camera is streaming (kind busy)`.
- [ ] `cargo run -p tetera-log --example log_tour` — expect six coloured lines on standard error, then the ring's six records from `trace` to `critical`, and the JSON file's path. That file holds six JSON lines, the last with `"level":"critical"`.
- [ ] `cargo run -p tetera-log --example log_tour -- --journald`, then `journalctl -t tetera-log-tour -o json -n 6` — expect priorities 7, 7, 6, 4, 3 and **2** for the critical record, with `FREE_GB=0` and the other fields. Nothing on standard error, since journald has the records.
- [ ] `RUST_LOG=warn cargo run -p tetera-log --example log_tour` — expect only the warn, error and critical records in the ring (the smoke check would fail with this filter, so don't pass `--smoke`).
- **Result:** see the [notes](notes/P1.02-core-types-and-logging.md#manual-test).
