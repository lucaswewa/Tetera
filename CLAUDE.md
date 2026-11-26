# Working on Tetera

Tetera is software for scientific lab instruments; its first is a motorised brightfield microscope (a Tucsen MIChrome 20, a grblHAL XYZ stage, objectives). Its backend is a layered Rust service on `teta-wot` v0.2.0, running in WSL2's Ubuntu 24.04; its frontend, from M4, is a Vue 3 app in the browser.

**Where to start:** [docs/README.md](docs/README.md). The plan is [docs/implementation-plan.md](docs/implementation-plan.md); Milestone 1's phase specs are [docs/m1/phases.md](docs/m1/phases.md); decisions are the [ADRs](docs/adr/README.md).

## Delivering a phase

- **In the plan's order,** one at a time, each from an up-to-date base after the previous one is merged.
- **One branch and one pull request a phase:** branch `m1/p1.07-short-name`, title `P1.07: <title>`. Upstream teta-wot phases: `u/u01-short-name` in `../teta-wot`.
- **The owner approves every pull request, and merges it.** Never push to `main`, never make a tag, and never push or open a pull request without the owner's word.
- **Before the code:** the second estimate, in five parts (code, tests, examples, docs-as-code, glue; doc comments and blank lines count), goes into the plan's change log in a commit of its own. Over 2,000 lines, ask the owner whether to split.
- **Spec-first phases** open their notes with the behaviour and its literature before any code.
- **Each pull request holds:** the code and its tests; examples with a README and a `--smoke` mode; the notes from [the template](docs/m1/notes/template.md); new ADRs, *Accepted*, with their rows in the index; the manual-test entry, run and recorded; the phase's status in [the notes' index](docs/m1/notes/README.md); the phase gate's summary in its description. The next phase's pull request marks the previous one *Done*.

## Rules

- **Clean room** ([ADR-0002](docs/adr/0002-clean-room-and-dependency-licences.md)): all code written fresh. OScope's documents may be read and their ideas reused, named in the notes; its source isn't opened while writing Tetera's code. Algorithms from published methods, cited. No Tucsen images or SDK in the repository.
- **Dependencies:** licences in `deny.toml`'s list; versions at least two weeks old (check crates.io's dates); a new class of dependency goes to the owner first.
- **Unsafe code** only in `tetera-compute`, `tetera-shm`, `tetera-ext`, `tetera-ext-sdk` and `tetera-tucsen`, each block with a `SAFETY:` comment.
- **Layers** ([ADR-0005](docs/adr/0005-layered-crates-and-the-local-gates.md)): a new crate, or an edge within a layer, changes the table in `xtask/src/layering.rs` in the pull request that needs it.
- **Platform:** Linux in WSL2 only ([ADR-0003](docs/adr/0003-the-runtime-platform-ubuntu-24-04-in-wsl2.md)). Work in `~/repos/Tetera`, never from `/mnt/`. LF line endings.
- **Tests** wait for a condition, never a fixed time; each gets its own data root, scratch folder and a port the system picks. Switches for what isn't always there: `TETERA_CUDA=1`, `TETERA_PYTHON=1`, `TETERA_GRBL_PORT`, `TETERA_TUCSEN=1`, `TETERA_VISUAL=1`. Anything that detects, supervises or recovers is broken on purpose once, and the notes say so.
- **Ports:** the service 8310; manual checks 8311–8319; Vite (M4) 8380. Stop only processes you started, by their ID.

## Gates

All local, in WSL; GitHub runs nothing ([ADR-0005](docs/adr/0005-layered-crates-and-the-local-gates.md)):

```bash
cargo xtask gate quick         # G0, before each commit
cargo xtask gate phase         # G1, before the pull request: what the change reaches, with stress runs
cargo xtask gate checkpoint    # G2, at a checkpoint: everything, on a fresh clone (add --mutants)
cargo xtask gate milestone     # G3, at a milestone's end
```

- Each run writes `target/gates/<time>-<level>/summary.md`; paste the phase gate's into the pull request.
- Add `--cuda` when a change reaches CUDA code without a `cuda` feature saying so.
- Mutants: write the list for what the phase detects or supervises, read the tests against it, then run `cargo mutants -p <crate> -f <file>`; the notes record what was missed.
- `cargo xtask doctor` checks the tools, the disk (40 GB free for G2) and the memory.
