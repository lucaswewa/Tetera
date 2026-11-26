# ADR-0002: Clean room and dependency licences

- Status: Accepted
- Date: 2026-11-26
- Phase: P1.01

## Context

- **OScope already built most of what Tetera needs,** under MIT, by the same owner. The owner chose a clean room over porting it (plan D1, Q1): Tetera's code is written fresh, and OScope's ideas, lessons and look are reused in Tetera's own words.
- **Tetera is MIT,** so every dependency's licence must allow distributing it under MIT.
- **New package versions are a supply-chain risk** in their first days.

## Decision

1. **All code is written fresh.** Nothing is copied from OScope, `microscope` or any other codebase, nor ported line by line.
2. **OScope's documents may be read and their ideas reused:** its plan, ADRs, notes, retrospectives, `look.md` and `hardware.md`. A phase's notes name the ideas it took. **Its source code isn't opened** while Tetera's code is written, until the owner answers plan Q1b otherwise.
3. **Facts are not code:** the camera's measured modes and rates, and the look's colour values, may be carried over.
4. **Algorithms come from published methods,** cited in the phase's notes.
5. **Tucsen's images are never committed,** and the Tucsen SDK is never redistributed unless its licence allows it.
6. **Dependencies:**
   - **licences:** MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unicode-3.0, CC0-1.0 and the Unlicense. Anything else, MPL-2.0 included, goes to the owner and becomes a named exception in `deny.toml`;
   - **age:** only versions at least two weeks old, checked against crates.io's release dates. For npm, from M4: `npm install --before=<date two weeks ago>`;
   - **sources:** crates.io, and teta-wot's repository;
   - **a new class of dependency** goes to the owner first: a C library, a CUDA library, a model's weights, a Python package with native code;
   - `cargo deny check` enforces the licences, the sources, the advisories and the bans in every phase gate.

## Consequences

- **Gains:** no question of where Tetera's code came from; every licence is checked on every gate.
- **Costs:** OScope's working code can't be reused directly, so each phase writes its own; each dependency's age is checked by hand.
- **To watch:** advisories against pinned versions, which `cargo deny` reports.

## Alternatives considered

- **Porting OScope** (the plan's first draft): faster, but the owner chose the clean room.
- **Allowing weak copyleft (MPL-2.0) freely:** it complicates redistribution; named exceptions keep it visible.
- **Newest versions always:** less lag, more exposure to a bad release.
