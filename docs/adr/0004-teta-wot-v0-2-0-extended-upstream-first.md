# ADR-0004: teta-wot v0.2.0, extended upstream first

- Status: Accepted
- Date: 2026-11-26
- Phase: P1.01

## Context

- **The spec names `teta-wot` v0.2.0 as Tetera's HTTP interface.** It serves Rust structs as Web of Things *Things*, with Thing Descriptions, OpenAPI, observation, binary WebSocket streams, router composition and pluggable security.
- **It isn't on crates.io:** applications depend on a tag of its repository.
- **Its toolchain is pinned to Rust 1.98.1,** its minimum.
- **If its types spread through Tetera,** every upgrade touches every crate.

## Decision

1. **Tetera depends on the `v0.2.0` tag** of `https://github.com/lucaswewa/teta-wot`. `deny.toml` allows that repository as a source.
2. **Only `tetera-api` depends on `teta-wot`.** The layering check fails any other crate that does (ADR-0005).
3. **Tetera's toolchain is teta-wot's,** Rust 1.98.1, in `rust-toolchain.toml`.
4. **A gap is fixed upstream first,** as a U-phase in the owner's `../teta-wot` clone, under the same approval rules, released as v0.3.0. If the upstream work stalls, `tetera-api` may mount a local route meanwhile, recorded in an ADR.

## Consequences

- **Gains:** one crate to change on an upgrade; the toolchain is one that teta-wot is tested with.
- **Costs:** a Tetera phase can wait on an upstream release; Rust's newer features wait for teta-wot's next toolchain.
- **To watch:** teta-wot's own advisories, such as its dependency on `chrono`, which OScope's notes flagged as unmaintained.

## Alternatives considered

- **Axum directly:** no Thing Descriptions or OpenAPI for free, and the spec names teta-wot.
- **A vendored copy:** upgrades become merges.
- **A newer toolchain than teta-wot's:** possible, but untested with it.
