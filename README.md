# Tetera

Tetera is software for scientific lab instruments. Its first instrument is a motorised brightfield microscope: a Tucsen MIChrome 20 camera, an XYZ stage driven by grblHAL, and a set of objectives.

- **The backend** is a layered Rust service on [`teta-wot`](https://github.com/lucaswewa/teta-wot) v0.2.0. It runs in WSL2's Ubuntu 24.04 on the instrument's Windows PC, with CUDA for image processing and the CPU as its fallback.
- **The frontend** is a Vue 3 app in the browser, from Milestone 4.
- **Without the instrument,** a virtual camera and stage image a simulated H&E slide, generated on the GPU.

Tetera is being built in five milestones; this is Milestone 1, the backend's core. The [implementation plan](docs/implementation-plan.md) has the architecture and every phase, and [docs/README.md](docs/README.md) indexes the rest.

## Getting started

Set up the toolchain with [docs/setup/toolchain.md](docs/setup/toolchain.md), then, in WSL:

```bash
cargo xtask doctor          # check the tools, the disk and the memory
cargo xtask gate quick      # the quick local gate: formatting, layers, lints, unit tests
cargo run -p tetera-core --example hello
```

`cargo xtask help` lists the gates. [CLAUDE.md](CLAUDE.md) has the working rules.

## Licence

[MIT](LICENSE).
