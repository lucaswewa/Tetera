# tetera-server

Tetera's server, L6 (plan §6.2): the binary that serves the instrument over HTTP, through `tetera-api`.

- **Its configuration** is `configs/tetera.toml`, or the file `--config` names; without either, the defaults. Every key has a default, so a file names only what it changes, and a key it doesn't know is refused with its name. The tables:
  - `[server]`: the address and port, `127.0.0.1:8310`;
  - `[data]`: the data root, `./.tetera/` in development, and the scratch root;
  - `[simulator]`: the sample, the clock's rate, the GPU, the camera's sensor, the light, where the stage starts;
  - `[objectives]`: the catalogue, and the objective fitted;
  - `[log]`: `tetera-log`'s filter, standard error, journald and files.

  Relative paths are from the file's folder, but the data root is from where the server runs, and log files go in the data root.
- **Its command line** overrides the file: `--address`, `--port`, `--data-root`. `--check` prints the configuration it would use, and exits.
- **It stops gracefully** on Ctrl-C or SIGTERM: unfinished invocations are cancelled, the Things are stopped, within 5 s.
- **As a service,** it runs under systemd in WSL: [`packaging/linux`](../../packaging/linux/README.md). A browser on Windows reaches it at <http://localhost:8310>: [`packaging/windows`](../../packaging/windows/README.md) ([ADR-0018](../../docs/adr/0018-the-service-in-wsl2-systemd-start-up-networking-and-devices.md)).

## How to try it

```bash
cargo run --release -p tetera-server
```

Then open <http://localhost:8310/docs> on Windows, or `curl` as `tetera-api`'s README shows. It logs `serving url=http://127.0.0.1:8310 backend=cuda sensor=[5472, 3648]` once it is ready.

```bash
cargo run -p tetera-server -- --check                  # the configuration it would use
cargo run -p tetera-server -- --port 9000 --data-root /tmp/tetera
```

It builds with the `cuda` feature by default, which needs no CUDA toolkit: on a machine without a GPU, or with `[simulator] gpu = false`, the camera renders on the CPU, on a 912 × 608 sensor.

VS Code's **Debug the server** configuration runs it under the debugger.
