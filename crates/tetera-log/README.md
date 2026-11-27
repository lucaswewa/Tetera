# tetera-log

Tetera's logging, on `tracing` ([ADR-0007](../../docs/adr/0007-logging.md)). Every event becomes one `LogRecord`, handed to:

- **the ring:** the newest records, which the `logs` Thing will serve (P1.20);
- **JSON lines in files rotated by size,** when configured. The disk they use is bounded at about `max_bytes × (keep + 1)`;
- **journald,** over its native protocol, when systemd started the process (or `journald = "on"`). A critical record is priority 2 (CRIT);
- **standard error,** as readable lines, unless journald already has them.

`tracing` has no critical level: `tetera_log::critical!` logs an error with `critical = true`, and every sink treats it as critical. `warn_on_drift` logs a warning when the wall clock has drifted more than a second from Tetera's clock.

```rust
let logs = tetera_log::init(&tetera_log::LogConfig::default())?;
tracing::info!(objective = "10x", "ready");
tetera_log::critical!("the scratch disk is full");
```

`RUST_LOG` overrides the configured filter.

## How to try it

```bash
cargo run -p tetera-log --example log_tour
cargo run -p tetera-log --example log_tour -- --journald
journalctl -t tetera-log-tour -o verbose -n 6
```

- **The first run** logs one record at each level, from trace to critical. It prints the ring's six records, and the path of the JSON file they also went to.
- **With `--journald`,** the records go to journald instead of standard error. `journalctl` then shows them with their priorities, 7 down to 2 for the critical one, and their fields (`FREE_GB`, `FRAME`, …).
- **`--dir <folder>`** keeps the log file in a folder of your choice.
- **`--smoke`** checks the ring holds the six records, and says so.
