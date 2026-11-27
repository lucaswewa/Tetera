# ADR-0007: Logging

- Status: Accepted
- Date: 2026-11-26
- Phase: P1.02

## Context

- **The spec's Logging tab** lists info, warn, error and critical. `tracing`, Rust's usual logging framework, has trace to error, and no critical.
- **Records must reach several places:** memory, for the `logs` Thing and its live tail (P1.20); files that survive a restart; journald, where a service under systemd is expected to log; and a terminal during development.
- **The instrument runs for weeks.** Its log files must not fill the disk, so they must be bounded by size.
- **journald's priorities** run from 0 (EMERG) to 7 (DEBUG); a critical record belongs at 2 (CRIT), where `journalctl -p crit` finds it.
- **The spec named `tracing-appender` and `tracing-journald`,** but P1.02 found that the first rotates only by time, and the second maps priority by level alone. So a critical record, an ERROR with a field, can't reach CRIT that way. The owner chose writers of Tetera's own (2026-11-26).

## Decision

1. **`tetera-core` holds the records:**
   - `LogLevel`, from trace to **critical**;
   - `LogRecord`, with its time, level, target, message and fields as text;
   - `LogRing`, the newest records up to a capacity. A full ring pushes out its oldest and counts it. Each record has a sequence number, so a tail asks only for what it hasn't seen. A panic elsewhere doesn't poison it.
2. **A critical record is an error with `critical = true`.** `tetera_log::critical!` writes one; every sink treats it as `Critical`, whatever its `tracing` level.
3. **`tetera-log` (L1)** sets up `tracing`. One `RecordLayer` makes each event into one `LogRecord` and hands it to every sink:
   - **the ring**;
   - **JSON lines in a file rotated by size:** before passing `max_bytes`, the file becomes `.1`, `.1` becomes `.2`, and so on, and the oldest beyond `keep` goes;
   - **journald, over its native protocol,** one datagram a record. Its priority comes from the level, critical being 2, with `SYSLOG_IDENTIFIER`, `TETERA_LEVEL`, `TETERA_TARGET` and the record's fields as journald fields.
4. **What is on:**
   - **journald:** on `Auto`, when systemd started the process (`JOURNAL_STREAM` or `INVOCATION_ID`), skipped quietly if its socket is missing. On `On`, the socket must connect;
   - **standard error,** as readable lines: on `Auto`, only when journald is off, since systemd already sends a service's standard error to journald;
   - **the filter,** in `tracing`'s syntax; `RUST_LOG` overrides it when set.
5. **A sink that fails keeps quiet:** a full disk or a missing journal never stops the instrument.
6. **Drift:** `warn_on_drift` logs a warning when the wall clock has drifted beyond a second from Tetera's clock (ADR-0006).

## Consequences

- **Gains:** the log's disk use is bounded; `journalctl -p crit` finds critical records; one record format everywhere, so the ring, the file and the API agree; no `tracing-appender` or `tracing-journald` dependency.
- **Costs:** about 250 lines of Tetera's own for the file and journald, with their tests. A record larger than one datagram, about 200 kB, doesn't reach journald: its memfd path isn't implemented. On standard error, a critical record shows as `ERROR` with `critical=true`.
- **To watch:** the cost of writing synchronously on busy paths. A background writer can come in P5.06 if the measurements ask for it.

## Alternatives considered

- **`tracing-appender` and `tracing-journald`,** as the spec first said: time-based rotation, and critical records at ERROR's priority.
- **A custom `tracing` level:** `tracing`'s levels are fixed.
- **`log` and `env_logger`:** no structured fields, and the ecosystem Tetera builds on uses `tracing`.
- **Writing every record to journald only:** no files to read on a machine without systemd, and no ring for the API without reading the journal back.
