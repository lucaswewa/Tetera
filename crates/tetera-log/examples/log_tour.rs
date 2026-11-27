//! A tour of tetera-log: a record at every level, to the ring, to a rotating JSON file and,
//! under systemd or with `--journald`, to journald.
//!
//! Run it with `cargo run -p tetera-log --example log_tour`. Options:
//!
//! - `--journald`: send to journald even outside systemd; then try
//!   `journalctl -t tetera-log-tour -o verbose -n 6`;
//! - `--dir <folder>`: where the log file goes (a temporary folder otherwise, removed after
//!   a smoke run);
//! - `--smoke`: check the ring holds the six records, and say so.

use std::path::PathBuf;
use tetera_log::{FileConfig, LogConfig, Toggle};

fn main() -> Result<(), tetera_core::Error> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|arg| arg == name);
    let chosen_dir = args
        .iter()
        .position(|arg| arg == "--dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let dir = chosen_dir.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("tetera-log-tour-{}", std::process::id()))
    });

    let config = LogConfig {
        filter: "trace".into(),
        identifier: "tetera-log-tour".into(),
        journald: if flag("--journald") {
            Toggle::On
        } else {
            Toggle::Auto
        },
        file: Some(FileConfig {
            dir: dir.clone(),
            ..FileConfig::default()
        }),
        ..LogConfig::default()
    };
    let logs = tetera_log::init(&config)?;

    tracing::trace!("tracing a problem");
    tracing::debug!(exposure_ms = 12.5, "detail for developers");
    tracing::info!(objective = "10x", "the instrument is ready");
    tracing::warn!(frame = 41, "a frame came late");
    tracing::error!(device = "stage", "a move failed");
    tetera_log::critical!(free_gb = 0, "the scratch disk is full");
    tetera_log::warn_on_drift(&tetera_core::SystemClock::new());

    println!("\nThe ring holds {} records:", logs.ring().len());
    for entry in logs.ring().snapshot() {
        println!(
            "  #{} {:<8} {}",
            entry.seq, entry.record.level, entry.record.message
        );
    }
    if let Some(path) = logs.file() {
        println!("The file: {}", path.display());
    }
    if logs.journald() {
        println!("journald: on; try `journalctl -t tetera-log-tour -o verbose -n 6`");
    } else {
        println!("journald: off (pass --journald, or run under systemd)");
    }

    if flag("--smoke") {
        assert_eq!(logs.ring().len(), 6, "a record at each of the six levels");
        if chosen_dir.is_none() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        println!("\nsmoke: ok");
    }
    Ok(())
}
