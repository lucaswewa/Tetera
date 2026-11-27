//! Tetera's logging (ADR-0007), on `tracing`. One [`RecordLayer`] turns every event into a
//! [`LogRecord`](tetera_core::LogRecord) and hands it to:
//!
//! - the [`LogRing`], which keeps the newest records for the `logs` Thing (P1.20);
//! - JSON lines in [rotating files](mod@file), when configured;
//! - [journald](mod@journald), when systemd started the process, with a critical record at
//!   priority 2 (CRIT).
//!
//! Readable lines go to standard error too, unless journald already has them. `tracing`
//! has no critical level: [`critical!`] logs an error with `critical = true`, and every
//! sink treats it as [`LogLevel::Critical`](tetera_core::LogLevel::Critical).
//!
//! ```no_run
//! let logs = tetera_log::init(&tetera_log::LogConfig::default())?;
//! tracing::info!(objective = "10x", "ready");
//! tetera_log::critical!("the scratch disk is full");
//! assert_eq!(logs.ring().len(), 2);
//! # Ok::<(), tetera_core::Error>(())
//! ```

pub mod config;
pub mod file;
pub mod journald;
pub mod layer;

pub use config::{FileConfig, LogConfig, Toggle};
pub use layer::{CRITICAL_FIELD, RecordLayer, RingSink, Sink};

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tetera_core::time::{Clock, drift_exceeds_limit};
use tetera_core::{Error, LogRing, Result};
use tracing::Subscriber;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;

#[doc(hidden)]
pub use tracing as __tracing;

/// Logs a critical record: an error with `critical = true`. It takes what `tracing::error!`
/// takes, a `target:` first included.
#[macro_export]
macro_rules! critical {
    (target: $target:expr, $($rest:tt)+) => {
        $crate::__tracing::error!(target: $target, critical = true, $($rest)+)
    };
    ($($rest:tt)+) => {
        $crate::__tracing::error!(critical = true, $($rest)+)
    };
}

/// What logging reads from the process's environment, kept apart so tests can choose it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Env {
    /// `RUST_LOG`, if set.
    pub rust_log: Option<String>,
    /// Whether systemd started the process.
    pub under_systemd: bool,
    /// journald's socket.
    pub journald_socket: PathBuf,
}

impl Env {
    /// The process's own environment.
    pub fn from_process() -> Self {
        let var = |name| std::env::var(name).ok();
        Self {
            rust_log: var("RUST_LOG"),
            under_systemd: config::under_systemd(
                var("JOURNAL_STREAM").as_deref(),
                var("INVOCATION_ID").as_deref(),
            ),
            journald_socket: PathBuf::from(journald::SOCKET),
        }
    }
}

/// What a logging set-up turned on, and its ring.
#[derive(Debug, Clone)]
pub struct LogHandle {
    ring: Arc<LogRing>,
    stderr: bool,
    journald: bool,
    file: Option<PathBuf>,
}

impl LogHandle {
    /// The ring of the newest records.
    pub fn ring(&self) -> &Arc<LogRing> {
        &self.ring
    }

    /// Whether standard error gets the records.
    pub fn stderr(&self) -> bool {
        self.stderr
    }

    /// Whether journald gets the records.
    pub fn journald(&self) -> bool {
        self.journald
    }

    /// The current log file, if records go to one.
    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }
}

/// Builds the subscriber `config` asks for, in `env`, without installing it.
pub fn build(
    config: &LogConfig,
    env: &Env,
) -> Result<(impl Subscriber + Send + Sync + 'static, LogHandle)> {
    let directives = config::filter_directives(&config.filter, env.rust_log.as_deref());
    let filter = EnvFilter::try_new(directives)
        .map_err(|e| Error::invalid_argument(format!("`{directives}` isn't a log filter: {e}")))?;

    let ring = Arc::new(LogRing::new(config.ring_capacity));
    let mut sinks: Vec<Box<dyn Sink>> = vec![Box::new(RingSink(ring.clone()))];

    let mut file_path = None;
    if let Some(file_config) = &config.file {
        let file = file::RotatingFile::open(file_config).map_err(|e| {
            Error::internal(format!(
                "the log file in {} won't open",
                file_config.dir.display()
            ))
            .with_source(e)
        })?;
        file_path = Some(file.path().to_path_buf());
        sinks.push(Box::new(file::FileSink::new(file)));
    }

    let mut journald_on = false;
    if config::journald_wanted(config.journald, env.under_systemd) {
        match journald::JournaldSink::connect(&env.journald_socket, &config.identifier) {
            Ok(sink) => {
                sinks.push(Box::new(sink));
                journald_on = true;
            }
            Err(e) if config.journald == Toggle::On => {
                return Err(Error::internal(format!(
                    "journald's socket {} won't connect",
                    env.journald_socket.display()
                ))
                .with_source(e));
            }
            // On `Auto`, a journal that isn't there is no reason to fail.
            Err(_) => {}
        }
    }

    let stderr_on = config::stderr_wanted(config.stderr, journald_on);
    let stderr = stderr_on.then(|| tracing_subscriber::fmt::layer().with_writer(std::io::stderr));
    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(RecordLayer::new(sinks))
        .with(stderr);
    let handle = LogHandle {
        ring,
        stderr: stderr_on,
        journald: journald_on,
        file: file_path,
    };
    Ok((subscriber, handle))
}

/// Builds the subscriber `config` asks for and installs it for the whole process.
pub fn init(config: &LogConfig) -> Result<LogHandle> {
    let (subscriber, handle) = build(config, &Env::from_process())?;
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| Error::internal("logging is already set up").with_source(e))?;
    Ok(handle)
}

/// Logs a warning if `clock`'s wall time has drifted beyond its limit from the mapping
/// made at its start, as WSL2's can after the host sleeps. Returns whether it had.
pub fn warn_on_drift(clock: &dyn Clock) -> bool {
    let drift = clock.drift();
    let beyond = drift_exceeds_limit(drift);
    if beyond {
        tracing::warn!(
            drift_ms = drift / 1_000_000,
            "the wall clock has drifted {:.3} s from Tetera's clock; times in records may be off",
            drift as f64 / 1e9
        );
    }
    beyond
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tetera_core::{LogLevel, ManualClock, UtcTime};

    fn env() -> Env {
        Env {
            rust_log: None,
            under_systemd: false,
            journald_socket: PathBuf::from("/nonexistent/journal.sock"),
        }
    }

    fn levels(handle: &LogHandle) -> Vec<LogLevel> {
        handle
            .ring()
            .snapshot()
            .into_iter()
            .map(|e| e.record.level)
            .collect()
    }

    #[test]
    fn the_filter_decides_what_reaches_the_ring() {
        let config = LogConfig {
            filter: "warn".into(),
            stderr: Toggle::Off,
            ..LogConfig::default()
        };
        let (subscriber, handle) = build(&config, &env()).unwrap();
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!("dropped");
            tracing::warn!("kept");
            critical!(target: "tetera::disk", "kept too");
        });
        assert_eq!(levels(&handle), [LogLevel::Warn, LogLevel::Critical]);
        assert_eq!(handle.ring().snapshot()[1].record.target, "tetera::disk");
    }

    #[test]
    fn rust_log_overrides_the_configured_filter() {
        let config = LogConfig {
            filter: "error".into(),
            stderr: Toggle::Off,
            ..LogConfig::default()
        };
        let env = Env {
            rust_log: Some("debug".into()),
            ..env()
        };
        let (subscriber, handle) = build(&config, &env).unwrap();
        tracing::subscriber::with_default(subscriber, || tracing::debug!("seen"));
        assert_eq!(levels(&handle), [LogLevel::Debug]);
    }

    #[test]
    fn a_bad_filter_is_refused() {
        let config = LogConfig {
            filter: "info,=[".into(),
            ..LogConfig::default()
        };
        let err = build(&config, &env()).map(|_| ()).unwrap_err();
        assert_eq!(err.kind(), tetera_core::ErrorKind::InvalidArgument);
    }

    #[test]
    fn auto_skips_a_missing_journal_but_on_insists() {
        let auto = LogConfig {
            journald: Toggle::Auto,
            ..LogConfig::default()
        };
        let (_, handle) = build(
            &auto,
            &Env {
                under_systemd: true,
                ..env()
            },
        )
        .unwrap();
        assert!(
            !handle.journald() && handle.stderr(),
            "no journal, so stderr carries the records"
        );
        let on = LogConfig {
            journald: Toggle::On,
            ..LogConfig::default()
        };
        let err = build(&on, &env()).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("won't connect"), "{err}");
    }

    #[test]
    fn under_systemd_records_reach_journald_instead_of_stderr() {
        let socket =
            std::env::temp_dir().join(format!("tetera-log-build-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&socket);
        let journal = std::os::unix::net::UnixDatagram::bind(&socket).unwrap();
        journal
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let env = Env {
            under_systemd: true,
            journald_socket: socket.clone(),
            ..env()
        };
        let (subscriber, handle) = build(&LogConfig::default(), &env).unwrap();
        assert!(handle.journald() && !handle.stderr());
        tracing::subscriber::with_default(subscriber, || critical!("power lost"));
        let mut buf = [0u8; 1024];
        let n = journal.recv(&mut buf).unwrap();
        assert!(
            std::str::from_utf8(&buf[..n])
                .unwrap()
                .starts_with("PRIORITY=2\n")
        );
        std::fs::remove_file(&socket).unwrap();
    }

    #[test]
    fn records_reach_the_file_when_configured() {
        let dir =
            std::env::temp_dir().join(format!("tetera-log-build-file-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = LogConfig {
            stderr: Toggle::Off,
            file: Some(FileConfig {
                dir: dir.clone(),
                ..FileConfig::default()
            }),
            ..LogConfig::default()
        };
        let (subscriber, handle) = build(&config, &env()).unwrap();
        tracing::subscriber::with_default(subscriber, || critical!(frame = 9, "lost a frame"));
        let path = handle.file().unwrap().to_path_buf();
        assert_eq!(path, dir.join("tetera.log"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains(r#""level":"critical""#) && text.contains(r#""frame":"9""#),
            "{text}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_log_folder_that_cant_be_made_is_an_error() {
        let config = LogConfig {
            file: Some(FileConfig {
                dir: PathBuf::from("/proc/tetera-cannot-be-here"),
                ..FileConfig::default()
            }),
            ..LogConfig::default()
        };
        let err = build(&config, &env()).map(|_| ()).unwrap_err();
        assert!(err.to_string().contains("won't open"), "{err}");
    }

    #[test]
    fn a_drift_beyond_a_second_is_logged_as_a_warning() {
        let config = LogConfig {
            stderr: Toggle::Off,
            ..LogConfig::default()
        };
        let (subscriber, handle) = build(&config, &env()).unwrap();
        let clock = ManualClock::new(UtcTime::UNIX_EPOCH);
        clock.advance(Duration::from_secs(5));
        tracing::subscriber::with_default(subscriber, || {
            clock.set_wall_offset(900_000_000);
            assert!(!warn_on_drift(&clock));
            clock.set_wall_offset(-2_500_000_000);
            assert!(warn_on_drift(&clock));
        });
        let records = handle.ring().snapshot();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].record.level, LogLevel::Warn);
        assert_eq!(records[0].record.fields["drift_ms"], "-2500");
        assert!(
            records[0].record.message.contains("-2.500 s"),
            "{}",
            records[0].record.message
        );
    }
}
