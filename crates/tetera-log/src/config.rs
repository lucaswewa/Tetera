//! Logging's configuration, as `tetera.toml`'s `[log]` section will hold it (P1.13), and
//! the decisions it leads to, each a small function so it can be tested alone.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// On, off, or decided from how the process was started.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Toggle {
    /// Decided from how the process was started.
    #[default]
    Auto,
    /// Always.
    On,
    /// Never.
    Off,
}

/// Where the JSON log files go, and how they rotate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FileConfig {
    /// Their folder, made if missing.
    pub dir: PathBuf,
    /// The current file's name. Older ones add `.1`, `.2`, … the higher the older.
    pub name: String,
    /// A file is rotated before it would pass this size, in bytes.
    pub max_bytes: u64,
    /// How many rotated files are kept beside the current one.
    pub keep: usize,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("logs"),
            name: "tetera.log".into(),
            max_bytes: 10 * 1024 * 1024,
            keep: 5,
        }
    }
}

/// How the process logs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LogConfig {
    /// Which records are kept, in `tracing`'s filter syntax, such as `info` or
    /// `info,tetera_sim=debug`. `RUST_LOG` overrides it when set.
    pub filter: String,
    /// The name records carry in journald.
    pub identifier: String,
    /// How many of the newest records the ring keeps.
    pub ring_capacity: usize,
    /// Readable lines on standard error. `Auto`: on, unless journald gets the records.
    pub stderr: Toggle,
    /// journald. `Auto`: on when systemd started the process.
    pub journald: Toggle,
    /// JSON lines in rotating files, if wanted.
    pub file: Option<FileConfig>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            filter: "info".into(),
            identifier: "tetera".into(),
            ring_capacity: 10_000,
            stderr: Toggle::Auto,
            journald: Toggle::Auto,
            file: None,
        }
    }
}

/// Whether systemd started this process: it sets `JOURNAL_STREAM` or `INVOCATION_ID`.
pub fn under_systemd(journal_stream: Option<&str>, invocation_id: Option<&str>) -> bool {
    journal_stream.is_some_and(|s| !s.is_empty()) || invocation_id.is_some_and(|s| !s.is_empty())
}

/// Whether journald gets the records.
pub fn journald_wanted(mode: Toggle, under_systemd: bool) -> bool {
    match mode {
        Toggle::On => true,
        Toggle::Off => false,
        Toggle::Auto => under_systemd,
    }
}

/// Whether standard error gets the records. On `Auto`, not when journald does: systemd
/// already sends a service's standard error to journald, so each record would appear twice.
pub fn stderr_wanted(mode: Toggle, journald_on: bool) -> bool {
    match mode {
        Toggle::On => true,
        Toggle::Off => false,
        Toggle::Auto => !journald_on,
    }
}

/// The filter to use: `RUST_LOG` when it is set and not blank, the configuration's otherwise.
pub fn filter_directives<'a>(configured: &'a str, rust_log: Option<&'a str>) -> &'a str {
    match rust_log {
        Some(env) if !env.trim().is_empty() => env,
        _ => configured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systemd_is_recognised_by_its_variables() {
        assert!(under_systemd(Some("8:12345"), None));
        assert!(under_systemd(None, Some("4f6c…")));
        assert!(!under_systemd(None, None));
        assert!(!under_systemd(Some(""), Some("")));
    }

    #[test]
    fn journald_follows_its_toggle_or_systemd() {
        assert!(journald_wanted(Toggle::On, false));
        assert!(!journald_wanted(Toggle::Off, true));
        assert!(journald_wanted(Toggle::Auto, true));
        assert!(!journald_wanted(Toggle::Auto, false));
    }

    #[test]
    fn stderr_steps_aside_for_journald_on_auto_only() {
        assert!(stderr_wanted(Toggle::Auto, false));
        assert!(!stderr_wanted(Toggle::Auto, true));
        assert!(stderr_wanted(Toggle::On, true));
        assert!(!stderr_wanted(Toggle::Off, false));
    }

    #[test]
    fn rust_log_overrides_the_configured_filter_unless_blank() {
        assert_eq!(
            filter_directives("info", Some("debug,hyper=warn")),
            "debug,hyper=warn"
        );
        assert_eq!(filter_directives("info", Some("  ")), "info");
        assert_eq!(filter_directives("info", None), "info");
    }

    #[test]
    fn a_partial_configuration_fills_in_the_defaults() {
        let config: LogConfig = serde_json::from_str(
            r#"{"filter":"debug","journald":"off","file":{"dir":"/var/log/tetera"}}"#,
        )
        .unwrap();
        assert_eq!(config.filter, "debug");
        assert_eq!(config.journald, Toggle::Off);
        assert_eq!(config.stderr, Toggle::Auto);
        assert_eq!(config.ring_capacity, 10_000);
        let file = config.file.unwrap();
        assert_eq!(
            (file.dir, file.name.as_str(), file.keep),
            (PathBuf::from("/var/log/tetera"), "tetera.log", 5)
        );
        assert_eq!(file.max_bytes, 10 * 1024 * 1024);
    }
}
