//! Log records (ADR-0007): what a log line is, wherever it goes, and [`LogRing`], the
//! bounded memory that keeps the newest records for the `logs` Thing (P1.20) to serve.
//! Writing records out is `tetera-log`'s; this module does no I/O.

use crate::error::Error;
use crate::time::UtcTime;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::str::FromStr;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// How severe a record is, from the least to the most.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Fine detail, for tracing a problem.
    Trace,
    /// Detail for developers.
    Debug,
    /// What the instrument is doing.
    Info,
    /// Something unexpected that it recovered from.
    Warn,
    /// Something failed.
    Error,
    /// Something failed that needs a person now: a crash of a device, data at risk.
    Critical,
}

impl LogLevel {
    /// Every level, from the least severe.
    pub const ALL: [LogLevel; 6] = [
        Self::Trace,
        Self::Debug,
        Self::Info,
        Self::Warn,
        Self::Error,
        Self::Critical,
    ];

    /// Its name, as it serialises: `trace` to `critical`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.name())
    }
}

impl FromStr for LogLevel {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        Self::ALL
            .into_iter()
            .find(|level| level.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "`{s}` isn't a log level: trace, debug, info, warn, error or critical"
                ))
            })
    }
}

/// One log record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LogRecord {
    /// When it was recorded.
    pub time: UtcTime,
    /// How severe it is.
    pub level: LogLevel,
    /// Where it came from: a module path, such as `tetera_sim::camera`.
    pub target: String,
    /// What it says.
    pub message: String,
    /// Its other fields, by name, as text.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, String>,
}

/// A record in the ring, with its sequence number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LogEntry {
    /// Its place in the order of records: 1 for the first, one more for each after.
    pub seq: u64,
    /// The record.
    pub record: LogRecord,
}

/// The newest records, up to a capacity. When full, a new record pushes out the oldest,
/// and the count of records pushed out goes up. Safe to share between threads.
#[derive(Debug)]
pub struct LogRing {
    inner: Mutex<Ring>,
}

#[derive(Debug)]
struct Ring {
    entries: VecDeque<LogEntry>,
    capacity: usize,
    last_seq: u64,
    dropped: u64,
}

impl LogRing {
    /// An empty ring that holds up to `capacity` records, and at least one.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Mutex::new(Ring {
                entries: VecDeque::with_capacity(capacity),
                capacity,
                last_seq: 0,
                dropped: 0,
            }),
        }
    }

    /// The ring, even if a thread panicked while holding it: a log must outlive a panic.
    fn lock(&self) -> MutexGuard<'_, Ring> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds `record`, pushing out the oldest if the ring is full. Returns its sequence number.
    pub fn push(&self, record: LogRecord) -> u64 {
        let mut ring = self.lock();
        if ring.entries.len() == ring.capacity {
            ring.entries.pop_front();
            ring.dropped += 1;
        }
        ring.last_seq += 1;
        let seq = ring.last_seq;
        ring.entries.push_back(LogEntry { seq, record });
        seq
    }

    /// How many records it holds at most.
    pub fn capacity(&self) -> usize {
        self.lock().capacity
    }

    /// How many records it holds.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Whether it holds no record.
    pub fn is_empty(&self) -> bool {
        self.lock().entries.is_empty()
    }

    /// How many records have been pushed out.
    pub fn dropped(&self) -> u64 {
        self.lock().dropped
    }

    /// The last record's sequence number, or 0 before the first.
    pub fn last_seq(&self) -> u64 {
        self.lock().last_seq
    }

    /// Every record it holds, oldest first.
    pub fn snapshot(&self) -> Vec<LogEntry> {
        self.lock().entries.iter().cloned().collect()
    }

    /// The records after sequence number `seq`, oldest first: what a tail hasn't seen yet.
    pub fn since(&self, seq: u64) -> Vec<LogEntry> {
        let ring = self.lock();
        let first = ring.entries.partition_point(|e| e.seq <= seq);
        ring.entries.range(first..).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(message: &str) -> LogRecord {
        LogRecord {
            time: UtcTime::UNIX_EPOCH,
            level: LogLevel::Info,
            target: "test".into(),
            message: message.into(),
            fields: BTreeMap::new(),
        }
    }

    fn messages(entries: &[LogEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.record.message.as_str()).collect()
    }

    #[test]
    fn levels_order_by_severity_and_read_their_names() {
        assert!(LogLevel::ALL.windows(2).all(|w| w[0] < w[1]));
        for level in LogLevel::ALL {
            assert_eq!(level.name().parse::<LogLevel>().unwrap(), level);
            assert_eq!(
                serde_json::to_string(&level).unwrap(),
                format!("\"{level}\"")
            );
        }
        assert_eq!("WARN".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!(
            format!("[{:<8}]", LogLevel::Info),
            "[info    ]",
            "width and alignment are honoured"
        );
        assert!("fatal".parse::<LogLevel>().is_err());
    }

    #[test]
    fn records_serialise_without_empty_fields() {
        let mut r = record("hello");
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(
            json,
            r#"{"time":"1970-01-01T00:00:00Z","level":"info","target":"test","message":"hello"}"#
        );
        r.fields.insert("objective".into(), "10x".into());
        let back: LogRecord = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn a_full_ring_pushes_out_its_oldest_and_counts_them() {
        let ring = LogRing::new(3);
        assert!(ring.is_empty() && ring.last_seq() == 0);
        for m in ["a", "b", "c", "d", "e"] {
            ring.push(record(m));
        }
        assert_eq!(messages(&ring.snapshot()), ["c", "d", "e"]);
        assert_eq!(
            (ring.len(), ring.capacity(), ring.dropped(), ring.last_seq()),
            (3, 3, 2, 5)
        );
    }

    #[test]
    fn a_ring_with_a_record_isnt_empty() {
        let ring = LogRing::new(2);
        ring.push(record("a"));
        assert!(!ring.is_empty());
    }

    #[test]
    fn a_tail_gets_only_what_it_hasnt_seen() {
        let ring = LogRing::new(10);
        let first = ring.push(record("a"));
        ring.push(record("b"));
        ring.push(record("c"));
        assert_eq!(first, 1);
        assert_eq!(messages(&ring.since(1)), ["b", "c"]);
        assert_eq!(messages(&ring.since(0)), ["a", "b", "c"]);
        assert!(ring.since(3).is_empty());
        let small = LogRing::new(2);
        for m in ["a", "b", "c", "d"] {
            small.push(record(m));
        }
        assert_eq!(
            messages(&small.since(1)),
            ["c", "d"],
            "a tail that fell behind gets what's left"
        );
    }

    #[test]
    fn a_ring_holds_at_least_one_record() {
        let ring = LogRing::new(0);
        ring.push(record("a"));
        ring.push(record("b"));
        assert_eq!(messages(&ring.snapshot()), ["b"]);
    }

    #[test]
    fn many_threads_push_without_losing_count() {
        let ring = LogRing::new(100);
        std::thread::scope(|s| {
            for t in 0..8 {
                let ring = &ring;
                s.spawn(move || {
                    for i in 0..1_000 {
                        ring.push(record(&format!("{t}:{i}")));
                    }
                });
            }
        });
        let entries = ring.snapshot();
        assert_eq!(
            (entries.len(), ring.dropped(), ring.last_seq()),
            (100, 7_900, 8_000)
        );
        assert!(entries.windows(2).all(|w| w[1].seq == w[0].seq + 1));
        assert_eq!(entries.last().unwrap().seq, 8_000);
    }

    #[test]
    fn a_panic_while_logging_doesnt_stop_the_log() {
        let ring = LogRing::new(4);
        let _ = std::thread::scope(|s| {
            s.spawn(|| {
                let _guard = ring.inner.lock().unwrap();
                panic!("a thread dies holding the ring");
            })
            .join()
        });
        ring.push(record("after"));
        assert_eq!(messages(&ring.snapshot()), ["after"]);
    }
}
