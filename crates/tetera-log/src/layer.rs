//! The `tracing` layer that turns each event into one [`LogRecord`] and hands it to every
//! [`Sink`]: the ring, the file, journald.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use tetera_core::{LogLevel, LogRecord, LogRing, UtcTime};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

/// The field that marks a record as critical: `critical = true`, which [`critical!`] sets.
///
/// [`critical!`]: crate::critical
pub const CRITICAL_FIELD: &str = "critical";

/// Somewhere records go.
pub trait Sink: Send + Sync {
    /// Takes one record. A sink that fails keeps quiet: logging must never stop the
    /// instrument.
    fn write(&self, record: &LogRecord);
}

/// The ring, as a sink.
pub struct RingSink(pub Arc<LogRing>);

impl Sink for RingSink {
    fn write(&self, record: &LogRecord) {
        self.0.push(record.clone());
    }
}

/// The layer that makes records and hands them to its sinks.
pub struct RecordLayer {
    sinks: Vec<Box<dyn Sink>>,
}

impl RecordLayer {
    /// A layer that writes to `sinks`.
    pub fn new(sinks: Vec<Box<dyn Sink>>) -> Self {
        Self { sinks }
    }
}

impl<S: Subscriber> Layer<S> for RecordLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let record = to_record(event, UtcTime::now());
        for sink in &self.sinks {
            sink.write(&record);
        }
    }
}

/// The level of a `tracing` event: `critical = true` makes it critical, whatever its level.
pub fn level_of(level: &Level, critical: bool) -> LogLevel {
    if critical {
        return LogLevel::Critical;
    }
    // `Level`'s five values are constants, so the match can't see it is complete: the last
    // arm is ERROR.
    match *level {
        Level::TRACE => LogLevel::Trace,
        Level::DEBUG => LogLevel::Debug,
        Level::INFO => LogLevel::Info,
        Level::WARN => LogLevel::Warn,
        _ => LogLevel::Error,
    }
}

/// The record an event makes, at `time`.
pub fn to_record(event: &Event<'_>, time: UtcTime) -> LogRecord {
    let mut fields = Fields::default();
    event.record(&mut fields);
    let meta = event.metadata();
    LogRecord {
        time,
        level: level_of(meta.level(), fields.critical),
        target: meta.target().to_string(),
        message: fields.message,
        fields: fields.rest,
    }
}

#[derive(Default)]
struct Fields {
    message: String,
    rest: BTreeMap<String, String>,
    critical: bool,
}

impl Fields {
    fn put(&mut self, field: &Field, value: String) {
        if field.name() == "message" {
            self.message = value;
        } else {
            self.rest.insert(field.name().to_string(), value);
        }
    }
}

impl Visit for Fields {
    fn record_bool(&mut self, field: &Field, value: bool) {
        if field.name() == CRITICAL_FIELD {
            self.critical = value;
        } else {
            self.put(field, value.to_string());
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.put(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.put(field, format!("{value:?}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    fn capture(emit: impl FnOnce()) -> Vec<LogRecord> {
        let ring = Arc::new(LogRing::new(100));
        let layer = RecordLayer::new(vec![Box::new(RingSink(ring.clone()))]);
        tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), emit);
        ring.snapshot().into_iter().map(|e| e.record).collect()
    }

    #[test]
    fn each_level_becomes_its_record_level() {
        let records = capture(|| {
            tracing::trace!("t");
            tracing::debug!("d");
            tracing::info!("i");
            tracing::warn!("w");
            tracing::error!("e");
            crate::critical!("c");
        });
        let levels: Vec<LogLevel> = records.iter().map(|r| r.level).collect();
        assert_eq!(levels, LogLevel::ALL);
        assert!(
            records.iter().all(|r| r.fields.is_empty()),
            "`critical` isn't kept as a field"
        );
    }

    #[test]
    fn messages_fields_and_targets_are_kept_as_text() {
        let records = capture(|| {
            tracing::info!(target: "tetera::stage", objective = "10x", z_um = 12.5, homed = true, count = 3u32, "moved {} µm", 40);
        });
        let r = &records[0];
        assert_eq!(r.target, "tetera::stage");
        assert_eq!(r.message, "moved 40 µm");
        let fields: Vec<(&str, &str)> = r
            .fields
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            fields,
            [
                ("count", "3"),
                ("homed", "true"),
                ("objective", "10x"),
                ("z_um", "12.5")
            ]
        );
    }

    #[test]
    fn critical_can_be_asked_of_any_level_and_false_isnt_critical() {
        let records = capture(|| {
            tracing::warn!(critical = true, "the scratch disk is full");
            tracing::error!(critical = false, "a frame was dropped");
        });
        assert_eq!(records[0].level, LogLevel::Critical);
        assert_eq!(records[1].level, LogLevel::Error);
    }

    #[test]
    fn records_carry_the_time_they_were_made() {
        let before = UtcTime::now();
        let records = capture(|| tracing::info!("now"));
        let after = UtcTime::now();
        assert!(records[0].time >= before && records[0].time <= after);
    }
}
