//! journald, over its native protocol (ADR-0007): each record is one datagram of fields to
//! journald's socket. Written here rather than through `tracing-journald`, which maps
//! priority by level only, so that a critical record carries journald's CRIT priority.
//!
//! The protocol is systemd's own: `NAME=value` lines, or, for a value with a newline, the
//! name, a newline, the value's length as 64-bit little-endian, the value and a newline.
//! Records larger than one datagram (about 200 kB) are dropped; journald's memfd way of
//! passing them isn't implemented.

use crate::layer::Sink;
use std::io;
use std::os::unix::net::UnixDatagram;
use std::path::Path;
use tetera_core::{LogLevel, LogRecord};

/// journald's socket for the native protocol.
pub const SOCKET: &str = "/run/systemd/journal/socket";

/// A level's syslog priority: CRIT is 2, ERR 3, WARNING 4, INFO 6 and DEBUG 7.
pub fn priority(level: LogLevel) -> u8 {
    match level {
        LogLevel::Critical => 2,
        LogLevel::Error => 3,
        LogLevel::Warn => 4,
        LogLevel::Info => 6,
        LogLevel::Debug | LogLevel::Trace => 7,
    }
}

/// A record field's name as journald accepts it: upper-case letters, digits and
/// underscores, at most 64 characters, starting with a letter. Names journald keeps for
/// itself, or that would clash with this module's own, get a `FIELD_` prefix.
pub fn field_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    let reserved = matches!(out.as_str(), "MESSAGE" | "PRIORITY" | "SYSLOG_IDENTIFIER")
        || out.starts_with("TETERA_");
    if reserved || !out.starts_with(|c: char| c.is_ascii_uppercase()) {
        out.insert_str(0, "FIELD_");
    }
    out.truncate(64);
    out
}

/// The datagram for `record`, from the process `identifier`.
pub fn encode(record: &LogRecord, identifier: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    push(&mut out, "PRIORITY", &priority(record.level).to_string());
    push(&mut out, "SYSLOG_IDENTIFIER", identifier);
    push(&mut out, "MESSAGE", &record.message);
    push(&mut out, "TETERA_LEVEL", record.level.name());
    push(&mut out, "TETERA_TARGET", &record.target);
    for (name, value) in &record.fields {
        push(&mut out, &field_name(name), value);
    }
    out
}

fn push(out: &mut Vec<u8>, name: &str, value: &str) {
    out.extend_from_slice(name.as_bytes());
    if value.contains('\n') {
        out.push(b'\n');
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    } else {
        out.push(b'=');
        out.extend_from_slice(value.as_bytes());
    }
    out.push(b'\n');
}

/// journald, as a sink.
pub struct JournaldSink {
    socket: UnixDatagram,
    identifier: String,
}

impl JournaldSink {
    /// A sink sending to the socket at `path`: [`SOCKET`], or another for tests.
    pub fn connect(path: &Path, identifier: &str) -> io::Result<Self> {
        let socket = UnixDatagram::unbound()?;
        socket.connect(path)?;
        Ok(Self {
            socket,
            identifier: identifier.to_string(),
        })
    }
}

impl Sink for JournaldSink {
    fn write(&self, record: &LogRecord) {
        // A full or missing journal mustn't stop the instrument.
        let _ = self.socket.send(&encode(record, &self.identifier));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tetera_core::UtcTime;

    fn record(level: LogLevel, message: &str, fields: &[(&str, &str)]) -> LogRecord {
        LogRecord {
            time: UtcTime::UNIX_EPOCH,
            level,
            target: "tetera::test".into(),
            message: message.into(),
            fields: fields
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn each_level_has_its_syslog_priority() {
        let priorities: Vec<u8> = LogLevel::ALL.into_iter().map(priority).collect();
        assert_eq!(priorities, [7, 7, 6, 4, 3, 2]);
    }

    #[test]
    fn field_names_become_names_journald_accepts() {
        assert_eq!(field_name("z_um"), "Z_UM");
        assert_eq!(field_name("frame.seq"), "FRAME_SEQ");
        assert_eq!(field_name("_hidden"), "FIELD__HIDDEN");
        assert_eq!(field_name("3d"), "FIELD_3D");
        assert_eq!(field_name("message"), "FIELD_MESSAGE");
        assert_eq!(field_name("priority"), "FIELD_PRIORITY");
        assert_eq!(field_name("syslog_identifier"), "FIELD_SYSLOG_IDENTIFIER");
        assert_eq!(field_name("tetera_level"), "FIELD_TETERA_LEVEL");
        assert_eq!(field_name(&"x".repeat(80)).len(), 64);
        assert_eq!(
            field_name("µm"),
            "FIELD__M",
            "each non-ASCII character becomes an underscore"
        );
    }

    #[test]
    fn a_record_becomes_fields_with_its_priority() {
        let bytes = encode(
            &record(LogLevel::Critical, "disk full", &[("free_gb", "0")]),
            "tetera",
        );
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "PRIORITY=2\nSYSLOG_IDENTIFIER=tetera\nMESSAGE=disk full\nTETERA_LEVEL=critical\n\
             TETERA_TARGET=tetera::test\nFREE_GB=0\n"
        );
    }

    #[test]
    fn a_value_with_a_newline_is_sent_with_its_length() {
        let bytes = encode(&record(LogLevel::Info, "two\nlines", &[]), "t");
        let start = bytes.windows(8).position(|w| w == b"MESSAGE\n").unwrap();
        let after_name = start + 8;
        let len = u64::from_le_bytes(bytes[after_name..after_name + 8].try_into().unwrap());
        assert_eq!(len, 9);
        assert_eq!(&bytes[after_name + 8..after_name + 8 + 9], b"two\nlines");
        assert_eq!(bytes[after_name + 17], b'\n');
    }

    #[test]
    fn the_sink_sends_one_datagram_a_record() {
        let path =
            std::env::temp_dir().join(format!("tetera-journald-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let fake_journal = UnixDatagram::bind(&path).unwrap();
        fake_journal
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let sink = JournaldSink::connect(&path, "probe").unwrap();
        sink.write(&record(LogLevel::Warn, "slow frame", &[]));
        let mut buf = [0u8; 4096];
        let n = fake_journal.recv(&mut buf).unwrap();
        let text = std::str::from_utf8(&buf[..n]).unwrap();
        assert!(
            text.starts_with("PRIORITY=4\nSYSLOG_IDENTIFIER=probe\nMESSAGE=slow frame\n"),
            "{text}"
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn connecting_to_no_journal_fails() {
        let missing =
            std::env::temp_dir().join(format!("tetera-no-journal-{}.sock", std::process::id()));
        assert!(JournaldSink::connect(&missing, "x").is_err());
    }
}
