//! JSON log files, rotated by size (ADR-0007): the current file is renamed `.1` before it
//! would pass its limit, `.1` becomes `.2`, and so on, and the oldest beyond `keep` goes.
//! The disk a log uses is bounded: about `max_bytes × (keep + 1)`.

use crate::config::FileConfig;
use crate::layer::Sink;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use tetera_core::LogRecord;

/// A file of lines that rotates by size.
#[derive(Debug)]
pub struct RotatingFile {
    path: PathBuf,
    max_bytes: u64,
    keep: usize,
    file: File,
    size: u64,
}

impl RotatingFile {
    /// Opens the configuration's file for appending, making its folder if needed.
    pub fn open(config: &FileConfig) -> io::Result<Self> {
        fs::create_dir_all(&config.dir)?;
        let path = config.dir.join(&config.name);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let size = file.metadata()?.len();
        Ok(Self {
            path,
            max_bytes: config.max_bytes.max(1),
            keep: config.keep,
            file,
            size,
        })
    }

    /// The current file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The `n`th rotated file: `.1` is the newest.
    pub fn rotated(&self, n: usize) -> PathBuf {
        let mut name = self.path.clone().into_os_string();
        name.push(format!(".{n}"));
        PathBuf::from(name)
    }

    /// Appends `line` and a newline, rotating first if the line would pass the limit. A
    /// line longer than the limit still goes into a file of its own.
    pub fn write_line(&mut self, line: &str) -> io::Result<()> {
        let len = line.len() as u64 + 1;
        if self.size > 0 && self.size + len > self.max_bytes {
            self.rotate()?;
        }
        let mut bytes = Vec::with_capacity(line.len() + 1);
        bytes.extend_from_slice(line.as_bytes());
        bytes.push(b'\n');
        self.file.write_all(&bytes)?;
        self.size += len;
        Ok(())
    }

    fn rotate(&mut self) -> io::Result<()> {
        if self.keep == 0 {
            self.file = File::create(&self.path)?;
            self.size = 0;
            return Ok(());
        }
        for n in (1..self.keep).rev() {
            let from = self.rotated(n);
            if from.exists() {
                fs::rename(&from, self.rotated(n + 1))?;
            }
        }
        fs::rename(&self.path, self.rotated(1))?;
        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        self.size = 0;
        Ok(())
    }
}

/// The rotating file, as a sink of JSON lines.
pub struct FileSink(Mutex<RotatingFile>);

impl FileSink {
    /// A sink that writes to `file`.
    pub fn new(file: RotatingFile) -> Self {
        Self(Mutex::new(file))
    }
}

impl Sink for FileSink {
    fn write(&self, record: &LogRecord) {
        let Ok(line) = serde_json::to_string(record) else {
            return;
        };
        let mut file = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        // A full disk or a removed folder mustn't stop the instrument; the ring and
        // journald still have the record.
        let _ = file.write_line(&line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tetera_core::{LogLevel, UtcTime};

    struct Folder(PathBuf);

    impl Folder {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("tetera-log-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            Self(dir)
        }

        fn config(&self, max_bytes: u64, keep: usize) -> FileConfig {
            FileConfig {
                dir: self.0.join("logs"),
                name: "t.log".into(),
                max_bytes,
                keep,
            }
        }
    }

    impl Drop for Folder {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn lines(path: &Path) -> Vec<String> {
        fs::read_to_string(path)
            .map(|t| t.lines().map(str::to_string).collect())
            .unwrap_or_default()
    }

    #[test]
    fn a_file_rotates_before_passing_its_limit_and_keeps_only_so_many() {
        let folder = Folder::new("rotate");
        let mut file = RotatingFile::open(&folder.config(20, 2)).unwrap();
        for n in 0..8 {
            file.write_line(&format!("line {n:04}")).unwrap(); // 10 bytes with its newline
        }
        assert_eq!(lines(file.path()), ["line 0006", "line 0007"]);
        assert_eq!(lines(&file.rotated(1)), ["line 0004", "line 0005"]);
        assert_eq!(lines(&file.rotated(2)), ["line 0002", "line 0003"]);
        assert!(
            !file.rotated(3).exists(),
            "older files beyond `keep` are removed"
        );
        for path in [file.path().to_path_buf(), file.rotated(1), file.rotated(2)] {
            assert!(fs::metadata(&path).unwrap().len() <= 20, "{path:?}");
        }
    }

    #[test]
    fn appending_resumes_where_the_file_left_off() {
        let folder = Folder::new("resume");
        let config = folder.config(25, 1);
        RotatingFile::open(&config)
            .unwrap()
            .write_line("first line")
            .unwrap(); // 11 bytes
        let mut again = RotatingFile::open(&config).unwrap();
        again.write_line("second one").unwrap(); // 22 bytes in all: still one file
        assert_eq!(lines(again.path()), ["first line", "second one"]);
        again.write_line("third one!").unwrap(); // would make 33: rotates first
        assert_eq!(lines(again.path()), ["third one!"]);
        assert_eq!(lines(&again.rotated(1)), ["first line", "second one"]);
    }

    #[test]
    fn keeping_no_rotated_file_starts_the_file_afresh() {
        let folder = Folder::new("keep0");
        let mut file = RotatingFile::open(&folder.config(10, 0)).unwrap();
        file.write_line("aaaa").unwrap();
        file.write_line("bbbb").unwrap();
        file.write_line("cccc").unwrap();
        assert_eq!(lines(file.path()), ["cccc"]);
        assert!(!file.rotated(1).exists());
    }

    #[test]
    fn a_line_longer_than_the_limit_still_gets_written() {
        let folder = Folder::new("long");
        let mut file = RotatingFile::open(&folder.config(4, 3)).unwrap();
        file.write_line("much longer than four").unwrap();
        file.write_line("again").unwrap();
        assert_eq!(lines(file.path()), ["again"]);
        assert_eq!(lines(&file.rotated(1)), ["much longer than four"]);
        assert!(!file.rotated(2).exists(), "an empty file is never rotated");
    }

    #[test]
    fn the_newline_counts_and_a_line_that_fits_exactly_stays() {
        let folder = Folder::new("exact");
        let mut file = RotatingFile::open(&folder.config(20, 1)).unwrap();
        file.write_line("123456789").unwrap(); // 10 bytes with its newline
        file.write_line("123456789").unwrap(); // exactly 20: no rotation
        assert_eq!(fs::metadata(file.path()).unwrap().len(), 20);
        assert!(!file.rotated(1).exists());
        file.write_line("x").unwrap(); // 22 would pass the limit
        assert_eq!(lines(file.path()), ["x"]);
        assert_eq!(lines(&file.rotated(1)).len(), 2);
    }

    #[test]
    fn the_sink_writes_records_as_json_lines() {
        let folder = Folder::new("sink");
        let sink = FileSink::new(RotatingFile::open(&folder.config(1 << 20, 1)).unwrap());
        let mut fields = BTreeMap::new();
        fields.insert("frame".to_string(), "7".to_string());
        let record = LogRecord {
            time: UtcTime::UNIX_EPOCH,
            level: LogLevel::Critical,
            target: "t".into(),
            message: "disk full".into(),
            fields,
        };
        sink.write(&record);
        let written = lines(&folder.0.join("logs/t.log"));
        assert_eq!(written.len(), 1);
        assert_eq!(
            serde_json::from_str::<LogRecord>(&written[0]).unwrap(),
            record
        );
        assert!(written[0].contains(r#""level":"critical""#));
    }
}
