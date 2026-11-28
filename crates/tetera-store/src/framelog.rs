//! The scratch frame log (ADR-0021): raw frames appended in order to preallocated segment
//! files, as fast as the camera makes them, each with a line in an index; read back in order,
//! for ingest.
//!
//! A log is a folder: `log.json`, its frames' format and size; `segment-NNNNN.bin`, each
//! `segment_frames` frames, their rows packed; and `index.jsonl`, a line for each frame, its
//! segment, its place in it, and its metadata. The index is written after the frame, so a
//! line names only a frame that is there; a crash can leave a torn last line, which is passed
//! over.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tetera_core::frame::meta::FrameMeta;
use tetera_core::{Error, Frame, PixelFormat, Result, UtcTime};

/// The log's header.
pub const HEADER: &str = "log.json";
/// The log's index.
pub const INDEX: &str = "index.jsonl";

fn io(path: &Path, e: std::io::Error) -> Error {
    Error::internal(format!("{}: {e}", path.display()))
}

fn segment_path(dir: &Path, n: u32) -> PathBuf {
    dir.join(format!("segment-{n:05}.bin"))
}

/// What a log holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogHeader {
    /// Its frames' format.
    pub format: PixelFormat,
    /// Their width.
    pub width: u32,
    /// Their height.
    pub height: u32,
    /// A frame's bytes, its rows packed.
    pub frame_bytes: u64,
    /// The frames in each segment.
    pub segment_frames: u64,
    /// When it began.
    pub created: UtcTime,
}

/// A frame's line in the index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogEntry {
    /// Its segment.
    pub segment: u32,
    /// Where it begins in the segment, bytes.
    pub offset: u64,
    /// Its metadata.
    pub meta: FrameMeta,
}

/// A log being written.
pub struct FrameLogWriter {
    dir: PathBuf,
    header: LogHeader,
    segment: Option<(u32, File)>,
    in_segment: u64,
    index: BufWriter<File>,
    written: u64,
    packed: Vec<u8>,
}

impl std::fmt::Debug for FrameLogWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameLogWriter")
            .field("dir", &self.dir)
            .field("written", &self.written)
            .finish_non_exhaustive()
    }
}

impl FrameLogWriter {
    /// A new log in `dir`, which must not exist, of `format` frames of `width` × `height`,
    /// `segment_frames` to a segment.
    pub fn create(
        dir: &Path,
        format: PixelFormat,
        width: u32,
        height: u32,
        segment_frames: u64,
    ) -> Result<Self> {
        if segment_frames == 0 || width == 0 || height == 0 || format.planes() != 1 {
            return Err(Error::invalid_argument(format!(
                "a frame log of {width} × {height} {format:?}, {segment_frames} frames a segment"
            )));
        }
        if dir.exists() {
            return Err(Error::invalid_argument(format!(
                "{} exists: a frame log makes its own folder",
                dir.display()
            )));
        }
        std::fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
        let header = LogHeader {
            format,
            width,
            height,
            frame_bytes: u64::from(width) * u64::from(height) * format.bytes_per_sample() as u64,
            segment_frames,
            created: UtcTime::now(),
        };
        let path = dir.join(HEADER);
        let json = serde_json::to_vec_pretty(&header)
            .map_err(|e| Error::internal(format!("a frame log's header: {e}")))?;
        std::fs::write(&path, json).map_err(|e| io(&path, e))?;
        let path = dir.join(INDEX);
        let index = BufWriter::new(File::create(&path).map_err(|e| io(&path, e))?);
        Ok(Self {
            dir: dir.to_path_buf(),
            header,
            segment: None,
            in_segment: 0,
            index,
            written: 0,
            packed: Vec::new(),
        })
    }

    /// The frames written.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Its folder.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Appends `frame`, the next.
    pub fn append(&mut self, frame: &Frame) -> Result<()> {
        let layout = frame.layout();
        let h = &self.header;
        if (layout.format, layout.width, layout.height) != (h.format, h.width, h.height) {
            return Err(Error::invalid_argument(format!(
                "a {} × {} {:?} frame in a log of {} × {} {:?}",
                layout.width, layout.height, layout.format, h.width, h.height, h.format
            )));
        }
        if self.segment.is_none() || self.in_segment == h.segment_frames {
            self.next_segment()?;
        }
        let (n, file) = self.segment.as_mut().expect("opened above");
        let row = layout.row_bytes();
        let bytes = if layout.stride == row {
            &frame.data()[..self.header.frame_bytes as usize]
        } else {
            self.packed.clear();
            for r in frame
                .data()
                .chunks(layout.stride)
                .take(layout.height as usize)
            {
                self.packed.extend_from_slice(&r[..row]);
            }
            &self.packed[..]
        };
        let path = segment_path(&self.dir, *n);
        file.write_all(bytes).map_err(|e| io(&path, e))?;
        let entry = LogEntry {
            segment: *n,
            offset: self.in_segment * self.header.frame_bytes,
            meta: frame.meta().clone(),
        };
        let line = serde_json::to_string(&entry)
            .map_err(|e| Error::internal(format!("a frame log's index: {e}")))?;
        writeln!(self.index, "{line}").map_err(|e| io(&self.dir.join(INDEX), e))?;
        self.in_segment += 1;
        self.written += 1;
        Ok(())
    }

    /// Closes the segment in use, cut to the frames in it, and opens the next, preallocated.
    fn next_segment(&mut self) -> Result<()> {
        let n = match self.segment.take() {
            Some((n, file)) => {
                self.close(n, file)?;
                n + 1
            }
            None => 0,
        };
        let path = segment_path(&self.dir, n);
        let file = File::create(&path).map_err(|e| io(&path, e))?;
        file.set_len(self.header.segment_frames * self.header.frame_bytes)
            .map_err(|e| io(&path, e))?;
        self.segment = Some((n, file));
        self.in_segment = 0;
        Ok(())
    }

    fn close(&self, n: u32, file: File) -> Result<()> {
        let path = segment_path(&self.dir, n);
        file.set_len(self.in_segment * self.header.frame_bytes)
            .map_err(|e| io(&path, e))?;
        file.sync_data().map_err(|e| io(&path, e))
    }

    /// Finishes it: the last segment cut to its frames, and everything on disk. The frames
    /// written.
    pub fn finish(mut self) -> Result<u64> {
        if let Some((n, file)) = self.segment.take() {
            self.close(n, file)?;
        }
        let path = self.dir.join(INDEX);
        self.index.flush().map_err(|e| io(&path, e))?;
        self.index.get_ref().sync_data().map_err(|e| io(&path, e))?;
        Ok(self.written)
    }
}

/// A log, read back.
#[derive(Debug)]
pub struct FrameLog {
    dir: PathBuf,
    header: LogHeader,
    entries: Vec<LogEntry>,
}

impl FrameLog {
    /// The log in `dir`. A torn last line of its index, left by a crash, is passed over.
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join(HEADER);
        let header: LogHeader =
            serde_json::from_slice(&std::fs::read(&path).map_err(|e| io(&path, e))?)
                .map_err(|e| Error::internal(format!("{}: {e}", path.display())))?;
        let path = dir.join(INDEX);
        let text = std::fs::read_to_string(&path).map_err(|e| io(&path, e))?;
        let lines: Vec<&str> = text.lines().collect();
        let mut entries = Vec::with_capacity(lines.len());
        for (k, line) in lines.iter().enumerate() {
            match serde_json::from_str::<LogEntry>(line) {
                Ok(entry) => entries.push(entry),
                Err(_) if k + 1 == lines.len() && !text.ends_with('\n') => {}
                Err(e) => {
                    return Err(Error::internal(format!(
                        "{}, line {}: {e}",
                        path.display(),
                        k + 1
                    )));
                }
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            header,
            entries,
        })
    }

    /// What it holds.
    pub fn header(&self) -> &LogHeader {
        &self.header
    }

    /// Its frames' entries, in order.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Its frames.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Frame `i`'s metadata, and its rows, packed.
    pub fn read(&self, i: usize) -> Result<(FrameMeta, Vec<u8>)> {
        let entry = self.entries.get(i).ok_or_else(|| {
            Error::invalid_argument(format!("frame {i} of a log of {}", self.len()))
        })?;
        let path = segment_path(&self.dir, entry.segment);
        let mut file = File::open(&path).map_err(|e| io(&path, e))?;
        file.seek(SeekFrom::Start(entry.offset))
            .map_err(|e| io(&path, e))?;
        let mut bytes = vec![0; self.header.frame_bytes as usize];
        file.read_exact(&mut bytes).map_err(|e| io(&path, e))?;
        Ok((entry.meta.clone(), bytes))
    }

    /// Deletes it.
    pub fn remove(self) -> Result<()> {
        std::fs::remove_dir_all(&self.dir).map_err(|e| io(&self.dir, e))
    }
}
