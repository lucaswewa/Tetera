//! A dataset on disk (ADR-0021): its folder, holding the frames as OME-Zarr v3 with OME-NGFF
//! 0.5's metadata, its `metadata.json`, and a JPEG thumbnail; written frame by frame, and read
//! back.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tetera_core::frame::format::{BayerPattern, Channel};
use tetera_core::frame::meta::FrameMeta;
use tetera_core::{Error, Frame, PixelFormat, Result, UtcTime};
use zarrs::array::Array;
use zarrs::array::codec::ZstdCodec;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

/// The OME-Zarr store's folder in a dataset's.
pub const IMAGE: &str = "image.ome.zarr";
/// The metadata's file.
pub const METADATA: &str = "metadata.json";
/// The thumbnail's file.
pub const THUMBNAIL: &str = "thumbnail.jpg";
/// The thumbnail's longest side, pixels.
pub const THUMBNAIL_SIZE: u32 = 256;
/// zstd's level for the frames: fast, and still a good ratio on raw frames.
const ZSTD_LEVEL: i32 = 3;

fn zarr(e: impl std::fmt::Display) -> Error {
    Error::internal(format!("OME-Zarr: {e}"))
}

/// What a new dataset holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DatasetSpec {
    /// Its name, as people see it.
    pub name: String,
    /// What made it, such as `capture` or `scan`.
    pub kind: String,
    /// Its frames' format: raw Bayer or grey, 8 or 16 bits.
    pub format: PixelFormat,
    /// Its frames' width.
    pub width: u32,
    /// Their height.
    pub height: u32,
    /// The frames it will hold, at most.
    pub frames: u64,
    /// A pixel's size at the sample, µm.
    pub pixel_size_um: f64,
    /// The time between frames, s; 0 for a single one.
    pub interval_s: f64,
    /// The instrument as it was: camera settings, objective, calibration and the like.
    pub instrument: serde_json::Value,
}

/// A dataset's `metadata.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Metadata {
    /// Its id.
    pub id: String,
    /// When it was made.
    pub created: UtcTime,
    /// What it holds.
    pub spec: DatasetSpec,
    /// Each frame's metadata: its number, times, settings and where the stage was.
    pub frames: Vec<FrameMeta>,
}

/// A frame's samples, as stored.
#[derive(Debug, Clone, PartialEq)]
pub enum Samples {
    /// One byte each.
    U8(Vec<u8>),
    /// Two bytes each.
    U16(Vec<u16>),
}

/// Whether `format` is one a dataset holds: raw Bayer or grey.
pub(crate) fn check_format(format: PixelFormat) -> Result<()> {
    match format {
        PixelFormat::Bayer8 { .. }
        | PixelFormat::Bayer16 { .. }
        | PixelFormat::Mono8
        | PixelFormat::Mono16 => Ok(()),
        other => Err(Error::invalid_argument(format!(
            "a dataset holds raw Bayer or grey frames, not {other:?}"
        ))),
    }
}

/// OME-NGFF 0.5's metadata for the image: axes t, c, y and x, one scale.
pub fn ngff(spec: &DatasetSpec) -> serde_json::Value {
    let interval = if spec.interval_s > 0.0 {
        spec.interval_s
    } else {
        1.0
    };
    serde_json::json!({
        "version": "0.5",
        "multiscales": [{
            "name": spec.name,
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": [{
                "path": "0",
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [interval, 1.0, spec.pixel_size_um, spec.pixel_size_um]
                }]
            }]
        }]
    })
}

/// The OME-Zarr image of `spec` made in `dir`: its group's metadata, and its empty array.
pub(crate) fn create_image(dir: &Path, spec: &DatasetSpec) -> Result<Array<FilesystemStore>> {
    let store = Arc::new(FilesystemStore::new(dir.join(IMAGE)).map_err(zarr)?);
    let mut group = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(zarr)?;
    group.attributes_mut().insert("ome".into(), ngff(spec));
    group.store_metadata().map_err(zarr)?;
    let data_type = if spec.format.bytes_per_sample() == 1 {
        zarrs::array::data_type::uint8()
    } else {
        zarrs::array::data_type::uint16()
    };
    let shape = vec![
        spec.frames,
        1,
        u64::from(spec.height),
        u64::from(spec.width),
    ];
    let chunk = vec![1, 1, u64::from(spec.height), u64::from(spec.width)];
    let fill: zarrs::array::FillValue = if spec.format.bytes_per_sample() == 1 {
        0u8.into()
    } else {
        0u16.into()
    };
    let array = zarrs::array::ArrayBuilder::new(shape, chunk, data_type, fill)
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(ZSTD_LEVEL, false))])
        .dimension_names(Some(["t", "c", "y", "x"]))
        .build(store, "/0")
        .map_err(zarr)?;
    array.store_metadata().map_err(zarr)?;
    Ok(array)
}

/// `frame`'s samples, its rows packed, after checking it fits `spec`.
pub(crate) fn samples(frame: &Frame, spec: &DatasetSpec) -> Result<Samples> {
    let layout = frame.layout();
    if (layout.format, layout.width, layout.height) != (spec.format, spec.width, spec.height) {
        return Err(Error::invalid_argument(format!(
            "a {} × {} {:?} frame in a dataset of {} × {} {:?}",
            layout.width, layout.height, layout.format, spec.width, spec.height, spec.format
        )));
    }
    let row = layout.width as usize * layout.format.bytes_per_sample();
    let rows = frame
        .data()
        .chunks(layout.stride)
        .take(layout.height as usize)
        .map(|r| &r[..row]);
    Ok(if layout.format.bytes_per_sample() == 1 {
        Samples::U8(rows.flatten().copied().collect())
    } else {
        Samples::U16(
            rows.flat_map(|r| r.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)))
                .collect(),
        )
    })
}

/// Stores `samples` as frame `t` of `array`.
pub(crate) fn store_frame(array: &Array<FilesystemStore>, t: u64, samples: Samples) -> Result<()> {
    match samples {
        Samples::U8(s) => array.store_chunk(&[t, 0, 0, 0], s),
        Samples::U16(s) => array.store_chunk(&[t, 0, 0, 0], s),
    }
    .map_err(zarr)
}

/// A dataset read back from its folder.
pub struct DatasetReader {
    array: Array<FilesystemStore>,
    metadata: Metadata,
}

impl std::fmt::Debug for DatasetReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatasetReader")
            .field("id", &self.metadata.id)
            .finish_non_exhaustive()
    }
}

impl DatasetReader {
    /// The dataset in `dir`.
    pub fn open(dir: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(dir.join(METADATA))
            .map_err(|e| Error::not_found(format!("{}: {e}", dir.join(METADATA).display())))?;
        let metadata: Metadata = serde_json::from_str(&json)
            .map_err(|e| Error::internal(format!("{}: {e}", dir.join(METADATA).display())))?;
        let store = Arc::new(FilesystemStore::new(dir.join(IMAGE)).map_err(zarr)?);
        let array = Array::open(store, "/0").map_err(zarr)?;
        Ok(Self { array, metadata })
    }

    /// Its metadata.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Its frames.
    pub fn frames(&self) -> u64 {
        self.array.shape()[0]
    }

    /// Frame `t`'s samples.
    pub fn read(&self, t: u64) -> Result<Samples> {
        if t >= self.frames() {
            return Err(Error::invalid_argument(format!(
                "frame {t} of a dataset of {}",
                self.frames()
            )));
        }
        let at = [t, 0, 0, 0];
        Ok(if self.metadata.spec.format.bytes_per_sample() == 1 {
            Samples::U8(self.array.retrieve_chunk::<Vec<u8>>(&at).map_err(zarr)?)
        } else {
            Samples::U16(self.array.retrieve_chunk::<Vec<u16>>(&at).map_err(zarr)?)
        })
    }
}

/// A JPEG thumbnail of `samples`, a frame of `spec`, of [`thumbnail_rgb`]'s pixels.
pub(crate) fn thumbnail(samples: &Samples, spec: &DatasetSpec) -> Result<Vec<u8>> {
    let (w, h, rgb) = thumbnail_rgb(samples, spec)?;
    let image = tetera_imaging::Image::new(w, h, 3, rgb)?;
    tetera_imaging::jpeg::encode(&image, 80)
}

/// A thumbnail's pixels, red, green and blue, and its size: each 2 × 2 Bayer cell as one
/// pixel, its red, the mean of its greens, and its blue (grey frames 2 × 2 binned), every
/// `step`th cell so that it is at most [`THUMBNAIL_SIZE`] each way, brightened by a display
/// gamma of 2.2.
pub(crate) fn thumbnail_rgb(samples: &Samples, spec: &DatasetSpec) -> Result<(u32, u32, Vec<u8>)> {
    let (w, h) = (spec.width as usize, spec.height as usize);
    let full = match spec.format {
        PixelFormat::Bayer16 {
            significant_bits, ..
        } => ((1u32 << significant_bits) - 1) as f32,
        PixelFormat::Mono16 => 65_535.0,
        _ => 255.0,
    };
    let at = |x: usize, y: usize| -> f32 {
        match samples {
            Samples::U8(s) => f32::from(s[y * w + x]),
            Samples::U16(s) => f32::from(s[y * w + x]),
        }
    };
    let pattern: Option<BayerPattern> = match spec.format {
        PixelFormat::Bayer8 { pattern } | PixelFormat::Bayer16 { pattern, .. } => Some(pattern),
        _ => None,
    };
    let (cw, ch) = (w / 2, h / 2);
    let step = cw
        .div_ceil(THUMBNAIL_SIZE as usize)
        .max(ch.div_ceil(THUMBNAIL_SIZE as usize))
        .max(1);
    // At least a pixel each way, however skinny the frame: a dataset's frames are 2 × 2 or more.
    let (tw, th) = ((cw / step).max(1), (ch / step).max(1));
    let gamma: Vec<u8> = (0..=255u32)
        .map(|v| ((v as f32 / 255.0).powf(1.0 / 2.2) * 255.0).round() as u8)
        .collect();
    let level = |v: f32| gamma[((v / full).clamp(0.0, 1.0) * 255.0).round() as usize];
    let mut rgb = Vec::with_capacity(tw * th * 3);
    for ty in 0..th {
        for tx in 0..tw {
            let (x, y) = (tx * step * 2, ty * step * 2);
            let mut sum = [0.0f32; 3];
            let mut n = [0.0f32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let c = match pattern.map(|p| p.channel_at((x + dx) as u32, (y + dy) as u32)) {
                    Some(Channel::Red) => 0,
                    Some(Channel::Blue) => 2,
                    Some(_) => 1,
                    None => 3,
                };
                let v = at(x + dx, y + dy);
                if c == 3 {
                    for k in 0..3 {
                        sum[k] += v;
                        n[k] += 1.0;
                    }
                } else {
                    sum[c] += v;
                    n[c] += 1.0;
                }
            }
            for k in 0..3 {
                rgb.push(level(sum[k] / n[k]));
            }
        }
    }
    Ok((tw as u32, th as u32, rgb))
}

/// The bytes the files under `dir` take.
pub(crate) fn size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => size(&e.path()),
            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
        })
        .sum()
}

/// The bytes the file at `path` takes.
pub(crate) fn size_of(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// `dir`'s path relative to `root`, as the index keeps it.
pub(crate) fn relative(dir: &Path, root: &Path) -> PathBuf {
    dir.strip_prefix(root).unwrap_or(dir).to_path_buf()
}
