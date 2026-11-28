//! The store: datasets written and read back exactly, 8 and 16 bits, raw and grey; their
//! OME-NGFF 0.5 metadata; the index across a restart; abandoned and deleted datasets; and bad
//! requests refused.

use crate::dataset::{IMAGE, METADATA, THUMBNAIL, THUMBNAIL_SIZE};
use crate::{DatasetSpec, Samples, Store};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tetera_core::frame::format::BayerPattern;
use tetera_core::frame::pool::HostAllocator;
use tetera_core::{ErrorKind, Frame, FrameLayout, FramePool, PixelFormat, Timestamp};

/// A folder of the test's own, removed when it drops.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("tetera-store-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn bayer16() -> PixelFormat {
    PixelFormat::bayer16(BayerPattern::Rggb, 12).unwrap()
}

fn spec(format: PixelFormat, frames: u64) -> DatasetSpec {
    DatasetSpec {
        name: "a burst".into(),
        kind: "capture".into(),
        format,
        width: 64,
        height: 48,
        frames,
        pixel_size_um: 0.24,
        interval_s: 0.05,
        instrument: json!({"objective": "20x", "exposure_ms": 5.0}),
    }
}

/// `n` frames of `format`, 64 × 48, each site a pattern of its place and the frame's number.
fn frames(format: PixelFormat, n: u64) -> (FramePool, Vec<Frame>) {
    frames_of(format, 64, n)
}

/// `n` frames of `format`, `width` × 48, each site a pattern of its place and the frame's
/// number.
fn frames_of(format: PixelFormat, width: u32, n: u64) -> (FramePool, Vec<Frame>) {
    let pool = FramePool::new(
        FrameLayout::new(format, width, 48).unwrap(),
        n as usize,
        &HostAllocator,
    )
    .unwrap();
    let frames = (0..n)
        .map(|t| {
            let mut f = pool.take(Duration::from_secs(5)).unwrap();
            let stride = f.layout().stride;
            let bytes = format.bytes_per_sample();
            for (y, row) in f.data_mut().chunks_mut(stride).enumerate() {
                for x in 0..width as usize {
                    let v = ((x * 31 + y * 17 + t as usize * 7) % 4_096) as u16;
                    if bytes == 1 {
                        row[x] = v as u8;
                    } else {
                        row[2 * x..2 * x + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
            let meta = f.meta_mut();
            meta.seq = 100 + t;
            meta.exposure_start = Timestamp(t * 50_000_000);
            f.freeze()
        })
        .collect();
    (pool, frames)
}

/// The samples frame `f` holds, as the store keeps them.
fn expected(f: &Frame) -> Samples {
    let l = f.layout();
    let rows = f.data().chunks(l.stride).take(l.height as usize);
    if l.format.bytes_per_sample() == 1 {
        Samples::U8(rows.flat_map(|r| r[..l.width as usize].to_vec()).collect())
    } else {
        Samples::U16(
            rows.flat_map(|r| {
                r[..2 * l.width as usize]
                    .chunks(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect::<Vec<_>>()
            })
            .collect(),
        )
    }
}

/// The bytes of every file under `dir`.
fn tree_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| {
            if e.path().is_dir() {
                tree_bytes(&e.path())
            } else {
                e.metadata().unwrap().len()
            }
        })
        .sum()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Checks `attrs` against OME-NGFF 0.5's rules for an image's multiscales: its version; two to
/// five axes, named once each, time first, then channel, then two or three space axes last,
/// with units the spec knows; and each dataset's single scale, one positive factor an axis.
fn check_ngff_05(attrs: &Value) {
    let ome = &attrs["ome"];
    assert_eq!(ome["version"], "0.5", "{attrs}");
    let ms = ome["multiscales"].as_array().expect("multiscales");
    assert!(!ms.is_empty());
    for m in ms {
        let axes = m["axes"].as_array().expect("axes");
        assert!((2..=5).contains(&axes.len()));
        let names: Vec<&str> = axes.iter().map(|a| a["name"].as_str().unwrap()).collect();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "axes named once each");
        let types: Vec<&str> = axes
            .iter()
            .map(|a| a["type"].as_str().unwrap_or(""))
            .collect();
        let rank = |t: &str| match t {
            "time" => 0,
            "channel" => 1,
            "space" => 2,
            other => panic!("an axis of type {other:?}"),
        };
        assert!(
            types.windows(2).all(|w| rank(w[0]) <= rank(w[1])),
            "{types:?}"
        );
        let space = types.iter().filter(|t| **t == "space").count();
        assert!((2..=3).contains(&space));
        for a in axes {
            match (a["type"].as_str(), a["unit"].as_str()) {
                (Some("space"), Some(u)) => {
                    assert!(["micrometer", "millimeter", "nanometer"].contains(&u));
                }
                (Some("time"), Some(u)) => assert!(["second", "millisecond"].contains(&u)),
                (Some("channel"), None) => {}
                other => panic!("an axis {a}: {other:?}"),
            }
        }
        let datasets = m["datasets"].as_array().expect("datasets");
        assert!(!datasets.is_empty());
        for d in datasets {
            assert!(d["path"].is_string());
            let t = d["coordinateTransformations"].as_array().unwrap();
            assert_eq!(t.len(), 1);
            assert_eq!(t[0]["type"], "scale");
            let scale = t[0]["scale"].as_array().unwrap();
            assert_eq!(scale.len(), axes.len());
            assert!(scale.iter().all(|s| s.as_f64().is_some_and(|s| s > 0.0)));
        }
    }
}

#[test]
fn a_dataset_reads_back_exactly_in_8_and_16_bits_raw_or_grey() {
    for format in [
        bayer16(),
        PixelFormat::Bayer8 {
            pattern: BayerPattern::Gbrg,
        },
        PixelFormat::Mono16,
        PixelFormat::Mono8,
    ] {
        let scratch = Scratch::new();
        let store = Store::open(&scratch.0).unwrap();
        store.create_project("Tonsil 3").unwrap();
        let (_pool, frames) = frames(format, 3);
        let mut w = store.create_dataset("Tonsil 3", spec(format, 3)).unwrap();
        for f in &frames {
            w.append(f).unwrap();
        }
        assert_eq!(w.written(), 3);
        assert!(format!("{w:?}").contains(w.id()));
        let info = w.finish().unwrap();
        assert!(info.complete);
        assert_eq!(
            (info.frames, info.project.as_str(), info.name.as_str()),
            (3, "Tonsil 3", "a burst")
        );
        let r = store.read(&info.id).unwrap();
        assert_eq!(r.frames(), 3);
        for (t, f) in frames.iter().enumerate() {
            assert_eq!(
                r.read(t as u64).unwrap(),
                expected(f),
                "{format:?}, frame {t}"
            );
        }
        assert!(r.read(3).is_err());
        let m = r.metadata();
        assert_eq!(
            m.frames.iter().map(|f| f.seq).collect::<Vec<_>>(),
            [100, 101, 102]
        );
        assert_eq!(m.spec.format, format);
        assert_eq!(m.spec.instrument["objective"], "20x");
        assert!(format!("{r:?}").contains(&info.id));
        // Its files, each counted.
        let files = store.files(&info.id).unwrap();
        let kinds: Vec<&str> = files.iter().map(|f| f.kind.as_str()).collect();
        assert_eq!(kinds, ["ome-zarr", "metadata", "thumbnail"]);
        assert!(files.iter().all(|f| f.bytes > 0));
        let dir = scratch.0.join(&info.path);
        assert_eq!(
            files[0].bytes,
            tree_bytes(&dir.join(IMAGE)),
            "the whole tree"
        );
        assert_eq!(
            files[1].bytes,
            std::fs::metadata(dir.join(METADATA)).unwrap().len()
        );
        assert_eq!(
            files[2].bytes,
            std::fs::metadata(dir.join(THUMBNAIL)).unwrap().len()
        );
        assert_eq!(info.bytes, files.iter().map(|f| f.bytes).sum::<u64>());
        // The thumbnail: a JPEG, each 2 × 2 cell a pixel.
        let jpeg = store.thumbnail(&info.id).unwrap();
        let image = tetera_imaging::jpeg::decode_rgb(&jpeg).unwrap();
        assert_eq!((image.width(), image.height()), (32, 24));
        assert_eq!(store.datasets("Tonsil 3").unwrap(), [info]);
    }
}

#[test]
fn a_frame_written_with_metadata_of_the_callers_keeps_it() {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0).unwrap();
    store.create_project("p").unwrap();
    let (_pool, frames) = frames(bayer16(), 2);
    let mut w = store.create_dataset("p", spec(bayer16(), 2)).unwrap();
    let mut meta = frames[0].meta().clone();
    meta.position = Some(tetera_core::StagePos::new(
        tetera_core::Um(1.0),
        tetera_core::Um(2.0),
        tetera_core::Um(3.0),
    ));
    assert_ne!(meta, *frames[0].meta());
    w.append_as(&frames[0], meta.clone()).unwrap();
    w.append(&frames[1]).unwrap();
    let r = store.read(&w.finish().unwrap().id).unwrap();
    assert_eq!(r.read(0).unwrap(), expected(&frames[0]));
    assert_eq!(r.read(1).unwrap(), expected(&frames[1]));
    assert_eq!(r.metadata().frames, [meta, frames[1].meta().clone()]);
}

#[test]
fn a_datasets_image_is_ome_zarr_v3_with_ome_ngff_0_5s_metadata() {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0).unwrap();
    store.create_project("p").unwrap();
    let (_pool, frames) = frames(bayer16(), 4);
    let mut w = store.create_dataset("p", spec(bayer16(), 10)).unwrap();
    for f in &frames {
        w.append(f).unwrap();
    }
    // Fewer frames than planned: the array is cut to those written.
    let info = w.finish().unwrap();
    assert_eq!(info.frames, 4);
    let dir = scratch.0.join(&info.path);
    let group = read_json(&dir.join(IMAGE).join("zarr.json"));
    assert_eq!(
        (group["zarr_format"].as_u64(), group["node_type"].as_str()),
        (Some(3), Some("group"))
    );
    check_ngff_05(&group["attributes"]);
    let first = &group["attributes"]["ome"]["multiscales"][0]["datasets"][0];
    assert_eq!(
        first["coordinateTransformations"][0]["scale"],
        json!([0.05, 1.0, 0.24, 0.24])
    );
    let array = read_json(&dir.join(IMAGE).join("0").join("zarr.json"));
    assert_eq!(array["node_type"], "array");
    assert_eq!(array["shape"], json!([4, 1, 48, 64]));
    assert_eq!(array["dimension_names"], json!(["t", "c", "y", "x"]));
    assert_eq!(array["data_type"], "uint16");
    assert!(
        array["codecs"].to_string().contains("zstd"),
        "{}",
        array["codecs"]
    );
    assert_eq!(store.read(&info.id).unwrap().metadata().spec.frames, 4);
    // A single frame: no interval, so time's scale is 1 s.
    let mut one = spec(bayer16(), 1);
    one.interval_s = 0.0;
    let mut w = store.create_dataset("p", one).unwrap();
    w.append(&frames[0]).unwrap();
    let single = w.finish().unwrap();
    let group = read_json(&scratch.0.join(single.path).join(IMAGE).join("zarr.json"));
    check_ngff_05(&group["attributes"]);
    let first = &group["attributes"]["ome"]["multiscales"][0]["datasets"][0];
    assert_eq!(first["coordinateTransformations"][0]["scale"][0], 1.0);
}

#[test]
fn an_abandoned_dataset_leaves_nothing_and_a_crashs_is_cleared_when_the_store_opens() {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0).unwrap();
    store.create_project("p").unwrap();
    let (_pool, frames) = frames(bayer16(), 2);
    // Dropped half way.
    let mut w = store.create_dataset("p", spec(bayer16(), 2)).unwrap();
    w.append(&frames[0]).unwrap();
    let dir = scratch.0.join("projects").join("p").join(w.id());
    assert!(dir.join(IMAGE).exists());
    let id = w.id().to_string();
    assert_eq!(
        store.dataset(&id).unwrap_err().kind(),
        ErrorKind::Busy,
        "being written"
    );
    drop(w);
    assert!(!dir.exists(), "its folder removed");
    assert_eq!(store.dataset(&id).unwrap_err().kind(), ErrorKind::NotFound);
    assert!(store.datasets("p").unwrap().is_empty());
    // Finished with no frames: refused, and left nothing.
    let w = store.create_dataset("p", spec(bayer16(), 2)).unwrap();
    let empty = scratch.0.join("projects").join("p").join(w.id());
    assert!(w.finish().is_err());
    assert!(!empty.exists());
    // A crash: the writer never dropped. The next store clears it.
    let mut w = store.create_dataset("p", spec(bayer16(), 2)).unwrap();
    w.append(&frames[0]).unwrap();
    let crashed = scratch.0.join("projects").join("p").join(w.id());
    let crashed_id = w.id().to_string();
    std::mem::forget(w);
    drop(store);
    assert!(crashed.exists());
    let store = Store::open(&scratch.0).unwrap();
    assert!(!crashed.exists(), "cleared when it opened");
    assert_eq!(
        store.dataset(&crashed_id).unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

#[test]
fn the_index_survives_a_restart_and_a_deleted_dataset_goes() {
    let scratch = Scratch::new();
    let kept = {
        let store = Store::open(&scratch.0).unwrap();
        store.create_project("b").unwrap();
        store.create_project("a").unwrap();
        let (_pool, frames) = frames(bayer16(), 1);
        let mut ids = Vec::new();
        for _ in 0..2 {
            let mut w = store.create_dataset("a", spec(bayer16(), 1)).unwrap();
            w.append(&frames[0]).unwrap();
            ids.push(w.finish().unwrap().id);
        }
        ids
    };
    let store = Store::open(&scratch.0).unwrap();
    assert_eq!(store.root(), scratch.0);
    let names: Vec<String> = store
        .projects()
        .unwrap()
        .into_iter()
        .map(|p| p.name)
        .collect();
    assert_eq!(names, ["a", "b"], "by name");
    let listed: Vec<String> = store
        .datasets("a")
        .unwrap()
        .into_iter()
        .map(|d| d.id)
        .collect();
    assert_eq!(
        listed,
        [kept[1].clone(), kept[0].clone()],
        "the newest first"
    );
    assert!(store.datasets("b").unwrap().is_empty());
    assert_eq!(store.read(&kept[0]).unwrap().frames(), 1);
    // Deleted: its folder, and its place in the index.
    let dir = scratch.0.join(store.dataset(&kept[0]).unwrap().path);
    store.delete(&kept[0]).unwrap();
    assert!(!dir.exists());
    assert_eq!(
        store.read(&kept[0]).unwrap_err().kind(),
        ErrorKind::NotFound
    );
    assert_eq!(
        store.delete(&kept[0]).unwrap_err().kind(),
        ErrorKind::NotFound
    );
    assert_eq!(store.datasets("a").unwrap().len(), 1);
    assert!(format!("{store:?}").contains("Store"));
}

#[test]
fn bad_projects_datasets_and_frames_are_refused() {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0).unwrap();
    let long = "x".repeat(65);
    for bad in [
        "",
        ".hidden",
        " lead",
        "trail ",
        "a/b",
        "a\\b",
        long.as_str(),
    ] {
        assert_eq!(
            store.create_project(bad).unwrap_err().kind(),
            ErrorKind::InvalidArgument,
            "{bad:?}"
        );
    }
    let made = store.create_project("Lab 2.0_x-y").unwrap();
    assert_eq!(store.project("Lab 2.0_x-y").unwrap(), made);
    assert_eq!(
        store.project("none").unwrap_err().kind(),
        ErrorKind::NotFound
    );
    assert_eq!(
        store.create_project("Lab 2.0_x-y").unwrap_err().kind(),
        ErrorKind::InvalidArgument,
        "a second of its name"
    );
    assert_eq!(
        store.datasets("none").unwrap_err().kind(),
        ErrorKind::NotFound
    );
    assert_eq!(
        store
            .create_dataset("none", spec(bayer16(), 1))
            .unwrap_err()
            .kind(),
        ErrorKind::NotFound
    );
    let p = "Lab 2.0_x-y";
    let good = spec(bayer16(), 1);
    for bad in [
        DatasetSpec {
            frames: 0,
            ..good.clone()
        },
        DatasetSpec {
            width: 1,
            ..good.clone()
        },
        DatasetSpec {
            height: 1,
            ..good.clone()
        },
        DatasetSpec {
            name: String::new(),
            ..good.clone()
        },
        DatasetSpec {
            pixel_size_um: 0.0,
            ..good.clone()
        },
        DatasetSpec {
            pixel_size_um: f64::NAN,
            ..good.clone()
        },
        DatasetSpec {
            interval_s: -1.0,
            ..good.clone()
        },
        DatasetSpec {
            interval_s: f64::INFINITY,
            ..good.clone()
        },
        spec(PixelFormat::Rgb8, 1),
    ] {
        assert!(store.create_dataset(p, bad.clone()).is_err(), "{bad:?}");
    }
    assert!(store.datasets(p).unwrap().is_empty());
    // Frames that don't fit, and one too many.
    let mut w = store.create_dataset(p, spec(bayer16(), 1)).unwrap();
    let (_pool8, wrong) = frames(PixelFormat::Mono8, 1);
    assert!(w.append(&wrong[0]).is_err());
    let (_pool, frames) = frames(bayer16(), 2);
    w.append(&frames[0]).unwrap();
    assert!(w.append(&frames[1]).is_err(), "it holds one");
    let info = w.finish().unwrap();
    assert_eq!(info.frames, 1);
    // The index is in `tetera.db`, which the jobs' history shares.
    let db = scratch.0.join(crate::store::DATABASE);
    let other = rusqlite::Connection::open(&db).unwrap();
    let n: i64 = other
        .query_row("SELECT count(*) FROM datasets", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
    assert!(scratch.0.join(&info.path).join(METADATA).exists());
    assert!(scratch.0.join(&info.path).join(THUMBNAIL).exists());
}

#[test]
fn a_large_frames_thumbnail_is_shrunk_to_fit() {
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0).unwrap();
    store.create_project("p").unwrap();
    let format = PixelFormat::Mono8;
    let pool = FramePool::new(
        FrameLayout::new(format, 1_200, 900).unwrap(),
        1,
        &HostAllocator,
    )
    .unwrap();
    let mut f = pool.take(Duration::from_secs(5)).unwrap();
    f.data_mut().fill(200);
    let frame = f.freeze();
    let mut w = store
        .create_dataset(
            "p",
            DatasetSpec {
                width: 1_200,
                height: 900,
                ..spec(format, 1)
            },
        )
        .unwrap();
    w.append(&frame).unwrap();
    let info = w.finish().unwrap();
    let image = tetera_imaging::jpeg::decode_rgb(&store.thumbnail(&info.id).unwrap()).unwrap();
    // 600 × 450 cells, every third: 200 × 150, no more than the thumbnail's size.
    assert_eq!((image.width(), image.height()), (200, 150));
    assert!(image.width() <= THUMBNAIL_SIZE);
    let mean = image.data().iter().map(|&v| f64::from(v)).sum::<f64>() / image.data().len() as f64;
    let expected = (200.0f64 / 255.0).powf(1.0 / 2.2) * 255.0;
    assert!((mean - expected).abs() < 3.0, "{mean} against {expected}");
}

/// The thumbnail's pixels worked out plainly: cell by cell, colour by colour.
fn reference_thumbnail(samples: &Samples, spec: &DatasetSpec) -> (u32, u32, Vec<u8>) {
    let (w, h) = (spec.width as usize, spec.height as usize);
    let at = |x: usize, y: usize| match samples {
        Samples::U8(s) => f64::from(s[y * w + x]),
        Samples::U16(s) => f64::from(s[y * w + x]),
    };
    let (full, pattern) = match spec.format {
        PixelFormat::Bayer16 {
            pattern,
            significant_bits,
        } => (f64::from((1u32 << significant_bits) - 1), Some(pattern)),
        PixelFormat::Bayer8 { pattern } => (255.0, Some(pattern)),
        PixelFormat::Mono16 => (65_535.0, None),
        _ => (255.0, None),
    };
    let (cw, ch) = (w / 2, h / 2);
    let mut step = 1;
    while cw / step > THUMBNAIL_SIZE as usize || ch / step > THUMBNAIL_SIZE as usize {
        step += 1;
    }
    let (tw, th) = ((cw / step).max(1), (ch / step).max(1));
    let level = |v: f64| {
        let v8 = ((v / full).clamp(0.0, 1.0) * 255.0).round();
        ((v8 / 255.0).powf(1.0 / 2.2) * 255.0).round() as u8
    };
    let mut rgb = Vec::new();
    for ty in 0..th {
        for tx in 0..tw {
            let (x0, y0) = (tx * step * 2, ty * step * 2);
            let cell = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)];
            match pattern {
                Some(p) => {
                    for want in [0, 1, 2] {
                        let vals: Vec<f64> = cell
                            .iter()
                            .filter(|&&(x, y)| {
                                let c = match p.channel_at(x as u32, y as u32) {
                                    tetera_core::frame::format::Channel::Red => 0,
                                    tetera_core::frame::format::Channel::Blue => 2,
                                    _ => 1,
                                };
                                c == want
                            })
                            .map(|&(x, y)| at(x, y))
                            .collect();
                        rgb.push(level(vals.iter().sum::<f64>() / vals.len() as f64));
                    }
                }
                None => {
                    let mean = cell.iter().map(|&(x, y)| at(x, y)).sum::<f64>() / 4.0;
                    rgb.extend([level(mean); 3]);
                }
            }
        }
    }
    (tw as u32, th as u32, rgb)
}

#[test]
fn a_thumbnails_pixels_are_each_cells_colours_every_steps_cell() {
    for (format, w, h) in [
        (bayer16(), 64, 48),
        (
            PixelFormat::bayer16(BayerPattern::Bggr, 10).unwrap(),
            1_100,
            520,
        ),
        (
            PixelFormat::Bayer8 {
                pattern: BayerPattern::Grbg,
            },
            36,
            1_030,
        ),
        (PixelFormat::Mono16, 2_000, 30),
        (PixelFormat::Mono8, 2, 2),
        (PixelFormat::Mono8, 2, 2_000),
        (PixelFormat::Mono16, 3_000, 2),
    ] {
        let s = DatasetSpec {
            width: w,
            height: h,
            ..spec(format, 1)
        };
        let n = (w * h) as usize;
        let value = |i: usize| (i * 2_654_435_761 % 65_536) as u32;
        let samples = if format.bytes_per_sample() == 1 {
            Samples::U8((0..n).map(|i| value(i) as u8).collect())
        } else {
            let full = match format {
                PixelFormat::Bayer16 {
                    significant_bits, ..
                } => 1u32 << significant_bits,
                _ => 65_536,
            };
            Samples::U16((0..n).map(|i| (value(i) % full) as u16).collect())
        };
        let got = crate::dataset::thumbnail_rgb(&samples, &s).unwrap();
        assert_eq!(
            got,
            reference_thumbnail(&samples, &s),
            "{format:?} {w} × {h}"
        );
        assert!(got.0 <= THUMBNAIL_SIZE && got.1 <= THUMBNAIL_SIZE);
    }
    // The least frame: 2 × 2, one pixel.
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0).unwrap();
    store.create_project("p").unwrap();
    let least = DatasetSpec {
        width: 2,
        height: 2,
        ..spec(PixelFormat::Mono8, 1)
    };
    let pool = FramePool::new(
        FrameLayout::new(PixelFormat::Mono8, 2, 2).unwrap(),
        1,
        &HostAllocator,
    )
    .unwrap();
    let mut f = pool.take(Duration::from_secs(5)).unwrap();
    f.data_mut().fill(9);
    let mut writer = store.create_dataset("p", least).unwrap();
    writer.append(&f.freeze()).unwrap();
    assert_eq!(writer.finish().unwrap().frames, 1);
}

/// `frame`'s rows, packed.
fn packed(frame: &Frame) -> Vec<u8> {
    let l = frame.layout();
    frame
        .data()
        .chunks(l.stride)
        .take(l.height as usize)
        .flat_map(|r| r[..l.row_bytes()].to_vec())
        .collect()
}

#[test]
fn a_frame_log_reads_back_in_order_across_its_segments() {
    use crate::framelog::{FrameLog, FrameLogWriter, INDEX};
    let scratch = Scratch::new();
    for (format, width) in [
        (bayer16(), 64u32),
        (PixelFormat::Mono8, 64),
        (PixelFormat::Mono8, 50),
    ] {
        let dir = scratch
            .0
            .join(format!("log-{width}-{}", format.bytes_per_sample()));
        let mut log = FrameLogWriter::create(&dir, format, width, 48, 2).unwrap();
        assert_eq!(log.dir(), dir);
        let (_pool, frames) = frames_of(format, width, 5);
        log.append(&frames[0]).unwrap();
        // Preallocated for two frames while it is written.
        let frame_bytes = u64::from(width) * 48 * format.bytes_per_sample() as u64;
        let first = dir.join("segment-00000.bin");
        assert_eq!(std::fs::metadata(&first).unwrap().len(), 2 * frame_bytes);
        for f in &frames[1..] {
            log.append(f).unwrap();
        }
        assert_eq!(log.written(), 5);
        assert!(format!("{log:?}").contains("written"));
        assert_eq!(log.finish().unwrap(), 5);
        // Three segments, of 2, 2 and 1 frames, each cut to its frames.
        for (n, frames_in) in [(0, 2), (1, 2), (2, 1)] {
            let len = std::fs::metadata(dir.join(format!("segment-{n:05}.bin")))
                .unwrap()
                .len();
            assert_eq!(len, frames_in * frame_bytes, "segment {n}");
        }
        let read = FrameLog::open(&dir).unwrap();
        assert_eq!((read.len(), read.is_empty()), (5, false));
        assert_eq!(read.header().frame_bytes, frame_bytes);
        assert_eq!(read.header().segment_frames, 2);
        for (i, f) in frames.iter().enumerate() {
            let (meta, bytes) = read.read(i).unwrap();
            assert_eq!(meta.seq, 100 + i as u64);
            assert_eq!(bytes, packed(f), "frame {i}");
        }
        assert_eq!(read.entries()[3].segment, 1);
        assert_eq!(read.entries()[3].offset, frame_bytes);
        assert!(read.read(5).is_err());
        // A crash's torn last line is passed over; a bad line elsewhere is not.
        let index = dir.join(INDEX);
        let whole = std::fs::read_to_string(&index).unwrap();
        std::fs::write(&index, format!("{whole}{{\"segment\": 2, \"off")).unwrap();
        assert_eq!(FrameLog::open(&dir).unwrap().len(), 5);
        std::fs::write(&index, format!("not json\n{whole}")).unwrap();
        assert!(FrameLog::open(&dir).is_err());
        std::fs::write(&index, format!("{whole}not json\n")).unwrap();
        assert!(
            FrameLog::open(&dir).is_err(),
            "a whole last line that is bad"
        );
        std::fs::write(&index, &whole).unwrap();
        FrameLog::open(&dir).unwrap().remove().unwrap();
        assert!(!dir.exists());
    }
    // Refused: an existing folder, a bad shape, a frame that doesn't fit.
    let taken = scratch.0.join("taken");
    std::fs::create_dir_all(&taken).unwrap();
    assert!(FrameLogWriter::create(&taken, bayer16(), 64, 48, 2).is_err());
    let fresh = scratch.0.join("fresh");
    for (w, h, n) in [(0, 48, 2), (64, 0, 2), (64, 48, 0)] {
        assert!(FrameLogWriter::create(&fresh, bayer16(), w, h, n).is_err());
    }
    assert!(
        FrameLogWriter::create(&fresh, PixelFormat::PlanarF32 { planes: 3 }, 64, 48, 2).is_err()
    );
    let mut log = FrameLogWriter::create(&fresh, bayer16(), 32, 48, 2).unwrap();
    let (_pool, frames) = frames(bayer16(), 1);
    assert!(log.append(&frames[0]).is_err(), "64 wide in a log of 32");
    // A frame appended packed, as the recorder copies it, and one of the wrong size.
    let (_pool32, narrow) = frames_of(bayer16(), 32, 1);
    let rows = packed(&narrow[0]);
    assert!(log.append_packed(narrow[0].meta(), &rows[1..]).is_err());
    log.append_packed(narrow[0].meta(), &rows).unwrap();
    assert_eq!(log.finish().unwrap(), 1);
    assert_eq!(FrameLog::open(&fresh).unwrap().read(0).unwrap().1, rows);
    // A log of no frames.
    let fresh = scratch.0.join("empty");
    let log = FrameLogWriter::create(&fresh, bayer16(), 32, 48, 2).unwrap();
    assert_eq!(log.finish().unwrap(), 0);
    let empty = FrameLog::open(&fresh).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
}

#[test]
fn ingest_writes_a_log_into_a_dataset_and_a_stopped_one_leaves_none() {
    use crate::framelog::{FrameLog, FrameLogWriter};
    let scratch = Scratch::new();
    let store = Store::open(&scratch.0).unwrap();
    store.create_project("p").unwrap();
    let dir = scratch.0.join("scratch").join("log");
    let mut log = FrameLogWriter::create(&dir, bayer16(), 64, 48, 3).unwrap();
    let (_pool, frames) = frames(bayer16(), 4);
    for f in &frames {
        log.append(f).unwrap();
    }
    log.finish().unwrap();
    let log = FrameLog::open(&dir).unwrap();
    // The spec names it; its format, size and frames come from the log.
    let described = DatasetSpec {
        width: 1,
        height: 1,
        frames: 1,
        format: PixelFormat::Mono8,
        ..spec(bayer16(), 1)
    };
    let mut steps = Vec::new();
    let info = crate::ingest(&store, "p", &log, described.clone(), |done, of| {
        steps.push((done, of));
        Ok(())
    })
    .unwrap();
    assert_eq!(steps, [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]);
    assert_eq!(info.frames, 4);
    let r = store.read(&info.id).unwrap();
    for (t, f) in frames.iter().enumerate() {
        assert_eq!(r.read(t as u64).unwrap(), expected(f));
        assert_eq!(r.metadata().frames[t].seq, f.meta().seq);
    }
    assert_eq!(
        (r.metadata().spec.width, r.metadata().spec.format),
        (64, bayer16())
    );
    // Stopped after two frames: an error, and no dataset left.
    let before = store.datasets("p").unwrap();
    let stopped = crate::ingest(&store, "p", &log, described.clone(), |done, _| {
        if done == 2 {
            Err(tetera_core::Error::cancelled("stopped"))
        } else {
            Ok(())
        }
    });
    assert_eq!(stopped.unwrap_err().kind(), ErrorKind::Cancelled);
    assert_eq!(store.datasets("p").unwrap(), before);
    let folders = std::fs::read_dir(scratch.0.join("projects").join("p"))
        .unwrap()
        .count();
    assert_eq!(folders, 1, "only the first dataset's");
    // Stopped once all are written: no dataset either.
    let at_end = crate::ingest(&store, "p", &log, described, |done, of| {
        if done == of {
            Err(tetera_core::Error::cancelled("stopped at the end"))
        } else {
            Ok(())
        }
    });
    assert!(at_end.is_err());
    assert_eq!(store.datasets("p").unwrap(), before);
    // A frame of the wrong size, appended raw, is refused.
    let mut w = store.create_dataset("p", spec(bayer16(), 1)).unwrap();
    assert!(w.append_packed(frames[0].meta().clone(), &[0; 10]).is_err());
}
