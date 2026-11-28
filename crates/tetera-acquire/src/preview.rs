//! A preview session (P1.14a): the newest frame from the bus, cropped to its region, binned
//! and resized to its viewport, demosaicked and coloured by a live recipe (fused into one
//! kernel on the GPU), then encoded as a JPEG. Its parameters change mid-stream; a slow
//! link's hook lowers its quality or size on top of them (P1.14b); each stage is timed. No
//! transport yet: each preview goes to a callback.

use crate::bus::{FrameBus, Latest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetera_compute::Compute;
use tetera_core::frame::meta::FrameMeta;
use tetera_core::{Error, Frame, Rect, Result};
use tetera_imaging::jpeg::JpegEncoder;
use tetera_imaging::{Image, Mosaic, MosaicLayout};
use tetera_pipeline::node::Context;
use tetera_pipeline::port::Value;
use tetera_pipeline::{Plan, Recipe, Registry};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a preview session makes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct PreviewParams {
    /// The viewport's width: the widest preview, pixels. The preview keeps the frame's shape
    /// within the viewport.
    pub width: u32,
    /// The viewport's height.
    pub height: u32,
    /// The region of the frame shown, in its pixels; the whole frame when `None`.
    pub roi: Option<Rect>,
    /// The JPEG's quality, 1 to 100.
    pub quality: u8,
    /// The most previews a second; the camera's own rate when it is slower.
    pub max_rate: f64,
    /// Red's, green's and blue's gains.
    pub white_balance: [f64; 3],
}

impl Default for PreviewParams {
    /// A 1920 × 1280 viewport, quality 80, up to 60 a second, no white balance.
    fn default() -> Self {
        Self {
            width: 1_920,
            height: 1_280,
            roi: None,
            quality: 80,
            max_rate: 60.0,
            white_balance: [1.0; 3],
        }
    }
}

impl PreviewParams {
    /// Whether a session can make it.
    pub fn check(&self) -> Result<()> {
        if !(16..=8_192).contains(&self.width)
            || !(16..=8_192).contains(&self.height)
            || !(1..=100).contains(&self.quality)
            || !(self.max_rate.is_finite() && self.max_rate > 0.0)
            || self
                .white_balance
                .iter()
                .any(|g| !(g.is_finite() && *g > 0.0))
            || self.roi.is_some_and(|r| r.width < 16 || r.height < 16)
        {
            return Err(Error::invalid_argument(format!(
                "a preview of {self:?}: a viewport of 16 to 8,192 pixels each way, a quality of 1 to 100, \
                 a rate and gains above zero, a region of 16 pixels or more each way"
            )));
        }
        Ok(())
    }
}

/// A slow link's hook, on top of the parameters: a cap on the quality, and a scale on the
/// viewport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Link {
    /// The most quality the link takes, if it caps it.
    pub quality: Option<u8>,
    /// The viewport's scale, above 0 and at most 1.
    pub scale: f64,
}

impl Default for Link {
    /// No cap, the whole viewport.
    fn default() -> Self {
        Self {
            quality: None,
            scale: 1.0,
        }
    }
}

/// How long each stage took, for one preview.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StageTimes {
    /// The frame to a mosaic, cropped: ms.
    pub convert: f64,
    /// The live recipe: binned, demosaicked, resized and coloured, with its copies: ms.
    pub isp: f64,
    /// The JPEG: ms.
    pub jpeg: f64,
    /// All of it, the frame taken to the JPEG made: ms.
    pub total: f64,
}

/// A preview.
#[derive(Debug, Clone)]
pub struct Preview {
    /// The frame's metadata.
    pub meta: FrameMeta,
    /// The preview's width, pixels.
    pub width: u32,
    /// Its height.
    pub height: u32,
    /// The quality it was encoded at.
    pub quality: u8,
    /// The JPEG.
    pub jpeg: Vec<u8>,
    /// How long each stage took.
    pub times: StageTimes,
}

/// What a session has done.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SessionStats {
    /// Previews made.
    pub previews: u64,
    /// Frames replaced by newer ones before the session took them: while it worked, or
    /// waited to keep to its maximum rate.
    pub replaced: u64,
    /// Frames that failed, and why the last did.
    pub failed: u64,
    /// Previews a second over the last second.
    pub rate: f64,
    /// Each stage's mean time over the last second.
    pub mean: StageTimes,
}

struct Shared {
    params: Mutex<(PreviewParams, Link)>,
    stats: Mutex<(SessionStats, VecDeque<(Instant, StageTimes)>)>,
    error: Mutex<Option<String>>,
    stop: AtomicBool,
}

/// A preview session.
pub struct Session {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("params", &self.params())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// The preview's size: the region's shape, as large as fits the viewport scaled by `scale`,
/// at least 16 pixels each way.
pub fn fit(region: (u32, u32), viewport: (u32, u32), scale: f64) -> (u32, u32) {
    let (vw, vh) = (
        ((f64::from(viewport.0) * scale) as u32).max(16),
        ((f64::from(viewport.1) * scale) as u32).max(16),
    );
    let aspect = f64::from(region.0) / f64::from(region.1);
    let (w, h) = if f64::from(vw) / f64::from(vh) > aspect {
        ((f64::from(vh) * aspect).round() as u32, vh)
    } else {
        (vw, (f64::from(vw) / aspect).round() as u32)
    };
    (w.clamp(16, region.0.max(16)), h.clamp(16, region.1.max(16)))
}

/// The largest binning, 1 to 4, that leaves the region at least the preview's size.
pub fn binning(region: (u32, u32), preview: (u32, u32)) -> u32 {
    (1..=4)
        .rev()
        .find(|b| region.0 / b >= preview.0 && region.1 / b >= preview.1)
        .unwrap_or(1)
}

/// The live recipe for a mosaic of `raw`, black at `black`, to a `size` preview balanced by
/// `white_balance`: `[bin →] demosaic → resize → colour`, the chain the pipeline fuses.
pub fn recipe(raw: &MosaicLayout, black: u16, size: (u32, u32), white_balance: [f64; 3]) -> Recipe {
    let full = ((1u32 << raw.significant_bits) - 1) as u16;
    let factor = binning((raw.width, raw.height), size);
    let mut nodes = Vec::new();
    let mut source = "raw".to_string();
    if factor > 1 {
        nodes.push(serde_json::json!({
            "id": "binned", "node": "tetera.bin",
            "params": {"factor": factor, "mode": "mean"},
            "inputs": {"raw": source},
        }));
        source = "binned.raw".into();
    }
    nodes.push(serde_json::json!({
        "id": "rgb", "node": "tetera.demosaic",
        "params": {"method": "malvar_he_cutler"},
        "inputs": {"raw": source},
    }));
    nodes.push(serde_json::json!({
        "id": "small", "node": "tetera.resize",
        "params": {"width": size.0, "height": size.1},
        "inputs": {"image": "rgb.image"},
    }));
    let level = serde_json::json!({"black": black.min(full - 1), "white": full});
    nodes.push(serde_json::json!({
        "id": "colour", "node": "tetera.colour",
        "params": {"white_balance": white_balance, "levels": [level, level, level], "curve": "srgb"},
        "inputs": {"image": "small.image"},
    }));
    let json = serde_json::json!({
        "name": "live preview",
        "summary": "A raw frame to a preview: binned, demosaicked, resized and coloured.",
        "inputs": {"raw": {"kind": "raw"}},
        "nodes": nodes,
        "outputs": {"image": "colour.image"},
    });
    Recipe::from_json(&json.to_string()).expect("the live recipe is well formed")
}

/// What the plan is made for: when any of it changes, it is made again.
#[derive(Debug, Clone, PartialEq)]
struct Made {
    raw: MosaicLayout,
    black: u16,
    size: (u32, u32),
    white_balance: [f64; 3],
}

impl Session {
    /// A session on `bus`'s frames, making previews as `params` says on `compute`, each handed
    /// to `deliver` on the session's thread.
    pub fn start(
        bus: &FrameBus,
        compute: Arc<Compute>,
        params: PreviewParams,
        deliver: impl FnMut(Preview) + Send + 'static,
    ) -> Result<Self> {
        params.check()?;
        let latest = bus.latest();
        let shared = Arc::new(Shared {
            params: Mutex::new((params, Link::default())),
            stats: Mutex::new((SessionStats::default(), VecDeque::new())),
            error: Mutex::new(None),
            stop: AtomicBool::new(false),
        });
        let running = shared.clone();
        let thread = std::thread::Builder::new()
            .name("preview session".into())
            .spawn(move || run(&running, &latest, &compute, deliver))
            .map_err(|e| Error::internal(format!("the preview session won't start: {e}")))?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// Its parameters.
    pub fn params(&self) -> PreviewParams {
        lock(&self.shared.params).0.clone()
    }

    /// Changes its parameters, from the next frame.
    pub fn set_params(&self, params: PreviewParams) -> Result<()> {
        params.check()?;
        lock(&self.shared.params).0 = params;
        Ok(())
    }

    /// Sets the slow link's hook, from the next frame.
    pub fn set_link(&self, link: Link) -> Result<()> {
        if !(link.scale > 0.0 && link.scale <= 1.0)
            || link.quality.is_some_and(|q| !(1..=100).contains(&q))
        {
            return Err(Error::invalid_argument(format!(
                "a link's scale is above 0 and at most 1, and its quality 1 to 100: {link:?}"
            )));
        }
        lock(&self.shared.params).1 = link;
        Ok(())
    }

    /// What it has done.
    pub fn stats(&self) -> SessionStats {
        lock(&self.shared.stats).0
    }

    /// Why its last frame failed, if one did.
    pub fn last_error(&self) -> Option<String> {
        lock(&self.shared.error).clone()
    }

    /// Stops: nothing more is delivered once it returns.
    pub fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.halt();
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The last second's previews, with one whose stages took `times` made at `now`: the rate,
/// previews a second, and each stage's mean time.
fn window(
    recent: &mut VecDeque<(Instant, StageTimes)>,
    now: Instant,
    times: StageTimes,
) -> (f64, StageTimes) {
    recent.push_back((now, times));
    while recent
        .front()
        .is_some_and(|(t, _)| now - *t > Duration::from_secs(1))
    {
        recent.pop_front();
    }
    let n = recent.len() as f64;
    let mean = |f: fn(&StageTimes) -> f64| recent.iter().map(|(_, t)| f(t)).sum::<f64>() / n;
    let means = StageTimes {
        convert: mean(|t| t.convert),
        isp: mean(|t| t.isp),
        jpeg: mean(|t| t.jpeg),
        total: mean(|t| t.total),
    };
    let span = recent
        .front()
        .map_or(0.0, |(t, _)| (now - *t).as_secs_f64());
    (if span > 0.0 { (n - 1.0) / span } else { 0.0 }, means)
}

/// The session's thread: the newest frame, at most at the maximum rate, to a preview.
fn run(shared: &Shared, latest: &Latest, compute: &Compute, mut deliver: impl FnMut(Preview)) {
    let registry = Registry::stock();
    let mut plan: Option<(Made, Plan)> = None;
    let mut encoder: Option<JpegEncoder> = None;
    let mut last: Option<Instant> = None;
    while !shared.stop.load(Ordering::Acquire) {
        // Keeping to the maximum rate: wait, then take the newest frame, so the frames that
        // came meanwhile give way to it, and the last one is never lost.
        let rate = lock(&shared.params).0.max_rate;
        if let Some(t) = last {
            let wait = Duration::from_secs_f64(1.0 / rate).saturating_sub(t.elapsed());
            if !wait.is_zero() {
                std::thread::sleep(wait.min(Duration::from_millis(20)));
                continue;
            }
        }
        let Some(frame) = latest.recv(Duration::from_millis(20)) else {
            continue;
        };
        let (params, link) = lock(&shared.params).clone();
        last = Some(Instant::now());
        match preview(
            &frame,
            &params,
            &link,
            compute,
            &registry,
            &mut plan,
            &mut encoder,
        ) {
            Ok(made) => {
                drop(frame);
                {
                    let mut stats = lock(&shared.stats);
                    let (rate, mean) = window(&mut stats.1, Instant::now(), made.times);
                    stats.0.previews += 1;
                    stats.0.replaced = latest.replaced();
                    stats.0.mean = mean;
                    stats.0.rate = rate;
                }
                deliver(made);
            }
            Err(e) => {
                lock(&shared.stats).0.failed += 1;
                *lock(&shared.error) = Some(e.to_string());
            }
        }
    }
}

/// One frame to a preview.
fn preview(
    frame: &Frame,
    params: &PreviewParams,
    link: &Link,
    compute: &Compute,
    registry: &Registry,
    plan: &mut Option<(Made, Plan)>,
    encoder: &mut Option<JpegEncoder>,
) -> Result<Preview> {
    let start = Instant::now();
    let mut mosaic = Mosaic::from_frame(frame)?;
    if let Some(roi) = params.roi {
        mosaic = mosaic.crop(roi)?;
    }
    let convert = start.elapsed();
    let raw = *mosaic.layout();
    let made = Made {
        raw,
        black: frame.meta().black_level,
        size: fit(
            (raw.width, raw.height),
            (params.width, params.height),
            link.scale,
        ),
        white_balance: params.white_balance,
    };
    if plan.as_ref().is_none_or(|(m, _)| *m != made) {
        let recipe = recipe(&made.raw, made.black, made.size, made.white_balance);
        *plan = Some((made.clone(), Plan::new(&recipe, registry)?));
    }
    let (_, live) = plan.as_mut().expect("the plan made above");
    let isp_start = Instant::now();
    let mut outputs = live.run(
        &Context { compute },
        BTreeMap::from([("raw".to_string(), Value::Raw(mosaic))]),
    )?;
    let image: Image<u8> = match outputs.remove("image") {
        Some(Value::U8(image)) => image,
        other => {
            return Err(Error::internal(format!(
                "the live recipe gave {:?}, not an 8-bit image",
                other.map(|v| v.port_type())
            )));
        }
    };
    let isp = isp_start.elapsed();
    let quality = link
        .quality
        .map_or(params.quality, |q| q.min(params.quality));
    if encoder.as_ref().is_none_or(|e| e.quality() != quality) {
        *encoder = Some(JpegEncoder::new(quality)?);
    }
    let encoder = encoder.as_mut().expect("made above");
    let jpeg_start = Instant::now();
    let jpeg = encoder.encode(&image)?;
    let jpeg_time = jpeg_start.elapsed();
    Ok(Preview {
        meta: frame.meta().clone(),
        width: image.width(),
        height: image.height(),
        quality: encoder.quality(),
        jpeg,
        times: StageTimes {
            convert: ms(convert),
            isp: ms(isp),
            jpeg: ms(jpeg_time),
            total: ms(start.elapsed()),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
    use tetera_core::frame::format::BayerPattern;
    use tetera_core::frame::pool::HostAllocator;
    use tetera_core::{FrameLayout, FramePool, PixelFormat, Timestamp};

    /// A pool of 256 × 192 RGGB frames in 12 bits.
    fn pool() -> FramePool {
        let format = PixelFormat::bayer16(BayerPattern::Rggb, 12).unwrap();
        let layout = FrameLayout::new(format, 256, 192).unwrap();
        FramePool::new(layout, 8, &HostAllocator).unwrap()
    }

    /// Frame `seq`: every site `value`, the black level 100.
    fn frame(pool: &FramePool, seq: u64, value: u16) -> Frame {
        let mut f = pool.take(Duration::from_secs(5)).unwrap();
        let stride = f.layout().stride;
        for row in f.data_mut().chunks_mut(stride) {
            for pair in row[..256 * 2].as_chunks_mut::<2>().0 {
                *pair = value.to_le_bytes();
            }
        }
        let meta = f.meta_mut();
        meta.seq = seq;
        meta.exposure_start = Timestamp(seq);
        meta.black_level = 100;
        f.freeze()
    }

    /// A session on `bus` on the CPU, its previews down a channel.
    fn session(bus: &FrameBus, params: PreviewParams) -> (Session, Receiver<Preview>) {
        let (tx, rx) = channel();
        let compute = Arc::new(Compute::cpu_only().unwrap());
        let s = Session::start(bus, compute, params, move |p| {
            let _ = tx.send(p);
        })
        .unwrap();
        (s, rx)
    }

    fn next(rx: &Receiver<Preview>) -> Preview {
        rx.recv_timeout(Duration::from_secs(10)).expect("a preview")
    }

    fn next_or(rx: &Receiver<Preview>, s: &Session) -> Preview {
        rx.recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|_| panic!("no preview: {:?}", s.last_error()))
    }

    #[test]
    fn a_preview_keeps_the_frames_shape_within_its_viewport_and_bins_down_to_it() {
        assert_eq!(fit((2_736, 1_824), (1_920, 1_280), 1.0), (1_920, 1_280));
        assert_eq!(fit((2_736, 1_824), (1_920, 1_920), 1.0), (1_920, 1_280));
        assert_eq!(fit((2_736, 1_824), (800, 400), 1.0), (600, 400));
        assert_eq!(fit((2_736, 1_824), (1_920, 1_280), 0.5), (960, 640));
        assert_eq!(
            fit((100, 50), (1_920, 1_280), 1.0),
            (100, 50),
            "never larger than the region"
        );
        assert_eq!(
            fit((5_000, 10), (64, 64), 1.0),
            (64, 16),
            "at least 16 each way"
        );
        assert_eq!(binning((2_736, 1_824), (1_920, 1_280)), 1);
        assert_eq!(binning((2_736, 1_824), (640, 427)), 4);
        assert_eq!(binning((2_736, 1_824), (960, 640)), 2);
        assert_eq!(binning((5_472, 3_648), (1_920, 1_280)), 2);
        // Each way on its own: a width that would bin by 4, a height that won't.
        assert_eq!(binning((2_000, 1_000), (500, 900)), 1);
        assert_eq!(binning((1_000, 2_000), (900, 500)), 1);
        assert_eq!(
            fit((2_736, 1_824), (1_920, 1_000), 0.5),
            (750, 500),
            "the height binds"
        );
        let raw = MosaicLayout {
            width: 2_736,
            height: 1_824,
            pattern: BayerPattern::Rggb,
            significant_bits: 8,
        };
        let r = recipe(&raw, 6, (640, 427), [1.5, 1.0, 2.0]);
        let json = r.to_json().replace([' ', '\n'], "");
        assert!(
            json.contains("tetera.bin") && json.contains("\"factor\":4"),
            "{json}"
        );
        assert!(
            json.contains("\"white\":255") && json.contains("\"black\":6"),
            "{json}"
        );
        assert!(
            !recipe(&raw, 6, (2_736, 1_824), [1.0; 3])
                .to_json()
                .contains("tetera.bin")
        );
    }

    #[test]
    fn a_session_makes_previews_of_its_viewports_size_and_times_each_stage() {
        let bus = FrameBus::new();
        let pool = pool();
        let (s, rx) = session(
            &bus,
            PreviewParams {
                width: 128,
                height: 128,
                ..PreviewParams::default()
            },
        );
        bus.publish(frame(&pool, 1, 2_000));
        let p = next(&rx);
        assert_eq!((p.width, p.height, p.meta.seq, p.quality), (128, 96, 1, 80));
        let decoded = tetera_imaging::jpeg::decode_rgb(&p.jpeg).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (128, 96));
        assert!(p.times.isp > 0.0 && p.times.jpeg > 0.0 && p.times.convert > 0.0);
        assert!(p.times.total >= p.times.isp + p.times.jpeg);
        assert!(
            p.times.total > 0.05 && p.times.total < 60_000.0,
            "milliseconds: {:?}",
            p.times
        );
        // An even grey: (2000 − 100) / (4095 − 100) of white in each colour, in sRGB.
        let level = (2_000.0 - 100.0) / (4_095.0 - 100.0);
        let srgb = 1.055 * f64::powf(level, 1.0 / 2.4) - 0.055;
        let mean =
            decoded.data().iter().map(|&v| f64::from(v)).sum::<f64>() / decoded.data().len() as f64;
        assert!(
            (mean - srgb * 255.0).abs() < 3.0,
            "{mean} against {}",
            srgb * 255.0
        );
        let stats = s.stats();
        assert_eq!((stats.previews, stats.failed), (1, 0));
        assert!(stats.mean.total > 0.0);
        assert!(format!("{s:?}").contains("Session"));
        assert_eq!(
            s.params(),
            PreviewParams {
                width: 128,
                height: 128,
                ..PreviewParams::default()
            }
        );
        s.stop();
        bus.publish(frame(&pool, 2, 2_000));
        assert!(
            matches!(
                rx.recv_timeout(Duration::from_millis(500)),
                Err(RecvTimeoutError::Disconnected)
            ),
            "nothing once stopped"
        );
    }

    #[test]
    fn its_parameters_and_its_link_change_mid_stream() {
        let bus = FrameBus::new();
        let pool = pool();
        let (s, rx) = session(&bus, PreviewParams::default());
        bus.publish(frame(&pool, 1, 1_000));
        assert_eq!(
            (next_or(&rx, &s).width, s.params().width),
            (256, 1_920),
            "the whole frame"
        );
        assert_eq!(s.last_error(), None, "none failed yet");
        s.set_params(PreviewParams {
            width: 64,
            height: 64,
            roi: Some(Rect::new(32, 16, 128, 64)),
            quality: 50,
            white_balance: [2.0, 1.0, 1.0],
            ..PreviewParams::default()
        })
        .unwrap();
        bus.publish(frame(&pool, 2, 1_000));
        let p = next_or(&rx, &s);
        assert_eq!(
            (p.width, p.height, p.quality),
            (64, 32, 50),
            "the region, fitted"
        );
        let decoded = tetera_imaging::jpeg::decode_rgb(&p.jpeg).unwrap();
        let red: f64 = decoded
            .data()
            .iter()
            .step_by(3)
            .map(|&v| f64::from(v))
            .sum();
        let green: f64 = decoded
            .data()
            .iter()
            .skip(1)
            .step_by(3)
            .map(|&v| f64::from(v))
            .sum();
        assert!(red > green * 1.2, "red's gain: {red} against {green}");
        s.set_link(Link {
            quality: Some(20),
            scale: 0.5,
        })
        .unwrap();
        bus.publish(frame(&pool, 3, 1_000));
        let p = next_or(&rx, &s);
        assert_eq!(
            (p.width, p.height, p.quality),
            (32, 16, 20),
            "the link's cap and scale"
        );
        for bad in [
            Link {
                quality: Some(0),
                scale: 1.0,
            },
            Link {
                quality: None,
                scale: 0.0,
            },
            Link {
                quality: None,
                scale: 1.5,
            },
        ] {
            assert!(s.set_link(bad).is_err(), "{bad:?}");
        }
        assert!(
            s.set_params(PreviewParams {
                roi: Some(Rect::new(250, 0, 64, 64)),
                ..PreviewParams::default()
            })
            .is_ok(),
            "checked against the frame when it comes"
        );
        bus.publish(frame(&pool, 4, 1_000));
        for _ in 0..200 {
            if s.stats().failed > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(s.stats().failed, 1, "a region outside the frame");
        assert!(s.last_error().is_some());
        drop(s);
        assert!(
            matches!(
                rx.recv_timeout(Duration::from_millis(500)),
                Err(RecvTimeoutError::Disconnected)
            ),
            "a session dropped stops"
        );
    }

    #[test]
    fn a_session_keeps_to_its_maximum_rate_and_bad_parameters_are_refused() {
        let bus = FrameBus::new();
        let pool = pool();
        let (s, rx) = session(
            &bus,
            PreviewParams {
                width: 64,
                height: 64,
                max_rate: 5.0,
                ..PreviewParams::default()
            },
        );
        for seq in 0..30 {
            bus.publish(frame(&pool, seq, 1_000));
            std::thread::sleep(Duration::from_millis(10));
        }
        // The last frame is shown, however soon after the one before it came.
        let mut made: Vec<Preview> = Vec::new();
        while made.last().is_none_or(|p| p.meta.seq < 29) {
            made.push(next(&rx));
        }
        assert!(
            made.len() <= 3,
            "at most 5 a second over 0.3 s: {}",
            made.len()
        );
        assert!(s.stats().replaced >= 25, "{:?}", s.stats());
        s.stop();
        for bad in [
            PreviewParams {
                width: 8,
                ..PreviewParams::default()
            },
            PreviewParams {
                quality: 0,
                ..PreviewParams::default()
            },
            PreviewParams {
                max_rate: 0.0,
                ..PreviewParams::default()
            },
            PreviewParams {
                white_balance: [1.0, f64::NAN, 1.0],
                ..PreviewParams::default()
            },
            PreviewParams {
                white_balance: [1.0, 0.0, 1.0],
                ..PreviewParams::default()
            },
            PreviewParams {
                white_balance: [f64::INFINITY, 1.0, 1.0],
                ..PreviewParams::default()
            },
            PreviewParams {
                roi: Some(Rect::new(0, 0, 8, 64)),
                ..PreviewParams::default()
            },
            PreviewParams {
                roi: Some(Rect::new(0, 0, 64, 8)),
                ..PreviewParams::default()
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
            assert!(
                Session::start(&bus, Arc::new(Compute::cpu_only().unwrap()), bad, |_| {}).is_err()
            );
        }
        assert!(
            PreviewParams {
                roi: Some(Rect::new(0, 0, 16, 16)),
                ..PreviewParams::default()
            }
            .check()
            .is_ok(),
            "a region of 16 × 16, the least"
        );
    }

    #[test]
    fn a_session_previews_each_frame_that_comes_slower_than_its_maximum_rate() {
        let bus = FrameBus::new();
        let pool = pool();
        let (s, rx) = session(
            &bus,
            PreviewParams {
                width: 64,
                height: 64,
                ..PreviewParams::default()
            },
        );
        for seq in 0..10 {
            bus.publish(frame(&pool, seq, 1_000));
            std::thread::sleep(Duration::from_millis(50));
        }
        let mut made: Vec<Preview> = Vec::new();
        while made.last().is_none_or(|p| p.meta.seq < 9) {
            made.push(next(&rx));
        }
        assert!(
            made.len() >= 6,
            "up to 60 a second, so most of 20 a second: {}",
            made.len()
        );
        s.stop();
    }

    #[test]
    fn the_rate_and_the_mean_times_are_over_the_last_second() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let times = |total: f64| StageTimes {
            convert: total / 4.0,
            isp: total / 2.0,
            jpeg: total / 8.0,
            total,
        };
        let mut recent = VecDeque::new();
        assert_eq!(
            window(&mut recent, at(0), times(8.0)),
            (0.0, times(8.0)),
            "no rate from one"
        );
        assert_eq!(
            window(&mut recent, at(500), times(16.0)),
            (2.0, times(12.0))
        );
        // One made a second ago is still in the last second.
        assert_eq!(
            window(&mut recent, at(1_000), times(24.0)),
            (2.0, times(16.0))
        );
        let (rate, mean) = window(&mut recent, at(1_600), times(32.0));
        assert_eq!(recent.len(), 2, "those more than a second old let go");
        assert!((rate - 1.0 / 0.6).abs() < 1e-9, "{rate}");
        assert_eq!(mean, times(28.0));
    }
}
