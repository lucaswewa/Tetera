//! Live statistics (P1.14a), from the raw frame, at up to 10 Hz: a histogram for each
//! colour, the saturation mask at the preview's size, line profiles through a row and a
//! column, the statistics of a region, and a focus value. [`statistics`] works them out from
//! a mosaic; [`LiveStats`] does it on a thread of its own, from a latest-wins subscriber.

use crate::bus::{FrameBus, Latest};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetera_core::frame::format::Channel;
use tetera_core::{Error, Rect, Result, Timestamp};
use tetera_imaging::Mosaic;

/// What the statistics take.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct StatsParams {
    /// Bins in each histogram, from 0 to the full scale.
    pub bins: usize,
    /// The saturation mask's size, as the preview's: cells across and down.
    pub mask: [u32; 2],
    /// The region whose statistics are worked out, in the frame's pixels; the whole frame
    /// when `None`.
    pub roi: Option<Rect>,
    /// The row of the row profile; the middle when `None`.
    pub row: Option<u32>,
    /// The column of the column profile; the middle when `None`.
    pub column: Option<u32>,
}

impl Default for StatsParams {
    fn default() -> Self {
        Self {
            bins: 256,
            mask: [192, 128],
            roi: None,
            row: None,
            column: None,
        }
    }
}

/// A colour's statistics in a region.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChannelStats {
    /// The mean count.
    pub mean: f64,
    /// The standard deviation.
    pub std: f64,
    /// The least count.
    pub min: u16,
    /// The greatest.
    pub max: u16,
    /// The sites counted.
    pub count: u64,
}

/// A frame's live statistics. Each array is red's, green's and blue's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LiveStatistics {
    /// The frame's number.
    pub seq: u64,
    /// When its exposure started.
    pub exposure_start: Timestamp,
    /// The full scale: the greatest count its bits hold.
    pub full_scale: u16,
    /// Each colour's histogram, its bins evenly from 0 to the full scale.
    pub histogram: [Vec<u64>; 3],
    /// The saturation mask's size, cells across and down.
    pub mask: [u32; 2],
    /// The mask, row by row: a cell is saturated when a site in it is at the full scale.
    pub saturated: Vec<bool>,
    /// The share of sites at the full scale.
    pub saturated_fraction: f64,
    /// Each colour along the profile's row, a value for each 2 × 2 cell.
    pub row_profile: [Vec<f32>; 3],
    /// Each colour down the profile's column.
    pub column_profile: [Vec<f32>; 3],
    /// Each colour's statistics in the region.
    pub roi: [ChannelStats; 3],
    /// Green's mean squared difference between sites two apart along a row (Brenner's
    /// measure), over the full scale squared: larger when sharper.
    pub focus: f64,
}

/// Red, green or blue, as 0, 1 or 2.
fn colour(channel: Channel) -> usize {
    match channel {
        Channel::Red => 0,
        Channel::GreenOnRed | Channel::GreenOnBlue => 1,
        Channel::Blue => 2,
    }
}

/// What one band of rows adds up to.
#[derive(Clone)]
struct Sums {
    histogram: [Vec<u64>; 3],
    saturated: Vec<bool>,
    at_full: u64,
    focus: (f64, u64),
}

impl Sums {
    fn new(bins: usize, cells: usize) -> Self {
        Self {
            histogram: std::array::from_fn(|_| vec![0; bins]),
            saturated: vec![false; cells],
            at_full: 0,
            focus: (0.0, 0),
        }
    }

    fn add(mut self, other: Sums) -> Sums {
        for (a, b) in self.histogram.iter_mut().zip(&other.histogram) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        for (a, b) in self.saturated.iter_mut().zip(&other.saturated) {
            *a |= b;
        }
        self.at_full += other.at_full;
        self.focus = (self.focus.0 + other.focus.0, self.focus.1 + other.focus.1);
        self
    }
}

/// `mosaic`'s live statistics, frame `seq` exposed from `exposure_start`, as `params` asks.
pub fn statistics(
    mosaic: &Mosaic,
    seq: u64,
    exposure_start: Timestamp,
    params: &StatsParams,
) -> Result<LiveStatistics> {
    let layout = *mosaic.layout();
    let (w, h) = (layout.width, layout.height);
    let [mw, mh] = params.mask;
    if params.bins == 0 || mw == 0 || mh == 0 || w < 4 || h < 4 {
        return Err(Error::invalid_argument(format!(
            "statistics of a {w} × {h} mosaic in {} bins, a {mw} × {mh} mask",
            params.bins
        )));
    }
    let roi = params.roi.unwrap_or(Rect::new(0, 0, w, h));
    if roi.is_empty() || !Rect::new(0, 0, w, h).contains_rect(&roi) {
        return Err(Error::invalid_argument(format!(
            "the region {roi:?} isn't inside the {w} × {h} frame"
        )));
    }
    let full = ((1u32 << layout.significant_bits) - 1) as u16;
    let sites = mosaic.sites();
    let pattern = layout.pattern;
    let bins = params.bins;
    let bin = |v: u16| ((usize::from(v.min(full)) * bins) / (usize::from(full) + 1)).min(bins - 1);
    let cells = (mw * mh) as usize;
    let sums = (0..h)
        .into_par_iter()
        .fold(
            || Sums::new(bins, cells),
            |mut s, y| {
                let row = &sites[(y * w) as usize..][..w as usize];
                let my = (u64::from(y) * u64::from(mh) / u64::from(h)) as usize;
                let green_from = if colour(pattern.channel_at(0, y)) == 1 {
                    0
                } else {
                    1
                };
                for (x, &v) in row.iter().enumerate() {
                    let c = colour(pattern.channel_at(x as u32, y));
                    s.histogram[c][bin(v)] += 1;
                    if v >= full {
                        s.at_full += 1;
                        let mx = (x as u64 * u64::from(mw) / u64::from(w)) as usize;
                        s.saturated[my * mw as usize + mx] = true;
                    }
                }
                let mut x = green_from;
                while x + 2 < w as usize {
                    let d = f64::from(row[x + 2]) - f64::from(row[x]);
                    s.focus.0 += d * d;
                    s.focus.1 += 1;
                    x += 2;
                }
                s
            },
        )
        .reduce(|| Sums::new(bins, cells), Sums::add);
    let mut stats = [(0.0f64, 0.0f64, u16::MAX, 0u16, 0u64); 3];
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let v = sites[(y * w + x) as usize];
            let s = &mut stats[colour(pattern.channel_at(x, y))];
            let f = f64::from(v);
            s.0 += f;
            s.1 += f * f;
            s.2 = s.2.min(v);
            s.3 = s.3.max(v);
            s.4 += 1;
        }
    }
    let roi_stats = stats.map(|(sum, squares, min, max, count)| {
        let n = count.max(1) as f64;
        let mean = sum / n;
        ChannelStats {
            mean,
            std: (squares / n - mean * mean).max(0.0).sqrt(),
            min: if count == 0 { 0 } else { min },
            max,
            count,
        }
    });
    // A 2 × 2 cell's colours: red, the mean of its greens, blue.
    let cell = |cx: u32, cy: u32| -> [f32; 3] {
        let mut sum = [0.0f32; 3];
        let mut n = [0u32; 3];
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (x, y) = (2 * cx + dx, 2 * cy + dy);
            let c = colour(pattern.channel_at(x, y));
            sum[c] += f32::from(sites[(y * w + x) as usize]);
            n[c] += 1;
        }
        std::array::from_fn(|c| sum[c] / n[c].max(1) as f32)
    };
    let (cw, ch) = (w / 2, h / 2);
    let row = params.row.unwrap_or(h / 2).min(h - 1) / 2;
    let column = params.column.unwrap_or(w / 2).min(w - 1) / 2;
    let profile = |values: Vec<[f32; 3]>| -> [Vec<f32>; 3] {
        std::array::from_fn(|c| values.iter().map(|v| v[c]).collect())
    };
    Ok(LiveStatistics {
        seq,
        exposure_start,
        full_scale: full,
        histogram: sums.histogram,
        mask: params.mask,
        saturated: sums.saturated,
        saturated_fraction: sums.at_full as f64 / f64::from(w) / f64::from(h),
        row_profile: profile((0..cw).map(|cx| cell(cx, row)).collect()),
        column_profile: profile((0..ch).map(|cy| cell(column, cy)).collect()),
        roi: roi_stats,
        focus: sums.focus.0 / sums.focus.1.max(1) as f64 / (f64::from(full) * f64::from(full)),
    })
}

/// The statistics worked out live, on a thread of their own.
pub struct LiveStats {
    params: Arc<Mutex<StatsParams>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for LiveStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveStats").finish_non_exhaustive()
    }
}

/// The fastest the statistics are worked out: 10 a second.
pub const FASTEST: Duration = Duration::from_millis(100);

impl LiveStats {
    /// Statistics of `bus`'s frames, as `params` asks, at most one every `every` (no faster
    /// than [`FASTEST`]), each handed to `deliver`. A frame that comes sooner is passed over.
    pub fn start(
        bus: &FrameBus,
        every: Duration,
        params: StatsParams,
        mut deliver: impl FnMut(Result<LiveStatistics>) + Send + 'static,
    ) -> Result<Self> {
        let latest: Latest = bus.latest();
        let every = every.max(FASTEST);
        let shared = Arc::new(Mutex::new(params));
        let stop = Arc::new(AtomicBool::new(false));
        let (params, flag) = (shared.clone(), stop.clone());
        let thread = std::thread::Builder::new()
            .name("live statistics".into())
            .spawn(move || {
                let mut last: Option<Instant> = None;
                while !flag.load(Ordering::Acquire) {
                    let Some(frame) = latest.recv(Duration::from_millis(20)) else {
                        continue;
                    };
                    if last.is_some_and(|t| t.elapsed() < every) {
                        continue;
                    }
                    last = Some(Instant::now());
                    let params = params
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .clone();
                    let meta = frame.meta().clone();
                    let worked = Mosaic::from_frame(&frame).and_then(|m| {
                        drop(frame);
                        statistics(&m, meta.seq, meta.exposure_start, &params)
                    });
                    deliver(worked);
                }
            })
            .map_err(|e| Error::internal(format!("the statistics' thread won't start: {e}")))?;
        Ok(Self {
            params: shared,
            stop,
            thread: Some(thread),
        })
    }

    /// Changes what is worked out, from the next frame.
    pub fn set_params(&self, params: StatsParams) {
        *self.params.lock().unwrap_or_else(PoisonError::into_inner) = params;
    }

    /// Stops: nothing more is delivered once it returns.
    pub fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for LiveStats {
    fn drop(&mut self) {
        self.halt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetera_core::frame::format::BayerPattern;
    use tetera_imaging::MosaicLayout;

    /// A mosaic of `w` × `h` sites in `bits`, each from `f`.
    fn mosaic(
        w: u32,
        h: u32,
        bits: u8,
        pattern: BayerPattern,
        f: impl Fn(u32, u32) -> u16,
    ) -> Mosaic {
        let layout = MosaicLayout {
            width: w,
            height: h,
            pattern,
            significant_bits: bits,
        };
        let sites = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect();
        Mosaic::new(layout, sites).unwrap()
    }

    /// The statistics again, one site at a time, as plainly as they are defined.
    fn reference(m: &Mosaic, p: &StatsParams) -> LiveStatistics {
        let l = *m.layout();
        let (w, h) = (l.width, l.height);
        let full = ((1u32 << l.significant_bits) - 1) as u16;
        let at = |x: u32, y: u32| m.sites()[(y * w + x) as usize];
        let c = |x: u32, y: u32| colour(l.pattern.channel_at(x, y));
        let mut histogram: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0; p.bins]);
        let [mw, mh] = p.mask;
        let mut saturated = vec![false; (mw * mh) as usize];
        let mut at_full = 0;
        let (mut focus, mut pairs) = (0.0, 0u64);
        for y in 0..h {
            for x in 0..w {
                let v = at(x, y);
                let b = (f64::from(v) / (f64::from(full) + 1.0) * p.bins as f64) as usize;
                histogram[c(x, y)][b.min(p.bins - 1)] += 1;
                if v >= full {
                    at_full += 1;
                    let cell = (y * mh / h) * mw + x * mw / w;
                    saturated[cell as usize] = true;
                }
                if c(x, y) == 1 && x + 2 < w {
                    let d = f64::from(at(x + 2, y)) - f64::from(at(x, y));
                    focus += d * d;
                    pairs += 1;
                }
            }
        }
        let roi = p.roi.unwrap_or(Rect::new(0, 0, w, h));
        let roi_stats = std::array::from_fn(|k| {
            let values: Vec<f64> = (roi.y..roi.y + roi.height)
                .flat_map(|y| (roi.x..roi.x + roi.width).map(move |x| (x, y)))
                .filter(|&(x, y)| c(x, y) == k)
                .map(|(x, y)| f64::from(at(x, y)))
                .collect();
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            ChannelStats {
                mean,
                std: var.sqrt(),
                min: values.iter().copied().fold(f64::MAX, f64::min) as u16,
                max: values.iter().copied().fold(0.0, f64::max) as u16,
                count: values.len() as u64,
            }
        });
        let cell = |cx: u32, cy: u32| -> [f32; 3] {
            let mut s = [0.0f32; 3];
            let mut n = [0.0f32; 3];
            for y in 2 * cy..2 * cy + 2 {
                for x in 2 * cx..2 * cx + 2 {
                    s[c(x, y)] += f32::from(at(x, y));
                    n[c(x, y)] += 1.0;
                }
            }
            [s[0] / n[0], s[1] / n[1], s[2] / n[2]]
        };
        let (row, column) = (p.row.unwrap_or(h / 2) / 2, p.column.unwrap_or(w / 2) / 2);
        let rows: Vec<[f32; 3]> = (0..w / 2).map(|cx| cell(cx, row)).collect();
        let columns: Vec<[f32; 3]> = (0..h / 2).map(|cy| cell(column, cy)).collect();
        LiveStatistics {
            seq: 0,
            exposure_start: Timestamp(0),
            full_scale: full,
            histogram,
            mask: p.mask,
            saturated,
            saturated_fraction: f64::from(at_full) / f64::from(w * h),
            row_profile: std::array::from_fn(|k| rows.iter().map(|v| v[k]).collect()),
            column_profile: std::array::from_fn(|k| columns.iter().map(|v| v[k]).collect()),
            roi: roi_stats,
            focus: focus / pairs as f64 / (f64::from(full) * f64::from(full)),
        }
    }

    /// Whether two sets of statistics agree, their floating sums to rounding.
    fn agree(a: &LiveStatistics, b: &LiveStatistics) {
        assert_eq!(a.histogram, b.histogram);
        assert_eq!(a.saturated, b.saturated);
        assert_eq!((a.full_scale, a.mask), (b.full_scale, b.mask));
        assert!((a.saturated_fraction - b.saturated_fraction).abs() < 1e-12);
        assert_eq!(a.row_profile, b.row_profile);
        assert_eq!(a.column_profile, b.column_profile);
        for (x, y) in a.roi.iter().zip(&b.roi) {
            assert_eq!((x.min, x.max, x.count), (y.min, y.max, y.count));
            assert!(
                (x.mean - y.mean).abs() < 1e-9 && (x.std - y.std).abs() < 1e-6,
                "{x:?} {y:?}"
            );
        }
        assert!(
            (a.focus - b.focus).abs() <= 1e-12 * b.focus.max(1e-12),
            "{} {}",
            a.focus,
            b.focus
        );
    }

    #[test]
    fn the_statistics_match_a_reference_worked_site_by_site() {
        let hash =
            |x: u32, y: u32| (x.wrapping_mul(2_654_435_761) ^ y.wrapping_mul(40_503)) % 4_096;
        for (pattern, bits, params) in [
            (BayerPattern::Rggb, 12, StatsParams::default()),
            (
                BayerPattern::Gbrg,
                12,
                StatsParams {
                    bins: 64,
                    mask: [7, 5],
                    roi: Some(Rect::new(10, 6, 31, 17)),
                    row: Some(13),
                    column: Some(41),
                },
            ),
            (BayerPattern::Bggr, 8, StatsParams::default()),
        ] {
            let max = (1u32 << bits) - 1;
            let m = mosaic(96, 64, bits, pattern, |x, y| {
                // Bright in a corner, so some sites saturate.
                if x < 10 && y < 6 {
                    max as u16
                } else {
                    (hash(x, y) % (max + 1)) as u16
                }
            });
            let got = statistics(&m, 0, Timestamp(0), &params).unwrap();
            agree(&got, &reference(&m, &params));
            assert!(got.saturated_fraction > 0.0 && got.saturated.iter().any(|s| *s));
            assert_eq!(
                got.histogram
                    .iter()
                    .map(|h| h.iter().sum::<u64>())
                    .sum::<u64>(),
                96 * 64
            );
        }
    }

    #[test]
    fn a_sharper_frame_has_a_higher_focus_value_and_bad_requests_are_refused() {
        let sharp = mosaic(64, 64, 12, BayerPattern::Rggb, |x, _| {
            if (x / 4) % 2 == 0 { 3_000 } else { 500 }
        });
        let soft = mosaic(64, 64, 12, BayerPattern::Rggb, |x, _| {
            1_750 + ((x as f64 / 8.0).sin() * 200.0) as u16
        });
        let p = StatsParams::default();
        let (a, b) = (
            statistics(&sharp, 0, Timestamp(0), &p).unwrap(),
            statistics(&soft, 0, Timestamp(0), &p).unwrap(),
        );
        assert!(a.focus > 10.0 * b.focus, "{} {}", a.focus, b.focus);
        assert_eq!(a.roi[1].mean, (3_000.0 + 500.0) / 2.0);
        for bad in [
            StatsParams {
                bins: 0,
                ..p.clone()
            },
            StatsParams {
                mask: [0, 4],
                ..p.clone()
            },
            StatsParams {
                roi: Some(Rect::new(60, 0, 10, 10)),
                ..p.clone()
            },
        ] {
            assert!(
                statistics(&sharp, 0, Timestamp(0), &bad).is_err(),
                "{bad:?}"
            );
        }
    }
}
