//! Scans (P1.17a, ADR-0022): a plan of stage positions, made before the scan starts with its
//! estimated time, and the stop-and-go engine that runs it.
//!
//! - **Planners:** an XY snake over a rectangle, from the field of view and an overlap, in
//!   serpentine order; a z-scan, by range and step, or N planes around the current z.
//! - **The estimate:** each move as the stage makes it, from its speeds and accelerations,
//!   and per position its report, the settle, a frame and its writing.
//! - [`engine`]: move, settle, autofocus if asked, a frame at full resolution, its position;
//!   pause, resume and cancel.

pub mod engine;

pub use engine::{AutofocusAt, ScanControl, ScanRequest, ScanResult, ScanRig, Tile, preview, scan};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tetera_core::{Error, Result, StagePos, Um};
use tetera_hal::stage::{Axis, StageCapabilities};

/// What a scan covers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Pattern {
    /// A rectangle of the stage's x and y covered with tiles, in serpentine order, at the
    /// current z.
    Snake {
        /// One corner's x, µm.
        x0: f64,
        /// Its y, µm.
        y0: f64,
        /// The opposite corner's x, µm.
        x1: f64,
        /// Its y, µm.
        y1: f64,
        /// What neighbouring tiles share, a fraction of the field: 0 to 0.5.
        overlap: f64,
    },
    /// z from `from` to `to` in `step`s, µm, rising, at the current x and y.
    ZRange {
        /// One end, µm.
        from: f64,
        /// The other, µm.
        to: f64,
        /// The step, µm.
        step: f64,
    },
    /// `planes` planes `step` µm apart, centred on the current z, rising.
    ZAround {
        /// How many.
        planes: u32,
        /// The step, µm.
        step: f64,
    },
}

/// A plan: the positions, in the order taken, and what it will take.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Plan {
    /// What it covers.
    pub pattern: Pattern,
    /// Each position, in order.
    pub positions: Vec<StagePos>,
    /// Tiles across and down for a snake; 1 by the planes for a z-scan.
    pub grid: [u32; 2],
    /// The field of view at the sample, µm, across and down.
    pub field: [f64; 2],
    /// The estimated time, s.
    pub estimate_s: f64,
}

/// One axis of a snake: the first tile's centre, the step, and how many, the grid centred on
/// `lo` to `hi` and covering it.
fn axis_tiles(lo: f64, hi: f64, field: f64, overlap: f64) -> (f64, f64, u32) {
    let step = field * (1.0 - overlap);
    let span = hi - lo;
    // A hair under each whole step, so a span of exactly n steps isn't taken for more.
    let n = if span <= field {
        1
    } else {
        ((span - field) / step - 1e-9).ceil() as u32 + 1
    };
    let covered = field + f64::from(n - 1) * step;
    (lo + field / 2.0 - (covered - span) / 2.0, step, n)
}

/// The positions `pattern` takes from `here`, with a field of view of `field` µm, each within
/// `capabilities`' travel; and the grid.
pub fn positions(
    pattern: &Pattern,
    here: StagePos,
    field: [f64; 2],
    capabilities: &StageCapabilities,
) -> Result<(Vec<StagePos>, [u32; 2])> {
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    let (positions, grid) = match *pattern {
        Pattern::Snake {
            x0,
            y0,
            x1,
            y1,
            overlap,
        } => {
            if !(finite(&[x0, y0, x1, y1, field[0], field[1]])
                && (0.0..=0.5).contains(&overlap)
                && field[0] > 0.0
                && field[1] > 0.0)
            {
                return Err(Error::invalid_argument(format!(
                    "a snake over ({x0}, {y0}) to ({x1}, {y1}) with an overlap of {overlap} and a field of {field:?} µm: corners, a field above 0, and an overlap of 0 to 0.5"
                )));
            }
            let (sx, dx, nx) = axis_tiles(x0.min(x1), x0.max(x1), field[0], overlap);
            let (sy, dy, ny) = axis_tiles(y0.min(y1), y0.max(y1), field[1], overlap);
            let mut at = Vec::with_capacity((nx * ny) as usize);
            for j in 0..ny {
                for k in 0..nx {
                    // Serpentine: odd rows run back.
                    let i = if j % 2 == 0 { k } else { nx - 1 - k };
                    at.push(StagePos::new(
                        Um(sx + f64::from(i) * dx),
                        Um(sy + f64::from(j) * dy),
                        here.z,
                    ));
                }
            }
            (at, [nx, ny])
        }
        Pattern::ZRange { from, to, step } => {
            if !(finite(&[from, to, step]) && step > 0.0) {
                return Err(Error::invalid_argument(format!(
                    "a z-scan from {from} to {to} µm in steps of {step}: a step above 0"
                )));
            }
            let (lo, hi) = (from.min(to), from.max(to));
            let n = ((hi - lo) / step + 1e-9).floor() as u32 + 1;
            let at = (0..n)
                .map(|k| StagePos::new(here.x, here.y, Um(lo + f64::from(k) * step)))
                .collect();
            (at, [1, n])
        }
        Pattern::ZAround { planes, step } => {
            if planes == 0 || !(step.is_finite() && step > 0.0) {
                return Err(Error::invalid_argument(format!(
                    "a z-scan of {planes} planes {step} µm apart: a plane or more, a step above 0"
                )));
            }
            let middle = f64::from(planes - 1) / 2.0;
            let at = (0..planes)
                .map(|k| {
                    StagePos::new(
                        here.x,
                        here.y,
                        Um(here.z.um() + (f64::from(k) - middle) * step),
                    )
                })
                .collect();
            (at, [1, planes])
        }
    };
    for p in &positions {
        capabilities.check_target(p)?;
    }
    Ok((positions, grid))
}

/// The time, s, the stage takes from `from` to `to`, as it makes a move: along the line, at
/// the speed and acceleration of its most limited axis, a trapezoid, or a triangle when short.
pub fn move_time(from: StagePos, to: StagePos, capabilities: &StageCapabilities) -> Result<f64> {
    let delta = [
        to.x.um() - from.x.um(),
        to.y.um() - from.y.um(),
        to.z.um() - from.z.um(),
    ];
    let length = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
    if length == 0.0 {
        return Ok(0.0);
    }
    let (mut v, mut a) = (f64::INFINITY, f64::INFINITY);
    for (d, axis) in delta.iter().zip(Axis::ALL) {
        let share = d.abs() / length;
        if share > 0.0 {
            let caps = capabilities.axis(axis)?;
            v = v.min(caps.max_speed / share);
            a = a.min(caps.acceleration / share);
        }
    }
    Ok(if v * v / a >= length {
        2.0 * (length / a).sqrt()
    } else {
        v / a + length / v
    })
}

/// What a scan's time is made of, beyond its moves.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Timing {
    /// From a move's end to the stage's report that it is still, ms.
    pub report_ms: f64,
    /// The settle after it, ms.
    pub settle_ms: f64,
    /// A frame at full resolution, ms.
    pub frame_ms: f64,
    /// Writing a frame, ms, overlapped with the next position's move and frame.
    pub write_ms: f64,
    /// The switch to full resolution and back, ms.
    pub switch_ms: f64,
    /// An autofocus, ms.
    pub autofocus_ms: f64,
}

/// The time, s, the scan of `positions` from `start` takes, an autofocus where `autofocus`
/// says: per position, the move, the report, the settle, a frame and an autofocus, or the
/// write before it if that is longer; then the last write, and the switches.
pub fn estimate(
    positions: &[StagePos],
    start: StagePos,
    capabilities: &StageCapabilities,
    timing: &Timing,
    autofocus: impl Fn(usize) -> bool,
) -> Result<f64> {
    let mut ms = timing.switch_ms;
    let mut from = start;
    for (i, &to) in positions.iter().enumerate() {
        let mut taking = move_time(from, to, capabilities)? * 1e3
            + timing.report_ms
            + timing.settle_ms
            + timing.frame_ms;
        if autofocus(i) {
            taking += timing.autofocus_ms;
        }
        ms += taking.max(timing.write_ms);
        from = to;
    }
    if !positions.is_empty() {
        ms += timing.write_ms;
    }
    Ok(ms / 1e3)
}

#[cfg(test)]
mod tests;
