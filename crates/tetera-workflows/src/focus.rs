//! Autofocus's search (P1.16a): a coarse sweep through z, a fine one around its peak, and a
//! parabola through the fine one's, every position approached from below against backlash.
//! The range and the steps are in depths of field. On a flat curve, or a peak at the range's
//! end, it finds no focus and goes back where it started. The choices are in P1.16a's
//! behaviour spec.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tetera_core::{Error, Result};

/// What the search drives: the stage's z, and a measure of the image there.
pub trait FocusProbe {
    /// The stage's z, µm, as it reports it.
    fn z(&mut self) -> Result<f64>;

    /// Moves to `z`, µm, and returns once the stage is still.
    fn move_to(&mut self, z: f64) -> Result<()>;

    /// The focus measure of an image exposed since the stage last stopped.
    fn measure(&mut self) -> Result<f64>;
}

/// How the search goes; distances in depths of field unless they say µm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct SearchOptions {
    /// The coarse sweep's reach either side of the start.
    pub range: f64,
    /// The coarse sweep's step.
    pub coarse_step: f64,
    /// The fine sweep's reach either side of the coarse peak.
    pub fine_range: f64,
    /// The fine sweep's step.
    pub fine_step: f64,
    /// How far below a sweep's first position, or the focus, the stage goes before moving up
    /// to it, µm: more than the stage's backlash.
    pub take_up_um: f64,
    /// How far above the coarse sweep's median its peak must rise to be a focus, as a
    /// fraction.
    pub min_contrast: f64,
}

impl Default for SearchOptions {
    /// ±8 depths of field in steps of one, then ±2 in halves; 10 µm of take-up; a peak 5 %
    /// above the median.
    fn default() -> Self {
        Self {
            range: 8.0,
            coarse_step: 1.0,
            fine_range: 2.0,
            fine_step: 0.5,
            take_up_um: 10.0,
            min_contrast: 0.05,
        }
    }
}

impl SearchOptions {
    /// Whether the search can run with them.
    pub fn check(&self) -> Result<()> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let fine = positive(self.range)
            && positive(self.coarse_step)
            && self.coarse_step <= self.range
            && positive(self.fine_range)
            && positive(self.fine_step)
            && self.fine_step <= self.fine_range
            && self.take_up_um.is_finite()
            && self.take_up_um >= 0.0
            && self.min_contrast.is_finite()
            && self.min_contrast >= 0.0;
        if fine {
            Ok(())
        } else {
            Err(Error::invalid_argument(format!(
                "a search of {self:?}: ranges and steps above 0, each step within its range, a take-up and a contrast of 0 or more"
            )))
        }
    }
}

/// Which sweep a position was measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Sweep {
    /// The coarse sweep, through the whole range.
    Coarse,
    /// The fine sweep, around the coarse peak.
    Fine,
}

/// A position measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FocusPoint {
    /// The stage's z, µm, as it reported it.
    pub z: f64,
    /// The measure there.
    pub value: f64,
    /// Its sweep.
    pub sweep: Sweep,
}

/// What a search found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Focus {
    /// The focus, µm, where the stage is now; `None` if it found none, and the stage is back
    /// at `start`.
    pub z: Option<f64>,
    /// Why it found none.
    pub why_not: Option<String>,
    /// Where it started, µm.
    pub start: f64,
    /// The depth of field its distances were in, µm.
    pub depth_of_field: f64,
    /// How far the coarse peak rose above the coarse sweep's median, as a fraction.
    pub contrast: f64,
    /// Each position measured, in the order taken.
    pub curve: Vec<FocusPoint>,
}

impl Focus {
    /// The focus, or a `not_found` error saying why there is none.
    pub fn found(&self) -> Result<f64> {
        self.z.ok_or_else(|| {
            Error::not_found(format!(
                "no focus: {}",
                self.why_not.as_deref().unwrap_or("none found")
            ))
        })
    }
}

/// Searches through z for the focus, from where `probe`'s stage is, `depth_of_field` µm
/// setting the distances, within `limits`, the stage's z travel, µm. `step(done, message)`
/// is called before each position, `done` from 0 to 1: an error from it stops the search
/// where it is.
pub fn search(
    probe: &mut dyn FocusProbe,
    depth_of_field: f64,
    limits: (f64, f64),
    options: &SearchOptions,
    step: &mut dyn FnMut(f64, &str) -> Result<()>,
) -> Result<Focus> {
    options.check()?;
    if !(depth_of_field.is_finite() && depth_of_field > 0.0) {
        return Err(Error::invalid_argument(format!(
            "a depth of field of {depth_of_field} µm: above 0"
        )));
    }
    let d = depth_of_field;
    let start = probe.z()?;
    let coarse = positions(start, options.range * d, options.coarse_step * d);
    let fine_count = positions(0.0, options.fine_range * d, options.fine_step * d).len();
    let lowest = coarse[0] - options.fine_range * d - options.take_up_um;
    let highest = coarse[coarse.len() - 1] + options.fine_range * d;
    if lowest < limits.0 || highest > limits.1 {
        return Err(Error::invalid_argument(format!(
            "a search from {lowest:.1} to {highest:.1} µm, beyond the stage's z travel, {:.1} to {:.1} µm",
            limits.0, limits.1
        )));
    }
    let mut run = Run {
        probe,
        step,
        take_up: options.take_up_um,
        total: coarse.len() + fine_count,
        curve: Vec::new(),
    };
    run.sweep(&coarse, Sweep::Coarse)?;
    let values: Vec<f64> = run.curve.iter().map(|p| p.value).collect();
    let (peak, best) = highest_of(&values);
    let median = median(&values);
    let contrast = if median > 0.0 {
        best / median - 1.0
    } else if best > median {
        f64::INFINITY
    } else {
        0.0
    };
    let mut focus = Focus {
        z: None,
        why_not: None,
        start,
        depth_of_field: d,
        contrast,
        curve: Vec::new(),
    };
    let why_not = if contrast < options.min_contrast {
        Some(format!(
            "the curve is flat: its peak is {:.1} % above its median, under {:.1} %",
            contrast * 100.0,
            options.min_contrast * 100.0
        ))
    } else if peak == 0 || peak == coarse.len() - 1 {
        Some(format!(
            "the peak is at the range's end, {:.1} µm: the focus may lie beyond ±{:.1} µm of {start:.1} µm",
            run.curve[peak].z,
            options.range * d
        ))
    } else {
        None
    };
    if let Some(why) = why_not {
        (run.step)(1.0, "no focus: back to the start")?;
        run.approach(start)?;
        focus.why_not = Some(why);
        focus.curve = run.curve;
        return Ok(focus);
    }
    let fine = positions(
        run.curve[peak].z,
        options.fine_range * d,
        options.fine_step * d,
    );
    run.sweep(&fine, Sweep::Fine)?;
    let fine_points: Vec<(f64, f64)> = run
        .curve
        .iter()
        .filter(|p| p.sweep == Sweep::Fine)
        .map(|p| (p.z, p.value))
        .collect();
    let z = vertex(&fine_points);
    (run.step)(1.0, &format!("to the focus, {z:.2} µm"))?;
    run.approach(z)?;
    focus.z = Some(z);
    focus.curve = run.curve;
    Ok(focus)
}

/// A search under way.
struct Run<'a> {
    probe: &'a mut dyn FocusProbe,
    step: &'a mut dyn FnMut(f64, &str) -> Result<()>,
    take_up: f64,
    total: usize,
    curve: Vec<FocusPoint>,
}

impl Run<'_> {
    /// To `z` from below: down past it by the take-up, then up.
    fn approach(&mut self, z: f64) -> Result<()> {
        self.probe.move_to(z - self.take_up)?;
        self.probe.move_to(z)
    }

    /// Each of `zs`, rising, measured: the first approached from below, the rest moved up to.
    fn sweep(&mut self, zs: &[f64], sweep: Sweep) -> Result<()> {
        for (i, &z) in zs.iter().enumerate() {
            let done = self.curve.len() as f64 / self.total as f64;
            (self.step)(
                done,
                &format!("{sweep:?} sweep: {} of {} at {z:.2} µm", i + 1, zs.len()).to_lowercase(),
            )?;
            if i == 0 {
                self.approach(z)?;
            } else {
                self.probe.move_to(z)?;
            }
            let at = self.probe.z()?;
            let value = self.probe.measure()?;
            self.curve.push(FocusPoint {
                z: at,
                value,
                sweep,
            });
        }
        Ok(())
    }
}

/// From `centre − reach` to `centre + reach` in `step`s, rising: an odd number of them,
/// `centre` in the middle.
fn positions(centre: f64, reach: f64, step: f64) -> Vec<f64> {
    let n = (reach / step).round() as i64;
    (-n..=n).map(|i| centre + i as f64 * step).collect()
}

/// The highest of `values` and where it is: the first, if several are.
fn highest_of(values: &[f64]) -> (usize, f64) {
    let mut best = (0, values[0]);
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

/// The median of `values`.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The peak of `points`, (z, measure), rising in z: the vertex of a parabola fitted by least
/// squares through the logarithm of the measure, if it opens downward and its vertex lies
/// among the points fitted; else the highest point. Fitted are the points around the highest
/// in the curve's top half, above the midpoint of its highest and lowest, and at least three:
/// all of a broad peak's, so its flat top has curvature enough to fit.
pub fn vertex(points: &[(f64, f64)]) -> f64 {
    let values: Vec<f64> = points.iter().map(|p| p.1).collect();
    let (peak, _) = highest_of(&values);
    let zp = points[peak].0;
    let (from, to) = window(&values);
    let window = &points[from..=to];
    if window.len() < 3 {
        return zp;
    }
    fit(window, zp).unwrap_or(zp)
}

/// The points [`vertex`] fits, first and last: around the highest of `values`, those in its
/// top half, then the higher neighbour each time until there are three, where there are.
fn window(values: &[f64]) -> (usize, usize) {
    let (peak, best) = highest_of(values);
    let lowest = values.iter().copied().fold(f64::INFINITY, f64::min);
    let half = (best + lowest) / 2.0;
    let (mut from, mut to) = (peak, peak);
    while from > 0 && values[from - 1] >= half {
        from -= 1;
    }
    while to + 1 < values.len() && values[to + 1] >= half {
        to += 1;
    }
    // At least three, taking the higher neighbour first.
    while to - from < 2 && (from > 0 || to + 1 < values.len()) {
        let left = if from > 0 {
            values[from - 1]
        } else {
            f64::NEG_INFINITY
        };
        let right = values.get(to + 1).copied().unwrap_or(f64::NEG_INFINITY);
        if left >= right {
            from -= 1;
        } else {
            to += 1;
        }
    }
    (from, to)
}

/// The vertex of the parabola through `window`'s logarithms, or its values if any is 0 or
/// below, `zp` the highest point's z: if it opens downward and the vertex lies within the
/// window.
fn fit(window: &[(f64, f64)], zp: f64) -> Option<f64> {
    // The logarithm, where every value is above 0: a Gaussian's is a parabola.
    let logs = window.iter().all(|p| p.1 > 0.0);
    let y = |v: f64| if logs { v.ln() } else { v };
    // y = a + b t + c t², t = z − zp, by the normal equations.
    let (mut s, mut st, mut st2, mut st3, mut st4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sy, mut sty, mut st2y) = (0.0, 0.0, 0.0);
    for &(z, v) in window {
        let t = z - zp;
        let yv = y(v);
        s += 1.0;
        st += t;
        st2 += t * t;
        st3 += t * t * t;
        st4 += t * t * t * t;
        sy += yv;
        sty += t * yv;
        st2y += t * t * yv;
    }
    let m = [[s, st, st2], [st, st2, st3], [st2, st3, st4]];
    let [_, b, c] = solve3(m, [sy, sty, st2y])?;
    if c >= 0.0 {
        return None;
    }
    let z = zp - b / (2.0 * c);
    (z >= window[0].0 && z <= window[window.len() - 1].0).then_some(z)
}

/// `m x = r` by Cramer's rule: `None` when `m` is singular.
fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&m);
    if d == 0.0 || !d.is_finite() {
        return None;
    }
    let mut x = [0.0; 3];
    for (col, out) in x.iter_mut().enumerate() {
        let mut mc = m;
        for row in 0..3 {
            mc[row][col] = r[row];
        }
        *out = det(&mc) / d;
    }
    Some(x)
}

pub mod synthetic;

#[cfg(test)]
mod tests;
