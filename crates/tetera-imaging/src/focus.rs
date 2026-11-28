//! Focus measures (P1.16a): how sharp a raw frame is, each higher when sharper, on its green
//! channel, binned. The choices, and their literature, are in P1.16a's behaviour spec.

use crate::image::Mosaic;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tetera_core::frame::format::Channel;
use tetera_core::{Error, Result};

/// The most values across that [`green`] leaves, binning by the whole number that brings the
/// green channel within it.
pub const GREEN_WIDTH: u32 = 512;

/// A focus measure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FocusMeasure {
    /// Brenner's gradient: the mean square of the difference two values apart, across.
    Brenner,
    /// The variance of the 4-neighbour Laplacian.
    LaplacianVariance,
    /// Normalised variance: the variance over the mean. The default (Groen, Young and
    /// Ligthart, 1985; Sun, Duthaler and Nelson, 2004).
    #[default]
    NormalisedVariance,
    /// Tenengrad: the mean square of the Sobel gradient's magnitude.
    Tenengrad,
}

impl FocusMeasure {
    /// Every measure.
    pub const ALL: [Self; 4] = [
        Self::Brenner,
        Self::LaplacianVariance,
        Self::NormalisedVariance,
        Self::Tenengrad,
    ];

    /// Its name, as JSON writes it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Brenner => "brenner",
            Self::LaplacianVariance => "laplacian_variance",
            Self::NormalisedVariance => "normalised_variance",
            Self::Tenengrad => "tenengrad",
        }
    }

    /// Its value on `image`.
    pub fn of(self, image: &Grey) -> f64 {
        match self {
            Self::Brenner => brenner(image),
            Self::LaplacianVariance => laplacian_variance(image),
            Self::NormalisedVariance => normalised_variance(image),
            Self::Tenengrad => tenengrad(image),
        }
    }

    /// Its value on `mosaic`'s green channel, binned to [`GREEN_WIDTH`] across.
    pub fn of_mosaic(self, mosaic: &Mosaic) -> Result<f64> {
        Ok(self.of(&green(mosaic, GREEN_WIDTH)?))
    }
}

/// A grey image of `f32` values, row by row: at least 3 × 3, so every measure has an
/// interior.
#[derive(Debug, Clone, PartialEq)]
pub struct Grey {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

impl Grey {
    /// An image of `values`, `width` × `height` of them.
    pub fn new(width: usize, height: usize, values: Vec<f32>) -> Result<Self> {
        if width < 3 || height < 3 || values.len() != width * height {
            return Err(Error::invalid_argument(format!(
                "a focus measure takes at least 3 × 3 values, as many as its size: {width} × {height}, {} values",
                values.len()
            )));
        }
        Ok(Self {
            width,
            height,
            values,
        })
    }

    /// Its width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Its height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Its values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    fn at(&self, x: usize, y: usize) -> f64 {
        f64::from(self.values[y * self.width + x])
    }
}

/// `mosaic`'s green channel: each whole 2 × 2 cell's two greens averaged, then binned by the
/// least whole number that leaves at most `max_width` values across, each the mean of a block.
/// A part block at the right or the bottom is left out.
pub fn green(mosaic: &Mosaic, max_width: u32) -> Result<Grey> {
    let layout = mosaic.layout();
    if max_width < 3 {
        return Err(Error::invalid_argument(format!(
            "the green channel binned to {max_width} across: 3 at least"
        )));
    }
    let (cells_x, cells_y) = (layout.width / 2, layout.height / 2);
    let k = cells_x.div_ceil(max_width).max(1);
    let (width, height) = ((cells_x / k) as usize, (cells_y / k) as usize);
    // The cell's two greens, by their place in it.
    let greens: Vec<(u32, u32)> = [(0, 0), (1, 0), (0, 1), (1, 1)]
        .into_iter()
        .filter(|&(x, y)| {
            matches!(
                layout.pattern.channel_at(x, y),
                Channel::GreenOnRed | Channel::GreenOnBlue
            )
        })
        .collect();
    let sites = mosaic.sites();
    let row = layout.width as usize;
    let mut values = vec![0f32; width * height];
    let block = f64::from(2 * k * k);
    for (by, out) in values.chunks_mut(width.max(1)).enumerate() {
        for (bx, v) in out.iter_mut().enumerate() {
            let mut sum = 0u64;
            for cy in 0..k {
                for cx in 0..k {
                    let x = 2 * (bx as u32 * k + cx);
                    let y = 2 * (by as u32 * k + cy);
                    for &(dx, dy) in &greens {
                        sum += u64::from(sites[(y + dy) as usize * row + (x + dx) as usize]);
                    }
                }
            }
            *v = (sum as f64 / block) as f32;
        }
    }
    Grey::new(width, height, values)
}

/// Brenner's gradient: the mean over the image of `(G[x + 2, y] − G[x, y])²`.
pub fn brenner(g: &Grey) -> f64 {
    let mut sum = 0.0;
    for y in 0..g.height {
        for x in 0..g.width - 2 {
            let d = g.at(x + 2, y) - g.at(x, y);
            sum += d * d;
        }
    }
    sum / (g.height * (g.width - 2)) as f64
}

/// The variance, over the interior, of the 4-neighbour Laplacian.
pub fn laplacian_variance(g: &Grey) -> f64 {
    let (mut sum, mut squares) = (0.0, 0.0);
    for y in 1..g.height - 1 {
        for x in 1..g.width - 1 {
            let l = g.at(x - 1, y) + g.at(x + 1, y) + g.at(x, y - 1) + g.at(x, y + 1)
                - 4.0 * g.at(x, y);
            sum += l;
            squares += l * l;
        }
    }
    let n = ((g.height - 2) * (g.width - 2)) as f64;
    let mean = sum / n;
    (squares / n - mean * mean).max(0.0)
}

/// Normalised variance: `Σ (G − μ)² / (N μ)`, 0 when the mean μ is 0 or below.
pub fn normalised_variance(g: &Grey) -> f64 {
    let n = g.values.len() as f64;
    let mean = g.values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let squares: f64 = g
        .values
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum();
    squares / (n * mean)
}

/// Tenengrad: the mean, over the interior, of `Gx² + Gy²`, the Sobel operator's gradients.
pub fn tenengrad(g: &Grey) -> f64 {
    let mut sum = 0.0;
    for y in 1..g.height - 1 {
        for x in 1..g.width - 1 {
            let gx = g.at(x + 1, y - 1) + 2.0 * g.at(x + 1, y) + g.at(x + 1, y + 1)
                - g.at(x - 1, y - 1)
                - 2.0 * g.at(x - 1, y)
                - g.at(x - 1, y + 1);
            let gy = g.at(x - 1, y + 1) + 2.0 * g.at(x, y + 1) + g.at(x + 1, y + 1)
                - g.at(x - 1, y - 1)
                - 2.0 * g.at(x, y - 1)
                - g.at(x + 1, y - 1);
            sum += gx * gx + gy * gy;
        }
    }
    sum / ((g.height - 2) * (g.width - 2)) as f64
}

#[cfg(test)]
mod tests;
