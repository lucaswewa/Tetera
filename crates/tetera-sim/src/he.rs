//! The H&E sample (P1.11a, ADR-0015): a section of stained tissue on a slide, as each stain's
//! density at a position and depth, and its colour by absorbance. Its behaviour spec is in
//! the phase's notes; the numbers below are the spec's.
//!
//! Positions are in µm on the slide, depths in µm above the section's lower face, densities
//! in optical density per µm of the light's path.

use crate::noise::{Seed, cellular, fbm, hash, improved, smoothstep, unit};
use tetera_core::{Error, Result};

/// Each stain's amount: optical density per µm at a point, or summed over a path.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stains {
    /// Haematoxylin: nuclei, blue-purple.
    pub haematoxylin: f32,
    /// Eosin: cytoplasm, collagen and red cells, pink.
    pub eosin: f32,
}

impl Stains {
    fn add(self, other: Stains) -> Stains {
        Stains {
            haematoxylin: self.haematoxylin + other.haematoxylin,
            eosin: self.eosin + other.eosin,
        }
    }

    fn scale(self, k: f32) -> Stains {
        Stains {
            haematoxylin: self.haematoxylin * k,
            eosin: self.eosin * k,
        }
    }
}

/// Haematoxylin's optical density per unit amount in red, green and blue (Ruifrok and
/// Johnston, 2001).
pub const HAEMATOXYLIN: [f32; 3] = [0.650, 0.704, 0.286];
/// Eosin's (Ruifrok and Johnston, 2001).
pub const EOSIN: [f32; 3] = [0.072, 0.990, 0.105];

/// The light that passes `stains`, summed over its path, in red, green and blue: Beer–Lambert,
/// `10^(−OD)` in each channel.
pub fn transmittance(stains: Stains) -> [f32; 3] {
    std::array::from_fn(|c| {
        let od = stains.haematoxylin * HAEMATOXYLIN[c] + stains.eosin * EOSIN[c];
        10f32.powf(-od)
    })
}

/// A transmittance as 8-bit sRGB.
pub fn srgb8(t: [f32; 3]) -> [u8; 3] {
    t.map(|v| {
        let v = v.clamp(0.0, 1.0);
        let e = if v <= 0.003_130_8 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (e * 255.0).round() as u8
    })
}

/// How the section is cut and laid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionConfig {
    /// Its thickness, µm: 3 to 5.
    pub thickness: f32,
    /// Its slope in x and in y, µm per µm.
    pub tilt: [f64; 2],
    /// Whether it lies folded in places.
    pub folds: bool,
}

impl Default for SectionConfig {
    fn default() -> Self {
        Self {
            thickness: 4.0,
            tilt: [0.0, 0.0],
            folds: false,
        }
    }
}

/// A nucleus: where it is and its shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nucleus {
    /// Its centre on the slide, µm.
    pub centre: (f64, f64),
    /// Its centre's depth in the section, µm.
    pub depth: f32,
    /// Its semi-axes, µm: along its orientation, across it, and in depth.
    pub axes: [f32; 3],
    /// Its orientation, radians.
    pub angle: f32,
    /// Its haematoxylin at its darkest, OD/µm.
    pub darkness: f32,
}

/// A red cell in a vessel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedCell {
    /// Its centre on the slide, µm.
    pub centre: (f64, f64),
    /// Its centre's depth in the section, µm.
    pub depth: f32,
}

// Salts, one for each use of the hash.
const MASK: u32 = 1;
const GLANDS: u32 = 10;
const FAT: u32 = 11;
const ANGLE: u32 = 12;
const WAVER: u32 = 13;
const CLEFT: u32 = 14;
const GLAND_CELLS: u32 = 15;
const GLAND_EDGE: u32 = 16;
const FAT_CELLS: u32 = 17;
const VESSELS: u32 = 18;
const NUCLEI: u32 = 20;
const NUCLEUS_EDGE: u32 = 21;
const CHROMATIN: u32 = 22;
const RED_CELLS: u32 = 23;
const FOLDS: u32 = 24;

/// The nuclei's grid, µm.
pub const NUCLEUS_CELL: f64 = 10.0;
/// The red cells' grid, µm.
pub const RED_CELL_CELL: f64 = 7.0;
const RED_CELL_RADIUS: f32 = 3.6;
/// The fibres' grid of orientations, µm.
const FIBRE_CELL: f64 = 40.0;

/// What the section holds at a point of the slide's plane, through its depth: everything but
/// nuclei and red cells is the same at every depth.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Column {
    /// The stains that don't vary with depth, OD/µm.
    pub stains: Stains,
    /// How much a nucleus is likely here, 0 to 1.
    pub nuclei: f32,
    /// How much this is a vessel's lumen, 0 to 1.
    pub vessel: f32,
    /// A nucleus's orientation here, radians.
    pub angle: f32,
}

/// A procedural H&E sample, from a seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeSample {
    seed: Seed,
    section: SectionConfig,
}

impl HeSample {
    /// The sample from `seed`, its section cut and laid as `section` says.
    pub fn new(seed: u64, section: SectionConfig) -> Result<Self> {
        if !(3.0..=5.0).contains(&section.thickness) || section.tilt.iter().any(|t| !t.is_finite())
        {
            return Err(Error::invalid_argument(format!(
                "a section {} µm thick, tilted {:?}: it is 3 to 5 µm thick, and tilted by a finite slope",
                section.thickness, section.tilt
            )));
        }
        Ok(Self {
            seed: Seed::new(seed),
            section,
        })
    }

    /// How its section is cut and laid.
    pub fn section(&self) -> SectionConfig {
        self.section
    }

    /// The section's lower face at (`x`, `y`), µm above the slide.
    pub fn lower_face(&self, x: f64, y: f64) -> f64 {
        self.section.tilt[0] * x + self.section.tilt[1] * y
    }

    /// Whether (`x`, `y`) lies in a fold, where the section is doubled.
    pub fn folded(&self, x: f64, y: f64) -> bool {
        self.section.folds && improved(self.seed, FOLDS, x, y, 2_000.0).abs() < 0.03
    }

    /// The section's thickness at (`x`, `y`): doubled in a fold.
    pub fn thickness(&self, x: f64, y: f64) -> f32 {
        if self.folded(x, y) {
            2.0 * self.section.thickness
        } else {
            self.section.thickness
        }
    }

    /// The column at (`x`, `y`).
    pub fn column(&self, x: f64, y: f64) -> Column {
        let s = self.seed;
        let tissue = smoothstep(-0.152, -0.148, fbm(s, MASK, x, y, 800.0, 4));
        if tissue == 0.0 {
            return Column::default();
        }
        let glands = smoothstep(0.175, 0.225, improved(s, GLANDS, x, y, 450.0));
        let fat = smoothstep(0.275, 0.325, improved(s, FAT, x, y, 350.0)) * (1.0 - glands);
        let mut stroma = 1.0 - glands - fat;
        let mut stains = Stains::default();
        let mut epithelium = 0.0;
        let mut angle = self.fibre_angle(x, y);
        if glands > 0.0 {
            let g = cellular(s, GLAND_CELLS, x, y, 180.0);
            let radius = 180.0 * (0.30 + 0.12 * unit(g.hash));
            let d = g.f1 * (1.0 + 0.06 * improved(s, GLAND_EDGE, x, y, 25.0));
            let lumen = 1.0 - smoothstep(radius - 17.0, radius - 15.0, d);
            let epi = smoothstep(radius - 17.0, radius - 15.0, d)
                * (1.0 - smoothstep(radius - 1.0, radius + 1.0, d));
            epithelium = glands * epi;
            stroma += glands * (1.0 - lumen - epi);
            stains = stains.add(
                Stains {
                    haematoxylin: 0.02,
                    eosin: 0.10,
                }
                .scale(epithelium),
            );
            if epi > 0.0 {
                // Columnar cells: nuclei along the radius.
                angle = g.offset.1.atan2(g.offset.0);
            }
        }
        // Vessels run through stroma and fat.
        let v = improved(s, VESSELS, x, y, 350.0).abs();
        let vessel = (1.0 - smoothstep(0.022, 0.028, v)) * (1.0 - glands);
        let wall =
            smoothstep(0.022, 0.028, v) * (1.0 - smoothstep(0.030, 0.034, v)) * (1.0 - glands);
        let open = 1.0 - vessel - wall;
        if stroma > 0.0 {
            stains = stains.add(
                Stains {
                    haematoxylin: 0.0,
                    eosin: self.collagen(x, y),
                }
                .scale(stroma * open),
            );
        }
        if fat > 0.0 {
            let c = cellular(s, FAT_CELLS, x, y, 70.0);
            let membrane = 1.0 - smoothstep(0.9, 1.5, c.f2 - c.f1);
            stains.eosin += 0.12 * membrane * fat * open;
        }
        stains.eosin += 0.16 * wall;
        let nuclei = (0.30 * stroma + 0.85 * epithelium + 0.06 * fat) * open;
        Column {
            stains: stains.scale(tissue),
            nuclei: nuclei * tissue,
            vessel: vessel * tissue,
            angle,
        }
    }

    /// The fibres' direction at (`x`, `y`), radians.
    fn fibre_angle(&self, x: f64, y: f64) -> f32 {
        std::f32::consts::PI * improved(self.seed, ANGLE, x, y, 250.0)
    }

    /// Collagen's eosin at (`x`, `y`): stripes across the fibres' direction, which sweeps in
    /// bundles, with pale clefts.
    fn collagen(&self, x: f64, y: f64) -> f32 {
        // Stripes on each corner of a 40 µm grid, in that corner's direction and its own
        // coordinates, blended across the cell, so the direction can turn without the
        // stripes tearing.
        let (fx, fy) = (x / FIBRE_CELL, y / FIBRE_CELL);
        let (cx, cy) = (fx.floor(), fy.floor());
        let (tx, ty) = ((fx - cx) as f32, (fy - cy) as f32);
        let waver = 2.5 * improved(self.seed, WAVER, x, y, 30.0);
        let stripe = |kx: f64, ky: f64| {
            let (ox, oy) = ((cx + kx) * FIBRE_CELL, (cy + ky) * FIBRE_CELL);
            let theta = self.fibre_angle(ox, oy);
            let (dx, dy) = ((x - ox) as f32, (y - oy) as f32);
            let across = -theta.sin() * dx + theta.cos() * dy;
            let bundle = 0.5 + 0.5 * (std::f32::consts::TAU * across / 6.0 + waver).sin();
            let fibril = 0.5 + 0.5 * (std::f32::consts::TAU * across / 1.7 + 2.0 * waver).sin();
            let s = 0.65 * bundle + 0.35 * fibril;
            s * s * s
        };
        let top = stripe(0.0, 0.0) * (1.0 - tx) + stripe(1.0, 0.0) * tx;
        let bottom = stripe(0.0, 1.0) * (1.0 - tx) + stripe(1.0, 1.0) * tx;
        let f = top * (1.0 - ty) + bottom * ty;
        let cleft = smoothstep(-0.44, -0.46, improved(self.seed, CLEFT, x, y, 40.0));
        let grain = 0.85 + 0.15 * improved(self.seed, WAVER + 100, x, y, 0.8);
        0.16 * (0.45 + 0.55 * f) * grain * (1.0 - cleft)
    }

    /// The nucleus in grid cell (`i`, `j`), if one is there: decided at its own centre.
    pub fn nucleus(&self, i: i32, j: i32) -> Option<Nucleus> {
        let h = hash(self.seed, NUCLEI, i, j, 0);
        let u = |k: u32| unit(hash(self.seed, NUCLEI + 100 + k, i, j, 0));
        let centre = (
            (f64::from(i) + 0.5 + 0.35 * f64::from(2.0 * u(0) - 1.0)) * NUCLEUS_CELL,
            (f64::from(j) + 0.5 + 0.35 * f64::from(2.0 * u(1) - 1.0)) * NUCLEUS_CELL,
        );
        let column = self.column(centre.0, centre.1);
        if unit(h) >= column.nuclei {
            return None;
        }
        let a = 2.2 + 1.6 * u(2);
        Some(Nucleus {
            centre,
            depth: self.thickness(centre.0, centre.1) * u(3),
            axes: [a, a * (0.55 + 0.45 * u(4)), 2.0 + u(5)],
            angle: column.angle,
            darkness: 0.50 * (0.8 + 0.4 * u(6)),
        })
    }

    /// The red cell in grid cell (`i`, `j`), if one is there: decided at its own centre.
    pub fn red_cell(&self, i: i32, j: i32) -> Option<RedCell> {
        let u = |k: u32| unit(hash(self.seed, RED_CELLS + k, i, j, 0));
        let centre = (
            (f64::from(i) + 0.5 + 0.25 * f64::from(2.0 * u(1) - 1.0)) * RED_CELL_CELL,
            (f64::from(j) + 0.5 + 0.25 * f64::from(2.0 * u(2) - 1.0)) * RED_CELL_CELL,
        );
        if u(0) >= 0.7 * self.column(centre.0, centre.1).vessel {
            return None;
        }
        let t = self.thickness(centre.0, centre.1);
        Some(RedCell {
            centre,
            depth: 1.0 + (t - 2.0) * u(3),
        })
    }

    /// The nuclei whose centres lie in [`x0`, `x1`) × [`y0`, `y1`).
    pub fn nuclei(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Nucleus> {
        let cells = |a: f64, b: f64| {
            (a / NUCLEUS_CELL).floor() as i32 - 1..=(b / NUCLEUS_CELL).floor() as i32 + 1
        };
        let mut found = Vec::new();
        for j in cells(y0, y1) {
            for i in cells(x0, x1) {
                if let Some(n) = self.nucleus(i, j)
                    && (x0..x1).contains(&n.centre.0)
                    && (y0..y1).contains(&n.centre.1)
                {
                    found.push(n);
                }
            }
        }
        found
    }

    /// A nucleus's haematoxylin at (`x`, `y`) and `depth` in the section, OD/µm.
    pub fn nucleus_density(&self, n: &Nucleus, x: f64, y: f64, depth: f32) -> f32 {
        let (dx, dy) = ((x - n.centre.0) as f32, (y - n.centre.1) as f32);
        let (sin, cos) = n.angle.sin_cos();
        let (along, across) = (cos * dx + sin * dy, -sin * dx + cos * dy);
        let dz = depth - n.depth;
        let [a, b, c] = n.axes;
        let lateral = (along / a).powi(2) + (across / b).powi(2);
        let q = lateral + (dz / c).powi(2);
        if q > 1.2 {
            return 0.0;
        }
        let edge = q * (1.0 + 0.08 * improved(self.seed, NUCLEUS_EDGE, x, y, 2.0));
        // Densest at its centre, fading toward its edge. Its membrane is a darker ring around
        // each depth's cross-section, not a shell, so its centre stays darkest at its depth.
        let body = smoothstep(1.0, 0.82, edge) * (1.0 - 0.25 * q);
        let cross_section = (1.0 - (dz / c).powi(2)).max(1e-3);
        let rim = 1.0 + 0.3 * smoothstep(0.6, 0.85, lateral / cross_section);
        let clumps = 1.0 - smoothstep(0.0, 0.35, cellular(self.seed, CHROMATIN, x, y, 0.7).f1);
        let grain = improved(self.seed, CHROMATIN + 100, x, y, 0.5);
        let chromatin = 0.62 + 0.25 * clumps + 0.13 * grain;
        n.darkness * body * rim * chromatin
    }

    /// A red cell's eosin at (`x`, `y`) and `depth` in the section, OD/µm: a disc 2 µm thick,
    /// paler at its centre.
    pub fn red_cell_density(&self, cell: &RedCell, x: f64, y: f64, depth: f32) -> f32 {
        let (dx, dy) = ((x - cell.centre.0) as f32, (y - cell.centre.1) as f32);
        let r = (dx * dx + dy * dy).sqrt() / RED_CELL_RADIUS;
        if r > 1.0 {
            return 0.0;
        }
        let disc = smoothstep(1.0, 0.9, r) * smoothstep(1.0, 0.8, (depth - cell.depth).abs());
        0.40 * (1.0 - 0.35 * (1.0 - r * r)) * disc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> HeSample {
        HeSample::new(7, SectionConfig::default()).unwrap()
    }

    #[test]
    fn colour_is_by_absorbance() {
        assert_eq!(
            transmittance(Stains::default()),
            [1.0, 1.0, 1.0],
            "no stain is white"
        );
        let h = transmittance(Stains {
            haematoxylin: 1.0,
            eosin: 0.0,
        });
        assert!(h[2] > h[0] && h[0] > h[1], "blue-purple: {h:?}");
        let e = transmittance(Stains {
            haematoxylin: 0.0,
            eosin: 1.0,
        });
        assert!(e[1] < e[2] && e[1] < e[0], "pink, green the darkest: {e:?}");
        let more = transmittance(Stains {
            haematoxylin: 2.0,
            eosin: 0.0,
        });
        assert!((0..3).all(|c| more[c] < h[c]), "more stain is darker");
        assert!((h[0] - 10f32.powf(-0.650)).abs() < 1e-6);
        assert_eq!(srgb8([1.0, 0.0, 0.5]), [255, 0, 188]);
        assert_eq!(srgb8([0.002, 2.0, -1.0]), [7, 255, 0]);
    }

    #[test]
    fn a_section_is_three_to_five_micrometres_thick_and_tilted_finitely() {
        for (thickness, tilt) in [(2.9, [0.0, 0.0]), (5.1, [0.0, 0.0]), (4.0, [f64::NAN, 0.0])] {
            assert!(
                HeSample::new(
                    1,
                    SectionConfig {
                        thickness,
                        tilt,
                        folds: false
                    }
                )
                .is_err()
            );
        }
        let tilted = HeSample::new(
            1,
            SectionConfig {
                tilt: [0.0005, -0.001],
                ..SectionConfig::default()
            },
        )
        .unwrap();
        assert_eq!(tilted.lower_face(2_000.0, 1_000.0), 0.0);
        assert!((tilted.lower_face(4_000.0, 0.0) - 2.0).abs() < 1e-12);
        assert_eq!(sample().section(), SectionConfig::default());
    }

    #[test]
    fn a_fold_doubles_the_section_within_its_band() {
        let folded = HeSample::new(
            7,
            SectionConfig {
                folds: true,
                ..SectionConfig::default()
            },
        )
        .unwrap();
        let mut inside = 0;
        for i in 0..4_000 {
            let x = f64::from(i) * 5.0;
            if folded.folded(x, 0.0) {
                inside += 1;
                assert_eq!(folded.thickness(x, 0.0), 8.0);
                assert_eq!(sample().thickness(x, 0.0), 4.0, "without folds, never");
            }
        }
        assert!(
            (1..4_000).contains(&inside),
            "{inside} of 4000 points in a fold"
        );
    }

    #[test]
    fn bare_glass_holds_nothing_and_tissue_its_parts() {
        let s = sample();
        let (mut glass, mut tissue, mut vessel, mut fat_or_lumen) = (0, 0, 0, 0);
        for j in 0..200 {
            for i in 0..200 {
                let c = s.column(f64::from(i) * 20.0, f64::from(j) * 20.0);
                if c == Column::default() {
                    glass += 1;
                } else {
                    tissue += 1;
                }
                if c.vessel > 0.95 {
                    vessel += 1;
                    assert!(c.stains.eosin < 0.02, "a lumen's own stain is faint: {c:?}");
                }
                if c.stains.eosin < 0.02 && c.stains.haematoxylin < 0.002 && c != Column::default()
                {
                    fat_or_lumen += 1;
                }
            }
        }
        let fraction = f64::from(tissue) / 40_000.0;
        assert!(
            (0.30..=0.85).contains(&fraction),
            "tissue covers {fraction}"
        );
        assert!(
            glass > 0 && vessel > 0 && fat_or_lumen > 0,
            "{glass} {vessel} {fat_or_lumen}"
        );
    }

    #[test]
    fn nuclei_are_dense_enough_and_darkest_at_their_own_depth() {
        let s = sample();
        let nuclei = s.nuclei(0.0, 0.0, 2_000.0, 2_000.0);
        let per_mm2 = nuclei.len() as f64 / 4.0;
        assert!(
            (1_000.0..=10_000.0).contains(&per_mm2),
            "{per_mm2} nuclei a mm²"
        );
        for n in nuclei.iter().take(200) {
            let (x, y) = n.centre;
            let at = s.nucleus_density(n, x, y, n.depth);
            assert!(at > 0.15, "{at} at its centre: {n:?}");
            for dz in [1.0, 2.0] {
                assert!(s.nucleus_density(n, x, y, n.depth + dz) < at);
                assert!(s.nucleus_density(n, x, y, n.depth - dz) < at);
            }
            assert_eq!(s.nucleus_density(n, x, y, n.depth + 4.0), 0.0, "beyond it");
            assert!((0.0..=s.thickness(x, y)).contains(&n.depth));
            let [a, b, c] = n.axes;
            assert!((2.2..=3.8).contains(&a) && b <= a && (2.0..=3.0).contains(&c));
        }
    }

    #[test]
    fn red_cells_fill_lumens_paler_at_their_centres() {
        let s = sample();
        let mut found = 0;
        for j in 0..400 {
            for i in 0..400 {
                if let Some(cell) = s.red_cell(i, j) {
                    found += 1;
                    let (x, y) = cell.centre;
                    assert!(s.column(x, y).vessel > 0.0);
                    let centre = s.red_cell_density(&cell, x, y, cell.depth);
                    let rim = s.red_cell_density(&cell, x + 2.5, y, cell.depth);
                    assert!(rim > centre && centre > 0.2, "{centre} then {rim}");
                    assert_eq!(s.red_cell_density(&cell, x + 4.0, y, cell.depth), 0.0);
                    assert_eq!(s.red_cell_density(&cell, x, y, cell.depth + 1.5), 0.0);
                }
            }
        }
        assert!(found > 0, "no red cells in 2.8 mm²");
    }
}
