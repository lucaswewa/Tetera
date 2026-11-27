//! Rendering a region of the H&E sample (P1.11a): each pixel's stains summed over a range of
//! depth, averaged over points within it, and its colour. This is the CPU twin, the
//! reference for the GPU (P1.11b).
//!
//! Two passes, as the GPU will make them: first the nuclei and red cells of every grid cell
//! the region touches, each decided at its own centre; then each pixel on its own, from its
//! column and the cells around it, in a fixed order, so threads never change a result.

use crate::he::{
    HeSample, NUCLEUS_CELL, Nucleus, RED_CELL_CELL, RedCell, Stains, srgb8, transmittance,
};
use rayon::prelude::*;
use tetera_core::{Error, Result};

/// Depth is summed in layers this thick, µm, each at its middle.
pub const LAYER: f32 = 0.5;
/// The farthest apart a pixel's points may be, µm.
pub const POINT_SPACING: f64 = 0.25;

/// Which depths a render sums, in µm above the section's lower face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Depth {
    /// The whole section: an image in focus.
    Whole,
    /// From one depth to another: a slice.
    Range(f32, f32),
}

/// A region of the slide to render.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    /// Its first pixel's corner on the slide, µm.
    pub origin: (f64, f64),
    /// Its width, pixels.
    pub width: u32,
    /// Its height, pixels.
    pub height: u32,
    /// Its pixel size, µm.
    pub pixel: f64,
    /// The depths it sums.
    pub depth: Depth,
}

impl Region {
    /// How many points a side each pixel averages: the fewest that keep them at most
    /// [`POINT_SPACING`] apart, from 1 to 4.
    pub fn points(&self) -> u32 {
        ((self.pixel / POINT_SPACING).ceil() as u32).clamp(1, 4)
    }

    fn check(&self) -> Result<()> {
        let depth_ok = match self.depth {
            Depth::Whole => true,
            Depth::Range(a, b) => a.is_finite() && b.is_finite() && a < b,
        };
        if self.width == 0
            || self.height == 0
            || !(self.pixel.is_finite() && self.pixel > 0.0)
            || !(self.origin.0.is_finite() && self.origin.1.is_finite())
            || !depth_ok
        {
            return Err(Error::invalid_argument(format!(
                "a region of {} × {} pixels of {} µm at {:?}, depth {:?}",
                self.width, self.height, self.pixel, self.origin, self.depth
            )));
        }
        Ok(())
    }
}

/// A rendered region: each pixel's stains, summed over its depths, row by row.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendered {
    /// Its width, pixels.
    pub width: u32,
    /// Its height, pixels.
    pub height: u32,
    /// Each pixel's stains, optical density.
    pub stains: Vec<Stains>,
}

impl Rendered {
    /// Each pixel's transmittance in red, green and blue.
    pub fn transmittance(&self) -> Vec<[f32; 3]> {
        self.stains.iter().map(|s| transmittance(*s)).collect()
    }

    /// Its colour as 8-bit sRGB, red, green and blue for each pixel.
    pub fn srgb8(&self) -> Vec<u8> {
        self.stains
            .iter()
            .flat_map(|s| srgb8(transmittance(*s)))
            .collect()
    }
}

/// Things of a grid, each cell's decided once.
struct Grid<T> {
    cell: f64,
    first: (i32, i32),
    columns: i32,
    things: Vec<Option<T>>,
}

impl<T: Send> Grid<T> {
    /// Every cell from `x0`, `y0` to `x1`, `y1` µm, and one more around, from `make`.
    fn new(
        cell: f64,
        (x0, y0): (f64, f64),
        (x1, y1): (f64, f64),
        make: impl Fn(i32, i32) -> Option<T> + Sync,
    ) -> Self {
        let first = (
            (x0 / cell).floor() as i32 - 1,
            (y0 / cell).floor() as i32 - 1,
        );
        let columns = (x1 / cell).floor() as i32 + 2 - first.0;
        let rows = (y1 / cell).floor() as i32 + 2 - first.1;
        let things = (0..rows * columns)
            .into_par_iter()
            .map(|k| make(first.0 + k % columns, first.1 + k / columns))
            .collect();
        Self {
            cell,
            first,
            columns,
            things,
        }
    }

    /// What lies in the 3 × 3 cells around (`x`, `y`).
    fn around(&self, x: f64, y: f64) -> impl Iterator<Item = &T> {
        let (i, j) = (
            (x / self.cell).floor() as i32,
            (y / self.cell).floor() as i32,
        );
        (-1..=1)
            .flat_map(move |dj| {
                (-1..=1).map(move |di| (i + di - self.first.0, j + dj - self.first.1))
            })
            // The grid holds every cell around every point of its region.
            .filter_map(move |(c, r)| self.things[(r * self.columns + c) as usize].as_ref())
    }
}

/// The stains at a point of the slide's plane, summed over the layers of `depth`.
fn point(
    sample: &HeSample,
    nuclei: &Grid<Nucleus>,
    red: &Grid<RedCell>,
    x: f64,
    y: f64,
    depth: Depth,
) -> Stains {
    let column = sample.column(x, y);
    let thickness = sample.thickness(x, y);
    let (from, to) = match depth {
        Depth::Whole => (0.0, thickness),
        Depth::Range(a, b) => (a.max(0.0), b.min(thickness)),
    };
    let mut sum = Stains::default();
    if column.stains == Stains::default() && column.nuclei == 0.0 && column.vessel == 0.0 {
        return sum;
    }
    let near_nuclei: Vec<&Nucleus> = nuclei.around(x, y).collect();
    let near_red: Vec<&RedCell> = red.around(x, y).collect();
    let mut z = from;
    while z < to {
        let layer = LAYER.min(to - z);
        let mid = z + layer / 2.0;
        let nucleus = near_nuclei
            .iter()
            .map(|n| sample.nucleus_density(n, x, y, mid))
            .fold(0.0, f32::max);
        let red_cell = near_red
            .iter()
            .map(|c| sample.red_cell_density(c, x, y, mid))
            .fold(0.0, f32::max);
        sum.haematoxylin += (column.stains.haematoxylin + nucleus) * layer;
        sum.eosin += (column.stains.eosin + red_cell) * layer;
        z += LAYER;
    }
    sum
}

/// Renders `region` of `sample`.
pub fn render(sample: &HeSample, region: &Region) -> Result<Rendered> {
    region.check()?;
    let (x0, y0) = region.origin;
    let end = (
        x0 + f64::from(region.width) * region.pixel,
        y0 + f64::from(region.height) * region.pixel,
    );
    let nuclei = Grid::new(NUCLEUS_CELL, region.origin, end, |i, j| {
        sample.nucleus(i, j)
    });
    let red = Grid::new(RED_CELL_CELL, region.origin, end, |i, j| {
        sample.red_cell(i, j)
    });
    let s = region.points();
    let weight = 1.0 / (s * s) as f32;
    let stains = (0..region.height)
        .into_par_iter()
        .flat_map_iter(|row| {
            let (nuclei, red) = (&nuclei, &red);
            (0..region.width).map(move |col| {
                let mut sum = Stains::default();
                for k in 0..s * s {
                    let x = x0
                        + (f64::from(col) + (f64::from(k % s) + 0.5) / f64::from(s)) * region.pixel;
                    let y = y0
                        + (f64::from(row) + (f64::from(k / s) + 0.5) / f64::from(s)) * region.pixel;
                    let p = point(sample, nuclei, red, x, y, region.depth);
                    sum.haematoxylin += p.haematoxylin * weight;
                    sum.eosin += p.eosin * weight;
                }
                sum
            })
        })
        .collect();
    Ok(Rendered {
        width: region.width,
        height: region.height,
        stains,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::he::SectionConfig;

    fn sample(seed: u64) -> HeSample {
        HeSample::new(seed, SectionConfig::default()).unwrap()
    }

    fn region(origin: (f64, f64), size: u32, pixel: f64, depth: Depth) -> Region {
        Region {
            origin,
            width: size,
            height: size,
            pixel,
            depth,
        }
    }

    fn bits(r: &Rendered) -> Vec<(u32, u32)> {
        r.stains
            .iter()
            .map(|s| (s.haematoxylin.to_bits(), s.eosin.to_bits()))
            .collect()
    }

    /// Somewhere with tissue in it, for the default seed.
    const TISSUE: (f64, f64) = (1_000.0, 1_000.0);

    #[test]
    fn a_pixel_averages_points_at_most_a_quarter_micrometre_apart() {
        let points = |pixel| region(TISSUE, 1, pixel, Depth::Whole).points();
        assert_eq!(
            [
                points(0.048),
                points(0.25),
                points(0.26),
                points(0.5),
                points(0.75),
                points(5.0)
            ],
            [1, 1, 2, 2, 3, 4]
        );
    }

    #[test]
    fn a_region_must_be_somewhere_and_something() {
        let s = sample(7);
        for bad in [
            region(TISSUE, 0, 0.5, Depth::Whole),
            region(TISSUE, 4, 0.0, Depth::Whole),
            region(TISSUE, 4, f64::NAN, Depth::Whole),
            region((f64::INFINITY, 0.0), 4, 0.5, Depth::Whole),
            region(TISSUE, 4, 0.5, Depth::Range(2.0, 1.0)),
            region(TISSUE, 4, 0.5, Depth::Range(1.0, 1.0)),
            Region {
                height: 0,
                ..region(TISSUE, 4, 0.5, Depth::Whole)
            },
            Region {
                width: 0,
                ..region(TISSUE, 4, 0.5, Depth::Whole)
            },
            region(TISSUE, 4, 0.5, Depth::Range(f32::NAN, 1.0)),
        ] {
            assert!(render(&s, &bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_bits_whole_or_in_tiles() {
        let s = sample(7);
        let whole = render(&s, &region(TISSUE, 64, 0.5, Depth::Whole)).unwrap();
        assert_eq!(
            bits(&whole),
            bits(&render(&s, &region(TISSUE, 64, 0.5, Depth::Whole)).unwrap())
        );
        assert!(
            whole.stains.iter().any(|p| p.haematoxylin > 0.0),
            "tissue there"
        );
        // Four tiles of 32 × 32, side by side.
        for (ty, tx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let origin = (
                TISSUE.0 + f64::from(tx) * 16.0,
                TISSUE.1 + f64::from(ty) * 16.0,
            );
            let tile = render(&s, &region(origin, 32, 0.5, Depth::Whole)).unwrap();
            for row in 0..32 {
                for col in 0..32 {
                    let a = tile.stains[(row * 32 + col) as usize];
                    let b = whole.stains[((ty * 32 + row) * 64 + tx * 32 + col) as usize];
                    assert_eq!(
                        (a.haematoxylin.to_bits(), a.eosin.to_bits()),
                        (b.haematoxylin.to_bits(), b.eosin.to_bits())
                    );
                }
            }
        }
    }

    #[test]
    fn a_seed_one_bit_different_changes_most_pixels() {
        let a = render(&sample(7), &region(TISSUE, 48, 1.0, Depth::Whole)).unwrap();
        let b = render(
            &sample(7 ^ (1 << 40)),
            &region(TISSUE, 48, 1.0, Depth::Whole),
        )
        .unwrap();
        let changed = a
            .stains
            .iter()
            .zip(&b.stains)
            .filter(|(p, q)| p != q)
            .count();
        assert!(
            changed > a.stains.len() / 2,
            "{changed} of {} changed",
            a.stains.len()
        );
    }

    #[test]
    fn a_nucleus_is_darkest_in_the_slice_at_its_own_depth() {
        let s = sample(7);
        let nuclei = s.nuclei(TISSUE.0, TISSUE.1, TISSUE.0 + 400.0, TISSUE.1 + 400.0);
        let n = nuclei
            .iter()
            .find(|n| (1.0..3.0).contains(&n.depth))
            .expect("a nucleus mid-section");
        let at = |from: f32| {
            let r = render(
                &s,
                &region(
                    (n.centre.0 - 0.1, n.centre.1 - 0.1),
                    1,
                    0.2,
                    Depth::Range(from, from + 0.5),
                ),
            )
            .unwrap();
            r.stains[0].haematoxylin
        };
        let own = at(n.depth - 0.25);
        assert!(
            own > at(n.depth + 1.25) && own > at(n.depth - 1.75),
            "{own} at its depth"
        );
        let whole = render(
            &s,
            &region((n.centre.0 - 0.1, n.centre.1 - 0.1), 1, 0.2, Depth::Whole),
        )
        .unwrap();
        assert!(
            whole.stains[0].haematoxylin > own,
            "the whole section sums its layers"
        );
    }

    #[test]
    fn depth_is_summed_in_layers_the_last_one_partial() {
        // A point of stroma with no nucleus or red cell: its stain is the same at every
        // depth, so a range sums it times its thickness.
        let s = sample(7);
        let (x, y) = (0..2_000)
            .map(|i| (TISSUE.0 + f64::from(i) * 0.37, TISSUE.1))
            .find(|&(x, y)| {
                let r = render(&s, &region((x, y), 1, 0.2, Depth::Whole)).unwrap();
                r.stains[0].haematoxylin == 0.0 && r.stains[0].eosin > 0.0
            })
            .expect("a point of stroma");
        let column = s.column(x + 0.1, y + 0.1).stains.eosin;
        for (depth, thickness) in [
            (Depth::Range(0.0, 0.8), 0.8),
            (Depth::Range(1.2, 3.0), 1.8),
            (Depth::Range(-1.0, 9.0), 4.0),
            (Depth::Whole, 4.0),
        ] {
            let r = render(&s, &region((x, y), 1, 0.2, depth)).unwrap();
            assert!(
                (r.stains[0].eosin - column * thickness).abs() < 1e-5,
                "{depth:?}: {} against {}",
                r.stains[0].eosin,
                column * thickness
            );
        }
    }

    #[test]
    fn its_colour_follows_its_stains() {
        let s = sample(7);
        let r = render(&s, &region(TISSUE, 16, 1.0, Depth::Whole)).unwrap();
        let rgb = r.srgb8();
        assert_eq!(rgb.len(), 16 * 16 * 3);
        let t = r.transmittance();
        for (k, p) in r.stains.iter().enumerate() {
            assert_eq!(&rgb[k * 3..k * 3 + 3], &srgb8(transmittance(*p)));
            assert_eq!(t[k], transmittance(*p));
        }
    }

    /// A region's fingerprint: its stains summed, and summed again weighted by where each
    /// pixel lies, so a change to any pixel shows.
    fn fingerprint(r: &Rendered) -> [f64; 4] {
        let mut f = [0.0; 4];
        for (k, p) in r.stains.iter().enumerate() {
            let w = (k % 97 + 1) as f64;
            let (h, e) = (f64::from(p.haematoxylin), f64::from(p.eosin));
            f[0] += h;
            f[1] += e;
            f[2] += h * w;
            f[3] += e * w;
        }
        f
    }

    /// The regions the golden test renders, which hold every part of the model: an
    /// overview with a vessel and fat, a gland, a fold, the tissue's edge, a red cell and a
    /// nucleus at 50×, in depth slices too.
    fn golden_regions() -> Vec<(&'static str, HeSample, Region)> {
        let s = sample(7);
        let folded = HeSample::new(
            7,
            SectionConfig {
                folds: true,
                ..SectionConfig::default()
            },
        )
        .unwrap();
        let red = s.red_cell(0, 0).expect("a red cell at the origin");
        let n = *s
            .nuclei(1_000.0, 1_000.0, 1_400.0, 1_400.0)
            .iter()
            .find(|n| (1.0..3.0).contains(&n.depth))
            .expect("a nucleus");
        let around = |(x, y): (f64, f64), size: u32, pixel: f64| {
            (
                x - f64::from(size) * pixel / 2.0,
                y - f64::from(size) * pixel / 2.0,
            )
        };
        vec![
            ("overview", s, region((0.0, 0.0), 128, 2.0, Depth::Whole)),
            (
                "gland",
                s,
                region(around((1_110.0, 0.0), 64, 0.48), 64, 0.48, Depth::Whole),
            ),
            (
                "fold",
                folded,
                region(around((20.0, 0.0), 64, 1.0), 64, 1.0, Depth::Whole),
            ),
            (
                "edge",
                s,
                region(around((2_480.0, 10.0), 64, 0.48), 64, 0.48, Depth::Whole),
            ),
            (
                "red cell",
                s,
                region(
                    around(red.centre, 64, 0.048),
                    64,
                    0.048,
                    Depth::Range(red.depth - 0.5, red.depth + 0.5),
                ),
            ),
            (
                "nucleus",
                s,
                region(around(n.centre, 64, 0.048), 64, 0.048, Depth::Whole),
            ),
            (
                "nucleus slice",
                s,
                region(
                    around(n.centre, 64, 0.048),
                    64,
                    0.048,
                    Depth::Range(n.depth + 0.5, n.depth + 1.5),
                ),
            ),
        ]
    }

    /// The fingerprints recorded when the model was written (P1.11a). A deliberate change to
    /// the model records them again, and says so in its phase's notes.
    const GOLDEN: [(&str, [f64; 4]); 7] = [
        (
            "overview",
            [
                749.1638391960266,
                3024.180845360899,
                37015.96078688305,
                147606.5736373446,
            ],
        ),
        (
            "gland",
            [
                554.5781022540447,
                1251.2954908898905,
                28363.73687429897,
                61408.00755330179,
            ],
        ),
        (
            "fold",
            [
                165.82605194568168,
                2350.020003245851,
                8553.84920612388,
                113644.72841460988,
            ],
        ),
        ("edge", [0.0, 534.3495314940193, 0.0, 25686.308438347485]),
        (
            "red cell",
            [0.0, 1134.5374239683151, 0.0, 55355.268564105034],
        ),
        (
            "nucleus",
            [
                4687.365117788315,
                1386.7999794334173,
                229002.42960363626,
                67620.95739085972,
            ],
        ),
        (
            "nucleus slice",
            [
                1351.7652901411057,
                346.6999947465956,
                66053.5228087008,
                16905.239337753505,
            ],
        ),
    ];

    #[test]
    fn the_model_renders_its_golden_regions_as_recorded() {
        let mut now = Vec::new();
        let mut wrong = Vec::new();
        for ((name, sample, region), (golden_name, golden)) in golden_regions().iter().zip(GOLDEN) {
            assert_eq!(*name, golden_name);
            let f = fingerprint(&render(sample, region).unwrap());
            // Within a maths library's last bits, and nothing more.
            if f.iter()
                .zip(golden)
                .any(|(a, b)| (a - b).abs() > 1e-5 * b.abs().max(1e-3))
            {
                wrong.push(*name);
            }
            now.push(format!("        ({name:?}, {f:?}),"));
        }
        assert!(
            wrong.is_empty(),
            "{wrong:?} changed; now:\n{}",
            now.join("\n")
        );
    }
}
