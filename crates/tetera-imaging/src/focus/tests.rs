//! The focus measures: their values on images whose answers are known, the green channel of
//! each Bayer pattern, binning, and each measure's peak through a synthetic stack.

use super::*;
use crate::image::MosaicLayout;
use tetera_core::frame::format::BayerPattern;

fn image(width: usize, height: usize, f: impl Fn(usize, usize) -> f64) -> Grey {
    let values = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| f(x, y) as f32)
        .collect();
    Grey::new(width, height, values).unwrap()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

#[test]
fn each_measure_gives_the_known_value_of_a_simple_image() {
    // A ramp across: every difference two apart is 2a.
    let ramp = image(6, 4, |x, _| 3.0 * x as f64 + 7.0);
    assert!(near(brenner(&ramp), 36.0), "{}", brenner(&ramp));
    // A plane: Sobel's gradients are 8a and 8b everywhere. Its size makes no two counts of
    // the interior alike.
    let plane = image(5, 6, |x, y| 2.0 * x as f64 + 3.0 * y as f64 + 1.0);
    assert!(
        near(tenengrad(&plane), 64.0 * 13.0),
        "{}",
        tenengrad(&plane)
    );
    // A checkerboard about 10: the Laplacian is ±8 in equal numbers; the variance is 1.
    let checker = image(4, 4, |x, y| {
        10.0 + if (x + y) % 2 == 0 { 1.0 } else { -1.0 }
    });
    assert!(near(laplacian_variance(&checker), 64.0));
    assert!(near(normalised_variance(&checker), 0.1));
    // A parabola's Laplacian is constant: no variance. Across, and down, where the
    // neighbours above and below count, its mean 6, the interior 3 × 4.
    let bowl = image(5, 5, |x, _| (x * x) as f64);
    assert!(near(laplacian_variance(&bowl), 0.0));
    let trough = image(5, 6, |_, y| 3.0 * (y * y) as f64);
    assert_eq!(laplacian_variance(&trough), 0.0);
    // A flat image is no sharper than any.
    let flat = image(5, 5, |_, _| 100.0);
    for m in FocusMeasure::ALL {
        assert_eq!(m.of(&flat), 0.0, "{m:?}");
    }
    // Normalised variance of an image whose mean is 0 or less is 0.
    assert_eq!(
        normalised_variance(&image(4, 4, |x, _| x as f64 - 1.5)),
        0.0
    );
    assert_eq!(
        normalised_variance(&image(4, 4, |x, _| x as f64 - 9.0)),
        0.0
    );
    assert_eq!(FocusMeasure::default(), FocusMeasure::NormalisedVariance);
    assert_eq!(
        FocusMeasure::ALL.map(FocusMeasure::name),
        [
            "brenner",
            "laplacian_variance",
            "normalised_variance",
            "tenengrad"
        ]
    );
    for m in FocusMeasure::ALL {
        assert_eq!(serde_json::to_value(m).unwrap(), m.name());
    }
}

#[test]
fn a_grey_image_too_small_or_misshapen_is_refused() {
    for (w, h, n) in [(2, 3, 6), (3, 2, 6), (3, 3, 8), (4, 4, 17)] {
        assert!(Grey::new(w, h, vec![0.0; n]).is_err(), "{w} × {h}, {n}");
    }
    let g = Grey::new(3, 4, vec![1.0; 12]).unwrap();
    assert_eq!((g.width(), g.height(), g.values().len()), (3, 4, 12));
}

/// A mosaic of `width` × `height` sites: the greens `g(cell x, cell y, which)`, red 60,000
/// and blue 7.
fn mosaic(
    pattern: BayerPattern,
    width: u32,
    height: u32,
    g: impl Fn(u32, u32, u32) -> u16,
) -> Mosaic {
    let mut sites = Vec::new();
    for y in 0..height {
        for x in 0..width {
            sites.push(match pattern.channel_at(x, y) {
                Channel::Red => 60_000,
                Channel::Blue => 7,
                Channel::GreenOnRed => g(x / 2, y / 2, 0),
                Channel::GreenOnBlue => g(x / 2, y / 2, 1),
            });
        }
    }
    let layout = MosaicLayout {
        width,
        height,
        pattern,
        significant_bits: 16,
    };
    Mosaic::new(layout, sites).unwrap()
}

#[test]
fn the_green_channel_is_each_cells_two_greens_averaged_for_every_pattern() {
    for pattern in [
        BayerPattern::Rggb,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
        BayerPattern::Bggr,
    ] {
        let m = mosaic(pattern, 10, 8, |x, y, which| {
            (100 * x + 10 * y + 4 * which) as u16
        });
        let g = green(&m, 512).unwrap();
        assert_eq!((g.width(), g.height()), (5, 4), "{pattern:?}");
        for y in 0..4 {
            for x in 0..5 {
                let expected = (100 * x + 10 * y) as f32 + 2.0;
                assert_eq!(g.values()[y * 5 + x], expected, "{pattern:?} at {x}, {y}");
            }
        }
    }
}

#[test]
fn the_green_channel_is_binned_to_fit_and_a_part_block_left_out() {
    // 23 × 9 sites: 11 × 4 whole cells, the odd column and row left out.
    let m = mosaic(BayerPattern::Rggb, 23, 9, |x, y, _| (x + 11 * y) as u16);
    let whole = green(&m, 11).unwrap();
    assert_eq!((whole.width(), whole.height()), (11, 4));
    // At most 5 across: binned 3 × 3, 3 × 1 blocks, the cells beyond left out.
    let binned = green(&m, 5).unwrap_err();
    assert_eq!(
        binned.kind(),
        tetera_core::ErrorKind::InvalidArgument,
        "1 row"
    );
    let m = mosaic(BayerPattern::Grbg, 24, 20, |x, y, _| (x + 12 * y) as u16);
    let binned = green(&m, 5).unwrap();
    assert_eq!(
        (binned.width(), binned.height()),
        (4, 3),
        "12 × 10 cells by 3"
    );
    // The first block: cells 0 to 2 across and down; their mean.
    let first: f64 = (0..3)
        .flat_map(|y| (0..3).map(move |x| f64::from(x + 12 * y)))
        .sum::<f64>()
        / 9.0;
    assert!(near(f64::from(binned.values()[0]), first));
    let last: f64 = (6..9)
        .flat_map(|y| (9..12).map(move |x| f64::from(x + 12 * y)))
        .sum::<f64>()
        / 9.0;
    assert!(near(f64::from(binned.values()[11]), last));
    // Exactly the width allowed isn't binned.
    assert_eq!(green(&m, 12).unwrap().width(), 12);
    assert_eq!(green(&m, 11).unwrap().width(), 6);
    assert!(green(&m, 2).is_err());
    assert!(green(&m, 0).is_err(), "refused, not divided by");
    let square = mosaic(BayerPattern::Rggb, 24, 24, |x, y, _| (x + y) as u16);
    let three = green(&square, 3).unwrap();
    assert_eq!((three.width(), three.height()), (3, 3), "binned 4 × 4");
    let tiny = mosaic(BayerPattern::Rggb, 4, 4, |_, _, _| 1);
    assert!(green(&tiny, 512).is_err(), "2 × 2 cells");
}

/// A texture with edges and detail at several scales, from a fixed seed.
fn texture(width: usize, height: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / u64::MAX as f64
    };
    let blobs: Vec<(f64, f64, f64, f64)> = (0..40)
        .map(|_| {
            (
                next() * width as f64,
                next() * height as f64,
                2.0 + 10.0 * next(),
                next() - 0.5,
            )
        })
        .collect();
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x as f64, y as f64)))
        .map(|(x, y)| {
            let mut v = 1_000.0
                + if (x / 16.0).floor() as i64 % 2 == 0 {
                    150.0
                } else {
                    0.0
                };
            for &(bx, by, r, a) in &blobs {
                if (x - bx).hypot(y - by) < r {
                    v += 400.0 * a;
                }
            }
            v
        })
        .collect()
}

/// `values` blurred by a Gaussian of `sigma` values, each way, its edges mirrored.
fn blurred(values: &[f64], width: usize, height: usize, sigma: f64) -> Vec<f64> {
    if sigma < 1e-6 {
        return values.to_vec();
    }
    let reach = (3.0 * sigma).ceil() as i64;
    let weights: Vec<f64> = (-reach..=reach)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let pass = |src: &[f64], across: bool| -> Vec<f64> {
        let mut out = vec![0.0; src.len()];
        for y in 0..height as i64 {
            for x in 0..width as i64 {
                let mut s = 0.0;
                for (k, w) in weights.iter().enumerate() {
                    let o = k as i64 - reach;
                    let (sx, sy) = if across { (x + o, y) } else { (x, y + o) };
                    let m = |i: i64, n: i64| {
                        let i = i.abs();
                        if i >= n { 2 * (n - 1) - i } else { i }
                    };
                    let (sx, sy) = (m(sx, width as i64), m(sy, height as i64));
                    s += w * src[(sy * width as i64 + sx) as usize];
                }
                out[(y * width as i64 + x) as usize] = s / total;
            }
        }
        out
    };
    pass(&pass(values, true), false)
}

#[test]
fn each_measure_peaks_at_a_synthetic_stacks_focus_and_falls_away_on_both_sides() {
    let (w, h) = (96, 72);
    let sharp = texture(w, h);
    let focus = 3i32;
    let mut noise = 0x1234_5678u64;
    let stack: Vec<Grey> = (-6..=10)
        .map(|z: i32| {
            let sigma = 0.7 * f64::from((z - focus).abs());
            let mut v = blurred(&sharp, w, h, sigma);
            // The sensor's noise, about 0.2 % of the signal.
            for s in &mut v {
                noise ^= noise << 13;
                noise ^= noise >> 7;
                noise ^= noise << 17;
                *s += 2.0 * ((noise % 1_000) as f64 / 1_000.0 - 0.5);
            }
            Grey::new(w, h, v.into_iter().map(|x| x as f32).collect()).unwrap()
        })
        .collect();
    for m in FocusMeasure::ALL {
        let values: Vec<f64> = stack.iter().map(|g| m.of(g)).collect();
        let best = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i32 - 6)
            .unwrap();
        assert_eq!(best, focus, "{m:?}: {values:?}");
        let at = |z: i32| values[(z + 6) as usize];
        for d in 1..=4 {
            assert!(
                at(focus + d) < at(focus + d - 1),
                "{m:?} above, {d}: {values:?}"
            );
            assert!(
                at(focus - d) < at(focus - d + 1),
                "{m:?} below, {d}: {values:?}"
            );
        }
    }
}

#[test]
fn a_mosaics_focus_is_its_greens() {
    // The greens a sharp checkerboard, red and blue flat: sharp. The greens flat: none.
    let sharp = mosaic(BayerPattern::Rggb, 64, 64, |x, y, _| {
        if (x / 2 + y / 2) % 2 == 0 {
            3_000
        } else {
            1_000
        }
    });
    let flat = mosaic(BayerPattern::Rggb, 64, 64, |_, _, _| 2_000);
    for m in FocusMeasure::ALL {
        assert!(m.of_mosaic(&sharp).unwrap() > 0.0, "{m:?}");
        assert_eq!(
            m.of_mosaic(&flat).unwrap(),
            0.0,
            "{m:?}: red and blue don't count"
        );
    }
    assert!(
        FocusMeasure::Brenner
            .of_mosaic(&mosaic(BayerPattern::Rggb, 4, 4, |_, _, _| 1))
            .is_err()
    );
}
