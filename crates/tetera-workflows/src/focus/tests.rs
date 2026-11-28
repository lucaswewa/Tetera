//! The search, through synthetic probes: known curves with noise, a stage with backlash and
//! steps, flat curves and peaks beyond the range, refusals, and the parabola's fit.

use super::synthetic::{Shape, Synthetic};
use super::*;

fn going(_: f64, _: &str) -> Result<()> {
    Ok(())
}

const TRAVEL: (f64, f64) = (0.0, 10_000.0);

/// The search, 100 times at 50×'s depth of field through the virtual stage's steps and 2 µm
/// of backlash, from starts within ±6 d, on Gaussian and Lorentzian peaks of `widths` depths
/// of field and `noise`: the trials within a quarter of the depth of field, and the worst
/// error, µm. The backlash shifts the window the carriage sees by its own size, so ±8 d
/// leaves room for 1.5 d.
fn trials(widths: &[f64], noise: f64) -> (usize, f64) {
    let d = 1.9;
    let mut within = 0;
    let mut worst = 0.0f64;
    for trial in 0..100u64 {
        let mut offsets = Synthetic::new(0.0, 0.0, 1_000 + trial);
        let offset = (offsets.uniform() * 2.0 - 1.0) * 6.0 * d;
        let mut probe = Synthetic::new(1_000.0, 1_000.0 + offset, trial);
        probe.backlash = 2.0;
        probe.resolution = 0.3125;
        probe.noise = noise;
        probe.width = d * widths[trial as usize % widths.len()];
        probe.shape = if trial % 2 == 0 {
            Shape::Gaussian
        } else {
            Shape::Lorentzian
        };
        let focus = search(&mut probe, d, TRAVEL, &SearchOptions::default(), &mut going).unwrap();
        let z = focus.found().unwrap();
        assert_eq!(
            probe.z().unwrap(),
            (z / 0.3125).round() * 0.3125,
            "the stage at the focus"
        );
        let error = (probe.carriage() - probe.focus).abs();
        worst = worst.max(error);
        within += usize::from(error <= d / 4.0);
        assert!(
            probe.measured_rising.iter().all(|&r| r),
            "every measure from below"
        );
        assert_eq!(focus.curve.len(), 17 + 9);
    }
    (within, worst)
}

#[test]
fn the_search_finds_a_known_peak_within_a_quarter_of_the_depth_of_field_in_99_of_100() {
    // As sharp as the gradient measures' on the simulator, with 1 % noise.
    let (within, worst) = trials(&[0.75, 1.0, 1.25], 0.01);
    assert!(
        within >= 99,
        "{within} of 100 within d / 4; the worst {worst:.3} µm"
    );
    // As broad as normalised variance's, and broader, with its smaller noise: the fine
    // sweep's reach and its fit of the top half see the flat top's fall.
    let (within, worst) = trials(&[2.5, 4.0, 5.0, 6.0], 0.002);
    assert!(
        within >= 99,
        "{within} of 100 within d / 4; the worst {worst:.3} µm"
    );
}

#[test]
fn backlash_is_taken_up_before_each_sweep_and_the_focus() {
    let d = 7.9;
    let mut probe = Synthetic::new(2_000.0, 2_013.0, 7);
    probe.backlash = 6.0;
    probe.width = d;
    let options = SearchOptions::default();
    let focus = search(&mut probe, d, TRAVEL, &options, &mut going).unwrap();
    // The carriage lags the screw by the backlash, measured and final alike.
    assert!(
        (probe.carriage() - 2_013.0).abs() <= d / 20.0,
        "{}",
        probe.carriage()
    );
    assert!((focus.z.unwrap() - 2_019.0).abs() <= d / 20.0, "{focus:?}");
    // Each sweep and the focus: down by the take-up, then up.
    let downs: Vec<(f64, f64)> = probe
        .moves
        .iter()
        .copied()
        .filter(|(from, to)| to < from)
        .collect();
    assert_eq!(
        downs.len(),
        3,
        "the coarse sweep, the fine, the focus: {downs:?}"
    );
    for (i, &(from, to)) in probe.moves.iter().enumerate() {
        if to < from {
            let (up_from, up_to) = probe.moves[i + 1];
            assert_eq!(up_from, to);
            assert!(
                (up_to - to - options.take_up_um).abs() < 1e-9,
                "taken up by 10 µm"
            );
        }
    }
    assert!(probe.measured_rising.iter().all(|&r| r));
    // The curve: the coarse sweep around the start, then the fine around its peak.
    let coarse: Vec<f64> = focus
        .curve
        .iter()
        .filter(|p| p.sweep == Sweep::Coarse)
        .map(|p| p.z)
        .collect();
    assert_eq!(coarse.len(), 17);
    assert!(
        (coarse[0] - (2_000.0 - 8.0 * d)).abs() < 1e-9
            && (coarse[16] - (2_000.0 + 8.0 * d)).abs() < 1e-9
    );
    let fine: Vec<f64> = focus
        .curve
        .iter()
        .filter(|p| p.sweep == Sweep::Fine)
        .map(|p| p.z)
        .collect();
    assert_eq!(fine.len(), 9);
    assert!(
        fine.windows(2)
            .all(|w| (w[1] - w[0] - d / 2.0).abs() < 1e-9)
    );
    assert!(
        (fine[4] - (2_000.0 + 2.0 * d)).abs() < 1e-9,
        "around the coarse peak: {fine:?}"
    );
    assert_eq!((focus.start, focus.depth_of_field), (2_000.0, d));
    assert!(focus.contrast > 1.0, "{}", focus.contrast);
}

#[test]
fn a_flat_curve_finds_no_focus_and_the_stage_goes_back_to_its_start_from_below() {
    let mut probe = Synthetic::new(1_500.0, 1_500.0, 3);
    probe.shape = Shape::Flat;
    probe.noise = 0.01;
    probe.backlash = 4.0;
    let focus = search(
        &mut probe,
        3.4,
        TRAVEL,
        &SearchOptions::default(),
        &mut going,
    )
    .unwrap();
    assert_eq!(focus.z, None);
    let why = focus.why_not.clone().unwrap();
    assert!(why.contains("flat"), "{why}");
    assert_eq!(probe.z().unwrap(), 1_500.0, "back at the start");
    assert!(
        probe.moves.last().is_some_and(|(from, to)| to > from),
        "from below"
    );
    assert_eq!(focus.curve.len(), 17, "no fine sweep");
    assert!(focus.contrast < 0.05, "{}", focus.contrast);
    let e = focus.found().unwrap_err();
    assert_eq!(e.kind(), tetera_core::ErrorKind::NotFound);
    assert!(e.to_string().contains("flat"), "{e}");
    // A curve rising just past the least contrast is a focus; one just short isn't.
    for (height, found) in [(0.06, true), (0.04, false)] {
        let mut probe = Synthetic::new(1_500.0, 1_500.0, 3);
        probe.height = height;
        probe.width = 3.4;
        let focus = search(
            &mut probe,
            3.4,
            TRAVEL,
            &SearchOptions::default(),
            &mut going,
        )
        .unwrap();
        assert_eq!(focus.z.is_some(), found, "{height}: {focus:?}");
    }
    // Nothing measured but 0: flat.
    let mut probe = Synthetic::new(1_500.0, 1_500.0, 3);
    probe.base = 0.0;
    probe.shape = Shape::Flat;
    let focus = search(
        &mut probe,
        3.4,
        TRAVEL,
        &SearchOptions::default(),
        &mut going,
    )
    .unwrap();
    assert_eq!((focus.z, focus.contrast), (None, 0.0));
    // A median of 0 or below and a peak above it: no contrast to divide, a focus.
    let mut probe = Synthetic::new(1_500.0, 1_500.0, 3);
    probe.base = -1.0;
    probe.width = 1.0;
    let focus = search(
        &mut probe,
        3.4,
        TRAVEL,
        &SearchOptions::default(),
        &mut going,
    )
    .unwrap();
    assert_eq!(focus.contrast, f64::INFINITY);
    assert!(focus.z.is_some());
}

#[test]
fn a_peak_at_the_ranges_end_finds_no_focus() {
    for (focus_at, end) in [(1_500.0 + 14.0 * 3.4, "16"), (1_500.0 - 14.0 * 3.4, "0")] {
        let mut probe = Synthetic::new(1_500.0, focus_at, 5);
        probe.width = 4.0 * 3.4;
        let focus = search(
            &mut probe,
            3.4,
            TRAVEL,
            &SearchOptions::default(),
            &mut going,
        )
        .unwrap();
        assert_eq!(focus.z, None, "{end}");
        let why = focus.why_not.unwrap();
        assert!(why.contains("range's end"), "{why}");
        assert_eq!(probe.z().unwrap(), 1_500.0);
        // The peak next to the end is fine.
        let inside = if focus_at > 1_500.0 {
            1_500.0 + 7.0 * 3.4
        } else {
            1_500.0 - 7.0 * 3.4
        };
        let mut probe = Synthetic::new(1_500.0, inside, 5);
        probe.width = 3.4;
        let focus = search(
            &mut probe,
            3.4,
            TRAVEL,
            &SearchOptions::default(),
            &mut going,
        )
        .unwrap();
        assert!(
            (focus.z.unwrap() - inside).abs() < 3.4 / 4.0,
            "{end}: {focus:?}"
        );
    }
}

#[test]
fn bad_options_and_a_range_beyond_the_travel_are_refused_before_a_move() {
    let base = SearchOptions::default();
    for bad in [
        SearchOptions { range: 0.0, ..base },
        SearchOptions {
            range: f64::NAN,
            ..base
        },
        SearchOptions {
            coarse_step: 0.0,
            ..base
        },
        SearchOptions {
            coarse_step: 9.0,
            ..base
        },
        SearchOptions {
            fine_range: -1.0,
            ..base
        },
        SearchOptions {
            fine_step: 0.0,
            ..base
        },
        SearchOptions {
            fine_step: 3.0,
            ..base
        },
        SearchOptions {
            take_up_um: -1.0,
            ..base
        },
        SearchOptions {
            take_up_um: f64::INFINITY,
            ..base
        },
        SearchOptions {
            min_contrast: -0.1,
            ..base
        },
        SearchOptions {
            min_contrast: f64::NAN,
            ..base
        },
    ] {
        let mut probe = Synthetic::new(1_000.0, 1_000.0, 1);
        let e = search(&mut probe, 2.0, TRAVEL, &bad, &mut going).unwrap_err();
        assert_eq!(e.kind(), tetera_core::ErrorKind::InvalidArgument, "{bad:?}");
        assert!(probe.moves.is_empty());
    }
    // Steps as long as their ranges are fine.
    let edge = SearchOptions {
        coarse_step: 8.0,
        fine_step: 2.0,
        take_up_um: 0.0,
        min_contrast: 0.0,
        ..base
    };
    assert!(edge.check().is_ok());
    for d in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut probe = Synthetic::new(1_000.0, 1_000.0, 1);
        assert!(
            search(&mut probe, d, TRAVEL, &base, &mut going).is_err(),
            "{d}"
        );
    }
    // ±8 d, a fine reach of 2 d below the first, and the take-up: 1,000 − 16 − 4 − 10 = 970.
    for (limits, fits) in [
        ((970.0, 1_020.0), true),
        ((970.1, 1_020.0), false),
        ((970.0, 1_019.9), false),
    ] {
        let mut probe = Synthetic::new(1_000.0, 1_000.0, 1);
        let r = search(&mut probe, 2.0, limits, &base, &mut going);
        assert_eq!(r.is_ok(), fits, "{limits:?}: {r:?}");
        assert_eq!(probe.moves.is_empty(), !fits);
    }
}

#[test]
fn the_step_says_where_the_search_is_and_its_error_stops_it() {
    let mut probe = Synthetic::new(1_000.0, 1_004.0, 2);
    let mut seen = Vec::new();
    search(
        &mut probe,
        2.0,
        TRAVEL,
        &SearchOptions::default(),
        &mut |done, msg| {
            seen.push((done, msg.to_string()));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen.len(), 17 + 9 + 1);
    assert_eq!(seen[0].0, 0.0);
    assert!(seen.windows(2).all(|w| w[1].0 >= w[0].0));
    assert!((seen[17].0 - 17.0 / 26.0).abs() < 1e-12, "{:?}", seen[17]);
    assert_eq!(seen[26].0, 1.0);
    assert!(
        seen[0].1.starts_with("coarse sweep: 1 of 17"),
        "{}",
        seen[0].1
    );
    assert!(
        seen[17].1.starts_with("fine sweep: 1 of 9"),
        "{}",
        seen[17].1
    );
    assert!(seen[26].1.starts_with("to the focus"), "{}", seen[26].1);
    // Stopped at the fifth position: no more moves.
    let mut probe = Synthetic::new(1_000.0, 1_004.0, 2);
    let mut calls = 0;
    let e = search(
        &mut probe,
        2.0,
        TRAVEL,
        &SearchOptions::default(),
        &mut |_, _| {
            calls += 1;
            if calls == 5 {
                Err(Error::cancelled("stopped"))
            } else {
                Ok(())
            }
        },
    )
    .unwrap_err();
    assert_eq!(e.kind(), tetera_core::ErrorKind::Cancelled);
    assert_eq!(probe.measured_rising.len(), 4);
}

#[test]
fn the_vertex_is_the_parabolas_through_the_peaks_logarithm() {
    // A Gaussian's logarithm is a parabola: exact, wherever the peak lies between points.
    for peak in [0.0, 0.1, -0.37, 0.49] {
        let points: Vec<(f64, f64)> = (-4..=4)
            .map(|i| {
                let z = f64::from(i) * 0.5;
                (z, 5.0 * (-(z - peak).powi(2) / 2.0).exp())
            })
            .collect();
        assert!(
            (vertex(&points) - peak).abs() < 1e-9,
            "{peak}: {}",
            vertex(&points)
        );
    }
    // At the first point: three are fitted, the vertex among them.
    let edge: Vec<(f64, f64)> = (0..5)
        .map(|i| {
            let z = f64::from(i);
            (z, (-(z - 0.3).powi(2)).exp())
        })
        .collect();
    assert!((vertex(&edge) - 0.3).abs() < 1e-9, "{}", vertex(&edge));
    // At the last.
    let last: Vec<(f64, f64)> = (0..5)
        .map(|i| {
            let z = f64::from(i);
            (z, (-(z - 3.8).powi(2)).exp())
        })
        .collect();
    assert!((vertex(&last) - 3.8).abs() < 1e-9, "{}", vertex(&last));
    // Opening upward, or flat: the highest point.
    let valley = [(0.0, 3.0), (1.0, 1.0), (2.0, 2.0), (3.0, 4.0), (4.0, 9.0)];
    assert_eq!(vertex(&valley), 4.0);
    let flat = [(0.0, 2.0), (1.0, 2.0), (2.0, 2.0)];
    assert_eq!(vertex(&flat), 0.0, "the first of equals");
    // Still rising at the last point: the vertex lies beyond, so the highest point.
    let rising = [(0.0, 1.0), (1.0, 1.1), (2.0, 1.2), (3.0, 1.3), (4.0, 1.4)];
    assert_eq!(vertex(&rising), 4.0);
    // Two points: the higher.
    assert_eq!(vertex(&[(0.0, 1.0), (1.0, 2.0)]), 1.0);
    // A value of 0 or below among those fitted: the measure itself, a parabola peaking at 2.3.
    let raw: Vec<(f64, f64)> = (0..5)
        .map(|i| (f64::from(i), 0.1 - 0.1 * (f64::from(i) - 2.3).powi(2)))
        .collect();
    assert!(raw[1].1 < 0.0);
    assert!((vertex(&raw) - 2.3).abs() < 1e-9, "{}", vertex(&raw));
    // A singular system: every z the same.
    assert_eq!(vertex(&[(1.0, 1.0), (1.0, 3.0), (1.0, 2.0)]), 1.0);
    // The window: the top half around the peak, a lower point beyond it left out.
    let dip = [
        (0.0, 9.0),
        (1.0, 1.0),
        (2.0, 5.0),
        (3.0, 10.0),
        (4.0, 6.0),
        (5.0, 1.0),
    ];
    let fitted = vertex(&dip);
    assert!(fitted > 2.0 && fitted < 4.0, "{fitted}");
}

#[test]
fn positions_and_the_median_are_as_said() {
    assert_eq!(positions(10.0, 2.0, 1.0), [8.0, 9.0, 10.0, 11.0, 12.0]);
    assert_eq!(positions(0.0, 1.0, 0.25).len(), 9);
    assert_eq!(
        positions(5.0, 0.4, 1.0),
        [5.0],
        "a step beyond the reach: the centre"
    );
    assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    assert_eq!(highest_of(&[1.0, 3.0, 2.0, 3.0]), (1, 3.0));
}

#[test]
fn the_synthetic_stage_has_backlash_and_steps_and_its_noise_is_normal() {
    let mut p = Synthetic::new(10.0, 0.0, 9);
    p.backlash = 2.0;
    p.resolution = 0.5;
    p.move_to(13.2).unwrap();
    assert_eq!(
        (p.z().unwrap(), p.carriage()),
        (13.0, 11.0),
        "up: the carriage 2 µm behind"
    );
    p.move_to(12.4).unwrap();
    assert_eq!(
        (p.z().unwrap(), p.carriage()),
        (12.5, 11.0),
        "down, within the play: still"
    );
    p.move_to(10.0).unwrap();
    assert_eq!(p.carriage(), 10.0, "down past it: pushed");
    p.move_to(10.0).unwrap();
    assert_eq!(p.moves.len(), 4);
    // The measure: (1 + 3 · shape) at the carriage, 2 µm wide.
    let mut p = Synthetic::new(1.0, 0.0, 9);
    assert!((p.measure().unwrap() - (1.0 + 3.0 * (-0.125f64).exp())).abs() < 1e-12);
    p.shape = Shape::Lorentzian;
    assert!((p.measure().unwrap() - (1.0 + 3.0 / 1.25)).abs() < 1e-12);
    p.shape = Shape::Flat;
    assert_eq!(p.measure().unwrap(), 1.0);
    assert_eq!(p.measured_rising, [false, false, false]);
    // Its noise: mean 0, standard deviation 1, from numbers in [0, 1).
    p.noise = 1.0;
    let n = 20_000;
    let draws: Vec<f64> = (0..n).map(|_| p.measure().unwrap() - 1.0).collect();
    let mean = draws.iter().sum::<f64>() / n as f64;
    let sd = (draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    assert!(mean.abs() < 0.03 && (sd - 1.0).abs() < 0.03, "{mean} {sd}");
    let u: Vec<f64> = (0..n).map(|_| p.uniform()).collect();
    assert!(u.iter().all(|&x| (0.0..1.0).contains(&x)));
    let below = u.iter().filter(|&&x| x < 0.25).count() as f64 / n as f64;
    assert!((below - 0.25).abs() < 0.02, "{below}");
}
