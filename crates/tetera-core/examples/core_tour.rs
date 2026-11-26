//! A tour of tetera-core's types: units, geometry, the clock, IDs and errors.
//!
//! Run it with `cargo run -p tetera-core --example core_tour`; `-- --smoke` does the same
//! and says so.

use tetera_core::geometry::{Micrometres, Pixels};
use tetera_core::time::drift_exceeds_limit;
use tetera_core::{
    Affine2, Clock, Error, Id, PointPx, Rect, StagePos, SystemClock, Um, VectorUm,
    name_and_version, units,
};

/// What the tour's IDs name.
struct Dataset;

fn main() {
    let smoke = std::env::args().any(|arg| arg == "--smoke");
    println!("{}\n", name_and_version());

    let travel = Um::from_mm(25.0);
    let pixel = Um(2.4);
    println!(
        "units:    {travel} of travel is {:.0} sensor pixels of {pixel}",
        travel / pixel
    );
    let exposure = units::duration_from_ms(12.5).expect("12.5 ms is a valid exposure");
    println!(
        "units:    12.5 ms is {exposure:?}; and back, {} ms",
        units::duration_ms(exposure)
    );

    // The camera-to-stage map at 50×: 0.048 µm a pixel, the camera turned by half a degree.
    let px_to_um = Affine2::<Pixels, Micrometres>::similarity(
        0.048,
        0.5_f64.to_radians(),
        VectorUm::new(1_000.0, 2_000.0),
    );
    let corner = PointPx::new(5_472.0, 3_648.0);
    let at = px_to_um.apply(corner);
    println!(
        "geometry: pixel ({}, {}) is at ({:.3}, {:.3}) µm on the stage",
        corner.x, corner.y, at.x, at.y
    );
    let back = px_to_um
        .inverse()
        .expect("a similarity has an inverse")
        .apply(at);
    println!("geometry: and back, ({:.6}, {:.6}) px", back.x, back.y);
    let sensor = Rect::new(0, 0, 5_472, 3_648);
    let roi = Rect::new(5_000, 3_000, 1_000, 1_000);
    println!(
        "geometry: an ROI past the sensor's corner keeps {:?}",
        sensor.intersection(&roi)
    );
    let from = StagePos::new(Um(0.0), Um(0.0), Um(100.0));
    let to = StagePos::new(Um(3_000.0), Um(4_000.0), Um(100.0));
    println!("geometry: the move is {:.1} long", from.distance(&to));

    let clock = SystemClock::new();
    let now = clock.now();
    println!(
        "clock:    {} ns since it started, at {} UTC",
        now.0,
        clock.utc(now)
    );
    let drift = clock.drift();
    println!(
        "clock:    the wall clock is {drift} ns from the mapping; beyond the limit: {}",
        drift_exceeds_limit(drift)
    );

    let (first, second) = (Id::<Dataset>::new(), Id::<Dataset>::new());
    println!(
        "ids:      {first}\n          {second}\n          in order: {}",
        first < second
    );

    let err = Error::busy("the camera is streaming")
        .context("setting the exposure")
        .context("capturing a burst");
    println!("errors:   {err} (kind {})", err.kind());

    if smoke {
        println!("\nsmoke: ok");
    }
}
