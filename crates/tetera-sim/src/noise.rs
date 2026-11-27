//! Procedural texture (P1.11a): integer hashing, improved noise (Perlin, 2002), cellular noise
//! (Worley, 1996) and fBm. Every function is pure, of a seed, a salt for each use and a
//! position, so the GPU computes the same values (ADR-0015).
//!
//! Positions are in µm, as `f64`: each is split into its lattice cell and the offset within
//! it in double precision, and everything after is single precision.

/// A seed, folded to 32 bits for hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Seed(pub u32);

impl Seed {
    /// A 64-bit seed, both its halves folded in.
    pub fn new(seed: u64) -> Self {
        let (lo, hi) = (seed as u32, (seed >> 32) as u32);
        Self(lowbias32(lo ^ lowbias32(hi ^ 0x9e37_79b9)))
    }
}

/// C. Wellons's "lowbias32" finaliser (hash-prospector, 2018).
fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// A hash of the lattice point (`x`, `y`, `z`), for the use `salt`.
pub fn hash(seed: Seed, salt: u32, x: i32, y: i32, z: i32) -> u32 {
    let mut h = lowbias32(seed.0 ^ salt.wrapping_mul(0x9e37_79b9));
    for v in [x, y, z] {
        h = lowbias32(h ^ v as u32);
    }
    h
}

/// A hash as a float in [0, 1): its top 24 bits, exact in single precision.
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// `x / wavelength` split into its lattice cell and the offset within it.
fn lattice(x: f64, wavelength: f64) -> (i32, f32) {
    let f = x / wavelength;
    let cell = f.floor();
    (cell as i64 as i32, (f - cell) as f32)
}

/// The 12 gradients of improved noise, toward the cube's edge midpoints, without their z:
/// in 2D, what is left of each.
const GRADIENTS: [(f32, f32); 12] = [
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (0.0, -1.0),
];

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Improved noise at (`x`, `y`) µm, on a lattice of `wavelength` µm: about −1 to 1, and 0 at
/// every lattice point.
pub fn improved(seed: Seed, salt: u32, x: f64, y: f64, wavelength: f64) -> f32 {
    let (ix, tx) = lattice(x, wavelength);
    let (iy, ty) = lattice(y, wavelength);
    let corner = |cx: i32, cy: i32, dx: f32, dy: f32| {
        let (gx, gy) = GRADIENTS
            [(hash(seed, salt, ix.wrapping_add(cx), iy.wrapping_add(cy), 0) % 12) as usize];
        gx * dx + gy * dy
    };
    let (u, v) = (fade(tx), fade(ty));
    lerp(
        lerp(corner(0, 0, tx, ty), corner(1, 0, tx - 1.0, ty), u),
        lerp(
            corner(0, 1, tx, ty - 1.0),
            corner(1, 1, tx - 1.0, ty - 1.0),
            u,
        ),
        v,
    )
}

/// fBm: `octaves` of improved noise from `wavelength` µm, each at twice the frequency and
/// half the amplitude of the last.
pub fn fbm(seed: Seed, salt: u32, x: f64, y: f64, wavelength: f64, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut w = wavelength;
    for octave in 0..octaves {
        sum += amplitude * improved(seed, salt.wrapping_add(octave), x, y, w);
        amplitude *= 0.5;
        w *= 0.5;
    }
    sum
}

/// Where a point lies among the feature points of cellular noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cellular {
    /// The distance to the nearest feature point, µm.
    pub f1: f32,
    /// The distance to the second nearest, µm.
    pub f2: f32,
    /// The nearest point's cell's hash, for what its feature holds.
    pub hash: u32,
    /// The point, from the nearest feature point, µm.
    pub offset: (f32, f32),
}

/// Cellular noise at (`x`, `y`) µm, on cells of `cell` µm, one feature point in each, over the
/// 3 × 3 cells around.
pub fn cellular(seed: Seed, salt: u32, x: f64, y: f64, cell: f64) -> Cellular {
    let (ix, tx) = lattice(x, cell);
    let (iy, ty) = lattice(y, cell);
    let size = cell as f32;
    let mut best = Cellular {
        f1: f32::INFINITY,
        f2: f32::INFINITY,
        hash: 0,
        offset: (0.0, 0.0),
    };
    for cy in -1..=1 {
        for cx in -1..=1 {
            let h = hash(seed, salt, ix.wrapping_add(cx), iy.wrapping_add(cy), 0);
            let fx = cx as f32 + unit(h);
            let fy = cy as f32 + unit(lowbias32(h));
            let (dx, dy) = ((tx - fx) * size, (ty - fy) * size);
            let d = (dx * dx + dy * dy).sqrt();
            if d < best.f1 {
                best.f2 = best.f1;
                best.f1 = d;
                best.hash = h;
                best.offset = (dx, dy);
            } else if d < best.f2 {
                best.f2 = d;
            }
        }
    }
    best
}

/// 0 below `a`, 1 above `b`, smooth between; with `a > b`, the other way round.
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: Seed = Seed(0x1234_5678);

    #[test]
    fn a_hash_is_the_same_each_time_and_each_seed_bit_matters() {
        assert_eq!(hash(SEED, 1, 2, 3, 4), hash(SEED, 1, 2, 3, 4));
        assert_ne!(hash(SEED, 1, 2, 3, 4), hash(SEED, 2, 2, 3, 4), "the salt");
        assert_ne!(hash(SEED, 1, 2, 3, 4), hash(SEED, 1, 2, 3, 5), "z");
        // Flipping any one of the 64 seed bits changes about half the hash's bits.
        let base: u64 = 0x0123_4567_89ab_cdef;
        for bit in 0..64 {
            let (a, b) = (Seed::new(base), Seed::new(base ^ (1 << bit)));
            let changed: u32 = (0..200)
                .map(|i| (hash(a, 7, i, -i, 3 * i) ^ hash(b, 7, i, -i, 3 * i)).count_ones())
                .sum();
            let mean = f64::from(changed) / 200.0;
            assert!((14.0..=18.0).contains(&mean), "seed bit {bit}: {mean} bits");
        }
    }

    #[test]
    fn units_fall_in_zero_to_one_about_a_half_on_average() {
        assert_eq!(unit(0), 0.0);
        assert!(unit(u32::MAX) < 1.0);
        assert_eq!(unit(u32::MAX), 16_777_215.0 / 16_777_216.0);
        let mean: f64 = (0..10_000)
            .map(|i| f64::from(unit(hash(SEED, 0, i, 0, 0))))
            .sum::<f64>()
            / 10_000.0;
        assert!((mean - 0.5).abs() < 0.01, "{mean}");
    }

    #[test]
    fn improved_noise_is_zero_on_its_lattice_bounded_and_continuous() {
        for i in -20..20 {
            assert_eq!(
                improved(SEED, 3, f64::from(i) * 7.0, f64::from(i * i) * 7.0, 7.0),
                0.0
            );
        }
        let mut sum = 0.0;
        let n = 200;
        for i in 0..n {
            for j in 0..n {
                let (x, y) = (f64::from(i) * 1.37 + 0.11, f64::from(j) * 1.91 - 50_000.3);
                let v = improved(SEED, 3, x, y, 10.0);
                assert!((-1.1..=1.1).contains(&v), "{v}");
                let step = improved(SEED, 3, x + 1e-3, y, 10.0);
                assert!((v - step).abs() < 1e-2, "{v} then {step}");
                sum += f64::from(v);
            }
        }
        assert!(
            (sum / f64::from(n * n)).abs() < 0.05,
            "mean {}",
            sum / f64::from(n * n)
        );
        // Far out on a 100 mm slide, as fine as near the origin.
        let (a, b) = (
            improved(SEED, 3, 99_999.0, 0.3, 0.7),
            improved(SEED, 3, 99_999.05, 0.3, 0.7),
        );
        assert_ne!(a, b);
    }

    #[test]
    fn fbm_adds_octaves_at_half_the_amplitude() {
        let (x, y) = (123.4, -567.8);
        let two = improved(SEED, 9, x, y, 100.0) + 0.5 * improved(SEED, 10, x, y, 50.0);
        assert_eq!(fbm(SEED, 9, x, y, 100.0, 2), two);
        assert_eq!(fbm(SEED, 9, x, y, 100.0, 1), improved(SEED, 9, x, y, 100.0));
        assert_eq!(fbm(SEED, 9, x, y, 100.0, 0), 0.0);
    }

    #[test]
    fn cellular_noise_orders_its_distances_and_is_zero_at_a_feature_point() {
        for i in 0..500 {
            let (x, y) = (f64::from(i) * 3.3 - 700.0, f64::from(i) * 1.7 + 20.0);
            let c = cellular(SEED, 5, x, y, 20.0);
            assert!(c.f1 <= c.f2, "{c:?}");
            assert!(c.f1 <= 20.0 * 2.0f32.sqrt(), "{c:?}");
            assert!(((c.offset.0.powi(2) + c.offset.1.powi(2)).sqrt() - c.f1).abs() < 1e-3);
            // At its nearest feature point, it is 0, and the same cell.
            let at = cellular(
                SEED,
                5,
                x - f64::from(c.offset.0),
                y - f64::from(c.offset.1),
                20.0,
            );
            assert!(at.f1 < 1e-3 && at.hash == c.hash, "{at:?}");
        }
    }

    #[test]
    fn smoothstep_rises_or_falls_between_its_ends() {
        assert_eq!(
            (
                smoothstep(0.0, 1.0, -1.0),
                smoothstep(0.0, 1.0, 0.5),
                smoothstep(0.0, 1.0, 2.0)
            ),
            (0.0, 0.5, 1.0)
        );
        assert_eq!(
            (smoothstep(1.0, 0.0, 2.0), smoothstep(1.0, 0.0, -1.0)),
            (0.0, 1.0)
        );
        assert!(smoothstep(0.0, 1.0, 0.25) < 0.25, "slow at its ends");
    }
}
