//! Half precision (P1.11b): IEEE 754 binary16, as the simulator's tiles hold their stains on
//! the GPU. The kernels round with PTX's `cvt.rn.f16.f32`, to nearest, ties to even; these are
//! its CPU twins, and how the host reads a tile.

/// `bits`, a half, as a float: exact, since every half is a float.
pub fn to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let magnitude = match (exponent, mantissa) {
        (0, 0) => 0,
        // A subnormal half is a normal float: mantissa × 2⁻²⁴.
        (0, m) => (m as f32 * (1.0 / 16_777_216.0)).to_bits(),
        (31, 0) => 0x7f80_0000,
        (31, m) => 0x7fc0_0000 | (m << 13),
        (e, m) => ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// `v` as a half, rounded to nearest, ties to even, as the GPU rounds; beyond the largest
/// half, infinity.
pub fn from_f32(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return sign | 0x7e00;
    }
    if magnitude >= 0x4780_0000 {
        // 2^16 and above is beyond every half; from 65520 the rounding below carries there.
        return sign | 0x7c00;
    }
    if magnitude < 0x3880_0000 {
        // Below the smallest normal half, 2⁻¹⁴: a subnormal, in units of 2⁻²⁴.
        let scaled = f32::from_bits(magnitude) * 16_777_216.0;
        return sign | round_even(scaled) as u16;
    }
    let exponent = (magnitude >> 23) - 112;
    let mantissa = magnitude & 0x7f_ffff;
    let mut half = (exponent << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

/// `v`, which is below 1,024 and not negative, to the nearest whole number, ties to even.
fn round_even(v: f32) -> u32 {
    let whole = v.floor();
    let rest = v - whole;
    let n = whole as u32;
    if rest > 0.5 || (rest == 0.5 && n % 2 == 1) {
        n + 1
    } else {
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_read_as_their_values() {
        for (bits, value) in [
            (0x0000, 0.0),
            (0x3c00, 1.0),
            (0x3800, 0.5),
            (0xc000, -2.0),
            (0x7bff, 65_504.0),
            (0x0400, 6.103_515_6e-5),
            (0x0001, 5.960_464_5e-8),
            (0x03ff, 6.097_555e-5),
            (0x3555, 0.333_251_95),
        ] {
            assert_eq!(to_f32(bits), value, "{bits:#06x}");
        }
        assert_eq!(to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(to_f32(0x7c00), f32::INFINITY);
        assert_eq!(to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(to_f32(0x7e00).is_nan());
    }

    #[test]
    fn floats_round_to_the_nearest_half_ties_to_even() {
        for (value, bits) in [
            (0.0, 0x0000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (65_504.0, 0x7bff),
            (65_519.0, 0x7bff),
            (65_520.0, 0x7c00),
            (1e9, 0x7c00),
            (1.0 / 3.0, 0x3555),
            (6.103_515_6e-5, 0x0400),
            (5.960_464_5e-8, 0x0001),
            (2.0e-8, 0x0000),
            (4.0e-8, 0x0001),
        ] {
            assert_eq!(from_f32(value), bits, "{value}");
        }
        // Exactly half way: to the even neighbour.
        assert_eq!(from_f32(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(from_f32(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(
            from_f32(5.960_464_5e-8 * 2.5),
            0x0002,
            "a subnormal tie, to even"
        );
        assert_eq!(from_f32(5.960_464_5e-8 * 3.5), 0x0004, "and up to even");
        assert_eq!(
            from_f32(5.960_464_5e-8 * 0.5),
            0x0000,
            "half the smallest, to even, zero"
        );
        assert_eq!(from_f32(-0.0), 0x8000);
        assert_eq!(from_f32(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(from_f32(f32::NAN) & 0x7e00, 0x7e00);
    }

    #[test]
    fn every_half_survives_a_round_trip() {
        for bits in 0..=u16::MAX {
            let v = to_f32(bits);
            if !v.is_nan() {
                assert_eq!(from_f32(v), bits, "{bits:#06x}");
            }
        }
    }

    #[test]
    fn a_half_is_within_one_part_in_two_thousand_of_its_float() {
        for i in 1..10_000 {
            let v = i as f32 * 0.000_731;
            let back = to_f32(from_f32(v));
            assert!((back - v).abs() <= v * 4.9e-4 + 3e-8, "{v} became {back}");
        }
    }
}
