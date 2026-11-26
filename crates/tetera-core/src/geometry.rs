//! Geometry (ADR-0006): points and vectors whose unit is part of their type, affine maps
//! from one unit to another, pixel rectangles, and stage positions.
//!
//! A [`PointPx`] can't be passed where a [`PointUm`] is wanted, and an
//! [`Affine2<Pixels, Micrometres>`] maps one to the other: the camera-to-stage calibration
//! is such a map (P1.18).

use crate::units::Um;
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A unit for points and vectors.
pub trait Unit: Copy + std::fmt::Debug + PartialEq + 'static {
    /// The unit's short name, as schemas show it.
    const NAME: &'static str;
}

/// Micrometres, the unit of stage coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Micrometres {}

/// Pixels, the unit of image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pixels {}

impl Unit for Micrometres {
    const NAME: &'static str = "Um";
}

impl Unit for Pixels {
    const NAME: &'static str = "Px";
}

/// A point in unit `U`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Point2<U: Unit> {
    /// Its x coordinate.
    pub x: f64,
    /// Its y coordinate.
    pub y: f64,
    #[serde(skip)]
    unit: PhantomData<U>,
}

/// A displacement in unit `U`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Vector2<U: Unit> {
    /// Its x component.
    pub x: f64,
    /// Its y component.
    pub y: f64,
    #[serde(skip)]
    unit: PhantomData<U>,
}

/// A point in micrometres.
pub type PointUm = Point2<Micrometres>;
/// A point in pixels.
pub type PointPx = Point2<Pixels>;
/// A displacement in micrometres.
pub type VectorUm = Vector2<Micrometres>;
/// A displacement in pixels.
pub type VectorPx = Vector2<Pixels>;

impl<U: Unit> Point2<U> {
    /// The origin.
    pub const ORIGIN: Self = Self::new(0.0, 0.0);

    /// The point at (`x`, `y`).
    pub const fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            unit: PhantomData,
        }
    }

    /// The distance to `other`.
    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }
}

impl<U: Unit> Vector2<U> {
    /// No displacement.
    pub const ZERO: Self = Self::new(0.0, 0.0);

    /// The displacement (`x`, `y`).
    pub const fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            unit: PhantomData,
        }
    }

    /// Its length.
    pub fn length(self) -> f64 {
        self.x.hypot(self.y)
    }

    /// The dot product with `other`.
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

impl<U: Unit> Sub for Point2<U> {
    type Output = Vector2<U>;
    fn sub(self, rhs: Self) -> Vector2<U> {
        Vector2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<U: Unit> Add<Vector2<U>> for Point2<U> {
    type Output = Self;
    fn add(self, rhs: Vector2<U>) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<U: Unit> Sub<Vector2<U>> for Point2<U> {
    type Output = Self;
    fn sub(self, rhs: Vector2<U>) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<U: Unit> Add for Vector2<U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<U: Unit> Sub for Vector2<U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<U: Unit> Neg for Vector2<U> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl<U: Unit> Mul<f64> for Vector2<U> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl<U: Unit> Div<f64> for Vector2<U> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

/// An affine map from unit `From` to unit `To`:
/// `x' = xx·x + xy·y + x0` and `y' = yx·x + yy·y + y0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Affine2<From: Unit, To: Unit> {
    /// How x' grows with x.
    pub xx: f64,
    /// How x' grows with y.
    pub xy: f64,
    /// x' at the origin.
    pub x0: f64,
    /// How y' grows with x.
    pub yx: f64,
    /// How y' grows with y.
    pub yy: f64,
    /// y' at the origin.
    pub y0: f64,
    #[serde(skip)]
    units: PhantomData<(From, To)>,
}

/// Below this determinant, a map is treated as having no inverse.
pub const SINGULAR: f64 = 1e-12;

impl<From: Unit, To: Unit> Affine2<From, To> {
    /// The map with these coefficients.
    pub const fn new(xx: f64, xy: f64, x0: f64, yx: f64, yy: f64, y0: f64) -> Self {
        Self {
            xx,
            xy,
            x0,
            yx,
            yy,
            y0,
            units: PhantomData,
        }
    }

    /// A similarity: scale by `scale`, rotate by `radians` counter-clockwise, then move by
    /// `offset`. The camera-to-stage map has this form.
    pub fn similarity(scale: f64, radians: f64, offset: Vector2<To>) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self::new(
            scale * cos,
            -scale * sin,
            offset.x,
            scale * sin,
            scale * cos,
            offset.y,
        )
    }

    /// Where `p` goes.
    pub fn apply(&self, p: Point2<From>) -> Point2<To> {
        Point2::new(
            self.xx * p.x + self.xy * p.y + self.x0,
            self.yx * p.x + self.yy * p.y + self.y0,
        )
    }

    /// Where the displacement `v` goes: the map without its translation.
    pub fn apply_vector(&self, v: Vector2<From>) -> Vector2<To> {
        Vector2::new(self.xx * v.x + self.xy * v.y, self.yx * v.x + self.yy * v.y)
    }

    /// This map, then `next`.
    pub fn then<Next: Unit>(&self, next: &Affine2<To, Next>) -> Affine2<From, Next> {
        Affine2::new(
            next.xx * self.xx + next.xy * self.yx,
            next.xx * self.xy + next.xy * self.yy,
            next.xx * self.x0 + next.xy * self.y0 + next.x0,
            next.yx * self.xx + next.yy * self.yx,
            next.yx * self.xy + next.yy * self.yy,
            next.yx * self.x0 + next.yy * self.y0 + next.y0,
        )
    }

    /// The determinant of its linear part: its area scale, negative if it mirrors.
    pub fn determinant(&self) -> f64 {
        self.xx * self.yy - self.xy * self.yx
    }

    /// The map back, unless this one flattens the plane.
    pub fn inverse(&self) -> Option<Affine2<To, From>> {
        let det = self.determinant();
        if det.abs() < SINGULAR {
            return None;
        }
        let (xx, xy, yx, yy) = (self.yy / det, -self.xy / det, -self.yx / det, self.xx / det);
        Some(Affine2::new(
            xx,
            xy,
            -(xx * self.x0 + xy * self.y0),
            yx,
            yy,
            -(yx * self.x0 + yy * self.y0),
        ))
    }
}

impl<U: Unit> Affine2<U, U> {
    /// The map that changes nothing.
    pub const IDENTITY: Self = Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

/// A rectangle of whole pixels: columns `x..x + width`, rows `y..y + height`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Rect {
    /// Its first column.
    pub x: u32,
    /// Its first row.
    pub y: u32,
    /// Its width in pixels.
    pub width: u32,
    /// Its height in pixels.
    pub height: u32,
}

impl Rect {
    /// The rectangle at (`x`, `y`) of `width` × `height` pixels.
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// The column just past it.
    pub fn right(&self) -> u64 {
        u64::from(self.x) + u64::from(self.width)
    }

    /// The row just past it.
    pub fn bottom(&self) -> u64 {
        u64::from(self.y) + u64::from(self.height)
    }

    /// Its area in pixels.
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Whether it holds no pixel.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Whether it holds the pixel at column `x`, row `y`.
    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && u64::from(x) < self.right() && y >= self.y && u64::from(y) < self.bottom()
    }

    /// Whether it holds every pixel of `other`.
    pub fn contains_rect(&self, other: &Rect) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// The pixels both hold, if any.
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if right <= u64::from(x) || bottom <= u64::from(y) {
            return None;
        }
        // Both differences fit in u32: each is at most one rectangle's own width or height.
        Some(Rect::new(
            x,
            y,
            (right - u64::from(x)) as u32,
            (bottom - u64::from(y)) as u32,
        ))
    }
}

/// A position of the stage, in micrometres.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StagePos {
    /// Along x.
    pub x: Um,
    /// Along y.
    pub y: Um,
    /// Along z, the focus axis.
    pub z: Um,
}

impl StagePos {
    /// The position (`x`, `y`, `z`).
    pub const fn new(x: Um, y: Um, z: Um) -> Self {
        Self { x, y, z }
    }

    /// Its x and y, as a point in the stage's plane.
    pub fn xy(&self) -> PointUm {
        PointUm::new(self.x.0, self.y.0)
    }

    /// The straight-line distance to `other`.
    pub fn distance(&self, other: &StagePos) -> Um {
        let (dx, dy, dz) = (
            self.x.0 - other.x.0,
            self.y.0 - other.y.0,
            self.z.0 - other.z.0,
        );
        Um((dx * dx + dy * dy + dz * dz).sqrt())
    }
}

fn xy_schema(description: &str) -> Schema {
    json_schema!({
        "description": description,
        "type": "object",
        "properties": { "x": { "type": "number" }, "y": { "type": "number" } },
        "required": ["x", "y"]
    })
}

impl<U: Unit> JsonSchema for Point2<U> {
    fn schema_name() -> Cow<'static, str> {
        format!("Point{}", U::NAME).into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        xy_schema(&format!("A point, in {}", U::NAME))
    }
}

impl<U: Unit> JsonSchema for Vector2<U> {
    fn schema_name() -> Cow<'static, str> {
        format!("Vector{}", U::NAME).into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        xy_schema(&format!("A displacement, in {}", U::NAME))
    }
}

impl<From: Unit, To: Unit> JsonSchema for Affine2<From, To> {
    fn schema_name() -> Cow<'static, str> {
        format!("Affine{}To{}", From::NAME, To::NAME).into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        let number = json_schema!({ "type": "number" });
        json_schema!({
            "description": format!("An affine map from {} to {}", From::NAME, To::NAME),
            "type": "object",
            "properties": {
                "xx": number, "xy": number, "x0": number,
                "yx": number, "yy": number, "y0": number
            },
            "required": ["xx", "xy", "x0", "yx", "yy", "y0"]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    type PxToUm = Affine2<Pixels, Micrometres>;

    fn close(a: f64, b: f64, scale: f64) -> bool {
        (a - b).abs() <= 1e-9 * scale.max(1.0)
    }

    #[test]
    fn points_and_vectors_combine_as_geometry_says() {
        let a = PointUm::new(1.0, 2.0);
        let b = PointUm::new(4.0, 6.0);
        let v = b - a;
        assert_eq!(v, VectorUm::new(3.0, 4.0));
        assert_eq!(v.length(), 5.0);
        assert_eq!(a.distance(b), 5.0);
        assert_eq!(a + v, b);
        assert_eq!(b - v, a);
        assert_eq!(v + v - v, v);
        assert_eq!(-v, VectorUm::new(-3.0, -4.0));
        assert_eq!(v * 2.0 / 4.0, VectorUm::new(1.5, 2.0));
        assert_eq!(v.dot(VectorUm::new(1.0, 0.0)), 3.0);
        assert_eq!(PointPx::ORIGIN, PointPx::new(0.0, 0.0));
        assert_eq!(VectorPx::ZERO.length(), 0.0);
    }

    #[test]
    fn a_similarity_scales_rotates_and_moves() {
        // 0.5 µm a pixel, a quarter turn, then 100 µm to the right.
        let map = PxToUm::similarity(0.5, std::f64::consts::FRAC_PI_2, VectorUm::new(100.0, 0.0));
        let p = map.apply(PointPx::new(10.0, 0.0));
        assert!(close(p.x, 100.0, 1.0) && close(p.y, 5.0, 1.0), "{p:?}");
        let v = map.apply_vector(VectorPx::new(10.0, 0.0));
        assert!(close(v.x, 0.0, 1.0) && close(v.y, 5.0, 1.0), "{v:?}");
        assert!(close(map.determinant(), 0.25, 1.0));
    }

    #[test]
    fn a_flattening_map_has_no_inverse() {
        let flat = PxToUm::new(1.0, 2.0, 0.0, 2.0, 4.0, 0.0);
        assert!(flat.inverse().is_none());
        assert!(
            PxToUm::new(1.0, 0.0, 5.0, 0.0, 1.0, 5.0)
                .inverse()
                .is_some()
        );
    }

    #[test]
    fn the_identity_changes_nothing() {
        let p = PointUm::new(3.5, -7.25);
        assert_eq!(Affine2::<Micrometres, Micrometres>::IDENTITY.apply(p), p);
    }

    #[test]
    fn maps_serialise_without_their_units() {
        let map = PxToUm::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(
            json,
            r#"{"xx":1.0,"xy":2.0,"x0":3.0,"yx":4.0,"yy":5.0,"y0":6.0}"#
        );
        assert_eq!(serde_json::from_str::<PxToUm>(&json).unwrap(), map);
        let p: PointUm = serde_json::from_str(r#"{"x":1.5,"y":-2}"#).unwrap();
        assert_eq!(p, PointUm::new(1.5, -2.0));
    }

    #[test]
    fn schemas_carry_the_unit_in_their_names() {
        assert_eq!(PointUm::schema_name(), "PointUm");
        assert_eq!(VectorPx::schema_name(), "VectorPx");
        assert_eq!(PxToUm::schema_name(), "AffinePxToUm");
        let schema = schemars::schema_for!(PxToUm);
        let required = schema.get("required").unwrap().as_array().unwrap();
        assert_eq!(required.len(), 6);
        let point = schemars::schema_for!(PointPx);
        assert_eq!(point.get("required").unwrap().as_array().unwrap().len(), 2);
    }

    #[test]
    fn the_dot_product_uses_both_components() {
        assert_eq!(VectorUm::new(3.0, 4.0).dot(VectorUm::new(2.0, 5.0)), 26.0);
    }

    #[test]
    fn a_similaritys_coefficients_are_exact() {
        let angle = 30_f64.to_radians();
        let map = PxToUm::similarity(2.0, angle, VectorUm::new(7.0, -3.0));
        let expected = [2.0 * angle.cos(), -1.0, 7.0, 1.0, 2.0 * angle.cos(), -3.0];
        let got = [map.xx, map.xy, map.x0, map.yx, map.yy, map.y0];
        for (g, e) in got.iter().zip(expected) {
            assert!((g - e).abs() < 1e-12, "{got:?} against {expected:?}");
        }
    }

    #[test]
    fn a_general_map_moves_points_and_turns_vectors() {
        let map = PxToUm::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(
            map.apply_vector(VectorPx::new(7.0, 8.0)),
            VectorUm::new(23.0, 68.0)
        );
        assert_eq!(map.apply(PointPx::new(7.0, 8.0)), PointUm::new(26.0, 74.0));
    }

    #[test]
    fn a_map_just_at_the_singular_limit_still_inverts() {
        assert!(
            PxToUm::new(SINGULAR, 0.0, 0.0, 0.0, 1.0, 0.0)
                .inverse()
                .is_some()
        );
        assert!(
            PxToUm::new(SINGULAR / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                .inverse()
                .is_none()
        );
    }

    #[test]
    fn distance_counts_every_axis() {
        let a = StagePos::new(Um(1.0), Um(2.0), Um(3.0));
        let b = StagePos::new(Um(4.0), Um(6.0), Um(15.0));
        assert_eq!(a.distance(&b), Um(13.0));
    }

    #[test]
    fn a_vectors_schema_requires_both_components() {
        let schema = schemars::schema_for!(VectorUm);
        assert_eq!(schema.get("required").unwrap().as_array().unwrap().len(), 2);
    }

    #[test]
    fn rectangles_measure_and_contain() {
        let r = Rect::new(10, 20, 30, 40);
        assert_eq!((r.right(), r.bottom(), r.area()), (40, 60, 1_200));
        assert!(r.contains(10, 20) && r.contains(39, 59));
        assert!(!r.contains(40, 20) && !r.contains(10, 60) && !r.contains(9, 20));
        assert!(r.contains_rect(&Rect::new(10, 20, 30, 40)));
        assert!(!r.contains_rect(&Rect::new(10, 20, 31, 40)));
        assert!(Rect::new(0, 0, 0, 5).is_empty() && !r.is_empty());
        let edge = Rect::new(u32::MAX - 1, 0, 2, 1);
        assert_eq!(edge.right(), u64::from(u32::MAX) + 1);
    }

    #[test]
    fn rectangles_that_only_touch_share_no_pixel() {
        let a = Rect::new(0, 0, 10, 10);
        assert_eq!(a.intersection(&Rect::new(10, 0, 5, 5)), None);
        assert_eq!(a.intersection(&Rect::new(0, 10, 5, 5)), None);
        assert_eq!(
            a.intersection(&Rect::new(5, 5, 10, 10)),
            Some(Rect::new(5, 5, 5, 5))
        );
    }

    #[test]
    fn stage_positions_measure_distance() {
        let a = StagePos::new(Um(0.0), Um(3.0), Um(0.0));
        let b = StagePos::new(Um(4.0), Um(0.0), Um(12.0));
        assert_eq!(a.distance(&b), Um(13.0));
        assert_eq!(b.xy(), PointUm::new(4.0, 0.0));
    }

    fn arb_map() -> impl Strategy<Value = PxToUm> {
        (
            0.01f64..100.0,
            -3.2f64..3.2,
            -1.0e4f64..1.0e4,
            -1.0e4f64..1.0e4,
        )
            .prop_map(|(s, r, x, y)| PxToUm::similarity(s, r, VectorUm::new(x, y)))
    }

    fn arb_rect() -> impl Strategy<Value = Rect> {
        (0u32..200, 0u32..200, 0u32..120, 0u32..120).prop_map(|(x, y, w, h)| Rect::new(x, y, w, h))
    }

    proptest! {
        #[test]
        fn a_map_then_its_inverse_returns_every_point(map in arb_map(), x in -1.0e4f64..1.0e4, y in -1.0e4f64..1.0e4) {
            let p = PointPx::new(x, y);
            let back = map.inverse().unwrap().apply(map.apply(p));
            prop_assert!(close(back.x, p.x, 1.0e4) && close(back.y, p.y, 1.0e4), "{p:?} -> {back:?}");
        }

        #[test]
        fn composing_maps_applies_them_in_order(
            a in arb_map(),
            coefficients in proptest::array::uniform6(-10.0f64..10.0),
            x in -1.0e3f64..1.0e3,
            y in -1.0e3f64..1.0e3,
        ) {
            let [xx, xy, x0, yx, yy, y0] = coefficients;
            let b = Affine2::<Micrometres, Micrometres>::new(xx, xy, x0, yx, yy, y0);
            let p = PointPx::new(x, y);
            let together = a.then(&b).apply(p);
            let one_by_one = b.apply(a.apply(p));
            prop_assert!(close(together.x, one_by_one.x, 1.0e7) && close(together.y, one_by_one.y, 1.0e7));
        }

        #[test]
        fn an_intersection_is_shared_and_symmetric(a in arb_rect(), b in arb_rect(), px in 0u32..330, py in 0u32..330) {
            let ab = a.intersection(&b);
            prop_assert_eq!(ab, b.intersection(&a));
            prop_assert_eq!(ab.is_some_and(|r| r.contains(px, py)), a.contains(px, py) && b.contains(px, py));
            if let Some(r) = ab {
                prop_assert!(!r.is_empty() && a.contains_rect(&r) && b.contains_rect(&r));
                prop_assert!(r.area() <= a.area().min(b.area()));
            }
        }
    }
}
