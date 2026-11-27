//! The glue nodes: crop, and statistics.

use crate::node::{Context, Descriptor, NoParameters, Node, Step, parameters};
use crate::port::{PortSpec, PortType, Table, Value};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::{Arc, LazyLock};
use tetera_core::{Error, Result};
use tetera_imaging::Image;

/// The glue nodes, for the registry.
pub fn nodes() -> Vec<Arc<dyn Node>> {
    vec![Arc::new(Crop), Arc::new(Statistics)]
}

/// `tetera.crop`: the part of an image inside a rectangle.
#[derive(Debug, Clone, Copy)]
pub struct Crop;

/// `tetera.crop`'s parameters: the rectangle, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CropParameters {
    /// Its first column.
    pub x: u32,
    /// Its first row.
    pub y: u32,
    /// Its width, at least 1.
    pub width: u32,
    /// Its height, at least 1.
    pub height: u32,
}

static CROP: LazyLock<Descriptor> = LazyLock::new(|| {
    Descriptor::new("tetera.crop", 1, "The part of an image inside a rectangle.")
        .input(PortSpec::new("image", PortType::ANY_IMAGE, "The image."))
        .output(
            PortSpec::new(
                "image",
                PortType::ANY_IMAGE,
                "Its part inside the rectangle.",
            )
            .following("image"),
        )
        .parameters::<CropParameters>()
});

impl Node for Crop {
    fn descriptor(&self) -> &Descriptor {
        &CROP
    }

    fn prepare(&self, params: &serde_json::Value) -> Result<Box<dyn Step>> {
        let p: CropParameters = parameters(params)?;
        if p.width == 0 || p.height == 0 {
            return Err(Error::invalid_argument(format!(
                "a crop is at least 1 × 1, not {} × {}",
                p.width, p.height
            )));
        }
        Ok(Box::new(p))
    }
}

/// `image`'s part inside `rect`, or why it doesn't fit.
fn crop<T: Copy>(image: &Image<T>, rect: &CropParameters) -> Result<Image<T>> {
    let fits =
        |start: u32, len: u32, size: u32| start.checked_add(len).is_some_and(|end| end <= size);
    if !fits(rect.x, rect.width, image.width()) || !fits(rect.y, rect.height, image.height()) {
        return Err(Error::invalid_argument(format!(
            "a crop of {} × {} at ({}, {}) runs past the {} × {} image",
            rect.width,
            rect.height,
            rect.x,
            rect.y,
            image.width(),
            image.height()
        )));
    }
    let c = usize::from(image.channels());
    let row = image.width() as usize * c;
    let mut data = Vec::with_capacity(rect.width as usize * rect.height as usize * c);
    for y in rect.y..rect.y + rect.height {
        let start = y as usize * row + rect.x as usize * c;
        data.extend_from_slice(&image.data()[start..start + rect.width as usize * c]);
    }
    Image::new(rect.width, rect.height, image.channels(), data)
}

impl Step for CropParameters {
    fn run(&mut self, _: &Context, inputs: &[&Value]) -> Result<Vec<Value>> {
        Ok(vec![match inputs[0] {
            Value::U8(i) => Value::U8(crop(i, self)?),
            Value::U16(i) => Value::U16(crop(i, self)?),
            Value::F32(i) => Value::F32(crop(i, self)?),
            other => {
                return Err(Error::internal(format!(
                    "crop was given {}",
                    other.port_type()
                )));
            }
        }])
    }
}

/// `tetera.statistics`: each channel's mean, standard deviation, least and greatest value.
#[derive(Debug, Clone, Copy)]
pub struct Statistics;

static STATISTICS: LazyLock<Descriptor> = LazyLock::new(|| {
    Descriptor::new(
        "tetera.statistics",
        1,
        "Each channel's mean, standard deviation, least and greatest value, and the pixels counted.",
    )
    .input(PortSpec::new("image", PortType::ANY_IMAGE, "The image."))
    .output(PortSpec::new(
        "table",
        PortType::Table,
        "`pixels`, and for each channel c, `mean.c`, `std.c`, `min.c` and `max.c`.",
    ))
});

impl Node for Statistics {
    fn descriptor(&self) -> &Descriptor {
        &STATISTICS
    }

    fn prepare(&self, params: &serde_json::Value) -> Result<Box<dyn Step>> {
        let _: NoParameters = parameters(params)?;
        Ok(Box::new(Statistics))
    }
}

/// `values`' statistics, `channels` interleaved.
fn statistics(values: impl Iterator<Item = f64>, channels: usize, pixels: usize) -> Table {
    let mut sum = vec![0.0; channels];
    let mut squares = vec![0.0; channels];
    let mut min = vec![f64::INFINITY; channels];
    let mut max = vec![f64::NEG_INFINITY; channels];
    for (i, v) in values.enumerate() {
        let c = i % channels;
        sum[c] += v;
        squares[c] += v * v;
        min[c] = min[c].min(v);
        max[c] = max[c].max(v);
    }
    let n = pixels as f64;
    let mut table = Table::from([("pixels".to_string(), n)]);
    for c in 0..channels {
        let mean = sum[c] / n;
        let variance = (squares[c] / n - mean * mean).max(0.0);
        table.insert(format!("mean.{c}"), mean);
        table.insert(format!("std.{c}"), variance.sqrt());
        table.insert(format!("min.{c}"), min[c]);
        table.insert(format!("max.{c}"), max[c]);
    }
    table
}

impl Step for Statistics {
    fn run(&mut self, _: &Context, inputs: &[&Value]) -> Result<Vec<Value>> {
        let table = match inputs[0] {
            Value::U8(i) => statistics(
                i.data().iter().map(|&v| f64::from(v)),
                i.channels().into(),
                pixels(i),
            ),
            Value::U16(i) => statistics(
                i.data().iter().map(|&v| f64::from(v)),
                i.channels().into(),
                pixels(i),
            ),
            Value::F32(i) => statistics(
                i.data().iter().map(|&v| f64::from(v)),
                i.channels().into(),
                pixels(i),
            ),
            other => {
                return Err(Error::internal(format!(
                    "statistics was given {}",
                    other.port_type()
                )));
            }
        };
        Ok(vec![Value::Table(table)])
    }
}

fn pixels<T: Copy>(image: &Image<T>) -> usize {
    image.width() as usize * image.height() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetera_compute::Compute;

    fn run(node: &dyn Node, params: serde_json::Value, input: Value) -> Result<Vec<Value>> {
        let compute = Compute::cpu_only().unwrap();
        let mut step = node.prepare(&params)?;
        step.run(&Context { compute: &compute }, &[&input])
    }

    #[test]
    fn crop_takes_the_rectangle_of_any_image() {
        // A 4 × 3 image of 2 channels, each value its index.
        let image = Image::new(4, 3, 2, (0..24u16).collect()).unwrap();
        let out = run(
            &Crop,
            serde_json::json!({"x": 1, "y": 1, "width": 2, "height": 2}),
            Value::U16(image),
        )
        .unwrap();
        let expected = Image::new(2, 2, 2, vec![10, 11, 12, 13, 18, 19, 20, 21]).unwrap();
        assert_eq!(out, [Value::U16(expected)]);
        let grey = Image::new(3, 1, 1, vec![1.0f32, 2.0, 3.0]).unwrap();
        let out = run(
            &Crop,
            serde_json::json!({"x": 2, "y": 0, "width": 1, "height": 1}),
            Value::F32(grey),
        )
        .unwrap();
        assert_eq!(out, [Value::F32(Image::new(1, 1, 1, vec![3.0]).unwrap())]);
        let bytes = Image::new(2, 2, 1, vec![1u8, 2, 3, 4]).unwrap();
        let whole = run(
            &Crop,
            serde_json::json!({"x": 0, "y": 0, "width": 2, "height": 2}),
            Value::U8(bytes.clone()),
        );
        assert_eq!(whole.unwrap(), [Value::U8(bytes)]);
    }

    #[test]
    fn crop_refuses_an_empty_rectangle_or_one_past_the_image() {
        let image = || Value::U8(Image::new(4, 3, 1, vec![0; 12]).unwrap());
        let err = |p| run(&Crop, p, image()).unwrap_err().to_string();
        assert_eq!(
            err(serde_json::json!({"x": 0, "y": 0, "width": 0, "height": 2})),
            "a crop is at least 1 × 1, not 0 × 2"
        );
        assert!(
            err(serde_json::json!({"x": 0, "y": 0, "width": 2, "height": 0})).contains("not 2 × 0")
        );
        assert_eq!(
            err(serde_json::json!({"x": 3, "y": 0, "width": 2, "height": 1})),
            "a crop of 2 × 1 at (3, 0) runs past the 4 × 3 image"
        );
        assert!(
            err(serde_json::json!({"x": 0, "y": 2, "width": 1, "height": 2})).contains("runs past")
        );
        assert!(
            err(serde_json::json!({"x": u32::MAX, "y": 0, "width": 2, "height": 1}))
                .contains("runs past")
        );
        assert!(
            err(serde_json::json!({"x": 0, "y": 0, "width": 1})).contains("missing field `height`")
        );
        assert!(
            run(
                &Crop,
                serde_json::json!({"x": 3, "y": 2, "width": 1, "height": 1}),
                image()
            )
            .is_ok()
        );
    }

    #[test]
    fn statistics_measure_each_channel() {
        // Two pixels of two channels: (1, 10) and (3, 30).
        let image = Image::new(2, 1, 2, vec![1u16, 10, 3, 30]).unwrap();
        let out = run(&Statistics, serde_json::json!({}), Value::U16(image)).unwrap();
        let Value::Table(t) = &out[0] else {
            panic!("a table")
        };
        let expected = [
            ("pixels", 2.0),
            ("mean.0", 2.0),
            ("std.0", 1.0),
            ("min.0", 1.0),
            ("max.0", 3.0),
            ("mean.1", 20.0),
            ("std.1", 10.0),
            ("min.1", 10.0),
            ("max.1", 30.0),
        ];
        assert_eq!(t.len(), expected.len());
        for (key, value) in expected {
            assert_eq!(t[key], value, "{key}");
        }
        let flat = Image::new(3, 1, 1, vec![0.5f32; 3]).unwrap();
        let out = run(&Statistics, serde_json::json!({}), Value::F32(flat)).unwrap();
        let Value::Table(t) = &out[0] else {
            panic!("a table")
        };
        assert_eq!((t["std.0"], t["mean.0"]), (0.0, 0.5));
        let bytes = Image::new(1, 1, 1, vec![200u8]).unwrap();
        let out = run(&Statistics, serde_json::json!({}), Value::U8(bytes)).unwrap();
        let Value::Table(t) = &out[0] else {
            panic!("a table")
        };
        assert_eq!((t["min.0"], t["max.0"], t["pixels"]), (200.0, 200.0, 1.0));
        assert!(
            run(
                &Statistics,
                serde_json::json!({"bins": 4}),
                Value::Scalar(1.0)
            )
            .is_err()
        );
    }

    #[test]
    fn a_node_given_the_wrong_value_says_so() {
        let compute = Compute::cpu_only().unwrap();
        let context = Context { compute: &compute };
        let mut crop = Crop
            .prepare(&serde_json::json!({"x": 0, "y": 0, "width": 1, "height": 1}))
            .unwrap();
        assert_eq!(
            crop.run(&context, &[&Value::Scalar(1.0)])
                .unwrap_err()
                .to_string(),
            "crop was given a scalar"
        );
        let mut stats = Statistics.prepare(&serde_json::json!({})).unwrap();
        assert_eq!(
            stats
                .run(&context, &[&Value::Table(Table::new())])
                .unwrap_err()
                .to_string(),
            "statistics was given a table"
        );
    }
}
