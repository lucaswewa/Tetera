//! Batch runs and tiling (P1.08e). A batch runs a plan over a list of inputs, in order. A tiled
//! run cuts its frames and images into tiles, each with a halo as wide as the recipe reaches,
//! runs each tile, and stitches what each tile's own part gives, so the result is what the
//! whole frame would give.

use crate::node::Context;
use crate::plan::{Plan, Source};
use crate::port::Value;
use std::collections::BTreeMap;
use tetera_core::geometry::Rect;
use tetera_core::{Error, Result};
use tetera_imaging::{Image, Mosaic, MosaicLayout};

impl Plan {
    /// The plan run on each of `inputs` in turn: their outputs, in order. A stateful node keeps
    /// its state from one to the next, as over the frames of a sequence.
    pub fn run_batch(
        &mut self,
        context: &Context,
        inputs: impl IntoIterator<Item = BTreeMap<String, Value>>,
    ) -> Result<Vec<BTreeMap<String, Value>>> {
        inputs
            .into_iter()
            .enumerate()
            .map(|(i, input)| {
                self.run(context, input)
                    .map_err(|e| e.context(format!("item {i} of the batch")))
            })
            .collect()
    }

    /// How far the recipe reaches, in pixels: along its longest path, the sum of each step's
    /// reach. Or the first node that needs the whole image, or keeps state, which tiles can't
    /// run.
    pub fn reach(&self) -> Result<u32> {
        let mut depth = vec![0u32; self.steps.len()];
        for (i, planned) in self.steps.iter().enumerate() {
            let why = if planned.descriptor.stateful {
                "keeps state from one run to the next"
            } else if planned.step.reach().is_none() {
                "needs the whole image"
            } else {
                ""
            };
            if !why.is_empty() {
                return Err(Error::invalid_argument(format!(
                    "recipe `{}` can't run in tiles: node `{}` ({}) {why}",
                    self.name, planned.id, planned.descriptor.id
                )));
            }
            let fed = planned
                .sources
                .iter()
                .map(|s| match *s {
                    Source::Output(j, _) => depth[j],
                    Source::Input(_) => 0,
                })
                .max()
                .unwrap_or(0);
            depth[i] = fed + planned.step.reach().unwrap_or(0);
        }
        Ok(depth.into_iter().max().unwrap_or(0))
    }

    /// The plan run on `inputs` in tiles of about `tile` pixels each way, each with a halo as
    /// wide as the recipe reaches, rounded up to keep a raw frame's Bayer pattern: the same
    /// outputs a whole run gives. Every frame and image given must be of one size, and every
    /// output a frame or an image of that size.
    pub fn run_tiled(
        &mut self,
        context: &Context,
        inputs: BTreeMap<String, Value>,
        tile: (u32, u32),
    ) -> Result<BTreeMap<String, Value>> {
        if self.held.iter().any(Option::is_some) {
            return Err(Error::invalid_argument(format!(
                "recipe `{}` holds inputs, which a tiled run doesn't cut: let them go",
                self.name
            )));
        }
        let halo = even(self.reach()?);
        let sizes: Vec<(u32, u32)> = inputs.values().filter_map(size).collect();
        let Some(&(width, height)) = sizes.first() else {
            return Err(Error::invalid_argument(format!(
                "recipe `{}`'s tiled run was given no frame or image to cut",
                self.name
            )));
        };
        if sizes.iter().any(|&s| s != (width, height)) {
            return Err(Error::invalid_argument(format!(
                "a tiled run cuts frames and images of one size, not {sizes:?}"
            )));
        }
        let (tw, th) = (even(tile.0.max(2)), even(tile.1.max(2)));
        let mut canvases: BTreeMap<String, Canvas> = BTreeMap::new();
        for y in (0..height).step_by(th as usize) {
            for x in (0..width).step_by(tw as usize) {
                let core = Rect::new(x, y, tw.min(width - x), th.min(height - y));
                let cut_at = around(core, halo, width, height);
                let (x0, y0) = (cut_at.x, cut_at.y);
                let mut part = BTreeMap::new();
                for (name, value) in &inputs {
                    let piece = if size(value).is_some() {
                        cut(value, cut_at)?
                    } else {
                        value.clone()
                    };
                    part.insert(name.clone(), piece);
                }
                let outputs = self.run(context, part)?;
                for (name, value) in outputs {
                    if size(&value) != Some((cut_at.width, cut_at.height)) {
                        return Err(Error::invalid_argument(format!(
                            "recipe `{}`'s output `{name}` is {}, not a frame or an image of its tile's size, which tiles can't stitch",
                            self.name,
                            value.port_type()
                        )));
                    }
                    let own = Rect::new(x - x0, y - y0, core.width, core.height);
                    let canvas = match canvases.entry(name) {
                        std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
                        std::collections::btree_map::Entry::Vacant(e) => {
                            e.insert(Canvas::new(&value, width, height)?)
                        }
                    };
                    canvas.paste(&value, own, (x, y))?;
                }
            }
        }
        canvases
            .into_iter()
            .map(|(name, canvas)| Ok((name, canvas.finish()?)))
            .collect()
    }
}

/// `core` with `halo` pixels around it, within a frame of `width × height`.
fn around(core: Rect, halo: u32, width: u32, height: u32) -> Rect {
    let x0 = core.x.saturating_sub(halo);
    let y0 = core.y.saturating_sub(halo);
    let x1 = (core.x + core.width + halo).min(width);
    let y1 = (core.y + core.height + halo).min(height);
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

/// `n` rounded up to even, so a raw frame's tiles keep its Bayer pattern.
fn even(n: u32) -> u32 {
    n + n % 2
}

/// A frame's or an image's width and height, on the host.
fn size(value: &Value) -> Option<(u32, u32)> {
    match value {
        Value::Raw(m) => Some((m.layout().width, m.layout().height)),
        Value::U8(i) => Some((i.width(), i.height())),
        Value::U16(i) => Some((i.width(), i.height())),
        Value::F32(i) => Some((i.width(), i.height())),
        _ => None,
    }
}

/// `value`'s part inside `rect`.
fn cut(value: &Value, rect: Rect) -> Result<Value> {
    Ok(match value {
        Value::Raw(m) => Value::Raw(m.crop(rect)?),
        Value::U8(i) => Value::U8(cut_image(i, rect)?),
        Value::U16(i) => Value::U16(cut_image(i, rect)?),
        Value::F32(i) => Value::F32(cut_image(i, rect)?),
        other => {
            return Err(Error::internal(format!(
                "a tiled run cut {}",
                other.port_type()
            )));
        }
    })
}

fn cut_image<T: Copy>(image: &Image<T>, rect: Rect) -> Result<Image<T>> {
    let c = usize::from(image.channels());
    let row = image.width() as usize * c;
    let mut data = Vec::with_capacity(rect.width as usize * rect.height as usize * c);
    for y in rect.y..rect.y + rect.height {
        let start = y as usize * row + rect.x as usize * c;
        data.extend_from_slice(&image.data()[start..start + rect.width as usize * c]);
    }
    Image::new(rect.width, rect.height, image.channels(), data)
}

/// An output being stitched: its values, row by row.
enum Canvas {
    Raw(MosaicLayout, Vec<u16>),
    U8(u32, u32, u8, Vec<u8>),
    U16(u32, u32, u8, Vec<u16>),
    F32(u32, u32, u8, Vec<f32>),
}

/// `part`'s rectangle `from` copied into `whole`, `width` wide, its corner at `to`.
fn paste_rows<T: Copy>(
    whole: &mut [T],
    width: u32,
    part: &[T],
    part_width: u32,
    channels: usize,
    from: Rect,
    to: (u32, u32),
) {
    let n = from.width as usize * channels;
    for row in 0..from.height {
        let src = ((from.y + row) as usize * part_width as usize + from.x as usize) * channels;
        let dst = ((to.1 + row) as usize * width as usize + to.0 as usize) * channels;
        whole[dst..dst + n].copy_from_slice(&part[src..src + n]);
    }
}

impl Canvas {
    /// A canvas of `width × height` for outputs like `first`; a raw frame's pattern is that of
    /// the first tile, at the corner.
    fn new(first: &Value, width: u32, height: u32) -> Result<Self> {
        let n = width as usize * height as usize;
        Ok(match first {
            Value::Raw(m) => Self::Raw(
                MosaicLayout {
                    width,
                    height,
                    ..*m.layout()
                },
                vec![0; n],
            ),
            Value::U8(i) => Self::U8(
                width,
                height,
                i.channels(),
                vec![0; n * usize::from(i.channels())],
            ),
            Value::U16(i) => Self::U16(
                width,
                height,
                i.channels(),
                vec![0; n * usize::from(i.channels())],
            ),
            Value::F32(i) => Self::F32(
                width,
                height,
                i.channels(),
                vec![0.0; n * usize::from(i.channels())],
            ),
            other => {
                return Err(Error::internal(format!(
                    "a canvas for {}",
                    other.port_type()
                )));
            }
        })
    }

    /// `part`'s own rectangle, `from`, placed with its corner at `to`.
    fn paste(&mut self, part: &Value, from: Rect, to: (u32, u32)) -> Result<()> {
        match (self, part) {
            (Self::Raw(layout, sites), Value::Raw(m)) => paste_rows(
                sites,
                layout.width,
                m.sites(),
                m.layout().width,
                1,
                from,
                to,
            ),
            (Self::U8(w, _, c, data), Value::U8(i)) if *c == i.channels() => {
                paste_rows(data, *w, i.data(), i.width(), usize::from(*c), from, to)
            }
            (Self::U16(w, _, c, data), Value::U16(i)) if *c == i.channels() => {
                paste_rows(data, *w, i.data(), i.width(), usize::from(*c), from, to)
            }
            (Self::F32(w, _, c, data), Value::F32(i)) if *c == i.channels() => {
                paste_rows(data, *w, i.data(), i.width(), usize::from(*c), from, to)
            }
            (_, other) => {
                return Err(Error::internal(format!(
                    "a tile gave {}, unlike the tiles before it",
                    other.port_type()
                )));
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<Value> {
        Ok(match self {
            Self::Raw(layout, sites) => Value::Raw(Mosaic::new(layout, sites)?),
            Self::U8(w, h, c, data) => Value::U8(Image::new(w, h, c, data)?),
            Self::U16(w, h, c, data) => Value::U16(Image::new(w, h, c, data)?),
            Self::F32(w, h, c, data) => Value::F32(Image::new(w, h, c, data)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Descriptor, NoParameters, parameters};
    use crate::port::{PortSpec, PortType};
    use crate::recipe::Recipe;
    use crate::registry::Registry;
    use crate::stock::{Stock, local};
    use std::sync::Arc;
    use tetera_compute::Compute;
    use tetera_imaging::pattern::{Sensor, noisy, test_pattern};

    fn compute() -> Compute {
        Compute::cpu_only().unwrap()
    }

    fn plan(text: &str) -> Plan {
        Plan::new(&Recipe::from_json(text).unwrap(), &Registry::stock()).unwrap()
    }

    /// Dark, demosaic, the bilateral filter, colour: a reach of 0 + 2 + 3 + 0.
    fn raw_chain(method: &str) -> String {
        format!(
            r#"{{"name": "raw chain", "inputs": {{"raw": {{"kind": "raw"}}, "dark": {{"kind": "raw"}}}},
            "nodes": [
                {{"id": "less_dark", "node": "tetera.dark", "inputs": {{"raw": "raw", "dark": "dark"}}}},
                {{"id": "rgb", "node": "tetera.demosaic", "params": {{"method": "{method}"}}, "inputs": {{"raw": "less_dark.raw"}}}},
                {{"id": "smooth", "node": "tetera.bilateral", "params": {{"sigma_spatial": 1.5, "sigma_range": 200}}, "inputs": {{"image": "rgb.image"}}}},
                {{"id": "colour", "node": "tetera.colour", "params": {{"levels": [{{"black": 0, "white": 4095}}, {{"black": 0, "white": 4095}}, {{"black": 0, "white": 4095}}]}}, "inputs": {{"image": "smooth.image"}}}}
            ],
            "outputs": {{"preview": "colour.image", "rgb": "rgb.image", "less_dark": "less_dark.raw"}}}}"#
        )
    }

    fn raw_inputs(w: u32, h: u32) -> BTreeMap<String, Value> {
        let sensor = Sensor {
            offset: 100,
            ..Sensor::default()
        };
        let clean = sensor.expose(&test_pattern(w, h, 12)).unwrap();
        let raw = Mosaic::new(*clean.layout(), noisy(clean.sites(), 30.0, 4_095, 1)).unwrap();
        BTreeMap::from([
            ("raw".to_string(), Value::Raw(raw)),
            ("dark".to_string(), Value::Raw(sensor.dark(w, h).unwrap())),
        ])
    }

    #[test]
    fn a_raw_chain_in_tiles_gives_what_the_whole_frame_gives() {
        let c = compute();
        for method in ["bilinear", "malvar_he_cutler"] {
            let mut p = plan(&raw_chain(method));
            assert_eq!(p.reach().unwrap(), if method == "bilinear" { 4 } else { 5 });
            let inputs = raw_inputs(150, 110);
            let whole = p.run(&Context { compute: &c }, inputs.clone()).unwrap();
            // Tiles that don't divide the frame, a tile larger than it, and tiles of 2.
            for tile in [(48, 40), (64, 64), (200, 200), (30, 22), (10, 6)] {
                let tiled = p
                    .run_tiled(&Context { compute: &c }, inputs.clone(), tile)
                    .unwrap();
                assert_eq!(tiled, whole, "{method}, tiles of {tile:?}");
            }
            // The smallest tiles, on a small frame.
            let small = raw_inputs(24, 16);
            let whole = p.run(&Context { compute: &c }, small.clone()).unwrap();
            assert_eq!(
                p.run_tiled(&Context { compute: &c }, small, (2, 2))
                    .unwrap(),
                whole
            );
        }
    }

    #[test]
    fn images_in_tiles_with_reaches_added_along_a_path() {
        let c = compute();
        // Two filters one after the other, and one beside them: the longest path reaches 3 + 4.
        let mut p = plan(
            r#"{"name": "filters", "inputs": {"image": {"kind": "image", "channels": 1, "sample": "u16"}},
            "nodes": [
                {"id": "a", "node": "tetera.bilateral", "params": {"sigma_spatial": 1.5, "sigma_range": 300}, "inputs": {"image": "image"}},
                {"id": "b", "node": "tetera.bilateral", "params": {"sigma_spatial": 2.0, "sigma_range": 300}, "inputs": {"image": "a.image"}},
                {"id": "c", "node": "tetera.bilateral", "params": {"sigma_spatial": 1.0, "sigma_range": 300}, "inputs": {"image": "image"}},
                {"id": "show", "node": "tetera.display", "params": {"levels": [{"black": 0, "white": 4095}]}, "inputs": {"image": "b.image"}}
            ],
            "outputs": {"show": "show.image", "c": "c.image"}}"#,
        );
        assert_eq!(p.reach().unwrap(), 7);
        let grey: Vec<u16> = test_pattern(75, 53, 12)
            .data()
            .iter()
            .step_by(3)
            .copied()
            .collect();
        let image = Image::new(75, 53, 1, noisy(&grey, 40.0, 4_095, 2)).unwrap();
        let inputs = BTreeMap::from([("image".to_string(), Value::U16(image))]);
        let whole = p.run(&Context { compute: &c }, inputs.clone()).unwrap();
        for tile in [(20, 16), (32, 32), (8, 50)] {
            assert_eq!(
                p.run_tiled(&Context { compute: &c }, inputs.clone(), tile)
                    .unwrap(),
                whole,
                "{tile:?}"
            );
        }
    }

    #[test]
    fn what_tiles_cant_run_is_refused() {
        let c = compute();
        let flat = plan(
            r#"{"name": "flat", "inputs": {"raw": {"kind": "raw"}, "flat": {"kind": "raw"}},
            "nodes": [{"id": "even", "node": "tetera.flat", "inputs": {"raw": "raw", "flat": "flat"}}],
            "outputs": {"raw": "even.raw"}}"#,
        );
        assert_eq!(
            flat.reach().unwrap_err().to_string(),
            "recipe `flat` can't run in tiles: node `even` (tetera.flat) needs the whole image"
        );
        let temporal = plan(
            r#"{"name": "mean", "inputs": {"raw": {"kind": "raw"}},
            "nodes": [{"id": "mean", "node": "tetera.temporal_mean", "params": {"window": 2}, "inputs": {"raw": "raw"}}],
            "outputs": {"raw": "mean.raw"}}"#,
        );
        assert!(
            temporal.reach().unwrap_err().to_string().ends_with(
                "node `mean` (tetera.temporal_mean) keeps state from one run to the next"
            )
        );
        // A node whose step says nothing of its reach needs the whole image.
        let stats = plan(
            r#"{"name": "stats", "inputs": {"image": {"kind": "image", "channels": 1, "sample": "u8"}},
            "nodes": [{"id": "s", "node": "tetera.statistics", "inputs": {"image": "image"}}],
            "outputs": {"t": "s.table"}}"#,
        );
        assert!(
            stats
                .reach()
                .unwrap_err()
                .to_string()
                .ends_with("node `s` (tetera.statistics) needs the whole image")
        );
        let mut p = plan(&raw_chain("bilinear"));
        let mut inputs = raw_inputs(64, 48);
        let other = raw_inputs(32, 48).remove("dark").unwrap();
        inputs.insert("dark".to_string(), other);
        assert!(
            p.run_tiled(&Context { compute: &c }, inputs, (16, 16))
                .unwrap_err()
                .to_string()
                .contains("of one size")
        );
        let mut held = plan(&raw_chain("bilinear"));
        let inputs = raw_inputs(64, 48);
        held.hold("dark", inputs["dark"].clone()).unwrap();
        let frame = BTreeMap::from([("raw".to_string(), inputs["raw"].clone())]);
        assert!(
            held.run_tiled(&Context { compute: &c }, frame, (16, 16))
                .unwrap_err()
                .to_string()
                .contains("let them go")
        );
    }

    #[test]
    fn an_output_tiles_cant_stitch_or_inputs_without_a_frame_are_refused() {
        // `test.count` reaches nothing, but gives a scalar.
        let d = Descriptor::new("test.count", 1, "counts")
            .input(PortSpec::new("image", PortType::ANY_IMAGE, "an image"))
            .output(PortSpec::new("n", PortType::Scalar, "pixels"));
        let mut registry = Registry::stock();
        registry
            .register(Arc::new(Stock::new(d, |params| {
                let _: NoParameters = parameters(params)?;
                local(0, |_, inputs| {
                    Ok(vec![Value::Scalar(
                        size(inputs[0]).map_or(0.0, |(w, h)| f64::from(w * h)),
                    )])
                })
            })))
            .unwrap();
        let recipe = Recipe::from_json(
            r#"{"name": "count", "inputs": {"image": {"kind": "image", "channels": 1, "sample": "u8"}, "k": {"kind": "scalar"}},
            "nodes": [{"id": "n", "node": "test.count", "inputs": {"image": "image"}}],
            "outputs": {"n": "n.n"}}"#,
        )
        .unwrap();
        let mut p = Plan::new(&recipe, &registry).unwrap();
        let c = compute();
        let image = Value::U8(Image::new(8, 8, 1, vec![0; 64]).unwrap());
        let inputs = BTreeMap::from([
            ("image".to_string(), image),
            ("k".to_string(), Value::Scalar(1.0)),
        ]);
        assert_eq!(
            p.run_tiled(&Context { compute: &c }, inputs, (4, 4))
                .unwrap_err()
                .to_string(),
            "recipe `count`'s output `n` is a scalar, not a frame or an image of its tile's size, which tiles can't stitch"
        );
        let numbers = Recipe::from_json(
            r#"{"name": "numbers", "inputs": {"image": {"kind": "image", "channels": 1, "sample": "u8"}},
            "nodes": [{"id": "n", "node": "test.count", "inputs": {"image": "image"}}],
            "outputs": {"n": "n.n"}}"#,
        )
        .unwrap();
        let mut p = Plan::new(&numbers, &registry).unwrap();
        assert!(
            p.run_tiled(&Context { compute: &c }, BTreeMap::new(), (4, 4))
                .unwrap_err()
                .to_string()
                .contains("no frame or image")
        );
    }

    #[test]
    fn a_tile_takes_its_halo_within_the_frame_and_halos_are_even() {
        assert_eq!(
            around(Rect::new(10, 20, 30, 40), 4, 100, 100),
            Rect::new(6, 16, 38, 48)
        );
        assert_eq!(
            around(Rect::new(2, 0, 10, 10), 4, 100, 100),
            Rect::new(0, 0, 16, 14)
        );
        assert_eq!(
            around(Rect::new(90, 94, 10, 6), 6, 100, 100),
            Rect::new(84, 88, 16, 12)
        );
        assert_eq!(
            around(Rect::new(0, 0, 8, 8), 0, 8, 8),
            Rect::new(0, 0, 8, 8)
        );
        assert_eq!([0, 1, 4, 5, 6, 7].map(even), [0, 2, 4, 6, 6, 8]);
    }

    #[test]
    fn values_cut_into_tiles_and_stitched_come_back_whole() {
        let layout = tetera_imaging::MosaicLayout {
            width: 8,
            height: 6,
            pattern: tetera_core::frame::format::BayerPattern::Grbg,
            significant_bits: 12,
        };
        let values = [
            Value::Raw(Mosaic::new(layout, (0..48).collect()).unwrap()),
            Value::U8(Image::new(8, 6, 3, (0..144).map(|v| v as u8).collect()).unwrap()),
            Value::U16(Image::new(8, 6, 1, (0..48).map(|v| v * 7).collect()).unwrap()),
            Value::F32(Image::new(8, 6, 3, (0..144).map(|v| v as f32 / 4.0).collect()).unwrap()),
        ];
        for value in values {
            let mut canvas: Option<Canvas> = None;
            for y in (0..6).step_by(2) {
                for x in (0..8).step_by(4) {
                    let part = cut(&value, Rect::new(x, y, 4, 2)).unwrap();
                    assert_eq!(size(&part), Some((4, 2)));
                    let c = canvas.get_or_insert_with(|| Canvas::new(&part, 8, 6).unwrap());
                    // Each tile's right half, from its cut with a column more on the left.
                    let wider =
                        cut(&value, Rect::new(x.saturating_sub(2), y, 4 + x.min(2), 2)).unwrap();
                    c.paste(&wider, Rect::new(x.min(2), 0, 4, 2), (x, y))
                        .unwrap();
                }
            }
            assert_eq!(canvas.unwrap().finish().unwrap(), value);
        }
        assert!(cut(&Value::Scalar(1.0), Rect::new(0, 0, 1, 1)).is_err());
        assert_eq!(size(&Value::Scalar(1.0)), None);
    }

    #[test]
    fn a_tile_unlike_the_ones_before_it_is_refused() {
        let rgb =
            |channels| Image::new(2, 2, channels, vec![0u16; 4 * usize::from(channels)]).unwrap();
        let mut canvas = Canvas::new(&Value::U16(rgb(3)), 4, 4).unwrap();
        let whole = Rect::new(0, 0, 2, 2);
        assert!(canvas.paste(&Value::U16(rgb(1)), whole, (0, 0)).is_err());
        assert!(canvas.paste(&Value::U16(rgb(3)), whole, (2, 2)).is_ok());
        let bytes = |channels| {
            Value::U8(Image::new(2, 2, channels, vec![0u8; 4 * usize::from(channels)]).unwrap())
        };
        let mut canvas = Canvas::new(&bytes(3), 4, 4).unwrap();
        assert!(canvas.paste(&bytes(1), whole, (0, 0)).is_err());
        assert!(canvas.paste(&bytes(3), whole, (0, 0)).is_ok());
        let floats = |channels| {
            Value::F32(Image::new(2, 2, channels, vec![0f32; 4 * usize::from(channels)]).unwrap())
        };
        let mut canvas = Canvas::new(&floats(1), 4, 4).unwrap();
        assert!(canvas.paste(&floats(3), whole, (0, 0)).is_err());
        assert!(canvas.paste(&floats(1), whole, (0, 0)).is_ok());
        assert!(
            canvas
                .paste(&bytes(1), whole, (0, 0))
                .unwrap_err()
                .to_string()
                .contains("unlike the tiles before it")
        );
        assert!(Canvas::new(&Value::Scalar(1.0), 4, 4).is_err());
    }

    #[test]
    fn a_batch_runs_in_order_its_state_carried() {
        let c = compute();
        let mut p = plan(
            r#"{"name": "mean", "inputs": {"raw": {"kind": "raw"}},
            "nodes": [{"id": "mean", "node": "tetera.temporal_mean", "params": {"window": 3}, "inputs": {"raw": "raw"}}],
            "outputs": {"raw": "mean.raw"}}"#,
        );
        let clean = Sensor::default().expose(&test_pattern(16, 8, 12)).unwrap();
        let frames: Vec<Mosaic> = (0..4)
            .map(|seed| {
                Mosaic::new(*clean.layout(), noisy(clean.sites(), 50.0, 4_095, seed)).unwrap()
            })
            .collect();
        let batch = frames
            .iter()
            .map(|f| BTreeMap::from([("raw".to_string(), Value::Raw(f.clone()))]));
        let out = p.run_batch(&Context { compute: &c }, batch).unwrap();
        let mut window = tetera_imaging::TemporalMean::new(3).unwrap();
        assert_eq!(out.len(), 4);
        for (o, f) in out.iter().zip(&frames) {
            assert_eq!(
                o["raw"],
                Value::Raw(window.push_cpu(c.cpu(), f, false).unwrap())
            );
        }
        let bad = vec![
            BTreeMap::from([("raw".to_string(), Value::Raw(frames[0].clone()))]),
            BTreeMap::new(),
        ];
        assert_eq!(
            p.run_batch(&Context { compute: &c }, bad)
                .unwrap_err()
                .to_string(),
            "item 1 of the batch: recipe `mean` takes input `raw`, which wasn't given"
        );
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use crate::recipe::Recipe;
    use crate::registry::Registry;
    use tetera_compute::Compute;
    use tetera_imaging::pattern::{Sensor, test_pattern};

    #[test]
    fn tiles_on_the_gpu_give_what_the_whole_frame_gives_on_the_cpu() {
        let _gpu = tetera_compute::cuda_or_skip!();
        let gpu = Compute::from_probe(&tetera_compute::SystemProbe).unwrap();
        let cpu = Compute::cpu_only().unwrap();
        let recipe = Recipe::from_json(
            r#"{"name": "on the GPU", "inputs": {"raw": {"kind": "raw"}},
            "nodes": [
                {"id": "rgb", "node": "tetera.demosaic", "inputs": {"raw": "raw"}},
                {"id": "smooth", "node": "tetera.bilateral", "params": {"sigma_spatial": 1.0, "sigma_range": 200}, "inputs": {"image": "rgb.image"}}
            ],
            "outputs": {"smooth": "smooth.image"}}"#,
        )
        .unwrap();
        let raw = Sensor::default()
            .expose(&test_pattern(128, 96, 12))
            .unwrap();
        let inputs = BTreeMap::from([("raw".to_string(), Value::Raw(raw))]);
        let mut whole = Plan::new(&recipe, &Registry::stock()).unwrap();
        let expected = whole
            .run(&Context { compute: &cpu }, inputs.clone())
            .unwrap();
        let mut tiled = Plan::new(&recipe, &Registry::stock()).unwrap();
        assert_eq!(
            tiled
                .run_tiled(&Context { compute: &gpu }, inputs, (40, 32))
                .unwrap(),
            expected
        );
    }
}
