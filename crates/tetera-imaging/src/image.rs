//! The images the operations take and give: a Bayer mosaic of 16-bit sites, and images of
//! interleaved channels, on the host; with the `cuda` feature, the same on a device. All are
//! stored row by row with no padding.

use tetera_core::frame::format::Channel;
use tetera_core::{Error, PixelFormat, PixelView, Rect, Result};

pub use tetera_compute::imaging::MosaicLayout;

/// A Bayer mosaic on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mosaic {
    layout: MosaicLayout,
    sites: Vec<u16>,
}

impl Mosaic {
    /// A mosaic of `sites`, which must number `layout.sites()`, with 1 to 16 significant bits.
    pub fn new(layout: MosaicLayout, sites: Vec<u16>) -> Result<Self> {
        if !(1..=16).contains(&layout.significant_bits) {
            return Err(Error::invalid_argument(format!(
                "a 16-bit site holds 1 to 16 significant bits, not {}",
                layout.significant_bits
            )));
        }
        if sites.len() != layout.sites() {
            return Err(Error::invalid_argument(format!(
                "a {} × {} mosaic has {} sites, not {}",
                layout.width,
                layout.height,
                layout.sites(),
                sites.len()
            )));
        }
        Ok(Self { layout, sites })
    }

    /// A raw frame's mosaic, from a Bayer16 or Bayer8 frame.
    pub fn from_frame(frame: &impl PixelView) -> Result<Self> {
        let frame_layout = *frame.layout();
        let (pattern, significant_bits) = match frame_layout.format {
            PixelFormat::Bayer16 {
                pattern,
                significant_bits,
            } => (pattern, significant_bits),
            PixelFormat::Bayer8 { pattern } => (pattern, 8),
            other => {
                return Err(Error::invalid_argument(format!(
                    "a {other:?} frame isn't a Bayer mosaic"
                )));
            }
        };
        let layout = MosaicLayout {
            width: frame_layout.width,
            height: frame_layout.height,
            pattern,
            significant_bits,
        };
        let mut sites = Vec::with_capacity(layout.sites());
        for y in 0..layout.height {
            match frame_layout.format {
                PixelFormat::Bayer8 { .. } => {
                    sites.extend(frame.row(0, y).iter().map(|&b| u16::from(b)))
                }
                _ => sites.extend(frame.samples_u16(y)),
            }
        }
        Self::new(layout, sites)
    }

    /// Its shape.
    pub fn layout(&self) -> &MosaicLayout {
        &self.layout
    }

    /// Its sites.
    pub fn sites(&self) -> &[u16] {
        &self.sites
    }

    /// The site at column `x`, row `y`, mirrored at the edges without repeating them: −1
    /// reads 1, and `width` reads `width − 2`. Mirroring that way keeps each site's colour.
    pub fn mirrored(&self, x: i32, y: i32) -> u16 {
        let (w, h) = (self.layout.width as i32, self.layout.height as i32);
        self.sites[(mirror(y, h) * w + mirror(x, w)) as usize]
    }

    /// The part of it inside `roi`, with its pattern moved for an odd column or row.
    pub fn crop(&self, roi: Rect) -> Result<Self> {
        let whole = Rect::new(0, 0, self.layout.width, self.layout.height);
        if roi.is_empty() || !whole.contains_rect(&roi) {
            return Err(Error::invalid_argument(format!(
                "{roi:?} isn't inside the {} × {} mosaic",
                self.layout.width, self.layout.height
            )));
        }
        let layout = MosaicLayout {
            width: roi.width,
            height: roi.height,
            pattern: self.layout.pattern.cropped(roi.x, roi.y),
            ..self.layout
        };
        let width = self.layout.width as usize;
        let mut sites = Vec::with_capacity(layout.sites());
        for y in roi.y..roi.y + roi.height {
            let start = y as usize * width + roi.x as usize;
            sites.extend_from_slice(&self.sites[start..start + roi.width as usize]);
        }
        Self::new(layout, sites)
    }
}

/// Index `i` mirrored into `0..n` without repeating the edge, for `n` of at least 2 and `i`
/// less than `n` beyond it.
pub(crate) fn mirror(i: i32, n: i32) -> i32 {
    let i = i.abs();
    if i >= n { 2 * (n - 1) - i } else { i }
}

/// An image of `channels` interleaved values a pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct Image<T> {
    width: u32,
    height: u32,
    channels: u8,
    data: Vec<T>,
}

impl<T: Copy> Image<T> {
    /// An image of `data`, which must hold `width × height × channels` values.
    pub fn new(width: u32, height: u32, channels: u8, data: Vec<T>) -> Result<Self> {
        let needed = width as usize * height as usize * usize::from(channels);
        if channels == 0 || data.len() != needed {
            return Err(Error::invalid_argument(format!(
                "a {width} × {height} image of {channels} channels holds {needed} values, not {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Its width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Its height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Values a pixel.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Its values, pixel by pixel.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Its values, taken.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// The pixel at column `x`, row `y`; empty beyond the image.
    pub fn pixel(&self, x: u32, y: u32) -> &[T] {
        if x >= self.width || y >= self.height {
            return &[];
        }
        let c = usize::from(self.channels);
        let start = (y as usize * self.width as usize + x as usize) * c;
        &self.data[start..start + c]
    }
}

/// A mosaic's four colour planes, each `width × height`, back to back in the order red, green
/// on a red row, green on a blue row, blue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BayerPlanes {
    width: u32,
    height: u32,
    data: Vec<u16>,
}

impl BayerPlanes {
    pub(crate) fn from_data(width: u32, height: u32, data: Vec<u16>) -> Self {
        debug_assert_eq!(data.len(), 4 * width as usize * height as usize);
        Self {
            width,
            height,
            data,
        }
    }

    /// Each plane's width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Each plane's height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// One colour's plane.
    pub fn plane(&self, channel: Channel) -> &[u16] {
        let size = self.width as usize * self.height as usize;
        let index = match channel {
            Channel::Red => 0,
            Channel::GreenOnRed => 1,
            Channel::GreenOnBlue => 2,
            Channel::Blue => 3,
        };
        &self.data[index * size..(index + 1) * size]
    }

    /// All four planes.
    pub fn data(&self) -> &[u16] {
        &self.data
    }
}

#[cfg(feature = "cuda")]
pub use device::{DeviceImage, DeviceMosaic};

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use tetera_compute::imaging::DemosaicOutput;
    use tetera_compute::{CudaBackend, DeviceBuffer};

    /// A Bayer mosaic on a device.
    #[derive(Debug)]
    pub struct DeviceMosaic {
        layout: MosaicLayout,
        buffer: DeviceBuffer<u16>,
    }

    impl DeviceMosaic {
        /// `mosaic` copied to `gpu`.
        pub fn upload(gpu: &CudaBackend, mosaic: &Mosaic) -> Result<Self> {
            Ok(Self {
                layout: mosaic.layout,
                buffer: gpu.upload(&mosaic.sites)?,
            })
        }

        /// It, copied back to the host.
        pub fn download(&self, gpu: &CudaBackend) -> Result<Mosaic> {
            Mosaic::new(self.layout, gpu.download(&self.buffer)?)
        }

        /// Its shape.
        pub fn layout(&self) -> &MosaicLayout {
            &self.layout
        }

        /// Its sites.
        pub fn buffer(&self) -> &DeviceBuffer<u16> {
            &self.buffer
        }
    }

    /// An image of interleaved channels on a device.
    #[derive(Debug)]
    pub struct DeviceImage<T> {
        width: u32,
        height: u32,
        channels: u8,
        buffer: DeviceBuffer<T>,
    }

    impl<T: DemosaicOutput> DeviceImage<T> {
        /// A zeroed image on `gpu`.
        pub fn zeros(gpu: &CudaBackend, width: u32, height: u32, channels: u8) -> Result<Self> {
            let len = width as usize * height as usize * usize::from(channels);
            Ok(Self {
                width,
                height,
                channels,
                buffer: gpu.zeros(len)?,
            })
        }

        /// `image` copied to `gpu`.
        pub fn upload(gpu: &CudaBackend, image: &Image<T>) -> Result<Self> {
            Ok(Self {
                width: image.width,
                height: image.height,
                channels: image.channels,
                buffer: gpu.upload(&image.data)?,
            })
        }

        /// It, copied back to the host.
        pub fn download(&self, gpu: &CudaBackend) -> Result<Image<T>> {
            Image::new(
                self.width,
                self.height,
                self.channels,
                gpu.download(&self.buffer)?,
            )
        }

        /// Its width in pixels.
        pub fn width(&self) -> u32 {
            self.width
        }

        /// Its height in pixels.
        pub fn height(&self) -> u32 {
            self.height
        }

        /// Values a pixel.
        pub fn channels(&self) -> u8 {
            self.channels
        }

        /// Its values.
        pub fn buffer(&self) -> &DeviceBuffer<T> {
            &self.buffer
        }

        /// Its values, to write.
        pub fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
            &mut self.buffer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tetera_core::frame::format::BayerPattern;
    use tetera_core::frame::{FramePool, HostAllocator};
    use tetera_core::{ErrorKind, FrameLayout};

    fn layout(width: u32, height: u32) -> MosaicLayout {
        MosaicLayout {
            width,
            height,
            pattern: BayerPattern::Rggb,
            significant_bits: 12,
        }
    }

    #[test]
    fn a_mosaic_checks_its_sites_and_bits() {
        let m = Mosaic::new(layout(3, 2), vec![0; 6]).unwrap();
        assert_eq!((m.layout().width, m.sites().len()), (3, 6));
        let err = Mosaic::new(layout(3, 2), vec![0; 5]).unwrap_err();
        assert_eq!(err.to_string(), "a 3 × 2 mosaic has 6 sites, not 5");
        for bits in [0, 17] {
            let bad = MosaicLayout {
                significant_bits: bits,
                ..layout(3, 2)
            };
            let err = Mosaic::new(bad, vec![0; 6]).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidArgument, "{bits} bits");
        }
        assert!(
            Mosaic::new(
                MosaicLayout {
                    significant_bits: 16,
                    ..layout(3, 2)
                },
                vec![0; 6]
            )
            .is_ok()
        );
        assert!(
            Mosaic::new(
                MosaicLayout {
                    significant_bits: 1,
                    ..layout(3, 2)
                },
                vec![0; 6]
            )
            .is_ok()
        );
    }

    #[test]
    fn sites_beyond_the_edges_mirror_without_repeating_them() {
        // 0 1 2 / 3 4 5 / 6 7 8
        let m = Mosaic::new(layout(3, 3), (0..9).collect()).unwrap();
        assert_eq!(m.mirrored(-1, 0), 1, "−1 reads 1");
        assert_eq!(m.mirrored(-2, 0), 2);
        assert_eq!(m.mirrored(3, 0), 1, "width reads width − 2");
        assert_eq!(m.mirrored(4, 0), 0);
        assert_eq!(m.mirrored(1, -1), 4);
        assert_eq!(m.mirrored(1, 3), 4);
        assert_eq!(m.mirrored(-1, -1), 4);
        assert_eq!(m.mirrored(2, 2), 8, "inside, itself");
        for (i, n) in [(-1, 2), (2, 2), (5, 7), (7, 7), (8, 7), (-2, 7)] {
            let j = mirror(i, n);
            assert!((0..n).contains(&j), "{i} in {n} → {j}");
            assert_eq!(i.rem_euclid(2), j.rem_euclid(2), "the colour is kept");
        }
    }

    #[test]
    fn a_crop_moves_the_pattern_and_keeps_its_rows() {
        let m = Mosaic::new(layout(4, 3), (0..12).collect()).unwrap();
        let c = m.crop(Rect::new(1, 1, 3, 2)).unwrap();
        assert_eq!(
            c.layout().pattern,
            BayerPattern::Bggr,
            "RGGB from (1, 1) starts on blue"
        );
        assert_eq!((c.layout().width, c.layout().height), (3, 2));
        assert_eq!(c.sites(), [5, 6, 7, 9, 10, 11]);
        assert_eq!(
            m.crop(Rect::new(2, 0, 2, 3)).unwrap().layout().pattern,
            BayerPattern::Rggb
        );
        assert_eq!(
            m.crop(Rect::new(1, 0, 1, 1)).unwrap().layout().pattern,
            BayerPattern::Grbg
        );
        assert_eq!(
            m.crop(Rect::new(0, 1, 1, 1)).unwrap().layout().pattern,
            BayerPattern::Gbrg
        );
        for roi in [
            Rect::new(0, 0, 5, 1),
            Rect::new(3, 2, 2, 1),
            Rect::new(0, 0, 0, 3),
        ] {
            assert!(m.crop(roi).is_err(), "{roi:?}");
        }
        let err = m.crop(Rect::new(0, 0, 4, 4)).unwrap_err();
        assert!(
            err.to_string().ends_with("isn't inside the 4 × 3 mosaic"),
            "{err}"
        );
    }

    #[test]
    fn a_raw_frame_becomes_a_mosaic() {
        let format = PixelFormat::bayer16(BayerPattern::Gbrg, 12).unwrap();
        let pool =
            FramePool::new(FrameLayout::new(format, 5, 3).unwrap(), 1, &HostAllocator).unwrap();
        let mut frame = pool.take(Duration::ZERO).unwrap();
        for y in 0..3 {
            for x in 0..5 {
                frame.set_sample_u16(x, y, (y * 10 + x) as u16);
            }
        }
        let m = Mosaic::from_frame(&frame.freeze()).unwrap();
        assert_eq!(
            *m.layout(),
            MosaicLayout {
                pattern: BayerPattern::Gbrg,
                ..layout(5, 3)
            }
        );
        assert_eq!(&m.sites()[..7], [0, 1, 2, 3, 4, 10, 11]);
        assert_eq!(m.sites()[14], 24);

        let eight = FrameLayout::new(
            PixelFormat::Bayer8 {
                pattern: BayerPattern::Bggr,
            },
            2,
            2,
        )
        .unwrap();
        let pool = FramePool::new(eight, 1, &HostAllocator).unwrap();
        let mut frame = pool.take(Duration::ZERO).unwrap();
        frame.row_mut(0, 1).copy_from_slice(&[200, 255]);
        let m = Mosaic::from_frame(&frame.freeze()).unwrap();
        assert_eq!(
            (m.layout().significant_bits, m.layout().pattern),
            (8, BayerPattern::Bggr)
        );
        assert_eq!(m.sites(), [0, 0, 200, 255]);

        let mono = FrameLayout::new(PixelFormat::Mono16, 2, 2).unwrap();
        let pool = FramePool::new(mono, 1, &HostAllocator).unwrap();
        let err = Mosaic::from_frame(&pool.take(Duration::ZERO).unwrap().freeze()).unwrap_err();
        assert_eq!(err.to_string(), "a Mono16 frame isn't a Bayer mosaic");
    }

    #[test]
    fn images_check_their_size_and_give_their_pixels() {
        let image = Image::new(2, 2, 3, (0..12u16).collect()).unwrap();
        assert_eq!((image.width(), image.height(), image.channels()), (2, 2, 3));
        assert_eq!(image.pixel(1, 1), [9, 10, 11]);
        assert_eq!(image.pixel(0, 1), [6, 7, 8]);
        assert!(image.pixel(2, 0).is_empty() && image.pixel(0, 2).is_empty());
        assert_eq!(image.data().len(), 12);
        assert_eq!(image.clone().into_data()[11], 11);
        let err = Image::new(2, 2, 3, vec![0u16; 11]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "a 2 × 2 image of 3 channels holds 12 values, not 11"
        );
        assert!(
            Image::new(0, 0, 0, Vec::<u16>::new()).is_err(),
            "no channels"
        );
        assert!(
            Image::new(0, 3, 1, Vec::<f32>::new()).is_ok(),
            "empty, but whole"
        );
    }

    #[test]
    fn planes_are_found_by_colour() {
        let planes = BayerPlanes::from_data(1, 2, (0..8).collect());
        assert_eq!((planes.width(), planes.height()), (1, 2));
        assert_eq!(planes.plane(Channel::Red), [0, 1]);
        assert_eq!(planes.plane(Channel::GreenOnRed), [2, 3]);
        assert_eq!(planes.plane(Channel::GreenOnBlue), [4, 5]);
        assert_eq!(planes.plane(Channel::Blue), [6, 7]);
        assert_eq!(planes.data().len(), 8);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn images_and_mosaics_keep_their_shape_on_the_device() {
        let gpu = tetera_compute::cuda_or_skip!();
        let image = Image::new(4, 3, 2, (0..24).map(|v| v as f32 * 0.5).collect()).unwrap();
        let on_device = DeviceImage::upload(&gpu, &image).unwrap();
        assert_eq!(
            (on_device.width(), on_device.height(), on_device.channels()),
            (4, 3, 2)
        );
        assert_eq!(on_device.buffer().len(), 24);
        assert_eq!(on_device.download(&gpu).unwrap(), image);
        let zeros = DeviceImage::<u16>::zeros(&gpu, 5, 2, 3).unwrap();
        assert_eq!((zeros.width(), zeros.height(), zeros.channels()), (5, 2, 3));
        assert!(zeros.download(&gpu).unwrap().data().iter().all(|&v| v == 0));
        let mosaic = Mosaic::new(layout(3, 2), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let on_device = DeviceMosaic::upload(&gpu, &mosaic).unwrap();
        assert_eq!(*on_device.layout(), layout(3, 2));
        assert_eq!(on_device.buffer().len(), 6);
        assert_eq!(on_device.download(&gpu).unwrap(), mosaic);
    }
}
