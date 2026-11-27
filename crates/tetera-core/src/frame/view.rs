//! Safe views of a frame's pixels: planes, rows, samples and Bayer sites. Samples are read
//! and written little-endian, as every supported host stores them, without reinterpreting
//! bytes as wider types.

use super::format::{Channel, FrameLayout};
use super::pool::{Frame, FrameMut};

/// Reading a frame's pixels. [`Frame`] and [`FrameMut`] both offer it.
pub trait PixelView {
    /// The frame's layout.
    fn layout(&self) -> &FrameLayout;

    /// The frame's bytes.
    fn data(&self) -> &[u8];

    /// Plane `p`'s bytes, padding included. Planes beyond the format's are empty.
    fn plane(&self, p: usize) -> &[u8] {
        let layout = self.layout();
        if p >= layout.format.planes() {
            return &[];
        }
        let size = layout.plane_bytes();
        &self.data()[p * size..(p + 1) * size]
    }

    /// Row `y` of plane `p`: its pixels' bytes, without the padding. Empty beyond the frame.
    fn row(&self, p: usize, y: u32) -> &[u8] {
        let layout = *self.layout();
        if y >= layout.height {
            return &[];
        }
        let start = y as usize * layout.stride;
        self.plane(p)
            .get(start..start + layout.row_bytes())
            .unwrap_or(&[])
    }

    /// Row `y`'s 16-bit samples, for a 16-bit format's first plane.
    fn samples_u16(&self, y: u32) -> Vec<u16> {
        if self.layout().format.bytes_per_sample() != 2 {
            return Vec::new();
        }
        self.row(0, y)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect()
    }

    /// The 16-bit sample at column `x`, row `y`, for a single-sample 16-bit format
    /// (Bayer16, Mono16); none otherwise, or beyond the frame.
    fn sample_u16(&self, x: u32, y: u32) -> Option<u16> {
        let format = self.layout().format;
        if format.bytes_per_sample() != 2
            || format.samples_per_pixel() != 1
            || x >= self.layout().width
        {
            return None;
        }
        let i = x as usize * 2;
        self.row(0, y)
            .get(i..i + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    }

    /// The 8-bit sample at column `x`, row `y`, for a single-sample 8-bit format
    /// (Bayer8, Mono8); none otherwise, or beyond the frame.
    fn sample_u8(&self, x: u32, y: u32) -> Option<u8> {
        let format = self.layout().format;
        if format.bytes_per_pixel() != 1 || x >= self.layout().width {
            return None;
        }
        self.row(0, y).get(x as usize).copied()
    }

    /// The colour of the site at column `x`, row `y`, for a Bayer format.
    fn channel_at(&self, x: u32, y: u32) -> Option<Channel> {
        self.layout()
            .format
            .bayer_pattern()
            .map(|p| p.channel_at(x, y))
    }
}

impl PixelView for Frame {
    fn layout(&self) -> &FrameLayout {
        Frame::layout(self)
    }

    fn data(&self) -> &[u8] {
        Frame::data(self)
    }
}

impl PixelView for FrameMut {
    fn layout(&self) -> &FrameLayout {
        FrameMut::layout(self)
    }

    fn data(&self) -> &[u8] {
        FrameMut::data(self)
    }
}

impl FrameMut {
    /// Row `y` of plane `p`, writable: its pixels' bytes, without the padding. Empty beyond
    /// the frame.
    pub fn row_mut(&mut self, p: usize, y: u32) -> &mut [u8] {
        let layout = *FrameMut::layout(self);
        if y >= layout.height || p >= layout.format.planes() {
            return &mut [];
        }
        let start = p * layout.plane_bytes() + y as usize * layout.stride;
        self.data_mut()
            .get_mut(start..start + layout.row_bytes())
            .unwrap_or(&mut [])
    }

    /// Writes the 16-bit sample at column `x`, row `y`, for a single-sample 16-bit format.
    /// Returns whether there was such a sample to write.
    pub fn set_sample_u16(&mut self, x: u32, y: u32, value: u16) -> bool {
        let layout = *FrameMut::layout(self);
        let format = layout.format;
        if format.bytes_per_sample() != 2 || format.samples_per_pixel() != 1 || x >= layout.width {
            return false;
        }
        let i = x as usize * 2;
        match self.row_mut(0, y).get_mut(i..i + 2) {
            Some(bytes) => {
                bytes.copy_from_slice(&value.to_le_bytes());
                true
            }
            None => false,
        }
    }

    /// Fills row `y` of a 16-bit format from `samples`, as far as both go. Returns how many
    /// samples were written.
    pub fn fill_row_u16(&mut self, y: u32, samples: &[u16]) -> usize {
        if FrameMut::layout(self).format.bytes_per_sample() != 2 {
            return 0;
        }
        let row = self.row_mut(0, y);
        let mut written = 0;
        for (bytes, value) in row.as_chunks_mut::<2>().0.iter_mut().zip(samples) {
            bytes.copy_from_slice(&value.to_le_bytes());
            written += 1;
        }
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::format::{BayerPattern, PixelFormat};
    use crate::frame::pool::{FramePool, HostAllocator};
    use std::time::Duration;

    fn take(format: PixelFormat, width: u32, height: u32) -> FrameMut {
        let layout = FrameLayout::new(format, width, height).unwrap();
        FramePool::new(layout, 1, &HostAllocator)
            .unwrap()
            .take(Duration::ZERO)
            .unwrap()
    }

    #[test]
    fn samples_are_written_and_read_little_endian() {
        let mut frame = take(PixelFormat::bayer16(BayerPattern::Rggb, 12).unwrap(), 5, 3);
        assert!(frame.set_sample_u16(4, 2, 0x0FFF));
        assert!(frame.set_sample_u16(0, 0, 0x0102));
        assert!(!frame.set_sample_u16(5, 0, 1), "past the last column");
        assert!(!frame.set_sample_u16(0, 3, 1), "past the last row");
        assert_eq!(&frame.row(0, 0)[..2], &[0x02, 0x01]);
        let frame = frame.freeze();
        assert_eq!(frame.sample_u16(4, 2), Some(0x0FFF));
        assert_eq!(frame.sample_u16(5, 2), None);
        assert_eq!(frame.samples_u16(2), [0, 0, 0, 0, 0x0FFF]);
        assert_eq!(
            frame.sample_u8(0, 0),
            None,
            "a 16-bit frame has no 8-bit samples"
        );
    }

    #[test]
    fn a_16_bit_sample_isnt_written_into_other_formats() {
        let mut bytes = take(PixelFormat::Mono8, 4, 2);
        assert!(!bytes.set_sample_u16(0, 0, 0xFFFF));
        assert!(bytes.data().iter().all(|&b| b == 0), "nothing written");
        let mut rgb = take(PixelFormat::Rgb16, 4, 2);
        assert!(
            !rgb.set_sample_u16(0, 0, 0xFFFF),
            "an RGB16 pixel is three samples"
        );
        assert!(rgb.data().iter().all(|&b| b == 0));
    }

    #[test]
    fn rows_leave_out_the_padding_and_planes_follow_each_other() {
        let mut frame = take(PixelFormat::PlanarF32 { planes: 2 }, 3, 2);
        let layout = *FrameMut::layout(&frame);
        assert_eq!((layout.stride, layout.row_bytes()), (64, 12));
        frame.row_mut(1, 1).fill(7);
        assert_eq!(frame.row(1, 1), &[7; 12]);
        assert_eq!(frame.plane(1)[64..76], [7; 12]);
        assert!(frame.plane(0).iter().all(|&b| b == 0));
        assert_eq!(frame.plane(2), &[] as &[u8]);
        assert_eq!(frame.row(0, 2), &[] as &[u8]);
        assert!(frame.row_mut(2, 0).is_empty() && frame.row_mut(0, 2).is_empty());
        assert_eq!(frame.data().len(), 2 * 64 * 2);
    }

    #[test]
    fn whole_rows_fill_from_samples() {
        let mut frame = take(PixelFormat::Mono16, 4, 2);
        assert_eq!(
            frame.fill_row_u16(1, &[10, 20, 30, 40, 50]),
            4,
            "a row holds four"
        );
        assert_eq!(frame.fill_row_u16(0, &[1, 2]), 2);
        assert_eq!(frame.samples_u16(1), [10, 20, 30, 40]);
        assert_eq!(frame.samples_u16(0), [1, 2, 0, 0]);
        let mut bytes = take(PixelFormat::Mono8, 4, 2);
        assert_eq!(
            bytes.fill_row_u16(0, &[1]),
            0,
            "an 8-bit frame takes no 16-bit samples"
        );
        assert!(bytes.samples_u16(0).is_empty());
    }

    #[test]
    fn eight_bit_samples_and_bayer_sites_read_back() {
        let mut frame = take(
            PixelFormat::Bayer8 {
                pattern: BayerPattern::Grbg,
            },
            4,
            4,
        );
        frame.row_mut(0, 3)[2] = 200;
        assert_eq!(frame.sample_u8(2, 3), Some(200));
        assert_eq!(frame.sample_u8(4, 3), None);
        assert_eq!(frame.sample_u16(2, 3), None);
        assert_eq!(frame.channel_at(0, 0), Some(Channel::GreenOnRed));
        assert_eq!(frame.channel_at(1, 1), Some(Channel::GreenOnBlue));
        let rgb = take(PixelFormat::Rgb8, 2, 2).freeze();
        assert_eq!(rgb.channel_at(0, 0), None);
        assert_eq!(rgb.sample_u8(0, 0), None, "an RGB pixel isn't one sample");
        let rgb16 = take(PixelFormat::Rgb16, 2, 2);
        assert_eq!(rgb16.sample_u16(0, 0), None, "nor is an RGB16 one");
        assert_eq!(rgb16.samples_u16(0).len(), 6, "but its row's samples read");
    }
}
