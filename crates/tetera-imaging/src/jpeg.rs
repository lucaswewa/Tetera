//! JPEG, on the CPU, through libjpeg-turbo and its SIMD code (the `turbojpeg` crate): RGB in
//! 4:2:0, or grey. MJPEG is a sequence of these, one a frame, so an encoder is made once and
//! kept for a stream.
//!
//! libjpeg-turbo is "based in part on the work of the Independent JPEG Group"; its licences
//! are in `THIRD_PARTY_NOTICES.md`.

use crate::image::Image;
use tetera_core::{Error, Result};
use turbojpeg::{Compressor, PixelFormat, Subsamp};

fn failed(what: &str, e: turbojpeg::Error) -> Error {
    Error::internal(format!("libjpeg-turbo couldn't {what}: {e}"))
}

/// A JPEG encoder at one quality, kept for a stream of frames.
pub struct JpegEncoder {
    compressor: Compressor,
    quality: u8,
}

impl std::fmt::Debug for JpegEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JpegEncoder")
            .field("quality", &self.quality)
            .finish_non_exhaustive()
    }
}

impl JpegEncoder {
    /// An encoder at `quality`, 1 to 100.
    pub fn new(quality: u8) -> Result<Self> {
        if !(1..=100).contains(&quality) {
            return Err(Error::invalid_argument(format!(
                "a JPEG's quality is 1 to 100, not {quality}"
            )));
        }
        let mut compressor = Compressor::new().map_err(|e| failed("start", e))?;
        compressor
            .set_quality(i32::from(quality))
            .map_err(|e| failed("set the quality", e))?;
        Ok(Self {
            compressor,
            quality,
        })
    }

    /// Its quality.
    pub fn quality(&self) -> u8 {
        self.quality
    }

    /// `image`, of three channels (RGB, in 4:2:0) or one (grey), as a JPEG file.
    pub fn encode(&mut self, image: &Image<u8>) -> Result<Vec<u8>> {
        let (format, subsamp) = match image.channels() {
            3 => (PixelFormat::RGB, Subsamp::Sub2x2),
            1 => (PixelFormat::GRAY, Subsamp::Gray),
            other => {
                return Err(Error::invalid_argument(format!(
                    "a JPEG holds RGB or grey, not {other} channels"
                )));
            }
        };
        if image.width() == 0 || image.height() == 0 {
            return Err(Error::invalid_argument(format!(
                "a JPEG needs a pixel at least, not {} × {}",
                image.width(),
                image.height()
            )));
        }
        self.compressor
            .set_subsamp(subsamp)
            .map_err(|e| failed("set the subsampling", e))?;
        let width = image.width() as usize;
        let pixels = turbojpeg::Image {
            pixels: image.data(),
            width,
            pitch: width * usize::from(image.channels()),
            height: image.height() as usize,
            format,
        };
        self.compressor
            .compress_to_vec(pixels)
            .map_err(|e| failed("compress", e))
    }
}

/// `image` as a JPEG file at `quality`, with an encoder made for it alone.
pub fn encode(image: &Image<u8>, quality: u8) -> Result<Vec<u8>> {
    JpegEncoder::new(quality)?.encode(image)
}

/// A JPEG file decoded to 8-bit RGB: for tests and checks, not the live path.
pub fn decode_rgb(jpeg: &[u8]) -> Result<Image<u8>> {
    let image = turbojpeg::decompress(jpeg, PixelFormat::RGB).map_err(|e| failed("decode", e))?;
    Image::new(image.width as u32, image.height as u32, 3, image.pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::{Curve, DisplayTransform, Levels};
    use crate::pattern::{psnr, test_pattern};
    use tetera_compute::CpuBackend;

    /// The test pattern in 8 bits, linear.
    fn rgb8(width: u32, height: u32) -> Image<u8> {
        let t = DisplayTransform::uniform(
            3,
            Levels {
                black: 0,
                white: 255,
            },
            Curve::Linear,
        )
        .unwrap();
        crate::display::display_cpu(
            &CpuBackend::new(2).unwrap(),
            &test_pattern(width, height, 8),
            &t,
        )
        .unwrap()
    }

    fn widen(image: &Image<u8>) -> Image<u16> {
        Image::new(
            image.width(),
            image.height(),
            image.channels(),
            image.data().iter().map(|&v| u16::from(v)).collect(),
        )
        .unwrap()
    }

    #[test]
    fn a_frame_comes_back_close_and_better_at_a_higher_quality() {
        let image = rgb8(320, 240);
        let low = encode(&image, 50).unwrap();
        let high = encode(&image, 95).unwrap();
        assert_eq!(&low[..3], [0xFF, 0xD8, 0xFF], "a JPEG's start");
        assert_eq!(&low[low.len() - 2..], [0xFF, 0xD9], "and its end");
        assert!(
            high.len() > low.len(),
            "{} bytes at 95, {} at 50",
            high.len(),
            low.len()
        );
        let (low, high) = (decode_rgb(&low).unwrap(), decode_rgb(&high).unwrap());
        assert_eq!(
            (high.width(), high.height(), high.channels()),
            (320, 240, 3)
        );
        let (low_db, high_db) = (
            psnr(&widen(&image), &widen(&low), 255).unwrap(),
            psnr(&widen(&image), &widen(&high), 255).unwrap(),
        );
        assert!(
            high_db > low_db && low_db > 25.0,
            "{low_db:.1} dB at 50, {high_db:.1} dB at 95"
        );
    }

    #[test]
    fn grey_and_flat_frames_encode_too() {
        let grey = Image::new(16, 8, 1, (0..128).map(|v| v as u8 * 2).collect()).unwrap();
        let jpeg = encode(&grey, 90).unwrap();
        let back = decode_rgb(&jpeg).unwrap();
        assert_eq!((back.width(), back.height()), (16, 8));
        assert!(
            back.data().chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]),
            "grey stays grey"
        );
        let flat = Image::new(9, 7, 3, [200u8, 100, 50].repeat(63)).unwrap();
        let back = decode_rgb(&encode(&flat, 100).unwrap()).unwrap();
        assert!(
            back.data().chunks(3).all(|p| p
                .iter()
                .zip([200u8, 100, 50])
                .all(|(&a, b)| a.abs_diff(b) <= 2)),
            "odd sizes"
        );
    }

    #[test]
    fn an_encoder_serves_a_stream_of_frames() {
        let mut encoder = JpegEncoder::new(80).unwrap();
        assert_eq!(encoder.quality(), 80);
        assert!(format!("{encoder:?}").contains("80"));
        let (a, b) = (rgb8(64, 48), rgb8(48, 64));
        let first = encoder.encode(&a).unwrap();
        let second = encoder.encode(&b).unwrap();
        let grey = encoder
            .encode(&Image::new(4, 4, 1, vec![9u8; 16]).unwrap())
            .unwrap();
        let again = encoder.encode(&a).unwrap();
        assert_eq!(
            first, again,
            "the same frame, the same file, after a grey one"
        );
        assert_eq!(decode_rgb(&second).unwrap().width(), 48);
        assert_eq!(decode_rgb(&grey).unwrap().width(), 4);
    }

    #[test]
    fn bad_qualities_shapes_and_files_are_refused() {
        for quality in [0, 101] {
            let err = JpegEncoder::new(quality).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!("a JPEG's quality is 1 to 100, not {quality}")
            );
        }
        assert!(JpegEncoder::new(1).is_ok() && JpegEncoder::new(100).is_ok());
        let rgba = Image::new(2, 2, 4, vec![0u8; 16]).unwrap();
        assert_eq!(
            encode(&rgba, 80).unwrap_err().to_string(),
            "a JPEG holds RGB or grey, not 4 channels"
        );
        let empty = Image::new(0, 4, 3, Vec::<u8>::new()).unwrap();
        assert_eq!(
            encode(&empty, 80).unwrap_err().to_string(),
            "a JPEG needs a pixel at least, not 0 × 4"
        );
        let err = decode_rgb(&[0xFF, 0xD8, 0, 1, 2]).unwrap_err();
        assert!(
            err.to_string().starts_with("libjpeg-turbo couldn't decode"),
            "{err}"
        );
    }
}
