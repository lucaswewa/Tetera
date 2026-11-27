//! Tetera's imaging algorithms, L1 (plan §6.6): each has a CPU twin here, and, with the `cuda`
//! feature, a CUDA kernel in `tetera-compute`, which is the crate allowed to launch one. The
//! two agree bit for bit wherever the arithmetic allows it.
//!
//! From P1.05a:
//! - [`image`]: Bayer mosaics and images of interleaved channels, on the host and the device;
//! - [`bayer`]: a mosaic's four colour planes, its half-resolution superpixel, and RGB sampled
//!   into a mosaic;
//! - [`demosaic`]: bilinear and Malvar-He-Cutler, to RGB in 16 bits or floats;
//! - [`pattern`]: a test pattern, and the PSNR of an image against it.
//!
//! From P1.05b:
//! - [`bayer`] bins a mosaic by 2, 3 or 4, keeping its layout;
//! - [`resize`]: area averaging to shrink, bilinear to enlarge, by integer weight tables;
//! - [`display`]: black and white levels and a tone curve, to 8 bits, by lookup tables;
//! - [`jpeg`]: JPEG on the CPU, through libjpeg-turbo.
//!
//! From P1.06a:
//! - [`correct`]: dark subtraction, with a library of master darks, and the flat field;
//! - [`colour`]: white balance and a colour matrix, then levels and a curve, to 8 bits;
//! - [`isp`]: the whole chain, from a raw frame to a preview, stage by stage.
//!
//! Each operation runs on whichever backend a [`Compute`](tetera_compute::Compute) chose;
//! `_cpu` runs its twin, and `_device` leaves data on the GPU.

pub mod bayer;
pub mod colour;
pub mod correct;
pub mod demosaic;
pub mod display;
pub mod image;
pub mod isp;
pub mod jpeg;
pub mod pattern;
pub mod resize;

pub use bayer::BinMode;
pub use colour::{Colour, ColourCorrection};
pub use correct::{DarkLibrary, FlatField, MasterDark};
pub use demosaic::{Demosaic, Sample};
pub use display::{Curve, DisplayTransform, Levels};
pub use image::{BayerPlanes, Image, Mosaic, MosaicLayout};
#[cfg(feature = "cuda")]
pub use image::{DeviceImage, DeviceMosaic};
pub use isp::{Isp, IspParams, Reduce};
pub use jpeg::JpegEncoder;
pub use resize::Resize;
