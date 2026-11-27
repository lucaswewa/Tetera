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
//! Each operation runs on whichever backend a [`Compute`](tetera_compute::Compute) chose;
//! `_cpu` runs its twin, and `_device` leaves data on the GPU.

pub mod bayer;
pub mod demosaic;
pub mod image;
pub mod pattern;

pub use demosaic::{Demosaic, Sample};
pub use image::{BayerPlanes, Image, Mosaic, MosaicLayout};
#[cfg(feature = "cuda")]
pub use image::{DeviceImage, DeviceMosaic};
