//! Ports, their types, and the values they carry.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use tetera_imaging::{Image, Mosaic};

/// The type of an image's samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SampleType {
    /// 8 bits, unsigned.
    U8,
    /// 16 bits, unsigned.
    U16,
    /// 32-bit floats.
    F32,
}

impl fmt::Display for SampleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::F32 => "f32",
        })
    }
}

/// How an encoded image is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    /// JPEG.
    Jpeg,
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Jpeg => "JPEG",
        })
    }
}

/// What a port carries. On a node's input, an image's channels or sample type left out means
/// any; on an output that follows an input, they are that input's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum PortType {
    /// A raw frame: a Bayer mosaic of 16-bit sites.
    Raw,
    /// An image of interleaved channels.
    Image {
        /// Its channels a pixel: 1 or 3.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        channels: Option<u8>,
        /// Its samples' type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sample: Option<SampleType>,
    },
    /// An encoded image: bytes in a format.
    Encoded {
        /// The format.
        format: Encoding,
    },
    /// One number.
    Scalar,
    /// Named numbers.
    Table,
}

impl PortType {
    /// An image of any channels and samples.
    pub const ANY_IMAGE: Self = Self::Image {
        channels: None,
        sample: None,
    };

    /// An image of `channels` channels of `sample`.
    pub fn image(channels: u8, sample: SampleType) -> Self {
        Self::Image {
            channels: Some(channels),
            sample: Some(sample),
        }
    }

    /// Whether a port of this type takes a value of `given`, a type with nothing left open.
    pub fn accepts(&self, given: &PortType) -> bool {
        match (self, given) {
            (
                Self::Image { channels, sample },
                Self::Image {
                    channels: c,
                    sample: s,
                },
            ) => channels.is_none_or(|n| Some(n) == *c) && sample.is_none_or(|t| Some(t) == *s),
            _ => self == given,
        }
    }

    /// Whether it leaves nothing open.
    pub fn is_concrete(&self) -> bool {
        !matches!(
            self,
            Self::Image { channels: None, .. } | Self::Image { sample: None, .. }
        )
    }

    /// This type with what it leaves open taken from `input`'s.
    pub fn following(&self, input: &PortType) -> PortType {
        match (self, input) {
            (
                Self::Image { channels, sample },
                Self::Image {
                    channels: c,
                    sample: s,
                },
            ) => Self::Image {
                channels: channels.or(*c),
                sample: sample.or(*s),
            },
            _ => *self,
        }
    }
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Raw => f.write_str("a raw frame"),
            Self::Encoded { format } => write!(f, "a {format} image"),
            Self::Scalar => f.write_str("a scalar"),
            Self::Table => f.write_str("a table"),
            Self::Image { channels, sample } => {
                f.write_str("an image")?;
                match channels {
                    Some(1) => f.write_str(" of 1 channel")?,
                    Some(n) => write!(f, " of {n} channels")?,
                    None => {}
                }
                match (channels, sample) {
                    (Some(_), Some(t)) => write!(f, " of {t}"),
                    (None, Some(t)) => write!(f, " of {t}"),
                    _ => Ok(()),
                }
            }
        }
    }
}

/// A port of a node: its name, its type, and what it is for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PortSpec {
    /// Its name, unique among the node's inputs, or among its outputs.
    pub name: &'static str,
    /// Its type.
    #[serde(rename = "type")]
    pub ty: PortType,
    /// On an output, the input whose type fills in what its own leaves open.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub follows: Option<&'static str>,
    /// What it is for, in a sentence.
    pub summary: &'static str,
}

impl PortSpec {
    /// A port named `name` of type `ty`.
    pub fn new(name: &'static str, ty: PortType, summary: &'static str) -> Self {
        Self {
            name,
            ty,
            follows: None,
            summary,
        }
    }

    /// The same output, its open parts taken from the input named `input`.
    pub fn following(self, input: &'static str) -> Self {
        Self {
            follows: Some(input),
            ..self
        }
    }
}

/// Named numbers: a node's measurements.
pub type Table = BTreeMap<String, f64>;

/// A value on a port.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A raw frame.
    Raw(Mosaic),
    /// An image of 8-bit samples.
    U8(Image<u8>),
    /// An image of 16-bit samples.
    U16(Image<u16>),
    /// An image of floats.
    F32(Image<f32>),
    /// An encoded image, its bytes.
    Encoded(Encoding, Vec<u8>),
    /// A raw frame on the GPU.
    #[cfg(feature = "cuda")]
    DeviceRaw(OnDevice<tetera_imaging::DeviceMosaic>),
    /// An image of 8-bit samples on the GPU.
    #[cfg(feature = "cuda")]
    DeviceU8(OnDevice<tetera_imaging::DeviceImage<u8>>),
    /// An image of 16-bit samples on the GPU.
    #[cfg(feature = "cuda")]
    DeviceU16(OnDevice<tetera_imaging::DeviceImage<u16>>),
    /// An image of floats on the GPU.
    #[cfg(feature = "cuda")]
    DeviceF32(OnDevice<tetera_imaging::DeviceImage<f32>>),
    /// One number.
    Scalar(f64),
    /// Named numbers.
    Table(Table),
}

impl Value {
    /// Its type, with nothing left open.
    pub fn port_type(&self) -> PortType {
        match self {
            Self::Raw(_) => PortType::Raw,
            Self::U8(i) => PortType::image(i.channels(), SampleType::U8),
            Self::U16(i) => PortType::image(i.channels(), SampleType::U16),
            Self::F32(i) => PortType::image(i.channels(), SampleType::F32),
            Self::Encoded(format, _) => PortType::Encoded { format: *format },
            #[cfg(feature = "cuda")]
            Self::DeviceRaw(_) => PortType::Raw,
            #[cfg(feature = "cuda")]
            Self::DeviceU8(i) => PortType::image(i.channels(), SampleType::U8),
            #[cfg(feature = "cuda")]
            Self::DeviceU16(i) => PortType::image(i.channels(), SampleType::U16),
            #[cfg(feature = "cuda")]
            Self::DeviceF32(i) => PortType::image(i.channels(), SampleType::F32),
            Self::Scalar(_) => PortType::Scalar,
            Self::Table(_) => PortType::Table,
        }
    }
}

#[cfg(feature = "cuda")]
pub use device::OnDevice;

#[cfg(feature = "cuda")]
mod device {
    use super::Value;
    use std::sync::Arc;
    use tetera_compute::CudaBackend;
    use tetera_core::Result;
    use tetera_imaging::{DeviceImage, DeviceMosaic};

    /// A value on the GPU, shared between its readers. Two are equal when they are the same
    /// buffer, so a node can tell its input is the one it saw last time without reading it.
    #[derive(Debug)]
    pub struct OnDevice<T>(Arc<T>);

    impl<T> OnDevice<T> {
        /// `value`, to share.
        pub fn new(value: T) -> Self {
            Self(Arc::new(value))
        }

        /// How many hold it, for tests that a value is let go.
        #[cfg(test)]
        pub(crate) fn holders(&self) -> usize {
            Arc::strong_count(&self.0)
        }
    }

    impl<T> Clone for OnDevice<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }

    impl<T> PartialEq for OnDevice<T> {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.0, &other.0)
        }
    }

    impl<T> std::ops::Deref for OnDevice<T> {
        type Target = T;

        fn deref(&self) -> &T {
            &self.0
        }
    }

    impl Value {
        /// Whether it is on the GPU.
        pub fn on_device(&self) -> bool {
            matches!(
                self,
                Self::DeviceRaw(_) | Self::DeviceU8(_) | Self::DeviceU16(_) | Self::DeviceF32(_)
            )
        }

        /// It copied to the GPU, with the bytes copied, if it is a frame or an image on the
        /// host; numbers, tables and encoded images stay on the host.
        pub fn upload(&self, gpu: &CudaBackend) -> Result<Option<(Value, u64)>> {
            let bytes = |n: usize, size: usize| (n * size) as u64;
            Ok(Some(match self {
                Self::Raw(m) => (
                    Self::DeviceRaw(OnDevice::new(DeviceMosaic::upload(gpu, m)?)),
                    bytes(m.sites().len(), 2),
                ),
                Self::U8(i) => (
                    Self::DeviceU8(OnDevice::new(DeviceImage::upload(gpu, i)?)),
                    bytes(i.data().len(), 1),
                ),
                Self::U16(i) => (
                    Self::DeviceU16(OnDevice::new(DeviceImage::upload(gpu, i)?)),
                    bytes(i.data().len(), 2),
                ),
                Self::F32(i) => (
                    Self::DeviceF32(OnDevice::new(DeviceImage::upload(gpu, i)?)),
                    bytes(i.data().len(), 4),
                ),
                _ => return Ok(None),
            }))
        }

        /// It copied to the host, with the bytes copied, if it is on the GPU.
        pub fn download(&self, gpu: &CudaBackend) -> Result<Option<(Value, u64)>> {
            let bytes = |n: usize, size: usize| (n * size) as u64;
            Ok(Some(match self {
                Self::DeviceRaw(m) => {
                    let host = m.download(gpu)?;
                    let n = bytes(host.sites().len(), 2);
                    (Self::Raw(host), n)
                }
                Self::DeviceU8(i) => {
                    let host = i.download(gpu)?;
                    let n = bytes(host.data().len(), 1);
                    (Self::U8(host), n)
                }
                Self::DeviceU16(i) => {
                    let host = i.download(gpu)?;
                    let n = bytes(host.data().len(), 2);
                    (Self::U16(host), n)
                }
                Self::DeviceF32(i) => {
                    let host = i.download(gpu)?;
                    let n = bytes(host.data().len(), 4);
                    (Self::F32(host), n)
                }
                _ => return Ok(None),
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_open_image_type_accepts_any_image_that_fills_it() {
        let rgb16 = PortType::image(3, SampleType::U16);
        assert!(PortType::ANY_IMAGE.accepts(&rgb16));
        let any_u16 = PortType::Image {
            channels: None,
            sample: Some(SampleType::U16),
        };
        let grey = PortType::Image {
            channels: Some(1),
            sample: None,
        };
        assert!(any_u16.accepts(&rgb16) && !any_u16.accepts(&PortType::image(3, SampleType::U8)));
        assert!(grey.accepts(&PortType::image(1, SampleType::F32)) && !grey.accepts(&rgb16));
        assert!(rgb16.accepts(&rgb16) && !rgb16.accepts(&PortType::image(1, SampleType::U16)));
        assert!(!PortType::ANY_IMAGE.accepts(&PortType::Raw));
        assert!(!PortType::Raw.accepts(&rgb16) && PortType::Raw.accepts(&PortType::Raw));
        assert!(
            PortType::Table.accepts(&PortType::Table)
                && !PortType::Table.accepts(&PortType::Scalar)
        );
    }

    #[test]
    fn a_following_type_fills_in_what_it_leaves_open() {
        let rgb16 = PortType::image(3, SampleType::U16);
        assert_eq!(PortType::ANY_IMAGE.following(&rgb16), rgb16);
        let any_u8 = PortType::Image {
            channels: None,
            sample: Some(SampleType::U8),
        };
        assert_eq!(any_u8.following(&rgb16), PortType::image(3, SampleType::U8));
        assert_eq!(PortType::Table.following(&rgb16), PortType::Table);
        assert_eq!(rgb16.following(&PortType::Raw), rgb16);
        assert!(rgb16.is_concrete() && PortType::Raw.is_concrete());
        assert!(!PortType::ANY_IMAGE.is_concrete() && !any_u8.is_concrete());
        let grey = PortType::Image {
            channels: Some(1),
            sample: None,
        };
        assert!(!grey.is_concrete());
    }

    #[test]
    fn types_read_as_phrases_and_round_trip_through_json() {
        let cases = [
            (PortType::Raw, "a raw frame", r#"{"kind":"raw"}"#),
            (PortType::Scalar, "a scalar", r#"{"kind":"scalar"}"#),
            (
                PortType::Encoded {
                    format: Encoding::Jpeg,
                },
                "a JPEG image",
                r#"{"kind":"encoded","format":"jpeg"}"#,
            ),
            (PortType::Table, "a table", r#"{"kind":"table"}"#),
            (PortType::ANY_IMAGE, "an image", r#"{"kind":"image"}"#),
            (
                PortType::image(3, SampleType::U16),
                "an image of 3 channels of u16",
                r#"{"kind":"image","channels":3,"sample":"u16"}"#,
            ),
            (
                PortType::Image {
                    channels: Some(1),
                    sample: None,
                },
                "an image of 1 channel",
                r#"{"kind":"image","channels":1}"#,
            ),
            (
                PortType::Image {
                    channels: None,
                    sample: Some(SampleType::F32),
                },
                "an image of f32",
                r#"{"kind":"image","sample":"f32"}"#,
            ),
        ];
        for (ty, phrase, json) in cases {
            assert_eq!(ty.to_string(), phrase);
            assert_eq!(serde_json::to_string(&ty).unwrap(), json);
            assert_eq!(serde_json::from_str::<PortType>(json).unwrap(), ty);
        }
        assert_eq!(SampleType::U8.to_string(), "u8");
    }

    #[test]
    fn values_know_their_types() {
        let rgb = Image::new(2, 1, 3, vec![0u16; 6]).unwrap();
        assert_eq!(
            Value::U16(rgb).port_type(),
            PortType::image(3, SampleType::U16)
        );
        let grey = Image::new(1, 1, 1, vec![0u8]).unwrap();
        assert_eq!(
            Value::U8(grey).port_type(),
            PortType::image(1, SampleType::U8)
        );
        let float = Image::new(1, 1, 1, vec![0f32]).unwrap();
        assert_eq!(
            Value::F32(float).port_type(),
            PortType::image(1, SampleType::F32)
        );
        assert_eq!(Value::Scalar(1.0).port_type(), PortType::Scalar);
        let jpeg = PortType::Encoded {
            format: Encoding::Jpeg,
        };
        assert_eq!(Value::Encoded(Encoding::Jpeg, vec![0xFF]).port_type(), jpeg);
        assert!(jpeg.is_concrete() && jpeg.accepts(&jpeg) && !jpeg.accepts(&PortType::Raw));
        assert_eq!(Value::Table(Table::new()).port_type(), PortType::Table);
        let layout = tetera_imaging::MosaicLayout {
            width: 2,
            height: 2,
            pattern: tetera_core::frame::format::BayerPattern::Rggb,
            significant_bits: 12,
        };
        let raw = Mosaic::new(layout, vec![0; 4]).unwrap();
        assert_eq!(Value::Raw(raw).port_type(), PortType::Raw);
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;

    #[test]
    fn values_go_to_the_gpu_and_back_and_device_values_are_known_by_their_buffer() {
        let gpu = tetera_compute::cuda_or_skip!();
        let layout = tetera_imaging::MosaicLayout {
            width: 4,
            height: 2,
            pattern: tetera_core::frame::format::BayerPattern::Rggb,
            significant_bits: 12,
        };
        let host = [
            (
                Value::Raw(Mosaic::new(layout, (0..8).collect()).unwrap()),
                16,
            ),
            (
                Value::U8(Image::new(2, 1, 3, vec![1u8, 2, 3, 4, 5, 6]).unwrap()),
                6,
            ),
            (Value::U16(Image::new(2, 1, 1, vec![7u16, 8]).unwrap()), 4),
            (
                Value::F32(Image::new(1, 1, 3, vec![0.5f32, 1.5, 2.5]).unwrap()),
                12,
            ),
        ];
        for (value, bytes) in host {
            let (up, sent) = value.upload(&gpu).unwrap().unwrap();
            assert!(up.on_device() && !value.on_device());
            assert_eq!((up.port_type(), sent), (value.port_type(), bytes));
            let (down, received) = up.download(&gpu).unwrap().unwrap();
            assert_eq!((down, received), (value.clone(), bytes));
            assert!(
                value.download(&gpu).unwrap().is_none(),
                "on the host already"
            );
            assert!(up.upload(&gpu).unwrap().is_none(), "on the GPU already");
            assert_eq!(up, up.clone(), "the same buffer");
            let (again, _) = value.upload(&gpu).unwrap().unwrap();
            assert_ne!(up, again, "another buffer, though the same values");
        }
        for stays in [
            Value::Scalar(1.0),
            Value::Table(Table::new()),
            Value::Encoded(Encoding::Jpeg, vec![1]),
        ] {
            assert!(stays.upload(&gpu).unwrap().is_none() && !stays.on_device());
        }
        let a = OnDevice::new(1);
        assert_eq!(a.holders(), 1);
        let b = a.clone();
        assert_eq!((a.holders(), *b), (2, 1));
    }
}
