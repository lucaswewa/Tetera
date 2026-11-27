# Third-party notices

Tetera is under the MIT licence. Its Rust dependencies' licences are checked by `cargo deny` (ADR-0002); this file holds the notices that some of them ask to be shown with Tetera itself.

## libjpeg-turbo

Tetera encodes JPEG with libjpeg-turbo 3.1.0, which the `turbojpeg-sys` crate builds from source and links into Tetera (P1.05b). libjpeg-turbo is under three permissive licences: the IJG licence, the Modified (3-clause) BSD licence, and, for its SIMD code, the zlib licence. Their full texts, and the project's own summary of how to comply, are in [licenses/libjpeg-turbo](licenses/libjpeg-turbo/): `LICENSE.md` and `README.ijg`.

As the IJG licence asks:

> This software is based in part on the work of the Independent JPEG Group.

Tetera's binaries use the TurboJPEG API, so the Modified BSD licence's text, in `licenses/libjpeg-turbo/LICENSE.md`, goes with them too.
