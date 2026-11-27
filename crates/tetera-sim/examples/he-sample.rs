//! The H&E sample on the CPU (P1.11a): regions rendered at the sampling of the MIChrome 20's
//! 2.4 µm pixels through the 5×, 20× and 50× objectives, depth slices through a nucleus at
//! 50×, and a fold, written as PNGs to `target/he-sample/` with their timings.
//!
//! Run it with `cargo run --release -p tetera-sim --example he-sample [-- seed]`; `-- --smoke`
//! renders small regions.

use std::path::Path;
use std::time::Instant;
use tetera_core::{Error, Result};
use tetera_sim::{Depth, HeSample, Region, Rendered, SectionConfig, render};

const PIXEL: f64 = 2.4;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let smoke = args.iter().any(|a| a == "--smoke");
    let seed: u64 = args.iter().find_map(|a| a.parse().ok()).unwrap_or(7);
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/he-sample");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out)
        .map_err(|e| Error::internal(format!("{}: {e}", out.display())))?;
    let scale = if smoke { 8 } else { 1 };
    let sample = HeSample::new(seed, SectionConfig::default())?;
    println!("The H&E sample from seed {seed}, a section 4 µm thick:");

    // The 5× field's middle, and a nucleus near it for 50×.
    let middle = (2_000.0, 2_000.0);
    let nucleus = sample
        .nuclei(
            middle.0 - 150.0,
            middle.1 - 150.0,
            middle.0 + 150.0,
            middle.1 + 150.0,
        )
        .into_iter()
        .find(|n| (1.0..3.0).contains(&n.depth))
        .ok_or_else(|| Error::not_found("no nucleus near the middle"))?;
    let views = [
        ("overview", PIXEL / 2.0, middle, 1_024, 1_024),
        ("5x", 5.0, middle, 1_536, 1_024),
        ("20x", 20.0, middle, 1_024, 768),
        ("50x", 50.0, nucleus.centre, 768, 768),
    ];
    for (name, magnification, centre, width, height) in views {
        let (width, height) = (width / scale, height / scale);
        let pixel = PIXEL / magnification;
        let region = centred(centre, width, height, pixel, Depth::Whole);
        let r = timed(&sample, &region, &format!("{name}, {pixel:.3} µm a pixel"))?;
        png(&out.join(format!("he-{name}.png")), &r, 1.0)?;
    }

    // Depth slices through the nucleus at 50×, each 1 µm, shown as dark as a whole section.
    println!(
        "\nSlices through the nucleus at ({:.1}, {:.1}) µm, its centre {:.2} µm deep:",
        nucleus.centre.0, nucleus.centre.1, nucleus.depth
    );
    for from in [0.0f32, 1.5, 3.0] {
        let region = centred(
            nucleus.centre,
            512 / scale,
            512 / scale,
            PIXEL / 50.0,
            Depth::Range(from, from + 1.0),
        );
        let r = timed(
            &sample,
            &region,
            &format!("{from:.1} to {:.1} µm", from + 1.0),
        )?;
        let mid = &r.stains[(r.height / 2 * r.width + r.width / 2) as usize];
        println!(
            "      its middle pixel's haematoxylin: {:.3} OD",
            mid.haematoxylin
        );
        png(&out.join(format!("he-50x-depth-{from:.1}.png")), &r, 4.0)?;
    }

    // A fold, at 5×.
    let folded = HeSample::new(
        seed,
        SectionConfig {
            folds: true,
            ..SectionConfig::default()
        },
    )?;
    if let Some(x) = (0..20_000)
        .map(|i| f64::from(i) * 5.0)
        .find(|&x| folded.folded(x, middle.1) && folded.column(x, middle.1).stains.eosin > 0.0)
    {
        println!("\nA fold at x = {x:.0} µm:");
        let region = centred(
            (x, middle.1),
            768 / scale,
            512 / scale,
            PIXEL / 5.0,
            Depth::Whole,
        );
        let r = timed(&folded, &region, "5x, folds on")?;
        png(&out.join("he-5x-fold.png"), &r, 1.0)?;
    }
    println!("\nWrote the PNGs to {}", out.display());
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

fn centred(centre: (f64, f64), width: u32, height: u32, pixel: f64, depth: Depth) -> Region {
    Region {
        origin: (
            centre.0 - f64::from(width) * pixel / 2.0,
            centre.1 - f64::from(height) * pixel / 2.0,
        ),
        width,
        height,
        pixel,
        depth,
    }
}

fn timed(sample: &HeSample, region: &Region, what: &str) -> Result<Rendered> {
    let start = Instant::now();
    let r = render(sample, region)?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let s = region.points();
    println!(
        "  {what}: {} × {} pixels of {}×{} points, {:.0} × {:.0} µm, in {ms:.0} ms ({:.2} µs a point)",
        region.width,
        region.height,
        s,
        s,
        f64::from(region.width) * region.pixel,
        f64::from(region.height) * region.pixel,
        ms * 1e3 / f64::from(region.width * region.height * s * s)
    );
    Ok(r)
}

/// Writes `r` as an 8-bit RGB PNG, its stains times `gain`.
fn png(path: &Path, r: &Rendered, gain: f32) -> Result<()> {
    let shown = Rendered {
        stains: r
            .stains
            .iter()
            .map(|s| tetera_sim::Stains {
                haematoxylin: s.haematoxylin * gain,
                eosin: s.eosin * gain,
            })
            .collect(),
        ..r.clone()
    };
    let rgb = shown.srgb8();
    let row = r.width as usize * 3;
    let mut raw = Vec::with_capacity((row + 1) * r.height as usize);
    for line in rgb.chunks(row) {
        raw.push(0);
        raw.extend_from_slice(line);
    }
    // zlib, in stored blocks: no compression, no dependency.
    let mut z = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (k, block) in blocks.iter().enumerate() {
        z.push(u8::from(k + 1 == blocks.len()));
        let len = block.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend_from_slice(&r.width.to_be_bytes());
    header.extend_from_slice(&r.height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &z);
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png).map_err(|e| Error::internal(format!("{}: {e}", path.display())))
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let crc = crc32(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in bytes {
        a = (a + u32::from(x)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}
