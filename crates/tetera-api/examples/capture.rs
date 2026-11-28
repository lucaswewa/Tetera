//! A capture (P1.15d, D5): the server on a loopback port, the virtual camera streaming in its
//! binned live mode, and a burst of 20 full frames into a new project through the `capture`
//! Thing, followed as a job. Then the dataset as the `datasets` Thing has it, its tree on disk,
//! and its frames read back.
//!
//! Run it with `cargo run --release -p tetera-api --features cuda --example capture`;
//! `-- --smoke` takes 5 frames of a tiny sensor on the CPU. `-- --keep` keeps the data root.

use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetera_api::instrument::{Instrument, ObjectivesConfig, SensorSize, SimulatorConfig};
use tetera_api::server::{Roots, build};
use tetera_store::Samples;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let keep = std::env::args().any(|a| a == "--keep");
    let count = if smoke { 5 } else { 20 };
    let instrument = Instrument::new(
        &SimulatorConfig {
            gpu: !smoke,
            sensor: if smoke {
                SensorSize::Tiny
            } else {
                SensorSize::Auto
            },
            ..SimulatorConfig::default()
        },
        &ObjectivesConfig::default(),
    )?;
    let compute = instrument.compute().clone();
    let scratch = std::env::temp_dir().join(format!("tetera-capture-{}", std::process::id()));
    let roots = Roots::make(scratch.join("data"), scratch.join("scratch"))?;
    let service = build(Arc::new(instrument), roots)?;
    let store = service.store().clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(service.serve(listener, async move {
        let _ = stopped.await;
    }));
    println!(
        "The server at {addr}; the camera renders on the {} ({} × {}), streaming in its binned live mode.",
        compute.backend, compute.sensor[0], compute.sensor[1]
    );
    invoke(addr, "/datasets/create_project", json!({"name": "Example"})).await?;
    invoke(addr, "/camera/start_streaming", json!({})).await?;
    tokio::time::sleep(Duration::from_millis(500)).await;
    let began = Instant::now();
    let job = invoke(
        addr,
        "/capture/capture",
        json!({"project": "Example", "name": "a burst", "kind": {"kind": "burst", "count": count}}),
    )
    .await?;
    let id = job["id"].as_str().unwrap_or_default().to_string();
    println!("A burst of {count} full frames: job {id}");
    // Followed as a job: where it is, each tenth of the way.
    let mut tenth = -1.0;
    let job = loop {
        let job = invoke(addr, "/jobs/get", json!({ "id": id })).await?;
        let progress = job["progress"].as_f64().unwrap_or(0.0);
        if (progress * 10.0).floor() > tenth {
            tenth = (progress * 10.0).floor();
            println!(
                "  {:>3.0} %  {}",
                progress * 100.0,
                job["message"].as_str().unwrap_or_default()
            );
        }
        if matches!(
            job["state"].as_str(),
            Some("succeeded" | "failed" | "cancelled")
        ) {
            break job;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    anyhow::ensure!(job["state"] == "succeeded", "the capture ended {job}");
    let r = &job["result"];
    println!(
        "  in {:.1} s: {} frames, {} lost; to full resolution in {:.0} ms, back to the live view in {:.0} ms",
        began.elapsed().as_secs_f64(),
        r["dataset"]["frames"],
        r["missing"],
        r["switch_ms"].as_f64().unwrap_or(0.0),
        r["back_ms"].as_f64().unwrap_or(0.0)
    );
    let streaming = String::from_utf8(http(addr, "GET", "/camera/streaming", None).await?)?;
    let mode = String::from_utf8(http(addr, "GET", "/camera/mode", None).await?)?;
    println!("  the live view: streaming {streaming}, in mode {mode}");
    // The dataset, as the `datasets` Thing has it.
    let dataset_id = r["dataset"]["id"].as_str().unwrap_or_default().to_string();
    let got = invoke(addr, "/datasets/get", json!({ "id": dataset_id })).await?;
    let spec = &got["metadata"]["spec"];
    println!(
        "\nDataset {dataset_id}: {} × {}, {}, {} µm a pixel",
        spec["width"], spec["height"], spec["format"], spec["pixel_size_um"]
    );
    for f in got["files"].as_array().into_iter().flatten() {
        println!(
            "  {:<14} {:<9} {:>8.1} MB",
            f["path"].as_str().unwrap_or_default(),
            f["kind"].as_str().unwrap_or_default(),
            f["bytes"].as_f64().unwrap_or(0.0) / 1e6
        );
    }
    let thumbnail = http(
        addr,
        "GET",
        got["thumbnail"].as_str().unwrap_or_default(),
        None,
    )
    .await?;
    println!("  its thumbnail served: {} bytes", thumbnail.len());
    // Its tree on disk.
    let dir = store
        .root()
        .join(r["dataset"]["path"].as_str().unwrap_or_default());
    println!("\nOn disk, {}:", dir.display());
    tree(&dir)?;
    // Its frames, read back.
    let reader = store.read(&dataset_id)?;
    let frames = &reader.metadata().frames;
    let first = reader.read(0)?;
    let last = reader.read(reader.frames() - 1)?;
    let range = |s: &Samples| match s {
        Samples::U16(v) => (v.iter().min().copied(), v.iter().max().copied()),
        Samples::U8(v) => (
            v.iter().min().map(|&x| u16::from(x)),
            v.iter().max().map(|&x| u16::from(x)),
        ),
    };
    let span = frames
        .last()
        .zip(frames.first())
        .map(|(l, f)| {
            l.exposure_start
                .saturating_since(f.exposure_start)
                .as_secs_f64()
        })
        .unwrap_or(0.0);
    println!(
        "\nRead back: {} frames, numbers {} to {}, {:.1} a second on the instrument's clock; the first's samples {:?}, the last's {:?}",
        reader.frames(),
        frames.first().map_or(0, |f| f.seq),
        frames.last().map_or(0, |f| f.seq),
        (reader.frames() - 1) as f64 / span.max(f64::EPSILON),
        range(&first),
        range(&last)
    );
    invoke(addr, "/camera/stop_streaming", json!({})).await?;
    let _ = stop.send(());
    serving.await??;
    if keep {
        println!("\nKept: {}", scratch.display());
    } else {
        let _ = std::fs::remove_dir_all(&scratch);
    }
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// The files under `dir`, by path: each with its size, the OME-Zarr image's chunks, one a
/// frame, summed up.
fn tree(dir: &Path) -> anyhow::Result<()> {
    let mut files = Vec::new();
    let mut folders = vec![dir.to_path_buf()];
    while let Some(folder) = folders.pop() {
        for e in std::fs::read_dir(&folder)? {
            let e = e?;
            if e.file_type()?.is_dir() {
                folders.push(e.path());
            } else {
                let path = e.path().strip_prefix(dir)?.to_string_lossy().to_string();
                files.push((path, e.metadata()?.len()));
            }
        }
    }
    files.sort();
    let (chunks, others): (Vec<_>, Vec<_>) = files.iter().partition(|(p, _)| p.contains("/c/"));
    for (path, bytes) in others {
        println!("  {path:<32} {bytes:>10} bytes");
    }
    if let Some((first, _)) = chunks.first() {
        let bytes: u64 = chunks.iter().map(|(_, b)| b).sum();
        println!(
            "  {first:<32} and {} more chunks, {:.1} MB in all, zstd",
            chunks.len() - 1,
            bytes as f64 / 1e6
        );
    }
    Ok(())
}

/// Invokes the action at `path` with `body`, and follows its invocation to its end: its output.
async fn invoke(addr: SocketAddr, path: &str, body: Value) -> anyhow::Result<Value> {
    let started: Value = serde_json::from_slice(&http(addr, "POST", path, Some(&body)).await?)?;
    let href = started["href"]
        .as_str()
        .unwrap_or_default()
        .replace(&format!("http://{addr}"), "");
    loop {
        let now: Value = serde_json::from_slice(&http(addr, "GET", &href, None).await?)?;
        match now["status"].as_str() {
            Some("pending" | "running") => tokio::time::sleep(Duration::from_millis(20)).await,
            Some("completed") => return Ok(now["output"].clone()),
            _ => anyhow::bail!("{path}: {now}"),
        }
    }
}

/// `method path`, with a JSON `body` if given, over a plain socket: the answer's body.
async fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> anyhow::Result<Vec<u8>> {
    let body = body.map(Value::to_string).unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    tokio::task::spawn_blocking(move || {
        let mut s = std::net::TcpStream::connect(addr)?;
        s.write_all(request.as_bytes())?;
        let mut answer = Vec::new();
        s.read_to_end(&mut answer)?;
        let end = answer
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map_or(answer.len(), |at| at + 4);
        let head = String::from_utf8_lossy(&answer[..end]).to_string();
        anyhow::ensure!(
            head.starts_with("HTTP/1.1 2"),
            "{head}{}",
            String::from_utf8_lossy(&answer[end..])
        );
        Ok(answer[end..].to_vec())
    })
    .await?
}
