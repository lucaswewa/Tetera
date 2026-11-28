//! The preview stream (P1.14b, ADR-0019): the server on a loopback port, the virtual camera
//! streaming in its binned live mode, and two clients of the `preview` Thing's stream. The
//! quick one shows each frame at once; the slow one takes 200 ms over each. Each prints the
//! frames it got, their rate, the time from each frame's exposure to its receipt, and the
//! quality its link came down to.
//!
//! Run it with `cargo run --release -p tetera-api --features cuda --example preview-stream`;
//! `-- --smoke` runs a small sensor on the CPU briefly.

use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetera_acquire::protocol::Header;
use tetera_api::instrument::{Instrument, ObjectivesConfig, SensorSize, SimulatorConfig};
use tetera_api::server::{Roots, build};
use tetera_core::UtcTime;
use tokio_tungstenite::tungstenite::Message;

/// What a client got.
struct Got {
    frames: usize,
    seconds: f64,
    /// From each frame's exposure's end to its receipt, ms.
    latencies: Vec<f64>,
    lowest_quality: u8,
    lowered: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let smoke = std::env::args().any(|a| a == "--smoke");
    let seconds = if smoke { 1.5 } else { 4.0 };
    let instrument = Instrument::new(
        &SimulatorConfig {
            gpu: !smoke,
            sensor: if smoke {
                SensorSize::Small
            } else {
                SensorSize::Auto
            },
            ..SimulatorConfig::default()
        },
        &ObjectivesConfig::default(),
    )?;
    let compute = instrument.compute().clone();
    let scratch =
        std::env::temp_dir().join(format!("tetera-preview-stream-{}", std::process::id()));
    let roots = Roots::make(scratch.join("data"), scratch.join("scratch"))?;
    let service = build(Arc::new(instrument), roots)?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(service.serve(listener, async move {
        let _ = stopped.await;
    }));
    println!(
        "The server at {addr}; the camera renders on the {} ({} × {}), in its binned live mode.",
        compute.backend, compute.sensor[0], compute.sensor[1]
    );
    http(addr, "POST", "/camera/start_streaming").await?;
    let quick = tokio::spawn(client(addr, Duration::ZERO, seconds));
    let slow = tokio::spawn(client(addr, Duration::from_millis(200), seconds));
    tokio::time::sleep(Duration::from_secs_f64(seconds / 2.0)).await;
    let sessions = http(addr, "GET", "/preview/sessions").await?;
    for (name, got) in [("quick", quick.await??), ("slow", slow.await??)] {
        let mut l = got.latencies.clone();
        l.sort_by(f64::total_cmp);
        let at = |q: f64| {
            l.get(((l.len() as f64 - 1.0) * q).round() as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        println!(
            "  the {name} client: {} frames in {:.1} s, {:.1} a second; exposure to receipt {:.0} ms median, {:.0} ms at the 95th percentile; {} lowered by its link, quality down to {}",
            got.frames,
            got.seconds,
            got.frames as f64 / got.seconds,
            at(0.5),
            at(0.95),
            got.lowered,
            got.lowest_quality
        );
    }
    let sessions: serde_json::Value = serde_json::from_str(&sessions)?;
    for s in sessions.as_array().into_iter().flatten() {
        println!(
            "  session {}: {} × {} viewport, link step {}, {} sent, {} in flight",
            s["id"],
            s["params"]["width"],
            s["params"]["height"],
            s["link_step"],
            s["sent"],
            s["in_flight"]
        );
    }
    http(addr, "POST", "/camera/stop_streaming").await?;
    let _ = stop.send(());
    serving.await??;
    let _ = std::fs::remove_dir_all(&scratch);
    if smoke {
        println!("\nsmoke: ok");
    }
    Ok(())
}

/// A client that takes `paint` over each frame, for `seconds`.
async fn client(addr: SocketAddr, paint: Duration, seconds: f64) -> anyhow::Result<Got> {
    let (mut socket, _) =
        tokio_tungstenite::connect_async(format!("ws://{addr}/preview/stream")).await?;
    let state = json!({"width": 1_280, "height": 853, "quality": 80});
    socket.send(Message::text(state.to_string())).await?;
    let started = Instant::now();
    let mut got = Got {
        frames: 0,
        seconds,
        latencies: Vec::new(),
        lowest_quality: 100,
        lowered: 0,
    };
    while started.elapsed().as_secs_f64() < seconds {
        let Ok(Some(message)) = tokio::time::timeout(Duration::from_secs(5), socket.next()).await
        else {
            anyhow::bail!("no frame within 5 s");
        };
        let Message::Binary(bytes) = message? else {
            continue;
        };
        let (header, _jpeg) = Header::parse(&bytes)?;
        let now = UtcTime::now().unix_nanos() / 1_000;
        got.latencies
            .push((now - header.exposure_end_utc_us) as f64 / 1e3);
        got.frames += 1;
        got.lowest_quality = got.lowest_quality.min(header.quality);
        got.lowered += usize::from(header.lowered);
        tokio::time::sleep(paint).await;
        let mut ack = state.clone();
        ack["ack"] = header.seq.into();
        socket.send(Message::text(ack.to_string())).await?;
    }
    got.seconds = started.elapsed().as_secs_f64();
    socket.close(None).await?;
    Ok(got)
}

/// `method path` over a plain socket: the answer's body.
async fn http(addr: SocketAddr, method: &str, path: &str) -> anyhow::Result<String> {
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    tokio::task::spawn_blocking(move || {
        let mut s = std::net::TcpStream::connect(addr)?;
        s.write_all(request.as_bytes())?;
        let mut answer = String::new();
        s.read_to_string(&mut answer)?;
        let (head, body) = answer.split_once("\r\n\r\n").unwrap_or((&answer, ""));
        anyhow::ensure!(head.starts_with("HTTP/1.1 2"), "{head}");
        Ok(body.to_string())
    })
    .await?
}
