//! The preview's stream over a real socket (ADR-0019): frames and their headers, the
//! acknowledgement window, a slow client, the client's state, and the sessions open.

use crate::instrument::{Instrument, ObjectivesConfig, SensorSize, SimulatorConfig};
use crate::server::{Roots, build};
use crate::tests::Scratch;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tetera_acquire::protocol::Header;
use tetera_core::frame::format::BayerPattern;
use tetera_core::frame::pool::HostAllocator;
use tetera_core::{Clock, FrameLayout, FramePool, PixelFormat};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;

type Socket = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>;

/// The server for a test instrument on a loopback port, until it drops.
struct Served {
    addr: SocketAddr,
    instrument: Arc<Instrument>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    _scratch: Scratch,
}

impl Served {
    async fn start() -> Self {
        let scratch = Scratch::new();
        let roots = Roots::make(scratch.0.join("data"), scratch.0.join("scratch")).unwrap();
        // On a real-time clock, so the frames' times in UTC are the wall clock's.
        let instrument = Arc::new(
            Instrument::new(
                &SimulatorConfig {
                    gpu: false,
                    sensor: SensorSize::Tiny,
                    ..SimulatorConfig::default()
                },
                &ObjectivesConfig::default(),
            )
            .unwrap(),
        );
        let service = build(instrument.clone(), roots).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(service.serve(listener, async move {
            let _ = stopped.await;
        }));
        Self {
            addr,
            instrument,
            stop: Some(stop),
            _scratch: scratch,
        }
    }

    async fn socket(&self) -> Socket {
        let url = format!("ws://{}/preview/stream", self.addr);
        tokio_tungstenite::connect_async(url).await.unwrap().0
    }

    /// `GET path`'s JSON, over a plain socket.
    async fn get(&self, path: &str) -> Value {
        self.http("GET", path, "").await
    }

    /// `PUT path` with `body`, over a plain socket: its answer's status line.
    async fn put(&self, path: &str, body: &Value) -> String {
        let answer = self.raw("PUT", path, &body.to_string()).await;
        answer.lines().next().unwrap_or_default().to_string()
    }

    async fn http(&self, method: &str, path: &str, body: &str) -> Value {
        let answer = self.raw(method, path, body).await;
        let body = answer.split_once("\r\n\r\n").expect("a body").1;
        serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {answer}"))
    }

    async fn raw(&self, method: &str, path: &str, body: &str) -> String {
        use std::io::{Read, Write};
        let addr = self.addr;
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        tokio::task::spawn_blocking(move || {
            let mut s = std::net::TcpStream::connect(addr).unwrap();
            s.write_all(request.as_bytes()).unwrap();
            let mut answer = String::new();
            s.read_to_string(&mut answer).unwrap();
            answer
        })
        .await
        .unwrap()
    }
}

impl Drop for Served {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}

/// Publishes frames into the instrument's bus every `every`, each exposed now, until it
/// drops: the last frame's number.
struct Publisher {
    last: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Publisher {
    fn start(instrument: &Arc<Instrument>, every: Duration) -> Self {
        let (last, stop) = (
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        let (bus, clock) = (instrument.bus.clone(), instrument.clock.clone());
        let (seen, stopping) = (last.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            let format = PixelFormat::Bayer8 {
                pattern: BayerPattern::Rggb,
            };
            let pool = FramePool::new(
                FrameLayout::new(format, 256, 192).unwrap(),
                8,
                &HostAllocator,
            )
            .unwrap();
            let mut seq = 0;
            while !stopping.load(Ordering::Relaxed) {
                seq += 1;
                let mut f = pool.take(Duration::from_secs(5)).unwrap();
                f.data_mut().fill(90);
                let now = clock.now();
                let meta = f.meta_mut();
                meta.seq = seq;
                meta.exposure_start = now;
                meta.exposure_end = now;
                bus.publish(f.freeze());
                seen.store(seq, Ordering::Relaxed);
                std::thread::sleep(every);
            }
        });
        Self {
            last,
            stop,
            thread: Some(thread),
        }
    }

    fn last(&self) -> u64 {
        self.last.load(Ordering::Relaxed)
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The next frame within `within`, skipping text.
async fn frame(socket: &mut Socket, within: Duration) -> Option<(Header, Vec<u8>)> {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        match tokio::time::timeout_at(deadline, socket.next())
            .await
            .ok()??
        {
            Ok(Message::Binary(b)) => {
                let (h, jpeg) = Header::parse(&b).unwrap();
                return Some((h, jpeg.to_vec()));
            }
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        }
    }
}

/// The next text message within `within`, skipping frames.
async fn text(socket: &mut Socket, within: Duration) -> Option<Value> {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        match tokio::time::timeout_at(deadline, socket.next())
            .await
            .ok()??
        {
            Ok(Message::Text(t)) => return serde_json::from_str(t.as_str()).ok(),
            Ok(_) => {}
            Err(e) => panic!("{e}"),
        }
    }
}

async fn send(socket: &mut Socket, state: Value) {
    socket.send(Message::text(state.to_string())).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_client_gets_two_frames_until_it_acknowledges_them() {
    let served = Served::start().await;
    let mut socket = served.socket().await;
    send(
        &mut socket,
        json!({"width": 64, "height": 64, "quality": 70}),
    )
    .await;
    let publisher = Publisher::start(&served.instrument, Duration::from_millis(10));
    // The state applies from the session's next preview: one made before it came takes the
    // defaults, and is acknowledged here.
    let (a, jpeg) = loop {
        let (h, jpeg) = frame(&mut socket, Duration::from_secs(10))
            .await
            .expect("a frame");
        if h.width == 64 {
            break (h, jpeg);
        }
        let ack = json!({"width": 64, "height": 64, "quality": 70, "ack": h.seq});
        send(&mut socket, ack).await;
    };
    let (b, _) = frame(&mut socket, Duration::from_secs(10))
        .await
        .expect("a second");
    assert!(b.seq > a.seq);
    assert_eq!(
        (a.width, a.height, a.quality, a.lowered),
        (64, 48, 70, false)
    );
    assert_eq!(&jpeg[..2], &[0xff, 0xd8]);
    let decoded = tetera_imaging::jpeg::decode_rgb(&jpeg).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (64, 48));
    let clock = &served.instrument.clock;
    let now_us = clock.utc(clock.now()).unix_nanos() / 1_000;
    assert!(a.exposure_end_utc_us <= a.sent_utc_us && a.sent_utc_us <= now_us);
    assert!(
        now_us - a.exposure_end_utc_us < 5_000_000,
        "exposed moments ago"
    );
    assert!(
        frame(&mut socket, Duration::from_millis(300))
            .await
            .is_none(),
        "no third before an acknowledgement"
    );
    send(
        &mut socket,
        json!({"width": 64, "height": 64, "quality": 70, "ack": a.seq}),
    )
    .await;
    let (c, _) = frame(&mut socket, Duration::from_secs(10))
        .await
        .expect("a third");
    assert!(c.seq > b.seq, "newer than the last sent");
    assert!(
        c.latency_us.is_some_and(|us| us >= 300_000),
        "a's latency, acknowledged after 300 ms: {:?}",
        c.latency_us
    );
    assert!(
        frame(&mut socket, Duration::from_millis(300))
            .await
            .is_none(),
        "two in flight again"
    );
    drop(publisher);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_slow_client_never_gets_a_stale_frame_and_its_link_steps_down() {
    let served = Served::start().await;
    let mut socket = served.socket().await;
    let publisher = Publisher::start(&served.instrument, Duration::from_millis(10));
    let started = Instant::now();
    let mut got = Vec::new();
    while started.elapsed() < Duration::from_millis(2_500) {
        let (h, _) = frame(&mut socket, Duration::from_secs(10))
            .await
            .expect("a frame");
        let behind = publisher.last() - h.seq;
        got.push((h, behind));
        // Painting takes this client 150 ms.
        tokio::time::sleep(Duration::from_millis(150)).await;
        send(
            &mut socket,
            json!({"width": 128, "height": 128, "ack": h.seq}),
        )
        .await;
    }
    drop(publisher);
    assert!(got.len() >= 10, "{}", got.len());
    let worst = got.iter().map(|(_, behind)| *behind).max().unwrap();
    let oldest = got
        .iter()
        .map(|(h, _)| h.sent_utc_us - h.exposure_end_utc_us)
        .max()
        .unwrap();
    assert!(
        worst <= 30,
        "a frame 30 behind the newest is stale: {worst}"
    );
    // Each frame was the newest when it was sent: none had waited for the window to open.
    assert!(oldest <= 100_000, "a frame {oldest} µs old when sent");
    let last = got.last().unwrap().0;
    assert!(
        last.lowered && last.quality <= 60,
        "the link stepped down: {last:?}"
    );
    // Each frame after the first carries the last acknowledged one's latency: at least the
    // client's 150 ms of painting, and less than the window's two paints and some.
    assert!(
        got.iter().skip(2).all(|(h, _)| h
            .latency_us
            .is_some_and(|us| (150_000..500_000).contains(&us))),
        "{:?}",
        got.iter().map(|(h, _)| h.latency_us).collect::<Vec<_>>()
    );
    let sessions = served.get("/preview/sessions").await;
    assert_eq!(sessions.as_array().map(Vec::len), Some(1), "{sessions}");
    assert_eq!(sessions[0]["id"], 1, "numbered from 1");
    assert!(sessions[0]["link_step"].as_u64() >= Some(1));
    assert!(sessions[0]["latency"].as_f64().is_some_and(|ms| ms > 100.0));
    drop(socket);
    for _ in 0..100 {
        if served.get("/preview/sessions").await == json!([]) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the session outlived its socket");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_client_that_says_nothing_gets_the_defaults_and_the_cameras_white_balance() {
    let served = Served::start().await;
    let defaults = json!({"width": 96, "height": 96, "quality": 55});
    assert!(
        served
            .put("/preview/defaults", &defaults)
            .await
            .contains(" 20")
    );
    let warm = json!({"red": 2.0, "green": 1.0, "blue": 1.0});
    assert!(
        served
            .put("/camera/white_balance", &warm)
            .await
            .contains(" 20")
    );
    let mut socket = served.socket().await;
    let _publisher = Publisher::start(&served.instrument, Duration::from_millis(10));
    let (h, jpeg) = frame(&mut socket, Duration::from_secs(10)).await.unwrap();
    assert_eq!((h.width, h.height, h.quality), (96, 72, 55));
    let rgb = tetera_imaging::jpeg::decode_rgb(&jpeg).unwrap();
    let sum = |c: usize| -> f64 {
        rgb.data()
            .iter()
            .skip(c)
            .step_by(3)
            .map(|&v| f64::from(v))
            .sum()
    };
    assert!(
        sum(0) > 1.2 * sum(1),
        "red's gain: {} against {}",
        sum(0),
        sum(1)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_clients_state_changes_its_preview_and_a_bad_one_is_answered() {
    let served = Served::start().await;
    let mut socket = served.socket().await;
    let _publisher = Publisher::start(&served.instrument, Duration::from_millis(10));
    let (first, _) = frame(&mut socket, Duration::from_secs(10)).await.unwrap();
    assert_eq!(
        (first.width, first.height),
        (256, 192),
        "the defaults: the whole frame"
    );
    send(
        &mut socket,
        json!({"width": 32, "height": 32, "ack": first.seq}),
    )
    .await;
    let (smaller, seen) = loop {
        let (h, _) = frame(&mut socket, Duration::from_secs(10)).await.unwrap();
        send(
            &mut socket,
            json!({"width": 32, "height": 32, "ack": h.seq}),
        )
        .await;
        if h.width == 32 {
            break (h, h.seq);
        }
    };
    assert_eq!(smaller.height, 24);
    for bad in [
        json!({"width": 32, "height": 32, "quality": 0}),
        json!({"zoom": 2}),
        json!("wide"),
    ] {
        send(&mut socket, bad.clone()).await;
        let answer = text(&mut socket, Duration::from_secs(10)).await;
        assert!(answer.is_some_and(|a| a["error"].is_string()), "{bad}");
    }
    socket.send(Message::text("not JSON")).await.unwrap();
    assert!(
        text(&mut socket, Duration::from_secs(10))
            .await
            .is_some_and(|a| a["error"].is_string())
    );
    send(&mut socket, json!({"width": 32, "height": 32, "ack": seen})).await;
    let (still, _) = frame(&mut socket, Duration::from_secs(10)).await.unwrap();
    assert_eq!(still.width, 32, "the last good state holds");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_mjpeg_stream_ends_with_the_server_at_once() {
    use std::io::{Read, Write};
    let served = Served::start().await;
    let _publisher = Publisher::start(&served.instrument, Duration::from_millis(10));
    let addr = served.addr;
    let (part_seen, seen) = tokio::sync::oneshot::channel();
    let reader = tokio::task::spawn_blocking(move || {
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        s.write_all(b"GET /api/v1/preview/mjpeg?width=64&height=64 HTTP/1.1\r\nHost: test\r\n\r\n")
            .unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut got = Vec::new();
        let mut buf = [0u8; 65_536];
        while !got.windows(14).any(|w| w == b"--tetera-frame") {
            let n = s.read(&mut buf).expect("a part within 10 s");
            assert!(n > 0, "the stream ended early");
            got.extend_from_slice(&buf[..n]);
        }
        let _ = part_seen.send(());
        // Then to its end, which a stream that ignores the shutdown never reaches: its parts
        // keep coming.
        let deadline = Instant::now() + Duration::from_secs(10);
        while s.read(&mut buf).expect("the end within 10 s") > 0 {
            assert!(Instant::now() < deadline, "the stream outlived the server");
        }
        Instant::now()
    });
    seen.await.unwrap();
    let stopping = Instant::now();
    drop(served);
    let ended = reader.await.unwrap();
    assert!(
        ended.duration_since(stopping) < Duration::from_secs(2),
        "the stream ended {:?} after the server began to stop, not after its 5 s grace",
        ended.duration_since(stopping)
    );
}
