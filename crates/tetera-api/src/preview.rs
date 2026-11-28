//! The `preview` Thing (ADR-0019): the defaults a client's preview starts from, the sessions
//! open, and the stream: a WebSocket of frames for each client, each a header and a JPEG,
//! paced by the client's acknowledgements and lowered on a slow link.

use crate::camera::CameraThing;
use crate::instrument::Instrument;
use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Instant;
use teta_wot::http::Subscriber;
use teta_wot::prelude::*;
use tetera_acquire::protocol::{AckWindow, ClientState, Header, LinkControl};
use tetera_acquire::{Preview, PreviewParams, Session, SessionStats};
use tetera_core::Clock;
use tokio::sync::Notify;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// An open session, as `sessions` shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SessionInfo {
    /// Its number, from 1 since the server started.
    pub id: u64,
    /// What it makes.
    pub params: PreviewParams,
    /// The slow link's step: 0 at its best, up to 4 (ADR-0019).
    pub link_step: usize,
    /// What its session has done.
    pub stats: SessionStats,
    /// Frames sent.
    pub sent: u64,
    /// Frames the socket hadn't taken yet, replaced by newer ones.
    pub dropped: u64,
    /// Frames sent and not yet acknowledged.
    pub in_flight: usize,
    /// From the last acknowledged frame's exposure's end to its acknowledgement, ms: an
    /// upper bound on the time to the client's paint.
    pub latency: Option<f64>,
}

/// The live preview.
#[derive(Thing)]
pub struct PreviewThing {
    /// What a client's preview starts from, and takes for what its state leaves out: the
    /// viewport, the region, the quality and the most frames a second. The white balance is
    /// the camera's.
    #[setting(default = PreviewParams::default())]
    defaults: Prop<PreviewParams>,
    instrument: OnceLock<Arc<Instrument>>,
    camera: OnceLock<Arc<CameraThing>>,
    open: Mutex<BTreeMap<u64, SessionInfo>>,
    next: AtomicU64,
}

/// The newest preview not yet sent, between a session's thread and its socket.
#[derive(Default)]
struct Slot {
    preview: Mutex<Option<Preview>>,
    ready: Notify,
    /// Previews replaced before they were sent: held back by the window.
    held: AtomicU64,
}

impl Slot {
    fn put(&self, preview: Preview) {
        if lock(&self.preview).replace(preview).is_some() {
            self.held.fetch_add(1, Ordering::Relaxed);
        }
        self.ready.notify_one();
    }

    fn take(&self) -> Option<Preview> {
        lock(&self.preview).take()
    }
}

impl PreviewThing {
    /// The defaults, with the camera's white balance.
    fn defaults(&self) -> PreviewParams {
        let mut defaults = self.defaults.get();
        if let Some(camera) = self.camera.get() {
            defaults.white_balance = camera.gains();
        }
        defaults
    }

    /// One client's stream, until it leaves or the server stops.
    async fn serve(&self, subscriber: &mut Subscriber<ClientState>) -> anyhow::Result<()> {
        let instrument = self.instrument.get().context("not started")?.clone();
        let wanted = |state: &ClientState| {
            let defaults = self.defaults();
            state.params(&defaults).unwrap_or(defaults)
        };
        let slot = Arc::new(Slot::default());
        let into = slot.clone();
        let session = Session::start(
            &instrument.bus,
            instrument.previews.clone(),
            wanted(&subscriber.params()),
            move |p| into.put(p),
        )?;
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let clock = instrument.camera.clock().clone();
        let mut window = AckWindow::default();
        let mut link = LinkControl::new(Instant::now());
        let mut held = 0;
        let mut exposures = BTreeMap::new();
        let mut latency = None;
        let result = loop {
            tokio::select! {
                changed = subscriber.changed() => {
                    if !changed {
                        break Ok(());
                    }
                    let state = subscriber.params();
                    if let Some(seq) = state.ack {
                        window.ack(seq, Instant::now());
                        if let Some(end) = exposures.remove(&seq) {
                            let since = clock.now().saturating_since(end);
                            latency = Some(since.as_secs_f64() * 1e3);
                        }
                        exposures.retain(|s, _| *s > seq);
                    }
                }
                () = slot.ready.notified() => {}
            }
            let params = wanted(&subscriber.params());
            if params != session.params()
                && let Err(e) = session.set_params(params)
            {
                break Err(e.into());
            }
            let now = Instant::now();
            let replaced = slot.held.load(Ordering::Relaxed);
            for _ in held..replaced {
                link.record(now, true);
            }
            held = replaced;
            if window.open()
                && let Some(p) = slot.take()
                && window.is_new(p.meta.seq)
            {
                // Both times through the clock's one mapping to UTC (ADR-0006), so their
                // difference is the monotonic clock's, whatever the wall clock does.
                let end = clock.utc(p.meta.exposure_end);
                let header = Header {
                    seq: p.meta.seq,
                    exposure_start: p.meta.exposure_start,
                    exposure_end: p.meta.exposure_end,
                    exposure_end_utc_us: end.unix_nanos() / 1_000,
                    sent_utc_us: clock.utc(clock.now()).unix_nanos() / 1_000,
                    width: p.width,
                    height: p.height,
                    quality: p.quality,
                    lowered: link.step() > 0,
                };
                if !subscriber.send(header.message(&p.jpeg)) {
                    break Ok(());
                }
                window.sent(p.meta.seq, now);
                exposures.insert(p.meta.seq, p.meta.exposure_end);
                link.record(now, false);
            }
            if let Err(e) = session.set_link(link.link()) {
                break Err(e.into());
            }
            lock(&self.open).insert(
                id,
                SessionInfo {
                    id,
                    params: session.params(),
                    link_step: link.step(),
                    stats: session.stats(),
                    sent: subscriber.sent(),
                    dropped: subscriber.dropped(),
                    in_flight: window.in_flight(),
                    latency,
                },
            );
        };
        lock(&self.open).remove(&id);
        tokio::task::spawn_blocking(move || session.stop()).await?;
        result
    }
}

#[thing_impl]
impl PreviewThing {
    /// Finds the instrument and the camera Thing.
    #[on_start]
    async fn start(&self, ctx: ThingCtx) -> anyhow::Result<()> {
        let server = ctx.server();
        let instrument = server
            .service::<Instrument>()
            .context("the server has no instrument")?;
        let _ = self.instrument.set(instrument);
        let camera = server
            .thing_ref::<CameraThing>("camera")
            .context("the server has no camera Thing")?;
        let _ = self.camera.set(Arc::clone(ThingRef::thing(&camera)));
        Ok(())
    }

    /// The previews open: each one's parameters and link, the frames sent, dropped and in
    /// flight, and its latency.
    #[property]
    async fn sessions(&self) -> Result<Vec<SessionInfo>, PropertyError> {
        Ok(lock(&self.open).values().cloned().collect())
    }

    /// The live preview: a WebSocket of frames, each a 64-byte header and a JPEG (ADR-0019).
    /// The client sends its whole state as JSON, `{"width", "height", "roi", "quality",
    /// "max_rate", "ack"}`, when it changes and once it has painted each frame, by its
    /// sequence number; no more than two frames go unacknowledged.
    #[binary_stream]
    async fn stream(&self, mut subscriber: Subscriber<ClientState>) {
        if let Err(e) = self.serve(&mut subscriber).await {
            tracing::warn!("a preview stream failed: {e:#}");
        }
    }
}

#[cfg(test)]
mod tests;
