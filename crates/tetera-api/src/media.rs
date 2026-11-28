//! Tetera's own routes (ADR-0017, ADR-0019), merged in front of the Things': the preview as
//! MJPEG, a session of its own for each viewer, and the `live-view` page.

use crate::instrument::Instrument;
use crate::preview::{PreviewThing, Slot};
use serde::Deserialize;
use std::sync::Arc;
use teta_wot::http::ShutdownToken;
use teta_wot::http::axum::Router;
use teta_wot::http::axum::body::{Body, Bytes};
use teta_wot::http::axum::extract::rejection::QueryRejection;
use teta_wot::http::axum::extract::{Query, State};
use teta_wot::http::axum::http::{StatusCode, header};
use teta_wot::http::axum::response::{Html, IntoResponse, Response};
use teta_wot::http::axum::routing::get;
use tetera_acquire::{PreviewParams, Session};

/// The MJPEG stream's path.
pub const MJPEG: &str = "/api/v1/preview/mjpeg";

/// The live-view page's path.
pub const LIVE_VIEW: &str = "/live-view";

/// The parts' boundary.
pub const BOUNDARY: &str = "tetera-frame";

/// The page: plain HTML and JavaScript, no framework.
const PAGE: &str = include_str!("live-view.html");

/// What the routes reach.
#[derive(Clone)]
struct Media {
    instrument: Arc<Instrument>,
    preview: Arc<PreviewThing>,
}

/// What an MJPEG viewer asks for in its query; the rest is the preview's defaults.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct MjpegQuery {
    width: Option<u32>,
    height: Option<u32>,
    quality: Option<u8>,
    max_rate: Option<f64>,
}

/// The routes, for `ThingServerBuilder::merge`.
pub(crate) fn routes(instrument: Arc<Instrument>, preview: Arc<PreviewThing>) -> Router {
    Router::new()
        .route(MJPEG, get(mjpeg))
        .route(LIVE_VIEW, get(|| async { Html(PAGE) }))
        .with_state(Media {
            instrument,
            preview,
        })
}

/// A session that stops off the async threads when its viewer goes.
struct Running(Option<Session>);

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(session) = self.0.take() {
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => drop(handle.spawn_blocking(move || session.stop())),
                Err(_) => session.stop(),
            }
        }
    }
}

/// The preview as `multipart/x-mixed-replace`, each part a JPEG, as fast as the viewer takes
/// them, up to its rate: a session of its own, which ends when the viewer goes or the server
/// shuts down.
async fn mjpeg(
    State(media): State<Media>,
    shutdown: ShutdownToken,
    query: Result<Query<MjpegQuery>, QueryRejection>,
) -> Response {
    let Ok(Query(query)) = query else {
        return (
            StatusCode::BAD_REQUEST,
            "the query takes width, height, quality and max_rate",
        )
            .into_response();
    };
    let defaults = media.preview.defaults();
    let params = PreviewParams {
        width: query.width.unwrap_or(defaults.width),
        height: query.height.unwrap_or(defaults.height),
        quality: query.quality.unwrap_or(defaults.quality),
        max_rate: query.max_rate.unwrap_or(defaults.max_rate),
        ..defaults
    };
    let slot = Arc::new(Slot::default());
    let into = slot.clone();
    let session = match Session::start(
        &media.instrument.bus,
        media.instrument.previews.clone(),
        params,
        move |p| into.put(p),
    ) {
        Ok(session) => Running(Some(session)),
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    let parts = futures_util::stream::unfold(
        (slot, session, shutdown),
        |(slot, session, shutdown)| async move {
            loop {
                if shutdown.is_shutting_down() {
                    return None;
                }
                if let Some(p) = slot.take() {
                    let part = Ok::<_, std::convert::Infallible>(part(&p.jpeg));
                    return Some((part, (slot, session, shutdown)));
                }
                tokio::select! {
                    () = slot.ready.notified() => {}
                    () = shutdown.cancelled() => return None,
                }
            }
        },
    );
    (
        [
            (
                header::CONTENT_TYPE,
                format!("multipart/x-mixed-replace; boundary={BOUNDARY}"),
            ),
            (header::CACHE_CONTROL, "no-store".to_string()),
        ],
        Body::from_stream(parts),
    )
        .into_response()
}

/// One part: its boundary, its headers, the JPEG, and a line's end.
fn part(jpeg: &[u8]) -> Bytes {
    let mut b = format!(
        "--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
        jpeg.len()
    )
    .into_bytes();
    b.extend_from_slice(jpeg);
    b.extend_from_slice(b"\r\n");
    Bytes::from(b)
}
