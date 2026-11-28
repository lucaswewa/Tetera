# ADR-0017: Control plane and media plane

- Status: Accepted
- Date: 2026-11-28
- Phase: P1.13a

## Context

- **Two kinds of traffic cross the server.** Control: small JSON, a property read, a move, a capture, a few a second. Media: a preview's frames at the camera's rate, deep-zoom tiles, volume bricks, downloads of gigabytes, with flow control and ranges.
- **teta-wot serves Things** with their Thing Descriptions, an OpenAPI document, observation over WebSocket and server-sent events, and interactive docs. The OpenAPI document becomes the web app's TypeScript types in M4 (plan §6.4).
- **teta-wot also has MJPEG and binary WebSocket streams,** but a preview needs more of its own: the client's viewport and ROI, acknowledgements that pace the sender, latest-wins frames (P1.14b).
- **LabThings' clients,** and teta-wot's own, expect a Thing at `/{thing}/`, the docs at `/docs` and the OpenAPI document at `/openapi.json`.
- **Devices block,** and some must stay on one thread; Thing code runs on tokio.
- **Only `tetera-api` may depend on teta-wot** (ADR-0004).

## Decision

1. **The control plane is teta-wot Things,** in `tetera-api`: one Thing for each subsystem (plan §6.4), served at the root, `/{thing}/`, with `/docs` and `/openapi.json` as teta-wot serves them.
2. **The media plane is Tetera's own axum routes under `/api/v1/`,** merged into teta-wot's router (`ThingServerBuilder::merge`), so they share its security, layers and shutdown. None is a Thing's name, so they can't clash.
3. **An action that starts long work returns a job's reference at once** (P1.15a). Short actions, a move or loading a sample, end their invocation when they finish.
4. **Devices reach the Things as services.** The server registers the instrument with teta-wot; each Thing finds it when it starts (`ThingCtx::server().service()`), and actions take it as a dependency. A device call that blocks runs off tokio's threads (`spawn_blocking`, or a blocking property).
5. **Tetera's errors become teta-wot's** (`tetera_api::error`):
   - a failure the device foresaw (a bad argument, busy, a fault, a timeout, something missing) ends an invocation as `error`, its message led by its kind, such as `hardware: an alarm stopped the stage`;
   - a cancellation ends it as `cancelled`;
   - a value a device refuses is a 422, as an invalid value is;
   - anything else is logged with its cause, and a property's is a 500.
6. **Things are served under fixed names:** `system` and `simulation` from P1.13a; `camera`, `stage`, `objective` and `illumination` from P1.13b; more as the plan's table says.

## Consequences

- **Gains:**
  - every control is described twice for free, in its Thing Description and the OpenAPI document, from its Rust types;
  - LabThings' Python client and teta-wot's tools work against Tetera;
  - the media plane is shaped for media, without bending the Things to carry it.
- **Costs:**
  - two kinds of routes to secure, document and test; the media plane's aren't in the OpenAPI document unless it adds them;
  - Things at the root take names a later route can't use.
- **To watch:**
  - whether teta-wot's binary streams grow what the preview needs, so a media route could become a Thing's stream (an upstream U-phase, ADR-0004);
  - how long a move's invocation may run before it should be a job.

## Alternatives considered

- **Things under `/api/v1/` too** (teta-wot's `api_prefix`): one prefix for everything, but LabThings' clients and teta-wot's tools would need it set, and the docs would move.
- **Media as Things' streams and Blobs:** one plane, but no acknowledgements or viewport, and a download of gigabytes would go through an action's output.
- **gRPC or a WebSocket protocol of Tetera's own for control:** typed, but no Thing Descriptions, no browser without a proxy, and teta-wot is the spec's interface.
