# tetera-api

Tetera's HTTP interface, L5 (plan §6.2, [ADR-0017](../../docs/adr/0017-control-plane-and-media-plane.md)): the instrument as [teta-wot](https://github.com/lucaswewa/teta-wot) v0.2.0 Things. It is the only crate that depends on teta-wot ([ADR-0004](../../docs/adr/0004-teta-wot-v0-2-0-extended-upstream-first.md)).

- **`Instrument`** is the virtual instrument, assembled from `configs/tetera.toml`'s `[simulator]` and `[objectives]` tables (`SimulatorConfig`, `ObjectivesConfig`). It holds:
  - the simulated clock;
  - the virtual stage;
  - the virtual camera, on the GPU when there is one;
  - the objective mount;
  - the light, on at the start;
  - the sample on the stage;
  - the frame bus, which the camera's stream goes into, and where previews are made, on the GPU when the camera renders there.

  `load` and `unload` change the sample while the camera runs.
- **The Things,** each with its Thing Description at `/{thing}/`:

  | Thing | Properties | Actions |
  |---|---|---|
  | `system` | `version`; `platform` (the kernel, the distribution, WSL's version); `compute` (the GPU or the CPU, and the camera's sensor); `memory`; `disks` (the data and scratch roots') | — |
  | `simulation` | `sample` (its seed and section, or `null`) | `load` (`seed`, `thickness`, `folds`), `unload`, `inject` (`stage_stall`, `stage_alarm`, `stage_limit`) |
  | `camera` | `capabilities`; `config`, and each of its settings: `mode`, `exposure` (ms), `gain`, `bit_depth`, `binning`, `roi`; `pixel_format`; `white_balance`, for the preview; `streaming`, `frames` (delivered and dropped), `last_frame` (its metadata) | `start_streaming`, `stop_streaming`: the stream goes into the frame bus |
  | `stage` | `capabilities`; `position` (µm), `state`, `moving`, kept current as it moves, so they can be watched; `speed`, the speed of moves that name none, a setting | `move_to`, `move_by` (each ends when the stage is still, and stops it if cancelled), `jog`, `stop_jog`, `halt`, `home` |
  | `objective` | `catalogue`; `fitted`, by name, written to say which is fitted; `objective`, in full | `remove`, to say none is |
  | `illumination` | `capabilities`; `on`, `intensity` (0 to 1), `colour_temperature` (K) | — |
  | `jobs` | `active`, the jobs queued and running; the `changed` event, each change to a job | `get` (`id`), `list` (`limit`, the newest first), `cancel` (`id`) |
  | `preview` | `defaults`, what a client's preview starts from (the viewport, region, quality and rate), a setting; `sessions`, the previews open; `statistics`, the live statistics of the camera's latest frame, up to 10 a second | —; its `stream` is the live preview's WebSocket |

  teta-wot takes a `null` body as a missing one, as FastAPI does, so a property can't be written to `null`: an action does that, as `remove` does for the objective.
- **`error`** maps Tetera's errors to teta-wot's.
  - **An action:** an anticipated failure ends its invocation as `error`, with a message led by its kind, such as `invalid_argument: …`. A cancellation ends it `cancelled`.
  - **A property:** a value a device refuses is a 422; anything else is a 500.
- **`media`** has Tetera's own routes, merged in front of the Things' (ADR-0017):
  - `/api/v1/preview/mjpeg`, the preview as MJPEG, for an `<img>` or any tool that plays it, a session for each viewer: `?width=&height=&quality=&max_rate=`, the rest from `preview/defaults`;
  - `/live-view`, a plain page: the live image over the stream, its rate and its time from exposure to paint, a histogram, the camera's start and stop, and buttons that jog the stage.
- **`jobs`** is the `jobs` Thing over the job service (`tetera-jobs`, ADR-0020), and `JobsOptions`, the server's `[jobs]`.
- **`server::build`** puts the Things on a teta-wot server, with the instrument, the data roots and the job service as its services. `build_with` takes the job service's options: its history is the data root's `tetera.db`, and its compute jobs share the GPU's budgets with the camera. `Service::serve` serves them until a shutdown signal, then stops gracefully. The OpenAPI document is at `/openapi.json` and the interactive docs at `/docs`.

## How to try it

Run the server (`tetera-server`'s README), then:

```bash
curl http://127.0.0.1:8310/system/              # system's Thing Description
curl http://127.0.0.1:8310/system/platform      # {"os":"linux","kernel":"…-microsoft-standard-WSL2",…}
curl http://127.0.0.1:8310/system/compute       # the GPU, or the CPU and why
curl http://127.0.0.1:8310/system/disks
curl http://127.0.0.1:8310/simulation/sample    # {"seed":7,"thickness":4.0,"folds":false}
curl -X POST -H 'Content-Type: application/json' -d '{"seed": 8}' http://127.0.0.1:8310/simulation/load
curl -X POST http://127.0.0.1:8310/simulation/unload
curl -X POST -H 'Content-Type: application/json' -d '{"imperfection": "stage_alarm"}' http://127.0.0.1:8310/simulation/inject
```

An action's `POST` answers `201 Created` at once, with its invocation's URL in `Location`; `curl` that URL to follow it to `completed`.

The devices:

```bash
curl http://127.0.0.1:8310/camera/config
curl -X PUT -H 'Content-Type: application/json' -d '20' http://127.0.0.1:8310/camera/exposure   # 20 ms
curl -X PUT -H 'Content-Type: application/json' -d '2' http://127.0.0.1:8310/camera/mode        # binned 3 × 3
curl -X POST http://127.0.0.1:8310/camera/start_streaming
curl http://127.0.0.1:8310/camera/frames                       # {"delivered":…,"dropped":0}
curl http://127.0.0.1:8310/camera/last_frame                   # its number, times, where the stage was
curl -X POST http://127.0.0.1:8310/camera/stop_streaming

curl http://127.0.0.1:8310/objective/catalogue
curl -X PUT -H 'Content-Type: application/json' -d '"50x"' http://127.0.0.1:8310/objective/fitted

curl -X PUT -H 'Content-Type: application/json' -d '0.5' http://127.0.0.1:8310/illumination/intensity
curl -X PUT -H 'Content-Type: application/json' -d '5000' http://127.0.0.1:8310/illumination/colour_temperature
```

A move, and its time: 10 mm along x at 20 mm/s takes 0.6 s, with 0.1 s each to speed up and slow down at 200 mm/s²:

```bash
curl http://127.0.0.1:8310/stage/position                      # {"x":2000.0,"y":2000.0,"z":1002.0}
href=$(curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"x": 12000, "y": 2000, "z": 1002, "speed": 20000}' \
  http://127.0.0.1:8310/stage/move_to | python3 -c 'import json,sys; print(json.load(sys.stdin)["href"])')
sleep 1; curl -s $href                                          # "status":"completed", and its times
curl -N http://127.0.0.1:8310/stage/position -H 'Accept: text/event-stream'   # watch it, in another shell
```

The invocation's `timeStarted` and `timeCompleted` are about 0.6 s apart, plus up to 25 ms for the stage's next report. Over WebSocket, `/stage/ws` with `{"messageType": "addPropertyObservation", "data": {"position": true}}` sends the position as it changes.

## The preview stream

`ws://127.0.0.1:8310/preview/stream` is a WebSocket of frames ([ADR-0019](../../docs/adr/0019-the-preview-protocol.md)). Each binary message is a frame: a 72-byte header, little-endian, then a JPEG. The header carries:

- the frame's sequence number;
- its exposure's start and end on the instrument's clock, and its end in UTC;
- when it was sent, in UTC;
- the image's size, its quality, and whether a slow link lowered it;
- the last acknowledged frame's latency: from its exposure's end to its acknowledgement, on the server's clock.

The client sends its whole state as JSON text, `{"width", "height", "roi", "quality", "max_rate", "ack"}`, when it changes and once it has painted each frame:

- a field left out takes `preview/defaults`;
- `ack` is the last frame painted, by its sequence number, and no more than two frames go unacknowledged;
- a state no preview can take is answered with `{"error": …}` and changes nothing.

When the window holds previews back for more than half of a second, the link steps down: the quality capped at 60, then 40, then the viewport scaled to 0.75 and 0.5. It steps back up after two quick seconds.

```bash
cargo run --release -p tetera-api --features cuda --example preview-stream
```

It serves the virtual instrument on a loopback port, streams the camera in its binned live mode, and connects two clients: a quick one, and one that takes 200 ms over each frame. On this PC the quick one gets the camera's 51 frames a second, 34 ms from each frame's exposure to its receipt. The slow one gets five a second, its link down to quality 40. `-- --smoke` runs a small sensor on the CPU.

## The live view

With the server running, open `http://localhost:8310/live-view` in a browser on Windows and press *Start*. The page shows the camera live, with its rate and its time from each frame's exposure to the page's paint, and a histogram of the frame. Hold an arrow to jog the stage. The MJPEG stream plays on its own at `http://localhost:8310/api/v1/preview/mjpeg?width=960&height=640`.

The tests run each Thing through teta-wot's in-process test client, and the preview's stream over a real socket: `cargo test -p tetera-api`.
