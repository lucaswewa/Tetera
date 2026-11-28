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
  | `capture` | `last`, the last capture's result: its dataset, the mode switch's times there and back, and the frames a burst lost | `capture` (`project`, `name`, `kind`, and `mode`, `bit_depth`, `exposure_ms` if not the defaults): queued as a hardware job, which it answers with |
  | `autofocus` | `last`, the last search's result: the focus or why there is none, the measure, and the curve | `autofocus` (`measure`, `options`, each optional): queued as a hardware job, which it answers with; a blank field fails it, the stage back where it was |
  | `datasets` | `projects` | `create_project` (`name`), `list` (`project`, the complete datasets, the newest first), `get` (`id`: the dataset, its files, its metadata and its thumbnail's path), `delete` (`id`) |
  | `preview` | `defaults`, what a client's preview starts from (the viewport, region, quality and rate), a setting; `sessions`, the previews open; `statistics`, the live statistics of the camera's latest frame, up to 10 a second | —; its `stream` is the live preview's WebSocket |

  teta-wot takes a `null` body as a missing one, as FastAPI does, so a property can't be written to `null`: an action does that, as `remove` does for the objective.
- **`error`** maps Tetera's errors to teta-wot's.
  - **An action:** an anticipated failure ends its invocation as `error`, with a message led by its kind, such as `invalid_argument: …`. A cancellation ends it `cancelled`.
  - **A property:** a value a device refuses is a 422; anything else is a 500.
- **`media`** has Tetera's own routes, merged in front of the Things' (ADR-0017):
  - `/api/v1/preview/mjpeg`, the preview as MJPEG, for an `<img>` or any tool that plays it, a session for each viewer: `?width=&height=&quality=&max_rate=`, the rest from `preview/defaults`;
  - `/live-view`, a plain page: the live image over the stream, its rate and its time from exposure to paint, a histogram, the camera's start and stop, and buttons that jog the stage;
  - `/api/v1/datasets/{id}/thumbnail.jpg`, a dataset's thumbnail: a 404 for one that isn't there, a 409 for one being written.
- **`jobs`** is the `jobs` Thing over the job service (`tetera-jobs`, ADR-0020), and `JobsOptions`, the server's `[jobs]`.
- **`autofocus`** runs `tetera-workflows`' autofocus as a hardware job, its curve logged, and the stream started through the `camera` Thing's sink if it wasn't running.
- **`capture`** runs `tetera-workflows`' capture as a hardware job. Its step is the job's progress and cancel check, and the live view restarts through the `camera` Thing's sink, so `camera/frames` goes on counting.
- **`server::build`** puts the Things on a teta-wot server, with its services: the instrument, the data roots, the job service and the dataset store. `build_with` takes the job service's options. The jobs' history and the datasets' index share the data root's `tetera.db`, and the compute jobs share the GPU's budgets with the camera. `Service::serve` serves them until a shutdown signal, then stops gracefully. The OpenAPI document is at `/openapi.json` and the interactive docs at `/docs`.

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

## Captures and datasets

A burst of 20 full frames into a new project, followed as a job:

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"name": "Tonsil 3"}' http://127.0.0.1:8310/datasets/create_project
curl -X POST http://127.0.0.1:8310/camera/start_streaming
curl -X POST -H 'Content-Type: application/json' \
  -d '{"project": "Tonsil 3", "name": "a burst", "kind": {"kind": "burst", "count": 20}}' \
  http://127.0.0.1:8310/capture/capture                        # its invocation, whose output is the job
curl -N http://127.0.0.1:8310/jobs/changed -H 'Accept: text/event-stream'   # the job as it goes
curl http://127.0.0.1:8310/capture/last                         # the dataset, and the switches' times
```

`{"kind": "single"}` takes one frame, and `{"kind": "series", "count": 10, "interval_ms": 1000}` ten, a second apart. `jobs/cancel` with the job's `id` stops a capture and leaves nothing behind. `datasets/list` with `{"project": "Tonsil 3"}` lists the datasets, `datasets/get` with a dataset's `id` reads one back, and its thumbnail is at `/api/v1/datasets/{id}/thumbnail.jpg`.

```bash
cargo run --release -p tetera-api --features cuda --example capture
```

It serves the virtual instrument on a loopback port, streams the camera in its binned live mode, and takes a burst of 20 full frames into a new project. It prints the job as it goes, the dataset as `datasets/get` has it, its tree on disk, and its frames read back. `-- --smoke` takes 5 frames of a tiny sensor on the CPU, and `-- --keep` keeps the data root.

## Autofocus

```bash
curl -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8310/autofocus/autofocus
curl -X POST -H 'Content-Type: application/json' -d '{"measure": "tenengrad"}' http://127.0.0.1:8310/autofocus/autofocus
curl http://127.0.0.1:8310/autofocus/last          # the focus, and the curve
```

```bash
cargo run --release -p tetera-api --features cuda --example autofocus
```

It starts the virtual instrument's live stream, then focuses 100 times at 5× and 100 at 50× from random starts within ±6 depths of field of the section's middle, the simulator's true focus, and prints each set's errors. Then it times a search at 10×, and focuses on a blank field, which finds none and leaves the stage where it was. `-- --compare` runs the other three measures too, and `-- --smoke` a single start on a tiny sensor on the CPU.

## Scans

A 3 × 3 snake at the objective fitted, planned, then scanned stop and go as a job, paused and resumed:

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"pattern": {"kind": "snake", "x0": 2000, "y0": 2000, "x1": 4500, "y1": 3700, "overlap": 0.1}}' \
  http://127.0.0.1:8310/scan/preview                            # the plan: positions, grid, estimated time
curl -X POST -H 'Content-Type: application/json' \
  -d '{"project": "Tonsil 3", "name": "a snake", "pattern": {"kind": "snake", "x0": 2000, "y0": 2000, "x1": 4500, "y1": 3700, "overlap": 0.1}}' \
  http://127.0.0.1:8310/scan/scan                              # its invocation, whose output is the job
curl -N http://127.0.0.1:8310/scan/progress -H 'Accept: text/event-stream'   # each tile as it goes
curl -X POST http://127.0.0.1:8310/scan/pause                   # held at its next tile
curl -X POST http://127.0.0.1:8310/scan/resume
curl http://127.0.0.1:8310/scan/last                            # the dataset, each tile's positions, the time
```

`{"kind": "z_around", "planes": 21, "step": 1.0}` scans 21 planes around the current z into a z-stack, and `{"kind": "z_range", "from": 990, "to": 1010, "step": 0.5}` a range. `autofocus` takes `{"at": "first"}` or `{"at": "every", "tiles": 5}`; `settle_ms`, `exposure_ms` and `write_mb_s` change the settle, the exposure and the disk's throughput the estimate assumes. `scan/running` says how far the running scan has got, and `jobs/cancel` stops it, keeping the tiles taken.

```bash
cargo run --release -p tetera-api --features cuda --example scan
```

It plans a 5 × 5 snake of full-resolution tiles at 10× on the virtual instrument, prints the plan and its estimated time, scans it stop and go, and prints the time taken against the estimate and each tile's recorded position against the plan and the simulator's truth. Then a 21-plane z-scan, read back as one 3D array; and a 3 × 3 snake through the `scan` Thing on a loopback port, paused at its fourth tile for a second and resumed. `-- --smoke` does the same, smaller, with a tiny sensor on the CPU.

## The live view

With the server running, open `http://localhost:8310/live-view` in a browser on Windows and press *Start*. The page shows the camera live, with its rate and its time from each frame's exposure to the page's paint, and a histogram of the frame. Hold an arrow to jog the stage. The MJPEG stream plays on its own at `http://localhost:8310/api/v1/preview/mjpeg?width=960&height=640`.

The tests run each Thing through teta-wot's in-process test client, and the preview's stream over a real socket. A capture is taken, read back, deleted, and cancelled through the `jobs` Thing: `cargo test -p tetera-api`.
