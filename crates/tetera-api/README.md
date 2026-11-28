# tetera-api

Tetera's HTTP interface, L5 (plan §6.2, [ADR-0017](../../docs/adr/0017-control-plane-and-media-plane.md)): the instrument as [teta-wot](https://github.com/lucaswewa/teta-wot) v0.2.0 Things. It is the only crate that depends on teta-wot ([ADR-0004](../../docs/adr/0004-teta-wot-v0-2-0-extended-upstream-first.md)).

- **`Instrument`** is the virtual instrument, assembled from `configs/tetera.toml`'s `[simulator]` and `[objectives]` tables (`SimulatorConfig`, `ObjectivesConfig`). It holds:
  - the simulated clock;
  - the virtual stage;
  - the virtual camera, on the GPU when there is one;
  - the objective mount;
  - the light, on at the start;
  - the sample on the stage.

  `load` and `unload` change the sample while the camera runs.
- **The Things,** each with its Thing Description at `/{thing}/`:

  | Thing | Properties | Actions |
  |---|---|---|
  | `system` | `version`; `platform` (the kernel, the distribution, WSL's version); `compute` (the GPU or the CPU, and the camera's sensor); `memory`; `disks` (the data and scratch roots') | — |
  | `simulation` | `sample` (its seed and section, or `null`) | `load` (`seed`, `thickness`, `folds`), `unload`, `inject` (`stage_stall`, `stage_alarm`, `stage_limit`) |

  The device Things, `camera`, `stage`, `objective` and `illumination`, join in P1.13b.
- **`error`** maps Tetera's errors to teta-wot's.
  - **An action:** an anticipated failure ends its invocation as `error`, with a message led by its kind, such as `invalid_argument: …`. A cancellation ends it `cancelled`.
  - **A property:** a value a device refuses is a 422; anything else is a 500.
- **`server::build`** puts the Things on a teta-wot server, with the instrument and the data roots as its services. `Service::serve` serves them until a shutdown signal, then stops gracefully. The OpenAPI document is at `/openapi.json` and the interactive docs at `/docs`.

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

The tests run each Thing through teta-wot's in-process test client: `cargo test -p tetera-api`.
