# tetera-hal

Tetera's hardware abstraction layer, L2 (plan §6.8, [ADR-0013](../../docs/adr/0013-the-hardware-abstraction-layer.md)): one interface for each kind of device, which a real driver, a virtual device, a fake, or a bridge to another process can each implement.

- **`Camera`:** its capabilities (modes with sizes, bit depths and rates, in 8 bits and in 16-bit words; binning; ROI steps; exposure and gain ranges; triggers), its configuration, streaming into a `FrameSink`, snapping, a software trigger, and its events. `CameraCapabilities::check` says exactly what is wrong with a configuration, and `layout` gives the frames it makes.
- **Ready for a bridge:** frames come from a pool the driver owns, and are shared, never borrowed; capabilities, configurations and events serialise.
- **Blocking methods on `Send + Sync` traits:** drivers block, and the API layer calls them off its async threads.
- **Events** go to each subscriber on a channel of its own.
- **`FakeCamera`:** a moving test pattern at a set rate, timestamped by a `Clock`, from a pool of 4 frames, with faults on demand: a refused configuration, dropped frames, a failed stream, a disconnection.
- **`contract::camera::all`:** the checks every camera must pass. The fake passes them now; the virtual camera (P1.12) and the real one (M5) will run them too.
- **`Stage`:** its capabilities (each axis's travel, fastest speed and acceleration, its resolution, where it homes), readings of where it is and what it is doing, timestamped; moves to and by, jogs, stop, homing, and its state changes as events. Moves, jogs and homing start and return, and `wait` waits until the stage is still.
- **`FakeStage`:** three axes that move at once or in time, on a straight line at each axis's speed, with faults on demand: a refused move, a stall into alarm, an alarm, a disconnection.
- **`contract::stage::all`:** the checks every stage must pass. The fake passes them at either pace; the virtual stage (P1.10b) and the grblHAL driver (M5) will run them too.
- **Each contract suite has tests of its own:** a device flawed on purpose, in one way for each check, fails that check.
- **`ObjectiveCatalogue`:** the objectives, from `configs/objectives.toml`, each checked; `ManualObjectiveMount`, where the user says which is fitted.
- **`Illumination`:** a light's channels, each with its intensity and colour temperature checked; `ManualIllumination` records what the user says of a light set by hand.

## How to try it

```bash
cargo run -p tetera-hal --example hal-fakes
```

`hal-fakes` prints the fake camera's capabilities, configures and snaps it, streams for a second at 30 fps with two frames dropped on purpose, triggers three frames in software, disconnects it, and prints every event it gave. Then it moves the fake stage, in time, to and by, jogs it, stalls it and homes it, printing its events; reads the objective catalogue and fits the 20×; and sets the light. `-- --smoke` streams and jogs for a quarter of the time.
