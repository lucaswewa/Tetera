# tetera-hal

Tetera's hardware abstraction layer, L2 (plan §6.8, [ADR-0013](../../docs/adr/0013-the-hardware-abstraction-layer.md)): one interface for each kind of device, which a real driver, a virtual device, a fake, or a bridge to another process can each implement.

- **`Camera`:** its capabilities (modes with sizes, bit depths and rates; binning; ROI steps; exposure and gain ranges; triggers), its configuration, streaming into a `FrameSink`, snapping, a software trigger, and its events. `CameraCapabilities::check` says exactly what is wrong with a configuration, and `layout` gives the frames it makes.
- **Ready for a bridge:** frames come from a pool the driver owns, and are shared, never borrowed; capabilities, configurations and events serialise.
- **Blocking methods on `Send + Sync` traits:** drivers block, and the API layer calls them off its async threads.
- **Events** go to each subscriber on a channel of its own.
- **`FakeCamera`:** a moving test pattern at a set rate, timestamped by a `Clock`, from a pool of 4 frames, with faults on demand: a refused configuration, dropped frames, a failed stream, a disconnection.
- **`contract::camera::all`:** the checks every camera must pass. The fake passes them now; the virtual camera (P1.12) and the real one (M5) will run them too.

The stage, the objectives and the light join in P1.09b.

## How to try it

```bash
cargo run -p tetera-hal --example hal-fakes
```

`hal-fakes` prints the fake camera's capabilities, configures and snaps it, streams for a second at 30 fps with two frames dropped on purpose, triggers three frames in software, disconnects it, and prints every event it gave. `-- --smoke` streams for a quarter of a second.
