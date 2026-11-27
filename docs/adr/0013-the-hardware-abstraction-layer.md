# ADR-0013: The hardware abstraction layer

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.09a; the stage, P1.09b

## Context

- **Tetera drives a camera (the MIChrome 20), a stage (grblHAL), an objective mount and a light** (plan §1). M1 builds against virtual devices (P1.10–P1.12), and M5 brings the real drivers.
- **The camera may not live in the same process.** If USB forwarding into WSL2 can't keep up, the owner's fallback (R1) is a bridge on Windows sending frames over shared memory or a socket. The camera's interface must work across that line from the start (plan §6.8).
- **Drivers block:** vendor SDKs wait on USB transfers and serial ports. The server is async (P1.13).
- **The frame model exists** (ADR-0008): frames come from a pool of buffers, frozen and shared once written, with their metadata.
- **OScope tested each device type against its own fake,** and each driver's quirks surfaced late. A suite every implementation runs is cheaper than a fake per test.

## Decision

1. **One trait for each kind of device:** `Camera` now; `Stage`, `ObjectiveMount` and `Illumination` in P1.09b. A real driver, a virtual device, a fake and a bridge each implement the same trait.
2. **The methods block, and the traits are `Send + Sync`,** taking `&self`, so one device is shared by every thread that uses it. The API layer calls them off its async threads.
3. **Capabilities, then configuration:**
   - a device says what it can do (`CameraCapabilities`: modes with sizes, bit depths and rates; binnings; ROI steps; exposure and gain ranges; triggers);
   - a configuration is checked against those capabilities by one shared function, which says exactly what is wrong.

   Both serialise, as do events, so a bridge or the API carries them unchanged.
4. **Frames are the driver's, and shared, never borrowed:**
   - a streaming camera takes buffers from a pool it owns, freezes each frame and hands it to a `FrameSink`;
   - a sink may keep frames as long as it likes; the camera drops frames meanwhile, says so as an event, and says on the next frame how many it lost;
   - no reference to a driver's memory crosses the interface, so the same trait works over a bridge.
5. **`stop` is a promise:** once it returns, no more frames reach the sink. `start` and `configure` while streaming are `Busy`, and so is a snap.
6. **Errors out of any call are events:** a failed stream emits an error and stops. Each subscriber gets every event on a channel of its own.
7. **Contract suites:**
   - `contract::camera::all` holds the checks every camera must pass, as public functions that panic with what went wrong;
   - the fake runs them now, the virtual camera (P1.12) and the drivers (M5) later;
   - a fake broken on purpose must fail them.
8. **Fakes with faults on demand:** `FakeCamera` gives a moving test pattern at a set rate from its own pool, timestamped by a `Clock`, and fails when asked: a refused configuration, dropped frames, a failed stream, a disconnection. `FakeStage` moves at once or in time, and fails when asked: a refused move, a stall into alarm, an alarm, a disconnection.
9. **A stage's moves start and return** (P1.09b):
   - `move_to`, `move_by`, `jog` and `home` check what they are asked, start, and return; `wait` blocks until the stage is still, and gives its reading then;
   - so a scan can move and expose at once, and `stop` can be called from another thread while a move is under way;
   - a stage's states are grblHAL's: idle, moving, jogging, homing, and alarm, which only homing clears.
10. **Each contract suite has tests of its own:** a device flawed on purpose, in one way for each check, fails that check, so a check that stops checking is caught.

## Consequences

- **Gains:**
  - a bridge, a virtual camera or a driver is one trait to implement, with a suite to prove it;
  - a slow consumer can't stall a camera: it loses frames, and is told;
  - configurations are checked the same way whatever the device.
- **Costs:**
  - each camera's thread hands frames to a sink, a copy-free handoff, but one more thread than a pull API;
  - the suite's timing checks wait in real time, a second or so.
- **To watch:**
  - an external trigger is declared, but the fake doesn't do it yet, and the suite doesn't test it;
  - a sink that blocks also blocks the camera's thread, and `stop` waits for it. Sinks should hand frames on, not process them.

## Alternatives considered

- **Async traits:** the server is async, but drivers block, and an async trait would push that onto every driver. The API layer moves blocking calls to its own threads instead.
- **A pull API** (`next_frame()`): simpler for one consumer, but a slow consumer stalls the camera, and it doesn't fit a bridge that pushes frames.
- **Moves that block until done:** simpler to call, but a scan couldn't expose while the stage settles, and stopping a move would need a second channel to the driver anyway.
- **Borrowed buffers:** fewer allocations, but a frame couldn't outlive the driver's callback, nor cross a process.
- **One fake for each test:** what OScope did; a shared suite catches more with less.
