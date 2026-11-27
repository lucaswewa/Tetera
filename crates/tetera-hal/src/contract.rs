//! The contract suites (ADR-0013): what any implementation of a device trait must do, as
//! checks that panic with what went wrong. Each implementation's tests run the suite: the
//! fakes now, the virtual devices (P1.10–P1.12) and the drivers (M5) later.

/// The camera's contract.
pub mod camera {
    use crate::camera::{Camera, CameraConfig, CameraEvent, FrameSink, Trigger};
    use std::sync::Arc;
    use std::sync::mpsc::{Receiver, channel};
    use std::time::Duration;
    use tetera_core::{ErrorKind, Frame};

    /// How long the suite waits for a frame.
    const WAIT: Duration = Duration::from_secs(5);

    /// Every check, each on a fresh camera from `make`.
    pub fn all(make: &dyn Fn() -> Box<dyn Camera>) {
        capabilities_are_sound(&*make());
        configurations_round_trip(&*make());
        bad_configurations_are_refused(&*make());
        a_snap_is_the_configured_frame(&*make());
        a_stream_runs_in_order_and_stops_dead(&*make());
        stopping_a_camera_not_streaming_does_nothing(&*make());
        a_software_trigger_gives_one_frame_each(&*make());
    }

    /// A sink that sends each frame down a channel.
    pub fn channel_sink() -> (Arc<dyn FrameSink>, Receiver<Frame>) {
        let (tx, rx) = channel();
        let sink: Arc<dyn FrameSink> = Arc::new(move |frame: Frame| {
            let _ = tx.send(frame);
        });
        (sink, rx)
    }

    fn kind(result: tetera_core::Result<impl std::fmt::Debug>) -> ErrorKind {
        result.expect_err("an error").kind()
    }

    /// It has a mode, its default configuration is one it takes, and its capabilities and
    /// configuration survive JSON.
    pub fn capabilities_are_sound(camera: &dyn Camera) {
        let caps = camera.capabilities();
        assert!(!caps.modes.is_empty(), "a camera has a mode");
        let config = caps.default_config();
        caps.check(&config)
            .expect("the default configuration is one it takes");
        caps.layout(&config)
            .expect("the default configuration lays out a frame");
        let json = serde_json::to_string(&caps).expect("capabilities are JSON");
        assert_eq!(
            serde_json::from_str::<crate::camera::CameraCapabilities>(&json).expect("and back"),
            caps
        );
        let json = serde_json::to_string(&camera.config()).expect("a configuration is JSON");
        assert_eq!(
            serde_json::from_str::<CameraConfig>(&json).expect("and back"),
            camera.config()
        );
    }

    /// Every mode at every bit depth configures, reads back as set, and says so.
    pub fn configurations_round_trip(camera: &dyn Camera) {
        let caps = camera.capabilities();
        let events = camera.subscribe();
        for mode in &caps.modes {
            for &bit_depth in &mode.bit_depths {
                let config = CameraConfig {
                    mode: mode.id,
                    bit_depth,
                    ..caps.default_config()
                };
                camera.configure(&config).expect("a configuration it takes");
                assert_eq!(
                    camera.config(),
                    config,
                    "mode {} at {bit_depth} bits reads back",
                    mode.id
                );
                assert_eq!(
                    events.recv_timeout(WAIT).expect("an event"),
                    CameraEvent::Configured { config },
                    "configuring says so"
                );
            }
        }
    }

    /// A configuration it doesn't take is refused as such, and changes nothing.
    pub fn bad_configurations_are_refused(camera: &dyn Camera) {
        let caps = camera.capabilities();
        let before = camera.config();
        let base = caps.default_config();
        let modes: Vec<u32> = caps.modes.iter().map(|m| m.id).collect();
        let bad = [
            CameraConfig {
                mode: (0..).find(|m| !modes.contains(m)).expect("a free number"),
                ..base.clone()
            },
            CameraConfig {
                bit_depth: 99,
                ..base.clone()
            },
            CameraConfig {
                exposure: Duration::from_secs_f64(caps.exposure_ms.max * 2.0 / 1e3),
                ..base.clone()
            },
            CameraConfig {
                gain: caps.gain.min - 1.0,
                ..base.clone()
            },
            CameraConfig {
                gain: f64::NAN,
                ..base
            },
        ];
        for config in bad {
            assert_eq!(
                kind(camera.configure(&config)),
                ErrorKind::InvalidArgument,
                "{config:?}"
            );
            assert_eq!(
                camera.config(),
                before,
                "a refused configuration changes nothing"
            );
        }
    }

    /// A snap is a frame of the configured layout, its metadata the configuration's, each
    /// numbered after the last.
    pub fn a_snap_is_the_configured_frame(camera: &dyn Camera) {
        let caps = camera.capabilities();
        let mode = caps.modes.last().expect("a mode");
        let config = CameraConfig {
            mode: mode.id,
            bit_depth: mode.bit_depths[0],
            gain: caps.gain.max,
            exposure: Duration::from_secs_f64(
                caps.exposure_ms.min.max(1.0).min(caps.exposure_ms.max) / 1e3,
            ),
            ..caps.default_config()
        };
        camera.configure(&config).expect("a configuration it takes");
        let a = camera.snap(WAIT).expect("a snap");
        let b = camera.snap(WAIT).expect("another");
        assert_eq!(
            *a.layout(),
            caps.layout(&config).expect("its layout"),
            "the configured layout"
        );
        let meta = a.meta();
        assert_eq!(
            (meta.mode, meta.gain, meta.exposure, meta.binning),
            (config.mode, config.gain, config.exposure, config.binning)
        );
        assert!(
            meta.exposure_end >= meta.exposure_start,
            "an exposure ends after it starts"
        );
        assert!(b.meta().seq > a.meta().seq, "numbered in order");
        assert_eq!(
            a.data().len(),
            a.layout().bytes(),
            "the frame's bytes are all there"
        );
    }

    /// Streaming delivers frames in order, refuses what it can't do meanwhile, and once
    /// `stop` returns, delivers nothing more.
    pub fn a_stream_runs_in_order_and_stops_dead(camera: &dyn Camera) {
        let events = camera.subscribe();
        let (sink, frames) = channel_sink();
        camera.start(sink.clone()).expect("a stream starts");
        assert!(camera.is_streaming());
        let mut last: Option<(u64, tetera_core::Timestamp)> = None;
        for _ in 0..5 {
            let frame = frames.recv_timeout(WAIT).expect("a frame while streaming");
            let (seq, at) = (frame.meta().seq, frame.meta().exposure_start);
            if let Some((s, t)) = last {
                assert!(seq > s && at >= t, "in order: {seq} after {s}");
            }
            last = Some((seq, at));
        }
        assert_eq!(kind(camera.start(sink)), ErrorKind::Busy, "a second start");
        assert_eq!(
            kind(camera.configure(&camera.config())),
            ErrorKind::Busy,
            "configuring while streaming"
        );
        assert_eq!(
            kind(camera.snap(WAIT)),
            ErrorKind::Busy,
            "a snap while streaming"
        );
        camera.stop().expect("the stream stops");
        assert!(!camera.is_streaming());
        frames.try_iter().for_each(drop);
        // Nothing more, however long it is waited for.
        assert!(
            frames.recv_timeout(Duration::from_millis(200)).is_err(),
            "a frame came after stop returned"
        );
        let seen: Vec<CameraEvent> = events.try_iter().collect();
        assert_eq!(seen.first(), Some(&CameraEvent::Started), "{seen:?}");
        assert_eq!(seen.last(), Some(&CameraEvent::Stopped), "{seen:?}");
        // It streams again after a stop.
        let (sink, frames) = channel_sink();
        camera.start(sink).expect("a second stream");
        frames.recv_timeout(WAIT).expect("a frame");
        camera.stop().expect("and stops");
    }

    /// Stopping a camera that isn't streaming does nothing, and isn't an error.
    pub fn stopping_a_camera_not_streaming_does_nothing(camera: &dyn Camera) {
        camera.stop().expect("nothing to stop");
        assert!(!camera.is_streaming());
        assert_eq!(
            kind(camera.trigger()),
            ErrorKind::InvalidArgument,
            "a trigger without a stream"
        );
    }

    /// On a software trigger, each trigger gives one frame, and none comes untriggered.
    pub fn a_software_trigger_gives_one_frame_each(camera: &dyn Camera) {
        let caps = camera.capabilities();
        if !caps.triggers.contains(&Trigger::Software) {
            return;
        }
        camera
            .configure(&CameraConfig {
                trigger: Trigger::Software,
                ..caps.default_config()
            })
            .expect("a software trigger");
        let (sink, frames) = channel_sink();
        camera.start(sink).expect("a stream");
        assert!(
            frames.recv_timeout(Duration::from_millis(200)).is_err(),
            "a frame without a trigger"
        );
        for _ in 0..3 {
            camera.trigger().expect("a trigger");
            frames.recv_timeout(WAIT).expect("its frame");
        }
        assert!(
            frames.recv_timeout(Duration::from_millis(100)).is_err(),
            "one frame for each trigger"
        );
        camera.stop().expect("the stream stops");
        camera
            .configure(&caps.default_config())
            .expect("free-running again");
        let (sink, _frames) = channel_sink();
        camera.start(sink).expect("a free-running stream");
        assert_eq!(
            kind(camera.trigger()),
            ErrorKind::InvalidArgument,
            "a trigger while free-running"
        );
        camera.stop().expect("the stream stops");
    }
}
