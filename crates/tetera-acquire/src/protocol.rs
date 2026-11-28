//! The preview protocol's messages (ADR-0019), with no network: each frame's header, the
//! client's state, the acknowledgement window that paces the sender, and the slow link's
//! controller, which lowers a preview through P1.14a's hook.

use crate::preview::{Link, PreviewParams};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tetera_core::{Error, Rect, Result, Timestamp};

/// A frame message's first four bytes.
pub const MAGIC: [u8; 4] = *b"TPV1";

/// The header's length, bytes.
pub const HEADER_LEN: usize = 64;

/// The payload's format: a JPEG.
pub const JPEG: u16 = 1;

/// The most frames sent and not yet acknowledged.
pub const WINDOW: usize = 2;

/// A frame message's header (ADR-0019).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// The frame's sequence number.
    pub seq: u64,
    /// Its exposure's start, on the instrument's clock.
    pub exposure_start: Timestamp,
    /// Its exposure's end, on the instrument's clock.
    pub exposure_end: Timestamp,
    /// Its exposure's end in UTC: µs since the Unix epoch.
    pub exposure_end_utc_us: i64,
    /// When the server sent it, in UTC: µs since the Unix epoch.
    pub sent_utc_us: i64,
    /// The image's width.
    pub width: u32,
    /// Its height.
    pub height: u32,
    /// The JPEG's quality.
    pub quality: u8,
    /// Whether the link lowered its quality or size.
    pub lowered: bool,
}

impl Header {
    /// The message: this header, then `jpeg`.
    pub fn message(&self, jpeg: &[u8]) -> Vec<u8> {
        let mut m = Vec::with_capacity(HEADER_LEN + jpeg.len());
        m.extend_from_slice(&MAGIC);
        m.extend_from_slice(&(HEADER_LEN as u16).to_le_bytes());
        m.extend_from_slice(&JPEG.to_le_bytes());
        m.extend_from_slice(&self.seq.to_le_bytes());
        m.extend_from_slice(&self.exposure_start.0.to_le_bytes());
        m.extend_from_slice(&self.exposure_end.0.to_le_bytes());
        m.extend_from_slice(&self.exposure_end_utc_us.to_le_bytes());
        m.extend_from_slice(&self.sent_utc_us.to_le_bytes());
        m.extend_from_slice(&self.width.to_le_bytes());
        m.extend_from_slice(&self.height.to_le_bytes());
        m.push(self.quality);
        m.push(u8::from(self.lowered));
        m.extend_from_slice(&[0, 0]);
        m.extend_from_slice(&(jpeg.len() as u32).to_le_bytes());
        m.extend_from_slice(jpeg);
        m
    }

    /// A message's header and its JPEG. A header longer than this one's is skipped past.
    pub fn parse(message: &[u8]) -> Result<(Header, &[u8])> {
        let bad = |why: &str| Error::invalid_argument(format!("a preview message {why}"));
        if message.len() < HEADER_LEN || message[..4] != MAGIC {
            return Err(bad("without its header"));
        }
        let u16_at = |i: usize| u16::from_le_bytes([message[i], message[i + 1]]);
        let u32_at = |i: usize| u32::from_le_bytes(message[i..i + 4].try_into().expect("4 bytes"));
        let u64_at = |i: usize| u64::from_le_bytes(message[i..i + 8].try_into().expect("8 bytes"));
        let len = usize::from(u16_at(4));
        if len < HEADER_LEN || u16_at(6) != JPEG {
            return Err(bad("of a header or format this reader doesn't know"));
        }
        let payload = u32_at(60) as usize;
        if message.len() != len + payload {
            return Err(bad("of the wrong length"));
        }
        let header = Header {
            seq: u64_at(8),
            exposure_start: Timestamp(u64_at(16)),
            exposure_end: Timestamp(u64_at(24)),
            exposure_end_utc_us: u64_at(32) as i64,
            sent_utc_us: u64_at(40) as i64,
            width: u32_at(48),
            height: u32_at(52),
            quality: message[56],
            lowered: message[57] & 1 == 1,
        };
        Ok((header, &message[len..]))
    }
}

/// What a client asks for: its whole state, in each message (ADR-0019). A field left out
/// takes the defaults.
#[derive(Debug, Clone, Default, PartialEq, Serialize, JsonSchema)]
pub struct ClientState {
    /// The viewport's width, 16 to 8,192 pixels.
    pub width: Option<u32>,
    /// The viewport's height.
    pub height: Option<u32>,
    /// The region of the frame shown, 16 pixels or more each way.
    pub roi: Option<Rect>,
    /// The JPEG's quality, 1 to 100.
    pub quality: Option<u8>,
    /// The most frames a second.
    pub max_rate: Option<f64>,
    /// The last frame painted, by its sequence number.
    pub ack: Option<u64>,
}

/// [`ClientState`] as it comes, before it is checked.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Unchecked {
    width: Option<u32>,
    height: Option<u32>,
    roi: Option<Rect>,
    quality: Option<u8>,
    max_rate: Option<f64>,
    ack: Option<u64>,
}

impl<'de> Deserialize<'de> for ClientState {
    /// Refuses a state no session can take, so the client is told at once.
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let u = Unchecked::deserialize(d)?;
        let state = ClientState {
            width: u.width,
            height: u.height,
            roi: u.roi,
            quality: u.quality,
            max_rate: u.max_rate,
            ack: u.ack,
        };
        state
            .params(&PreviewParams::default())
            .map_err(serde::de::Error::custom)?;
        Ok(state)
    }
}

impl ClientState {
    /// The preview's parameters: this state over `defaults`, checked.
    pub fn params(&self, defaults: &PreviewParams) -> Result<PreviewParams> {
        let params = PreviewParams {
            width: self.width.unwrap_or(defaults.width),
            height: self.height.unwrap_or(defaults.height),
            roi: self.roi.or(defaults.roi),
            quality: self.quality.unwrap_or(defaults.quality),
            max_rate: self.max_rate.unwrap_or(defaults.max_rate),
            white_balance: defaults.white_balance,
        };
        params.check()?;
        Ok(params)
    }
}

/// The frames sent and not yet acknowledged: no more than [`WINDOW`], so a socket never
/// holds frames its client hasn't seen.
#[derive(Debug, Default)]
pub struct AckWindow {
    sent: VecDeque<(u64, Instant)>,
    last: Option<u64>,
}

impl AckWindow {
    /// Whether another frame may be sent.
    pub fn open(&self) -> bool {
        self.sent.len() < WINDOW
    }

    /// Whether frame `seq` is newer than the last one sent.
    pub fn is_new(&self, seq: u64) -> bool {
        self.last.is_none_or(|last| seq > last)
    }

    /// Frame `seq` sent at `at`.
    pub fn sent(&mut self, seq: u64, at: Instant) {
        self.sent.push_back((seq, at));
        self.last = Some(seq);
    }

    /// Every frame up to `seq` acknowledged at `at`: `seq`'s round trip, if it was waiting.
    pub fn ack(&mut self, seq: u64, at: Instant) -> Option<Duration> {
        let mut round_trip = None;
        while let Some(&(s, t)) = self.sent.front() {
            if s > seq {
                break;
            }
            if s == seq {
                round_trip = Some(at.saturating_duration_since(t));
            }
            self.sent.pop_front();
        }
        round_trip
    }

    /// Frames sent and not yet acknowledged.
    pub fn in_flight(&self) -> usize {
        self.sent.len()
    }
}

/// The steps a slow link takes down, each lower than the one before: the quality capped,
/// then the viewport scaled.
pub const STEPS: [Link; 5] = [
    Link {
        quality: None,
        scale: 1.0,
    },
    Link {
        quality: Some(60),
        scale: 1.0,
    },
    Link {
        quality: Some(40),
        scale: 1.0,
    },
    Link {
        quality: Some(40),
        scale: 0.75,
    },
    Link {
        quality: Some(40),
        scale: 0.5,
    },
];

/// Lowers a preview on a slow link, and raises it again (ADR-0019): a step down when the
/// window held previews back for more than half of the last second, a step up after two
/// seconds held back for less than a tenth of the time.
#[derive(Debug)]
pub struct LinkControl {
    step: usize,
    changed: Instant,
    /// Each preview of the last two seconds, and whether the window held it back.
    recent: VecDeque<(Instant, bool)>,
}

impl LinkControl {
    /// A link at its best, from `now`.
    pub fn new(now: Instant) -> Self {
        Self {
            step: 0,
            changed: now,
            recent: VecDeque::new(),
        }
    }

    /// The link's step now: 0, its best, to `STEPS.len() - 1`.
    pub fn step(&self) -> usize {
        self.step
    }

    /// The hook for the session.
    pub fn link(&self) -> Link {
        STEPS[self.step]
    }

    /// A preview at `now`, sent, or held back by the window: the step from now.
    pub fn record(&mut self, now: Instant, held: bool) -> usize {
        self.recent.push_back((now, held));
        while self
            .recent
            .front()
            .is_some_and(|(t, _)| now.saturating_duration_since(*t) > Duration::from_secs(2))
        {
            self.recent.pop_front();
        }
        let since = now.saturating_duration_since(self.changed);
        let share = |within: Duration| {
            let (held, all) = self
                .recent
                .iter()
                .filter(|(t, _)| now.saturating_duration_since(*t) <= within)
                .fold((0usize, 0usize), |(h, a), (_, held)| {
                    (h + usize::from(*held), a + 1)
                });
            held as f64 / all.max(1) as f64
        };
        if since >= Duration::from_secs(1)
            && share(Duration::from_secs(1)) > 0.5
            && self.step + 1 < STEPS.len()
        {
            self.step += 1;
            self.changed = now;
        } else if since >= Duration::from_secs(2)
            && share(Duration::from_secs(2)) < 0.1
            && self.step > 0
        {
            self.step -= 1;
            self.changed = now;
        }
        self.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Header {
        Header {
            seq: 41,
            exposure_start: Timestamp(1_000_000),
            exposure_end: Timestamp(6_000_000),
            exposure_end_utc_us: 1_795_000_000_000_000,
            sent_utc_us: 1_795_000_000_020_000,
            width: 1_280,
            height: 853,
            quality: 72,
            lowered: true,
        }
    }

    #[test]
    fn a_frames_header_reads_back_and_a_bad_one_is_refused() {
        let jpeg = [0xff, 0xd8, 1, 2, 3, 0xff, 0xd9];
        let message = header().message(&jpeg);
        assert_eq!(message.len(), HEADER_LEN + jpeg.len());
        assert_eq!(&message[..4], b"TPV1");
        assert_eq!(&message[60..64], &(jpeg.len() as u32).to_le_bytes());
        let (got, payload) = Header::parse(&message).unwrap();
        assert_eq!((got, payload), (header(), &jpeg[..]));
        let plain = Header {
            lowered: false,
            ..header()
        };
        assert!(!Header::parse(&plain.message(&jpeg)).unwrap().0.lowered);
        assert_eq!(
            Header::parse(&header().message(&[])).unwrap(),
            (header(), &[][..]),
            "an empty payload"
        );
        // A longer header, from a later version, is skipped past.
        let mut longer = message[..HEADER_LEN].to_vec();
        longer[4..6].copy_from_slice(&72u16.to_le_bytes());
        longer.extend_from_slice(&[9; 8]);
        longer.extend_from_slice(&jpeg);
        assert_eq!(Header::parse(&longer).unwrap(), (header(), &jpeg[..]));
        let mut bad_magic = message.clone();
        bad_magic[0] = b'X';
        let mut shorter_header = message.clone();
        shorter_header[4..6].copy_from_slice(&60u16.to_le_bytes());
        let mut png = message.clone();
        png[6] = 2;
        for bad in [
            &message[..HEADER_LEN - 1],
            &message[..message.len() - 1],
            &bad_magic[..],
            &shorter_header[..],
            &png[..],
        ] {
            assert!(Header::parse(bad).is_err(), "{} bytes", bad.len());
        }
    }

    #[test]
    fn a_clients_state_lays_over_the_defaults_and_a_bad_one_is_refused() {
        let defaults = PreviewParams {
            white_balance: [1.5, 1.0, 2.0],
            ..PreviewParams::default()
        };
        let state: ClientState =
            serde_json::from_str(r#"{"width": 640, "quality": 50, "ack": 12}"#).unwrap();
        assert_eq!(state.ack, Some(12));
        let p = state.params(&defaults).unwrap();
        assert_eq!(
            (p.width, p.height, p.quality, p.max_rate, p.white_balance),
            (640, 1_280, 50, 60.0, [1.5, 1.0, 2.0])
        );
        let all: ClientState = serde_json::from_str(
            r#"{"width": 320, "height": 200, "roi": {"x": 8, "y": 8, "width": 64, "height": 32},
                "quality": 90, "max_rate": 15}"#,
        )
        .unwrap();
        let p = all.params(&defaults).unwrap();
        assert_eq!(
            (p.width, p.height, p.roi, p.quality, p.max_rate),
            (320, 200, Some(Rect::new(8, 8, 64, 32)), 90, 15.0)
        );
        assert_eq!(
            serde_json::from_str::<ClientState>("{}").unwrap(),
            ClientState::default()
        );
        for bad in [
            r#"{"width": 8}"#,
            r#"{"height": 9000}"#,
            r#"{"quality": 0}"#,
            r#"{"max_rate": -1}"#,
            r#"{"roi": {"x": 0, "y": 0, "width": 4, "height": 64}}"#,
            r#"{"zoom": 2}"#,
            r#"{"width": "wide"}"#,
            r#"[1, 2]"#,
        ] {
            assert!(serde_json::from_str::<ClientState>(bad).is_err(), "{bad}");
        }
        let narrow = ClientState {
            width: Some(4),
            ..ClientState::default()
        };
        assert!(narrow.params(&defaults).is_err());
    }

    #[test]
    fn the_window_holds_two_frames_and_acknowledgements_open_it() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut w = AckWindow::default();
        assert!(w.open() && w.is_new(0));
        w.sent(3, at(0));
        assert!(w.open() && !w.is_new(3) && w.is_new(4));
        w.sent(5, at(10));
        assert_eq!(w.in_flight(), 2);
        assert!(!w.open(), "two in flight");
        assert_eq!(
            w.ack(4, at(30)),
            None,
            "4 wasn't sent, but 3 is acknowledged with it"
        );
        assert_eq!((w.in_flight(), w.open()), (1, true));
        w.sent(6, at(40));
        assert_eq!(
            w.ack(6, at(55)),
            Some(Duration::from_millis(15)),
            "6's round trip"
        );
        assert_eq!(w.in_flight(), 0, "5 is acknowledged with 6");
        assert_eq!(w.ack(2, at(60)), None);
        assert!(!w.is_new(6) && w.is_new(7));
    }

    #[test]
    fn a_slow_link_steps_down_and_a_quick_one_back_up() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut down = LinkControl::new(t0);
        assert_eq!((down.step(), down.link()), (0, STEPS[0]));
        // Every frame held back: a step down a second after the start, then each second after.
        for ms in (20..1_000).step_by(20) {
            assert_eq!(down.record(at(ms), true), 0, "not before a second: {ms} ms");
        }
        assert_eq!(down.record(at(1_000), true), 1);
        assert_eq!(down.link(), STEPS[1]);
        for ms in (1_020..2_000).step_by(20) {
            assert_eq!(
                down.record(at(ms), true),
                1,
                "a second between steps: {ms} ms"
            );
        }
        assert_eq!(down.record(at(2_000), true), 2);
        for ms in (2_020..=6_000).step_by(20) {
            down.record(at(ms), true);
        }
        assert_eq!(down.step(), STEPS.len() - 1, "no lower than the last");
        assert_eq!(down.link().scale, 0.5);
        // Nothing held back: a step up two seconds after the last, then each two seconds after.
        let mut up = LinkControl::new(t0);
        up.step = 4;
        for ms in (20..2_000).step_by(20) {
            assert_eq!(up.record(at(ms), false), 4, "{ms} ms");
        }
        assert_eq!(up.record(at(2_000), false), 3);
        for ms in (2_020..4_000).step_by(20) {
            assert_eq!(up.record(at(ms), false), 3, "{ms} ms");
        }
        assert_eq!(up.record(at(4_000), false), 2);
        // At its best, a quick link stays there.
        let mut best = LinkControl::new(t0);
        for ms in (20..=3_000).step_by(20) {
            assert_eq!(best.record(at(ms), false), 0, "{ms} ms");
        }
        // Exactly half held back is not more than half.
        let mut half = LinkControl::new(t0);
        half.record(at(990), true);
        assert_eq!(half.record(at(1_010), false), 0);
        // A frame a second old is still in the last second.
        let mut edge = LinkControl::new(t0);
        edge.record(at(0), true);
        edge.record(at(500), true);
        assert_eq!(edge.record(at(1_000), false), 1, "two of three held back");
        // Exactly a tenth held back is not less than a tenth, and a frame two seconds old is
        // remembered; a moment later it is forgotten.
        let mut tenth = LinkControl::new(t0);
        tenth.step = 1;
        tenth.record(at(0), true);
        for ms in (400..2_000).step_by(200) {
            assert_eq!(tenth.record(at(ms), false), 1, "{ms} ms");
        }
        assert_eq!(tenth.record(at(2_000), false), 1, "one in ten");
        assert_eq!(tenth.record(at(2_200), false), 0, "none in ten");
    }
}
