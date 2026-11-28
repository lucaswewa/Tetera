# ADR-0019: The preview protocol

- Status: Accepted
- Date: 2026-11-28
- Phase: P1.14b

## Context

- **The live view is JPEG over a binary WebSocket, sized to the client's viewport, latest frame wins, never faster than the camera** (D5). P1.14a's preview session makes the JPEGs: a viewport, a region, a quality and a maximum rate, changed mid-stream, and a slow link's hook that caps the quality and scales the viewport.
- **A socket's buffers hold frames the client hasn't seen.** A browser that falls behind still drains its socket in order, so without pacing it shows frames that are seconds old.
- **teta-wot's binary streams** (v0.2.0) run a producer for each WebSocket, with a one-slot sink that never waits, and parameters the client sends as JSON text, each message replacing the last; a message that doesn't deserialise is answered with `{"error": …}`. The Thing Description links the stream with its parameters' schema.
- **ADR-0017** put media on Tetera's own routes under `/api/v1/`, and left open whether teta-wot's binary streams would grow what the preview needs.
- **The instrument's clock maps its timestamps to UTC** (ADR-0006), so a frame's exposure has a wall-clock time a page can compare with its own.
- **MJPEG** (`multipart/x-mixed-replace`) plays in an `<img>` and in tools such as VLC and OpenCV (Q23).

## Decision

1. **The preview's stream is the `preview` Thing's binary stream,** `/preview/stream`: each connection is a preview session of its own. teta-wot's binary streams carry what the preview needs once the client's message is its whole state (2), so the preview is a Thing's stream after all, and ADR-0017's question is closed for it.
2. **The client sends its whole state as JSON text,** each time it changes or it shows a frame: `{"width", "height", "roi", "quality", "max_rate", "ack"}`. A field left out takes the Thing's default. A message that isn't valid is answered with `{"error": …}` and changes nothing. As each message replaces the last, a lost one is made good by the next.
3. **Each binary message is one frame:** a 72-byte header, little-endian, then the JPEG. P1.14b's first header was 64 bytes, without the latency; readers take both.

   | Offset | Size | Field |
   |---|---|---|
   | 0 | 4 | `TPV1` |
   | 4 | 2 | the header's length, 72 |
   | 6 | 2 | the payload's format: 1, JPEG |
   | 8 | 8 | the frame's sequence number |
   | 16 | 8 | its exposure's start, ns on the instrument's clock |
   | 24 | 8 | its exposure's end, ns on the instrument's clock |
   | 32 | 8 | its exposure's end in UTC, µs since the Unix epoch |
   | 40 | 8 | when the server sent it, in UTC, µs since the Unix epoch |
   | 48 | 4 | the image's width |
   | 52 | 4 | its height |
   | 56 | 1 | the JPEG's quality |
   | 57 | 1 | flags: bit 0, the link lowered it |
   | 58 | 2 | reserved, 0 |
   | 60 | 4 | the payload's length |
   | 64 | 4 | the latency of the last frame acknowledged before this one was sent: from its exposure's end to its acknowledgement, µs on the server's clock; `0xFFFFFFFF` before any (P1.14c) |
   | 68 | 4 | reserved, 0 |

   A reader skips a header longer than it knows, so fields can be added at the end.
4. **Acknowledgements pace the sender.** The client acknowledges a frame, by its sequence number, once it has painted it. The server sends a frame only while fewer than two are sent and unacknowledged, so the socket never holds more than two.
5. **No stale frame:** one slot lies between the session and the socket, holding the newest preview; a preview is sent only if it is newer than the last one sent; and teta-wot's sink replaces a message the socket hasn't taken.
6. **A slow link lowers the preview,** through P1.14a's hook. When the window held previews back for more than half of the last second, the link steps down: the quality capped at 60, then 40, then the viewport scaled to 0.75, then 0.5. After two seconds held back for less than a tenth of the time, it steps back up one. The header's flag says when the link lowered a frame.
7. **Both UTC times in the header go through the instrument clock's one mapping to UTC** (ADR-0006), so their difference is the monotonic clock's: the time the frame waited before it was sent. The wall clock would not do: in this phase's tests, WSL2's stepped 3.3 s while a stream ran.
8. **Latency is measured on the server's clock alone,** from a frame's exposure's end to its acknowledgement, which the client sends once it has painted the frame: an upper bound on the time to the paint, by the ack's trip, a fraction of a millisecond on localhost. Each frame's header carries the last acknowledged frame's (P1.14c), and `sessions` shows it. **A client's own clock can't stand in for the server's:**
   - a browser on Windows found the server's mapping to UTC 3 s behind its clock, as WSL2's wall clock had been when the server started; WSL2 steps its wall clock by seconds to keep up with Windows';
   - on this PC, WSL2's monotonic clock runs at 0.90 of Windows' (the `tsc` clock source), so a disagreement measured once, as the least gap between a receipt and its sending, grows by about 100 ms a second.

   A client on the server's own clock, such as P1.14b's example in WSL, can still take the exposure's end from its receipt, less that least gap.
9. **MJPEG is a media route** (P1.14c), `/api/v1/preview/mjpeg`, with the viewport, quality and rate in its query: a session of its own for each viewer, paced by its socket, as `multipart/x-mixed-replace`. It ends with the server.
10. **`live-view`,** a plain page with no framework, is served at `/live-view` until the web app arrives in M4 (P1.14c).

## Consequences

- **Gains:**
  - a slow client never stalls the camera or other clients, and each frame it gets was the newest when it was sent; the second of the two in flight waits while the client paints the first, so it is one paint old when read;
  - the stream is in the preview Thing's Thing Description, with its parameters;
  - each frame carries what a client needs to place and time it, without a second channel.
- **Costs:**
  - each client is a session of its own: two clients of the same size do the work twice;
  - acknowledgements cost a message a frame upstream;
  - a client that never acknowledges gets two frames.
- **To watch:**
  - clocks that disagree: WSL2's wall clock is stepped to Windows', and its monotonic clock runs at another rate on this PC. Every time the server measures runs on WSL2's clock: 53 frames a second by it are 48 by Windows'. A real camera's own clock, and a clock source that keeps time, are for M5 and the owner (ADR-0006, ADR-0018);
  - whether the step-down's thresholds suit a real remote link (M5).

## Alternatives considered

- **A WebSocket route of Tetera's own under `/api/v1/`,** as ADR-0017 expected: full control of the messages, but teta-wot's stream already does what the preview needs, with its Thing Description and authentication.
- **Acknowledgements as their own messages:** natural, but teta-wot keeps only the client's latest message, so a settings change followed at once by an acknowledgement could be lost.
- **teta-wot's MJPEG stream:** a viewer page and a TD link for free, but one stream of one size, encoded whether anyone watches or not.
- **WebRTC or H.264:** less bandwidth, but much more machinery, and a latency the codec adds; H.264 waits for M5 if remote links need it (Q18).
