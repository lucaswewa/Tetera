# ADR-0016: The virtual camera's model

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.12a, the field; P1.12b, the sensor and motion; P1.12c, the camera

## Context

- **Checkpoint A** (P1.12c) is the virtual instrument rendering H&E live on the GPU, while the stage moves. The virtual camera must look like the MIChrome 20 through the fitted objective: its field of view, its depth of field, its light, its sensor, and its rate.
- **The sample** (ADR-0015) gives stain densities by depth, in tiles cached on the GPU. Physics on stains, not on a picture, lets the optics and the light act as they do on glass.
- **The budget** (§6.15) is 40 ms a full 20 MP frame on the GPU, still or jogging. A frame rendered from scratch at full detail takes 80 to 190 ms; the cache is what makes the budget possible.
- **The CPU fallback** must match the GPU, as every kernel's twin does (ADR-0010), and run in tests on a scaled-down sensor.
- **Q27:** linear by default, the SDK's processing as an option, rolling shutter modelled.

## Decision

1. **The field is light at the sensor:** each pixel's irradiance in red, green and blue, in photoelectrons a ms at quantum efficiency 1. The sensor (P1.12b) turns it into counts.
2. **The field of view** comes from the magnification, the 2.4 µm pixels and the camera's mounting angle, centred on where the stage itself is, not on its last report.
3. **Depth in groups as thick as the depth of field:**
   - the depth of field is `λ / NA² + e / (M · NA)`, e the sensor's pixel;
   - the section is split into groups of whole 0.5 µm layers, at most that thick. At 5× and 10× the whole section is one group; at 50×, three.
4. **Blur as Gaussians:** each group is blurred by the Airy disc's Gaussian, `σ = 0.21 λ / NA`, and the circle of confusion of its middle's distance from focus, `σ = |Δz| · NA / (2 √(1 − NA²))`, added in quadrature. Stains, which are optical densities, are blurred and summed over the groups, then turned into light by Beer–Lambert. That is the thin-specimen approximation.
5. **The light:**
   - an intensity;
   - a colour temperature, as Planck's law at each channel's middle wavelength, relative to green, with no white balance, as a raw frame is;
   - a field uneven by a set fraction;
   - vignetting falling as the radius squared.
6. **The field on the GPU comes from the tile cache** at the level just finer than the sensor's pixel:
   - resampled bilinearly into the sensor's (turned) grid;
   - blurred in shared memory;
   - lit;
   - in planes kept from frame to frame, so a still frame allocates nothing.

   The CPU renders the same region and resamples it the same way, so the twins agree.
7. **The sensor** (P1.12b): RGGB, quantum efficiency, shot, read and dark noise, black level, gain, 12 bits in 16 or 8. Its noise is hashed from the seed, the pixel and the frame's number, not drawn from a stream, so either backend draws any pixel alone and a frame repeats bit for bit. Binning sums the light and the charge: the field is a 2.4 µm pixel's light, times the binned area.
8. **Motion** (P1.12b): constant over a frame. Each pixel averages its row's path during its exposure, one sample for each pixel moved, at most 16; row r starts r line times after the first, the line time being the mode's readout over its rows.
9. **Timing** (P1.12c): the MIChrome 20's modes and rates, on the stage's simulated clock. A frame is made once its last row has read out, from where the stage's plan had it from the first row's start to the last row's end. Free-running, frames start an interval apart, the readout or the exposure if longer; a camera that falls whole frames behind drops them and says so.
10. **Defocus is held where the blur reaches 48 sensor pixels** (P1.12c), so a frame far from focus takes no longer than one near it: beyond about 25 µm at 20× and 7 µm at 50×, a frame is as blurred as it gets.

## Consequences

- **Gains:**
  - changing the objective changes the field and the blur; focusing changes which groups are sharp;
  - the full field still takes 5 ms at 5×, 10 ms at 20× and 25 ms at 50× on this PC, once the resampling divides nothing in double precision (P1.12b); a raw frame, still, 17 to 27 ms into the camera's pinned frames (P1.12c);
  - every mode reaches its rate, still;
  - the twins agree within 1e-4 of the light, and the sensors but for a count in 20 million.
- **Costs:**
  - a moving field makes the tiles newly in view. At 20×, a frame is made within the budget jogging at up to 250 µm/s; at 500 µm/s it takes about 50 ms, at 1 mm/s about 75 ms, still at the full rate (P1.12c);
  - the first frame of a new place makes all its tiles: 0.3 to 0.9 s;
  - the tiles of a 50× field at three depth groups hold about 620 MB of VRAM.
- **To watch:**
  - whether making the newly visible tiles in one launch, not one each, brings faster jogging within the budget;
  - whether the held defocus is far enough for autofocus at 50× (P1.16), or wants a lower-resolution blur beyond it;
  - whether a Gaussian is a close enough defocus for autofocus (P1.16) to behave as on real optics.

## Alternatives considered

- **Rendering each frame from scratch at the sensor's sampling:** simple, and no cache, but 80 to 190 ms a frame.
- **A proper defocused PSF** (P1.07b's) by FFT: closer to real optics, but tens of ms a group at 20 MP.
- **Blurring the light, not the stains:** right for a thick specimen, but each group would need its own light and exponent; for a section 4 µm thick the difference is small.
- **Every 0.5 µm layer blurred on its own:** finer in depth, but eight blurs at 50× instead of three, for little visible gain.
