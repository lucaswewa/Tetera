# ADR-0016: The virtual camera's model

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.12a, the field; the sensor and the camera, P1.12b

## Context

- **Checkpoint A** (P1.12b) is the virtual instrument rendering H&E live on the GPU, while the stage moves. The virtual camera must look like the MIChrome 20 through the fitted objective: its field of view, its depth of field, its light, its sensor, and its rate.
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
7. **The sensor, motion and timing** (P1.12b):
   - RGGB, quantum efficiency, shot, read and dark noise, black level, gain, 12 bits in 16;
   - the stage's path during the exposure and the rolling shutter's line time;
   - the MIChrome 20's modes and rates on the simulated clock.

## Consequences

- **Gains:**
  - changing the objective changes the field and the blur; focusing changes which groups are sharp;
  - still frames take 9 ms at 5×, 19 ms at 20× and 38 ms at 50× on this PC, within the budget;
  - the twins agree within 4e-5 of the light.
- **Costs:**
  - a moving field makes the tiles newly in view. At 50×, moving 100 pixels a frame, a frame takes 40 to 45 ms, a little over the budget; faster moves take longer. P1.12b measures jogging and reports it;
  - the first frame of a new place makes all its tiles: 0.3 to 0.9 s;
  - the tiles of a 50× field at three depth groups hold about 620 MB of VRAM.
- **To watch:**
  - whether making the newly visible tiles in one launch, not one each, brings jogging within the budget;
  - whether a Gaussian is a close enough defocus for autofocus (P1.16) to behave as on real optics.

## Alternatives considered

- **Rendering each frame from scratch at the sensor's sampling:** simple, and no cache, but 80 to 190 ms a frame.
- **A proper defocused PSF** (P1.07b's) by FFT: closer to real optics, but tens of ms a group at 20 MP.
- **Blurring the light, not the stains:** right for a thick specimen, but each group would need its own light and exponent; for a section 4 µm thick the difference is small.
- **Every 0.5 µm layer blurred on its own:** finer in depth, but eight blurs at 50× instead of three, for little visible gain.
