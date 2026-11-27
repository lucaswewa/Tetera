# ADR-0015: Simulated samples on the GPU

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.11a; the GPU and the tiles, P1.11b

## Context

- **M1 builds against a simulated instrument** (plan D8, D9): the virtual camera images a simulated H&E section through simulated optics. It must look right at 5×, 20× and 50×, and have depth, for z-scans, autofocus and depth from focus (Q26).
- **The camera wants 20 MP frames** at its frame rate. At 50× that is a field of 260 × 175 µm sampled at 0.048 µm; at 5×, 2.6 × 1.75 mm at 0.48 µm. A stored image of a whole slide at 50× would be terabytes.
- **Tests and checkpoints need the same sample every time,** on every machine, and the GPU must be checkable against the CPU (ADR-0010).
- **Ground truth matters later:** stitching, depth from focus and object recognition (P3) compare their results with where things really are.

## Decision

1. **The sample is procedural:** a pure function of a seed, a position and a depth, giving each stain's density. Nothing is stored but the seed.
2. **Its texture is from the literature:**
   - improved noise (Perlin, 2002) and cellular noise (Worley, 1996), and fBm of them;
   - tissue built from them as fields: a mask, regions, nuclei, collagen, glands, vessels with red cells, fat;
   - a section a few µm thick, nuclei at depths in it, a tilt and folds.
3. **Integer hashes, not tables:** every random choice is a hash of integer coordinates, the seed and a salt for each use, so there is no permutation table to share, and the GPU computes the same choices. Positions are split into lattice cell and offset in double precision; everything else is single precision.
4. **Colour by absorbance:**
   - stain densities in optical density per µm, with Ruifrok and Johnston's (2001) stain vectors;
   - Beer–Lambert for transmitted light.

   So optics, defocus and the light (P1.12) act on physical quantities, not on a picture.
5. **Two passes:**
   - the discrete things (nuclei, red cells) are decided once for each grid cell, each at its own centre;
   - then each pixel is computed on its own, from its column and the cells around it, in a fixed order.

   So threads, tiles and backends can't change a result, and the nuclei can be listed as ground truth.
6. **The CPU twin is the reference;** the GPU (P1.11b) computes the same functions. The phase's behaviour spec states their tolerance.
7. **Tiles on the GPU** (P1.11b):
   - 256 × 256 pixels, keyed by level of detail, position and depth layer;
   - half precision;
   - made on demand, and kept in an LRU cache within a VRAM budget.

## Consequences

- **Gains:**
  - any part of a slide, at any magnification and depth, from a seed;
  - deterministic: the same seed, the same bits, on a backend;
  - ground truth for free: the nuclei and red cells are listed by `nuclei` and `red_cell`;
  - the CPU twin renders a 1,536 × 1,024 field at 5× in about 0.2 s on 8 cores, enough for tests and examples.
- **Costs:**
  - it is a model, not a slide: plausible, judged by eye at Checkpoint A, not histologically exact;
  - each pixel recomputes its fields; the GPU and its cache are what make it fast enough for the live view;
  - floats differ in their last bits between backends, so the twins agree within a tolerance, not exactly.
- **To watch:**
  - whether the camera's rate needs more than the cache gives at 50×;
  - whether P3's object recognition wants more kinds of cell than this model has.

## Alternatives considered

- **Scanned slides,** stored as pyramids: real, but large, fixed, without depth, and with no ground truth.
- **Synthetic images from a generative model:** realistic, but not deterministic across machines, slow, and with no physical stains to blur and illuminate.
- **Permutation-table noise,** Perlin's original: a 512-entry table to share with the GPU, and a period of 256 lattice cells, short on a 100 mm slide.
- **Single precision positions:** too coarse on a 100 mm slide for the 0.05 µm a 50× pixel needs.
