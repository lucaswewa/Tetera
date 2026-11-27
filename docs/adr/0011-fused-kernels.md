# ADR-0011: Fused kernels

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.06b

## Context

- **The live view runs one chain on every frame:** crop or bin, dark, flat, demosaic, resize, colour, 8 bits (plan D7). Unfused, each stage reads its input from device memory and writes its output back, and allocates it.
- **The unfused chain on the GPU** takes 9.6 ms from a full 20 MP frame to 1920 × 1280, and 2.4 ms from a 2 × 2 binned one (P1.06a). Most of it is the stages' memory traffic and their outputs' allocation: a full frame's demosaicked RGB is 120 MB, written and read once each.
- **The plan's budget** is 10 ms for 20 MP to a 1920 × 1280 preview on the GPU (plan §6.15). The camera gives up to 67 frames a second in its binned modes.
- **Q19 asked:** one hand-written fused kernel for the standard chain, or fusion generated at run time with NVRTC for any chain of fusable nodes? The plan answered hand-written (D7).
- **Every stage of the unfused chain is integer arithmetic** (P1.05a–P1.06a). A kernel that does the same arithmetic in the same order gives the same values.

## Decision

1. **One hand-written fused kernel,** `fused_isp` in `tetera-compute`'s `kernels/fused.cu`, for the standard chain:
   - crop or bin → dark → flat → demosaic (either method) → resize → white balance and CCM → levels and curve → 8 bits;
   - resizing before colour, in linear light, as P1.06a set the chain.
2. **It is a gather:** one thread per preview pixel. A thread walks its resize window. At each tap it demosaics the pixel from the sites around it, reducing, dark-subtracting and flat-fielding each site as it reads it. It applies colour once, to the resized value.
3. **It shares the demosaic's device code** with `demosaic.cu`, in `kernels/demosaic.cuh`: the sums are templates over any source of sites, the fused kernel's being the corrected mosaic, computed site by site. `build.rs` inlines the header into the source it embeds, for NVRTC.
4. **It is the same arithmetic as the unfused chain,** in the same order, with the same rounding. So it matches the chain exactly. The spec's tolerance, within 1 level in 8 bits for 99.9 % of values and 2 for all, is measured by `compare_previews`, and the tests ask for exact equality.
5. **Its reference is the unfused chain on the CPU** (`Isp::run_cpu`). The unfused chain stays, stage by stage, on either backend: for the pipeline engine's other recipes, and as the reference.
6. **Its parameters** travel as one `#[repr(C)]` struct, by value. The safe launch, `CudaBackend::fused_isp`, checks everything its reads rely on before the `unsafe` launch:
   - every buffer's length against the layouts;
   - the crop or bin against the raw frame;
   - the resize's inputs against the reduced size;
   - the method's minimum size.
7. **No tiling for now:** the direct kernel meets the budget. Shared-memory tiles, each block demosaicking its window once, are the next step if a larger preview or a slower GPU needs them.

## Consequences

- **Gains:**
  - 20 MP to 1920 × 1280 in **5.3 ms** on this PC, with no intermediate buffers or allocations;
  - the live view's chain in one launch;
  - exact equality with the reference, so a test can't hide a fused kernel's drift inside a tolerance.
- **Costs:**
  - every tap recomputes its pixel's demosaic, and each site's correction, so the fused kernel does several times the unfused chain's arithmetic. It wins on memory traffic, not on arithmetic;
  - **binning in software inside the fused kernel is slower than unfused:** 5.5 ms against 2.4 ms from a 2 × 2 binned frame, since each site averages four raw sites again at every tap. A camera's binned modes give frames already binned, which the kernel takes as a full frame;
  - a second implementation of the chain to keep equal to the first; the equivalence tests over random parameters guard it.
- **To watch:**
  - previews larger than 1920 × 1280, or full-resolution views, which would make tiling worth its complexity;
  - the pipeline engine (P1.08), which runs this kernel when a recipe matches the standard chain, and the unfused stages otherwise.

## Alternatives considered

- **Fusion generated at run time,** NVRTC compiling any chain of fusable nodes: the plan's Q19 alternative. It is general, but a code generator and its own equivalence problem, before any chain but the standard one needs it.
- **Tiling through shared memory now:** each block demosaics its window once, and resizes from shared memory. It is faster in arithmetic, but bounded by shared memory, which large reductions (thumbnails) exceed. The direct kernel meets the budget without it.
- **Keeping the plan's order,** resizing last on 8-bit values: colour would run once per tap rather than once per pixel, and averaging gamma-encoded values darkens fine detail.
- **Floating point in the fused kernel:** simpler sums, but compiler contraction and rounding would differ from the reference in the last bit, and the equivalence would need its tolerance.
