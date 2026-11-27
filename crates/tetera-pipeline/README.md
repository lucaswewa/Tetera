# tetera-pipeline

Tetera's pipeline engine, L3 (plan §6.7, [ADR-0012](../../docs/adr/0012-the-pipeline-model.md)): image processing as a graph of nodes with typed ports, written as a JSON recipe, checked before it runs, and run in order.

- **Ports** carry a raw frame, an image (its channels and sample type: `u8`, `u16` or `f32`), an encoded image (a JPEG), a scalar, or a table of named numbers. A node's input may leave an image's channels or sample type open, to take any; an output may follow an input, taking what it leaves open from what that input was given.
- **A node** declares a `Descriptor`: a stable id such as `tetera.crop`, a version, its input and output ports, its parameters' JSON Schema, its backends, and whether it can fuse. It is prepared once with its parameters into a `Step`, which then runs on one set of inputs after another, and may keep state between them.
- **A recipe** is JSON with a schema (`Recipe::schema`): its inputs and their types, its nodes, each fed from the recipe's inputs or other nodes' outputs (`node.port`), and its outputs. Recipes in [`recipes/`](recipes/) show the format.
- **`Plan::check`** checks a recipe against the `Registry` and reports every problem, each where it is and what is wrong: an unknown node or version, an unknown port, mismatched types, an input left unfed or given twice, a cycle, missing or invalid parameters. A problem downstream of another isn't reported again.
- **`Plan::run`** runs the nodes in topological order, each after those that feed it, and gives the recipe's outputs. It checks the inputs it is given, and that each node gives what it declared.
- **The registry** holds the nodes by id and version, a recipe getting the latest unless it names one:
  - **glue:** `tetera.crop` and `tetera.statistics`;
  - **the raw path** (P1.08b): `tetera.bin`, `tetera.bayer_split`, `tetera.superpixel`, `tetera.demosaic`, `tetera.dark`, `tetera.flat`, `tetera.resize`, `tetera.colour`, `tetera.display` and `tetera.jpeg`. Each wraps its `tetera-imaging` function, on whichever backend `Compute` chose; JPEG is always on the CPU. `tetera.flat` makes its field once for each master flat;
  - **denoise and deconvolution** (P1.08d): `tetera.temporal_mean` (stateful, its window on either backend), `tetera.bilateral` (on either), and `tetera.richardson_lucy`, its PSF made once from the optics in its parameters. Richardson–Lucy takes its image on the host, and `tetera-imaging` runs it on the GPU when there is one, with its own copies.
- **[`recipes/live-chain.json`](recipes/live-chain.json)** is the live chain, node by node: dark, flat, demosaic, resize, colour and JPEG. It gives exactly the preview `Isp::run` gives.

**Live** (P1.08d): `Live::start` runs a plan on a thread of its own, its other inputs held, and `Live::offer` gives it frames. A frame offered while one runs waits; a frame offered while one waits takes its place. So at most one frame waits, and the next to run is always the latest. `LiveStats` counts the frames offered, run, given way and dropped at the stop, and each run's outputs, or its error, go to a callback.

**Batches and tiles** (P1.08e):
- **`Plan::run_batch`** runs a plan over a list of inputs in order, a stateful node carrying its state from one to the next.
- **Each step says how far it reaches,** in pixels: 0 for a pixel's own, 2 for Malvar-He-Cutler's window, a bilateral filter's radius from its σ. Or it needs the whole image (flat-fielding, by its master's mean; resizing; statistics), or keeps state. `Plan::reach` is the longest path's sum.
- **`Plan::run_tiled`** cuts a recipe's frames and images into tiles, each with a halo as wide as the recipe reaches, rounded up to even to keep the Bayer pattern. It runs each tile and stitches each tile's own part: the outputs are exactly the whole frame's. It refuses a recipe with a node that needs the whole image, or keeps state.

**On the GPU** (P1.08c), with the `cuda` feature and `Compute` choosing it:
- **Values stay where their readers are.** A node that runs on the GPU takes its frames and images there, and gives its own there. A value goes to or from the host only where a step on the other side reads it, once, however many steps read it there. `Plan::copies` counts each run's copies. A value is freed after its last reader.
- **Held inputs:** `Plan::hold` keeps an input, such as a master dark, from run to run. The steps it alone feeds run once and keep their outputs, unless a node is stateful, so a master flat's dark subtraction and its flat field are worked once.
- **The fusion pass** finds the live chain, `[bin →] [dark →] [flat →] demosaic → resize → colour`, each step feeding only the next. On the GPU it runs P1.06b's fused kernel in the chain's place, and gives exactly what the chain gives. `Plan::fused` names the nodes it replaces; `Plan::set_fusion(false)` turns it off; `Plan::ran` names the steps a run ran.

## How to try it

```bash
cargo run --release -p tetera-pipeline --example recipe --features cuda
```

`recipe` lists the registry, refuses [`recipes/broken.json`](recipes/broken.json) with each of its six problems, then runs [`recipes/crop-and-measure.json`](recipes/crop-and-measure.json) on a full-frame test pattern, in about 5 ms. Then it runs the live chain's recipe on a 20 MP raw frame from a synthetic sensor, to a 1920 × 1280 preview and a JPEG, the dark and the flat held. It runs on the CPU, on CUDA node by node, and on CUDA fused, and checks each preview is the ISP's own. The JPEG goes to `target/recipe/preview.jpg`. On this PC, a run takes about 130 ms on the CPU, 33 ms on CUDA node by node, and 30 ms fused, each run on CUDA copying the frame up and the preview down, 47 MB. A run's time includes copying the frame from the host, uploading it, and the JPEG. Last, it runs the live chain live, 29 frames from a synthetic camera at 14.5 fps: on CUDA, node by node or fused, every frame runs, at 14.5 fps; on the CPU, about half give way to a later one. Then it writes four 20 MP frames to `target/recipe/frames` as 16-bit PPM files, reads them back, and runs [`recipes/denoise.json`](recipes/denoise.json) over them as a batch, whole and in tiles of 1,024 × 1,024, the two exactly the same.

`-- PATH.json` checks another recipe; `-- --schema` prints the recipe schema; `-- --catalogue` prints every node's descriptor; and `-- --smoke` takes a 640 × 480 frame.
