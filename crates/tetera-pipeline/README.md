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
  - the denoise and deconvolution nodes join in P1.08d.
- **[`recipes/live-chain.json`](recipes/live-chain.json)** is the live chain, node by node: dark, flat, demosaic, resize, colour and JPEG. It gives exactly the preview `Isp::run` gives.

With the `cuda` feature, the nodes run on the GPU when `Compute` chose it. Each still copies its data to the GPU and back; keeping it there between nodes, and fusing the live chain, is P1.08c's.

## How to try it

```bash
cargo run --release -p tetera-pipeline --example recipe --features cuda
```

`recipe` lists the registry, refuses [`recipes/broken.json`](recipes/broken.json) with each of its six problems, then runs [`recipes/crop-and-measure.json`](recipes/crop-and-measure.json) on a full-frame test pattern, in about 5 ms. Then it runs the live chain's recipe on a 20 MP raw frame from a synthetic sensor, to a 1920 × 1280 preview and a JPEG, on each backend. Each preview is checked to be the ISP's own, and the JPEG goes to `target/recipe/preview.jpg`. On this PC, 170 ms on the CPU and 215 ms on CUDA, where each node copies its data to the GPU and back.

`-- PATH.json` checks another recipe; `-- --schema` prints the recipe schema; `-- --catalogue` prints every node's descriptor; and `-- --smoke` takes a 640 × 480 frame.
