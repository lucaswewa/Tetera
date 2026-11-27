# tetera-pipeline

Tetera's pipeline engine, L3 (plan §6.7, [ADR-0012](../../docs/adr/0012-the-pipeline-model.md)): image processing as a graph of nodes with typed ports, written as a JSON recipe, checked before it runs, and run in order.

- **Ports** carry a raw frame, an image (its channels and sample type: `u8`, `u16` or `f32`), a scalar, or a table of named numbers. A node's input may leave an image's channels or sample type open, to take any; an output may follow an input, taking what it leaves open from what that input was given.
- **A node** declares a `Descriptor`: a stable id such as `tetera.crop`, a version, its input and output ports, its parameters' JSON Schema, its backends, and whether it can fuse. It is prepared once with its parameters into a `Step`, which then runs on one set of inputs after another, and may keep state between them.
- **A recipe** is JSON with a schema (`Recipe::schema`): its inputs and their types, its nodes, each fed from the recipe's inputs or other nodes' outputs (`node.port`), and its outputs. Recipes in [`recipes/`](recipes/) show the format.
- **`Plan::check`** checks a recipe against the `Registry` and reports every problem, each where it is and what is wrong: an unknown node or version, an unknown port, mismatched types, an input left unfed or given twice, a cycle, missing or invalid parameters. A problem downstream of another isn't reported again.
- **`Plan::run`** runs the nodes in topological order, each after those that feed it, and gives the recipe's outputs. It checks the inputs it is given, and that each node gives what it declared.
- **The registry** holds the nodes by id and version, a recipe getting the latest unless it names one. It starts with two glue nodes, `tetera.crop` and `tetera.statistics`; the stock nodes join in P1.08b.

## How to try it

```bash
cargo run --release -p tetera-pipeline --example recipe
```

`recipe` lists the registry, refuses [`recipes/broken.json`](recipes/broken.json) with each of its six problems, then runs [`recipes/crop-and-measure.json`](recipes/crop-and-measure.json) on a full-frame test pattern, in about 5 ms, most of it the crop's copy. `-- PATH.json` checks another recipe; `-- --schema` prints the recipe schema; `-- --catalogue` prints every node's descriptor; and `-- --smoke` takes a 640 × 480 frame.
