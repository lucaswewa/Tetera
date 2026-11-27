# ADR-0012: The pipeline model

- Status: Accepted
- Date: 2026-11-27
- Phase: P1.08a

## Context

- **The plan describes processing as a graph** (§6.7): a recipe is a DAG of nodes with typed ports, written as JSON with a schema. It runs per frame on the live stream, or in batch over a dataset. Each node declares its ports, its parameters as JSON Schema, its backends, and whether it can fuse.
- **Recipes come from outside the code:** the web app's editor (M4), extensions (M2), and people writing JSON by hand. A mistake must be refused before anything runs, with a message saying where and what, not found halfway through a scan.
- **The stock algorithms exist** (P1.05–P1.07), each a function on the CPU and on CUDA. Some keep state between frames: temporal averaging keeps a window, and the bilateral filter its tables.
- **The pipeline engine was split in four** (P1.08a–d). This ADR fixes the model the other three build on: GPU residency and fusion (P1.08c), and tiling (P1.08d), change how a plan runs, not what a recipe says.

## Decision

1. **Port types:** a raw frame (a Bayer mosaic), an image (channels, and a sample type of `u8`, `u16` or `f32`), a scalar, or a table of named numbers. A node's input may leave an image's channels or sample type open, to take any. An output may *follow* an input, taking what it leaves open from what that input is given, so `tetera.crop` gives what it was given.
2. **A node is a descriptor and a factory:**
   - the `Descriptor` holds a stable dotted id (`tetera.crop`), a version, the ports, the parameters' JSON Schema, the backends, and whether it can fuse;
   - `Node::prepare` reads the parameters into the node's own typed struct, and gives a `Step`;
   - `Step::run` then runs on one set of inputs after another, and keeps any state between them.

   Parameters are checked by deserialising them into that struct, `deny_unknown_fields` on, so serde says precisely what is wrong; `schemars` makes the schema from the same struct, so the two can't drift.
3. **A recipe is JSON:**
   - its inputs, named and typed with nothing left open;
   - its nodes, each with an id, the registry's node id and an optional version, its parameters, and what feeds each input: a recipe input's name, or another node's `node.port`;
   - its outputs, each from a `node.port`.

   Feeding inputs from the node's side makes "an input fed twice" a key given twice, which the reader refuses. The schema comes from the same Rust types (`Recipe::schema`).
4. **Validation reports every problem, each once,** with where it is (`` node `stats`, input `image` ``) and what is wrong:
   - unknown nodes and versions, ports, and references;
   - unfed inputs;
   - types that don't match;
   - cycles, named node by node in the order data flows;
   - parameters that don't read;
   - outputs that come from nowhere.

   Types are checked in topological order, so each input is checked against what feeds it. A node fed by a problem isn't checked again, so one mistake gives one message.
5. **A `Plan`** is a recipe checked against a `Registry`: its nodes prepared, in an order where each runs after those that feed it, ties kept in the recipe's order. `Plan::run` checks the inputs it is given, runs each step, and checks that each node gives the types it declared.
6. **The registry** holds nodes by id and version. A recipe gets the latest version unless it names one, so old recipes keep working as nodes gain versions.

## Consequences

- **Gains:**
  - a recipe is checked whole before anything runs, and its problems read as a list to fix;
  - nodes are plain Rust types: their parameters, schema and checks come from one struct;
  - a stateful node is a `Step` like any other;
  - the run order is deterministic, so runs repeat exactly.
- **Costs:**
  - each stock node needs a wrapper of about 30 lines (P1.08b);
  - a typed parameter struct per node;
  - serde's message names a parameter but not its path in nested parameters. No stock node nests them.
- **To watch:**
  - values are kept until the run ends, which costs memory on long recipes of large images. P1.08c's residency work frees each when its last reader is done;
  - the type system has no variants beyond these four kinds, and a node that takes "a raw frame or an image" must be two nodes, or take an image.

## Alternatives considered

- **Edges as a separate list** (`from`, `to`): the format of many graph tools, and what a graph editor draws. Feeding each input from the node's side keeps a node's wiring next to its parameters for people writing JSON, and the editor (M4) can still draw edges from it.
- **Validating parameters against their JSON Schema with a validator crate:** a large dependency. And the node would still read the parameters into its own types, so there would be two checks to keep in step.
- **Dynamic values without declared types,** checked only when they run: simpler nodes, but mistakes found mid-run, which the plan rules out.
- **Stopping at the first problem:** simpler, but a recipe with three mistakes takes three tries.
