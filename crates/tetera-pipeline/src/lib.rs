//! Tetera's pipeline engine, L3 (plan §6.7, ADR-0012): image processing as a graph of nodes
//! with typed ports, written as a JSON recipe, checked before it runs, and run in order.
//!
//! From P1.08a:
//! - [`port`]: what a port carries (a raw frame, an image, a scalar, a table) and the values;
//! - [`node`]: a node's [`Descriptor`], and the [`Node`] and [`Step`] traits: a node is
//!   prepared once with its parameters, then run on one set of inputs after another;
//! - [`recipe`]: recipes as JSON, with a schema;
//! - [`plan`]: a recipe checked against the [`Registry`], each problem said precisely, its
//!   nodes in the order they run; and running it;
//! - [`registry`]: the nodes by stable id and version;
//! - [`glue`]: the first nodes, crop and statistics.
//!
//! From P1.08b:
//! - [`port`] carries encoded images too, JPEG first;
//! - [`stock`]: the raw path's nodes, from binning to JPEG, each on the backend chosen; and
//!   the live chain as a recipe, `recipes/live-chain.json`.

pub mod glue;
pub mod node;
pub mod plan;
pub mod port;
pub mod recipe;
pub mod registry;
pub mod stock;

pub use node::{Backends, Context, Descriptor, Node, Step};
pub use plan::{Plan, Problem};
pub use port::{Encoding, PortSpec, PortType, SampleType, Table, Value};
pub use recipe::{NodeUse, Recipe};
pub use registry::Registry;
