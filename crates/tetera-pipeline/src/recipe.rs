//! Recipes: a graph of nodes as JSON, with a schema. A recipe names its inputs and their
//! types, its nodes, each fed from the recipe's inputs or other nodes' outputs, and its
//! outputs. A reference is a recipe input's name, or `node.port` for a node's output.
//!
//! ```json
//! {
//!   "name": "crop and measure",
//!   "inputs": { "frame": { "kind": "image", "channels": 3, "sample": "u16" } },
//!   "nodes": [
//!     { "id": "crop", "node": "tetera.crop",
//!       "params": { "x": 0, "y": 0, "width": 64, "height": 64 },
//!       "inputs": { "image": "frame" } },
//!     { "id": "stats", "node": "tetera.statistics", "inputs": { "image": "crop.image" } }
//!   ],
//!   "outputs": { "statistics": "stats.table" }
//! }
//! ```

use crate::port::PortType;
use schemars::JsonSchema;
use serde::de::{Deserializer, Error as _, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use tetera_core::{Error, Result};

/// A recipe, as its JSON reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// Its name.
    pub name: String,
    /// What it does.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
    /// What it takes, by name, each of a type with nothing left open.
    #[serde(default, deserialize_with = "unique_keys")]
    pub inputs: BTreeMap<String, PortType>,
    /// Its nodes, in any order.
    pub nodes: Vec<NodeUse>,
    /// What it gives, by name, each from a node's output: `node.port`.
    #[serde(deserialize_with = "unique_keys")]
    pub outputs: BTreeMap<String, String>,
}

/// A node in a recipe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NodeUse {
    /// Its id in the recipe, which references to its outputs use.
    pub id: String,
    /// The registry's id for the node, such as `tetera.crop`.
    pub node: String,
    /// The node's version; the latest when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    /// Its parameters, as its schema says.
    #[serde(default = "empty_object")]
    pub params: serde_json::Value,
    /// What feeds each of its inputs: a recipe input's name, or another node's `node.port`.
    #[serde(default, deserialize_with = "unique_keys")]
    pub inputs: BTreeMap<String, String>,
}

fn empty_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

impl Recipe {
    /// A recipe from its JSON, or where and why the JSON doesn't make one.
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text)
            .map_err(|e| Error::invalid_argument(format!("not a recipe: {e}")))
    }

    /// It as JSON, laid out for people.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a recipe is JSON")
    }

    /// The JSON Schema every recipe follows.
    pub fn schema() -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(Recipe)).expect("a schema is JSON")
    }
}

/// A map whose keys must each appear once: JSON allows a key twice, and would keep the last.
fn unique_keys<'de, D, V>(deserializer: D) -> std::result::Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct Unique<V>(PhantomData<V>);

    impl<'de, V: Deserialize<'de>> Visitor<'de> for Unique<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a map of names")
        }

        fn visit_map<A: MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, V>()? {
                if out.contains_key(&key) {
                    return Err(A::Error::custom(format!("`{key}` is given twice")));
                }
                out.insert(key, value);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(Unique(PhantomData))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::SampleType;

    const EXAMPLE: &str = r#"{
  "name": "crop and measure",
  "inputs": { "frame": { "kind": "image", "channels": 3, "sample": "u16" } },
  "nodes": [
    { "id": "crop", "node": "tetera.crop",
      "params": { "x": 0, "y": 0, "width": 64, "height": 64 },
      "inputs": { "image": "frame" } },
    { "id": "stats", "node": "tetera.statistics", "version": 1, "inputs": { "image": "crop.image" } }
  ],
  "outputs": { "statistics": "stats.table" }
}"#;

    #[test]
    fn a_recipe_reads_from_json_and_back() {
        let recipe = Recipe::from_json(EXAMPLE).unwrap();
        assert_eq!(recipe.name, "crop and measure");
        assert_eq!(recipe.inputs["frame"], PortType::image(3, SampleType::U16));
        assert_eq!(recipe.nodes.len(), 2);
        assert_eq!(recipe.nodes[0].params["width"], 64);
        assert_eq!(recipe.nodes[0].version, None);
        assert_eq!(recipe.nodes[1].version, Some(1));
        assert_eq!(
            recipe.nodes[1].params,
            serde_json::json!({}),
            "no parameters, an empty object"
        );
        assert_eq!(recipe.nodes[1].inputs["image"], "crop.image");
        assert_eq!(recipe.outputs["statistics"], "stats.table");
        assert_eq!(Recipe::from_json(&recipe.to_json()).unwrap(), recipe);
        assert!(
            !recipe.to_json().contains("summary"),
            "an empty summary is left out"
        );
    }

    #[test]
    fn json_that_makes_no_recipe_says_where_and_why() {
        let err = |text: &str| Recipe::from_json(text).unwrap_err().to_string();
        assert_eq!(
            err(&EXAMPLE.replace("\"nodes\"", "\"nodez\"")),
            "not a recipe: unknown field `nodez`, expected one of `name`, `summary`, `inputs`, `nodes`, `outputs` at line 4 column 9"
        );
        assert!(
            err(&EXAMPLE.replace("\"node\": \"tetera.crop\",", ""))
                .contains("missing field `node`")
        );
        assert!(
            err(&EXAMPLE.replace("\"kind\": \"image\"", "\"kind\": \"picture\""))
                .contains("unknown variant `picture`")
        );
        let twice = EXAMPLE.replace(
            r#""inputs": { "image": "frame" }"#,
            r#""inputs": { "image": "frame", "image": "frame" }"#,
        );
        assert_eq!(
            err(&twice),
            "not a recipe: `image` is given twice at line 7 column 54"
        );
        assert!(
            err(r#"{"name": "x", "nodes": [], "outputs": {"a": "b", "a": "c"}}"#)
                .contains("`a` is given twice")
        );
        assert!(
            err(r#"{"name": "x", "nodes": [], "outputs": []}"#).contains("expected a map of names")
        );
        assert!(err("{").starts_with("not a recipe: EOF while parsing"));
    }

    #[test]
    fn the_schema_describes_every_field() {
        let schema = Recipe::schema();
        let text = schema.to_string();
        for field in [
            "name", "summary", "inputs", "nodes", "outputs", "params", "version", "kind", "u16",
        ] {
            assert!(text.contains(&format!("\"{field}\"")), "{field}");
        }
        assert_eq!(schema["title"], "Recipe");
        assert_eq!(
            schema["required"],
            serde_json::json!(["name", "nodes", "outputs"])
        );
    }
}
