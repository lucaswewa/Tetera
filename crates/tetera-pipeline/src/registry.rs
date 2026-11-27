//! The registry: every node a recipe can name, by its stable id and version.

use crate::node::{Descriptor, Node};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use tetera_core::{Error, Result};

/// Nodes by id, then version.
#[derive(Clone, Default)]
pub struct Registry {
    nodes: BTreeMap<&'static str, BTreeMap<u32, Arc<dyn Node>>>,
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let all: Vec<String> = self
            .descriptors()
            .map(|d| format!("{} {}", d.id, d.version))
            .collect();
        f.debug_struct("Registry").field("nodes", &all).finish()
    }
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Tetera's own nodes.
    pub fn stock() -> Self {
        let mut registry = Self::new();
        for node in crate::glue::nodes()
            .into_iter()
            .chain(crate::stock::nodes())
        {
            registry
                .register(node)
                .expect("the stock nodes are sound and distinct");
        }
        registry
    }

    /// `node`, added, if its descriptor is sound and its id and version are new.
    pub fn register(&mut self, node: Arc<dyn Node>) -> Result<()> {
        let d = node.descriptor();
        d.check()?;
        let versions = self.nodes.entry(d.id).or_default();
        if versions.contains_key(&d.version) {
            return Err(Error::invalid_argument(format!(
                "`{}` version {} is registered already",
                d.id, d.version
            )));
        }
        versions.insert(d.version, node);
        Ok(())
    }

    /// The node `id` at `version`, or its latest when that is `None`; or why there is none.
    pub fn find(
        &self,
        id: &str,
        version: Option<u32>,
    ) -> std::result::Result<Arc<dyn Node>, String> {
        let Some(versions) = self.nodes.get(id) else {
            return Err(format!("no node `{id}` is registered"));
        };
        let found = match version {
            Some(v) => versions.get(&v),
            None => versions.values().next_back(),
        };
        found.cloned().ok_or_else(|| {
            let have: Vec<String> = versions.keys().map(u32::to_string).collect();
            format!(
                "`{id}` has no version {}; the registry has {}",
                version.unwrap_or_default(),
                have.join(", ")
            )
        })
    }

    /// Every node's descriptor, by id, then version.
    pub fn descriptors(&self) -> impl Iterator<Item = &Descriptor> {
        self.nodes
            .values()
            .flat_map(|v| v.values())
            .map(|n| n.descriptor())
    }

    /// The catalogue: every descriptor, as JSON.
    pub fn catalogue(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.descriptors()
                .map(|d| serde_json::to_value(d).expect("a descriptor is JSON"))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Context, Step};
    use crate::port::Value;

    /// A node that only declares itself.
    struct Bare(Descriptor);

    impl Node for Bare {
        fn descriptor(&self) -> &Descriptor {
            &self.0
        }

        fn prepare(&self, _: &serde_json::Value) -> Result<Box<dyn Step>> {
            Err(Error::internal("not run"))
        }
    }

    impl Step for Bare {
        fn run(&mut self, _: &Context, _: &[&Value]) -> Result<Vec<Value>> {
            Ok(Vec::new())
        }
    }

    fn bare(id: &'static str, version: u32) -> Arc<dyn Node> {
        Arc::new(Bare(Descriptor::new(id, version, "nothing")))
    }

    #[test]
    fn the_stock_registry_holds_the_glue_and_raw_path_nodes() {
        let registry = Registry::stock();
        let ids: Vec<(&str, u32)> = registry.descriptors().map(|d| (d.id, d.version)).collect();
        let expected = [
            "tetera.bayer_split",
            "tetera.bin",
            "tetera.colour",
            "tetera.crop",
            "tetera.dark",
            "tetera.demosaic",
            "tetera.display",
            "tetera.flat",
            "tetera.jpeg",
            "tetera.resize",
            "tetera.statistics",
            "tetera.superpixel",
        ];
        assert_eq!(ids, expected.map(|id| (id, 1)));
        let catalogue = registry.catalogue();
        let entry = |id: &str| {
            catalogue
                .as_array()
                .unwrap()
                .iter()
                .find(|d| d["id"] == id)
                .unwrap()
                .clone()
        };
        assert_eq!(
            entry("tetera.crop")["parameters"]["required"],
            serde_json::json!(["x", "y", "width", "height"])
        );
        assert_eq!(
            entry("tetera.statistics")["outputs"][0]["type"]["kind"],
            "table"
        );
        assert_eq!(entry("tetera.demosaic")["backends"]["cuda"], true);
        assert_eq!(
            entry("tetera.jpeg")["backends"]["cuda"],
            false,
            "libjpeg-turbo is on the CPU"
        );
        assert_eq!(entry("tetera.crop")["backends"]["cuda"], false);
        let debug = format!("{registry:?}");
        assert!(debug.starts_with("Registry { nodes: [\"tetera.bayer_split 1\", \"tetera.bin 1\""));
        assert_eq!(Registry::new().descriptors().count(), 0);
    }

    #[test]
    fn nodes_are_found_by_id_and_version_the_latest_by_default() {
        let mut registry = Registry::new();
        registry.register(bare("test.node", 2)).unwrap();
        registry.register(bare("test.node", 1)).unwrap();
        registry.register(bare("test.node", 10)).unwrap();
        let version = |v| registry.find("test.node", v).unwrap().descriptor().version;
        assert_eq!(
            (version(None), version(Some(1)), version(Some(2))),
            (10, 1, 2)
        );
        assert_eq!(
            registry.find("test.node", Some(3)).err().unwrap(),
            "`test.node` has no version 3; the registry has 1, 2, 10"
        );
        assert_eq!(
            registry.find("test.other", None).err().unwrap(),
            "no node `test.other` is registered"
        );
    }

    #[test]
    fn a_node_registered_twice_or_unsound_is_refused() {
        let mut registry = Registry::stock();
        assert_eq!(
            registry
                .register(bare("tetera.crop", 1))
                .unwrap_err()
                .to_string(),
            "`tetera.crop` version 1 is registered already"
        );
        assert!(registry.register(bare("tetera.crop", 2)).is_ok());
        assert_eq!(
            registry
                .register(bare("NoDots", 1))
                .unwrap_err()
                .to_string(),
            "node `NoDots`: an id is lower-case words joined by dots, such as `tetera.crop`"
        );
        assert!(registry.find("NoDots", None).is_err(), "nothing kept of it");
    }
}
