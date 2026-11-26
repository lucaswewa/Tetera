//! The parts of `cargo metadata` that the tasks read.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The workspace, from `cargo metadata --no-deps`: its members and folders.
#[derive(Debug, Deserialize)]
pub struct Metadata {
    /// The workspace's members.
    pub packages: Vec<Package>,
    /// The folder holding the workspace's `Cargo.toml`.
    pub workspace_root: PathBuf,
    /// Where builds go: `target/`, or `CARGO_TARGET_DIR`.
    pub target_directory: PathBuf,
}

/// A workspace member.
#[derive(Debug, Clone, Deserialize)]
pub struct Package {
    /// Its name.
    pub name: String,
    /// Cargo's identifier for it, as build messages give it.
    pub id: String,
    /// Its `Cargo.toml`.
    pub manifest_path: PathBuf,
    /// What it depends on, of every kind.
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    /// Its library, binaries, examples, tests and benchmarks.
    #[serde(default)]
    pub targets: Vec<Target>,
    /// Its features, by name.
    #[serde(default)]
    pub features: BTreeMap<String, Vec<String>>,
}

/// One dependency of a member.
#[derive(Debug, Clone, Deserialize)]
pub struct Dependency {
    /// The package depended on.
    pub name: String,
    /// Where it comes from: absent for a path dependency.
    pub source: Option<String>,
    /// `dev` or `build`; absent for a normal dependency.
    pub kind: Option<String>,
}

/// One build target of a member.
#[derive(Debug, Clone, Deserialize)]
pub struct Target {
    /// Its name.
    pub name: String,
    /// Its kinds: `lib`, `bin`, `example`, `test`, and so on.
    pub kind: Vec<String>,
}

const LIBRARY_KINDS: &[&str] = &["lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"];

impl Package {
    /// The folder holding its `Cargo.toml`.
    pub fn dir(&self) -> &Path {
        self.manifest_path.parent().unwrap_or(Path::new("."))
    }

    /// Whether it has a library.
    pub fn has_lib(&self) -> bool {
        self.targets
            .iter()
            .any(|t| t.kind.iter().any(|k| LIBRARY_KINDS.contains(&k.as_str())))
    }

    /// Whether it has binaries.
    pub fn has_bins(&self) -> bool {
        self.targets_of("bin").next().is_some()
    }

    /// Its examples' names.
    pub fn examples(&self) -> Vec<String> {
        self.targets_of("example").map(|t| t.name.clone()).collect()
    }

    fn targets_of<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Target> + 'a {
        self.targets
            .iter()
            .filter(move |t| t.kind.iter().any(|k| k == kind))
    }
}

impl Metadata {
    /// Runs `cargo metadata` in `dir`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let out = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--no-deps"])
            .current_dir(dir)
            .output()
            .map_err(|e| format!("cargo metadata didn't start: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "cargo metadata failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        serde_json::from_slice(&out.stdout).map_err(|e| format!("cargo metadata's output: {e}"))
    }

    /// The members' names.
    pub fn member_names(&self) -> BTreeSet<&str> {
        self.packages.iter().map(|p| p.name.as_str()).collect()
    }

    /// The path dependencies between members, as (dependent, dependency), of every kind.
    pub fn member_edges(&self) -> Vec<(String, String)> {
        let members = self.member_names();
        let mut edges = Vec::new();
        for p in &self.packages {
            for d in &p.dependencies {
                if d.source.is_none() && members.contains(d.name.as_str()) {
                    edges.push((p.name.clone(), d.name.clone()));
                }
            }
        }
        edges
    }

    /// The member with Cargo's identifier `id`.
    pub fn package_by_id(&self, id: &str) -> Option<&Package> {
        self.packages.iter().find(|p| p.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(name: &str, kind: &str) -> Target {
        Target {
            name: name.into(),
            kind: vec![kind.into()],
        }
    }

    fn dep(name: &str, source: Option<&str>) -> Dependency {
        Dependency {
            name: name.into(),
            source: source.map(Into::into),
            kind: None,
        }
    }

    fn package(name: &str, targets: Vec<Target>, dependencies: Vec<Dependency>) -> Package {
        Package {
            name: name.into(),
            id: format!("id-{name}"),
            manifest_path: format!("/ws/{name}/Cargo.toml").into(),
            dependencies,
            targets,
            features: BTreeMap::new(),
        }
    }

    #[test]
    fn a_packages_targets_say_what_it_holds() {
        let lib_only = package("a", vec![target("a", "lib"), target("it", "test")], vec![]);
        assert!(lib_only.has_lib() && !lib_only.has_bins());
        assert!(lib_only.examples().is_empty());

        let macros = package("m", vec![target("m", "proc-macro")], vec![]);
        assert!(macros.has_lib());

        let tool = package(
            "t",
            vec![
                target("t", "bin"),
                target("one", "example"),
                target("two", "example"),
            ],
            vec![],
        );
        assert!(!tool.has_lib() && tool.has_bins());
        assert_eq!(tool.examples(), ["one", "two"]);
        assert_eq!(tool.dir(), Path::new("/ws/t"));
    }

    #[test]
    fn member_edges_are_path_dependencies_on_members() {
        let meta = Metadata {
            packages: vec![
                package(
                    "core",
                    vec![],
                    vec![dep("serde", Some("registry+https://x"))],
                ),
                package(
                    "app",
                    vec![],
                    vec![
                        dep("core", None),
                        dep("helper", None),
                        dep("core-git", Some("git+https://x")),
                    ],
                ),
                package("core-git", vec![], vec![]),
            ],
            workspace_root: "/ws".into(),
            target_directory: "/ws/target".into(),
        };
        assert_eq!(
            meta.member_edges(),
            [("app".to_string(), "core".to_string())]
        );
        assert_eq!(
            meta.package_by_id("id-app").map(|p| p.name.as_str()),
            Some("app")
        );
        assert!(meta.package_by_id("id-none").is_none());
    }
}
