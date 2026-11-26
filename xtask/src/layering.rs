//! The layer rules of ADR-0005, checked against the workspace.
//!
//! Each `tetera-*` crate sits in a layer of plan §6.2. It may depend on crates in lower
//! layers, and on the few crates in its own layer that [`ALLOWED_WITHIN_LAYER`] names.
//! The rules hold for every kind of dependency: normal, dev and build.

use crate::metadata::Metadata;
use crate::toml_lite;
use std::path::Path;

/// Every planned crate and its layer, from L0 (foundation) to L6 (applications). A crate is
/// placed here before it exists, so the plan decides its place, not its code.
pub const LAYERS: &[(&str, u8)] = &[
    ("tetera-core", 0),
    ("tetera-compute", 1),
    ("tetera-log", 1),
    ("tetera-shm", 1),
    ("tetera-imaging", 1),
    ("tetera-hal", 2),
    ("tetera-sim", 2),
    ("tetera-grbl", 2),
    ("tetera-tucsen", 2),
    ("tetera-store", 2),
    ("tetera-jobs", 3),
    ("tetera-pipeline", 3),
    ("tetera-ext", 3),
    ("tetera-ext-sdk", 3),
    ("tetera-acquire", 4),
    ("tetera-workflows", 4),
    ("tetera-api", 5),
    ("tetera-server", 6),
];

/// The edges allowed within a layer, as (dependent, dependency). Each one the plan implies.
pub const ALLOWED_WITHIN_LAYER: &[(&str, &str)] = &[
    // Every algorithm has a CPU and a CUDA path (§6.6).
    ("tetera-imaging", "tetera-compute"),
    // Simulated and real devices implement the device traits (§6.8).
    ("tetera-sim", "tetera-hal"),
    ("tetera-grbl", "tetera-hal"),
    ("tetera-tucsen", "tetera-hal"),
    // Batch pipelines run as compute jobs (§6.7).
    ("tetera-pipeline", "tetera-jobs"),
    // Extensions add nodes, run as jobs, and are built against the SDK (§6.12).
    ("tetera-ext", "tetera-jobs"),
    ("tetera-ext", "tetera-pipeline"),
    ("tetera-ext", "tetera-ext-sdk"),
    // Workflows wait on the frame bus (§6.10).
    ("tetera-workflows", "tetera-acquire"),
];

/// The crates allowed unsafe code (plan §5.3). They set their own lints, not the workspace's.
pub const UNSAFE_ALLOWED: &[&str] = &[
    "tetera-compute",
    "tetera-shm",
    "tetera-ext",
    "tetera-ext-sdk",
    "tetera-tucsen",
];

/// The only crate that may depend on `teta-wot` (ADR-0004).
pub const API_CRATE: &str = "tetera-api";

/// How a crate sets its lints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lints {
    /// `[lints]` with `workspace = true`.
    Workspace,
    /// Its own `[lints.rust]` and `[lints.clippy]`.
    Own {
        /// Whether `unsafe_code` is allowed.
        unsafe_allowed: bool,
        /// Whether clippy's `undocumented_unsafe_blocks` is denied.
        undocumented_unsafe_denied: bool,
    },
    /// Neither.
    Missing,
}

/// A dependency, as the rules see it.
#[derive(Debug, Clone)]
pub struct Dep {
    /// The package depended on.
    pub name: String,
    /// `normal`, `dev` or `build`.
    pub kind: String,
    /// Whether it is a workspace member.
    pub member: bool,
}

/// A workspace member, as the rules see it.
#[derive(Debug, Clone)]
pub struct Crate {
    /// Its name.
    pub name: String,
    /// Its dependencies, of every kind.
    pub deps: Vec<Dep>,
    /// How it sets its lints.
    pub lints: Lints,
}

/// The layer the table places `name` in.
pub fn layer_of(name: &str) -> Option<u8> {
    LAYERS.iter().find(|(n, _)| *n == name).map(|(_, l)| *l)
}

fn is_layered(name: &str) -> bool {
    name.starts_with("tetera-")
}

/// Checks `crates` against the rules. Each broken rule is reported in one sentence that
/// names the crate, and the edge where there is one.
pub fn check(crates: &[Crate], workspace_forbids_unsafe: bool) -> Vec<String> {
    let mut found = Vec::new();
    if !workspace_forbids_unsafe {
        found.push(
            "The workspace's Cargo.toml must set `unsafe_code = \"forbid\"` under \
             `[workspace.lints.rust]`."
                .to_string(),
        );
    }
    for krate in crates {
        check_lints(krate, &mut found);
        if !is_layered(&krate.name) {
            continue;
        }
        check_teta_wot(krate, &mut found);
        let Some(layer) = layer_of(&krate.name) else {
            found.push(format!(
                "`{}` isn't in the layer table in xtask/src/layering.rs: place it in a layer first.",
                krate.name
            ));
            continue;
        };
        for dep in krate.deps.iter().filter(|d| d.member) {
            check_edge(&krate.name, layer, dep, &mut found);
        }
    }
    found
}

fn check_lints(krate: &Crate, found: &mut Vec<String>) {
    if UNSAFE_ALLOWED.contains(&krate.name.as_str()) {
        let own = Lints::Own {
            unsafe_allowed: true,
            undocumented_unsafe_denied: true,
        };
        if krate.lints != own {
            found.push(format!(
                "`{}` may use unsafe code, so it sets its own lints: `unsafe_code = \"allow\"` under \
                 `[lints.rust]`, and `undocumented_unsafe_blocks = \"deny\"` under `[lints.clippy]`.",
                krate.name
            ));
        }
    } else if krate.lints != Lints::Workspace {
        found.push(format!(
            "`{}` must take the workspace's lints: add `[lints]` with `workspace = true` to its Cargo.toml.",
            krate.name
        ));
    }
}

fn check_teta_wot(krate: &Crate, found: &mut Vec<String>) {
    if krate.name == API_CRATE {
        return;
    }
    for dep in &krate.deps {
        if dep.name == "teta-wot" || dep.name.starts_with("teta-wot-") {
            found.push(format!(
                "`{}` depends on `{}`: only `{API_CRATE}` may (ADR-0004).",
                krate.name, dep.name
            ));
        }
    }
}

fn check_edge(from: &str, layer: u8, dep: &Dep, found: &mut Vec<String>) {
    let to = dep.name.as_str();
    let kind = match dep.kind.as_str() {
        "dev" => " (a dev-dependency)",
        "build" => " (a build-dependency)",
        _ => "",
    };
    match layer_of(to) {
        Some(to_layer) if to_layer < layer => {}
        Some(to_layer) if to_layer == layer => {
            if !ALLOWED_WITHIN_LAYER.contains(&(from, to)) {
                found.push(format!(
                    "`{from}` (L{layer}) depends on `{to}` (L{to_layer}){kind}: an edge within a layer \
                     that the table doesn't allow."
                ));
            }
        }
        Some(to_layer) => found.push(format!(
            "`{from}` (L{layer}) depends on `{to}` (L{to_layer}){kind}, a higher layer."
        )),
        // An unplaced `tetera-*` crate is reported where that crate is checked.
        None if is_layered(to) => {}
        None => found.push(format!(
            "`{from}` depends on `{to}`{kind}, which isn't a layered crate: layered crates depend \
             only on layered crates."
        )),
    }
}

/// How a manifest sets its lints, read from its text.
pub fn lints_of(manifest: &str) -> Lints {
    let s = toml_lite::sections(manifest);
    if toml_lite::value(&s, "lints", "workspace") == Some("true") {
        return Lints::Workspace;
    }
    if !s.contains_key("lints.rust") && !s.contains_key("lints.clippy") {
        return Lints::Missing;
    }
    let unsafe_allowed =
        toml_lite::value(&s, "lints.rust", "unsafe_code").is_some_and(|v| v.contains("allow"));
    let undocumented_unsafe_denied =
        toml_lite::value(&s, "lints.clippy", "undocumented_unsafe_blocks")
            .is_some_and(|v| v.contains("deny") || v.contains("forbid"));
    Lints::Own {
        unsafe_allowed,
        undocumented_unsafe_denied,
    }
}

/// Whether the workspace's manifest forbids unsafe code for the crates that take its lints.
pub fn workspace_forbids_unsafe(manifest: &str) -> bool {
    let s = toml_lite::sections(manifest);
    toml_lite::value(&s, "workspace.lints.rust", "unsafe_code")
        .is_some_and(|v| v.contains("forbid"))
}

/// Checks the workspace that holds `dir`: its members, from `cargo metadata`, and their manifests.
pub fn check_workspace(dir: &Path) -> Result<Vec<String>, String> {
    let meta = Metadata::load(dir)?;
    let members = meta.member_names();
    let read = |path: &Path| {
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))
    };
    let mut crates = Vec::new();
    for p in &meta.packages {
        crates.push(Crate {
            name: p.name.clone(),
            lints: lints_of(&read(&p.manifest_path)?),
            deps: p
                .dependencies
                .iter()
                .map(|d| Dep {
                    name: d.name.clone(),
                    kind: d.kind.clone().unwrap_or_else(|| "normal".to_string()),
                    member: d.source.is_none() && members.contains(d.name.as_str()),
                })
                .collect(),
        });
    }
    let root_manifest = read(&meta.workspace_root.join("Cargo.toml"))?;
    Ok(check(&crates, workspace_forbids_unsafe(&root_manifest)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn krate(name: &str, deps: &[(&str, &str)]) -> Crate {
        Crate {
            name: name.to_string(),
            deps: deps
                .iter()
                .map(|(dep, kind)| Dep {
                    name: dep.to_string(),
                    kind: kind.to_string(),
                    member: dep.starts_with("tetera-") || *dep == "xtask",
                })
                .collect(),
            lints: if UNSAFE_ALLOWED.contains(&name) {
                Lints::Own {
                    unsafe_allowed: true,
                    undocumented_unsafe_denied: true,
                }
            } else {
                Lints::Workspace
            },
        }
    }

    fn one(crates: &[Crate]) -> String {
        let found = check(crates, true);
        assert_eq!(found.len(), 1, "{found:#?}");
        found[0].clone()
    }

    #[test]
    fn edges_to_lower_layers_and_allowed_edges_within_a_layer_pass() {
        let crates = [
            krate("tetera-core", &[]),
            krate("tetera-compute", &[("tetera-core", "normal")]),
            krate(
                "tetera-imaging",
                &[("tetera-compute", "normal"), ("tetera-core", "dev")],
            ),
            krate(
                "tetera-api",
                &[("tetera-core", "normal"), ("teta-wot", "normal")],
            ),
            krate("xtask", &[("tetera-core", "normal"), ("serde", "normal")]),
        ];
        assert_eq!(check(&crates, true), Vec::<String>::new());
    }

    #[test]
    fn an_upward_edge_is_named() {
        let found = one(&[krate("tetera-core", &[("tetera-hal", "normal")])]);
        assert_eq!(
            found,
            "`tetera-core` (L0) depends on `tetera-hal` (L2), a higher layer."
        );
    }

    #[test]
    fn an_upward_dev_dependency_is_named_as_one() {
        let found = one(&[krate("tetera-compute", &[("tetera-api", "dev")])]);
        assert!(
            found.contains("(a dev-dependency), a higher layer"),
            "{found}"
        );
    }

    #[test]
    fn an_edge_within_a_layer_needs_the_tables_permission() {
        let found = one(&[krate("tetera-compute", &[("tetera-imaging", "normal")])]);
        assert!(found.contains("an edge within a layer"), "{found}");
    }

    #[test]
    fn teta_wot_belongs_to_the_api_crate_alone() {
        let found = one(&[krate("tetera-acquire", &[("teta-wot", "normal")])]);
        assert_eq!(
            found,
            "`tetera-acquire` depends on `teta-wot`: only `tetera-api` may (ADR-0004)."
        );
    }

    #[test]
    fn a_layered_crate_may_not_depend_on_tools_or_examples() {
        let found = one(&[krate("tetera-core", &[("xtask", "dev")])]);
        assert!(found.contains("isn't a layered crate"), "{found}");
    }

    #[test]
    fn an_unplaced_crate_must_be_placed_first() {
        let found = one(&[krate("tetera-gadget", &[])]);
        assert!(found.contains("isn't in the layer table"), "{found}");
    }

    #[test]
    fn an_edge_to_an_unplaced_crate_is_reported_once_at_that_crate() {
        let found = one(&[
            krate("tetera-core", &[("tetera-gadget", "normal")]),
            krate("tetera-gadget", &[]),
        ]);
        assert!(
            found.starts_with("`tetera-gadget` isn't in the layer table"),
            "{found}"
        );
    }

    #[test]
    fn an_upward_build_dependency_is_named_as_one() {
        let found = one(&[krate("tetera-core", &[("tetera-compute", "build")])]);
        assert!(
            found.contains("(a build-dependency), a higher layer"),
            "{found}"
        );
    }

    #[test]
    fn rust_lints_alone_are_a_crates_own_lints() {
        assert_eq!(
            lints_of("[lints.rust]\nunsafe_code = \"allow\"\n"),
            Lints::Own {
                unsafe_allowed: true,
                undocumented_unsafe_denied: false
            }
        );
        assert_eq!(
            lints_of("[lints.clippy]\nundocumented_unsafe_blocks = \"deny\"\n"),
            Lints::Own {
                unsafe_allowed: false,
                undocumented_unsafe_denied: true
            }
        );
    }

    #[test]
    fn every_crate_takes_the_workspaces_lints_unless_allowed_unsafe_code() {
        let mut core = krate("tetera-core", &[]);
        core.lints = Lints::Missing;
        assert!(one(&[core]).contains("must take the workspace's lints"));

        let mut compute = krate("tetera-compute", &[]);
        compute.lints = Lints::Own {
            unsafe_allowed: true,
            undocumented_unsafe_denied: false,
        };
        assert!(one(&[compute]).contains("sets its own lints"));

        let mut compute = krate("tetera-compute", &[]);
        compute.lints = Lints::Workspace;
        assert!(one(&[compute]).contains("sets its own lints"));
    }

    #[test]
    fn the_workspace_must_forbid_unsafe_code() {
        let found = check(&[krate("tetera-core", &[])], false);
        assert_eq!(found.len(), 1);
        assert!(found[0].contains("unsafe_code = \"forbid\""));
    }

    #[test]
    fn lints_are_read_from_the_manifest() {
        assert_eq!(
            lints_of("[package]\nname = \"a\"\n[lints]\nworkspace = true\n"),
            Lints::Workspace
        );
        assert_eq!(lints_of("[package]\nname = \"a\"\n"), Lints::Missing);
        assert_eq!(
            lints_of(
                "[lints.rust]\nunsafe_code = \"allow\"\n[lints.clippy]\nundocumented_unsafe_blocks = \"deny\"\n"
            ),
            Lints::Own {
                unsafe_allowed: true,
                undocumented_unsafe_denied: true
            }
        );
        assert!(workspace_forbids_unsafe(
            "[workspace.lints.rust]\nunsafe_code = \"forbid\"\n"
        ));
        assert!(!workspace_forbids_unsafe(
            "[workspace.lints.rust]\nunsafe_code = \"warn\"\n"
        ));
    }

    #[test]
    fn every_allowed_edge_stays_within_one_layer() {
        for (from, to) in ALLOWED_WITHIN_LAYER {
            assert_eq!(layer_of(from), layer_of(to), "{from} -> {to}");
            assert!(layer_of(from).is_some(), "{from} isn't placed");
        }
        for name in UNSAFE_ALLOWED {
            assert!(layer_of(name).is_some(), "{name} isn't placed");
        }
    }
}
