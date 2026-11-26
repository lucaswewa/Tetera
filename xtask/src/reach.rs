//! What a change reaches (ADR-0005): the packages that hold changed files and, for the
//! phase gate, every package that depends on them. Some files reach every check.

use crate::exec;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

/// Files and folders whose change reaches every check: the build's own configuration, and
/// the gate's code. A folder ends in `/`.
pub const REACHES_EVERYTHING: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "rustfmt.toml",
    "deny.toml",
    ".cargo/",
    "xtask/",
];

/// A workspace member and its folder, as reach sees them.
#[derive(Debug, Clone)]
pub struct PackageDir {
    /// The member's name.
    pub name: String,
    /// Its folder, relative to the workspace's root, with `/` between parts.
    pub dir: String,
}

/// What a change reaches.
#[derive(Debug, Default, Clone)]
pub struct Reach {
    /// Why every check runs, when the change reaches everything.
    pub everything: Option<String>,
    /// The packages reached, each with why.
    pub packages: BTreeMap<String, String>,
}

/// Works out what `changed` reaches. `edges` are the members' dependencies as (dependent,
/// dependency). With `dependents`, the packages that depend on a reached one, directly or
/// not, are reached too.
pub fn compute(
    changed: &[String],
    packages: &[PackageDir],
    edges: &[(String, String)],
    dependents: bool,
) -> Reach {
    let mut reach = Reach::default();
    for path in changed {
        let rule = REACHES_EVERYTHING
            .iter()
            .find(|rule| match rule.strip_suffix('/') {
                Some(dir) => path.starts_with(&format!("{dir}/")),
                None => path == *rule,
            });
        if let Some(rule) = rule {
            reach.everything.get_or_insert_with(|| {
                format!("`{path}` changed, and `{rule}` reaches every check")
            });
        } else if let Some(owner) = owner_of(path, packages) {
            reach
                .packages
                .entry(owner.to_string())
                .or_insert_with(|| format!("changed: `{path}`"));
        }
    }
    if reach.everything.is_some() {
        reach.packages = packages
            .iter()
            .map(|p| (p.name.clone(), "everything is reached".to_string()))
            .collect();
        return reach;
    }
    if dependents {
        let mut queue: VecDeque<String> = reach.packages.keys().cloned().collect();
        while let Some(reached) = queue.pop_front() {
            for (dependent, dependency) in edges {
                if *dependency == reached && !reach.packages.contains_key(dependent) {
                    reach
                        .packages
                        .insert(dependent.clone(), format!("depends on `{reached}`"));
                    queue.push_back(dependent.clone());
                }
            }
        }
    }
    reach
}

/// The member whose folder holds `path`: the deepest, so a crate nested in another's folder wins.
fn owner_of<'a>(path: &str, packages: &'a [PackageDir]) -> Option<&'a str> {
    packages
        .iter()
        .filter(|p| p.dir.is_empty() || path.starts_with(&format!("{}/", p.dir)))
        .max_by_key(|p| p.dir.len())
        .map(|p| p.name.as_str())
}

/// The files changed since `base`'s merge base with `HEAD`, committed or not, and untracked
/// files; with the merge base itself.
pub fn changed_files(root: &Path, base: &str) -> Result<(String, Vec<String>), String> {
    let git = |args: &[&str]| exec::output("git", args, Some(root));
    let merge_base = git(&["merge-base", base, "HEAD"]).ok_or_else(|| {
        format!("git finds no merge base between `{base}` and HEAD: pass another `--base`")
    })?;
    let tracked = git(&["diff", "--name-only", &merge_base]).ok_or("git diff failed")?;
    let untracked =
        git(&["ls-files", "--others", "--exclude-standard"]).ok_or("git ls-files failed")?;
    let files: BTreeSet<String> = tracked
        .lines()
        .chain(untracked.lines())
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    Ok((merge_base, files.into_iter().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packages() -> Vec<PackageDir> {
        [
            ("tetera-core", "crates/tetera-core"),
            ("tetera-imaging", "crates/tetera-imaging"),
            ("tetera-api", "crates/tetera-api"),
            ("ex-nested", "crates/tetera-api/examples/nested"),
            ("xtask", "xtask"),
        ]
        .iter()
        .map(|(name, dir)| PackageDir {
            name: name.to_string(),
            dir: dir.to_string(),
        })
        .collect()
    }

    fn edges() -> Vec<(String, String)> {
        [
            ("tetera-imaging", "tetera-core"),
            ("tetera-api", "tetera-imaging"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
    }

    fn reached(changed: &[&str], dependents: bool) -> Reach {
        let changed: Vec<String> = changed.iter().map(|s| s.to_string()).collect();
        compute(&changed, &packages(), &edges(), dependents)
    }

    #[test]
    fn a_leaf_change_reaches_its_own_package() {
        let r = reached(&["crates/tetera-api/src/lib.rs"], true);
        assert_eq!(r.packages.keys().collect::<Vec<_>>(), ["tetera-api"]);
        assert_eq!(
            r.packages["tetera-api"],
            "changed: `crates/tetera-api/src/lib.rs`"
        );
        assert!(r.everything.is_none());
    }

    #[test]
    fn a_core_change_reaches_every_dependent_with_why() {
        let r = reached(&["crates/tetera-core/src/lib.rs"], true);
        assert_eq!(
            r.packages.keys().collect::<Vec<_>>(),
            ["tetera-api", "tetera-core", "tetera-imaging"]
        );
        assert_eq!(r.packages["tetera-imaging"], "depends on `tetera-core`");
        assert_eq!(r.packages["tetera-api"], "depends on `tetera-imaging`");
    }

    #[test]
    fn without_dependents_only_changed_packages_are_reached() {
        let r = reached(&["crates/tetera-core/src/lib.rs"], false);
        assert_eq!(r.packages.keys().collect::<Vec<_>>(), ["tetera-core"]);
    }

    #[test]
    fn the_build_configuration_and_the_gate_reach_everything() {
        for path in [
            "Cargo.toml",
            "Cargo.lock",
            ".cargo/config.toml",
            "xtask/src/reach.rs",
        ] {
            let r = reached(&[path], false);
            assert!(r.everything.as_deref().unwrap().contains(path), "{path}");
            assert_eq!(r.packages.len(), packages().len(), "{path}");
        }
    }

    #[test]
    fn a_crates_own_manifest_reaches_only_that_crate() {
        let r = reached(&["crates/tetera-core/Cargo.toml"], false);
        assert!(r.everything.is_none());
        assert_eq!(r.packages.keys().collect::<Vec<_>>(), ["tetera-core"]);
    }

    #[test]
    fn docs_reach_nothing() {
        let r = reached(
            &[
                "docs/implementation-plan.md",
                "README.md",
                ".vscode/launch.json",
            ],
            true,
        );
        assert!(r.everything.is_none());
        assert!(r.packages.is_empty());
    }

    #[test]
    fn the_deepest_folder_owns_a_file() {
        let r = reached(&["crates/tetera-api/examples/nested/src/main.rs"], false);
        assert_eq!(r.packages.keys().collect::<Vec<_>>(), ["ex-nested"]);
    }

    #[test]
    fn a_folder_name_that_only_starts_alike_is_another_folder() {
        let r = reached(
            &["crates/tetera-core-extra/src/lib.rs", "xtasks/notes.txt"],
            false,
        );
        assert!(r.packages.is_empty(), "{:?}", r.packages);
        assert!(r.everything.is_none());
    }
}
