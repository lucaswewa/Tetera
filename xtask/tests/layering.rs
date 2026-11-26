//! The layer rules hold for this workspace (ADR-0005), and the check finds a broken rule in
//! a real workspace. Part of `cargo test`, so every gate that runs xtask's tests checks them.

use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn the_workspace_follows_the_layer_rules() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits in the workspace's root");
    let found = xtask::layering::check_workspace(root).expect("cargo metadata runs");
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// A throwaway workspace in the temporary folder, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("xtask-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("Cargo.toml"),
            "[workspace]\nresolver = \"3\"\nmembers = [\"crates/*\"]\nexclude = [\"helpers\"]\n\n\
             [workspace.lints.rust]\nunsafe_code = \"forbid\"\n",
        )
        .unwrap();
        Self(dir)
    }

    fn krate(&self, folder: &str, name: &str, dependencies: &str) {
        let dir = self.0.join(folder);
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("src/lib.rs"), "//! A scratch crate.\n").unwrap();
        fs::write(
            dir.join("Cargo.toml"),
            format!(
                "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n\
                 [dependencies]\n{dependencies}\n[lints]\nworkspace = true\n"
            ),
        )
        .unwrap();
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn an_upward_edge_in_a_real_workspace_is_found() {
    let ws = Scratch::new("upward");
    ws.krate("crates/hal", "tetera-hal", "");
    ws.krate(
        "crates/core",
        "tetera-core",
        "tetera-hal = { path = \"../hal\" }\n",
    );
    let found = xtask::layering::check_workspace(&ws.0).expect("cargo metadata runs");
    assert_eq!(
        found,
        ["`tetera-core` (L0) depends on `tetera-hal` (L2), a higher layer."]
    );
}

#[test]
fn a_path_dependency_outside_the_workspace_isnt_a_member() {
    // A crate the workspace excludes is no member, so no member rule applies to it.
    let ws = Scratch::new("outside");
    let helper = ws.0.join("helpers/helper");
    fs::create_dir_all(helper.join("src")).unwrap();
    fs::write(helper.join("src/lib.rs"), "//! Outside the workspace.\n").unwrap();
    fs::write(
        helper.join("Cargo.toml"),
        "[package]\nname = \"helper\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    ws.krate(
        "crates/core",
        "tetera-core",
        "helper = { path = \"../../helpers/helper\" }\n",
    );
    let found = xtask::layering::check_workspace(&ws.0).expect("cargo metadata runs");
    assert!(found.is_empty(), "{}", found.join("\n"));
}
