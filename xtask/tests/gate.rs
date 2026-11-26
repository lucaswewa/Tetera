//! The gate end to end, on scratch workspaces (ADR-0005): it passes a good change, fails a
//! failing test and a broken layer, names only what a quick gate reaches, and its stress
//! step tells a passing test binary from a failing one.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use xtask::metadata::Metadata;

const PASSING: &str = "//! A scratch crate.

/// Adds one.
pub fn add_one(x: u32) -> u32 {
    x + 1
}

#[cfg(test)]
mod tests {
    #[test]
    fn adds_one() {
        assert_eq!(super::add_one(1), 2);
    }
}
";

/// A git repository holding a workspace, removed when dropped: `tetera-core`, and
/// `tetera-log`, which depends on it, committed on `main`.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("xtask-gate-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let ws = Self(dir);
        ws.write(
            "Cargo.toml",
            "[workspace]\nresolver = \"3\"\nmembers = [\"crates/*\"]\n\n\
             [workspace.lints.rust]\nunsafe_code = \"forbid\"\n",
        );
        ws.krate("core", "tetera-core", "", PASSING);
        ws.krate(
            "log",
            "tetera-log",
            "tetera-core = { path = \"../core\" }\n",
            PASSING,
        );
        ws.lock();
        ws.run("git", &["init", "--quiet", "--initial-branch=main"]);
        ws.run("git", &["add", "."]);
        ws.run("git", &["commit", "--quiet", "-m", "base"]);
        ws
    }

    fn write(&self, path: &str, text: &str) {
        let path = self.0.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn krate(&self, folder: &str, name: &str, dependencies: &str, lib: &str) {
        self.write(
            &format!("crates/{folder}/Cargo.toml"),
            &format!(
                "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n\
                 [dependencies]\n{dependencies}\n[lints]\nworkspace = true\n"
            ),
        );
        self.write(&format!("crates/{folder}/src/lib.rs"), lib);
    }

    /// Brings `Cargo.lock` up to date, as the gate's `--locked` builds need.
    fn lock(&self) {
        self.run("cargo", &["generate-lockfile", "--offline", "--quiet"]);
    }

    fn run(&self, program: &str, args: &[&str]) {
        let out = Command::new(program)
            .args(args)
            .current_dir(&self.0)
            .env_remove("CARGO_TARGET_DIR")
            .env("GIT_AUTHOR_NAME", "xtask")
            .env("GIT_AUTHOR_EMAIL", "xtask@example.invalid")
            .env("GIT_COMMITTER_NAME", "xtask")
            .env("GIT_COMMITTER_EMAIL", "xtask@example.invalid")
            .output()
            .expect("the program runs");
        assert!(
            out.status.success(),
            "{program} {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// Runs `cargo xtask gate <level>` here; returns whether it passed, and its summary.
    fn gate(&self, level: &str) -> (bool, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["gate", level, "--jobs", "1"])
            .current_dir(&self.0)
            .env_remove("CARGO_TARGET_DIR")
            .output()
            .expect("xtask runs");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let path = stdout
            .lines()
            .find_map(|l| l.strip_prefix("summary: "))
            .unwrap_or_else(|| {
                panic!(
                    "no summary:\n{stdout}\n{}",
                    String::from_utf8_lossy(&out.stderr)
                )
            });
        (out.status.success(), fs::read_to_string(path).unwrap())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn the_quick_gate_passes_a_good_change_and_names_only_the_changed_package() {
    let ws = Scratch::new("good");
    ws.write(
        "crates/core/src/lib.rs",
        &PASSING.replace("Adds one.", "Adds one to `x`."),
    );
    let (passed, summary) = ws.gate("quick");
    assert!(passed, "{summary}");
    assert!(
        summary.contains("working tree with uncommitted changes"),
        "{summary}"
    );
    assert!(
        summary.contains("- `tetera-core`: changed: `crates/core/src/lib.rs`"),
        "{summary}"
    );
    assert!(
        !summary.contains("`tetera-log`"),
        "a quick gate leaves out dependents:\n{summary}"
    );
    assert!(
        summary.contains("| unit tests tetera-core | pass |"),
        "{summary}"
    );
}

#[test]
fn the_quick_gate_fails_a_failing_test_and_a_broken_layer() {
    let ws = Scratch::new("bad");
    ws.krate("hal", "tetera-hal", "", PASSING);
    let failing = PASSING.replace("add_one(1), 2", "add_one(1), 3");
    ws.krate(
        "core",
        "tetera-core",
        "tetera-hal = { path = \"../hal\" }\n",
        &failing,
    );
    ws.lock();
    let (passed, summary) = ws.gate("quick");
    assert!(!passed, "{summary}");
    assert!(
        summary.contains("- **Result:** FAIL, 2 step(s)"),
        "{summary}"
    );
    assert!(summary.contains("| layering | FAIL |"), "{summary}");
    assert!(
        summary.contains("| unit tests tetera-core | FAIL |"),
        "{summary}"
    );
    assert!(
        summary.contains("| unit tests tetera-hal | pass |"),
        "{summary}"
    );
}

#[test]
fn the_stress_step_tells_a_passing_binary_from_a_failing_one() {
    if std::env::var_os("CARGO_TARGET_DIR").is_some() {
        // The scratch build would share the target folder whose lock the running tests hold.
        eprintln!("skipped: CARGO_TARGET_DIR is set");
        return;
    }
    let ws = Scratch::new("stress");
    let meta = Metadata::load(&ws.0).unwrap();
    let log = ws.0.join("stress.log");
    let core = ["tetera-core".to_string()];

    let (passed, note) = xtask::stress::run(&ws.0, &meta, &core, 2, 1, &log).unwrap();
    assert!(passed, "{note}");
    assert_eq!(note, "1 binaries × 2 on 1 slots of 2 CPUs; 0 failed");

    ws.write(
        "crates/core/src/lib.rs",
        &PASSING.replace("add_one(1), 2", "add_one(1), 3"),
    );
    let (passed, note) = xtask::stress::run(&ws.0, &meta, &core, 2, 1, &log).unwrap();
    assert!(!passed, "{note}");
    assert!(note.ends_with("; 2 failed"), "{note}");
    let text = fs::read_to_string(&log).unwrap();
    assert!(
        text.contains("FAIL tetera-core/tetera_core (lib) #1"),
        "{text}"
    );
    assert!(text.contains("assertion `left == right` failed"), "{text}");
}

#[test]
fn the_stress_step_reports_a_build_that_fails() {
    if std::env::var_os("CARGO_TARGET_DIR").is_some() {
        eprintln!("skipped: CARGO_TARGET_DIR is set");
        return;
    }
    let ws = Scratch::new("unbuildable");
    ws.write(
        "crates/core/src/lib.rs",
        "//! Doesn't build.\npub fn broken() -> u32 { \"no\" }\n",
    );
    let meta = Metadata::load(&ws.0).unwrap();
    let log = ws.0.join("stress.log");
    let (passed, note) =
        xtask::stress::run(&ws.0, &meta, &["tetera-core".to_string()], 1, 1, &log).unwrap();
    assert!(!passed);
    assert_eq!(note, "the test binaries didn't build");
}
