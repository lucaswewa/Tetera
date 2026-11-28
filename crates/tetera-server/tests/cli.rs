//! The binary, run as a user runs it: what it prints and the code it exits with. Nothing
//! here serves, so each run ends by itself; one that doesn't within its limit is killed.

use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Runs `tetera-server` with `args`, killed if it runs 30 s.
fn server(args: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tetera-server"))
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the binary starts");
    let start = Instant::now();
    while child.try_wait().expect("it can be waited for").is_none() {
        if start.elapsed() > Duration::from_secs(30) {
            child.kill().expect("it can be killed");
            panic!("tetera-server {args:?} ran on");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    child.wait_with_output().expect("its output")
}

const SHIPPED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/tetera.toml");

#[test]
fn check_prints_the_configuration_with_its_overrides() {
    let out = server(&["--config", SHIPPED, "--port", "9400", "--check"]);
    assert_eq!(out.status.code(), Some(0));
    let printed = String::from_utf8_lossy(&out.stdout);
    assert!(
        printed.contains("port: 9400") && printed.contains("20x"),
        "{printed}"
    );
}

#[test]
fn a_configuration_that_wont_do_is_exit_code_2_and_says_why() {
    let out = server(&["--config", "/nowhere/tetera.toml"]);
    assert_eq!(out.status.code(), Some(2));
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("/nowhere/tetera.toml"), "{said}");
    let bad = std::env::temp_dir().join(format!("tetera-cli-test-{}.toml", std::process::id()));
    std::fs::write(&bad, "[server]\nprot = 1\n").unwrap();
    let out = server(&["--config", bad.to_str().unwrap()]);
    let _ = std::fs::remove_file(&bad);
    assert_eq!(out.status.code(), Some(2));
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("prot"), "{said}");
    assert_eq!(
        server(&["--port", "port"]).status.code(),
        Some(2),
        "clap's own refusal"
    );
}

#[test]
fn version_and_help_answer() {
    let out = server(&["--version"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains(env!("CARGO_PKG_VERSION")));
    let out = server(&["--help"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains("--data-root"));
}
