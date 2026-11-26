//! The command line, through the built binary: its exit codes, and the doctor's report.

use std::process::{Command, Output};

fn xtask(args: &[&str], path: Option<&str>) -> (Option<i32>, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xtask"));
    cmd.args(args).env_remove("CARGO_TARGET_DIR");
    if let Some(path) = path {
        cmd.env("PATH", path);
    }
    let Output {
        status,
        stdout,
        stderr,
    } = cmd.output().expect("xtask runs");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    (status.code(), text)
}

#[test]
fn help_succeeds_no_task_fails_and_an_unknown_task_is_an_error() {
    let (code, text) = xtask(&["help"], None);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("usage: cargo xtask"), "{text}");

    let (code, text) = xtask(&[], None);
    assert_eq!(code, Some(1), "{text}");
    assert!(text.contains("usage: cargo xtask"), "{text}");

    let (code, text) = xtask(&["bogus"], None);
    assert_eq!(code, Some(2), "{text}");
    assert!(text.contains("unknown task `bogus`"), "{text}");
}

#[test]
fn the_doctor_reports_every_tool_and_its_exit_code_follows_its_verdict() {
    let (code, text) = xtask(&["doctor"], None);
    for tool in [
        "rustc",
        "mold",
        "cargo-deny",
        "taskset",
        "git identity",
        "disk",
        "memory",
    ] {
        assert!(text.contains(tool), "{tool} is missing from:\n{text}");
    }
    assert_eq!(
        code == Some(0),
        text.contains("Everything needed now is here."),
        "{text}"
    );
}

#[test]
fn the_doctor_fails_when_the_tools_arent_there() {
    let (code, text) = xtask(&["doctor"], Some("/nonexistent"));
    assert_eq!(code, Some(1), "{text}");
    assert!(text.contains("MISSING  rustc"), "{text}");
    assert!(text.contains("Something needed now is missing."), "{text}");
}
