//! `changed_files` against a real git repository: committed, unstaged and untracked changes
//! since the merge base all count (ADR-0005).

use std::fs;
use std::path::Path;
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args([
            "-c",
            "user.name=xtask",
            "-c",
            "user.email=xtask@example.invalid",
        ])
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn every_kind_of_change_since_the_merge_base_counts() {
    let dir = std::env::temp_dir().join(format!("xtask-reach-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    git(&dir, &["init", "--quiet", "--initial-branch=main"]);
    fs::write(dir.join("base.txt"), "base\n").unwrap();
    fs::write(dir.join("edited.txt"), "before\n").unwrap();
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "--quiet", "-m", "base"]);
    let base = git(&dir, &["rev-parse", "HEAD"]);

    git(&dir, &["switch", "--quiet", "-c", "phase"]);
    fs::write(dir.join("committed.txt"), "new\n").unwrap();
    git(&dir, &["add", "committed.txt"]);
    git(&dir, &["commit", "--quiet", "-m", "phase"]);
    fs::write(dir.join("edited.txt"), "after\n").unwrap();
    fs::write(dir.join("untracked.txt"), "loose\n").unwrap();

    let (merge_base, files) = xtask::reach::changed_files(&dir, "main").unwrap();
    assert_eq!(merge_base, base);
    assert_eq!(files, ["committed.txt", "edited.txt", "untracked.txt"]);

    let err = xtask::reach::changed_files(&dir, "no-such-branch").unwrap_err();
    assert!(err.contains("no merge base"), "{err}");
    fs::remove_dir_all(&dir).unwrap();
}
