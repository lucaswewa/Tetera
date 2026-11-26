//! The stress step (ADR-0005): each test binary runs several times side by side, each run
//! pinned to two CPUs with `taskset`, as a small CI runner would have. Races that a machine
//! with many threads hides show up here.

use crate::exec;
use crate::metadata::Metadata;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A test binary that `cargo test --no-run` built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestBinary {
    /// The executable.
    pub exe: PathBuf,
    /// Its package's folder, where cargo would run it.
    pub dir: PathBuf,
    /// A name for the log: the package, then the target.
    pub label: String,
}

/// The test binaries listed in `cargo test --no-run --message-format=json`'s output.
pub fn binaries_from_messages(messages: &str, meta: &Metadata) -> Vec<TestBinary> {
    let mut found = Vec::new();
    for line in messages.lines() {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let is_test = msg["reason"] == "compiler-artifact" && msg["profile"]["test"] == true;
        let (Some(exe), Some(id)) = (msg["executable"].as_str(), msg["package_id"].as_str()) else {
            continue;
        };
        if !is_test {
            continue;
        }
        let Some(package) = meta.package_by_id(id) else {
            continue;
        };
        let target = msg["target"]["name"].as_str().unwrap_or("?");
        // A library and a binary often share a name: the kind tells their runs apart.
        let kind = msg["target"]["kind"][0].as_str().unwrap_or("?");
        found.push(TestBinary {
            exe: PathBuf::from(exe),
            dir: package.dir().to_path_buf(),
            label: format!("{}/{target} ({kind})", package.name),
        });
    }
    found
}

/// How many runs go side by side: `jobs` if given, or one for every two CPUs, at most eight.
pub fn slots(jobs: usize, cpus: usize) -> usize {
    let most = (cpus / 2).max(1);
    if jobs > 0 {
        jobs.min(most)
    } else {
        most.min(8)
    }
}

/// The CPUs that slot `k` is pinned to.
pub fn cpus_of_slot(k: usize, cpus: usize) -> String {
    if cpus < 2 {
        "0".to_string()
    } else {
        format!("{},{}", 2 * k, 2 * k + 1)
    }
}

/// Builds the packages' test binaries, then runs each `repeats` times side by side.
/// Returns whether every run passed, and a line for the summary. Details go to `log`.
pub fn run(
    root: &Path,
    meta: &Metadata,
    packages: &[String],
    repeats: usize,
    jobs: usize,
    log: &Path,
) -> Result<(bool, String), String> {
    let mut build = Command::new("cargo");
    build
        .args(["test", "--locked", "--no-run", "--message-format=json"])
        .current_dir(root);
    for p in packages {
        build.args(["-p", p]);
    }
    let out = build
        .output()
        .map_err(|e| format!("cargo didn't start: {e}"))?;
    let write = |text: &str| exec::append(log, text).map_err(|e| e.to_string());
    write(&String::from_utf8_lossy(&out.stderr))?;
    if !out.status.success() {
        return Ok((false, "the test binaries didn't build".to_string()));
    }
    let binaries = binaries_from_messages(&String::from_utf8_lossy(&out.stdout), meta);
    if binaries.is_empty() {
        return Ok((true, "no test binaries".to_string()));
    }

    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let slot_count = slots(jobs, cpus);
    let queue: Mutex<VecDeque<(usize, usize)>> = Mutex::new(
        (0..binaries.len())
            .flat_map(|b| (1..=repeats).map(move |r| (b, r)))
            .collect(),
    );
    let results: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let failures = Mutex::new(0usize);
    std::thread::scope(|scope| {
        for k in 0..slot_count {
            let (queue, results, failures, binaries) = (&queue, &results, &failures, &binaries);
            scope.spawn(move || {
                let pin = cpus_of_slot(k, cpus);
                loop {
                    let Some((b, repeat)) = queue.lock().unwrap().pop_front() else {
                        break;
                    };
                    let line = run_one(&binaries[b], repeat, &pin, failures);
                    results.lock().unwrap().push(line);
                }
            });
        }
    });
    let failed = *failures.lock().unwrap();
    for line in results.into_inner().unwrap() {
        write(&line)?;
    }
    let note = format!(
        "{} binaries × {repeats} on {slot_count} slots of 2 CPUs; {failed} failed",
        binaries.len()
    );
    Ok((failed == 0, note))
}

fn run_one(binary: &TestBinary, repeat: usize, pin: &str, failures: &Mutex<usize>) -> String {
    let start = Instant::now();
    let out = Command::new("taskset")
        .arg("-c")
        .arg(pin)
        .arg(&binary.exe)
        .current_dir(&binary.dir)
        .env("CARGO_MANIFEST_DIR", &binary.dir)
        .stdin(Stdio::null())
        .output();
    let took = exec::human(start.elapsed().max(Duration::from_millis(1)));
    match out {
        Ok(out) if out.status.success() => {
            format!("pass {} #{repeat} on CPUs {pin}, {took}\n", binary.label)
        }
        other => {
            *failures.lock().unwrap() += 1;
            let detail = match other {
                Ok(out) => tail(&format!(
                    "{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                )),
                Err(e) => format!("didn't start: {e}"),
            };
            format!(
                "FAIL {} #{repeat} on CPUs {pin}, {took}\n{detail}\n",
                binary.label
            )
        }
    }
}

/// The last 40 lines of `text`.
fn tail(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(40)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Package;
    use std::collections::BTreeMap;

    fn meta() -> Metadata {
        Metadata {
            packages: vec![Package {
                name: "tetera-core".into(),
                id: "path+file:///ws/crates/tetera-core#0.1.0".into(),
                manifest_path: "/ws/crates/tetera-core/Cargo.toml".into(),
                dependencies: vec![],
                targets: vec![],
                features: BTreeMap::new(),
            }],
            workspace_root: "/ws".into(),
            target_directory: "/ws/target".into(),
        }
    }

    #[test]
    fn test_binaries_are_found_in_cargos_messages() {
        let messages = r#"
{"reason":"compiler-artifact","package_id":"path+file:///ws/crates/tetera-core#0.1.0","target":{"name":"tetera_core","kind":["lib"]},"profile":{"test":true},"executable":"/ws/target/debug/deps/tetera_core-1"}
{"reason":"compiler-artifact","package_id":"path+file:///ws/crates/tetera-core#0.1.0","target":{"name":"tetera_core"},"profile":{"test":false},"executable":null}
{"reason":"compiler-artifact","package_id":"registry+https://x#serde@1.0.0","target":{"name":"serde"},"profile":{"test":true},"executable":"/elsewhere"}
{"reason":"build-finished","success":true}
not json
"#;
        let found = binaries_from_messages(messages, &meta());
        assert_eq!(
            found,
            [TestBinary {
                exe: "/ws/target/debug/deps/tetera_core-1".into(),
                dir: "/ws/crates/tetera-core".into(),
                label: "tetera-core/tetera_core (lib)".into(),
            }]
        );
    }

    #[test]
    fn slots_take_two_cpus_each_and_at_most_eight_by_default() {
        assert_eq!(slots(0, 16), 8);
        assert_eq!(slots(0, 24), 8);
        assert_eq!(slots(0, 6), 3);
        assert_eq!(slots(0, 1), 1);
        assert_eq!(slots(4, 16), 4);
        assert_eq!(slots(20, 16), 8);
        assert_eq!(cpus_of_slot(0, 16), "0,1");
        assert_eq!(cpus_of_slot(3, 16), "6,7");
        assert_eq!(cpus_of_slot(0, 1), "0");
    }

    #[test]
    fn a_failures_tail_keeps_the_last_forty_lines() {
        let text: String = (1..=50).map(|i| format!("line {i}\n")).collect();
        let kept = tail(&text);
        assert!(kept.starts_with("line 11") && kept.ends_with("line 50"));
    }
}
