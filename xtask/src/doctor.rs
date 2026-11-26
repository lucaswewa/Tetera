//! `cargo xtask doctor`: whether this machine has what the build and the gates need.
//! Required tools fail the check when missing; tools a later phase needs are reported only.

use crate::exec;
use std::path::Path;

/// Whether a check's subject is needed now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    /// Needed now: missing fails the doctor.
    Now,
    /// Needed from a later phase, which this names.
    From(&'static str),
}

/// How a check came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Present and as expected.
    Ok,
    /// Present, but worth a look.
    Warn,
    /// Not there.
    Missing,
}

/// One line of the doctor's report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// What was checked.
    pub name: &'static str,
    /// How it came out.
    pub status: Status,
    /// What was found, or what to do.
    pub detail: String,
    /// When it is needed.
    pub need: Need,
}

/// Runs every check for the workspace at `root`, prints the report, and returns whether
/// everything needed now is there.
pub fn run(root: &Path) -> Result<bool, String> {
    let checks = checks(root);
    for c in &checks {
        let status = match c.status {
            Status::Ok => "ok",
            Status::Warn => "warn",
            Status::Missing => "MISSING",
        };
        let need = match c.need {
            Need::Now => String::new(),
            Need::From(when) => format!(" (needed {when})"),
        };
        println!("  {status:<8} {:<16} {}{need}", c.name, c.detail);
    }
    let ok = passes(&checks);
    println!(
        "\n{}",
        if ok {
            "Everything needed now is here."
        } else {
            "Something needed now is missing."
        }
    );
    Ok(ok)
}

/// Whether nothing needed now is missing.
pub fn passes(checks: &[Check]) -> bool {
    !checks
        .iter()
        .any(|c| c.need == Need::Now && c.status == Status::Missing)
}

fn checks(root: &Path) -> Vec<Check> {
    let run = |program: &str, args: &[&str]| exec::output(program, args, Some(root));
    let first_line = |s: String| s.lines().next().unwrap_or_default().to_string();
    let tool = |name: &'static str, program: &str, args: &[&str], need: Need| -> Check {
        match run(program, args) {
            Some(out) => Check {
                name,
                status: Status::Ok,
                detail: first_line(out),
                need,
            },
            None => Check {
                name,
                status: Status::Missing,
                detail: format!("`{program}` isn't there"),
                need,
            },
        }
    };

    let mut out = Vec::new();
    let wanted = std::fs::read_to_string(root.join("rust-toolchain.toml"))
        .ok()
        .and_then(|t| toolchain_channel(&t));
    out.push(match (run("rustc", &["--version"]), wanted) {
        (Some(v), Some(w)) if v.starts_with(&format!("rustc {w} ")) => ok("rustc", v),
        (Some(v), Some(w)) => warn(
            "rustc",
            format!("{v}, but rust-toolchain.toml pins {w}"),
            Need::Now,
        ),
        (Some(v), None) => warn(
            "rustc",
            format!("{v}; rust-toolchain.toml has no channel"),
            Need::Now,
        ),
        (None, _) => missing(
            "rustc",
            "install rustup: docs/setup/toolchain.md",
            Need::Now,
        ),
    });
    out.push(tool("rustfmt", "cargo", &["fmt", "--version"], Need::Now));
    out.push(tool("clippy", "cargo", &["clippy", "--version"], Need::Now));
    out.push(tool("clang", "clang", &["--version"], Need::Now));
    out.push(tool("mold", "mold", &["--version"], Need::Now));
    out.push(tool(
        "cargo-deny",
        "cargo",
        &["deny", "--version"],
        Need::Now,
    ));
    out.push(tool(
        "cargo-mutants",
        "cargo",
        &["mutants", "--version"],
        Need::Now,
    ));
    out.push(tool("taskset", "taskset", &["--version"], Need::Now));

    let name = run("git", &["config", "user.name"]).unwrap_or_default();
    let email = run("git", &["config", "user.email"]).unwrap_or_default();
    out.push(if name.is_empty() || email.is_empty() {
        missing("git identity", "set user.name and user.email", Need::Now)
    } else {
        ok("git identity", format!("{name} <{email}>"))
    });
    out.push(match run("git", &["config", "core.autocrlf"]).as_deref() {
        Some("true") => warn(
            "core.autocrlf",
            "true: it writes CRLF; set it to input".into(),
            Need::Now,
        ),
        Some(v) => ok("core.autocrlf", v.to_string()),
        None => ok("core.autocrlf", "unset".to_string()),
    });
    out.push(if root.starts_with("/mnt/") {
        warn(
            "location",
            format!(
                "{} is on a Windows drive: builds are slow; clone into WSL",
                root.display()
            ),
            Need::Now,
        )
    } else {
        ok("location", root.display().to_string())
    });

    let free_gb = run("df", &["-Pk", "."])
        .and_then(|s| df_available_kb(&s))
        .map(|kb| kb / 1_048_576);
    out.push(match free_gb {
        Some(gb) if gb >= MIN_FREE_GB => ok("disk", format!("{gb} GB free")),
        Some(gb) => warn(
            "disk",
            format!("{gb} GB free; the checkpoint gate wants {MIN_FREE_GB}"),
            Need::Now,
        ),
        None => warn("disk", "df gave nothing readable".into(), Need::Now),
    });
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mem_gb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| meminfo_total_kb(&s))
        .map(|kb| kb as f64 / 1_048_576.0);
    out.push(match mem_gb {
        Some(gb) => {
            let detail = format!(
                "{gb:.1} GB, {cpus} CPUs; suggested CARGO_BUILD_JOBS={}",
                suggested_jobs(gb, cpus)
            );
            if gb < 8.0 {
                warn("memory", detail, Need::Now)
            } else {
                ok("memory", detail)
            }
        }
        None => warn("memory", "/proc/meminfo isn't readable".into(), Need::Now),
    });

    let gpu = ["--query-gpu=name,driver_version", "--format=csv,noheader"];
    out.push(tool(
        "nvidia-smi",
        "nvidia-smi",
        &gpu,
        Need::From("from P1.04"),
    ));
    out.push(tool(
        "nvcc",
        "nvcc",
        &["--version"],
        Need::From("from P1.04"),
    ));
    out.push(match run("node", &["--version"]) {
        Some(v) if node_major(&v).is_some_and(|m| m >= 24) => {
            ok("node", v).needed(Need::From("from M4"))
        }
        Some(v) => warn(
            "node",
            format!("{v}; M4 needs 24 or newer"),
            Need::From("from M4"),
        ),
        None => missing("node", "install Node 24 LTS", Need::From("from M4")),
    });
    out.push(tool("uv", "uv", &["--version"], Need::From("from M2")));
    out
}

/// The free disk space the checkpoint gate wants, in GB.
pub const MIN_FREE_GB: u64 = 40;

fn ok(name: &'static str, detail: String) -> Check {
    Check {
        name,
        status: Status::Ok,
        detail,
        need: Need::Now,
    }
}

fn warn(name: &'static str, detail: String, need: Need) -> Check {
    Check {
        name,
        status: Status::Warn,
        detail,
        need,
    }
}

fn missing(name: &'static str, detail: &str, need: Need) -> Check {
    Check {
        name,
        status: Status::Missing,
        detail: detail.to_string(),
        need,
    }
}

impl Check {
    fn needed(mut self, need: Need) -> Self {
        self.need = need;
        self
    }
}

/// The `channel` that a `rust-toolchain.toml` pins.
pub fn toolchain_channel(text: &str) -> Option<String> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("channel"))
        .filter_map(|rest| rest.trim().strip_prefix('='))
        .map(|v| v.trim().trim_matches('"').to_string())
        .next()
}

/// The available kilobytes in `df -Pk`'s output.
pub fn df_available_kb(text: &str) -> Option<u64> {
    text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()
}

/// `MemTotal` in kilobytes, from `/proc/meminfo`.
pub fn meminfo_total_kb(text: &str) -> Option<u64> {
    text.lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Build jobs that fit in `mem_gb`, at about 1.5 GB a job, and no more than `cpus`.
pub fn suggested_jobs(mem_gb: f64, cpus: usize) -> usize {
    ((mem_gb / 1.5) as usize).clamp(1, cpus.max(1))
}

/// The major version in `node --version`'s output, such as 24 from `v24.11.0`.
pub fn node_major(version: &str) -> Option<u32> {
    version
        .trim()
        .trim_start_matches('v')
        .split('.')
        .next()?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pinned_channel_is_read() {
        let text = "[toolchain]\n# a comment\nchannel = \"1.98.1\"\ncomponents = [\"rustfmt\"]\n";
        assert_eq!(toolchain_channel(text).as_deref(), Some("1.98.1"));
        assert_eq!(toolchain_channel("[toolchain]\n"), None);
    }

    #[test]
    fn disk_and_memory_are_read_from_df_and_meminfo() {
        let df = "Filesystem     1024-blocks      Used Available Capacity Mounted on\n/dev/sdd 1055762868 5452184 996594300 1% /\n";
        assert_eq!(df_available_kb(df), Some(996_594_300));
        assert_eq!(df_available_kb("garbage"), None);
        let meminfo = "MemTotal:       11963340 kB\nMemFree:         9000000 kB\n";
        assert_eq!(meminfo_total_kb(meminfo), Some(11_963_340));
    }

    #[test]
    fn build_jobs_fit_memory_and_cpus() {
        assert_eq!(suggested_jobs(11.4, 16), 7);
        assert_eq!(suggested_jobs(64.0, 16), 16);
        assert_eq!(suggested_jobs(1.0, 16), 1);
    }

    #[test]
    fn node_versions_are_read() {
        assert_eq!(node_major("v24.11.0"), Some(24));
        assert_eq!(node_major("v18.19.1\n"), Some(18));
        assert_eq!(node_major("node"), None);
    }

    #[test]
    fn only_a_missing_tool_needed_now_fails() {
        let later = Check {
            name: "uv",
            status: Status::Missing,
            detail: String::new(),
            need: Need::From("from M2"),
        };
        let warned = Check {
            name: "disk",
            status: Status::Warn,
            detail: String::new(),
            need: Need::Now,
        };
        assert!(passes(&[later.clone(), warned]));
        let now = Check {
            name: "mold",
            status: Status::Missing,
            detail: String::new(),
            need: Need::Now,
        };
        assert!(!passes(&[later, now]));
    }
}
