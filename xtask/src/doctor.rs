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
    out.push(rustc_check(run("rustc", &["--version"]), wanted));
    out.push(tool("rustfmt", "cargo", &["fmt", "--version"], Need::Now));
    out.push(tool("clippy", "cargo", &["clippy", "--version"], Need::Now));
    out.push(tool("clang", "clang", &["--version"], Need::Now));
    out.push(tool("mold", "mold", &["--version"], Need::Now));
    // libjpeg-turbo is built from source, with its SIMD code (P1.05b).
    out.push(tool("cmake", "cmake", &["--version"], Need::Now));
    out.push(tool("nasm", "nasm", &["-v"], Need::Now));
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
    out.push(identity_check(&name, &email));
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

    out.push(disk_check(
        run("df", &["-Pk", "."]).and_then(|s| df_available_kb(&s)),
    ));
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mem_kb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| meminfo_total_kb(&s));
    out.push(memory_check(mem_kb, cpus));

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
    out.push(node_check(run("node", &["--version"])));
    out.push(tool("uv", "uv", &["--version"], Need::From("from M2")));
    out
}

/// The free disk space the checkpoint gate wants, in GB.
pub const MIN_FREE_GB: u64 = 40;

/// Less memory than this, in GB, is worth a warning.
pub const LOW_MEMORY_GB: f64 = 8.0;

/// The check of `rustc --version`'s output against the channel `rust-toolchain.toml` pins.
pub fn rustc_check(found: Option<String>, wanted: Option<String>) -> Check {
    match (found, wanted) {
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
    }
}

/// The check of git's `user.name` and `user.email`: both must be set.
pub fn identity_check(name: &str, email: &str) -> Check {
    if name.is_empty() || email.is_empty() {
        missing("git identity", "set user.name and user.email", Need::Now)
    } else {
        ok("git identity", format!("{name} <{email}>"))
    }
}

/// The check of the free disk space, given in kilobytes.
pub fn disk_check(free_kb: Option<u64>) -> Check {
    match free_kb.map(|kb| kb / 1_048_576) {
        Some(gb) if gb >= MIN_FREE_GB => ok("disk", format!("{gb} GB free")),
        Some(gb) => warn(
            "disk",
            format!("{gb} GB free; the checkpoint gate wants {MIN_FREE_GB}"),
            Need::Now,
        ),
        None => warn("disk", "df gave nothing readable".into(), Need::Now),
    }
}

/// The check of the memory, given in kilobytes, with the build jobs it suggests.
pub fn memory_check(total_kb: Option<u64>, cpus: usize) -> Check {
    let Some(kb) = total_kb else {
        return warn("memory", "/proc/meminfo isn't readable".into(), Need::Now);
    };
    let gb = kb as f64 / 1_048_576.0;
    let detail = format!(
        "{gb:.1} GB, {cpus} CPUs; suggested CARGO_BUILD_JOBS={}",
        suggested_jobs(gb, cpus)
    );
    if gb < LOW_MEMORY_GB {
        warn("memory", detail, Need::Now)
    } else {
        ok("memory", detail)
    }
}

/// The check of `node --version`'s output: M4 needs 24 or newer.
pub fn node_check(found: Option<String>) -> Check {
    let need = Need::From("from M4");
    match found {
        Some(v) if node_major(&v).is_some_and(|m| m >= 24) => ok("node", v).needed(need),
        Some(v) => warn("node", format!("{v}; M4 needs 24 or newer"), need),
        None => missing("node", "install Node 24 LTS", need),
    }
}

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
    fn rustc_must_be_the_pinned_version() {
        let pinned = Some("1.98.1".to_string());
        let found = |v: &str| Some(v.to_string());
        assert_eq!(
            rustc_check(found("rustc 1.98.1 (48a229cea 2026-09-01)"), pinned.clone()).status,
            Status::Ok
        );
        let other = rustc_check(found("rustc 1.98.10 (x)"), pinned.clone());
        assert_eq!(other.status, Status::Warn);
        assert!(other.detail.contains("pins 1.98.1"));
        assert_eq!(
            rustc_check(found("rustc 1.98.1 (x)"), None).status,
            Status::Warn
        );
        assert_eq!(rustc_check(None, pinned).status, Status::Missing);
    }

    #[test]
    fn a_git_identity_needs_a_name_and_an_email() {
        assert_eq!(
            identity_check("Lucas", "l@example.invalid").status,
            Status::Ok
        );
        assert_eq!(identity_check("Lucas", "").status, Status::Missing);
        assert_eq!(
            identity_check("", "l@example.invalid").status,
            Status::Missing
        );
    }

    #[test]
    fn the_checkpoint_wants_forty_gigabytes_free() {
        let gb = |n: u64| Some(n * 1_048_576);
        let plenty = disk_check(gb(947));
        assert_eq!(
            (plenty.status, plenty.detail.as_str()),
            (Status::Ok, "947 GB free")
        );
        assert_eq!(disk_check(gb(MIN_FREE_GB)).status, Status::Ok);
        assert_eq!(disk_check(gb(MIN_FREE_GB - 1)).status, Status::Warn);
        assert_eq!(disk_check(None).status, Status::Warn);
    }

    #[test]
    fn little_memory_is_worth_a_warning() {
        let enough = memory_check(Some(12_268_000), 16);
        assert_eq!(enough.status, Status::Ok);
        assert_eq!(
            enough.detail,
            "11.7 GB, 16 CPUs; suggested CARGO_BUILD_JOBS=7"
        );
        assert_eq!(memory_check(Some(8 * 1_048_576), 16).status, Status::Ok);
        assert_eq!(
            memory_check(Some(8 * 1_048_576 - 1), 16).status,
            Status::Warn
        );
        assert_eq!(memory_check(None, 16).status, Status::Warn);
    }

    #[test]
    fn node_24_is_needed_from_m4() {
        let found = |v: &str| Some(v.to_string());
        let current = node_check(found("v24.0.0"));
        assert_eq!(
            (current.status, current.need),
            (Status::Ok, Need::From("from M4"))
        );
        assert_eq!(node_check(found("v23.9.0")).status, Status::Warn);
        assert_eq!(node_check(None).status, Status::Missing);
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
