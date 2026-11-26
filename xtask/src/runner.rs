//! Runs a gate: works out what the change reaches, runs each step with its log, and writes
//! `target/gates/<time>-<level>/summary.md`.

use crate::exec::{self, human};
use crate::gate::{self, Action, Level, Options, PackageInfo, Step};
use crate::layering;
use crate::metadata::Metadata;
use crate::reach::{self, PackageDir, Reach};
use crate::stress;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// How a step ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// It ran and passed.
    Pass,
    /// It ran and failed, or didn't start.
    Fail,
    /// Nothing reached it.
    Skip,
}

impl Status {
    fn word(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "FAIL",
            Self::Skip => "skip",
        }
    }
}

/// A step's result.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// How it ended.
    pub status: Status,
    /// How long it took.
    pub elapsed: Duration,
    /// Anything worth a line in the summary.
    pub note: String,
}

impl StepResult {
    fn new(status: Status, elapsed: Duration, note: impl Into<String>) -> Self {
        Self {
            status,
            elapsed,
            note: note.into(),
        }
    }
}

/// Everything the summary reports.
pub struct Summary<'a> {
    /// The gate's options.
    pub opts: &'a Options,
    /// The commit, and whether the working tree had changes.
    pub commit: String,
    /// Whether the working tree had uncommitted changes.
    pub dirty: bool,
    /// The merge base the change was measured from, where one was.
    pub merge_base: Option<String>,
    /// When the gate started, in UTC.
    pub started: String,
    /// How long it took.
    pub elapsed: Duration,
    /// What the change reached.
    pub reach: &'a Reach,
    /// Each step and its result.
    pub steps: &'a [(Step, StepResult)],
}

/// Runs the gate `opts` asks for, in the workspace that holds the current folder. Returns
/// whether it passed.
pub fn run(opts: &Options) -> Result<bool, String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let meta = Metadata::load(&cwd)?;
    let root = meta.workspace_root.clone();
    let started =
        exec::output("date", &["-u", "+%Y%m%d-%H%M%S"], None).unwrap_or_else(|| "now".into());
    let dir = meta
        .target_directory
        .join("gates")
        .join(format!("{started}-{}", opts.level.name()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    let (merge_base, reach) = work_out_reach(opts, &root, &meta)?;
    let steps = gate::plan(opts, &reach, &package_infos(&meta));
    println!(
        "{} ({}): {} steps; logs in {}",
        opts.level.gate(),
        opts.level.name(),
        steps.len(),
        dir.display()
    );

    let clock = Instant::now();
    let mut done = Vec::new();
    for (i, step) in steps.into_iter().enumerate() {
        let log = dir.join(format!("{:02}-{}.log", i + 1, slug(&step.name)));
        print!("[{:>2}] {:<32} ", i + 1, step.name);
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let result = execute(&step, &log, &root, &meta, opts, merge_base.as_deref(), &dir);
        println!(
            "{} {}{}",
            result.status.word(),
            human(result.elapsed),
            note_suffix(&result.note)
        );
        done.push((step, result));
    }

    let git = |args: &[&str]| exec::output("git", args, Some(&root));
    let summary = Summary {
        opts,
        commit: git(&["rev-parse", "--short", "HEAD"]).unwrap_or_default(),
        dirty: git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty()),
        merge_base,
        started,
        elapsed: clock.elapsed(),
        reach: &reach,
        steps: &done,
    };
    let path = dir.join("summary.md");
    std::fs::write(&path, render(&summary))
        .map_err(|e| format!("writing {}: {e}", path.display()))?;
    let passed = done.iter().all(|(_, r)| r.status != Status::Fail);
    println!(
        "{}: {}",
        opts.level.gate(),
        if passed { "pass" } else { "FAIL" }
    );
    println!("summary: {}", path.display());
    Ok(passed)
}

fn note_suffix(note: &str) -> String {
    if note.is_empty() {
        String::new()
    } else {
        format!(" ({note})")
    }
}

fn work_out_reach(
    opts: &Options,
    root: &Path,
    meta: &Metadata,
) -> Result<(Option<String>, Reach), String> {
    if opts.level.on_fresh_clone() {
        let everything = Reach {
            everything: Some(format!("{} runs everything", opts.level.gate())),
            packages: meta
                .packages
                .iter()
                .map(|p| (p.name.clone(), "everything".to_string()))
                .collect(),
        };
        let merge_base = if opts.mutants && !opts.in_clone {
            Some(reach::changed_files(root, &opts.base)?.0)
        } else {
            None
        };
        return Ok((merge_base, everything));
    }
    let (merge_base, changed) = reach::changed_files(root, &opts.base)?;
    let dirs: Vec<PackageDir> = meta
        .packages
        .iter()
        .map(|p| PackageDir {
            name: p.name.clone(),
            dir: p
                .dir()
                .strip_prefix(root)
                .unwrap_or(p.dir())
                .to_string_lossy()
                .replace('\\', "/"),
        })
        .collect();
    let dependents = opts.level == Level::Phase;
    Ok((
        Some(merge_base),
        reach::compute(&changed, &dirs, &meta.member_edges(), dependents),
    ))
}

fn package_infos(meta: &Metadata) -> Vec<PackageInfo> {
    meta.packages
        .iter()
        .map(|p| PackageInfo {
            name: p.name.clone(),
            has_lib: p.has_lib(),
            has_bins: p.has_bins(),
            examples: p.examples(),
            cuda: p.features.contains_key("cuda"),
        })
        .collect()
}

fn execute(
    step: &Step,
    log: &Path,
    root: &Path,
    meta: &Metadata,
    opts: &Options,
    merge_base: Option<&str>,
    gate_dir: &Path,
) -> StepResult {
    let start = Instant::now();
    let failed = |message: String| StepResult::new(Status::Fail, start.elapsed(), message);
    match &step.action {
        Action::Skip(reason) => StepResult::new(Status::Skip, Duration::ZERO, reason.clone()),
        Action::Layering => match layering::check_workspace(root) {
            Ok(found) if found.is_empty() => {
                let _ = std::fs::write(log, "The layer rules hold.\n");
                StepResult::new(Status::Pass, start.elapsed(), "")
            }
            Ok(found) => {
                let _ = std::fs::write(log, found.join("\n") + "\n");
                StepResult::new(Status::Fail, start.elapsed(), found[0].clone())
            }
            Err(e) => failed(e),
        },
        Action::Cargo {
            args,
            env,
            limit_secs,
        } => {
            let mut cmd = Command::new("cargo");
            cmd.args(args).current_dir(root);
            for (key, value) in env {
                cmd.env(key, value);
            }
            match exec::run_logged(cmd, log, limit_secs.map(Duration::from_secs)) {
                Ok(o) if o.success => StepResult::new(Status::Pass, o.elapsed, ""),
                Ok(o) => {
                    StepResult::new(Status::Fail, o.elapsed, format!("see {}", file_name(log)))
                }
                Err(e) => failed(format!("cargo didn't start: {e}")),
            }
        }
        Action::Stress { packages, repeats } => {
            match stress::run(root, meta, packages, *repeats, opts.jobs, log) {
                Ok((true, note)) => StepResult::new(Status::Pass, start.elapsed(), note),
                Ok((false, note)) => failed(format!("{note}; see {}", file_name(log))),
                Err(e) => failed(e),
            }
        }
        Action::FreshClone => match fresh_clone(root, meta, opts, merge_base, gate_dir, log) {
            Ok((passed, note)) => {
                let status = if passed { Status::Pass } else { Status::Fail };
                StepResult::new(status, start.elapsed(), note)
            }
            Err(e) => failed(e),
        },
    }
}

/// Clones the commit into `target/gate-clone/src`, kept between runs with its own target
/// folder so it starts warm, cleans it, and runs the gate inside it.
fn fresh_clone(
    root: &Path,
    meta: &Metadata,
    opts: &Options,
    merge_base: Option<&str>,
    gate_dir: &Path,
    log: &Path,
) -> Result<(bool, String), String> {
    let home = meta.target_directory.join("gate-clone");
    let src = home.join("src");
    let git = |args: &[&str], dir: &Path| -> Result<(), String> {
        let ok = Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .is_ok_and(|s| s.success());
        if ok {
            Ok(())
        } else {
            Err(format!("git {} failed", args.join(" ")))
        }
    };
    std::fs::create_dir_all(&home).map_err(|e| e.to_string())?;
    if !src.join(".git").exists() {
        git(
            &[
                "clone",
                "--quiet",
                "--no-checkout",
                &root.display().to_string(),
                "src",
            ],
            &home,
        )?;
    }
    git(
        &[
            "fetch",
            "--quiet",
            "--force",
            &root.display().to_string(),
            "HEAD",
        ],
        &src,
    )?;
    git(
        &["checkout", "--quiet", "--force", "--detach", "FETCH_HEAD"],
        &src,
    )?;
    git(&["clean", "--quiet", "-ffdx"], &src)?;

    let mut args: Vec<String> = ["xtask", "gate", opts.level.name(), "--in-clone"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    args.extend(["--jobs".to_string(), opts.jobs.to_string()]);
    if opts.cuda {
        args.push("--cuda".to_string());
    }
    if opts.python {
        args.push("--python".to_string());
    }
    if let (true, Some(base)) = (opts.mutants, merge_base) {
        let diff: PathBuf = gate_dir.join("change.diff");
        let text =
            exec::output("git", &["diff", base, "HEAD"], Some(root)).ok_or("git diff failed")?;
        std::fs::write(&diff, text + "\n").map_err(|e| e.to_string())?;
        args.extend(["--mutants-diff".to_string(), diff.display().to_string()]);
    }
    let mut cmd = Command::new("cargo");
    cmd.args(&args)
        .current_dir(&src)
        .env("CARGO_TARGET_DIR", home.join("target"));
    let outcome =
        exec::run_logged(cmd, log, None).map_err(|e| format!("cargo didn't start: {e}"))?;

    let text = std::fs::read_to_string(log).unwrap_or_default();
    let inner = text
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("summary: "))
        .unwrap_or("no summary: see the log");
    let dirty =
        exec::output("git", &["status", "--porcelain"], Some(root)).is_some_and(|s| !s.is_empty());
    let warning = if dirty {
        "; uncommitted changes weren't included"
    } else {
        ""
    };
    Ok((outcome.success, format!("{inner}{warning}")))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// A step's name as part of a file name.
pub fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

/// The summary, as Markdown.
pub fn render(s: &Summary) -> String {
    let failed = s
        .steps
        .iter()
        .filter(|(_, r)| r.status == Status::Fail)
        .count();
    let mut out = format!(
        "# Gate {} ({})\n\n",
        s.opts.level.gate(),
        s.opts.level.name()
    );
    let tree = if s.dirty {
        "with uncommitted changes"
    } else {
        "clean"
    };
    out += &format!("- **Commit:** `{}`, working tree {tree}\n", s.commit);
    if let Some(base) = &s.merge_base {
        out += &format!(
            "- **Base:** `{}`, merge base `{}`\n",
            s.opts.base,
            &base[..base.len().min(10)]
        );
    }
    out += &format!(
        "- **Started:** {} UTC; took {}\n",
        s.started,
        human(s.elapsed)
    );
    out += &match failed {
        0 => "- **Result:** pass\n".to_string(),
        n => format!("- **Result:** FAIL, {n} step(s)\n"),
    };

    out += "\n## What the change reaches\n\n";
    if let Some(why) = &s.reach.everything {
        out += &format!("Everything: {why}.\n");
    } else if s.reach.packages.is_empty() {
        out += "No package: no package holds a changed file.\n";
    } else {
        if s.opts.level == Level::Quick {
            out += "The changed packages (the quick gate leaves out their dependents):\n\n";
        }
        for (name, why) in &s.reach.packages {
            out += &format!("- `{name}`: {why}\n");
        }
    }

    out += "\n## Steps\n\n| # | Step | Result | Time | Why | Note |\n|---|---|---|---|---|---|\n";
    for (i, (step, result)) in s.steps.iter().enumerate() {
        out += &format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            i + 1,
            step.name,
            result.status.word(),
            human(result.elapsed),
            step.why.replace('|', "/"),
            result.note.replace('|', "/"),
        );
    }
    out += "\nEach step's log is beside this file, numbered as the step.\n";
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_summary_reports_the_result_the_reach_and_each_step() {
        let mut opts = Options::new(Level::Phase);
        opts.base = "main".into();
        let reach = Reach {
            everything: None,
            packages: [(
                "tetera-core".to_string(),
                "changed: `crates/tetera-core/src/lib.rs`".to_string(),
            )]
            .into(),
        };
        let step = |name: &str, status| {
            (
                Step {
                    name: name.into(),
                    why: "a | b".into(),
                    action: Action::Layering,
                },
                StepResult::new(status, Duration::from_millis(1500), ""),
            )
        };
        let steps = [
            step("fmt", Status::Pass),
            step("tests", Status::Fail),
            step("cuda", Status::Skip),
        ];
        let text = render(&Summary {
            opts: &opts,
            commit: "abc1234".into(),
            dirty: true,
            merge_base: Some("0123456789abcdef".into()),
            started: "20261126-120000".into(),
            elapsed: Duration::from_secs(75),
            reach: &reach,
            steps: &steps,
        });
        assert!(text.starts_with("# Gate G1 (phase)\n"));
        assert!(text.contains("`abc1234`, working tree with uncommitted changes"));
        assert!(text.contains("merge base `0123456789`"));
        assert!(text.contains("took 1 min 15 s"));
        assert!(text.contains("- **Result:** FAIL, 1 step(s)"));
        assert!(text.contains("- `tetera-core`: changed: `crates/tetera-core/src/lib.rs`"));
        assert!(text.contains("| 2 | tests | FAIL | 1.5 s | a / b |  |"));
        assert!(text.contains("| 3 | cuda | skip |"));
    }

    #[test]
    fn step_names_become_file_names() {
        assert_eq!(slug("smoke tetera-core/hello"), "smoke-tetera-core-hello");
        assert_eq!(slug("Release Build"), "release-build");
    }
}
