//! The local gates of plan §8: the steps each level runs, and why. [`plan`] decides; the
//! [`runner`](crate::runner) runs.

use crate::reach::Reach;
use std::path::PathBuf;

/// A gate's level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// G0, before each commit: the changed packages only.
    Quick,
    /// G1, before a pull request: everything the change reaches.
    Phase,
    /// G2, at a checkpoint: everything, on a fresh clone.
    Checkpoint,
    /// G3, at a milestone's end: the checkpoint, and the release's checks.
    Milestone,
}

impl Level {
    /// Reads a level from its name.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "quick" => Some(Self::Quick),
            "phase" => Some(Self::Phase),
            "checkpoint" => Some(Self::Checkpoint),
            "milestone" => Some(Self::Milestone),
            _ => None,
        }
    }

    /// Its name, as typed.
    pub fn name(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Phase => "phase",
            Self::Checkpoint => "checkpoint",
            Self::Milestone => "milestone",
        }
    }

    /// Its number in plan §8, G0 to G3.
    pub fn gate(self) -> &'static str {
        match self {
            Self::Quick => "G0",
            Self::Phase => "G1",
            Self::Checkpoint => "G2",
            Self::Milestone => "G3",
        }
    }

    /// Whether it runs on a fresh clone of the commit.
    pub fn on_fresh_clone(self) -> bool {
        matches!(self, Self::Checkpoint | Self::Milestone)
    }
}

/// A gate run's options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// The level.
    pub level: Level,
    /// The ref whose merge base with `HEAD` the change is measured from.
    pub base: String,
    /// Run the CUDA tests even if no reached package has a `cuda` feature.
    pub cuda: bool,
    /// Run the Python tests.
    pub python: bool,
    /// How many runs go side by side in the stress step; 0 picks from the CPUs.
    pub jobs: usize,
    /// At a checkpoint or milestone, run `cargo mutants` on the change.
    pub mutants: bool,
    /// Set when the gate runs inside its fresh clone.
    pub in_clone: bool,
    /// Inside the clone: the change as a diff, for `cargo mutants --in-diff`.
    pub mutants_diff: Option<PathBuf>,
}

impl Options {
    /// The options for `level`, with every other choice at its default.
    pub fn new(level: Level) -> Self {
        Self {
            level,
            base: "main".to_string(),
            cuda: false,
            python: false,
            jobs: 0,
            mutants: false,
            in_clone: false,
            mutants_diff: None,
        }
    }
}

/// What a step does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Runs cargo.
    Cargo {
        /// Its arguments.
        args: Vec<String>,
        /// Environment variables to set.
        env: Vec<(String, String)>,
        /// A time limit in seconds, after which it fails.
        limit_secs: Option<u64>,
    },
    /// Checks the layer rules.
    Layering,
    /// Runs the packages' test binaries several times side by side, two CPUs each.
    Stress {
        /// The packages whose test binaries run.
        packages: Vec<String>,
        /// How many times each binary runs.
        repeats: usize,
    },
    /// Clones the commit into a clean folder, and runs the gate there.
    FreshClone,
    /// Runs nothing, for this reason.
    Skip(String),
}

/// One step of a gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// A short name, for the terminal and the summary.
    pub name: String,
    /// Why it runs, or why not.
    pub why: String,
    /// What it does.
    pub action: Action,
}

impl Step {
    fn new(name: impl Into<String>, why: impl Into<String>, action: Action) -> Self {
        Self {
            name: name.into(),
            why: why.into(),
            action,
        }
    }
}

/// A member, as the plan of steps sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageInfo {
    /// Its name.
    pub name: String,
    /// Whether it has a library.
    pub has_lib: bool,
    /// Whether it has binaries.
    pub has_bins: bool,
    /// Its examples.
    pub examples: Vec<String>,
    /// Whether it has a `cuda` feature.
    pub cuda: bool,
}

/// How many times the phase gate runs each reached test binary (plan §8).
pub const STRESS_REPEATS: usize = 3;
/// How many mutants `cargo mutants` builds and tests side by side.
pub const MUTANTS_JOBS: &str = "4";
/// How long an example's smoke run may take before it fails.
pub const SMOKE_LIMIT_SECS: u64 = 120;

/// The steps a gate with `opts` runs, for a change that reaches `reach`.
pub fn plan(opts: &Options, reach: &Reach, packages: &[PackageInfo]) -> Vec<Step> {
    if opts.level.on_fresh_clone() && !opts.in_clone {
        return vec![Step::new(
            "fresh clone",
            format!(
                "{} runs everything on a fresh clone of the commit",
                opts.level.gate()
            ),
            Action::FreshClone,
        )];
    }
    let whole = opts.level.on_fresh_clone();
    let reached: Vec<&PackageInfo> = packages
        .iter()
        .filter(|p| whole || reach.packages.contains_key(&p.name))
        .collect();
    let why = if whole {
        "every package, as a checkpoint runs".to_string()
    } else {
        describe(&reached)
    };
    let select = selection(whole, &reached);

    let mut steps = vec![
        Step::new(
            "fmt",
            "formatting is checked across the workspace",
            cargo(&["fmt", "--all", "--check"]),
        ),
        Step::new(
            "layering",
            "the layer rules hold for every crate",
            Action::Layering,
        ),
    ];
    if reached.is_empty() {
        let skip = Action::Skip("no package is reached".to_string());
        steps.push(Step::new("clippy", "nothing to lint", skip.clone()));
        steps.push(Step::new("tests", "nothing to test", skip));
        if opts.level == Level::Phase {
            steps.push(deny_step());
        }
        return steps;
    }

    let mut clippy = vec!["clippy", "--locked", "--all-targets"];
    clippy.extend(select.iter().map(String::as_str));
    clippy.extend(["--", "-D", "warnings"]);
    steps.push(Step::new("clippy", why.clone(), cargo(&clippy)));

    if opts.level == Level::Quick {
        for p in &reached {
            steps.push(unit_tests(p));
        }
        return steps;
    }

    let mut test = vec!["test", "--locked"];
    test.extend(select.iter().map(String::as_str));
    steps.push(Step::new("tests", why.clone(), cargo(&test)));

    for p in &reached {
        for example in &p.examples {
            let args = [
                "run",
                "--locked",
                "-p",
                &p.name,
                "--example",
                example,
                "--",
                "--smoke",
            ];
            let mut action = cargo(&args);
            if let Action::Cargo { limit_secs, .. } = &mut action {
                *limit_secs = Some(SMOKE_LIMIT_SECS);
            }
            steps.push(Step::new(
                format!("smoke {}/{example}", p.name),
                format!("an example of `{}`", p.name),
                action,
            ));
        }
    }

    let mut doc = vec!["doc", "--locked", "--no-deps"];
    doc.extend(select.iter().map(String::as_str));
    let mut doc = cargo(&doc);
    if let Action::Cargo { env, .. } = &mut doc {
        env.push(("RUSTDOCFLAGS".to_string(), "-D warnings".to_string()));
    }
    steps.push(Step::new("docs", why.clone(), doc));
    steps.push(deny_step());

    if opts.level == Level::Phase {
        steps.push(Step::new(
            "stress",
            format!("{why}; each test binary {STRESS_REPEATS} times side by side, on 2 CPUs each"),
            Action::Stress {
                packages: reached.iter().map(|p| p.name.clone()).collect(),
                repeats: STRESS_REPEATS,
            },
        ));
    }

    steps.extend(cuda_steps(opts, &reached));
    steps.push(Step::new(
        "python",
        "the Python SDK's tests",
        Action::Skip("no Python SDK until M2".to_string()),
    ));

    if whole {
        steps.push(match &opts.mutants_diff {
            Some(diff) => Step::new(
                "mutants",
                "cargo mutants on the change, to find tests that pass for the wrong reason",
                cargo(&[
                    "mutants",
                    "--in-diff",
                    &diff.display().to_string(),
                    "--jobs",
                    MUTANTS_JOBS,
                ]),
            ),
            None => Step::new(
                "mutants",
                "cargo mutants on the change",
                Action::Skip("pass --mutants to run it".to_string()),
            ),
        });
    }
    if opts.level == Level::Milestone {
        steps.push(Step::new(
            "licences",
            "the licence report of every dependency",
            cargo(&["deny", "list"]),
        ));
        steps.push(Step::new(
            "release build",
            "the workspace builds for release",
            cargo(&["build", "--locked", "--workspace", "--release"]),
        ));
        let later = |what: &str| Action::Skip(format!("{what} arrives in P1.21"));
        steps.push(Step::new(
            "perf",
            "the budgets of plan §6.15",
            later("`cargo xtask perf`"),
        ));
        steps.push(Step::new(
            "soak",
            "an hour of the virtual instrument",
            later("`cargo xtask soak`"),
        ));
    }
    steps
}

fn cargo(args: &[&str]) -> Action {
    Action::Cargo {
        args: args.iter().map(|a| a.to_string()).collect(),
        env: Vec::new(),
        limit_secs: None,
    }
}

fn deny_step() -> Step {
    Step::new(
        "deny",
        "licences, advisories, bans and sources",
        cargo(&["deny", "check"]),
    )
}

fn unit_tests(p: &PackageInfo) -> Step {
    let name = format!("unit tests {}", p.name);
    if !p.has_lib && !p.has_bins {
        return Step::new(
            name,
            "changed",
            Action::Skip("no library or binary".to_string()),
        );
    }
    let mut args = vec!["test", "--locked", "-p", p.name.as_str()];
    if p.has_lib {
        args.push("--lib");
    }
    if p.has_bins {
        args.push("--bins");
    }
    Step::new(name, format!("`{}` changed", p.name), cargo(&args))
}

/// G1c: for the reached crates with a `cuda` feature, clippy, the tests and each example's
/// smoke run with the feature on, the tests and examples with `TETERA_CUDA=1`. Without the
/// feature, a crate's CUDA code isn't compiled, so the other steps never see it.
fn cuda_steps(opts: &Options, reached: &[&PackageInfo]) -> Vec<Step> {
    let with_cuda: Vec<&PackageInfo> = reached.iter().copied().filter(|p| p.cuda).collect();
    if with_cuda.is_empty() {
        let reason = if opts.cuda {
            "--cuda was given, but no reached package has a `cuda` feature"
        } else {
            "no reached package has a `cuda` feature"
        };
        return vec![Step::new(
            "cuda",
            "the CUDA twins",
            Action::Skip(reason.to_string()),
        )];
    }
    let names: Vec<&str> = with_cuda.iter().map(|p| p.name.as_str()).collect();
    let why = format!("reached with a `cuda` feature: {}", names.join(", "));
    let features: Vec<String> = names.iter().map(|n| format!("{n}/cuda")).collect();
    let select: Vec<&str> = names
        .iter()
        .zip(&features)
        .flat_map(|(name, feature)| ["-p", *name, "--features", feature.as_str()])
        .collect();
    let on_gpu = |mut action: Action| {
        if let Action::Cargo { env, .. } = &mut action {
            env.push(("TETERA_CUDA".to_string(), "1".to_string()));
        }
        action
    };

    let mut clippy = vec!["clippy", "--locked", "--all-targets"];
    clippy.extend(&select);
    clippy.extend(["--", "-D", "warnings"]);
    let mut test = vec!["test", "--locked"];
    test.extend(&select);
    let mut steps = vec![
        Step::new("cuda clippy", why.clone(), cargo(&clippy)),
        Step::new("cuda tests", why, on_gpu(cargo(&test))),
    ];
    for p in &with_cuda {
        let feature = format!("{}/cuda", p.name);
        for example in &p.examples {
            let args = [
                "run",
                "--locked",
                "-p",
                &p.name,
                "--features",
                &feature,
                "--example",
                example,
                "--",
                "--smoke",
            ];
            let mut action = on_gpu(cargo(&args));
            if let Action::Cargo { limit_secs, .. } = &mut action {
                *limit_secs = Some(SMOKE_LIMIT_SECS);
            }
            steps.push(Step::new(
                format!("cuda smoke {}/{example}", p.name),
                format!("an example of `{}`, with CUDA", p.name),
                action,
            ));
        }
    }
    steps
}

fn selection(whole: bool, reached: &[&PackageInfo]) -> Vec<String> {
    if whole {
        return vec!["--workspace".to_string()];
    }
    reached
        .iter()
        .flat_map(|p| ["-p".to_string(), p.name.clone()])
        .collect()
}

fn describe(reached: &[&PackageInfo]) -> String {
    let names: Vec<&str> = reached.iter().map(|p| p.name.as_str()).collect();
    match names.len() {
        0 => "nothing is reached".to_string(),
        1..=5 => format!("reached: {}", names.join(", ")),
        n => format!("reached: {} and {} more", names[..4].join(", "), n - 4),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, examples: &[&str], cuda: bool) -> PackageInfo {
        PackageInfo {
            name: name.to_string(),
            has_lib: name != "xtask-bin-only",
            has_bins: name == "xtask" || name == "xtask-bin-only",
            examples: examples.iter().map(|e| e.to_string()).collect(),
            cuda,
        }
    }

    fn workspace() -> Vec<PackageInfo> {
        vec![
            info("tetera-core", &["hello"], false),
            info("tetera-compute", &["info"], true),
            info("xtask", &[], false),
        ]
    }

    fn reach(names: &[&str]) -> Reach {
        Reach {
            everything: None,
            packages: names
                .iter()
                .map(|n| (n.to_string(), "changed".to_string()))
                .collect(),
        }
    }

    fn names(steps: &[Step]) -> Vec<&str> {
        steps.iter().map(|s| s.name.as_str()).collect()
    }

    #[test]
    fn quick_lints_and_unit_tests_the_changed_packages_only() {
        let steps = plan(
            &Options::new(Level::Quick),
            &reach(&["tetera-core"]),
            &workspace(),
        );
        assert_eq!(
            names(&steps),
            ["fmt", "layering", "clippy", "unit tests tetera-core"]
        );
        assert_eq!(
            steps[3].action,
            cargo(&["test", "--locked", "-p", "tetera-core", "--lib"])
        );
        let Action::Cargo { args, .. } = &steps[2].action else {
            panic!()
        };
        assert!(args.windows(2).any(|w| w == ["-p", "tetera-core"]));
        assert!(!args.contains(&"tetera-compute".to_string()));
    }

    #[test]
    fn quick_with_nothing_reached_still_checks_formatting_and_layers() {
        let steps = plan(&Options::new(Level::Quick), &reach(&[]), &workspace());
        assert_eq!(names(&steps), ["fmt", "layering", "clippy", "tests"]);
        assert!(matches!(steps[2].action, Action::Skip(_)));
    }

    #[test]
    fn phase_runs_tests_smoke_docs_deny_stress_and_cuda_for_the_reach() {
        let steps = plan(
            &Options::new(Level::Phase),
            &reach(&["tetera-core", "tetera-compute"]),
            &workspace(),
        );
        assert_eq!(
            names(&steps),
            [
                "fmt",
                "layering",
                "clippy",
                "tests",
                "smoke tetera-core/hello",
                "smoke tetera-compute/info",
                "docs",
                "deny",
                "stress",
                "cuda clippy",
                "cuda tests",
                "cuda smoke tetera-compute/info",
                "python"
            ]
        );
        let smoke = &steps[4].action;
        assert_eq!(
            *smoke,
            Action::Cargo {
                args: [
                    "run",
                    "--locked",
                    "-p",
                    "tetera-core",
                    "--example",
                    "hello",
                    "--",
                    "--smoke"
                ]
                .iter()
                .map(|s| s.to_string())
                .collect(),
                env: vec![],
                limit_secs: Some(SMOKE_LIMIT_SECS),
            }
        );
        let Action::Cargo { env, .. } = &steps[6].action else {
            panic!()
        };
        assert_eq!(
            env,
            &[("RUSTDOCFLAGS".to_string(), "-D warnings".to_string())]
        );
        assert_eq!(
            steps[8].action,
            Action::Stress {
                packages: vec!["tetera-core".into(), "tetera-compute".into()],
                repeats: 3
            }
        );
        let Action::Cargo { args, env, .. } = &steps[9].action else {
            panic!("{:?}", steps[9])
        };
        assert!(
            args.windows(4)
                .any(|w| w == ["-p", "tetera-compute", "--features", "tetera-compute/cuda"]),
            "{args:?}"
        );
        assert_eq!(args.last().map(String::as_str), Some("warnings"));
        assert!(env.is_empty(), "clippy needs no GPU");
        let Action::Cargo { args, env, .. } = &steps[10].action else {
            panic!("{:?}", steps[10])
        };
        assert_eq!(args[0], "test");
        assert_eq!(env, &[("TETERA_CUDA".to_string(), "1".to_string())]);
        let Action::Cargo {
            args,
            env,
            limit_secs,
        } = &steps[11].action
        else {
            panic!("{:?}", steps[11])
        };
        assert_eq!(
            args,
            &[
                "run",
                "--locked",
                "-p",
                "tetera-compute",
                "--features",
                "tetera-compute/cuda",
                "--example",
                "info",
                "--",
                "--smoke"
            ]
        );
        assert_eq!((env.len(), *limit_secs), (1, Some(SMOKE_LIMIT_SECS)));
    }

    #[test]
    fn phase_with_nothing_reached_still_checks_the_licences() {
        let steps = plan(&Options::new(Level::Phase), &reach(&[]), &workspace());
        assert_eq!(
            names(&steps),
            ["fmt", "layering", "clippy", "tests", "deny"]
        );
    }

    #[test]
    fn a_checkpoint_outside_its_clone_only_makes_the_clone() {
        let steps = plan(&Options::new(Level::Checkpoint), &reach(&[]), &workspace());
        assert_eq!(names(&steps), ["fresh clone"]);
        assert_eq!(steps[0].action, Action::FreshClone);
    }

    #[test]
    fn a_checkpoint_in_its_clone_runs_the_whole_workspace_without_stress() {
        let mut opts = Options::new(Level::Checkpoint);
        opts.in_clone = true;
        let steps = plan(&opts, &reach(&[]), &workspace());
        assert_eq!(
            names(&steps),
            [
                "fmt",
                "layering",
                "clippy",
                "tests",
                "smoke tetera-core/hello",
                "smoke tetera-compute/info",
                "docs",
                "deny",
                "cuda clippy",
                "cuda tests",
                "cuda smoke tetera-compute/info",
                "python",
                "mutants"
            ]
        );
        let Action::Cargo { args, .. } = &steps[3].action else {
            panic!()
        };
        assert_eq!(args, &["test", "--locked", "--workspace"]);
        assert!(matches!(steps[12].action, Action::Skip(_)));

        opts.mutants_diff = Some(PathBuf::from("/tmp/change.diff"));
        let steps = plan(&opts, &reach(&[]), &workspace());
        assert_eq!(
            steps[12].action,
            cargo(&[
                "mutants",
                "--in-diff",
                "/tmp/change.diff",
                "--jobs",
                MUTANTS_JOBS
            ])
        );
    }

    #[test]
    fn a_milestone_adds_the_release_checks() {
        let mut opts = Options::new(Level::Milestone);
        opts.in_clone = true;
        let steps = plan(&opts, &reach(&[]), &workspace());
        assert!(names(&steps).ends_with(&["mutants", "licences", "release build", "perf", "soak"]));
    }

    #[test]
    fn a_package_without_a_library_runs_its_binaries_tests() {
        let step = unit_tests(&info("xtask-bin-only", &[], false));
        assert_eq!(
            step.action,
            cargo(&["test", "--locked", "-p", "xtask-bin-only", "--bins"])
        );
    }

    #[test]
    fn long_reaches_are_shortened() {
        let many: Vec<PackageInfo> = (0..7).map(|i| info(&format!("p{i}"), &[], false)).collect();
        let refs: Vec<&PackageInfo> = many.iter().collect();
        assert_eq!(describe(&refs), "reached: p0, p1, p2, p3 and 3 more");
    }

    #[test]
    fn levels_read_and_print_their_names() {
        for level in [
            Level::Quick,
            Level::Phase,
            Level::Checkpoint,
            Level::Milestone,
        ] {
            assert_eq!(Level::parse(level.name()), Some(level));
        }
        assert_eq!(Level::parse("full"), None);
        assert_eq!(Level::Checkpoint.gate(), "G2");
    }
}
