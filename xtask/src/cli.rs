//! The command line: `cargo xtask <task> [options]`.

use crate::gate::{Level, Options};
use crate::{doctor, layering, runner};
use std::path::PathBuf;

/// What `cargo xtask help` prints.
pub const USAGE: &str = "\
usage: cargo xtask <task>

tasks:
  layering                  check the layer rules (ADR-0005)
  gate <level> [options]    run a local gate (plan §8):
                              quick (G0)       the changed packages: fmt, layering, clippy, unit tests
                              phase (G1)       everything the change reaches, with stress runs
                              checkpoint (G2)  everything, on a fresh clone of the commit
                              milestone (G3)   the checkpoint, and the release's checks
      --base <ref>          measure the change from this ref's merge base (default: main)
      --cuda                run the CUDA tests even where no reached package has a `cuda` feature
      --python              run the Python tests
      --jobs <n>            runs side by side in the stress step (default: one per 2 CPUs, at most 8)
      --mutants             checkpoint and milestone: run cargo mutants on the change
  doctor                    check this machine's tools, disk and memory
  help                      print this
";

/// Runs the task `args` names. Returns whether it passed.
pub fn run(args: &[String]) -> Result<bool, String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    match args.first().map(String::as_str) {
        Some("layering") => {
            let found = layering::check_workspace(&cwd)?;
            if found.is_empty() {
                println!("The layer rules hold.");
            }
            for line in &found {
                println!("{line}");
            }
            Ok(found.is_empty())
        }
        Some("gate") => runner::run(&parse_gate(&args[1..])?),
        Some("doctor") => doctor::run(&cwd),
        Some("help" | "--help" | "-h") => {
            print!("{USAGE}");
            Ok(true)
        }
        None => {
            print!("{USAGE}");
            Ok(false)
        }
        Some(other) => Err(format!("unknown task `{other}`\n\n{USAGE}")),
    }
}

/// Reads the gate's level and options.
pub fn parse_gate(args: &[String]) -> Result<Options, String> {
    let mut args = args.iter();
    let level = args
        .next()
        .ok_or("gate needs a level: quick, phase, checkpoint or milestone")?;
    let level = Level::parse(level).ok_or_else(|| {
        format!("`{level}` isn't a level: choose quick, phase, checkpoint or milestone")
    })?;
    let mut opts = Options::new(level);
    while let Some(arg) = args.next() {
        let mut value = || args.next().cloned().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--base" => opts.base = value()?,
            "--jobs" => {
                let v = value()?;
                opts.jobs = v
                    .parse()
                    .map_err(|_| format!("--jobs takes a number, not `{v}`"))?;
            }
            "--cuda" => opts.cuda = true,
            "--python" => opts.python = true,
            "--mutants" => opts.mutants = true,
            // Set by the gate itself, when it runs inside its fresh clone.
            "--in-clone" => opts.in_clone = true,
            "--mutants-diff" => opts.mutants_diff = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown option `{other}` for gate\n\n{USAGE}")),
        }
    }
    Ok(opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        parse_gate(&args.iter().map(|a| a.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn the_level_comes_first_and_options_follow() {
        let opts = parse(&["phase", "--base", "origin/main", "--jobs", "4", "--cuda"]).unwrap();
        assert_eq!(opts.level, Level::Phase);
        assert_eq!(opts.base, "origin/main");
        assert_eq!(opts.jobs, 4);
        assert!(opts.cuda && !opts.python && !opts.mutants && !opts.in_clone);
    }

    #[test]
    fn defaults_measure_from_main() {
        assert_eq!(parse(&["quick"]).unwrap(), Options::new(Level::Quick));
    }

    #[test]
    fn mistakes_are_explained() {
        assert!(parse(&[]).unwrap_err().contains("needs a level"));
        assert!(parse(&["full"]).unwrap_err().contains("isn't a level"));
        assert!(
            parse(&["phase", "--jobs", "x"])
                .unwrap_err()
                .contains("takes a number")
        );
        assert!(
            parse(&["phase", "--base"])
                .unwrap_err()
                .contains("needs a value")
        );
        assert!(
            parse(&["phase", "--fast"])
                .unwrap_err()
                .contains("unknown option")
        );
    }

    #[test]
    fn the_clones_own_options_are_read() {
        let opts = parse(&["checkpoint", "--in-clone", "--mutants-diff", "/tmp/d"]).unwrap();
        assert!(opts.in_clone);
        assert_eq!(opts.mutants_diff, Some(PathBuf::from("/tmp/d")));
    }
}
