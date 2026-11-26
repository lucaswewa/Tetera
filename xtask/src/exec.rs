//! Running commands: their output to a log, timed, with an optional time limit.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// What a finished command left behind.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    /// Whether it exited with status 0 within its time limit.
    pub success: bool,
    /// How long it ran.
    pub elapsed: Duration,
}

/// Runs `cmd` with its standard output and error going to `log`. With a `limit`, a command
/// still running after it is stopped and counts as failed.
pub fn run_logged(
    mut cmd: Command,
    log: &Path,
    limit: Option<Duration>,
) -> std::io::Result<Outcome> {
    let out = File::create(log)?;
    let err = out.try_clone()?;
    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err));
    let start = Instant::now();
    let mut child = cmd.spawn()?;
    let status = match limit {
        None => child.wait()?,
        Some(limit) => loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if start.elapsed() > limit {
                child.kill()?;
                child.wait()?;
                append(
                    log,
                    &format!(
                        "\nxtask: stopped after its limit of {} s\n",
                        limit.as_secs()
                    ),
                )?;
                return Ok(Outcome {
                    success: false,
                    elapsed: start.elapsed(),
                });
            }
            std::thread::sleep(Duration::from_millis(50));
        },
    };
    Ok(Outcome {
        success: status.success(),
        elapsed: start.elapsed(),
    })
}

/// Runs `program` with `args`, in `dir` if given, and returns its standard output, trimmed,
/// if it ran and succeeded.
pub fn output(program: &str, args: &[&str], dir: Option<&Path>) -> Option<String> {
    let mut cmd = Command::new(program);
    cmd.args(args).stdin(Stdio::null()).stderr(Stdio::null());
    if let Some(dir) = dir {
        cmd.current_dir(dir);
    }
    let out = cmd.output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Appends `text` to the file at `path`.
pub fn append(path: &Path, text: &str) -> std::io::Result<()> {
    OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?
        .write_all(text.as_bytes())
}

/// A duration as a person reads it: `0.4 s`, `12 s`, `3 min 12 s`.
pub fn human(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 10.0 {
        format!("{secs:.1} s")
    } else if secs < 60.0 {
        format!("{secs:.0} s")
    } else {
        let whole = d.as_secs();
        format!("{} min {} s", whole / 60, whole % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_read_naturally() {
        assert_eq!(human(Duration::from_millis(420)), "0.4 s");
        assert_eq!(human(Duration::from_secs(12)), "12 s");
        assert_eq!(human(Duration::from_secs(192)), "3 min 12 s");
    }

    #[test]
    fn a_command_over_its_limit_is_stopped_and_fails() {
        let dir = std::env::temp_dir().join(format!("xtask-exec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("sleep.log");
        let mut cmd = Command::new("sleep");
        cmd.arg("5");
        let outcome = run_logged(cmd, &log, Some(Duration::from_millis(200))).unwrap();
        assert!(!outcome.success);
        assert!(outcome.elapsed < Duration::from_secs(4));
        assert!(
            std::fs::read_to_string(&log)
                .unwrap()
                .contains("stopped after its limit")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
