//! What the `system` Thing reports of the host: the platform, memory and disks, read from
//! Linux's own files and tools. Each reader has a parser of its own, tested on samples, and
//! reports nothing, rather than failing, where the host doesn't have what it reads.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// The operating system Tetera runs on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Platform {
    /// The operating system: `linux`, `windows`, …
    pub os: String,
    /// The kernel's release, such as `6.6.87.2-microsoft-standard-WSL2`.
    pub kernel: Option<String>,
    /// The distribution, such as `Ubuntu 24.04.3 LTS`.
    pub distribution: Option<String>,
    /// Inside WSL, which, and WSL's own version when Windows says: `WSL2 2.6.1.0`.
    pub wsl: Option<String>,
}

/// The host's memory, bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Memory {
    /// In all.
    pub total: u64,
    /// Available to start new work without swapping.
    pub available: u64,
}

/// A disk Tetera writes to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Disk {
    /// What Tetera keeps there: `data` or `scratch`.
    pub role: String,
    /// The folder.
    pub path: String,
    /// The file system's size, bytes.
    pub total: u64,
    /// Free to Tetera, bytes.
    pub free: u64,
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// The platform: the kernel, the distribution, and WSL's version, which asks Windows and so
/// takes a moment.
pub fn platform() -> Platform {
    let kernel = read("/proc/sys/kernel/osrelease");
    let distribution = read("/etc/os-release").and_then(|t| os_release_name(&t));
    let wsl = kernel
        .as_deref()
        .and_then(wsl_from_kernel)
        .map(|which| match wsl_version() {
            Some(version) => format!("{which} {version}"),
            None => which.to_string(),
        });
    Platform {
        os: std::env::consts::OS.to_string(),
        kernel,
        distribution,
        wsl,
    }
}

/// The distribution's name from `/etc/os-release`: its `PRETTY_NAME`, else its `NAME`.
pub fn os_release_name(text: &str) -> Option<String> {
    let value = |key: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
            .map(|v| v.trim().trim_matches('"').to_string())
            .filter(|v| !v.is_empty())
    };
    value("PRETTY_NAME").or_else(|| value("NAME"))
}

/// Which WSL a kernel's release is: WSL2's kernels are `microsoft-standard-WSL2`, WSL1's
/// report `Microsoft`.
pub fn wsl_from_kernel(release: &str) -> Option<&'static str> {
    let release = release.to_ascii_lowercase();
    if release.contains("microsoft-standard-wsl2") {
        Some("WSL2")
    } else if release.contains("microsoft") {
        Some("WSL1")
    } else {
        None
    }
}

/// WSL's version, from `wsl.exe --version` through interop, if Windows answers.
pub fn wsl_version() -> Option<String> {
    let out = Command::new("wsl.exe").arg("--version").output().ok()?;
    out.status
        .success()
        .then(|| parse_wsl_version(&out.stdout))
        .flatten()
}

/// The version on `wsl.exe --version`'s first line, which Windows writes in UTF-16:
/// `WSL version: 2.6.1.0` gives `2.6.1.0`.
pub fn parse_wsl_version(bytes: &[u8]) -> Option<String> {
    let text = if bytes.len() >= 2 && bytes.iter().skip(1).step_by(2).all(|b| *b == 0) {
        let units: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_le_bytes(*p))
            .collect();
        String::from_utf16_lossy(&units)
    } else {
        String::from_utf8_lossy(bytes).into_owned()
    };
    let first = text.trim_start_matches('\u{feff}').lines().next()?;
    let version = first.rsplit(':').next()?.trim();
    (!version.is_empty() && version.chars().all(|c| c.is_ascii_digit() || c == '.'))
        .then(|| version.to_string())
}

/// The host's memory, from `/proc/meminfo`.
pub fn memory() -> Option<Memory> {
    parse_meminfo(&read("/proc/meminfo")?)
}

/// `MemTotal` and `MemAvailable` from `/proc/meminfo`, in bytes.
pub fn parse_meminfo(text: &str) -> Option<Memory> {
    let kib = |key: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(key))?;
        let number = line.split_whitespace().nth(1)?;
        number.parse::<u64>().ok().map(|v| v * 1024)
    };
    Some(Memory {
        total: kib("MemTotal:")?,
        available: kib("MemAvailable:")?,
    })
}

/// The disk that holds `path`, Tetera's `role` for it, from `df`.
pub fn disk(role: &str, path: &Path) -> Option<Disk> {
    let out = Command::new("df")
        .args(["-P", "-B1"])
        .arg(path)
        .output()
        .ok()?;
    let (total, free) = parse_df(&String::from_utf8_lossy(&out.stdout))?;
    Some(Disk {
        role: role.to_string(),
        path: path.display().to_string(),
        total,
        free,
    })
}

/// The size and the free space from `df -P -B1`'s second line: its second and fourth
/// columns.
pub fn parse_df(text: &str) -> Option<(u64, u64)> {
    let line = text.lines().nth(1)?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    let number = |k: usize| fields.get(k)?.parse::<u64>().ok();
    Some((number(1)?, number(3)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_distribution_is_named_from_os_release() {
        let ubuntu = "NAME=\"Ubuntu\"\nVERSION_ID=\"24.04\"\nPRETTY_NAME=\"Ubuntu 24.04.3 LTS\"\n";
        assert_eq!(
            os_release_name(ubuntu).as_deref(),
            Some("Ubuntu 24.04.3 LTS")
        );
        assert_eq!(os_release_name("NAME=Debian\n").as_deref(), Some("Debian"));
        assert_eq!(
            os_release_name("PRETTY_NAME=\"\"\nNAME=Alpine").as_deref(),
            Some("Alpine")
        );
        assert_eq!(os_release_name("ID=x\n"), None);
    }

    #[test]
    fn wsl_is_told_from_the_kernel() {
        assert_eq!(
            wsl_from_kernel("6.6.87.2-microsoft-standard-WSL2"),
            Some("WSL2")
        );
        assert_eq!(wsl_from_kernel("4.4.0-19041-Microsoft"), Some("WSL1"));
        assert_eq!(wsl_from_kernel("6.8.0-48-generic"), None);
    }

    #[test]
    fn wsls_version_is_read_in_utf16_or_utf8() {
        let text = "WSL version: 2.6.1.0\r\nKernel version: 6.6.87.2-1\r\n";
        let utf16: Vec<u8> = text.encode_utf16().flat_map(u16::to_le_bytes).collect();
        assert_eq!(parse_wsl_version(&utf16).as_deref(), Some("2.6.1.0"));
        assert_eq!(
            parse_wsl_version(text.as_bytes()).as_deref(),
            Some("2.6.1.0")
        );
        // Another language: the label translated, the version the same.
        let german: Vec<u8> = "WSL-Version: 2.6.1.0\r\n"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(parse_wsl_version(&german).as_deref(), Some("2.6.1.0"));
        assert_eq!(parse_wsl_version(b"Usage: wsl.exe [Argument]"), None);
        assert_eq!(parse_wsl_version(b""), None);
    }

    #[test]
    fn memory_is_read_from_meminfo() {
        let text =
            "MemTotal:       16303536 kB\nMemFree:         1000 kB\nMemAvailable:   12000000 kB\n";
        assert_eq!(
            parse_meminfo(text),
            Some(Memory {
                total: 16_303_536 * 1024,
                available: 12_000_000 * 1024
            })
        );
        assert_eq!(parse_meminfo("MemTotal: 5 kB\n"), None, "no MemAvailable");
        assert!(memory().is_some_and(|m| m.total > 0 && m.available <= m.total));
    }

    #[test]
    fn a_disk_is_read_from_df() {
        let text = "Filesystem         1-blocks        Used    Available Capacity Mounted on\n\
                    /dev/sdd      1081101176832 99540361216 926561841152      10% /\n";
        assert_eq!(parse_df(text), Some((1_081_101_176_832, 926_561_841_152)));
        assert_eq!(parse_df("Filesystem\n"), None);
        assert_eq!(parse_df(""), None);
        let here = disk("data", Path::new(".")).expect("df answers");
        assert!(here.total > 0 && here.free <= here.total && here.role == "data");
    }

    #[test]
    fn the_platform_is_this_one() {
        let p = platform();
        assert_eq!(p.os, std::env::consts::OS);
        if cfg!(target_os = "linux") {
            assert!(p.kernel.is_some() && p.distribution.is_some(), "{p:?}");
        }
    }
}
