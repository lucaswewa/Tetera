//! The server's configuration: `configs/tetera.toml` (plan D16). Every key has a default, so
//! a file names only what it changes; a key the server doesn't know is refused, with its
//! name. Relative paths in it are from the file's own folder, except the data root, which
//! is from where the server runs: `./.tetera/` in development.

use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};
use tetera_api::{ObjectivesConfig, SimulatorConfig};
use tetera_core::{Error, Result};
use tetera_log::LogConfig;

/// Where the server listens: `[server]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Listen {
    /// The address: `127.0.0.1`, this PC only (Q11).
    pub address: IpAddr,
    /// The port: 8310, beside OScope's 8210 (Q48).
    pub port: u16,
}

impl Default for Listen {
    fn default() -> Self {
        Self {
            address: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 8310,
        }
    }
}

/// Where Tetera keeps its files: `[data]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Data {
    /// Datasets and settings: `./.tetera/` in development, `/var/lib/tetera/` for the
    /// service.
    pub root: PathBuf,
    /// Raw frame logs and other scratch; `scratch` in the data root when not set.
    pub scratch: Option<PathBuf>,
}

impl Default for Data {
    fn default() -> Self {
        Self {
            root: PathBuf::from(".tetera"),
            scratch: None,
        }
    }
}

impl Data {
    /// The scratch root.
    pub fn scratch(&self) -> PathBuf {
        self.scratch
            .clone()
            .unwrap_or_else(|| self.root.join("scratch"))
    }
}

/// The server's configuration.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Where it listens.
    pub server: Listen,
    /// Where it keeps its files.
    pub data: Data,
    /// The simulated instrument.
    pub simulator: SimulatorConfig,
    /// The objectives.
    pub objectives: ObjectivesConfig,
    /// Logging.
    pub log: LogConfig,
}

/// Where the server looks for its configuration when none is named.
pub const DEFAULT_PATH: &str = "configs/tetera.toml";

impl Config {
    /// The configuration in `text`, its relative paths from `folder`.
    pub fn from_toml(text: &str, folder: &Path) -> Result<Self> {
        let mut config: Config = toml::from_str(text)
            .map_err(|e| Error::invalid_argument(format!("the configuration: {e}")))?;
        if let Some(catalogue) = &config.objectives.catalogue
            && catalogue.is_relative()
        {
            config.objectives.catalogue = Some(folder.join(catalogue));
        }
        if let Some(dir) = config.log.file.as_mut().map(|f| &mut f.dir)
            && dir.is_relative()
        {
            *dir = config.data.root.join(&*dir);
        }
        if config.simulator.clock_rate.is_nan() || config.simulator.clock_rate < 1.0 {
            return Err(Error::invalid_argument(format!(
                "the configuration: the simulator's clock runs at 1 or more, not {}",
                config.simulator.clock_rate
            )));
        }
        Ok(config)
    }

    /// The configuration in the file at `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            Error::not_found(format!(
                "the configuration {} won't open: {e}",
                path.display()
            ))
        })?;
        let folder = path.parent().unwrap_or(Path::new("."));
        Self::from_toml(&text, folder).map_err(|e| e.context(path.display().to_string()))
    }

    /// The configuration in `named` if given; otherwise in [`DEFAULT_PATH`] if it is there;
    /// otherwise the defaults. Also where it came from.
    pub fn find(named: Option<&Path>) -> Result<(Self, Option<PathBuf>)> {
        Self::find_in(named, Path::new("."))
    }

    /// The same, [`DEFAULT_PATH`] looked for in `folder`.
    pub fn find_in(named: Option<&Path>, folder: &Path) -> Result<(Self, Option<PathBuf>)> {
        let default = folder.join(DEFAULT_PATH);
        match named {
            Some(path) => Ok((Self::load(path)?, Some(path.to_path_buf()))),
            None if default.is_file() => Ok((Self::load(&default)?, Some(default))),
            None => Ok((Self::default(), None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetera_api::{Sample, SensorSize};

    #[test]
    fn the_defaults_are_the_plans() {
        let c = Config::default();
        assert_eq!(c.server.address.to_string(), "127.0.0.1");
        assert_eq!(c.server.port, 8310);
        assert_eq!(c.data.root, PathBuf::from(".tetera"));
        assert_eq!(c.data.scratch(), PathBuf::from(".tetera/scratch"));
        assert_eq!(c.simulator.sample, Some(Sample::default()));
        assert_eq!(c.simulator.sensor, SensorSize::Auto);
        assert!(c.simulator.gpu);
        assert_eq!(c.objectives.fitted.as_deref(), Some("20x"));
        assert_eq!(c.objectives.catalogue, None);
        assert_eq!(c.log, LogConfig::default());
        assert_eq!(
            Config::from_toml("", Path::new(".")).unwrap(),
            c,
            "an empty file"
        );
    }

    #[test]
    fn the_shipped_file_is_the_defaults_but_for_its_catalogue() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let shipped = Config::load(&root.join(DEFAULT_PATH)).unwrap();
        let catalogue = shipped.objectives.catalogue.clone().expect("a catalogue");
        assert!(
            catalogue.ends_with("configs/./objectives.toml")
                || catalogue.ends_with("configs/objectives.toml")
        );
        assert!(catalogue.is_file(), "{}", catalogue.display());
        assert_eq!(
            Config {
                objectives: ObjectivesConfig {
                    catalogue: None,
                    ..shipped.objectives.clone()
                },
                ..shipped
            },
            Config::default()
        );
    }

    #[test]
    fn a_file_names_only_what_it_changes_and_relative_paths_are_from_it() {
        let text = r#"
            [server]
            port = 9000
            [data]
            root = "/var/lib/tetera"
            [simulator]
            gpu = false
            sensor = "small"
            clock_rate = 10.0
            sample = { seed = 3, thickness = 3.5, folds = true }
            [objectives]
            catalogue = "lenses.toml"
            fitted = "50x"
            [log]
            filter = "debug"
            file = { dir = "logs" }
        "#;
        let c = Config::from_toml(text, Path::new("/etc/tetera")).unwrap();
        assert_eq!(
            (c.server.address.to_string(), c.server.port),
            ("127.0.0.1".into(), 9000)
        );
        assert_eq!(c.data.scratch(), PathBuf::from("/var/lib/tetera/scratch"));
        assert_eq!(
            c.simulator.sample,
            Some(Sample {
                seed: 3,
                thickness: 3.5,
                folds: true
            })
        );
        assert_eq!(
            (c.simulator.gpu, c.simulator.sensor, c.simulator.clock_rate),
            (false, SensorSize::Small, 10.0)
        );
        assert_eq!(
            c.objectives.catalogue,
            Some(PathBuf::from("/etc/tetera/lenses.toml"))
        );
        assert_eq!(c.log.filter, "debug");
        assert_eq!(
            c.log.file.map(|f| f.dir),
            Some(PathBuf::from("/var/lib/tetera/logs")),
            "logs in the data root"
        );
        let absolute = Config::from_toml(
            "[objectives]\ncatalogue = \"/opt/lenses.toml\"",
            Path::new("/etc/tetera"),
        )
        .unwrap();
        assert_eq!(
            absolute.objectives.catalogue,
            Some(PathBuf::from("/opt/lenses.toml"))
        );
    }

    #[test]
    fn a_bad_configuration_is_refused_with_what_is_wrong() {
        for (text, says) in [
            ("[server]\nprot = 8000", "prot"),
            ("[server]\nport = \"eighty\"", "port"),
            ("[server]\nport = 70000", "port"),
            ("[server]\naddress = \"localhost\"", "address"),
            ("[simulator]\nsensor = \"huge\"", "huge"),
            ("[simulator]\nclock_rate = 0.5", "clock"),
            (
                "[simulator]\nsample = { seed = 1, colour = \"red\" }",
                "colour",
            ),
            ("[stage]\nspeed = 1", "stage"),
            ("server = 3", "server"),
        ] {
            let e = Config::from_toml(text, Path::new(".")).expect_err(text);
            assert_eq!(e.kind(), tetera_core::ErrorKind::InvalidArgument);
            assert!(e.to_string().contains(says), "{text}: {e}");
        }
        let missing = Config::load(Path::new("/nowhere/tetera.toml")).unwrap_err();
        assert_eq!(missing.kind(), tetera_core::ErrorKind::NotFound);
        assert!(missing.to_string().contains("/nowhere/tetera.toml"));
        assert!(Config::find(Some(Path::new("/nowhere/tetera.toml"))).is_err());
    }

    #[test]
    fn without_a_named_file_it_looks_for_configs_tetera_toml() {
        let folder =
            std::env::temp_dir().join(format!("tetera-config-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        std::fs::create_dir_all(&folder).unwrap();
        let (config, from) = Config::find_in(None, &folder).unwrap();
        assert_eq!(
            (config, from),
            (Config::default(), None),
            "nothing there: the defaults"
        );
        std::fs::create_dir_all(folder.join("configs")).unwrap();
        std::fs::write(folder.join(DEFAULT_PATH), "[server]\nport = 9200\n").unwrap();
        let (config, from) = Config::find_in(None, &folder).unwrap();
        assert_eq!(config.server.port, 9200);
        assert_eq!(from, Some(folder.join(DEFAULT_PATH)));
        let named = folder.join("other.toml");
        std::fs::write(&named, "[server]\nport = 9300\n").unwrap();
        let (config, from) = Config::find_in(Some(&named), &folder).unwrap();
        assert_eq!((config.server.port, from), (9300, Some(named)));
        std::fs::write(folder.join(DEFAULT_PATH), "[server]\nport = -1\n").unwrap();
        assert!(
            Config::find_in(None, &folder).is_err(),
            "a bad file there is refused"
        );
        let _ = std::fs::remove_dir_all(&folder);
    }
}
