//! The objectives (plan Q24): a catalogue, read from `configs/objectives.toml`, and the mount,
//! where the user says which objective is fitted. They are swapped by hand.

use crate::event::Events;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, PoisonError};
use tetera_core::{Error, ErrorKind, Result};

/// An objective, as the catalogue describes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    /// The name the user knows it by, such as `10x`.
    pub name: String,
    /// Its maker and model.
    pub model: String,
    /// Its magnification, with the tube lens it was made for.
    pub magnification: f64,
    /// Its numerical aperture.
    pub numerical_aperture: f64,
    /// From its front to the sample in focus, in mm.
    pub working_distance_mm: f64,
}

/// The objectives an instrument has: dry ones, each named once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveCatalogue {
    /// Each objective, written `[[objective]]` in TOML.
    #[serde(rename = "objective")]
    pub objectives: Vec<Objective>,
}

impl ObjectiveCatalogue {
    /// The catalogue written in `text`, as TOML, checked.
    pub fn from_toml(text: &str) -> Result<Self> {
        let catalogue: Self = toml::from_str(text).map_err(|e| {
            Error::invalid_argument(format!("the objective catalogue: {}", e.message()))
        })?;
        catalogue.check()?;
        Ok(catalogue)
    }

    /// The catalogue in the file at `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            let kind = match e.kind() {
                std::io::ErrorKind::NotFound => ErrorKind::NotFound,
                _ => ErrorKind::Internal,
            };
            Error::new(kind, format!("couldn't read {}: {e}", path.display()))
        })?;
        Self::from_toml(&text).map_err(|e| e.context(format!("reading {}", path.display())))
    }

    /// Whether every objective is sound and named once, saying which isn't and why.
    pub fn check(&self) -> Result<()> {
        if self.objectives.is_empty() {
            return Err(Error::invalid_argument(
                "the objective catalogue has no objectives",
            ));
        }
        for (i, o) in self.objectives.iter().enumerate() {
            let fault = if o.name.trim().is_empty() {
                "it has no name"
            } else if self.objectives[..i].iter().any(|p| p.name == o.name) {
                "it is named twice"
            } else if !(o.magnification.is_finite() && o.magnification > 0.0) {
                "its magnification is above zero"
            } else if !(o.numerical_aperture > 0.0 && o.numerical_aperture < 1.0) {
                "a dry objective's numerical aperture is between 0 and 1"
            } else if !(o.working_distance_mm.is_finite() && o.working_distance_mm > 0.0) {
                "its working distance is above zero"
            } else {
                continue;
            };
            return Err(Error::invalid_argument(format!(
                "objective {} ({:?}): {fault}",
                i + 1,
                o.name
            )));
        }
        Ok(())
    }

    /// The objective named `name`.
    pub fn get(&self, name: &str) -> Result<&Objective> {
        self.objectives
            .iter()
            .find(|o| o.name == name)
            .ok_or_else(|| {
                Error::not_found(format!(
                    "no objective named {name:?}; the catalogue has {}",
                    self.names().join(", ")
                ))
            })
    }

    /// The objectives' names, in order.
    pub fn names(&self) -> Vec<&str> {
        self.objectives.iter().map(|o| o.name.as_str()).collect()
    }
}

/// What an objective mount tells its subscribers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ObjectiveEvent {
    /// This objective is fitted now, or none is.
    Fitted {
        /// The objective.
        objective: Option<Objective>,
    },
}

/// Where the objective is fitted: which one it is, from a catalogue.
pub trait ObjectiveMount: Send + Sync {
    /// The objectives it can take.
    fn catalogue(&self) -> ObjectiveCatalogue;

    /// The objective fitted, if any.
    fn fitted(&self) -> Option<Objective>;

    /// Says which objective is fitted, by its name in the catalogue, or that none is.
    fn fit(&self, name: Option<&str>) -> Result<()>;

    /// A channel of its events from now on.
    fn subscribe(&self) -> Receiver<ObjectiveEvent>;
}

/// A mount where objectives are swapped by hand, and the user says which is fitted. None is,
/// to begin with.
#[derive(Debug)]
pub struct ManualObjectiveMount {
    catalogue: ObjectiveCatalogue,
    fitted: Mutex<Option<Objective>>,
    events: Events<ObjectiveEvent>,
}

impl ManualObjectiveMount {
    /// A mount for the objectives in `catalogue`.
    pub fn new(catalogue: ObjectiveCatalogue) -> Self {
        Self {
            catalogue,
            fitted: Mutex::new(None),
            events: Events::default(),
        }
    }
}

impl ObjectiveMount for ManualObjectiveMount {
    fn catalogue(&self) -> ObjectiveCatalogue {
        self.catalogue.clone()
    }

    fn fitted(&self) -> Option<Objective> {
        self.fitted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn fit(&self, name: Option<&str>) -> Result<()> {
        let objective = name
            .map(|name| self.catalogue.get(name).cloned())
            .transpose()?;
        *self.fitted.lock().unwrap_or_else(PoisonError::into_inner) = objective.clone();
        self.events.emit(ObjectiveEvent::Fitted { objective });
        Ok(())
    }

    fn subscribe(&self) -> Receiver<ObjectiveEvent> {
        self.events.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIPPED: &str = include_str!("../../../configs/objectives.toml");

    fn one(fields: &str) -> String {
        format!("[[objective]]\n{fields}\n")
    }

    const TEN: &str = r#"name = "10x"
model = "a 10x"
magnification = 10.0
numerical_aperture = 0.28
working_distance_mm = 33.5"#;

    #[test]
    fn the_shipped_catalogue_has_the_four_mitutoyo_objectives() {
        let catalogue = ObjectiveCatalogue::from_toml(SHIPPED).unwrap();
        assert_eq!(catalogue.names(), ["5x", "10x", "20x", "50x"]);
        let table: Vec<(f64, f64, f64)> = catalogue
            .objectives
            .iter()
            .map(|o| (o.magnification, o.numerical_aperture, o.working_distance_mm))
            .collect();
        assert_eq!(
            table,
            [
                (5.0, 0.14, 34.0),
                (10.0, 0.28, 33.5),
                (20.0, 0.42, 20.0),
                (50.0, 0.55, 13.0)
            ]
        );
        assert!(catalogue.get("20x").unwrap().model.contains("Mitutoyo"));
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/objectives.toml");
        assert_eq!(ObjectiveCatalogue::load(&path).unwrap(), catalogue);
    }

    #[test]
    fn each_thing_wrong_with_a_catalogue_is_said() {
        let without = |field: &str| {
            TEN.lines()
                .filter(|l| !l.starts_with(field))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let with = |field: &str, value: &str| format!("{}\n{field} = {value}", without(field));
        for (text, says) in [
            (String::new(), "missing field `objective`"),
            ("objective = []".to_string(), "has no objectives"),
            (one(&without("model")), "missing field `model`"),
            (one(&format!("{TEN}\nzoom = 2")), "unknown field `zoom`"),
            (
                one(&with("name", r#"" ""#)),
                "objective 1 (\" \"): it has no name",
            ),
            (
                format!("{}{}", one(TEN), one(TEN)),
                "objective 2 (\"10x\"): it is named twice",
            ),
            (
                one(&with("magnification", "0.0")),
                "magnification is above zero",
            ),
            (
                one(&with("magnification", "inf")),
                "magnification is above zero",
            ),
            (one(&with("numerical_aperture", "0.0")), "between 0 and 1"),
            (one(&with("numerical_aperture", "1.0")), "between 0 and 1"),
            (one(&with("numerical_aperture", "nan")), "between 0 and 1"),
            (
                one(&with("working_distance_mm", "0.0")),
                "working distance is above zero",
            ),
            (
                one(&with("working_distance_mm", "-1.0")),
                "working distance is above zero",
            ),
            (
                one(&with("working_distance_mm", "inf")),
                "working distance is above zero",
            ),
            ("[[objective]\n".to_string(), "the objective catalogue: "),
        ] {
            let e = ObjectiveCatalogue::from_toml(&text).unwrap_err();
            assert_eq!(e.kind(), ErrorKind::InvalidArgument, "{text}");
            assert!(e.message().contains(says), "{text}: {e}");
        }
        assert!(ObjectiveCatalogue::from_toml(&one(TEN)).is_ok());
    }

    #[test]
    fn a_missing_or_bad_file_says_which() {
        let dir = std::env::temp_dir().join(format!("tetera-objectives-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("missing.toml");
        let e = ObjectiveCatalogue::load(&missing).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::NotFound);
        assert!(e.message().contains("missing.toml"), "{e}");
        let e = ObjectiveCatalogue::load(&dir).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::Internal, "a folder: {e}");
        let bad = dir.join("bad.toml");
        std::fs::write(&bad, "objective = []").unwrap();
        let e = ObjectiveCatalogue::load(&bad).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidArgument);
        assert!(e.contexts().any(|c| c.contains("bad.toml")), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_user_says_which_objective_is_fitted() {
        let mount = ManualObjectiveMount::new(ObjectiveCatalogue::from_toml(SHIPPED).unwrap());
        let events = mount.subscribe();
        assert_eq!(mount.fitted(), None);
        assert_eq!(mount.catalogue().names().len(), 4);
        mount.fit(Some("20x")).unwrap();
        assert_eq!(mount.fitted().unwrap().numerical_aperture, 0.42);
        let e = mount.fit(Some("40x")).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::NotFound);
        assert!(
            e.message().contains("the catalogue has 5x, 10x, 20x, 50x"),
            "{e}"
        );
        assert_eq!(mount.fitted().unwrap().name, "20x", "unchanged");
        mount.fit(None).unwrap();
        assert_eq!(mount.fitted(), None);
        let seen: Vec<Option<String>> = events
            .try_iter()
            .map(|ObjectiveEvent::Fitted { objective }| objective.map(|o| o.name))
            .collect();
        assert_eq!(seen, [Some("20x".to_string()), None]);
        let json = serde_json::to_string(&ObjectiveEvent::Fitted { objective: None }).unwrap();
        assert_eq!(json, r#"{"event":"fitted","objective":null}"#);
    }
}
