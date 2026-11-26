//! IDs: [`Id<T>`] is a UUID v7 typed by what it names, so a project's ID can't be passed
//! as a dataset's. IDs made by one process sort by when they were made (`uuid`'s `now_v7`
//! guarantees it), which keeps lists in creation order without a separate field.

use crate::error::{Error, Result};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::str::FromStr;
use uuid::Uuid;

/// The ID of a `T`.
pub struct Id<T> {
    uuid: Uuid,
    kind: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    /// A new ID, later than every ID this process made before it.
    pub fn new() -> Self {
        Self::from_uuid(Uuid::now_v7())
    }

    /// The ID that `uuid` is.
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self {
            uuid,
            kind: PhantomData,
        }
    }

    /// The UUID it is.
    pub const fn uuid(&self) -> Uuid {
        self.uuid
    }
}

impl<T> Default for Id<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.uuid.cmp(&other.uuid)
    }
}

impl<T> Hash for Id<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uuid.hash(state);
    }
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id({})", self.uuid)
    }
}

impl<T> fmt::Display for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.uuid.hyphenated().fmt(f)
    }
}

impl<T> FromStr for Id<T> {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Uuid::parse_str(s)
            .map(Self::from_uuid)
            .map_err(|_| Error::invalid_argument(format!("`{s}` isn't an ID")))
    }
}

impl<T> Serialize for Id<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de, T> Deserialize<'de> for Id<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = Cow::<str>::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

impl<T> JsonSchema for Id<T> {
    fn schema_name() -> Cow<'static, str> {
        "Id".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({ "type": "string", "format": "uuid" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    struct Project;
    struct Dataset;

    #[test]
    fn ids_sort_by_when_they_were_made() {
        let ids: Vec<Id<Project>> = (0..10_000).map(|_| Id::new()).collect();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(ids.iter().copied().collect::<HashSet<_>>().len(), ids.len());
        assert_eq!(ids[0].uuid().get_version_num(), 7);
    }

    #[test]
    fn ids_made_on_many_threads_are_all_different() {
        let ids: Vec<Id<Dataset>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| s.spawn(|| (0..1_000).map(|_| Id::new()).collect::<Vec<_>>()))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), 8_000);
    }

    #[test]
    fn ids_print_parse_and_serialise_as_uuids() {
        let id: Id<Project> = Id::new();
        let text = id.to_string();
        assert_eq!(text.len(), 36);
        assert_eq!(text.parse::<Id<Project>>().unwrap(), id);
        assert_eq!(format!("{id:?}"), format!("Id({text})"));
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{text}\""));
        assert_eq!(serde_json::from_str::<Id<Project>>(&json).unwrap(), id);
        assert!("not-an-id".parse::<Id<Project>>().is_err());
        assert!(serde_json::from_str::<Id<Project>>("\"x\"").is_err());
        assert_eq!(Id::<Project>::schema_name(), "Id");
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn a_copy_or_clone_is_the_same_id_and_hashes_the_same() {
        use std::hash::{BuildHasher, RandomState};
        let id: Id<Project> = Id::new();
        assert_eq!(id.clone(), id);
        let hasher = RandomState::new();
        assert_eq!(hasher.hash_one(id), hasher.hash_one(id.clone()));
        assert_ne!(hasher.hash_one(id), hasher.hash_one(Id::<Project>::new()));
    }

    #[test]
    fn an_ids_schema_is_a_uuid_string() {
        let schema = schemars::schema_for!(Id<Project>);
        assert_eq!(schema.get("format").and_then(|f| f.as_str()), Some("uuid"));
    }

    #[test]
    fn a_default_id_is_a_new_one() {
        let (a, b) = (Id::<Project>::default(), Id::<Project>::default());
        assert!(a < b);
    }
}
