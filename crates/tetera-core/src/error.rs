//! Errors: one [`Error`] type with a [`ErrorKind`] that callers act on, a message, the
//! context it passed through on its way up, and the error that caused it, if any.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;

/// What kind of failure an error is: what a caller, or the API's status code, acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The request is wrong: a value out of range, a malformed string.
    InvalidArgument,
    /// What was asked for isn't there.
    NotFound,
    /// It can't be done now: the device or resource is in use.
    Busy,
    /// A device failed or reported a fault.
    Hardware,
    /// It took longer than allowed.
    Timeout,
    /// It was cancelled before it finished.
    Cancelled,
    /// Tetera itself went wrong.
    Internal,
}

impl ErrorKind {
    /// Its name, as it serialises: `invalid_argument`, `not_found`, and so on.
    pub fn name(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid_argument",
            Self::NotFound => "not_found",
            Self::Busy => "busy",
            Self::Hardware => "hardware",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.name())
    }
}

/// An error: its kind, its message, the context it was raised in, and its cause.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    context: Vec<String>,
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

/// Tetera's result.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// An error of `kind` that says `message`.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            context: Vec::new(),
            source: None,
        }
    }

    /// An [`ErrorKind::InvalidArgument`] error.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidArgument, message)
    }

    /// An [`ErrorKind::NotFound`] error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotFound, message)
    }

    /// An [`ErrorKind::Busy`] error.
    pub fn busy(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Busy, message)
    }

    /// An [`ErrorKind::Hardware`] error.
    pub fn hardware(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Hardware, message)
    }

    /// An [`ErrorKind::Timeout`] error.
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Timeout, message)
    }

    /// An [`ErrorKind::Cancelled`] error.
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Cancelled, message)
    }

    /// An [`ErrorKind::Internal`] error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }

    /// The same error, caused by `source`.
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// The same error, raised while doing `what`: `what` is printed before the message.
    pub fn context(mut self, what: impl Into<String>) -> Self {
        self.context.push(what.into());
        self
    }

    /// Its kind.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Its own message, without the context.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The context it passed through, outermost first.
    pub fn contexts(&self) -> impl Iterator<Item = &str> {
        self.context.iter().rev().map(String::as_str)
    }
}

impl fmt::Display for Error {
    /// The context, outermost first, then the message: `capturing: reading the camera: busy`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for what in self.contexts() {
            write!(f, "{what}: ")?;
        }
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

/// Adds context to a result's error on its way up.
pub trait ResultExt<T> {
    /// The same result, its error raised while doing `what`.
    fn context(self, what: impl Into<String>) -> Result<T>;

    /// The same result, its error raised while doing what `what` says; `what` runs only
    /// on an error.
    fn with_context(self, what: impl FnOnce() -> String) -> Result<T>;
}

impl<T> ResultExt<T> for Result<T> {
    fn context(self, what: impl Into<String>) -> Result<T> {
        self.map_err(|e| e.context(what))
    }

    fn with_context(self, what: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|e| e.context(what()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn context_reads_outermost_first() {
        let err = Error::busy("the camera is streaming")
            .context("setting the exposure")
            .context("capturing a burst");
        assert_eq!(
            err.to_string(),
            "capturing a burst: setting the exposure: the camera is streaming"
        );
        assert_eq!(err.kind(), ErrorKind::Busy);
        assert_eq!(err.message(), "the camera is streaming");
        assert_eq!(
            err.contexts().collect::<Vec<_>>(),
            ["capturing a burst", "setting the exposure"]
        );
    }

    #[test]
    fn results_gain_context_only_on_failure() {
        let failed: Result<u32> = Err(Error::timeout("no frame in 2 s"));
        let err = failed.context("autofocus").unwrap_err();
        assert_eq!(err.to_string(), "autofocus: no frame in 2 s");
        let mut ran = false;
        let ok: Result<u32> = Ok(3);
        assert_eq!(
            ok.with_context(|| {
                ran = true;
                "never".into()
            })
            .unwrap(),
            3
        );
        assert!(!ran, "the context isn't built for a success");
        let failed: Result<u32> = Err(Error::not_found("dataset 7"));
        assert_eq!(
            failed
                .with_context(|| "opening".into())
                .unwrap_err()
                .to_string(),
            "opening: dataset 7"
        );
    }

    #[test]
    fn a_cause_is_its_source() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err = Error::hardware("the port won't open").with_source(io);
        assert_eq!(err.source().unwrap().to_string(), "denied");
        assert!(Error::internal("x").source().is_none());
    }

    #[test]
    fn each_constructor_gives_its_kind_and_name() {
        let cases = [
            (
                Error::invalid_argument("x"),
                ErrorKind::InvalidArgument,
                "invalid_argument",
            ),
            (Error::not_found("x"), ErrorKind::NotFound, "not_found"),
            (Error::busy("x"), ErrorKind::Busy, "busy"),
            (Error::hardware("x"), ErrorKind::Hardware, "hardware"),
            (Error::timeout("x"), ErrorKind::Timeout, "timeout"),
            (Error::cancelled("x"), ErrorKind::Cancelled, "cancelled"),
            (Error::internal("x"), ErrorKind::Internal, "internal"),
        ];
        for (err, kind, name) in cases {
            assert_eq!(err.kind(), kind);
            assert_eq!(kind.to_string(), name);
            assert_eq!(format!("{kind:>20}").trim_start(), name);
            assert_eq!(serde_json::to_string(&kind).unwrap(), format!("\"{name}\""));
        }
    }
}
