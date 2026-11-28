//! Tetera's errors as teta-wot's (ADR-0017).
//!
//! - **An action's failure** ends its invocation as `error`. A failure the device or the
//!   instrument foresaw (a bad argument, a busy or faulty device, a timeout, something not
//!   there) is teta-wot's handled error, its message led by its kind; a cancellation ends the
//!   invocation as `cancelled`; anything else is logged with its cause, as unexpected.
//! - **A property's failure:** a value the device refuses, or names something it doesn't
//!   have, is a 422, as an invalid value is; anything else is a 500 with its kind and message.

use teta_wot::prelude::{ActionError, Cancelled, PropertyError};
use teta_wot::validate::{LocItem, ValidationError};
use tetera_core::{Error, ErrorKind};

/// `error` as an action's failure.
pub fn action(error: Error) -> ActionError {
    match error.kind() {
        ErrorKind::Cancelled => ActionError::from(Cancelled),
        ErrorKind::Internal => ActionError::from(anyhow::Error::new(error)),
        kind => ActionError::handled(format!("{}: {error}", kind.name())),
    }
}

/// `error` as a property's failure, `value` the value written, if one was.
pub fn property(error: Error, value: Option<&serde_json::Value>) -> PropertyError {
    match error.kind() {
        ErrorKind::InvalidArgument | ErrorKind::NotFound => {
            PropertyError::Invalid(ValidationError::value_error(
                vec![LocItem::Key("body".into())],
                &error.to_string(),
                value.cloned().unwrap_or(serde_json::Value::Null),
            ))
        }
        kind => PropertyError::failed(anyhow::anyhow!("{}: {error}", kind.name())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_foreseen_failure_is_handled_and_led_by_its_kind() {
        for (made, kind) in [
            (Error::invalid_argument("a speed of -1"), "invalid_argument"),
            (Error::busy("the stage is moving"), "busy"),
            (Error::hardware("an alarm stopped the stage"), "hardware"),
            (Error::timeout("no report"), "timeout"),
            (Error::not_found("no objective 7x"), "not_found"),
        ] {
            let message = made.message().to_string();
            let failed = action(made);
            assert!(failed.is_handled(), "{failed}");
            assert!(!failed.is_cancelled());
            let shown = failed.to_string();
            assert!(
                shown.starts_with(kind) && shown.contains(&message),
                "{shown}"
            );
        }
        assert!(action(Error::cancelled("stopped")).is_cancelled());
        let unexpected = action(Error::internal("a bug"));
        assert!(!unexpected.is_handled() && !unexpected.is_cancelled());
        assert!(unexpected.to_string().contains("a bug"));
    }

    #[test]
    fn a_refused_value_is_a_422_and_anything_else_a_500() {
        let value = serde_json::json!(5000.0);
        let refused = property(Error::invalid_argument("at most 2,000 ms"), Some(&value));
        // The HTTP binding answers an invalid value with a 422 and its issues.
        assert!(matches!(refused, PropertyError::Invalid(_)), "{refused:?}");
        assert!(format!("{refused:?}").contains("at most 2,000 ms"));
        let failed = property(Error::hardware("the camera is gone"), None);
        let problem = failed.problem();
        assert_eq!(problem.status, Some(500));
        assert!(format!("{failed}").contains("hardware: the camera is gone"));
    }
}
