//! Tetera's foundation layer, L0.
//!
//! It will hold the types every other crate shares: units, geometry, the clock, IDs,
//! frames and their metadata, and errors (plan §6.2). It does no I/O and has no async
//! code. P1.01 creates it nearly empty, so the gates and the debugger have a crate to
//! run; P1.02 and P1.03 fill it.

/// Tetera's version, as Cargo knows it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Tetera's name with its version, such as `Tetera 0.1.0`.
pub fn name_and_version() -> String {
    format!("Tetera {VERSION}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_and_version_carries_the_package_version() {
        assert_eq!(
            name_and_version(),
            format!("Tetera {}", env!("CARGO_PKG_VERSION"))
        );
    }
}
