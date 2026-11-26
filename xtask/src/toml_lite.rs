//! A reader for the plain TOML that the workspace's manifests are written in: `[section]`
//! headers and `key = value` lines. It reads what the layering check needs, and no more:
//! values stay as written, and arrays over several lines are skipped.

use std::collections::BTreeMap;

/// Each section's keys and their values as written, by section name. Keys before any
/// header are under `""`.
pub type Sections = BTreeMap<String, BTreeMap<String, String>>;

/// Reads `text` into its sections, without comments.
pub fn sections(text: &str) -> Sections {
    let mut out = Sections::new();
    let mut current = String::new();
    for line in text.lines() {
        let line = strip_comment(line).trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            // An array of tables, such as `[[bin]]`, is never one the checks read.
            current = if header.starts_with('[') {
                String::from("[[array]]")
            } else {
                header.trim().to_string()
            };
        } else if let Some((key, value)) = line.split_once('=') {
            out.entry(current.clone()).or_default().insert(
                key.trim().trim_matches('"').to_string(),
                value.trim().to_string(),
            );
        }
    }
    out
}

/// The value of `key` in `section`, as written, if both are there.
pub fn value<'a>(sections: &'a Sections, section: &str, key: &str) -> Option<&'a str> {
    sections.get(section)?.get(key).map(String::as_str)
}

fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..i],
            _ => {}
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_sections_keys_and_values_without_comments() {
        let text = r##"
            name = "x" # a comment
            [lints]
            workspace = true
            [lints.clippy]
            all = { level = "warn", priority = -1 }
            "quoted" = "a # inside a string"
            [[bin]]
            name = "ignored"
        "##;
        let s = sections(text);
        assert_eq!(value(&s, "", "name"), Some("\"x\""));
        assert_eq!(value(&s, "lints", "workspace"), Some("true"));
        assert_eq!(
            value(&s, "lints.clippy", "all"),
            Some("{ level = \"warn\", priority = -1 }")
        );
        assert_eq!(
            value(&s, "lints.clippy", "quoted"),
            Some("\"a # inside a string\"")
        );
        assert_eq!(value(&s, "lints", "name"), None);
    }
}
