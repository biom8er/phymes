use anyhow::{Result, anyhow};
use clap::ValueEnum;
use diff_match_patch_rs::{DiffMatchPatch, Efficient};
use phymes_diagnostics::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::patch::apply_v4a_diff::apply_v4a_patch;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, ValueEnum, Default)]
pub enum PatchOperator {
    #[default]
    #[value(name = "Create")]
    Create,
    #[value(name = "Update")]
    Update,
    #[value(name = "Delete")]
    Delete,
}
impl std::fmt::Display for PatchOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create => write!(f, "Create"),
            Self::Update => write!(f, "Update"),
            Self::Delete => write!(f, "Delete"),
        }
    }
}

#[derive(Debug)]
pub struct PatchOperation {
    pub path: std::path::PathBuf,
    pub diff: String,
    pub operator: PatchOperator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, ValueEnum)]
pub enum DiffType {
    /// DMP (Universal Diff format)
    #[default]
    #[value(name = "Dmp")]
    Dmp,
    /// V4A based on markers instead of line numbers
    #[value(name = "V4A")]
    V4A,
    /// HashMap merging
    #[value(name = "Map")]
    Map,
    #[value(name = "Unknown")]
    #[serde(other)]
    Unknown,
}
impl std::fmt::Display for DiffType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dmp => write!(f, "Dmp"),
            Self::V4A => write!(f, "V4A"),
            Self::Map => write!(f, "Map"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Heuristic classifier for V4A vs DMP.
///
/// ### DMP features:
/// - `@@ -<n>,<n> +<n>,<n> @@` style header
/// 
/// ### V4A features:
/// - bare `@@` or `@@ <context>` without line/col metadata
/// - section markers: `*** End Patch`, `*** End of File`, `*** Update File:`,
///   `*** Add File:`, `*** Delete File:`
/// - leading `+`/`-`/` ` lines without any DMP header present
/// 
/// ### Map features:
/// - leading and trailing brackets i.e., `{...}`
/// - JSON deserializable
fn classify_diff(diff: &str) -> DiffType {
    if serde_json::from_str::<Map<String, Value>>(diff).is_ok() {
        DiffType::Map
    } else {
        let mut dmp_score = 0;
        let mut v4a_score = 0;
        for line in diff.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("@@") {
                // DMP header: @@ -a,b +c,d @@
                let has_minus = trimmed.contains('-');
                let has_plus = trimmed.contains('+');
                let has_comma = trimmed.contains(',');
                let has_trailing_at = trimmed.ends_with("@@");

                if has_minus && has_plus && has_comma && has_trailing_at {
                    dmp_score += 3;
                } else {
                    // Bare anchor or context-only anchor => V4A signal
                    v4a_score += 2;
                }
            } else if trimmed.starts_with("*** End Patch")
                || trimmed.starts_with("*** End of File")
                || trimmed.starts_with("*** Update File:")
                || trimmed.starts_with("*** Add File:")
                || trimmed.starts_with("*** Delete File:")
            {
                v4a_score += 2;
            } else if let Some(first) = trimmed.chars().next()
                && (first == '+' || first == '-' || first == ' ')
            {
                // Only count as V4A-ish if we haven't seen any strong DMP header yet.
                if dmp_score == 0 {
                    v4a_score += 1;
                }
            }
        }

        if dmp_score >= 3 && v4a_score == 0 {
            DiffType::Dmp
        } else if v4a_score >= 3 && dmp_score == 0 {
            DiffType::V4A
        } else {
            DiffType::Unknown
        }        
    }
}

pub fn apply_patch_auto(original: &str, diff: &str, create: bool) -> Result<String> {
    match classify_diff(diff) {
        DiffType::V4A => apply_v4a_patch(original, diff, create),
        DiffType::Dmp => {
            let dmp = DiffMatchPatch::new();
            let patches = dmp
                .patch_from_text::<Efficient>(diff)
                .map_err(|e| anyhow!("{e:?}"))?;

            let (new_content, results) = dmp
                .patch_apply(&patches, original)
                .map_err(|e| anyhow!("{e:?}"))?;

            if results.iter().any(|applied| !applied) {
                return Err(anyhow!("Not all DiffMatchPatch patches applied"));
            }

            Ok(new_content)
        }
        DiffType::Map => {
            if create {
                Ok(diff.to_string())
            } else {
                let diff_map = serde_json::from_str::<Map<String, Value>>(diff)?.into_iter().collect::<HashMap<_, _>>();
                let mut original_map = serde_json::from_str::<Map<String, Value>>(original)?.into_iter().collect::<HashMap<_, _>>();
                original_map.extend(diff_map);
                let patch_map = original_map.into_iter().collect::<Map<_, _>>();
                let patch = serde_json::to_string(&patch_map)?;
                Ok(patch)
            }
        }
        DiffType::Unknown => Err(anyhow!(
            "Unknown diff format. Only `Universal Diff` and `V4A Diff` formats are currently supported."
        )),
    }
}

#[cfg(test)]
pub mod tests {
    use diff_match_patch_rs::PatchInput;

    use super::*;

    /// Helper: generate a DMP patch text for a simple change.
    fn make_dmp_patch(original: &str, modified: &str) -> Result<String> {
        let dmp = DiffMatchPatch::new();
        let diffs = dmp
            .diff_main::<Efficient>(original, modified)
            .map_err(|e| anyhow!("{e:?}"))?;
        let patches = dmp
            .patch_make(PatchInput::new_diffs(&diffs))
            .map_err(|e| anyhow!("{e:?}"))?;
        let patch_txt = dmp.patch_to_text(&patches);
        Ok(patch_txt)
    }

    #[test]
    fn test_apply_patch_auto_v4a_end_patch_uses_v4a_engine_and_matches_direct() {
        let original = "";
        let diff = "+hello\r\n+world\r\n*** End Patch\r\n";
        let direct = apply_v4a_patch(original, diff, true).unwrap();
        let auto = apply_patch_auto(original, diff, true).unwrap();
        assert_eq!(auto, direct);
    }

    #[test]
    fn test_apply_patch_auto_v4a_bare_anchor_uses_v4a_engine_and_matches_direct() {
        let original = "";
        let diff = "@@\n+hello\n";
        let direct = apply_v4a_patch(original, diff, false).unwrap();
        let auto = apply_patch_auto(original, diff, false).unwrap();
        assert_eq!(auto, direct);
    }

    #[test]
    fn test_apply_patch_auto_dmp_header_uses_dmp_engine_and_matches_direct() {
        let original = "a\nb\nc\n";
        let modified = "a\nB\nc\n";
        let diff = make_dmp_patch(original, modified).unwrap();

        let auto = apply_patch_auto(original, &diff, false).unwrap();

        let dmp = DiffMatchPatch::new();
        let patches = dmp.patch_from_text::<Efficient>(&diff).unwrap();
        let (direct, results) = dmp.patch_apply(&patches, original).unwrap();
        assert!(results.iter().all(|b| *b));

        assert_eq!(auto, direct);
    }

    #[test]
    fn test_apply_patch_auto_map_brackets_extend() {
        let json = serde_json::json!({"a": "A", "b": 2});
        let original = serde_json::to_string(&json).unwrap();
        let json = serde_json::json!({"b": 1, "c": "C"});
        let diff = serde_json::to_string(&json).unwrap();
        let auto = apply_patch_auto(&original, &diff, false).unwrap();
        assert_eq!(auto, "{\"a\":\"A\",\"b\":1,\"c\":\"C\"}");
    }

    #[test]
    fn test_apply_patch_auto_map_brackets_create() {
        let original = "";
        let json = serde_json::json!({"b": 1, "c": "C"});
        let diff = serde_json::to_string(&json).unwrap();
        let auto = apply_patch_auto(original, &diff, true).unwrap();
        assert_eq!(auto, "{\"b\":1,\"c\":\"C\"}");
    }

    #[test]
    fn test_apply_patch_auto_mixed_format_to_unknown() {
        let original = "a\nb\nc\n";
        let modified = "a\nB\nc\n";
        let dmp_diff = make_dmp_patch(original, modified).unwrap();

        // Prepend a V4A-looking marker but keep a valid DMP header.
        let mixed = format!("*** End Patch\n{dmp_diff}");

        let auto = apply_patch_auto(original, &mixed, false);
        assert!(auto.is_err());
    }

    #[test]
    fn test_apply_patch_auto_ambiguous_no_strong_signals_to_unknown() {
        let original = "a\nb\n";
        let modified = "a\nB\n";
        let diff = make_dmp_patch(original, modified).unwrap();

        // Strip the header to make it ambiguous.
        let stripped = diff
            .lines()
            .filter(|l| !l.starts_with("@@"))
            .collect::<Vec<_>>()
            .join("\n");

        let auto = apply_patch_auto(original, &stripped, false);
        assert!(auto.is_err());
    }
}
