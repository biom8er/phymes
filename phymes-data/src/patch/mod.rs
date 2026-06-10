/// Patching module inspired by OpenAI Codex patching tools
mod apply_patch;
mod apply_search_and_replace_diff;
mod apply_v4a_diff;
mod tool_parser;
#[cfg(feature = "api")]
mod workspace_editor;

pub use apply_patch::{DiffType, PatchOperation, PatchOperator, apply_patch_auto, compute_diff};
pub use apply_search_and_replace_diff::{
    apply_search_and_replace_patch, parse_fill_in_the_middle_output,
    parse_search_and_replace_output,
};
pub use apply_v4a_diff::{ApplyDiffMode, apply_v4a_diff};
pub use tool_parser::{extract_fim_str, extract_tool_calls_str, format_tool_calls_str};

#[cfg(feature = "api")]
pub use workspace_editor::WorkspaceEditor;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// The code completion type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, ValueEnum)]
pub enum CodeCompletionType {
    /// Fill-in-the-middle
    #[value(name = "FIM")]
    FIM,
    /// Search and Replace Infilling
    #[default]
    #[value(name = "SRI")]
    SRI,
}
