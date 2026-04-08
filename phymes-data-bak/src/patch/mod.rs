/// Patching module inspired by OpenAI Codex patching tools
pub mod apply_patch;
pub mod apply_v4a_diff;
#[cfg(feature = "api")]
pub mod workspace_editor;

pub use apply_patch::{DiffType, PatchOperation, PatchOperator, apply_patch_auto, compute_diff};
pub use apply_v4a_diff::{ApplyDiffMode, apply_v4a_diff};

#[cfg(feature = "api")]
pub use workspace_editor::WorkspaceEditor;
