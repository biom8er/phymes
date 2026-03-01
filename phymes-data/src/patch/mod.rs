/// Patching module inspired by OpenAI Codex patching tools
/// 
/// # Example
/// 
/// === src/main.rs
/// ```rust
/// use std::path::PathBuf;
/// 
/// use workspace_editor_v4a::{PatchKind, PatchOperation, WorkspaceEditor};
/// 
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let editor = WorkspaceEditor::new("./workspace");
/// 
///     // Example V4A-style create diff (bare @@)
///     let diff_text = "@@\n+Hello, world!\n";
/// 
///     let op = PatchOperation {
///         path: PathBuf::from("hello.txt"),
///         diff: diff_text.to_string(),
///         kind: PatchKind::Create,
///     };
/// 
///     editor.apply_operation(&op)?;
/// 
///     Ok(())
/// }
/// ```

pub mod apply_diff;
pub mod v4a_patch;
pub mod patch_engine;
#[cfg(feature = "api")]
pub mod workspace_editor;

pub use apply_diff::{ApplyDiffMode, apply_diff};
pub use patch_engine::{PatchKind, PatchOperation, apply_patch_auto};

#[cfg(feature = "api")]
pub use workspace_editor::WorkspaceEditor;