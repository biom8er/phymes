use anyhow::{anyhow, Result};
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_diagnostics::create_timestamp_micros;
use phymes_schemas::create_chat_record_batch;
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use serde::{Deserialize, Serialize};

use crate::{CodeCompletionType, DataConfig, DataOperatorTrait};

/// Compute the normalized start and end times in a [RecordBatch]
#[derive(Debug, Serialize, Deserialize)]
pub struct FromMessagesToPatches {
    code_completion: CodeCompletionType,
}

impl MappableTrait for FromMessagesToPatches {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for FromMessagesToPatches {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        from_messages_to_patches(lhs_args, &self.code_completion, device)
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let code_completion = config.code_completion.clone().ok_or(anyhow!(
            "Missing `code_completion` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(FromMessagesToPatches { code_completion })
    }
}

/// Custom function to convert a fill-in-the-middle (FIM) code completion response to a patch
///
/// # Notes
///
/// * LHS schema is Workspace
/// * RHS schema is Message
/// * Output schema is Patch
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es with the assistant FIM code completion
/// * `device` - The compute device
pub fn from_messages_to_patches(
    lhs_args: &[RecordBatch],
    code_completion: &CodeCompletionType,
    _device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs and rhs into tables
    let lhs_table = Subject::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("from_messages_to_patches Code Completion")
        .build()?;

    // Get the workspace information
    let repository_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("repository")?;
    let path_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("path")?;
    let content_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("content")?;

    // Create the promopt
    let prompt_vec: Result<Vec<(String, String, String)>> = repository_vec.into_iter()
        .zip(path_vec.into_iter())
        .zip(content_vec.into_iter())
        .filter(|((r, p), c)| match code_completion { 
            CodeCompletionType::FIM => c.contains("<|fim_prefix|>") && c.contains("<|fim_suffix|>"),
            CodeCompletionType::SRI => c.contains("/* MIDDLE CODE TO COMPLETE */"),
        })
        .map(|((r, p), c)| {

            // Extract the filename
            let path = std::path::Path::new(&p);
            match path.file_name() {
                Some(filename) => {
                    // Convert OsStr to &str safely
                    match filename.to_str() {
                        Some(name_str) => Ok((r, name_str.to_string(), c)),
                        None => Err(anyhow!("Filename contains invalid UTF-8.")),
                    }
                }
                None => Err(anyhow!("No filename found in the given path.")),
            }
        })
        .collect();

    todo!()
    // Create the patch batch
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::device;
    use arrow::array::{ArrayRef, StringArray};

    use super::*;

    #[test]
    fn test_from_messages_to_patches() -> Result<()> {
        // Create the mock repository
        let repo_names = [
            "test_repo",
            "test_repo",
            "test_repo",
            "test_repo",
            "test_repo",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let repo_paths = [
            "/home/sandbox/Cargo.toml",
            "/home/sandbox/src/main.rs",
            "/home/sandbox/src/lib.rs",
            "/home/sandbox/src/extras/mod.rs",
            "/home/sandbox/src/extras/todo.rs",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let code = [
            r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }"#,
            r#"use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}"#,
            "pub mod extra;",
            r#"mod todo;
pub use todo::Todo"#,
            "pub struct Todo {}",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let repo_names: ArrayRef = Arc::new(StringArray::from(repo_names));
        let repo_paths: ArrayRef = Arc::new(StringArray::from(repo_paths));
        let code: ArrayRef = Arc::new(StringArray::from(code));
        let repo_batch = RecordBatch::try_from_iter(vec![
            ("repository", repo_names),
            ("path", repo_paths),
            ("content", code),
        ])?;

        // Make the device
        let device = device(false)?;

        let result = from_messages_to_patches(&[repo_batch], &CodeCompletionType::FIM, &device)?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let cols = result_table.get_column_as_vec_str("role");
        assert_eq!(cols, ["user"]);
        let cols = result_table.get_column_as_vec_str("content");
        assert_eq!(cols, [r#"<|repo_name|>test_repo
<|file_sep|>Cargo.toml
[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }
<|file_sep|>main.rs
use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}
<|file_sep|>lib.rs
pub mod extra;
<|file_sep|>mod.rs
mod todo;
pub use todo::Todo
<|file_sep|>todo.rs
pub struct Todo {}
<|fim_middle|>"#]);

        Ok(())
    }
}
