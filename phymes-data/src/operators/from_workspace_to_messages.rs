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
pub struct FromWorkspaceToMessages {
    code_completion: CodeCompletionType,
}

impl MappableTrait for FromWorkspaceToMessages {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for FromWorkspaceToMessages {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        from_workspace_to_messages(lhs_args, &self.code_completion, device)
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let code_completion = config.code_completion.clone().ok_or(anyhow!(
            "Missing `code_completion` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(FromWorkspaceToMessages { code_completion })
    }
}

/// Custom function to convert `Workspace` to a prompt for fill-in-the-middle (FIM) code completion
///
/// # Notes
///
/// * LHS is Workspace
/// * Output schema is Message
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es
/// * ``
/// * `device` - The compute device
pub fn from_workspace_to_messages(
    lhs_args: &[RecordBatch],
    code_completion: &CodeCompletionType,
    _device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs and rhs into tables
    let lhs_table = Subject::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("from_workspace_to_messages")
        .build()?;

    // Get the workspace information
    let repository_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("repository")?;
    let path_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("path")?;
    // Assume that <|fim_prefix|> and <|fim_suffix|> have already been added to the content?
    let content_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("content")?;

    // Create the prompt
    let prompt_vec: Result<Vec<String>> = repository_vec.into_iter()
        .zip(path_vec.into_iter())
        .zip(content_vec.into_iter())
        .enumerate()
        .map(|(i, ((r, p), c))| {

            // Extract the filename
            let path = std::path::Path::new(&p);
            match path.file_name() {
                Some(filename) => {
                    // Convert OsStr to &str safely
                    match filename.to_str() {
                        Some(name_str) => {
                            // Create the string
                            if i == 0 {
                                let prompt = format!("<|repo_name|>{r}\n<|file_sep|>{name_str}\n{c}\n");
                                Ok(prompt)
                            } else {
                                let prompt = format!("<|file_sep|>{name_str}\n{c}\n");
                                Ok(prompt)
                            }
                        },
                        None => Err(anyhow!("Filename contains invalid UTF-8.")),
                    }
                }
                None => Err(anyhow!("No filename found in the given path.")),
            }
        })
        .collect();
    let mut prompt = prompt_vec?.join("");
    match code_completion {
        CodeCompletionType::FIM => prompt.push_str("<|fim_middle|>"),
        CodeCompletionType::SRI => {},
    }

    // Create the message
    create_chat_record_batch(vec!["user".to_string()], vec![prompt], vec![create_timestamp_micros()])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::device;
    use arrow::array::{ArrayRef, StringArray};

    use super::*;

    #[test]
    fn test_from_workspace_to_messages_fim() -> Result<()> {
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

        let result = from_workspace_to_messages(&[repo_batch], &CodeCompletionType::FIM, &device)?;
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

    #[test]
    fn test_from_workspace_to_messages_sri() -> Result<()> {
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

        let result = from_workspace_to_messages(&[repo_batch], &CodeCompletionType::SRI, &device)?;
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
pub struct Todo {}"#]);

        Ok(())
    }
}
