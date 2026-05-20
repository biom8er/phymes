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

/// Fill-in-the-middle System Prompt Based on https://arxiv.org/pdf/2601.13384 with modifications
const FIM_SYSTEM_TEMPLATE: &str = r#"You are a code completion assistant. Use the minimal amount of tokens to fill in the middle code.
            
Requirements:

1. Encapsulate the MIDDLE code in MarkDown code block that includes the LANGUAGE of the code. 
2. Provide the FILENAME for the middle code above the code block.
3. Do not repeat or modify any existing code from the context, prefix, or suffix
4. Do not add any other text or comments

Output template:

FILENAME
```LANGUAGE
MIDDLE
```

Example output:

hello.py
```python
    print("Hello world!")
```

"#;
/// Search and Replace Infilling Based on https://arxiv.org/pdf/2601.13384 with modifications
const SRI_SYSTEM_TEMPLATE: &str =  r#"You are a code edit assistant. Your task is to implement ONLY the middle code that needs to be completed while keeping all other code exactly as is. When you see a code file containing special comment markers /* MIDDLE CODE TO COMPLETE*/, you should:

1. Generate a search/replace format output that:

- Identifies the PATH containing the /* MIDDLE CODE TO COMPLETE */ marker
- Identifies the exact region containing the /* MIDDLE CODE TO COMPLETE */ marker
- Provides the code that should replace the marker

2.a. Output format:

```
PATH
<<<<<<< SEARCH
Code section containing /* MIDDLE CODE TO COMPLETE */
=======
Same code section with ONLY the middle code implemented
>>>>>>> REPLACE
```

2.b. Example output (no bugs within a 10-line window):

```
/src/hello.py
<<<<<<< SEARCH
    /* MIDDLE CODE TO COMPLETE */
=======
    print("Hello world!")
>>>>>>> REPLACE
```

2.c. Example output (bugs within a 10-line window):

```
/src/hello.py
<<<<<<< SEARCH
    three = 4
    assert(2 + 1 == three)
    /* MIDDLE CODE TO COMPLETE */
=======
    three = 3
    assert(2 + 1 == three)
    print("Hello world!")
>>>>>>> REPLACE
```

3. Requirements:

- Only edit the code within a 10-line window around the identifier.
- The search section MUST contain the /* MIDDLE CODE TO COMPLETE */ marker
"#;

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
    let path_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("path")?;
    let content_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("content")?;
    let repository_vec = lhs_table.get_column_as_vec_nonprimitive::<String>("repository")
        .unwrap_or_else(|_| (0..path_vec.len()).map(|_| "repo".to_string()).collect::<Vec<_>>());

    // Create the prompt
    let prompt_vec: Result<Vec<String>> = repository_vec.into_iter()
        .zip(path_vec.into_iter())
        .zip(content_vec.into_iter())
        .enumerate()
        .map(|(i, ((r, p), c))| {
            // Create the string
            if i == 0 {
                let prompt = format!("<|repo_name|>{r}\n<|file_sep|>{p}\n{c}\n");
                Ok(prompt)
            } else {
                let prompt = format!("<|file_sep|>{p}\n{c}\n");
                Ok(prompt)
            }
        })
        .collect();
    let mut prompt = prompt_vec?.join("");
    match code_completion {
        CodeCompletionType::FIM => prompt.push_str("<|fim_middle|>"),
        CodeCompletionType::SRI => {},
    }

    // Create the system prompt
    let system = match code_completion {
        CodeCompletionType::FIM => FIM_SYSTEM_TEMPLATE,
        CodeCompletionType::SRI => SRI_SYSTEM_TEMPLATE,
    };

    // Create the message
    let role = vec!["system".to_string(), "user".to_string()];
    let content = vec![system.to_string(), prompt];
    let timestamp = vec![create_timestamp_micros(), create_timestamp_micros()];
    create_chat_record_batch(role, content, timestamp)
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
        assert_eq!(cols, ["system", "user"]);
        let cols = result_table.get_column_as_vec_str("content");
        assert_eq!(cols, [FIM_SYSTEM_TEMPLATE, r#"<|repo_name|>test_repo
<|file_sep|>/home/sandbox/Cargo.toml
[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }
<|file_sep|>/home/sandbox/src/main.rs
use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}
<|file_sep|>/home/sandbox/src/lib.rs
pub mod extra;
<|file_sep|>/home/sandbox/src/extras/mod.rs
mod todo;
pub use todo::Todo
<|file_sep|>/home/sandbox/src/extras/todo.rs
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
        assert_eq!(cols, ["system", "user"]);
        let cols = result_table.get_column_as_vec_str("content");
        assert_eq!(cols, [SRI_SYSTEM_TEMPLATE ,r#"<|repo_name|>test_repo
<|file_sep|>/home/sandbox/Cargo.toml
[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = { version = "1", default-features = false }
<|file_sep|>/home/sandbox/src/main.rs
use anyhow::Result;
fn main() -> Result<()> {
    Ok(())
}
<|file_sep|>/home/sandbox/src/lib.rs
pub mod extra;
<|file_sep|>/home/sandbox/src/extras/mod.rs
mod todo;
pub use todo::Todo
<|file_sep|>/home/sandbox/src/extras/todo.rs
pub struct Todo {}
"#]);

        Ok(())
    }
}
