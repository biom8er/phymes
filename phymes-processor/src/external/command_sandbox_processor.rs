use std::sync::Arc;

use anyhow::{Result, anyhow};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::CommandSandboxStream;
use phymes_subject::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};

use crate::ProcessorTrait;

/// Runs commands in a sandboxed environment and returns the result or error message
///
/// # Notes
/// * Current sandboxed environments include docker and wasmtime
/// * Stderr messages are routed to `SessionError`s
#[derive(Debug)]
pub struct CommandSandboxProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for CommandSandboxProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for CommandSandboxProcessor {
    fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
        }
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn line_and_file(&self) -> (u32, String) {
        (line!(), file!().to_string())
    }

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Run the stream
        let out = Box::pin(CommandSandboxStream::new(
            message,
            config,
            Arc::clone(&runtime_env),
            diagnostic_builder.cloned(),
        )?);

        // Prepare the message builder
        let mut builder_map = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let builder = SendableRecordBatchStreamMessage::get_builder()
            .with_name(self.get_name())
            .with_message(out);
        let _ = builder_map.insert(self.get_name().to_string(), builder);

        Ok(builder_map)
    }
}

pub mod test_command_sandbox_processor {
    use arrow::array::{ArrayRef, RecordBatch, StringArray, Int64Array};

    use super::*;

    pub fn create_messages() -> Result<RecordBatch> {
        let names = vec!["Alice", "Bob"];
        let names_arr: ArrayRef = Arc::new(StringArray::from(names));
        let ages = vec![30, 25];
        let ages_arr: ArrayRef = Arc::new(Int64Array::from(ages));
        let batch = RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, RecordBatch, StringArray, Int64Array};
    use futures::TryStreamExt;
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};
    use phymes_event::Publication;
    use phymes_schemas::create_chat_record_batch;
    use phymes_streams::{
        CommandSandboxConfig, CommandSandboxEnvironments, CommandSandboxRunners, DataIOMethod,
    };
    use phymes_subject::{SubjectBuilder, SubjectBuilderTrait, SubjectTrait};
    use std::{fs::File, io::Write};
    use tempfile::{NamedTempFile, TempDir};

    use super::*;

    #[tokio::test]
    async fn test_command_sandbox_processor_wasmtime_no_workspace_no_messages() -> Result<()> {
        let name = "CommandSandboxProcessor";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // --- From config with wasm module env ---

        // Create the wasm module
        let wasm_module_str = r#"(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )
)"#;
        let mut wasm_module_file = NamedTempFile::new()?;
        writeln!(wasm_module_file, "{wasm_module_str}")?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            timeout: 5,
            runner: CommandSandboxRunners::Wasmtime,
            environment: CommandSandboxEnvironments::WasmModule,
            container_image: wasm_module_file.path().to_str().unwrap().to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("add".to_string()), // DM: mimic module style CLI without WAVE
            container_args: Some(vec!["run".to_string(), "--invoke".to_string()]), // DM: mimic module style CLI without WAVE
            cli_args: Some(vec!["1".to_string(), "2".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["3\n"]);

        // --- From config with wasm component env ---

        // Create the wasm module
        let wasm_module_str = r#"(component
  ;; Define a core module
  (core module $math
    (func $add (param $a i32) (param $b i32) (result i32)
      local.get $a
      local.get $b
      i32.add
    )
    (export "add" (func $add))
  )

  ;; Instantiate the core module
  (core instance $math-inst
    (instantiate $math)
  )

  ;; Lift the core function into the component world
  (func (export "add") (param "a" u32) (param "b" u32) (result u32)
    (canon lift (core func $math-inst "add"))
  )
)"#;
        let mut wasm_module_file = NamedTempFile::new()?;
        writeln!(wasm_module_file, "{wasm_module_str}")?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Wasmtime,
            environment: CommandSandboxEnvironments::WasmComponent,
            container_image: wasm_module_file.path().to_str().unwrap().to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("add".to_string()),
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["1".to_string(), "2".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["3\n"]);

        // --- From stdio ---
        // DM: requires named arguments which is only possible with .wasm
        // DM, todo: create a small wasm example that takes a RecordBatch, modifies it, and returns the modified RecordBatch

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_wasmtime_workspace_no_messages() -> Result<()> {
        let name = "CommandSandboxProcessor";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Create project directory
        let project_name = "phymes-wasm-workspace";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = std::fs::create_dir(&project_dir);

        // --- From config with wasm module env ---

        // Create the workspace
        let workspace_table = CommandSandboxEnvironments::WasmModule.to_default_workspace(None)?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            timeout: 5,
            runner: CommandSandboxRunners::Wasmtime,
            environment: CommandSandboxEnvironments::WasmModule,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_image: project_dir
                .join("src/main.wat")
                .as_path()
                .to_str()
                .unwrap()
                .to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: None,
            container_args: Some(vec![
                "run".to_string(),
                "--invoke".to_string(),
                "add".to_string(),
            ]), // DM: mimic module style CLI without WAVE
            cli_args: Some(vec!["1".to_string(), "2".to_string()]),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["3\n"]);

        // --- From config with wasm component env ---
        let workspace_table =
            CommandSandboxEnvironments::WasmComponent.to_default_workspace(None)?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Wasmtime,
            environment: CommandSandboxEnvironments::WasmComponent,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_image: project_dir
                .join("src/main.wat")
                .as_path()
                .to_str()
                .unwrap()
                .to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("add".to_string()),
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["1".to_string(), "2".to_string()]),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["3\n"]);

        // --- From stdio ---
        // DM: requires named arguments which is only possible with .wasm
        // DM, todo: create a small wasm example that takes a RecordBatch, modifies it, and returns the modified RecordBatch

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_bash_no_workspace() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // --- From config ---
        // based on docker run --rm alpine echo "Hello from Docker!"

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Bash,
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("echo".to_string()),
            timeout: 5,
            cli_args: Some(vec!["Hello from Docker!".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains("Hello from Docker"));

        // --- From Stdio ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Bash,
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::None,
            command: Some("echo".to_string()),
            timeout: 5,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_batch = create_chat_record_batch(
            vec!["user".to_string()],
            vec!["Hello from Docker!".to_string()],
            vec![0],
        )?;
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![message_batch])?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains(
            "--input [{\"content\":\"Hello from Docker!\",\"role\":\"user\",\"timestamp\":"
        ));

        // --- From TempFile ---

        // Create project directory
        let project_name = "phymes-bash-project";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = std::fs::create_dir(&project_dir);

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Bash,
            container_image: "alpine".to_string(),
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::None,
            command: Some("echo".to_string()),
            timeout: 5,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(
            result
                .first()
                .unwrap()
                .contains("--input-file /home/sandbox/input.ipc")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_bash_workspace() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Create project directory
        let project_name = "phymes-bash-workspace";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = std::fs::create_dir(&project_dir);

        // --- From config ---
        // based on docker run --rm alpine echo "Hello from Docker!"

        // Create the workspace
        let workspace_table = CommandSandboxEnvironments::Bash.to_default_workspace(None)?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Bash,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.sh".to_string()),
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            cli_args: Some(vec!["Hello from Docker!".to_string()]),
            timeout: 5,
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_bash_workspace From config")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains("Hello from Docker"));

        // --- From Stdio ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Bash,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.sh".to_string()),
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::None,
            timeout: 5,
            subject_name: Some(messages.to_string()),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_batch = create_chat_record_batch(
            vec!["user".to_string()],
            vec!["Hello from Docker!".to_string()],
            vec![0],
        )?;
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .with_record_batches(vec![message_batch])?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_bash_workspace From Stdio")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains(
            "--input\n[{\"content\":\"Hello from Docker!\",\"role\":\"user\",\"timestamp\":"
        ));

        // --- From TempFile ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Bash,
            container_image: "alpine".to_string(),
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.sh".to_string()),
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::None,
            timeout: 5,
            subject_name: Some(messages.to_string()),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_bash_workspace From Tempfile")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(
            result
                .first()
                .unwrap()
                .contains("--input-file\n/home/sandbox/input.ipc")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py_no_workspace_no_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // --- From config ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("-c".to_string()),
            timeout: 5,
            cli_args: Some(vec![
                "import json, sys; data=json.loads(sys.argv[1]); print(json.dumps([{\"name\": item[\"name\"]} for item in data]))".to_string(),
                "[{\"name\": \"Alice\", \"age\": 30}, {\"name\": \"Bob\", \"age\": 25}]".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_run from Config")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(
            result
                .first()
                .unwrap()
                .contains("[{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]")
        );

        // --- From Stdio ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::Stdio,
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            command: Some("-c".to_string()),
            timeout: 5,
            cli_args: Some(vec!["import json, argparse; \
                parser = argparse.ArgumentParser(); \
                parser.add_argument('--input', required=True); \
                args = parser.parse_args(); \
                data = json.loads(args.input); \
                print(json.dumps([{\"name\": item[\"name\"], \"age\": item[\"age\"] + 10} for item in data]))".to_string()]),
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_run from STDIO")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py_no_workspace_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Create project directory
        let project_name = "phymes-py-project";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = std::fs::create_dir(&project_dir);

        // Create src directory
        let src_path = format!("{}/src", project_dir.as_path().to_str().unwrap());
        let _ = std::fs::create_dir(&src_path);

        // Create the requirements.txt
        let requirements_file_path = format!(
            "{}/requirements.txt",
            project_dir.as_path().to_str().unwrap()
        );
        let mut requirements_file =
            File::create(&requirements_file_path).expect("Failed to create requirements.txt");
        let requirements_str = r#"pandas==2.2.3
pyarrow==17.0.0"#;
        let _ = requirements_file.write(requirements_str.as_bytes())?;
        requirements_file.flush()?;

        // Create the initialization script
        let initialization_str = r#"#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt"#;

        // Create the run script
        let run_str = r#"#!/usr/bin/env python3
import argparse
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    # Read using PyArrow IPC file reader (works with Rust arrow FileWriter output)
    with open(args.input_file, "rb") as f:
        table = ipc.open_file(f).read_all()

    # Convert to pandas for your transformation
    df = table.to_pandas()

    # Your transformation
    df['age'] = df['age'] + 10
    
    # Write back out as Feather v2 (fully compatible with Pandas)
    #df.to_feather(args.output_file, version=2)

    # Convert pandas back to Arrow
    table_out = pa.Table.from_pandas(df)

    # Write Arrow IPC File format (Rust-compatible)
    with pa.OSFile(args.output_file, "wb") as f:
        writer = ipc.RecordBatchFileWriter(f, table_out.schema)
        writer.write_table(table_out)
        writer.close()"#;

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: Some(initialization_str.to_string()),
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: None,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_install")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py_workspace_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Create project directory
        let project_name = "phymes-py-workspace";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = std::fs::create_dir(&project_dir);

        // Create the workspace
        let workspace_table = CommandSandboxEnvironments::Python.to_default_workspace(None)?;

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.py".to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: None,
            subject_name: Some(messages.to_string()),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_install")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_rs_no_workspace_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Create project directory
        let project_name = "phymes-rs-project";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = std::fs::create_dir(&project_dir);

        // Create src directory
        let src_path = format!("{}/src", project_dir.as_path().to_str().unwrap());
        let _ = std::fs::create_dir(&src_path);

        // Create the cargo.toml
        let requirements_file_path =
            format!("{}/Cargo.toml", project_dir.as_path().to_str().unwrap());
        let mut requirements_file =
            File::create(&requirements_file_path).expect("Failed to create Cargo.toml");
        let requirements_str = r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = { version = "1", default-features = false }
arrow = "58.0.0"
serde_json = "1.0.133"
serde = { version = "1.0.215", features = ["derive"] }
clap = { version = "4.5.4", features = ["derive"] }"#;
        let _ = requirements_file.write(requirements_str.as_bytes())?;
        requirements_file.flush()?;

        // Create the run script
        let run_str = r#"use std::io::Write;
use anyhow::Result;
use clap::Parser;

/// Minimal Feather/IPC editor using a single CLI argument.
#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
struct Args {
    /// Input string
    #[arg(long)]
    input: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Data {
    /// Name
    name: String,
    /// Age
    age: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = serde_json::from_str::<Vec<Data>>(&args.input)?;
    let modified = input.into_iter()
        .map(|d| Data { name: d.name, age: d.age + 10 })
        .collect::<Vec<_>>();
    let serialized = serde_json::to_string(&modified)?;
    let _bytes = std::io::stdout().write(serialized.as_bytes())?;
    std::io::stdout().flush().unwrap();

    Ok(())
}"#;

        // --- from Config, no initialization, and Error ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            project_dir: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: None,
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec![
                "--release".to_string(),
                "--".to_string(),
                "--input".to_string(),
                "[{\"name\": \"Alice\", \"age\": 30}, {\"name\": \"Bob\", \"age\": 25}]"
                    .to_string(),
            ]),
            subject_name: None,
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await;
        if let Err(err) = &result {
            assert!(
                err.to_string()
                    .contains("error: cannot find derive macro `Serialize` in this scope")
            );
            assert!(
                err.to_string()
                    .contains("error: cannot find derive macro `Deserialize` in this scope")
            );
            assert!(err.to_string().contains(
                "error[E0277]: the trait bound `Data: serde::Deserialize<'de>` is not satisfied"
            ));
        } else {
            panic!("Should have generated an Error.")
        }

        // Create the run script
        let run_str = r#"use std::io::Write;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

/// Minimal Feather/IPC editor using a single CLI argument.
#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
struct Args {
    /// Input string
    #[arg(long)]
    input: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Data {
    /// Name
    name: String,
    /// Age
    age: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = serde_json::from_str::<Vec<Data>>(&args.input)?;
    let modified = input.into_iter()
        .map(|d| Data { name: d.name, age: d.age + 10 })
        .collect::<Vec<_>>();
    let serialized = serde_json::to_string(&modified)?;
    let _bytes = std::io::stdout().write(serialized.as_bytes())?;
    std::io::stdout().flush().unwrap();

    Ok(())
}"#;

        // --- from Config, no initialization ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            project_dir: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: None,
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec![
                "--release".to_string(),
                "--".to_string(),
                "--input".to_string(),
                "[{\"name\": \"Alice\", \"age\": 30}, {\"name\": \"Bob\", \"age\": 25}]"
                    .to_string(),
            ]),
            subject_name: None,
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(
            result,
            ["[{\"name\":\"Alice\",\"age\":40},{\"name\":\"Bob\",\"age\":35}]"]
        );

        // --- from StdIO, no initialization ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::Stdio,
            project_dir: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: None,
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch.clone()])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        // --- from TempFile, no initialization ---

        // Create the run script
        let run_str = r#"use arrow::array::{ArrayRef, StringArray, Int64Array};
use arrow::error::{ArrowError, Result};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use clap::Parser;
use std::fs::File;
use std::sync::Arc;

/// Minimal Feather/IPC editor using a single CLI argument.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the input Feather/IPC file
    #[arg(long)]
    input_file: String,
    /// Path to the output Feather/IPC file
    #[arg(long)]
    output_file: String,
}

/// Helper function to add 10 to years to the age column in a RecordBatch
fn add_10_yrs_to_age(batch: &RecordBatch) -> arrow::error::Result<RecordBatch> {
    let names = batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap_or_default().to_string())
        .collect::<Vec<String>>();
    let ages = batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap_or_default() + 10)
        .collect::<Vec<u32>>();
    let names_arr: ArrayRef = Arc::new(StringArray::from(names));
    let ages_arr: ArrayRef = Arc::new(Int64Array::from(ages));
    RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_path = args.input_file.as_str();
    let output_path = args.output_file.as_str();

    // --- Step 1: Read Feather/IPC file ---
    let file = File::open(input_path)?;
    let reader = FileReader::try_new(file, None)?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<_>>()?;

    // --- Step 2: Modify the batches ---
    let batches = batches.into_iter()
        .map(|batch| add_10_yrs_to_age(&batch).unwrap())
        .collect::<Vec<_>>();

    // --- Step 3: Write modified batches to output file ---
    let file = File::create(output_path)?;
    let mut writer = FileWriter::try_new(file, &batches[0].schema())?;

    for batch in batches {
        writer.write(&batch)?;
    }

    writer.finish()?;

    Ok(())
}"#;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: None,
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch.clone()])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        // --- from TempFile (multiple batches), no initialization ---

        // State for the command processor config
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch_1 = test_command_sandbox_processor::create_messages()?;
        let names = vec!["Joe"];
        let names_arr: ArrayRef = Arc::new(StringArray::from(names));
        let ages = vec![40];
        let ages_arr: ArrayRef = Arc::new(Int64Array::from(ages));
        let batch_2 = RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch_1, batch_2])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob", "Joe"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35, 50]);

        // --- from TempFile, initialization ---

        // Create the initialization script
        let initialization_str = r#"#!/usr/bin/env bash
apt update
apt install --assume-yes protobuf-compiler clang"#;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: Some(initialization_str.to_string()),
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_rs_workspace_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Create project directory
        let project_name = "phymes-rs-workspace";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = std::fs::create_dir(&project_dir);

        // --- from TempFile, initialization ---

        // Create the workspace
        let workspace_table = CommandSandboxEnvironments::Rust.to_default_workspace(None)?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.rs".to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            subject_name: Some(messages.to_string()),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }
}
