use phymes_data::{AvailableOperators, DataConfig, DataStreamManager};
use phymes_event::{Publication, Subscription};
use phymes_network::{
    DynamicTaskNetworkBuilder, DynamicTaskNetworkNames, DynamicTaskNetworkTypes, NetworkBuilder,
    NetworkBuilderMermaidTrait,
};
use phymes_processor::{AvailableProcessors, test_command_sandbox_processor};
use phymes_schemas::{
    AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait, DataEncoding, DataFormat,
};
use phymes_subject::{
    BuildableTrait, BuilderTrait, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectPlan,
    SubjectPlanBuilderTrait,
};

use crate::{GenerateTextNetworkBuilder, PatchWorkspaceNetworkBuilderStaticWSubject};

/// OpenAlex network
pub struct GenerateCodeNetworkBuilder {
    pub inner: Option<NetworkBuilder>,
}

impl Default for GenerateCodeNetworkBuilder {
    fn default() -> Self {
        // From workspace to messages
        let generate_code_network_builder = {
            let network_name = "from_workspace_to_messages";
            let config = DataConfig {
                lhs_name: Some(AvailableSubjects::Workspace.to_string()),
                code_completion: Some(phymes_data::CodeCompletionType::SRI),
                cpu: false,
                operator: AvailableOperators::FromWorkspaceToMessages,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config).unwrap();
            let subject = SubjectBuilder::new()
                .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                .with_json(&config_json, 1)
                .unwrap()
                .build()
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            // Workspace and UserMessages are instantiated in patch_workspace_network_builder and generate_text_network_builder, respectively
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                dynamic_type: DynamicTaskNetworkTypes::Static,
                processor: AvailableProcessors::CandleDataProcessor,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: AvailableSubjects::Workspace.to_string(),
                },
                publication: Publication::Replace {
                    subject_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                },
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };

        // From messages to patch
        let network_builder = {
            let network_name = "from_messages_to_patches";
            let config = DataConfig {
                lhs_name: Some(AvailableInterfaceSubjects::AssistantMessages.to_string()),
                code_completion: Some(phymes_data::CodeCompletionType::SRI),
                cpu: false,
                operator: AvailableOperators::FromMessagesToPatches,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config).unwrap();
            let subject = SubjectBuilder::new()
                .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                .with_json(&config_json, 1)
                .unwrap()
                .build()
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            // WorkspacePatch and AssistantMessages are instantiated in patch_workspace_network_builder and generate_text_network_builder, respectively
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                dynamic_type: DynamicTaskNetworkTypes::Static,
                processor: AvailableProcessors::CandleDataProcessor,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                },
                publication: Publication::Extend {
                    subject_name: AvailableSubjects::WorkspacePatch.to_string(),
                },
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };
        let generate_code_network_builder = generate_code_network_builder
            .extend(network_builder)
            .unwrap();

        // Generate code
        let generate_text_network = GenerateTextNetworkBuilder::new(
            "generate_text_network",
            // Some("QwenV2p5_0p5bCoder".to_string()), // DM: less than 7b is not accurate enough with Qwen v2.5
            Some("QwenV2p5_7bCoder".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        );
        let network_builder = NetworkBuilder::from_mermaid_flowchart(
            &generate_text_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            &generate_text_network.as_mermaid_erdiagram(),
            false,
            true,
        )
        .unwrap()
        .with_name(generate_text_network.network_name);
        let generate_code_network_builder = generate_code_network_builder
            .extend(network_builder)
            .unwrap();

        // Patch workspace
        let patch_workspace_network = PatchWorkspaceNetworkBuilderStaticWSubject::default();
        let network_builder = patch_workspace_network.inner.build_dynamic();
        let generate_code_network_builder = generate_code_network_builder
            .extend(network_builder)
            .unwrap();

        // Input table from CSV
        let subject_name_i = "subject_name_i";
        let subject_schema_io = test_command_sandbox_processor::create_messages()
            .unwrap()
            .schema();
        let network_builder = {
            let network_name = "extract_csv";
            let config = DataConfig {
                lhs_name: Some(AvailableInterfaceSubjects::UserCsv.to_string()),
                lhs_pk: Some("filename".to_string()),
                lhs_values: Some(
                    ["bytes"]
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                ),
                format: Some(DataFormat::CsvDefault),
                encoding: Some(DataEncoding::None),
                schema: Some(AvailableSubjects::Empty),
                cpu: false,
                operator: AvailableOperators::ExtractTabular,
                lhs_stream: DataStreamManager::Stream,
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config).unwrap();
            let subject = SubjectBuilder::new()
                .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                .with_json(&config_json, 1)
                .unwrap()
                .build()
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableInterfaceSubjects::UserCsv
                .to_subject(None, None)
                .unwrap();
            let subject_lhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = Subject::get_builder()
                .with_name(subject_name_i)
                .with_schema(subject_schema_io.clone())
                .with_record_batches(Vec::new())
                .unwrap()
                .build()
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                dynamic_type: DynamicTaskNetworkTypes::Static,
                processor: AvailableProcessors::ExtractTabular,
                subscription_lhs: Subscription::OnUpdateDrainRecordBatches {
                    subject_name: AvailableInterfaceSubjects::UserCsv.to_string(),
                },
                publication: Publication::Replace {
                    subject_name: subject_name_i.to_string(),
                },
                subject_lhs: Some(subject_lhs),
                subject_out: Some(subject_out),
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };
        let generate_code_network_builder = generate_code_network_builder
            .extend(network_builder)
            .unwrap();

        // Output table to CSV
        let subject_name_o = "subject_name_o";
        let network_builder = {
            let network_name = "pack_csv";
            let config = DataConfig {
                lhs_name: Some(subject_name_o.to_string()),
                format: Some(DataFormat::CsvDefault),
                encoding: Some(DataEncoding::None),
                schema: Some(AvailableSubjects::Attachments),
                doc_name: Some(subject_name_o.to_string()),
                cpu: false,
                operator: AvailableOperators::PackTabular,
                lhs_stream: DataStreamManager::Stream,
                ..Default::default()
            };
            let config_json = serde_json::to_vec(&config).unwrap();
            let subject = SubjectBuilder::new()
                .with_name(&DynamicTaskNetworkNames::Processor(network_name).to_string())
                .with_json(&config_json, 1)
                .unwrap()
                .build()
                .unwrap();
            let subject_processor = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = Subject::get_builder()
                .with_name(subject_name_o)
                .with_schema(subject_schema_io.clone())
                .with_record_batches(Vec::new())
                .unwrap()
                .build()
                .unwrap();
            let subject_lhs = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let subject = AvailableInterfaceSubjects::AssistantCsv
                .to_subject(None, None)
                .unwrap();
            let subject_out = SubjectPlan::get_builder()
                .with_subject(subject)
                .build()
                .unwrap();
            let builder = DynamicTaskNetworkBuilder {
                network_name: network_name.to_string(),
                dynamic_type: DynamicTaskNetworkTypes::Static,
                processor: AvailableProcessors::PackTabular,
                subscription_lhs: Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_name_o.to_string(),
                },
                publication: Publication::Replace {
                    subject_name: AvailableInterfaceSubjects::AssistantCsv.to_string(),
                },
                subject_lhs: Some(subject_lhs),
                subject_out: Some(subject_out),
                subject_processor,
                ..Default::default()
            };
            builder.build_dynamic()
        };
        let generate_code_network_builder = generate_code_network_builder
            .extend(network_builder)
            .unwrap();

        GenerateCodeNetworkBuilder {
            inner: Some(generate_code_network_builder.with_name("generate_code_network")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{
        DynamicTaskNetworkNames, NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream,
    };
    use phymes_schemas::{
        AttachmentBuilderTraitExt, AvailableInterfaceSubjects, AvailableSubjects,
        AvailableSubjectsTrait, CsvFormat, create_workspace_batch,
    };
    use phymes_streams::CommandSandboxEnvironments;
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, RuntimeEnvBuilderTrait, Subject,
        SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use crate::ExecuteWorkspaceNetwork;

    use super::*;

    /// `cargo test -p phymes-templates test_generate_code_network_py --features api,gpu,hf_hub --release -- --nocapture`
    /// # Notes
    /// * should require two iterations to complete the code:
    ///   1. Fill in the middle
    ///   2. Correct the error message
    /// `cargo test -p phymes-templates test_generate_code_network_py --features api,gpu,hf_hub --release -- --nocapture`
    #[tokio::test]
    async fn test_generate_code_network_py() -> Result<()> {
        // Constants
        let subject_name_i = "subject_name_i";
        let subject_name_o = "subject_name_o";
        let workspace_name = "apply_patch_s";

        // Initialize the network
        let generate_code_network_builder =
            GenerateCodeNetworkBuilder::default().inner.take().unwrap();

        // Extend with execute_workspace_network
        let execute_workspace_network = ExecuteWorkspaceNetwork::new(
            "execute_workspace_network_py",
            None,
            Some(subject_name_i),
            subject_name_o,
            &CommandSandboxEnvironments::Python,
        );
        let network_builder = NetworkBuilder::from_mermaid_flowchart(
            &execute_workspace_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            &execute_workspace_network.as_mermaid_erdiagram().unwrap(),
            false,
            true,
        )
        .unwrap()
        .with_name(execute_workspace_network.network_name);
        let generate_code_network_builder =
            generate_code_network_builder.extend(network_builder)?;

        let network_name = generate_code_network_builder.name.clone().unwrap();
        let (network, network_messages) = generate_code_network_builder
            .with_runtime_env(
                RuntimeEnv::get_builder()
                    .with_name(
                        DynamicTaskNetworkNames::RuntimeEnv(&network_name)
                            .to_string()
                            .as_str(),
                    )
                    .with_max_steps(100)
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // Make the workspace data ready for SRI
        let path = ["requirements.txt", "src/main.py", "install.sh"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let content = [
            r#"pandas==2.2.3
pyarrow==17.0.0"#,
            r#"#!/usr/bin/env python3
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

    # Convert pandas back to Arrow
    #table_out = pa.Table.from_pandas(df)
    /* MIDDLE CODE TO COMPLETE */

    # One-liner to make all fields non-nullable
    new_schema = p.schema([pa.field(f.name, f.type, nullable=False) for f in table_out.schema])

    # Cast the table to the new schema
    # This will fail if there are actual nulls in the data
    try:
        table_non_nullable = table_out.cast(new_schema)
    except pa.ArrowInvalid as e:
        raise ValueError(
            "Cannot cast to non-nullable schema because null values exist in the data."
        ) from e

    # Write Arrow IPC File format (Rust-compatible)
    with pa.OSFile(args.output_file, "wb") as f:
        writer = ipc.RecordBatchFileWriter(f, table_non_nullable.schema)
        writer.write_table(table_non_nullable)
        writer.close()"#,
            r#"#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt"#,
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let batch = create_workspace_batch(path, content)?;
        let workspace_subject = AvailableSubjects::Workspace.to_subject(None, Some(vec![batch]))?;
        let _ = message_map.insert(
            workspace_subject.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(workspace_subject.get_name())
                .with_publisher(&network_name)
                .with_subject(workspace_subject.get_name())
                .with_update(&Publication::Replace {
                    subject_name: workspace_subject.get_name().to_string(),
                })
                .with_message(workspace_subject.to_ipc_stream()?)
                .build()?,
        );

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;
        let tabular_data = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(subject_name_i)
            .build()?;
        let csv_format = CsvFormat::default();
        let bytes = tabular_data.to_csv(csv_format.delimiter, csv_format.header)?;
        let attachments = AvailableInterfaceSubjects::UserCsv
            .to_subject_builder(None)
            .with_attachment(None, Some("csv"), &bytes, None)?
            .build()?;
        let _ = message_map.insert(
            attachments.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(attachments.get_name())
                .with_publisher(&network_name)
                .with_subject(attachments.get_name())
                .with_update(&Publication::Replace {
                    subject_name: attachments.get_name().to_string(),
                })
                .with_message(attachments.to_ipc_stream()?)
                .build()?,
        );

        let _ = network_arc
            .update_subjects_from_messages(network_messages.unwrap_or_default(), 0)
            .await;

        // 1. Run the network
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));

        // DM: Requires HuggingFaceHub
        if cfg!(feature = "hf_hub") {
            let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

            assert_eq!(response.len(), 0);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(
                    AvailableInterfaceSubjects::AssistantMessages
                        .to_string()
                        .as_str(),
                )
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 2);
            let column = subject.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            assert_eq!(column.get(1).unwrap(), &"assistant");
            let column = subject.get_column_as_vec_str("content");
            assert!(column.first().unwrap().contains("src/main.py"));
            assert!(
                column
                    .first()
                    .unwrap()
                    .contains("table_out = pa.Table.from_pandas(df)")
            );
            assert!(column.get(1).unwrap().contains("src/main.py"));
            assert!(column.get(1).unwrap().contains("new_schema = pa.schema([pa.field(f.name, f.type, nullable=False) for f in table_out.schema])"));
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for t in column {
                assert!(t > 0);
            }

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: workspace_name.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(workspace_name)
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 6);
            let column = subject.get_column_as_vec_str("path");
            assert_eq!(column.get(4).unwrap(), &"src/main.py");
            let column = subject.get_column_as_vec_str("content");
            assert!(
                !column
                    .get(4)
                    .unwrap()
                    .contains("/* MIDDLE CODE TO COMPLETE */")
            );
            assert!(
                column
                    .get(4)
                    .unwrap()
                    .contains("table_out = pa.Table.from_pandas(df)")
            );
            assert!(column.get(4).unwrap().contains("new_schema = pa.schema([pa.field(f.name, f.type, nullable=False) for f in table_out.schema])"));

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: subject_name_o.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(subject_name_o)
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("name");
            assert_eq!(column, ["Alice", "Bob"]);
            let column = subject.get_column_as_vec_primitive::<i64>("age")?;
            assert_eq!(column, [40, 35]);
        }

        Ok(())
    }

    /// # Notes
    /// * should require two iterations to complete the code:
    ///   1. Fill in the middle
    ///   2. Correct the error message
    /// `cargo test -p phymes-templates test_generate_code_network_rs --features api,gpu,hf_hub --release -- --nocapture`
    #[tokio::test]
    async fn test_generate_code_network_rs() -> Result<()> {
        // Constants
        let subject_name_i = "subject_name_i";
        let subject_name_o = "subject_name_o";
        let workspace_name = "apply_patch_s";

        // Initialize the network
        let generate_code_network_builder =
            GenerateCodeNetworkBuilder::default().inner.take().unwrap();

        // Extend with execute_workspace_network
        let execute_workspace_network = ExecuteWorkspaceNetwork::new(
            "execute_workspace_network_rs",
            None,
            Some(subject_name_i),
            subject_name_o,
            &CommandSandboxEnvironments::Rust,
        );
        let network_builder = NetworkBuilder::from_mermaid_flowchart(
            &execute_workspace_network.as_mermaid_flowchart(),
            false,
        )
        .unwrap()
        .with_subjects_from_mermaid_erdiagram(
            &execute_workspace_network.as_mermaid_erdiagram().unwrap(),
            false,
            true,
        )
        .unwrap()
        .with_name(execute_workspace_network.network_name);
        let generate_code_network_builder =
            generate_code_network_builder.extend(network_builder)?;

        let network_name = generate_code_network_builder.name.clone().unwrap();
        let (network, network_messages) = generate_code_network_builder
            .with_runtime_env(
                RuntimeEnv::get_builder()
                    .with_name(
                        DynamicTaskNetworkNames::RuntimeEnv(&network_name)
                            .to_string()
                            .as_str(),
                    )
                    .with_max_steps(100)
                    .build_arc()?,
            )
            .with_diagnostics(true)
            .add_processor_subjects()?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let network_arc = Arc::new(network);

        // Make the test data
        let mut message_map = HashMap::<String, IPCMessage>::new();

        // Make the workspace data ready for SRI
        let path = ["Cargo.toml", "src/main.rs", "install.sh"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let content = [
            r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = { version = "1", default-features = false }
arrow = "58.0.0"
serde_json = "1.0.133"
serde = { version = "1.0.215", features = ["derive"] }
clap = { version = "4.5.4", features = ["derive"] }"#,
            r#"use arrow::array::{ArrayRef, Int64Array, StringArray};
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
        .collect::<Vec<u32>>(); // Should be i64
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
    // let batches = batches.into_iter()
    //      .map(|batch| add_10_yrs_to_age(&batch).unwrap())
    //      .collect::<Vec<_>>();
    /* MIDDLE CODE TO COMPLETE */

    // --- Step 3: Write modified batches to output file ---
    let file = File::create(output_path)?;
    let mut writer = FileWriter::try_new(file, &batches[0].schema())?;

    for batch in batches {
        writer.write(&batch)?;
    }

    writer.finish()?;

    Ok(())
}"#,
            r#"#!/usr/bin/env bash
apt update
apt install --assume-yes protobuf-compiler clang"#,
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let batch = create_workspace_batch(path, content)?;
        let workspace_subject = AvailableSubjects::Workspace.to_subject(None, Some(vec![batch]))?;
        let _ = message_map.insert(
            workspace_subject.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(workspace_subject.get_name())
                .with_publisher(&network_name)
                .with_subject(workspace_subject.get_name())
                .with_update(&Publication::Replace {
                    subject_name: workspace_subject.get_name().to_string(),
                })
                .with_message(workspace_subject.to_ipc_stream()?)
                .build()?,
        );

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;
        let tabular_data = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(subject_name_i)
            .build()?;
        let csv_format = CsvFormat::default();
        let bytes = tabular_data.to_csv(csv_format.delimiter, csv_format.header)?;
        let attachments = AvailableInterfaceSubjects::UserCsv
            .to_subject_builder(None)
            .with_attachment(None, Some("csv"), &bytes, None)?
            .build()?;
        let _ = message_map.insert(
            attachments.get_name().to_string(),
            IPCMessage::get_builder()
                .with_name(attachments.get_name())
                .with_publisher(&network_name)
                .with_subject(attachments.get_name())
                .with_update(&Publication::Replace {
                    subject_name: attachments.get_name().to_string(),
                })
                .with_message(attachments.to_ipc_stream()?)
                .build()?,
        );

        let _ = network_arc
            .update_subjects_from_messages(network_messages.unwrap_or_default(), 0)
            .await;

        // 1. Run the network
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));

        if cfg!(feature = "hf_hub") {
            let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

            assert_eq!(response.len(), 0);

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(
                    AvailableInterfaceSubjects::AssistantMessages
                        .to_string()
                        .as_str(),
                )
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 2);
            let column = subject.get_column_as_vec_str("role");
            assert_eq!(column.first().unwrap(), &"assistant");
            assert_eq!(column.get(1).unwrap(), &"assistant");
            let column = subject.get_column_as_vec_str("content");
            assert!(column.first().unwrap().contains("src/main.rs"));
            assert!(column.first().unwrap().contains(
                r#"    let batches = batches.into_iter()
            .map(|batch| add_10_yrs_to_age(&batch).unwrap())
            .collect::<Vec<_>>();"#
            ));
            assert!(column.get(1).unwrap().contains("src/main.rs"));
            assert!(column.get(1).unwrap().contains(
                r#"    let ages = batch
            .column_by_name("age")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default() + 10)
            .collect::<Vec<i64>>();"#
            ));
            let column = subject.get_column_as_vec_primitive::<i64>("timestamp")?;
            for t in column {
                assert!(t > 0);
            }

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: workspace_name.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(workspace_name)
                .with_record_batches(batches)?
                .build()?;
            assert_eq!(subject.count_rows(), 6);
            let column = subject.get_column_as_vec_str("path");
            assert_eq!(column.get(5).unwrap(), &"src/main.rs");
            let column = subject.get_column_as_vec_str("content");
            assert!(
                !column
                    .get(5)
                    .unwrap()
                    .contains("/* MIDDLE CODE TO COMPLETE */")
            );
            assert!(column.get(5).unwrap().contains(
                r#"    let batches = batches.into_iter()
            .map(|batch| add_10_yrs_to_age(&batch).unwrap())
            .collect::<Vec<_>>();"#
            ));
            assert!(column.get(5).unwrap().contains(
                r#"    let ages = batch
            .column_by_name("age")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .map(|s| s.unwrap_or_default() + 10)
            .collect::<Vec<i64>>();"#
            ));

            let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
                subject_name: subject_name_o.to_string(),
            }
            .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
            .unwrap()
            .try_collect()
            .await?;
            let subject = Subject::get_builder()
                .with_name(subject_name_o)
                .with_record_batches(batches)?
                .build()?;
            let column = subject.get_column_as_vec_str("name");
            assert_eq!(column, ["Alice", "Bob"]);
            let column = subject.get_column_as_vec_primitive::<i64>("age")?;
            assert_eq!(column, [40, 35]);
        }

        Ok(())
    }
}
