use phymes_data::{AvailableOperators, DataConfig, DataStreamManager, PatchOperator};
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_processor::AvailableProcessors;
use phymes_schemas::{
    AvailableSubjects, AvailableSubjectsTrait, create_workspace_batch, create_workspace_patch_batch,
};
use phymes_network::{DynamicTaskNetworkBuilder, DynamicTaskNetworkNames};
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectPlan,
    SubjectPlanBuilderTrait,
};

pub struct PatchWorkspaceNetworkBuilderStaticWSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for PatchWorkspaceNetworkBuilderStaticWSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "patch";
        let subject_name_out = "apply_patch_s";

        // Processor subject
        let config = DataConfig {
            lhs_name: Some(AvailableSubjects::Workspace.to_string()),
            rhs_name: Some(AvailableSubjects::WorkspacePatch.to_string()),
            lhs_values: Some(vec!["content".to_string()]),
            rhs_values: Some(vec!["diff".to_string(), "operator".to_string()]),
            lhs_pk: Some("path".to_string()),
            rhs_pk: Some("filename".to_string()),
            doc_patch: Some("[\"\"]".to_string()), // DM: equivalent of serde_json::to_string(&[serde_json::to_value("")?])?;
            cpu: false,
            operator: AvailableOperators::Patch,
            lhs_stream: DataStreamManager::Accumulate,
            rhs_stream: Some(DataStreamManager::Accumulate),
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

        // Subscriptions and publications
        let subject = AvailableSubjects::Workspace.to_subject(None, None).unwrap();
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableSubjects::WorkspacePatch
            .to_subject(None, None)
            .unwrap();
        let subject_rhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableSubjects::Workspace
            .to_subject(Some(subject_name_out), None)
            .unwrap();
        let subject_out = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Initialize the network
        let builder = DynamicTaskNetworkBuilder {
            network_name: network_name.to_string(),
            is_dynamic: false,
            processor: AvailableProcessors::Patch,
            subscription_lhs: Subscription::AlwaysAllRecordBatches {
                subject_name: subject_lhs.get_name().to_string(),
            },
            subscription_rhs: Some(Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_rhs.get_name().to_string(),
            }),
            publication: Publication::Extend {
                subject_name: subject_out.get_name().to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_lhs: Some(subject_lhs),
            subject_rhs: Some(subject_rhs),
            subject_out: Some(subject_out),
            subject_processor,
            ..Default::default()
        };

        Self { inner: builder }
    }
}

pub struct PatchWorkspaceNetworkBuilderDynamicWSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for PatchWorkspaceNetworkBuilderDynamicWSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "patch";
        let subject_name_out = "apply_patch_s";

        // Processor subject
        let subject = AvailableSubjects::Bytes
            .to_subject(
                Some(&DynamicTaskNetworkNames::Processor(network_name).to_string()),
                None,
            )
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Subscriptions and publications
        let subject = {
            // Create the mock repository
            let path = [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let content = [
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
            let batch = create_workspace_batch(path, content).unwrap();
            AvailableSubjects::Workspace
                .to_subject(None, Some(vec![batch]))
                .unwrap()
        };
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = {
            // Create the mock patches
            let filename = [
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/other.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let operator = vec![
                PatchOperator::Delete.to_string(),
                PatchOperator::Update.to_string(),
                PatchOperator::Create.to_string(),
            ];
            let content = [
                "",
                "@@ pub mod extra;\n+pub mod other;\n",
                "+pub struct Other {}\n*** End Patch",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let batch = create_workspace_patch_batch(filename, content, operator).unwrap();
            AvailableSubjects::WorkspacePatch
                .to_subject(None, Some(vec![batch]))
                .unwrap()
        };
        let subject_rhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableSubjects::Workspace
            .to_subject(Some(subject_name_out), None)
            .unwrap();
        let subject_out = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Initialize the network
        let builder = DynamicTaskNetworkBuilder {
            network_name: network_name.to_string(),
            is_dynamic: true,
            processor: AvailableProcessors::Patch,
            subscription_lhs: Subscription::AlwaysAllRecordBatches {
                subject_name: subject_lhs.get_name().to_string(),
            },
            subscription_rhs: Some(Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_rhs.get_name().to_string(),
            }),
            publication: Publication::Extend {
                subject_name: subject_out.get_name().to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_lhs: Some(subject_lhs),
            subject_rhs: Some(subject_rhs),
            subject_out: Some(subject_out),
            subject_processor,
            ..Default::default()
        };

        Self { inner: builder }
    }
}

pub struct PatchWorkspaceNetworkBuilderDynamicWOSubject {
    pub inner: DynamicTaskNetworkBuilder,
}

impl Default for PatchWorkspaceNetworkBuilderDynamicWOSubject {
    fn default() -> Self {
        // Initialize the task data
        let network_name = "patch";
        let subject_name_out = "apply_patch_s";

        // Processor subject
        let subject = AvailableSubjects::Bytes
            .to_subject(
                Some(&DynamicTaskNetworkNames::Processor(network_name).to_string()),
                None,
            )
            .unwrap();
        let subject_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Subscriptions and publications
        let subject = AvailableSubjects::Workspace.to_subject(None, None).unwrap();
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableSubjects::WorkspacePatch
            .to_subject(None, None)
            .unwrap();
        let subject_rhs = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject = AvailableSubjects::Workspace
            .to_subject(Some(subject_name_out), None)
            .unwrap();
        let subject_out = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();

        // Initialize the network
        let builder = DynamicTaskNetworkBuilder {
            network_name: network_name.to_string(),
            is_dynamic: true,
            processor: AvailableProcessors::Patch,
            subscription_lhs: Subscription::AlwaysAllRecordBatches {
                subject_name: subject_lhs.get_name().to_string(),
            },
            subscription_rhs: Some(Subscription::OnUpdateAllRecordBatches {
                subject_name: subject_rhs.get_name().to_string(),
            }),
            publication: Publication::Extend {
                subject_name: subject_out.get_name().to_string(),
            },
            subscribe: AvailableSubscribeEvents::AllSubjectNamesSubscribe,
            subject_lhs: Some(subject_lhs),
            subject_rhs: Some(subject_rhs),
            subject_out: Some(subject_out),
            subject_processor,
            ..Default::default()
        };

        Self { inner: builder }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_data::{AvailableOperators, DataConfig, DataStreamManager, PatchOperator};
    use phymes_diagnostics::HashMap;
    use phymes_event::{Publication, Subscription};
    use phymes_message::{IPCMessage, MessageBuilderTrait};
    use phymes_network::{NetworkBuilderAppsTrait, NetworkBuilderTrait, NetworkStream};
    use phymes_schemas::{
        AvailableSubjects, AvailableSubjectsTrait, WorkspacePatchSubject,
        create_bytes_record_batch, create_workspace_batch, create_workspace_patch_batch,
    };
    use phymes_subject::{
        BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnvBuilder, Subject, SubjectBuilder,
        SubjectBuilderTrait, SubjectTrait,
    };
    use phymes_task::SubscriptionTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_network_dynamic_w_subjects() -> Result<()> {
        let patch_workspace_network = PatchWorkspaceNetworkBuilderDynamicWSubject::default();
        let (network, session_messages) = patch_workspace_network
            .inner
            .build_dynamic()
            .with_runtime_env(
                RuntimeEnvBuilder::default()
                    .with_name(
                        &DynamicTaskNetworkNames::Task(&patch_workspace_network.inner.network_name)
                            .to_string(),
                    )
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

        // Apply patch data
        {
            let apply_patch_config = DataConfig {
                lhs_name: Some(AvailableSubjects::Workspace.to_string()),
                rhs_name: Some(AvailableSubjects::WorkspacePatch.to_string()),
                lhs_values: Some(vec!["content".to_string()]),
                rhs_values: Some(vec!["diff".to_string(), "operator".to_string()]),
                lhs_pk: Some("path".to_string()),
                rhs_pk: Some("filename".to_string()),
                doc_patch: Some("[\"\"]".to_string()), // DM: equivalent of serde_json::to_string(&[serde_json::to_value("")?])?;
                cpu: false,
                operator: AvailableOperators::Patch,
                lhs_stream: DataStreamManager::Accumulate,
                rhs_stream: Some(DataStreamManager::Accumulate),
                ..Default::default()
            };
            let apply_patch_config_json = serde_json::to_vec(&apply_patch_config)?;
            let apply_patch_config_batch =
                create_bytes_record_batch(vec![apply_patch_config_json])?;
            let apply_patch_config_table = SubjectBuilder::new()
                .with_name(patch_workspace_network.inner.subject_processor.get_name())
                .with_record_batches(vec![apply_patch_config_batch])?
                .build()?;
            let _ = message_map.insert(
                apply_patch_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(apply_patch_config_table.get_name())
                    .with_publisher(&patch_workspace_network.inner.network_name)
                    .with_subject(apply_patch_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: apply_patch_config_table.get_name().to_string(),
                    })
                    .with_message(apply_patch_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "apply_patch_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("apply_patch_s")
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("path");
        assert_eq!(
            column,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/other.rs"
            ]
        );
        let column = subject.get_column_as_vec_str("content");
        assert_eq!(
            column,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub mod extra;",
                "pub struct Other {}"
            ]
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_network_dynamic_wo_subjects() -> Result<()> {
        let patch_workspace_network = PatchWorkspaceNetworkBuilderDynamicWOSubject::default();
        let (network, session_messages) = patch_workspace_network
            .inner
            .build_dynamic()
            .with_runtime_env(
                RuntimeEnvBuilder::default()
                    .with_name(
                        &DynamicTaskNetworkNames::Task(&patch_workspace_network.inner.network_name)
                            .to_string(),
                    )
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

        // Apply patch data
        {
            // Create the mock repository
            let path = [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let content = [
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
            let batch = create_workspace_batch(path, content)?;
            let table = AvailableSubjects::Workspace.to_subject(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(&patch_workspace_network.inner.network_name)
                    .with_subject(table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: table.get_name().to_string(),
                    })
                    .with_message(table.to_ipc_stream()?)
                    .build()?,
            );

            // Create the mock patches
            let filename = [
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/other.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let operator = vec![
                PatchOperator::Delete.to_string(),
                PatchOperator::Update.to_string(),
                PatchOperator::Create.to_string(),
            ];
            let content = [
                "",
                "@@ pub mod extra;\n+pub mod other;\n",
                "+pub struct Other {}\n*** End Patch",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let values = filename
                .into_iter()
                .zip(content)
                .zip(operator)
                .map(|((filename, diff), operator)| {
                    let patch = WorkspacePatchSubject {
                        filename,
                        diff,
                        operator,
                    };
                    serde_json::to_value(&patch).unwrap()
                })
                .collect::<Vec<_>>();
            let doc_patch = serde_json::to_string(&values)?;

            let apply_patch_config = DataConfig {
                lhs_name: Some(AvailableSubjects::Workspace.to_string()),
                rhs_name: None,
                lhs_values: Some(vec!["content".to_string()]),
                rhs_values: Some(vec!["diff".to_string(), "operator".to_string()]),
                lhs_pk: Some("path".to_string()),
                rhs_pk: Some("filename".to_string()),
                doc_patch: Some(doc_patch),
                cpu: false,
                operator: AvailableOperators::Patch,
                lhs_stream: DataStreamManager::Accumulate,
                rhs_stream: Some(DataStreamManager::Accumulate),
                ..Default::default()
            };
            let apply_patch_config_json = serde_json::to_vec(&apply_patch_config)?;
            let apply_patch_config_batch =
                create_bytes_record_batch(vec![apply_patch_config_json])?;
            let apply_patch_config_table = SubjectBuilder::new()
                .with_name(patch_workspace_network.inner.subject_processor.get_name())
                .with_record_batches(vec![apply_patch_config_batch])?
                .build()?;
            let _ = message_map.insert(
                apply_patch_config_table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(apply_patch_config_table.get_name())
                    .with_publisher(&patch_workspace_network.inner.network_name)
                    .with_subject(apply_patch_config_table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: apply_patch_config_table.get_name().to_string(),
                    })
                    .with_message(apply_patch_config_table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "apply_patch_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("apply_patch_s")
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("path");
        assert_eq!(
            column,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/other.rs"
            ]
        );
        let column = subject.get_column_as_vec_str("content");
        assert_eq!(
            column,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub mod extra;",
                "pub struct Other {}"
            ]
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_patch_workspace_network_static() -> Result<()> {
        let patch_workspace_network = PatchWorkspaceNetworkBuilderStaticWSubject::default();
        let (network, session_messages) = patch_workspace_network
            .inner
            .build_dynamic()
            .with_runtime_env(
                RuntimeEnvBuilder::default()
                    .with_name(
                        &DynamicTaskNetworkNames::Task(&patch_workspace_network.inner.network_name)
                            .to_string(),
                    )
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

        // Apply patch data
        {
            // Create the mock repository
            let path = [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let content = [
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
            let batch = create_workspace_batch(path, content)?;
            let table = AvailableSubjects::Workspace.to_subject(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(&patch_workspace_network.inner.network_name)
                    .with_subject(table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: table.get_name().to_string(),
                    })
                    .with_message(table.to_ipc_stream()?)
                    .build()?,
            );

            // Create the mock patches
            let filename = [
                "/home/sandbox/src/main.rs",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/other.rs",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let operator = vec![
                PatchOperator::Delete.to_string(),
                PatchOperator::Update.to_string(),
                PatchOperator::Create.to_string(),
            ];
            let content = [
                "",
                "@@ pub mod extra;\n+pub mod other;\n",
                "+pub struct Other {}\n*** End Patch",
            ]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
            let batch = create_workspace_patch_batch(filename, content, operator)?;
            let table = AvailableSubjects::WorkspacePatch.to_subject(None, Some(vec![batch]))?;
            let _ = message_map.insert(
                table.get_name().to_string(),
                IPCMessage::get_builder()
                    .with_name(table.get_name())
                    .with_publisher(&patch_workspace_network.inner.network_name)
                    .with_subject(table.get_name())
                    .with_update(&Publication::Replace {
                        subject_name: table.get_name().to_string(),
                    })
                    .with_message(table.to_ipc_stream()?)
                    .build()?,
            );
        }

        // Run the session
        let _ = network_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;
        let network_stream = NetworkStream::new(message_map, Arc::clone(&network_arc));
        let response: Vec<HashMap<String, IPCMessage>> = network_stream.try_collect().await?;

        assert_eq!(response.len(), 0);

        // Test supsersteps
        let batches: Vec<_> = Subscription::AlwaysAllRecordBatches {
            subject_name: "apply_patch_s".to_string(),
        }
        .subscribe_to_subject(network_arc.runtime_env(), network_arc.get_name())?
        .unwrap()
        .try_collect()
        .await?;
        let subject = Subject::get_builder()
            .with_name("apply_patch_s")
            .with_record_batches(batches)?
            .build()?;
        let column = subject.get_column_as_vec_str("path");
        assert_eq!(
            column,
            [
                "/home/sandbox/Cargo.toml",
                "/home/sandbox/src/extras/mod.rs",
                "/home/sandbox/src/extras/todo.rs",
                "/home/sandbox/src/lib.rs",
                "/home/sandbox/src/extras/other.rs"
            ]
        );
        let column = subject.get_column_as_vec_str("content");
        assert_eq!(
            column,
            [
                "[package]\nname = \"phymes_rs\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[dependencies]\nanyhow = { version = \"1\", default-features = false }",
                "pub mod other;\nmod todo;\npub use todo::Todo",
                "pub struct Todo {}",
                "pub mod extra;",
                "pub struct Other {}"
            ]
        );
        Ok(())
    }
}
