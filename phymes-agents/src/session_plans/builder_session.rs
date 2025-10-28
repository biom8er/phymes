use crate::{
    session_plans::AvailableSessionPlans,
    session_traits::{CustomAgentsBuilderTrait, SessionContextBuilderMermaidTrait},
};
use anyhow::Result;
use phymes_core::{
    AllTableNamesSubscribe, AvailableSubjects, BuildableTrait, BuilderTrait, ProcessorEcho,
    ProcessorTrait, RuntimeEnv, RuntimeEnvTrait, SubscribeTrait, Table, TableBuilderTrait,
    TablePublish, TableSubscribe, TaskPlan, create_session_mermaid_batch,
};
use phymes_diagnostics::create_timestamp_micros;
use std::sync::Arc;

/// Example Mermaid diagrams for chat, doc, and tool agent sessions
pub fn make_example_mermaid_table(deployable: bool, builder: bool) -> Result<Table> {
    let available_session_plans = if deployable {
        AvailableSessionPlans::get_deployable_session_plan_names()
    } else {
        AvailableSessionPlans::get_all_session_plan_names()
    };

    // Initialize with chat, doc, and tool agent session diagrams
    let mut session_context_names = Vec::new();
    let mut flowchart_diagram = Vec::new();
    let mut er_diagram = Vec::new();
    let mut timestamp = Vec::new();
    for session_context_name in available_session_plans {
        let builder = AvailableSessionPlans::get_session_context_builder_by_name(
            &session_context_name,
            &session_context_name,
        )?
        .with_name(&session_context_name);
        flowchart_diagram.push(builder.to_mermaid_flowchart(false, false)?);
        er_diagram.push(builder.to_mermaid_erdiagram(false, true)?);
        session_context_names.push(session_context_name);
        timestamp.push(create_timestamp_micros());
    }

    // Create the table
    let table_name = if builder {
        AvailableSubjects::BuilderMermaid.to_string()
    } else {
        AvailableSubjects::SessionMermaid.to_string()
    };
    let batch = create_session_mermaid_batch(
        session_context_names,
        flowchart_diagram,
        er_diagram,
        timestamp,
    )?;
    Table::get_builder()
        .with_name(table_name.as_str())
        .with_record_batches(vec![batch])?
        .build()
}

/// Session for building new sessions via Mermaid diagrams
pub struct BuilderSession<'a> {
    /// Session and state
    pub session_context_name: &'a str,
}

impl Default for BuilderSession<'_> {
    fn default() -> Self {
        BuilderSession {
            session_context_name: "session_context_1",
        }
    }
}

impl<'a> BuilderSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        BuilderSession {
            session_context_name,
        }
    }
}

impl CustomAgentsBuilderTrait for BuilderSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![TaskPlan {
            task_name: self.session_context_name.to_string(),
            runtime_env_name: "rt_default".to_string(),
            processor_names: vec![self.session_context_name.to_string()],
        }];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![ProcessorEcho::new_arc_with_pub_sub(
            self.session_context_name,
            &[TablePublish::Extend {
                table_name: AvailableSubjects::BuilderMermaid.to_string(),
            }],
            &[TableSubscribe::OnUpdateLastRecordBatch {
                table_name: AvailableSubjects::BuilderMermaid.to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        )];

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![RuntimeEnv::new().with_name("rt_default")])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
        Some(vec![make_example_mermaid_table(true, true).unwrap()])
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use parking_lot::RwLock;
    use phymes_core::SessionStreamState;

    use crate::session_traits::SessionContextBuilderAgentsTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_builder_agent_session() -> Result<()> {
        // initialize the session
        let builder_agent_session = BuilderSession::default();
        let session_ctx = builder_agent_session
            .build()
            .with_name(builder_agent_session.session_context_name)
            .build_with_tables()?;
        let _session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        Ok(())
    }
}
