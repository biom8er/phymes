use std::sync::Arc;

use phymes_core::{
    schemas::{available_subjects::AvailableSubjects, builder::BuilderBuilderTraitExt}, session::{
        common_traits::{BuildableTrait, BuilderTrait},
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context_builder::TaskPlan,
    }, table::{
        table::Table, table_publish::TablePublish, table_subscribe::{AllTableNamesSubscribe, SubscribeTrait, TableSubscribe}
    }, task::processor::{ProcessorEcho, ProcessorTrait}
};

use crate::{session_plans::available_session_plans::AvailableSessionPlans, session_traits::{agents::CustomAgentsBuilderTrait, mermaid::SessionContextBuilderMermaidTrait}};

pub struct BuilderAgentSession<'a> {
    /// Session and state
    pub session_context_name: &'a str,
}

impl Default for BuilderAgentSession<'_> {
    fn default() -> Self {
        BuilderAgentSession {
            session_context_name: "session_context_1",
        }
    }
}

impl<'a> BuilderAgentSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        BuilderAgentSession {
            session_context_name,
            ..Default::default()
        }
    }
}

impl CustomAgentsBuilderTrait for BuilderAgentSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let mut tasks = Vec::new();

        tasks.push(TaskPlan {
            task_name: self.session_context_name.to_string(),
            runtime_env_name: "rt_default".to_string(),
            processor_names: vec![self.session_context_name.to_string()],
        });

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        // The order is the order in which the processors are called in the task
        let mut processors = Vec::new();
        processors.push(ProcessorEcho::new_arc_with_pub_sub(
            self.session_context_name,
            &[
                TablePublish::Extend {
                    table_name: AvailableSubjects::Builder.to_string(),
                },
            ],
            &[TableSubscribe::OnUpdateLastRecordBatch {
                table_name: AvailableSubjects::Builder.to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        ));

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name("rt_default"),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
        // Initialize with chat, doc, and tool agent session diagrams
        let mut table_builder = Table::get_builder()
            .with_name(AvailableSubjects::Builder.to_string().as_str());
        for session_context_name in ["Chat", "DocChat", "ToolChat"] {
            let builder = AvailableSessionPlans::get_session_context_builder_by_name(session_context_name, session_context_name).unwrap();
            let flowchart_diagram = builder.to_mermaid_flowchart().unwrap();
            let er_diagram = builder.to_mermaid_erdiagram().unwrap();
            table_builder = table_builder.with_builder(
                Some(AvailableSessionPlans::Chat.to_string().as_str()), 
                Some(&flowchart_diagram), 
                Some(&er_diagram)).unwrap();
        }
        Some(vec![           
            table_builder.build().unwrap(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use parking_lot::RwLock;
    use phymes_core::{metrics::ArrowTaskMetricsSet, session::{session_context::SessionStreamState, session_context_builder::SessionContextBuilderTrait}};

    use crate::session_traits::agents::SessionContextBuilderAgentsTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_builder_agent_session() -> Result<()> {
        // initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // initialize the session
        let builder_agent_session = BuilderAgentSession::default();
        let session_ctx = builder_agent_session
            .build()
            .with_metrics(metrics.clone())
            .with_name(builder_agent_session.session_context_name)
            .build_with_tables()?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        Ok(())
    }
}
