use std::sync::Arc;

use anyhow::Result;
use phymes_core::{BuildableTrait, BuilderTrait, RuntimeEnv, Subject, SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait};
use phymes_diagnostics::create_timestamp_micros;
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_schemas::{AvailableSubjects, create_session_mermaid_batch};
use phymes_processor::{AvailableProcessors, ProcessorPlan, ProcessorPlanBuilder};
use phymes_task::TaskPlan;

use crate::{AvailableSessionPlans, CustomAgentsBuilderTrait, SessionContextBuilderMermaidTrait};

/// Example Mermaid diagrams for chat, doc, and tool agent sessions
pub fn make_example_mermaid_table(deployable: bool, builder: bool) -> Result<Subject> {
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
    let subject_name = if builder {
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
    Subject::get_builder()
        .with_name(subject_name.as_str())
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
            processor_names: vec![self.session_context_name.to_string()],
        }];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ProcessorEcho.build_arc(self.session_context_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: AvailableSubjects::BuilderMermaid.to_string(),
                }])
                .with_subscriptions(&[Subscription::OnUpdateLastRecordBatch {
                    subject_name: AvailableSubjects::BuilderMermaid.to_string(),
                }])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
        ];

        Some(processors)
    }

    fn make_runtime_env(&self) -> Option<Arc<RuntimeEnv>> {
        Some(
            RuntimeEnv::get_builder()
                .with_name("rt_default")
                .build_arc()
                .unwrap(),
        )
    }

    fn make_subjects(&self) -> Option<Vec<SubjectPlan>> {
        let subject_plan = SubjectPlan::get_builder()
            .with_subject(make_example_mermaid_table(true, true).unwrap())
            .build()
            .unwrap();
        Some(vec![subject_plan])
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::SessionContextBuilderAgentsTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_builder_agent_session() -> Result<()> {
        // initialize the session
        let builder_agent_session = BuilderSession::default();
        let _session_ctx = builder_agent_session
            .build()
            .with_name(builder_agent_session.session_context_name)
            .build_with_tables()?;

        Ok(())
    }
}
