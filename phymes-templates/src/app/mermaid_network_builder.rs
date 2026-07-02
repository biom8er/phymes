use std::sync::Arc;

use anyhow::Result;
use phymes_diagnostics::create_timestamp_micros;
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_network::{NetworkBuilderCustomTrait, NetworkBuilderMermaidTrait};
use phymes_processor::{AvailableProcessors, ProcessorPlan, ProcessorPlanBuilder};
use phymes_schemas::{AvailableSubjects, create_network_mermaid_batch};
use phymes_subject::{
    BuildableTrait, BuilderTrait, RuntimeEnv, Subject, SubjectBuilderTrait, SubjectPlan,
    SubjectPlanBuilderTrait,
};
use phymes_task::TaskPlan;

use crate::AvailableNetworks;

/// Example Mermaid diagrams for available app sessions
pub fn make_example_mermaid_table(deployable: bool, builder: bool) -> Result<Subject> {
    let available_session_plans = if deployable {
        AvailableNetworks::get_deployable_session_plan_names()
    } else {
        AvailableNetworks::get_all_session_plan_names()
    };

    // Initialize with available app session diagrams
    let mut network_names = Vec::new();
    let mut flowchart_diagram = Vec::new();
    let mut er_diagram = Vec::new();
    let mut timestamp = Vec::new();
    for network_name in available_session_plans {
        let builder = AvailableNetworks::get_network_builder_by_name(&network_name, &network_name)?
            .with_name(&network_name);
        flowchart_diagram.push(builder.to_mermaid_flowchart(false, false)?);
        er_diagram.push(builder.to_mermaid_erdiagram(false, true)?);
        network_names.push(network_name);
        timestamp.push(create_timestamp_micros());
    }

    // Create the table
    let subject_name = if builder {
        AvailableSubjects::BuilderMermaid.to_string()
    } else {
        AvailableSubjects::SessionMermaid.to_string()
    };
    let batch =
        create_network_mermaid_batch(network_names, flowchart_diagram, er_diagram, timestamp)?;
    Subject::get_builder()
        .with_name(subject_name.as_str())
        .with_record_batches(vec![batch])?
        .build()
}

/// Session for building new networks via Mermaid diagrams
pub struct MermaidNetworkBuilder<'a> {
    /// Session and state
    pub network_name: &'a str,
}

impl Default for MermaidNetworkBuilder<'_> {
    fn default() -> Self {
        MermaidNetworkBuilder {
            network_name: "network_1",
        }
    }
}

impl<'a> MermaidNetworkBuilder<'a> {
    pub fn new_with_network_name(network_name: &'a str) -> Self {
        MermaidNetworkBuilder { network_name }
    }
}

impl NetworkBuilderCustomTrait for MermaidNetworkBuilder<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![TaskPlan {
            task_name: self.network_name.to_string(),
            processor_names: vec![self.network_name.to_string()],
        }];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorEcho.build_arc(self.network_name))
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

    use phymes_network::NetworkBuilderAppsTrait;

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_mermaid_network_builder() -> Result<()> {
        // initialize the session
        let builder_network = MermaidNetworkBuilder::default();
        let _network = builder_network
            .build()
            .with_name(builder_network.network_name)
            .build_with_tables()?;

        Ok(())
    }
}
