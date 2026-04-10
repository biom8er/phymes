use std::sync::Arc;

use phymes_subject::{
    BuildableTrait, BuilderTrait, RuntimeEnv, SubjectBuilder, SubjectBuilderTrait, SubjectPlan,
    SubjectPlanBuilderTrait,
};
use phymes_data::{AvailableOperators, DataConfig};
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_processor::{AvailableProcessors, ProcessorPlan, ProcessorPlanBuilder};
use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjects, AvailableSubjectsTrait};
use phymes_task::TaskPlan;
use phymes_network::CustomAgentsBuilderTrait;

/// Template dynamic (or static) task creation network
///   that is intended to be extended with a base network to enable dynamic task invokation
///   or extended with a network of the same name to create a static processor pipeline
pub struct DynamicTaskNetwork<'a> {
    /// Network name (task name)
    pub network_name: &'a str,
    /// Dynamic pipeline (e.g., tool call) or static pipeline
    pub is_dynamic: bool,
    /// The processor to use
    pub processor: AvailableProcessors,
    /// LHS subject
    pub subject_name_lhs: &'a str,
    /// RHS subject
    pub subject_name_rhs: Option<&'a str>,
    /// Output subject
    pub subject_name_o: &'a str,
}

impl Default for DynamicTaskNetwork<'_> {
    fn default() -> Self {
        DynamicTaskNetwork {
            network_name: "network_1",
            is_dynamic: false,
            processor: AvailableProcessors::default(),
            subject_name_lhs: "subject_name_lhs",
            subject_name_rhs: None,
            subject_name_o: "subject_name_o",
        }
    }
}

impl<'a> DynamicTaskNetwork<'a> {
    pub fn new_with_network_name(network_name: &'a str) -> Self {
        DynamicTaskNetwork {
            network_name,
            ..Default::default()
        }
    }
    fn task_name(&self) -> String {
        format!("{}_t", self.network_name)
    }
    fn processor_name(&self) -> String {
        format!("{}_p", self.network_name)
    }
    fn subject_name(&self, subject_name: &str) -> String {
        format!("{subject_name}_s")
    }
}

impl CustomAgentsBuilderTrait for DynamicTaskNetwork<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![
            TaskPlan {
                task_name: self.task_name(),
                processor_names: vec![self.processor_name()],
            },
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        // Build the subscriptions based on dynamic and RHS
        let mut subscriptions = Vec::new();

        // LHS
        {
            let subscription = Subscription::OnUpdateAllRecordBatches {
                subject_name: self.subject_name(self.subject_name_lhs),
            };
            subscriptions.push(subscription);
        }

        // RHS
        if let Some(subject_name_rhs) = self.subject_name_rhs {
            let subscription = Subscription::OnUpdateAllRecordBatches {
                subject_name: self.subject_name(subject_name_rhs),
            };
            subscriptions.push(subscription);
        }

        // Dynamic
        if self.is_dynamic {
            let subscription = Subscription::OnUpdateLastRecordBatch {
                subject_name: self.processor_name(),
            };
            subscriptions.push(subscription)
        } else {
            let subscription = Subscription::AlwaysLastRecordBatch {
                subject_name: self.processor_name(),
            };
            subscriptions.push(subscription)
        }

        // Build the processor
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(
                    self.processor.build_arc(&self.processor_name()),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.subject_name(self.subject_name_o),
                }])
                .with_subscriptions(&subscriptions)
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
        ];

        Some(processors)
    }

    fn make_runtime_env(&self) -> Option<Arc<RuntimeEnv>> {
        // Intended to be extended so the runtime should be defined in the base Network
        None
    }

    fn make_subjects(&self) -> Option<Vec<SubjectPlan>> {
        // Intended to be extended so the subjects should be defined in the base Network
        let mut subjects = Vec::new();
        subjects.push(AvailableSubjects::Bytes
            .to_subject(Some(&self.subject_name(self.subject_name_lhs)), None)
            .unwrap());
        if let Some(subject_name_rhs) = self.subject_name_rhs {
            subjects.push(AvailableSubjects::Bytes
                .to_subject(Some(&self.subject_name(subject_name_rhs)), None)
                .unwrap());
        }
        subjects.push(AvailableSubjects::Bytes
            .to_subject(Some(&self.subject_name(self.subject_name_o)), None)
            .unwrap());
        subjects.push(AvailableSubjects::Bytes
            .to_subject(Some(&self.processor_name()), None)
            .unwrap());
        
        // Wrap into the subject plan
        let subject_plans = subjects
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();
        Some(subject_plans)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_subject::{BuildableTrait, MappableTrait, SubjectTrait};
    use phymes_diagnostics::HashMap;
    use phymes_message::{IPCMessage, MessageBuilderTrait, MessageTrait, create_message_map};
    use phymes_streams::ChatBuilderTraitExt;
    use phymes_network::{NetworkBuilderAgentsTrait, NetworkStream};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_chat_agent_network() -> Result<()> {

        Ok(())
    }
}
