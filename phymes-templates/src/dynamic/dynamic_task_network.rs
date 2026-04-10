use std::sync::Arc;

use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv, Subject, SubjectPlan, SubjectPlanBuilderTrait
};
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_processor::{AvailableProcessors, ProcessorPlan, ProcessorPlanBuilder};
use phymes_schemas::{AvailableSubjects, AvailableSubjectsTrait};
use phymes_task::TaskPlan;
use phymes_network::CustomAgentsBuilderTrait;

/// Template dynamic (or static) task creation network
///   that is intended to be extended with a base network to enable dynamic task invokation
///   or extended with a network of the same name to create a static processor pipeline
pub struct DynamicTaskNetwork {
    /// Network name (task name)
    pub network_name: String,
    /// Dynamic pipeline (e.g., tool call) or static pipeline
    pub is_dynamic: bool,
    /// The processor to use
    pub processor: AvailableProcessors,
    /// LHS subscription
    pub subscription_lhs: Subscription,
    /// RHS subscription
    pub subscription_rhs: Option<Subscription>,
    /// Output publication
    pub publication: Publication,
    /// Subscribe event
    pub subscribe: AvailableSubscribeEvents,
    /// LHS subject
    pub subject_lhs: SubjectPlan,
    /// RHS subject
    pub subject_rhs: Option<SubjectPlan>,
    /// Output subject
    pub subject_out: SubjectPlan,
    /// Config data
    pub config: Option<Subject>
}

impl Default for DynamicTaskNetwork {
    fn default() -> Self {
        let subject_lhs = SubjectPlan::get_builder()
            .with_subject(AvailableSubjects::Bytes.to_subject(Some("lhs_s"), None).unwrap())
            .build()
            .unwrap();
        let subject_out = SubjectPlan::get_builder()
            .with_subject(AvailableSubjects::Bytes.to_subject(Some("out_s"), None).unwrap())
            .build()
            .unwrap();
        DynamicTaskNetwork {
            network_name: "network_1".to_string(),
            is_dynamic: false,
            processor: AvailableProcessors::default(),
            subscription_lhs: Subscription::OnUpdateAllRecordBatches { subject_name: subject_lhs.get_name().to_string() },
            subscription_rhs: None,
            publication: Publication::Replace { subject_name: subject_out.get_name().to_string() },
            subscribe: AvailableSubscribeEvents::default(),
            subject_lhs,
            subject_rhs: None,
            subject_out,
            config: None,
        }
    }
}

impl DynamicTaskNetwork {
    pub fn new_with_network_name(network_name: &str) -> Self {
        DynamicTaskNetwork {
            network_name: network_name.to_string(),
            ..Default::default()
        }
    }
    fn task_name(&self) -> String {
        format!("{}_t", self.network_name)
    }
    fn processor_name(&self) -> String {
        format!("{}_p", self.network_name)
    }
}

impl CustomAgentsBuilderTrait for DynamicTaskNetwork {
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
        subscriptions.push(self.subscription_lhs.clone());

        // RHS
        if let Some(subscription_rhs) = self.subscription_rhs.as_ref() {
            subscriptions.push(subscription_rhs.clone());
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
                .with_publications(&[self.publication.clone()])
                .with_subscriptions(&subscriptions)
                .with_subscribe_policy(self.subscribe.clone().build())
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
        let mut subject_plans = Vec::new();
        subject_plans.push(self.subject_lhs.clone());
        if let Some(subject_rhs) = self.subject_rhs.as_ref() {
            subject_plans.push(subject_rhs.clone());
        }
        subject_plans.push(self.subject_out.clone());
        if let Some(config) = self.config.as_ref() {
            let subject_plan = SubjectPlan::get_builder().with_subject(config.to_owned()).build().unwrap();
            subject_plans.push(subject_plan);
        } else {
            let subject = AvailableSubjects::Bytes
                .to_subject(Some(&self.processor_name()), None)
                .unwrap();
            let subject_plan = SubjectPlan::get_builder().with_subject(subject).build().unwrap();
            subject_plans.push(subject_plan);
        }

        Some(subject_plans)
    }
}
