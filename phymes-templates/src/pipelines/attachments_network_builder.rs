use phymes_data::DataConfigTrait;
use phymes_event::{AvailableSubscribeEvents, Publication, Subscription};
use phymes_network::{DynamicTaskNetworkNames, NetworkBuilderCustomTrait};
use phymes_processor::{AvailableProcessors, ProcessorPlan, ProcessorPlanBuilder};
use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjectsTrait};
use phymes_subject::{
    BuildableTrait, BuilderTrait, SubjectBuilder, SubjectBuilderTrait, SubjectPlan,
    SubjectPlanBuilderTrait,
};
use phymes_task::TaskPlan;

/// Attatchments aggregator network builder
///
/// # Todo
/// * Generalize to aggregator network builder
pub struct AttachmentsNetworkBuilder<'a> {
    pub network_name: &'a str,
    // todo: Change to `subject_names_lhs`
    pub subject_names: &'a [&'a str],
    // pub subject_out: &'a SubjectPlan,
}

impl<'a> AttachmentsNetworkBuilder<'a> {
    pub fn new(network_name: &'a str, subject_names: &'a [&'a str]) -> Self {
        Self {
            network_name,
            subject_names,
        }
    }
}

impl<'a> NetworkBuilderCustomTrait for AttachmentsNetworkBuilder<'a> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        let tasks = vec![TaskPlan {
            task_name: DynamicTaskNetworkNames::Task(self.network_name).to_string(),
            processor_names: vec![
                DynamicTaskNetworkNames::Processor(
                    &AvailableProcessors::AggregatorProcessor.to_string(),
                )
                .to_string(),
            ],
        }];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        let subscriptions = self
            .subject_names
            .iter()
            .map(|s| Subscription::OnUpdateAllRecordBatches {
                subject_name: s.to_string(),
            })
            .chain([Subscription::AlwaysLastRecordBatch {
                subject_name: DynamicTaskNetworkNames::Processor(
                    &AvailableProcessors::AggregatorProcessor.to_string(),
                )
                .to_string(),
            }])
            .collect::<Vec<_>>();
        let publications = [Publication::Extend {
            subject_name: AvailableInterfaceSubjects::AggregatedAttachments.to_string(),
        }];
        let processor = AvailableProcessors::AggregatorProcessor.build_arc(
            &DynamicTaskNetworkNames::Processor(
                &AvailableProcessors::AggregatorProcessor.to_string(),
            )
            .to_string(),
        );
        let subscribe_policy = AvailableSubscribeEvents::AnySubjectNameSubscribe.build();

        // Build the processor
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(processor)
                .with_publications(&publications)
                .with_subscriptions(&subscriptions)
                .with_subscribe_policy(subscribe_policy)
                .build()
                .unwrap(),
        ];

        Some(processors)
    }

    fn make_subjects(&self) -> Option<Vec<SubjectPlan>> {
        let config_json = AvailableProcessors::AggregatorProcessor
            .to_example_json()
            .unwrap();
        let subject = SubjectBuilder::new()
            .with_name(
                &DynamicTaskNetworkNames::Processor(
                    &AvailableProcessors::AggregatorProcessor.to_string(),
                )
                .to_string(),
            )
            .with_json(&config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let subject_plan_processor = SubjectPlan::get_builder()
            .with_subject(subject)
            .build()
            .unwrap();
        let subject_plan = AvailableInterfaceSubjects::AggregatedAttachments
            .to_subject_plan(None, None)
            .unwrap();
        let subject_plans = vec![subject_plan_processor, subject_plan];

        Some(subject_plans)
    }
}
