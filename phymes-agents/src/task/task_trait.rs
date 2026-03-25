use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::record_batch::RecordBatch;
use phymes_core::{
    BuildableTrait, MappableTrait, ProcessorSubjectsMap, ProcessorTrait, RuntimeEnv,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, Subscription,
};
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, TraceBuilderTrait};
use tracing::{Level, event};

use crate::{TaskBuilder, build_and_publish_to_stream, subscribe_to_subject, update_publisher};

/// Trait to implement the actual task which could involve one or
///   more operators over [`RecordBatch`]s often originating from
///   structs implementing the [`SubjectTrait`].
///
/// [`SubjectTrait`]: phymes_core::SubjectTrait
///
/// The trait allows for the schema of the data to change (e.g. after joins),
///   but the logic must be implemented by the user
/// The trait allows for tasks to have access to local data
///
/// # Example: Chaining
///
/// Result = Task (t1) -> Task (t2) -> Task (t3)
/// where t3 represents the leaf node in the computation and t1 and t2 represent
///   intermediate nodes in the computation tree. The `run` method of `RunnableTrait` of
///   t3 will most likely just produce a stream of its underlying [`RecordBatch`]s
///   while t1 and t2 will operate over incoming streams of [`RecordBatch`]s
///
/// Chaining use cases would include RAG, database query, etc.
///
/// # Example: Directed Cyclic Graph
///
/// Result = Task (t1) -> Or(Task (t2) -> Task (t1), Task(t3) -> Task (t1), End)
/// where t1 can call one or more tasks which that run a task and return the results to t1
///    or stop the loop when a criteria is reached
///
/// DCG use cases would include an agentic AI application, etc.
///
/// # Example: Parallel execution
///
/// Result = Apply Task (t1) over (Taable (d1), Taable (d2), Taable (d3), ...)
/// where the same task is run over different ArrowTables in parallel.
///
/// Parallel execution could be integrated into any uses case to improve execution speed
pub trait TaskTrait: MappableTrait + BuildableTrait + Sync + Send {
    /// Run the computation
    fn run(
        &self,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        processor_subjects: &ProcessorSubjectsMap,
        runtime_env: &Arc<RuntimeEnv>,
        session_name: &str,
    ) -> Result<SendableRecordBatchStreamMessageMap>;

    /// Get an immutable reference to the processors
    fn get_processors(&self) -> &Vec<Arc<dyn ProcessorTrait>>;
}

/// The actual task to execute
#[derive(Default, Debug)]
pub struct Task {
    /// Name of the task
    pub(crate) name: String,
    /// Processor sequence
    pub(crate) processor: Vec<Arc<dyn ProcessorTrait>>,
}

impl MappableTrait for Task {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl BuildableTrait for Task {
    type T = TaskBuilder;
    fn get_builder() -> Self::T
    where
        Self: Sized,
    {
        Self::T::default()
    }
}

impl TaskTrait for Task {
    fn run(
        &self,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        processor_subjects: &ProcessorSubjectsMap,
        runtime_env: &Arc<RuntimeEnv>,
        session_name: &str,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        event!(Level::INFO, "Running task {}", self.get_name());

        // Build the tracer for the task and processors, and trace the subscriptions
        let (trace, trace_builder) = if let Some(diagnostic_builder) = diagnostic_builder {
            let trace_builder = diagnostic_builder.clone().to_child(self.get_name())?;
            let trace = trace_builder
                .clone()
                .messages(line!(), file!(), self.get_name());
            trace.enter(
                &processor_subjects
                    .values()
                    .flat_map(|p| &p.subscriptions)
                    .collect::<Vec<_>>(),
            );
            (Some(trace), Some(trace_builder))
        } else {
            (None, None)
        };

        // Run the processing sequence and collect the messages
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        for processor in self.processor.iter() {
            // Subscribe to the processor subjects
            let processor_subject = processor_subjects.get(processor.get_name()).ok_or(anyhow!(
                "Processor `{}` not found in processor subscriptions and publications `{:?}`.",
                processor.get_name(),
                processor_subjects.keys()
            ))?;
            let message_sub: HashMap<String, SendableRecordBatchStreamMessage> =
                subscribe_to_subject(
                    &processor_subject.subscriptions,
                    &processor_subject.publications,
                    &runtime_env,
                    session_name,
                    &mut messages,
                )?;

            // Trace the processor subscribed messages
            let (trace, trace_builder) = if let Some(diagnostic_builder) = trace_builder.as_ref() {
                let trace_builder = diagnostic_builder.clone().to_child(processor.get_name())?;
                let (line, file) = processor.line_and_file();
                let trace = trace_builder
                    .clone()
                    .messages(line, &file, processor.get_name());
                trace.enter(&message_sub.values().collect::<Vec<_>>());
                (Some(trace), Some(trace_builder))
            } else {
                (None, None)
            };

            // Run the processor
            let message_builder =
                processor.process(message_sub, trace_builder.as_ref(), runtime_env.clone())?;

            // Build and trace the processor published messages
            let message_pub = build_and_publish_to_stream(
                processor.get_name(),
                &processor_subject.publications.iter().collect::<Vec<_>>(),
                message_builder,
            )?;
            if let Some(trace) = trace {
                trace.exit(&message_pub.values().collect::<Vec<_>>());
            }

            // Update the message stream
            messages.extend(message_pub);
        }

        // Prepare the messages to publish
        let messages = update_publisher(self.get_name(), messages)?;

        // Trace the messages to publish
        if let Some(trace) = trace {
            trace.exit(&messages.values().collect::<Vec<_>>());
        }
        Ok(messages)
    }
    fn get_processors(&self) -> &Vec<Arc<dyn ProcessorTrait>> {
        &self.processor
    }
}

/// Mock objects and functions for task testing
pub mod test_task {
    use super::*;
    use crate::TaskBuilderTrait;
    use phymes_core::{
        BuildableTrait, BuilderTrait, IPCMessage, IPCMessageBuilder, IPCMessageMap, MappableTrait,
        MessageBuilderTrait, ProcessorBuilder, ProcessorSubjects, ProcessorSubjectsBuilder,
        Publication, RuntimeEnv, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectPlan,
        SubjectPlanBuilderTrait, SubjectTrait, test_processor, test_subject,
    };

    use arrow::array::{ArrayRef, BooleanArray, StringArray};
    use phymes_diagnostics::HashMap;
    use std::sync::Arc;

    pub fn make_config_tables(config_name: &str) -> Result<Subject> {
        // mock config for the task
        let a: ArrayRef = Arc::new(StringArray::from(vec!["HumanInTheLoop".to_string()]));
        let b: ArrayRef = Arc::new(BooleanArray::from(vec![true]));
        let c: ArrayRef = Arc::new(StringArray::from(vec!["Accumulate".to_string()]));
        let batch =
            RecordBatch::try_from_iter(vec![("operator", a), ("cpu", b), ("lhs_stream", c)])?;
        let config = SubjectBuilder::new()
            .with_name(config_name)
            .with_record_batches(vec![batch])?
            .build()?;
        Ok(config)
    }

    pub fn make_subject_tables(subject_name: &str, config_name: &str) -> Result<Vec<Subject>> {
        // mock config for the task
        let config = make_config_tables(config_name)?;

        // mock table for the task
        let table = test_subject::make_test_subject(subject_name, 4, 8, 3)?;
        Ok(vec![config, table])
    }

    pub fn make_subject_tables_empty(
        subject_name: &str,
        config_name: &str,
    ) -> Result<Vec<Subject>> {
        // mock config for the task
        let a: ArrayRef = Arc::new(StringArray::from(vec!["".to_string()]));
        let b: ArrayRef = Arc::new(BooleanArray::from(vec![true]));
        let c: ArrayRef = Arc::new(StringArray::from(vec!["".to_string()]));
        let batch =
            RecordBatch::try_from_iter(vec![("operator", a), ("cpu", b), ("lhs_stream", c)])?;
        let config = SubjectBuilder::new()
            .with_name(config_name)
            .with_record_batches(vec![batch])?
            .build()?;

        // mock table for the task
        let table = test_subject::make_test_subject(subject_name, 1, 8, 1)?;
        Ok(vec![config, table])
    }

    pub fn make_subjects(subject_name: &str, config_name: &str) -> Result<Vec<SubjectPlan>> {
        let tables = make_subject_tables(subject_name, config_name)?;
        let mut subject_plans = Vec::new();
        for table in tables.into_iter() {
            let plan = SubjectPlan::get_builder().with_subject(table).build()?;
            subject_plans.push(plan);
        }
        Ok(subject_plans)
    }

    pub fn make_runtime_env(name: &str) -> Result<Arc<RuntimeEnv>> {
        let rt = RuntimeEnv::get_builder().with_name(name).build()?;
        Ok(Arc::new(rt))
    }

    pub fn make_test_task_single_processor(
        task_name: &str,
        processor_name: &str,
        subject_name: &str,
    ) -> Result<(Task, ProcessorSubjectsMap)> {
        let processor = ProcessorBuilder::default()
            .with_name(processor_name)
            .with_type(test_processor::ProcessorMock::get_static_name())
            .build_arc::<test_processor::ProcessorMock>()?;
        let task = Task::get_builder()
            .with_name(task_name)
            .with_processor(vec![processor])
            .build()?;
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name(processor_name)
            .with_subscriptions(&[
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: subject_name.to_string(),
                },
                Subscription::AlwaysAllRecordBatches {
                    subject_name: processor_name.to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: subject_name.to_string(),
            }])
            .build()?;
        let mut processor_subjects_map = HashMap::<String, ProcessorSubjects>::new();
        let _ = processor_subjects_map.insert(processor_name.to_string(), processor_subjects);
        Ok((task, processor_subjects_map))
    }

    pub fn make_test_task_chained_processor(
        task_name: &str,
        processor_name: &str,
        subject_name: &str,
    ) -> Result<(Task, ProcessorSubjectsMap)> {
        let processor_name_1 = format!("{processor_name}_1");
        let processor_name_2 = format!("{processor_name}_2");
        let processor_name_3 = format!("{processor_name}_3");
        let table_name_1 = format!("{subject_name}_1");
        let task = Task::get_builder()
            .with_name(task_name)
            .with_processor(vec![
                ProcessorBuilder::default()
                    .with_name(processor_name_1.as_str())
                    .with_type(test_processor::ProcessorMock::get_static_name())
                    .build_arc::<test_processor::ProcessorMock>()?,
                ProcessorBuilder::default()
                    .with_name(processor_name_2.as_str())
                    .with_type(test_processor::ProcessorMock::get_static_name())
                    .build_arc::<test_processor::ProcessorMock>()?,
                ProcessorBuilder::default()
                    .with_name(processor_name_3.as_str())
                    .with_type(test_processor::ProcessorMock::get_static_name())
                    .build_arc::<test_processor::ProcessorMock>()?,
            ])
            .build()?;
        let mut processor_subjects_map = HashMap::<String, ProcessorSubjects>::new();
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name(&processor_name_1)
            .with_subscriptions(&[
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: table_name_1.to_string(),
                },
                Subscription::AlwaysAllRecordBatches {
                    subject_name: processor_name_1.to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: table_name_1.to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert(processor_name_1, processor_subjects);
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name(&processor_name_2)
            .with_subscriptions(&[
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: table_name_1.to_string(),
                },
                Subscription::AlwaysAllRecordBatches {
                    subject_name: processor_name_2.to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: table_name_1.to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert(processor_name_2, processor_subjects);
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name(&processor_name_3)
            .with_subscriptions(&[
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: table_name_1.to_string(),
                },
                Subscription::AlwaysAllRecordBatches {
                    subject_name: processor_name_3.to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: table_name_1.to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert(processor_name_3, processor_subjects);
        Ok((task, processor_subjects_map))
    }

    pub fn make_test_task_multiple_subscriptions(
        task_name: &str,
        processor_name: &str,
        subject_name: &str,
    ) -> Result<(Task, ProcessorSubjectsMap)> {
        let processor_name_1 = format!("{processor_name}_1");
        let processor_name_2 = format!("{processor_name}_2");
        let processor_name_3 = format!("{processor_name}_3");
        let table_name_1 = format!("{subject_name}_1");
        let table_name_2 = format!("{subject_name}_2");
        let table_name_3 = format!("{subject_name}_3");
        let task = Task::get_builder()
            .with_name(task_name)
            .with_processor(vec![
                ProcessorBuilder::default()
                    .with_name(processor_name_1.as_str())
                    .with_type(test_processor::ProcessorMock::get_static_name())
                    .build_arc::<test_processor::ProcessorMock>()?,
                ProcessorBuilder::default()
                    .with_name(processor_name_2.as_str())
                    .with_type(test_processor::ProcessorMock::get_static_name())
                    .build_arc::<test_processor::ProcessorMock>()?,
                ProcessorBuilder::default()
                    .with_name(processor_name_3.as_str())
                    .with_type(test_processor::ProcessorMock::get_static_name())
                    .build_arc::<test_processor::ProcessorMock>()?,
            ])
            .build()?;
        let mut processor_subjects_map = HashMap::<String, ProcessorSubjects>::new();
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name(&processor_name_1)
            .with_subscriptions(&[
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: table_name_1.to_string(),
                },
                Subscription::AlwaysAllRecordBatches {
                    subject_name: processor_name_1.to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: table_name_1.to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert(processor_name_1, processor_subjects);
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name(&processor_name_2)
            .with_subscriptions(&[
                Subscription::AlwaysAllRecordBatches {
                    subject_name: table_name_1.to_string(),
                },
                Subscription::AlwaysAllRecordBatches {
                    subject_name: processor_name_2.to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: table_name_2.to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert(processor_name_2, processor_subjects);
        let processor_subjects = ProcessorSubjectsBuilder::default()
            .with_name(&processor_name_3)
            .with_subscriptions(&[
                Subscription::OnUpdateAllRecordBatches {
                    subject_name: table_name_1.to_string(),
                },
                Subscription::AlwaysAllRecordBatches {
                    subject_name: processor_name_3.to_string(),
                },
            ])
            .with_publications(&[Publication::Extend {
                subject_name: table_name_3.to_string(),
            }])
            .build()?;
        let _ = processor_subjects_map.insert(processor_name_3, processor_subjects);
        Ok((task, processor_subjects_map))
    }

    pub fn make_test_input_message(
        name: &str,
        publisher: &str,
        subject: &str,
        subject_name: &str,
        update: &Publication,
        test_table: bool,
    ) -> Result<IPCMessageMap> {
        // mock table as input
        let table = if test_table {
            test_subject::make_test_subject(subject_name, 4, 8, 3)?
        } else {
            test_subject::make_test_subject_chat(subject_name)?
        };

        // build the message
        let message = IPCMessageBuilder::new()
            .with_name(name)
            .with_subject(subject)
            .with_publisher(publisher)
            .with_message(table.to_ipc_stream()?)
            .with_update(update)
            .build()?;

        // finish the message map
        let mut map = HashMap::<String, IPCMessage>::new();
        map.insert(message.get_name().to_string(), message);
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use crate::PublicationTrait;

    use super::*;
    use futures::TryStreamExt;
    use phymes_core::{
        BuilderTrait, MessageTrait, Publication, SubjectBuilder, SubjectBuilderTrait,
        SubjectPlanTrait, SubjectTrait,
    };
    use phymes_diagnostics::{Diagnostics, SpanBuilder};

    /// A compilation test to ensure that the `Task::get_name()` method can
    /// be called from a trait object.
    #[allow(dead_code)]
    fn use_task_name_as_trait_object(plan: &dyn TaskTrait<T = TaskBuilder>) {
        let _ = plan.get_name();
    }

    #[tokio::test]
    async fn test_run_task_single_processor() -> Result<()> {
        let span = SpanBuilder::default()
            .with_span("test_run_task_single_processor")
            .build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let (test_task, test_procesor_subjects) = test_task::make_test_task_single_processor(
            "test_task",
            "test_processor",
            "test_table",
        )?;
        let subjects = test_task::make_subjects("test_table", "test_processor")?;
        let runtime_env = test_task::make_runtime_env("rt")?;
        for subject in subjects {
            let _publication: Vec<_> = Publication::Extend {
                subject_name: subject.subject().get_name().to_string(),
            }
            .publish_to_subject(
                &runtime_env,
                subject.subject_own().get_record_batches_own(),
                0,
                "",
                "test_session",
            )?
            .unwrap()
            .try_collect()
            .await?;
        }
        let mut response = test_task.run(
            Some(&diagnostic_builder),
            &test_procesor_subjects,
            &runtime_env,
            "test_session",
        )?;
        assert_eq!(response.len(), 1);
        assert!(response.get("from_test_task_on_test_table").is_some());
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_publisher(),
            "test_task"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table")
                .unwrap()
                .get_subject(),
            "test_table"
        );
        let stream = response.remove("from_test_task_on_test_table").unwrap();
        let partitions =
            SubjectBuilder::new_from_sendable_record_batch_stream(stream.get_message_own())
                .await?
                .with_name("")
                .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 15); // 3 * (4 + 1) from input + 1 added to each batch
        Ok(())
    }

    #[tokio::test]
    async fn test_run_task_chained_processor() -> Result<()> {
        let span = SpanBuilder::default()
            .with_span("test_run_task_chained_processor")
            .build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let (test_task, test_procesor_subjects) = test_task::make_test_task_chained_processor(
            "test_task",
            "test_processor",
            "test_table",
        )?;
        let mut subjects = test_task::make_subjects("test_table_1", "test_processor_1")?;
        subjects.extend(test_task::make_subjects(
            "test_table_2",
            "test_processor_2",
        )?);
        subjects.extend(test_task::make_subjects(
            "test_table_3",
            "test_processor_3",
        )?);
        let runtime_env = test_task::make_runtime_env("rt")?;
        for subject in subjects {
            let _publication: Vec<_> = Publication::Extend {
                subject_name: subject.subject().get_name().to_string(),
            }
            .publish_to_subject(
                &runtime_env,
                subject.subject_own().get_record_batches_own(),
                0,
                "",
                "test_session",
            )?
            .unwrap()
            .try_collect()
            .await?;
        }
        let mut response = test_task.run(
            Some(&diagnostic_builder),
            &test_procesor_subjects,
            &runtime_env,
            "test_session",
        )?;
        assert_eq!(response.len(), 1);
        assert!(response.get("from_test_task_on_test_table_1").is_some());
        assert_eq!(
            response
                .get("from_test_task_on_test_table_1")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table_1"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table_1")
                .unwrap()
                .get_publisher(),
            "test_task"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table_1")
                .unwrap()
                .get_subject(),
            "test_table_1"
        );
        let stream = response.remove("from_test_task_on_test_table_1").unwrap();
        let partitions =
            SubjectBuilder::new_from_sendable_record_batch_stream(stream.get_message_own())
                .await?
                .with_name("")
                .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 21);
        Ok(())
    }

    #[tokio::test]
    async fn test_run_task_multiple_subscriptions() -> Result<()> {
        let span = SpanBuilder::default()
            .with_span("test_run_task_multiple_subscriptions")
            .build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let (test_task, test_procesor_subjects) = test_task::make_test_task_multiple_subscriptions(
            "test_task",
            "test_processor",
            "test_table",
        )?;
        let mut subjects = test_task::make_subjects("test_table_1", "test_processor_1")?;
        subjects.extend(test_task::make_subjects(
            "test_table_2",
            "test_processor_2",
        )?);
        subjects.extend(test_task::make_subjects(
            "test_table_3",
            "test_processor_3",
        )?);
        let runtime_env = test_task::make_runtime_env("rt")?;
        for subject in subjects {
            let _publication: Vec<_> = Publication::Extend {
                subject_name: subject.subject().get_name().to_string(),
            }
            .publish_to_subject(
                &runtime_env,
                subject.subject_own().get_record_batches_own(),
                0,
                "",
                "test_session",
            )?
            .unwrap()
            .try_collect()
            .await?;
        }
        let mut response = test_task.run(
            Some(&diagnostic_builder),
            &test_procesor_subjects,
            &runtime_env,
            "test_session",
        )?;
        assert_eq!(response.len(), 2);
        assert!(response.get("from_test_task_on_test_table_2").is_some());
        assert_eq!(
            response
                .get("from_test_task_on_test_table_2")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table_2"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table_2")
                .unwrap()
                .get_publisher(),
            "test_task"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table_2")
                .unwrap()
                .get_subject(),
            "test_table_2"
        );
        let stream = response.remove("from_test_task_on_test_table_2").unwrap();
        let partitions =
            SubjectBuilder::new_from_sendable_record_batch_stream(stream.get_message_own())
                .await?
                .with_name("")
                .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 18);
        assert!(response.get("from_test_task_on_test_table_3").is_some());
        assert_eq!(
            response
                .get("from_test_task_on_test_table_3")
                .unwrap()
                .get_name(),
            "from_test_task_on_test_table_3"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table_3")
                .unwrap()
                .get_publisher(),
            "test_task"
        );
        assert_eq!(
            response
                .get("from_test_task_on_test_table_3")
                .unwrap()
                .get_subject(),
            "test_table_3"
        );
        let stream = response.remove("from_test_task_on_test_table_3").unwrap();
        let partitions =
            SubjectBuilder::new_from_sendable_record_batch_stream(stream.get_message_own())
                .await?
                .with_name("")
                .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 15);
        Ok(())
    }
}
