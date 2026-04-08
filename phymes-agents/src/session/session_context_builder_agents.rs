use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Schema, SchemaRef},
};
use clap::ValueEnum;
use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, AvailableSubscribeEvents, BuildableTrait,
    BuilderTrait, IPCMessage, IPCMessageMap, MappableTrait, MessageBuilderTrait, ProcessorPlan,
    ProcessorPlanBuilder, Publication, RuntimeEnv, Subject, SubjectBuilderTrait, SubjectPlan,
    SubjectPlanBuilderTrait, SubjectPlanTrait, SubjectTrait, Subscription, create_bytes_fields,
    create_values_fields,
};
use phymes_data::{
    AvailableOperators, DataConfig, DataConfigTrait, LimitConfig, ObjectStoreConfig, device,
};
#[cfg(feature = "api")]
use phymes_data::{CommandSandboxConfig, HTTPClientConfig};
use phymes_diagnostics::{HashMap, HashSet};
use phymes_ml::{CandleChatConfig, CandleEmbedConfig, ToolCallConfig};

use crate::{
    AvailableInterfaceSubjects, AvailableProcessors, SessionContext, SessionContextBuilder,
    SessionContextBuilderMermaidTrait, SessionContextBuilderTabularTrait,
    SessionContextBuilderTrait, TaskMap, TaskPlan,
    plans::{CountSubjectRowsSession, NextSuperstepSession, NextTaskSession},
};

type SessionContextInput = (
    String,
    TaskMap,
    HashMap<String, SchemaRef>,
    Vec<Subject>,
    Arc<RuntimeEnv>,
    bool,
);

/// Trait extension for [SessionContextBuilderTrait] to facilitate building agentic workflows
pub trait SessionContextBuilderAgentsTrait {
    fn build_inner_with_tables(self) -> Result<SessionContextInput>;

    /// Build the [SessionContext] objects along with the [SessionContext] schema tables
    fn build_with_tables(self) -> Result<(SessionContext, Option<IPCMessageMap>)>
    where
        Self: Sized;

    /// Check the [DataConfig] entries with their linked [ProcessorTrait] subscriptions
    ///
    /// # Notes
    /// 1. Check for consistency between the `lhs_name` and `rhs_name` in any [DataConfig]s and the subscriptions of the [ProcessorTrait]s
    /// 2. Check for consistency between the `lhs_pk`, `rhs_pk`, `lhs_fk`, `rhs_fk`, `lhs_values`, and `rhs_values` in any [DataConfig]s and the subscriptions of the [ProcessorTrait]s
    ///
    /// [ProcessorTrait]: phymes_core::ProcessorTrait
    fn check_data_config_subjects(&self) -> Result<()>;

    /// Check that all processor configs can be built
    ///
    /// # Notes
    /// 1. Check that [DataOperatorTrait]s of [CandleDataProcessor]s can be build with the specified [DataConfig]s
    /// 2. Check that all other configs can be generated from the provided table
    /// 3. Check that all config schemas match their processor
    ///
    /// [DataOperatorTrait]: phymes_data::DataOperatorTrait
    /// [CandleDataProcessor]: phymes_data::CandleDataProcessor
    fn check_processor_config_builds(&self) -> Result<()>;

    /// Check that all [ProcessorTrait]s subscribe to a subject of the same name
    ///
    /// [ProcessorTrait]: phymes_core::ProcessorTrait
    fn check_processor_config_subjects(&self) -> Result<()>;

    /// Add processor subjects to the state with defaults
    ///
    /// # Notes
    /// 1. Add a proessor config subject to the subscription of all processors (if it is not present already)
    /// 2. Add a processor config table to the state with defaults (if it is not present already)
    /// 3. Add a subject table to the state with schema (it is not already present and the schema is known based on the subject name)
    fn add_processor_subjects(self) -> Result<Self>
    where
        Self: Sized;

    /// Add tasks, processors, and runtime environments for a session interface that
    /// subscribes to all [AvailableInterfaceSubjects]
    ///
    /// # Arguments
    /// `subscriptions` - Optional list of subscriptions to listen on in addition to [AvailableInterfaceSubjects]
    fn add_session_interface(self, subscriptions: Option<&[&str]>) -> Result<Self>
    where
        Self: Sized;

    /// Add tasks that automatically update the number of subject rows
    ///
    /// # Notes
    /// * See [CountSubjectRowsSession] for stand alone session and testing
    fn add_subjects_num_rows(self) -> Result<Self>
    where
        Self: Sized;

    /// Add tasks that dynamically compute the next set of tasks that are ready to subscribe to their subjects
    ///
    /// # Notes
    /// * See [NextTaskSession] for stand alone session and testing
    fn add_next_tasks(self) -> Result<Self>
    where
        Self: Sized;

    /// Add tasks that dynamically compute the next superstep
    ///
    /// # Notes
    /// * See [NextSuperstepSession] for stand alone session and testing
    fn add_next_supersteps(self) -> Result<Self>
    where
        Self: Sized;
}

impl SessionContextBuilderAgentsTrait for SessionContextBuilder {
    fn build_with_tables(self) -> Result<(SessionContext, Option<IPCMessageMap>)> {
        // Check that we can build
        self.check_tasks()?;
        self.check_processors()?;
        self.check_runtime_env()?;
        self.check_subjects()?;
        self.check_processor_config_subjects()?;
        self.check_data_config_subjects()?;
        self.check_processor_config_builds()?;

        // build the tasks, state, and runtime objects
        let (name, tasks, schemas, subjects, runtime_env, diagnostics) =
            self.build_inner_with_tables()?;
        let messages: Result<IPCMessageMap> = subjects
            .into_iter()
            .map(|s| {
                let subject_name = s.get_name().to_string();
                let message = IPCMessage::get_builder()
                    .with_publisher(&name)
                    .with_subject(s.get_name())
                    .with_update(&Publication::Extend {
                        subject_name: s.get_name().to_string(),
                    })
                    .with_message(s.to_ipc_stream()?)
                    .make_name()?
                    .build()?;
                Ok((subject_name, message))
            })
            .collect();

        // ready to build the session
        let session_context = SessionContext::new(name, tasks, schemas, runtime_env, diagnostics);
        Ok((session_context, Some(messages?)))
    }

    fn build_inner_with_tables(self) -> Result<SessionContextInput> {
        let tables = self.to_subject_plans(true, true, true, true, true)?;
        let (name, tasks, mut schemas, mut subjects, runtime_envs, diagnostics) =
            self.build_inner()?;

        // Update the schemas and subjects
        for subject in tables {
            let _ = schemas.insert(
                subject.subject().get_name().to_string(),
                subject.subject().get_schema().clone(),
            );
            if subject.subject().count_rows() > 0 {
                subjects.push(subject.subject_own());
            }
        }

        Ok((name, tasks, schemas, subjects, runtime_envs, diagnostics))
    }

    fn check_data_config_subjects(&self) -> Result<()> {
        let state_map = self
            .subjects
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| (t.get_name().to_string(), t))
            .collect::<HashMap<_, _>>();

        // Find the config subject for each process
        let processors = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .filter(|p| {
                p.get_subscriptions()
                    .iter()
                    .map(|s| s.subject_name())
                    .collect::<Vec<_>>()
                    .contains(&p.get_name())
            })
            .collect::<Vec<_>>();

        // Iterate through the LHS and RHS entries
        for processor in processors {
            let subject = state_map.get(processor.get_name()).unwrap();
            let column_names = subject
                .subject()
                .get_schema()
                .fields()
                .iter()
                .map(|f| f.name().to_string())
                .collect::<HashSet<_>>();

            // Check the messages entries
            if column_names.contains("messages") {
                let vec_str = subject.subject().get_column_as_vec_str("messages");
                let name = vec_str.last().unwrap();

                let subscriptions = processor
                    .get_subscriptions()
                    .iter()
                    .filter_map(|s| {
                        if &s.subject_name() == name {
                            Some(name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if subscriptions.is_empty() {
                    return Err(anyhow!(
                        "A subscriptions with the same name as the `ChatConfig` messages was not found for processor {} with messages {name}.",
                        processor.get_name()
                    ));
                }
            }

            // Check the tools entries
            if column_names.contains("tools") {
                let vec_str = subject.subject().get_column_as_vec_str("tools");
                let name = vec_str.last().unwrap();

                let subscriptions = processor
                    .get_subscriptions()
                    .iter()
                    .filter_map(|s| {
                        if &s.subject_name() == name {
                            Some(name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if subscriptions.is_empty() {
                    return Err(anyhow!(
                        "A subscriptions with the same name as the `ChatConfig` tools was not found for processor {} with tools {name}.",
                        processor.get_name()
                    ));
                }
            }

            // Check the documents entries
            if column_names.contains("documents") {
                let vec_str = subject.subject().get_column_as_vec_str("documents");
                let name = vec_str.last().unwrap();

                let subscriptions = processor
                    .get_subscriptions()
                    .iter()
                    .filter_map(|s| {
                        if &s.subject_name() == name {
                            Some(name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if subscriptions.is_empty() {
                    return Err(anyhow!(
                        "A subscriptions with the same name as the `EmbedConfig` documents was not found for processor {} with documents {name}.",
                        processor.get_name()
                    ));
                }
            }

            // Check the subject_name entries
            if column_names.contains("subject_name") {
                let vec_str = subject.subject().get_column_as_vec_str("subject_name");
                let name = vec_str.last().unwrap();

                let subscriptions = processor
                    .get_subscriptions()
                    .iter()
                    .filter_map(|s| {
                        if &s.subject_name() == name {
                            Some(name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if subscriptions.is_empty() {
                    return Err(anyhow!(
                        "A subscriptions with the same name as the `Config` subject_name was not found for processor {} with subject_name {name}.",
                        processor.get_name()
                    ));
                }
            }

            // Check the subject_names entries
            if column_names.contains("subject_names") {
                let mut vec_str = subject
                    .subject()
                    .get_column_as_vec_nested_nonprimitive::<String>("subject_names")?;
                if let Some(names) = vec_str.pop() {
                    let subscriptions = processor
                        .get_subscriptions()
                        .iter()
                        .filter_map(|s| {
                            if names.contains(&s.subject_name().to_string()) {
                                Some(s.subject_name().to_string())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    if subscriptions.is_empty() {
                        return Err(anyhow!(
                            "A subscriptions with the same name as the `ToolCallProcessor` subject_names was not found for processor {} with subject_name {names:?}.",
                            processor.get_name()
                        ));
                    }
                }
            }

            // Check the LHS entries
            if column_names.contains("lhs_name") {
                let vec_str = subject.subject().get_column_as_vec_str("lhs_name");
                let name = vec_str.last().unwrap();

                let subscriptions = processor
                    .get_subscriptions()
                    .iter()
                    .filter_map(|s| {
                        if &s.subject_name() == name {
                            Some(name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if subscriptions.is_empty() {
                    return Err(anyhow!(
                        "A subscriptions with the same name as the `DataConfig` lhs_name was not found for processor {} with lhs_name {name}.",
                        processor.get_name()
                    ));
                }
                let subscription_table = state_map.get(subscriptions.first().unwrap()).unwrap();
                let subscription_col_names = subscription_table
                    .subject()
                    .get_schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().to_string())
                    .collect::<HashSet<_>>();

                if !subscription_col_names.is_empty() {
                    if column_names.contains("lhs_pk") {
                        let vec_str = subject.subject().get_column_as_vec_str("lhs_pk");
                        let pk = vec_str.last().unwrap();
                        if !subscription_col_names.contains(*pk) {
                            return Err(anyhow!(
                                "Subscription {} does not have a column for `DataConfig` lhs_pk {pk} for processor {} with lhs_name {name}.",
                                subscription_table.get_name(),
                                processor.get_name()
                            ));
                        }
                    }
                    if column_names.contains("lhs_fk") {
                        let vec_str = subject.subject().get_column_as_vec_str("lhs_fk");
                        let fk = vec_str.last().unwrap();
                        if !subscription_col_names.contains(*fk) {
                            return Err(anyhow!(
                                "Subscription {} does not have a column for `DataConfig` lhs_fk {fk} for processor {} with lhs_name {name}.",
                                subscription_table.get_name(),
                                processor.get_name()
                            ));
                        }
                    }
                    if column_names.contains("lhs_values") {
                        let vec_str = subject
                            .subject()
                            .get_column_as_vec_nested_nonprimitive::<String>("lhs_values")?;
                        let values = vec_str.last().unwrap();

                        // Check for any renamed or initiated columns
                        let init_col_names = if let Ok(as_columns) = subject
                            .subject()
                            .get_column_as_vec_nested_nonprimitive::<String>("as_columns")
                        {
                            if let Some(as_columns) = as_columns.last() {
                                as_columns
                                    .iter()
                                    .filter_map(|c| {
                                        if c.is_empty() {
                                            None
                                        } else {
                                            Some(c.to_string())
                                        }
                                    })
                                    .collect::<HashSet<_>>()
                            } else {
                                HashSet::new()
                            }
                        } else {
                            HashSet::new()
                        };

                        let mut missing = values
                            .iter()
                            .filter(|v| {
                                !(subscription_col_names.contains(v.as_str())
                                    || init_col_names.contains(v.as_str()))
                            })
                            .collect::<Vec<_>>();
                        missing.sort();
                        if !missing.is_empty() {
                            return Err(anyhow!(
                                "Subscription {} does not have columns for `DataConfig` lhs_values {missing:?} for processor {} with lhs_name {name}.",
                                subscription_table.get_name(),
                                processor.get_name()
                            ));
                        }
                    }
                }
            }

            // Check the RHS entries
            if column_names.contains("rhs_name") {
                let vec_str = subject.subject().get_column_as_vec_str("rhs_name");
                let name = vec_str.last().unwrap();

                let subscriptions = processor
                    .get_subscriptions()
                    .iter()
                    .filter_map(|s| {
                        if &s.subject_name() == name {
                            Some(name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if subscriptions.is_empty() {
                    return Err(anyhow!(
                        "A subscriptions with the same name as the `DataConfig` rhs_name was not found for processor {} with rhs_name {name}.",
                        processor.get_name()
                    ));
                }
                let subscription_table = state_map.get(subscriptions.first().unwrap()).unwrap();
                let subscription_col_names = subscription_table
                    .subject()
                    .get_schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().to_string())
                    .collect::<HashSet<_>>();

                if !subscription_col_names.is_empty() {
                    if column_names.contains("rhs_pk") {
                        let vec_str = subject.subject().get_column_as_vec_str("rhs_pk");
                        let pk = vec_str.last().unwrap();
                        if !subscription_col_names.contains(*pk) {
                            return Err(anyhow!(
                                "Subscription {} does not have a column for `DataConfig` rhs_pk {pk} for processor {} with rhs_name {name}.",
                                subscription_table.get_name(),
                                processor.get_name()
                            ));
                        }
                    }
                    if column_names.contains("rhs_fk") {
                        let vec_str = subject.subject().get_column_as_vec_str("rhs_fk");
                        let fk = vec_str.last().unwrap();
                        if !subscription_col_names.contains(*fk) {
                            return Err(anyhow!(
                                "Subscription {} does not have a column for `DataConfig` rhs_fk {fk} for processor {} with rhs_name {name}.",
                                subscription_table.get_name(),
                                processor.get_name()
                            ));
                        }
                    }
                    if column_names.contains("rhs_values") {
                        let vec_str = subject
                            .subject()
                            .get_column_as_vec_nested_nonprimitive::<String>("rhs_values")?;
                        let values = vec_str.last().unwrap();

                        // Check for any renamed or initiated columns
                        let init_col_names = if let Ok(as_columns) = subject
                            .subject()
                            .get_column_as_vec_nested_nonprimitive::<String>("as_columns")
                        {
                            if let Some(as_columns) = as_columns.last() {
                                as_columns
                                    .iter()
                                    .filter_map(|c| {
                                        if c.is_empty() {
                                            None
                                        } else {
                                            Some(c.to_string())
                                        }
                                    })
                                    .collect::<HashSet<_>>()
                            } else {
                                HashSet::new()
                            }
                        } else {
                            HashSet::new()
                        };

                        let mut missing = values
                            .iter()
                            .filter(|v| {
                                !(subscription_col_names.contains(v.as_str())
                                    || init_col_names.contains(v.as_str()))
                            })
                            .collect::<Vec<_>>();
                        missing.sort();
                        if !missing.is_empty() {
                            return Err(anyhow!(
                                "Subscription {} does not have columns for `DataConfig` rhs_values {missing:?} for processor {} with rhs_name {name}.",
                                subscription_table.get_name(),
                                processor.get_name()
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }
    fn check_processor_config_builds(&self) -> Result<()> {
        let state_map = self
            .subjects
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| (t.get_name().to_string(), t))
            .collect::<HashMap<_, _>>();

        let tasks = self
            .tasks
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| t.task_name.to_string())
            .collect::<HashSet<_>>();

        // Find the config tables whereby a config table is defined as
        //  a subject with the same name as a processor
        let config_tables = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|p| {
                if p.get_subscriptions()
                    .iter()
                    .map(|s| s.subject_name())
                    .collect::<Vec<_>>()
                    .contains(&p.get_name())
                {
                    Some((
                        state_map.get(p.get_name()).unwrap(),
                        p.get_type(),
                        p.get_name(),
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Try to build each config
        let mut data_config_vec = Vec::new();
        for (subject_plan, r#type, name) in config_tables {
            // Check for processors names that are tasks names with empty rows
            if tasks.contains(name) && subject_plan.subject().count_rows() == 0 {
                continue;
            // Check for `values` schema
            } else if subject_plan.subject().get_schema().fields() == &create_values_fields() {
                continue;
            // Check for `bytes` schema
            } else if subject_plan.subject().get_schema().fields() == &create_bytes_fields() {
                continue;
            // Ignore Echo processors
            } else if let Ok(processor) = AvailableProcessors::from_str(r#type, false)
                && processor == AvailableProcessors::ProcessorEcho
            {
                continue;
            }

            // Check guarded configs
            let mut passed_config_checks = false;
            #[cfg(feature = "api")]
            if let Ok(_config) = HTTPClientConfig::from_table(subject_plan.subject()) {
                if let Ok(processor) = AvailableProcessors::from_str(r#type, false) {
                    if processor.config_type() != "HTTPClientConfig" {
                        return Err(anyhow!(
                            "Schema for `HTTPClientConfig` from subject `{}` for processor type `{}` does not match the expected processor type HTTPClientRequestProcessor.",
                            subject_plan.get_name(),
                            r#type
                        ));
                    } else {
                        passed_config_checks = true;
                    }
                } else {
                    return Err(anyhow!(
                        "Processor type `{}` for `HTTPClientConfig` from subject `{}` does not match any of the supported processor types {:?}.",
                        r#type,
                        subject_plan.get_name(),
                        AvailableProcessors::all_varient_names()
                    ));
                }
            } else if let Ok(_config) = CommandSandboxConfig::from_table(subject_plan.subject()) {
                if let Ok(processor) = AvailableProcessors::from_str(r#type, false) {
                    if processor.config_type() != "CommandSandboxConfig" {
                        return Err(anyhow!(
                            "Schema for `CommandSandboxConfig` from subject `{}` for processor type `{}` does not match the expected processor type CommandSandboxProcessor.",
                            subject_plan.get_name(),
                            r#type
                        ));
                    } else {
                        passed_config_checks = true;
                    }
                } else {
                    return Err(anyhow!(
                        "Processor type `{}` for `CommandSandboxConfig` from subject `{}` does not match any of the supported processor types {:?}.",
                        r#type,
                        subject_plan.get_name(),
                        AvailableProcessors::all_varient_names()
                    ));
                }
            }

            // Check everything else
            if let Ok(_config) = CandleChatConfig::from_table(subject_plan.subject()) {
                if let Ok(processor) = AvailableProcessors::from_str(r#type, false) {
                    if processor.config_type() != "CandleChatConfig" {
                        return Err(anyhow!(
                            "Schema for `CandleChatConfig` from subject `{}` for processor type `{}` does not match the expected processor types CandleChatProcessor, MessageParserProcessor, or OpenAIChatProcessor.",
                            subject_plan.get_name(),
                            r#type
                        ));
                    } else {
                        passed_config_checks = true;
                    }
                } else {
                    return Err(anyhow!(
                        "Processor type `{}` for `CandleChatConfig` from subject `{}` does not match any of the supported processor types {:?}.",
                        r#type,
                        subject_plan.get_name(),
                        AvailableProcessors::all_varient_names()
                    ));
                }
            } else if let Ok(_config) = CandleEmbedConfig::from_table(subject_plan.subject()) {
                if let Ok(processor) = AvailableProcessors::from_str(r#type, false) {
                    if processor.config_type() != "CandleEmbedConfig" {
                        return Err(anyhow!(
                            "Schema for `CandleEmbedConfig` from subject `{}` for processor type `{}` does not match the expected processor types CandleEmbedProcessor or OpenAIEmbedProcessor.",
                            subject_plan.get_name(),
                            r#type
                        ));
                    } else {
                        passed_config_checks = true;
                    }
                } else {
                    return Err(anyhow!(
                        "Processor type `{}` for `CandleEmbedConfig` from subject `{}` does not match any of the supported processor types {:?}.",
                        r#type,
                        subject_plan.get_name(),
                        AvailableProcessors::all_varient_names()
                    ));
                }
            } else if let Ok(_config) = ToolCallConfig::from_table(subject_plan.subject()) {
                if let Ok(processor) = AvailableProcessors::from_str(r#type, false) {
                    if processor.config_type() != "ToolCallConfig" {
                        return Err(anyhow!(
                            "Schema for `ToolCallConfig` from subject `{}` for processor type `{}` does not match the expected processor types ToolCallProcessor.",
                            subject_plan.get_name(),
                            r#type
                        ));
                    } else {
                        passed_config_checks = true;
                    }
                } else {
                    return Err(anyhow!(
                        "Processor type `{}` for `ToolCallConfig` from subject `{}` does not match any of the supported processor types {:?}.",
                        r#type,
                        subject_plan.get_name(),
                        AvailableProcessors::all_varient_names()
                    ));
                }
            } else if let Ok(_config) = LimitConfig::from_table(subject_plan.subject()) {
                if let Ok(processor) = AvailableProcessors::from_str(r#type, false) {
                    if processor.config_type() != "LimitConfig" {
                        return Err(anyhow!(
                            "Schema for `LimitConfig` from subject `{}` for processor type `{}` does not match the expected processor type BatchCoalesceProcessor and LimitProcessor.",
                            subject_plan.get_name(),
                            r#type
                        ));
                    } else {
                        passed_config_checks = true;
                    }
                } else {
                    return Err(anyhow!(
                        "Processor type `{}` for `LimitConfig` from subject `{}` does not match any of the supported processor types {:?}.",
                        r#type,
                        subject_plan.get_name(),
                        AvailableProcessors::all_varient_names()
                    ));
                }
            } else if let Ok(_config) = ObjectStoreConfig::from_table(subject_plan.subject()) {
                if let Ok(processor) = AvailableProcessors::from_str(r#type, false) {
                    if processor.config_type() != "ObjectStoreConfig" {
                        return Err(anyhow!(
                            "Schema for `ObjectStoreConfig` from subject `{}` for processor type `{}` does not match the expected processor type ObjectStoreProcessor.",
                            subject_plan.get_name(),
                            r#type
                        ));
                    } else {
                        passed_config_checks = true;
                    }
                } else {
                    return Err(anyhow!(
                        "Processor type `{}` for `ObjectStoreConfig` from subject `{}` does not match any of the supported processor types {:?}.",
                        r#type,
                        subject_plan.get_name(),
                        AvailableProcessors::all_varient_names()
                    ));
                }
            } else if let Ok(config) = DataConfig::from_table(subject_plan.subject()) {
                if let Ok(processor) = AvailableProcessors::from_str(r#type, false) {
                    if processor.config_type() == "DataConfig" {
                        if config.operator.to_string().as_str() != r#type
                            && r#type != AvailableProcessors::ProcessorMock.to_string().as_str()
                            && r#type != AvailableProcessors::ProcessorError.to_string().as_str()
                            && r#type
                                != AvailableProcessors::AggregatorProcessor
                                    .to_string()
                                    .as_str()
                            && r#type
                                != AvailableProcessors::AggregatorProcessor
                                    .to_string()
                                    .as_str()
                            && r#type
                                != AvailableProcessors::CandleDataProcessor
                                    .to_string()
                                    .as_str()
                        {
                            return Err(anyhow!(
                                "Operator `{}` for `DataConfig` from subject `{}` does not match the expected for processor type `{}`.",
                                config.operator,
                                subject_plan.get_name(),
                                r#type
                            ));
                        } else if config.operator.to_string().as_str()
                            != AvailableProcessors::Sort.to_string().as_str()
                            && (r#type
                                == AvailableProcessors::AggregatorProcessor
                                    .to_string()
                                    .as_str()
                                || r#type
                                    == AvailableProcessors::AggregatorProcessor
                                        .to_string()
                                        .as_str())
                        {
                            return Err(anyhow!(
                                "Operator `{}` for `DataConfig` from subject `{}` for processor type `{}` with does not match the expected for processor type of `{}` required by {} and {}.",
                                config.operator,
                                subject_plan.get_name(),
                                r#type,
                                AvailableProcessors::Sort,
                                AvailableProcessors::AggregatorProcessor,
                                AvailableProcessors::AggregatorProcessor
                            ));
                        } else {
                            passed_config_checks = true;
                        }
                    } else {
                        return Err(anyhow!(
                            "Schema for `DataConfig` from subject `{}` for processor type `{}` does not match the expected processor type DataProcessor nor any of the `CandleOperator`s {:?}.",
                            subject_plan.get_name(),
                            r#type,
                            AvailableOperators::all_varient_names()
                        ));
                    }
                } else {
                    return Err(anyhow!(
                        "Processor type `{}` for `DataConfig` from subject `{}` does not match any of the supported processor types {:?}.",
                        r#type,
                        subject_plan.get_name(),
                        AvailableProcessors::all_varient_names()
                    ));
                }
                data_config_vec.push((config, subject_plan.get_name().to_string()));
            }

            // Return an error if the config didn't pass one of the checks
            if !passed_config_checks {
                if let Err(err) = DataConfig::from_table(subject_plan.subject()) {
                    return Err(anyhow!(
                        "Config could not be built for subject `{}` and Error `{err}` when trying to build for DataConfig with table `{subject_plan:?}`.",
                        subject_plan.get_name()
                    ));
                } else {
                    return Err(anyhow!(
                        "Config could not be built for subject `{}` and table `{subject_plan:?}`.",
                        subject_plan.get_name()
                    ));
                }
            }
        }

        // Try to build each DataOperator
        for (config, name) in data_config_vec {
            let data_operator = match config.operator.build(&config) {
                Ok(data_operator) => data_operator,
                Err(err) => {
                    return Err(anyhow!(
                        "Failed to build `{}` with DataConfig from subject `{name}`. {err}",
                        config.operator
                    ));
                }
            };

            // Try to run each operator
            if let Some(lhs_name) = config.lhs_name
                && let Some(lhs_table) = state_map.get(&lhs_name)
                && lhs_table.subject().count_rows() > 0
                && let Some(rhs_name) = config.rhs_name
            {
                let device = device(false)?;
                if let Some(rhs_table) = state_map.get(&rhs_name) {
                    if rhs_table.subject().count_rows() > 0
                        && let Err(err) = data_operator.forward(
                            lhs_table.subject().get_record_batches(),
                            Some(rhs_table.subject().get_record_batches()),
                            &device,
                        )
                    {
                        return Err(anyhow!(
                            "Failed to run `{}` with DataConfig from subject `{name}` and lhs_args {:?} and rhs_args {:?}. {err}",
                            config.operator,
                            lhs_table.subject().get_record_batches(),
                            rhs_table.subject().get_record_batches()
                        ));
                    }
                } else if let Err(err) =
                    data_operator.forward(lhs_table.subject().get_record_batches(), None, &device)
                {
                    return Err(anyhow!(
                        "Failed to run `{}` with DataConfig from subject `{name}` and lhs_args {:?}. {err}",
                        config.operator,
                        lhs_table.subject().get_record_batches()
                    ));
                }
            }
        }

        Ok(())
    }
    fn check_processor_config_subjects(&self) -> Result<()> {
        let processor_names = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|p| {
                if self.name.as_ref().unwrap() == p.get_name()
                    || p.get_subscriptions()
                        .iter()
                        .map(|s| s.subject_name())
                        .collect::<Vec<_>>()
                        .contains(&p.get_name())
                {
                    None
                } else {
                    Some(p.get_name().to_string())
                }
            })
            .collect::<Vec<_>>();
        if !processor_names.is_empty() {
            return Err(anyhow!(
                "A subscription with the same name as the processor (i.e., its config) is not provided for processors {processor_names:?}."
            ));
        }

        Ok(())
    }
    fn add_processor_subjects(mut self) -> Result<Self> {
        if self.processors.is_none() {
            return Err(anyhow!(
                "Add processors before making the default processor configuration subjects."
            ));
        }
        if self.name.is_none() {
            return Err(anyhow!(
                "Add a name for the session before making the default processor configuration subjects."
            ));
        }

        // Find the processors that are missing a config
        let processors_to_update = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|p| {
                if self.name.as_ref().unwrap() != p.get_name()
                    && !p
                        .get_subscriptions()
                        .iter()
                        .map(|s| s.subject_name())
                        .collect::<Vec<_>>()
                        .contains(&p.get_name())
                {
                    Some(p.get_processor().clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Add the default configuration to the subjects if it does not exist
        let subjects = processors_to_update
            .iter()
            .filter_map(|p| {
                if self.subjects.is_some()
                    && self
                        .subjects
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|t| t.get_name())
                        .collect::<Vec<_>>()
                        .contains(&p.get_name())
                {
                    None
                } else {
                    let new_processor = AvailableProcessors::from_str(p.get_type(), false).unwrap();

                    // Make the default config
                    let config = new_processor.to_example_json().unwrap();
                    let subject = Subject::get_builder()
                        .with_name(p.get_name())
                        .with_json(&config, 1)
                        .unwrap()
                        .build()
                        .unwrap();
                    let plan = SubjectPlan::get_builder()
                        .with_subject(subject)
                        .build()
                        .unwrap();

                    // DM: potentially where we could override the defaults to update with the known lhs_name and rhs_name
                    Some(plan)
                }
            })
            .collect::<Vec<_>>();

        // Remake the state
        if !subjects.is_empty() {
            let mut state = self.subjects.take().unwrap_or_default();
            state.extend(subjects);
            self.subjects.replace(state);
        }

        // Add empty tables for all subjects missing in the state with a schema if found
        let subjects = self
            .get_subject_names_from_processors()
            .iter()
            .filter_map(|s| {
                if self.subjects.is_some()
                    && self
                        .subjects
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|t| t.get_name())
                        .collect::<Vec<_>>()
                        .contains(&s.as_str())
                {
                    None
                } else {
                    let schema =
                        if let Ok(new_subject) = AvailableInterfaceSubjects::from_str(s, false) {
                            new_subject.to_schema()
                        } else if let Ok(new_subject) = AvailableSubjects::from_str(s, false) {
                            new_subject.to_schema()
                        } else {
                            Arc::new(Schema::empty())
                        };

                    // Make the default config
                    let subject = Subject::get_builder()
                        .with_schema(schema.clone())
                        .with_record_batches(vec![RecordBatch::new_empty(schema)])
                        .unwrap()
                        .with_name(s)
                        .build()
                        .unwrap();
                    let plan = SubjectPlan::get_builder()
                        .with_subject(subject)
                        .build()
                        .unwrap();
                    Some(plan)
                }
            })
            .collect::<Vec<_>>();

        // Remake the state
        if !subjects.is_empty() {
            let mut state = self.subjects.take().unwrap_or_default();
            state.extend(subjects);
            self.subjects.replace(state);
        }

        // Remake the processors (consuming the update)
        let mut processors_to_update = processors_to_update
            .into_iter()
            .map(|p| (p.get_name().to_string(), p))
            .collect::<HashMap<_, _>>();
        let mut processors = Vec::new();
        for processor in self.processors.take().unwrap().into_iter() {
            if let Some(to_update) = processors_to_update.remove(processor.get_name()) {
                let subscriptions = processor
                    .get_subscriptions()
                    .iter()
                    .chain([&Subscription::AlwaysAllRecordBatches {
                        subject_name: to_update.get_name().to_string(),
                    }])
                    .cloned()
                    .collect::<Vec<_>>();
                let new_processor = ProcessorPlanBuilder::default()
                    .with_processor(processor.get_processor().clone())
                    .with_publications(processor.get_publications())
                    .with_subscriptions(&subscriptions)
                    .with_subscribe_policy(processor.get_subscribe_policy_owned())
                    .build()?;
                processors.push(new_processor)
            } else {
                // Move over the other processors and preserve order
                processors.push(processor)
            }
        }

        // Remake the processors
        Ok(self.with_processors(processors))
    }

    fn add_session_interface(mut self, subscriptions: Option<&[&str]>) -> Result<Self>
    where
        Self: Sized,
    {
        if self.name.is_none() {
            return Err(anyhow!(
                "Add a name for the session before making the session interface."
            ));
        }
        if self.subjects.is_none() {
            return Err(anyhow!(
                "Add state for the session before making the session interface."
            ));
        }

        // Add the session task
        let session_name = self.name.as_ref().unwrap().to_string();
        let mut tasks = self.tasks.take().unwrap_or_default();
        tasks.push(TaskPlan {
            task_name: session_name.to_string(),
            processor_names: vec![session_name.to_string()],
        });
        self.tasks.replace(tasks);

        // Add the processors
        // DM: Since we use [ProcessorEcho], we also need to include the subscription in the publications so that it is "echoed" to the session!
        let mut publications = subscriptions
            .map(|s| {
                s.iter()
                    .map(|s| Publication::Extend {
                        subject_name: s.to_string(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut subscriptions = subscriptions
            .map(|s| {
                s.iter()
                    .map(|s| Subscription::OnUpdateLastRecordBatch {
                        subject_name: s.to_string(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(subjects) = self.subjects.as_ref() {
            for subject_plan in subjects.iter() {
                if let Ok(subject) =
                    AvailableInterfaceSubjects::from_str(subject_plan.get_name(), false)
                {
                    // DM: Leave the option for AvailableInterfaceSubjects to be both subscriptions and publications
                    // DM: Use publications/subscription policies that retain the information across sessions
                    if subject.is_session_publication() {
                        publications.push(Publication::Extend {
                            subject_name: subject.to_string(),
                        });
                    }
                    if subject.is_session_subscription() {
                        subscriptions.push(Subscription::OnUpdateLastRecordBatch {
                            subject_name: subject.to_string(),
                        });
                        // DM: Since we use [ProcessorEcho], we also need to include the subscription in the publications so that it is "echoed" to the session!
                        publications.push(Publication::Extend {
                            subject_name: subject.to_string(),
                        });
                    }
                }
            }
        }
        let mut processors = self.processors.take().unwrap_or_default();
        let processor = AvailableProcessors::ProcessorEcho.build_arc(session_name.as_str());
        let processor_plan = ProcessorPlanBuilder::default()
            .with_processor(processor)
            .with_subscriptions(&subscriptions)
            .with_publications(&publications)
            .with_subscribe_policy(AvailableSubscribeEvents::AnySubjectNameSubscribe.build())
            .build()?;
        processors.push(processor_plan);
        self.processors.replace(processors);

        Ok(self)
    }

    fn add_subjects_num_rows(self) -> Result<Self>
    where
        Self: Sized,
    {
        // Initialize the subjects num rows session
        let subjects_session = CountSubjectRowsSession::default();
        let other_builder = SessionContextBuilder::from_mermaid_flowchart(
            subjects_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(subjects_session.as_mermaid_erdiagram(), false, true)?
        .with_name(subjects_session.session_context_name)
        .add_processor_subjects()?;

        // Extend the current session context builder
        self.extend(other_builder)
    }

    fn add_next_tasks(self) -> Result<Self>
    where
        Self: Sized,
    {
        // Initialize the task subscribe and publish session
        let next_task_session = NextTaskSession::default();
        let other_builder = SessionContextBuilder::from_mermaid_flowchart(
            next_task_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            next_task_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(next_task_session.session_context_name)
        .add_processor_subjects()?;

        // Extend the current session context builder
        self.extend(other_builder)
    }

    fn add_next_supersteps(self) -> Result<Self>
    where
        Self: Sized,
    {
        // Initialize the task subscribe and publish session
        let next_superstep_session = NextSuperstepSession::default();
        let other_builder = SessionContextBuilder::from_mermaid_flowchart(
            next_superstep_session.as_mermaid_flowchart(),
            false,
        )?
        .with_subjects_from_mermaid_erdiagram(
            next_superstep_session.as_mermaid_erdiagram(),
            false,
            true,
        )?
        .with_name(next_superstep_session.session_context_name)
        .add_processor_subjects()?;

        // Extend the current session context builder
        self.extend(other_builder)
    }
}

/// Create custom [SessionContextBuilder]
///
/// # Notes
///
/// * Intended to be used when implementing custom traits or types not yet supported
///   when building from tabular or memermaid.js formats
/// * Users can mix and match custom types with tabular or mermaid.js formats
///   as all return values are optional
/// * Useful for prototyping with static type checking support
pub trait CustomAgentsBuilderTrait {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        None
    }
    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        None
    }
    fn make_runtime_env(&self) -> Option<Arc<RuntimeEnv>> {
        None
    }
    fn make_subjects(&self) -> Option<Vec<SubjectPlan>> {
        None
    }
    fn build(&self) -> SessionContextBuilder {
        let mut builder = SessionContextBuilder::default();
        if let Some(task_plans) = self.make_task_plans() {
            builder = builder.with_tasks(task_plans);
        }
        if let Some(processors) = self.make_processors() {
            builder = builder.with_processors(processors);
        }
        if let Some(runtime_env) = self.make_runtime_env() {
            builder = builder.with_runtime_env(runtime_env);
        }
        if let Some(subjects) = self.make_subjects() {
            builder = builder.with_subjects(subjects);
        }
        builder
    }
}

pub mod test_session_context_builder_agents {

    use std::vec;

    use crate::{test_session_context_builder, test_task};
    use phymes_core::{
        AvailableSubscribeEvents, BuildableTrait, BuilderTrait, Publication, SubjectBuilderTrait,
        Subscription, test_subject,
    };
    use phymes_data::{AvailableOperators, DataConfig, DataJoinOperator};

    use super::*;

    pub fn make_test_state_agents() -> Result<Vec<Subject>> {
        let mut state = vec![
            test_subject::make_test_subject("state_1", 4, 8, 3)?,
            test_subject::make_test_subject("state_2", 4, 8, 3)?,
            test_subject::make_test_subject("state_3", 4, 8, 3)?,
        ];
        let processor_1 = DataConfig {
            lhs_name: Some("state_1".to_string()),
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&processor_1).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_1")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        state.push(join_config_state);
        let processor_2 = DataConfig {
            lhs_name: Some("state_2".to_string()),
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&processor_2).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_2")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        state.push(join_config_state);
        Ok(state)
    }

    pub fn make_test_processors_agents() -> Result<Vec<ProcessorPlan>> {
        let processor_plans = vec![
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_1"))
                .with_publications(&[Publication::Extend {
                    subject_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: "state_1".to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: "processor_1".to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()?,
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_2"))
                .with_publications(&[Publication::Extend {
                    subject_name: "state_2".to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: "state_2".to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: "processor_2".to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()?,
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::Join.build_arc("processor_3"))
                .with_publications(&[Publication::Extend {
                    subject_name: "state_3".to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: "state_1".to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: "state_2".to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: "processor_3".to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()?,
        ];
        Ok(processor_plans)
    }

    #[allow(dead_code)]
    pub fn make_test_session_builder_agents(name: &str) -> Result<SessionContextBuilder> {
        let processor_plans = make_test_processors_agents()?;
        let mut subjects = make_test_state_agents()?;

        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            lhs_fk: Some("id".to_string()),
            rhs_fk: Some("id".to_string()),
            lhs_pk: Some("id".to_string()),
            rhs_pk: Some("id".to_string()),
            operator: AvailableOperators::Join,
            join_operators: Some(DataJoinOperator::default()),
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_3")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        subjects.push(join_config_state);
        let subjects_plan = subjects
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();

        let builder = SessionContextBuilder::new()
            .with_name(name)
            .with_tasks(
                test_session_context_builder::make_test_session_context_builder_parallel_tasks(),
            )
            .with_processors(processor_plans)
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_subjects(subjects_plan)
            .with_diagnostics(true);
        Ok(builder)
    }
}

#[cfg(test)]
mod tests {
    use crate::{TaskTrait, test_session_context_builder, test_task};
    use phymes_core::{BuildableTrait, BuilderTrait, SubjectBuilderTrait};
    use phymes_data::{AvailableOperators, DataConfig, DataJoinOperator, DataStreamManager};

    use super::*;

    #[test]
    fn test_session_context_builder_agents_build_with_tables_success() -> Result<()> {
        let (session, messages) =
            test_session_context_builder_agents::make_test_session_builder_agents("session_1")?
                .build_with_tables()?;
        assert_eq!(session.subjects().len(), 19);
        assert_eq!(session.tasks().len(), 3);
        assert_eq!(session.get_name(), "session_1");
        assert_eq!(session.get_max_steps(), 25);
        assert!(session.get_diagnostics());
        let mut keys = messages
            .unwrap()
            .keys()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            [
                "SessionMermaid",
                "SessionProcessors",
                "SessionRuntimeEnvs",
                "SessionSubjectSchemas",
                "SessionTasks",
                "SessionTasksRunLog",
                "SubjectsChangeLog",
                "SubjectsNumRows",
                "SubjectsObjectStoreMeta",
                "processor_1",
                "processor_2",
                "processor_3",
                "state_1",
                "state_2",
                "state_3"
            ]
        );
        Ok(())
    }

    #[test]
    fn test_session_context_builder_agents_build_with_tables_add_session_interface() -> Result<()> {
        let (session, messages) =
            test_session_context_builder_agents::make_test_session_builder_agents("session_1")?
                .add_session_interface(Some(&["state_1"]))?
                .build_with_tables()?;
        assert_eq!(session.subjects().len(), 19);
        assert_eq!(session.tasks().len(), 4);
        assert_eq!(
            session.tasks().get("session_1").unwrap().get_name(),
            "session_1"
        );
        let test = session
            .tasks()
            .get("session_1")
            .unwrap()
            .get_processors()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, ["session_1"]);
        assert_eq!(session.get_name(), "session_1");
        assert_eq!(session.get_max_steps(), 25);
        assert!(session.get_diagnostics());
        let mut keys = messages
            .unwrap()
            .keys()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            [
                "SessionMermaid",
                "SessionProcessors",
                "SessionRuntimeEnvs",
                "SessionSubjectSchemas",
                "SessionTasks",
                "SessionTasksRunLog",
                "SubjectsChangeLog",
                "SubjectsNumRows",
                "SubjectsObjectStoreMeta",
                "processor_1",
                "processor_2",
                "processor_3",
                "state_1",
                "state_2",
                "state_3"
            ]
        );
        Ok(())
    }

    #[test]
    fn test_session_context_builder_agents_build_with_tables_missing_processor_configs_subjects()
    -> Result<()> {
        // Check that missing config subscriptions can be identified
        let mut subjects = test_session_context_builder_agents::make_test_state_agents()?;
        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            lhs_fk: Some("id".to_string()),
            rhs_fk: Some("id".to_string()),
            lhs_pk: Some("id".to_string()),
            rhs_pk: Some("id".to_string()),
            operator: AvailableOperators::Join,
            join_operators: Some(DataJoinOperator::default()),
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_3")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        subjects.push(join_config_state);
        let subjects_plan = subjects
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();
        let mut task_plans =
            test_session_context_builder::make_test_session_context_builder_parallel_tasks();
        task_plans.push(TaskPlan {
            task_name: "task_4".to_string(),
            processor_names: vec!["processor_4".to_string()],
        });
        let mut processor_plans =
            test_session_context_builder_agents::make_test_processors_agents()?;
        processor_plans.push(
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_4"))
                .with_publications(&[Publication::Extend {
                    subject_name: "state_3".to_string(),
                }])
                .with_subscriptions(&[Subscription::OnUpdateAllRecordBatches {
                    subject_name: "state_3".to_string(),
                }])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()?,
        );
        let result = SessionContextBuilder::new()
            .with_name("session_1")
            .with_tasks(task_plans.clone())
            .with_processors(processor_plans)
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_diagnostics(true)
            .with_subjects(subjects_plan.clone())
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "A subscription with the same name as the processor (i.e., its config) is not provided for processors [\"processor_4\"]."
            ),
        }

        // Check that the default processor subjects fix the issue
        let mut processor_plans =
            test_session_context_builder_agents::make_test_processors_agents()?;
        processor_plans.push(
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_4"))
                .with_publications(&[Publication::Extend {
                    subject_name: "state_3".to_string(),
                }])
                .with_subscriptions(&[Subscription::OnUpdateAllRecordBatches {
                    subject_name: "state_3".to_string(),
                }])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()?,
        );
        let (session, messages) = SessionContextBuilder::new()
            .with_name("session_1")
            .with_tasks(task_plans)
            .with_processors(processor_plans)
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_diagnostics(true)
            .with_subjects(subjects_plan.clone())
            .add_processor_subjects()?
            .build_with_tables()?;
        assert_eq!(session.subjects().len(), 20);
        assert_eq!(session.tasks().len(), 4);
        assert_eq!(session.get_name(), "session_1");
        assert_eq!(session.get_max_steps(), 25);
        let mut keys = messages
            .unwrap()
            .keys()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            [
                "SessionMermaid",
                "SessionProcessors",
                "SessionRuntimeEnvs",
                "SessionSubjectSchemas",
                "SessionTasks",
                "SessionTasksRunLog",
                "SubjectsChangeLog",
                "SubjectsNumRows",
                "SubjectsObjectStoreMeta",
                "processor_1",
                "processor_2",
                "processor_3",
                "processor_4",
                "state_1",
                "state_2",
                "state_3"
            ]
        );
        Ok(())
    }

    #[test]
    fn test_session_context_builder_agents_build_with_tables_missing_data_config_subjects()
    -> Result<()> {
        let subjects = test_session_context_builder_agents::make_test_state_agents()?;

        // Test that mismatches in the lhs/rhs name are identified
        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("missing_state".to_string()),
            operator: AvailableOperators::Join,
            lhs_stream: DataStreamManager::Accumulate,
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_3")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let mut subjects_test = subjects.clone();
        subjects_test.push(join_config_state);
        let subjects_plans = subjects_test
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();

        let result = SessionContextBuilder::new()
            .with_tasks(
                test_session_context_builder::make_test_session_context_builder_parallel_tasks(),
            )
            .with_processors(test_session_context_builder_agents::make_test_processors_agents()?)
            .with_name("session_1")
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_subjects(subjects_plans)
            .with_diagnostics(true)
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "A subscriptions with the same name as the `DataConfig` rhs_name was not found for processor processor_3 with rhs_name missing_state."
            ),
        }

        // Test that mismatches in the lhs/rhs pk, fk are identified
        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            lhs_fk: Some("id".to_string()),
            rhs_fk: Some("id".to_string()),
            lhs_pk: Some("title".to_string()),
            rhs_pk: Some("missing_pk".to_string()),
            operator: AvailableOperators::Join,
            lhs_stream: DataStreamManager::Accumulate,
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_3")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let mut subjects_test = subjects.clone();
        subjects_test.push(join_config_state);
        let subjects_plans = subjects_test
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();

        let result = SessionContextBuilder::new()
            .with_tasks(
                test_session_context_builder::make_test_session_context_builder_parallel_tasks(),
            )
            .with_processors(test_session_context_builder_agents::make_test_processors_agents()?)
            .with_name("session_1")
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_subjects(subjects_plans)
            .with_diagnostics(true)
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Subscription state_2 does not have a column for `DataConfig` rhs_pk missing_pk for processor processor_3 with rhs_name state_2."
            ),
        }

        // Test that mismatches in the lhs/rhs pk, fk, and values are identified
        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            lhs_fk: Some("id".to_string()),
            rhs_fk: Some("id".to_string()),
            lhs_pk: Some("title".to_string()),
            rhs_pk: Some("title".to_string()),
            lhs_values: Some(vec!["metadata".to_string(), "missing_value".to_string()]),
            rhs_values: Some(vec!["metadata".to_string(), "score".to_string()]),
            operator: AvailableOperators::Join,
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_3")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let mut subjects_test = subjects.clone();
        subjects_test.push(join_config_state);
        let subjects_plans = subjects_test
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();

        let result = SessionContextBuilder::new()
            .with_tasks(
                test_session_context_builder::make_test_session_context_builder_parallel_tasks(),
            )
            .with_processors(test_session_context_builder_agents::make_test_processors_agents()?)
            .with_name("session_1")
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_subjects(subjects_plans)
            .with_diagnostics(true)
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Subscription state_1 does not have columns for `DataConfig` lhs_values [\"missing_value\"] for processor processor_3 with lhs_name state_1."
            ),
        }

        Ok(())
    }

    #[test]
    fn test_session_context_builder_agents_build_with_tables_failing_processor_config_builds()
    -> Result<()> {
        let subjects = test_session_context_builder_agents::make_test_state_agents()?;

        // Test for mismatch between processor and config types
        let join_config = LimitConfig {
            fetch: 0,
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_3")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let mut subjects_test = subjects.clone();
        subjects_test.push(join_config_state);
        let subjects_plans = subjects_test
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();

        let result = SessionContextBuilder::new()
            .with_tasks(
                test_session_context_builder::make_test_session_context_builder_parallel_tasks(),
            )
            .with_processors(test_session_context_builder_agents::make_test_processors_agents()?)
            .with_name("session_1")
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_subjects(subjects_plans)
            .with_diagnostics(true)
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Schema for `LimitConfig` from subject `processor_3` for processor type `Join` does not match the expected processor type BatchCoalesceProcessor and LimitProcessor."
            ),
        }

        // Test for a mismatch between operators
        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            lhs_values: Some(vec!["id".to_string()]),
            cpu: false,
            operator: AvailableOperators::NormalizeTime,
            lhs_stream: DataStreamManager::Accumulate,
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_3")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let mut subjects_test = subjects.clone();
        subjects_test.push(join_config_state);
        let subjects_plans = subjects_test
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();

        let result = SessionContextBuilder::new()
            .with_tasks(
                test_session_context_builder::make_test_session_context_builder_parallel_tasks(),
            )
            .with_processors(test_session_context_builder_agents::make_test_processors_agents()?)
            .with_name("session_1")
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_subjects(subjects_plans)
            .with_diagnostics(true)
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Operator `NormalizeTime` for `DataConfig` from subject `processor_3` does not match the expected for processor type `Join`."
            ),
        }

        // Test for missing required operator members
        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            lhs_fk: Some("id".to_string()),
            rhs_fk: Some("id".to_string()),
            operator: AvailableOperators::Join,
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Subject::get_builder()
            .with_name("processor_3")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let mut subjects_test = subjects.clone();
        subjects_test.push(join_config_state);
        let subjects_plans = subjects_test
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();

        let result = SessionContextBuilder::new()
            .with_tasks(
                test_session_context_builder::make_test_session_context_builder_parallel_tasks(),
            )
            .with_processors(test_session_context_builder_agents::make_test_processors_agents()?)
            .with_name("session_1")
            .with_runtime_env(test_task::make_runtime_env("rt_1")?)
            .with_subjects(subjects_plans)
            .with_diagnostics(true)
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Failed to build `Join` with DataConfig from subject `processor_3`. Missing `lhs_pk` for `Join`."
            ),
        }

        Ok(())
    }
}
