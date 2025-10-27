use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::Schema};
use clap::ValueEnum;
use parking_lot::{Mutex, RwLock};
use phymes_core::{
    AnyTableNameSubscribe, AvailableSubjects, AvailableSubjectsTrait, BuildableTrait, BuilderTrait,
    MappableTrait, ProcessorEcho, ProcessorTrait, RuntimeEnv, RuntimeEnvTrait, SessionContext,
    SessionContextBuilder, SessionContextBuilderTrait, StateMap, SubscribeTrait, Table,
    TableBuilderTrait, TablePublish, TableSubscribe, TableTrait, TaskMap, TaskPlan,
};
use phymes_diagnostics::HashMap;

use crate::{
    AvailableInterfaceSubjects, AvailableProcessors,
    session_traits::tabular::SessionContextBuilderTabularTrait,
};

type SessionContextInput = (
    String,
    TaskMap,
    StateMap,
    HashMap<String, Arc<Mutex<RuntimeEnv>>>,
    usize,
    bool,
    Vec<Table>,
);

/// Trait extension for [SessionContextBuilderTrait] to facilitate building agentic workflows
pub trait SessionContextBuilderAgentsTrait {
    fn build_inner_with_tables(self) -> Result<SessionContextInput>;

    /// Build the [SessionContext] objects along with the [SessionContext] schema tables
    fn build_with_tables(self) -> Result<SessionContext>
    where
        Self: Sized;

    /// Check for consistency between the `lhs_name` and `rhs_name` in any [DataConfig]s and the subscriptions of the [ProcessorTrait]s
    fn check_data_config_subjects(&self) -> Result<()>;

    /// Check that all [ProcessorTrait]s subscribe to a subject of the same name
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
}

impl SessionContextBuilderAgentsTrait for SessionContextBuilder {
    fn build_with_tables(self) -> Result<SessionContext> {
        // Check that we can build
        self.check_tasks()?;
        self.check_processors()?;
        self.check_runtime_envs()?;
        self.check_state()?;
        self.check_processor_config_subjects()?;
        self.check_data_config_subjects()?;

        // build the tasks, state, and runtime objects
        let (name, tasks, mut state, runtime_envs, max_iter, diagnostics, tables) =
            self.build_inner_with_tables()?;

        // update the state with the schema tables
        for table in tables.into_iter() {
            state.insert(table.get_name().to_string(), Arc::new(RwLock::new(table)));
        }

        // ready to build the session
        Ok(SessionContext::new(
            name,
            tasks,
            state,
            runtime_envs,
            max_iter,
            diagnostics,
        ))
    }
    fn build_inner_with_tables(self) -> Result<SessionContextInput> {
        let (tables, _state) = self.to_arrow_tables(false, true, true)?;
        let (name, tasks, state, runtime_envs, max_iter, diagnostics) = self.build_inner()?;
        Ok((
            name,
            tasks,
            state,
            runtime_envs,
            max_iter,
            diagnostics,
            tables,
        ))
    }
    fn check_data_config_subjects(&self) -> Result<()> {
        let state_map = self
            .state
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| (t.get_name().to_string(), t))
            .collect::<HashMap<_, _>>();

        // Find the config subject for each process and check if the subscriptions are in the lhs_name/rhs_name
        let processor_names = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .filter(|p| {
                p.get_subscriptions()
                    .iter()
                    .map(|s| s.get_table_name())
                    .collect::<Vec<_>>()
                    .contains(&p.get_name())
            })
            .filter_map(|p| {
                let mut names = Vec::new();
                let table = state_map.get(p.get_name()).unwrap();
                if let Ok(_index) = table.get_schema().index_of("lhs_name") {
                    let vec_str = table.get_column_as_vec_str("lhs_name");
                    let name = vec_str.last().unwrap();
                    if !name.is_empty() {
                        names.push(name.to_string());
                    }
                }
                if let Ok(_index) = table.get_schema().index_of("rhs_name") {
                    let vec_str = table.get_column_as_vec_str("rhs_name");
                    let name = vec_str.last().unwrap();
                    if !name.is_empty() {
                        names.push(name.to_string());
                    }
                }
                let missing = names
                    .iter()
                    .filter_map(|n| {
                        if p.get_subscriptions()
                            .iter()
                            .map(|s| s.get_table_name())
                            .collect::<Vec<_>>()
                            .contains(&n.as_str())
                        {
                            None
                        } else {
                            Some(n.to_string())
                        }
                    })
                    .collect::<Vec<_>>();
                if missing.is_empty() {
                    None
                } else {
                    Some((p.get_name(), missing))
                }
            })
            .collect::<Vec<_>>();

        if !processor_names.is_empty() {
            return Err(anyhow!(
                "A subscriptions with the same names as the `DataConfig` lhs_name and rhs_name were not found for processors with lhs_name and rhs_name {processor_names:?}."
            ));
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
                        .map(|s| s.get_table_name())
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
        // DM: need to find a way to customize the default further for `DataConfig`
        let name = "";

        // Find the processors that are missing a config
        let processors_to_update = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .filter(|p| {
                self.name.as_ref().unwrap() != p.get_name()
                    && !p
                        .get_subscriptions()
                        .iter()
                        .map(|s| s.get_table_name())
                        .collect::<Vec<_>>()
                        .contains(&p.get_name())
            })
            .cloned()
            .collect::<Vec<_>>();

        // Add the default configuration to the subjects if it does not exist
        let subjects = processors_to_update
            .iter()
            .filter_map(|p| {
                if self.state.is_some()
                    && self
                        .state
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
                    let config = new_processor.to_example_config_json(name).unwrap();
                    let table = Table::get_builder()
                        .with_name(p.get_name())
                        .with_json(&config, 1)
                        .unwrap()
                        .build()
                        .unwrap();
                    Some(table)
                }
            })
            .collect::<Vec<_>>();

        // Remake the state
        if !subjects.is_empty() {
            let mut state = self.state.take().unwrap_or_default();
            state.extend(subjects);
            self.state.replace(state);
        }

        // Add empty tables for all subjects missing in the state with a schema if found
        let subjects = self
            .get_subject_names_from_processors()
            .iter()
            .filter_map(|s| {
                if self.state.is_some()
                    && self
                        .state
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
                    let table = Table::get_builder()
                        .with_schema(schema.clone())
                        .with_record_batches(vec![RecordBatch::new_empty(schema)])
                        .unwrap()
                        .with_name(s)
                        .build()
                        .unwrap();
                    Some(table)
                }
            })
            .collect::<Vec<_>>();

        // Remake the state
        if !subjects.is_empty() {
            let mut state = self.state.take().unwrap_or_default();
            state.extend(subjects);
            self.state.replace(state);
        }

        // Remake the processors (consuming the update)
        let mut processors_to_update = processors_to_update
            .into_iter()
            .map(|p| (p.get_name().to_string(), p))
            .collect::<HashMap<_, _>>();
        let mut processors = Vec::new();
        for processor in self.processors.as_ref().unwrap().iter() {
            if let Some(to_update) = processors_to_update.remove(processor.get_name()) {
                // Rebuild the updated processor
                let mut subscriptions = to_update
                    .get_subscriptions()
                    .into_iter()
                    .map(|e| e.to_owned())
                    .collect::<Vec<TableSubscribe>>();
                subscriptions.push(TableSubscribe::AlwaysFullTable {
                    table_name: to_update.get_name().to_string(),
                });
                let new_processor =
                    AvailableProcessors::from_str(to_update.get_type(), false).unwrap();
                let new_processor = new_processor.build_arc_with_pub_sub(
                    to_update.get_name(),
                    &to_update
                        .get_publications()
                        .into_iter()
                        .map(|e| e.to_owned())
                        .collect::<Vec<TablePublish>>(),
                    &subscriptions,
                    to_update.get_subscribe().clone_boxed(),
                );
                processors.push(new_processor)
            } else {
                // Move over the other processors an preserve order
                processors.push(processor.clone())
            }
        }

        // Remake the processors
        self.processors.replace(processors);

        Ok(self)
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
        if self.state.is_none() {
            return Err(anyhow!(
                "Add state for the session before making the session interface."
            ));
        }

        // Add the session task
        let session_name = self.name.as_ref().unwrap().to_string();
        let runtime_env_name = format!("{session_name}-runtime_env");
        let mut tasks = self.tasks.take().unwrap_or_default();
        tasks.push(TaskPlan {
            task_name: session_name.to_string(),
            runtime_env_name: runtime_env_name.to_owned(),
            processor_names: vec![session_name.to_string()],
        });
        self.tasks.replace(tasks);

        // Add the processors
        // DM: Since we use [ProcessorEcho], we also need to include the subscription in the publications so that it is "echoed" to the session!
        let mut publications = subscriptions
            .map(|s| {
                s.into_iter()
                    .map(|s| TablePublish::Extend {
                        table_name: s.to_string(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut subscriptions = subscriptions
            .map(|s| {
                s.into_iter()
                    .map(|s| TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: s.to_string(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(state) = self.state.as_ref() {
            for table in state.iter() {
                if let Ok(subject) = AvailableInterfaceSubjects::from_str(table.get_name(), false) {
                    // DM: Leave the option for AvailableInterfaceSubjects to be both subscriptions and publications
                    // DM: Use publications/subscription policies that retain the information across sessions
                    if subject.is_session_publication() {
                        publications.push(TablePublish::Extend {
                            table_name: subject.to_string(),
                        });
                    }
                    if subject.is_session_subscription() {
                        subscriptions.push(TableSubscribe::OnUpdateLastRecordBatch {
                            table_name: subject.to_string(),
                        });
                        // DM: Since we use [ProcessorEcho], we also need to include the subscription in the publications so that it is "echoed" to the session!
                        publications.push(TablePublish::Extend {
                            table_name: subject.to_string(),
                        });
                    }
                }
            }
        }
        let mut processors = self.processors.take().unwrap_or_default();
        processors.push(ProcessorEcho::new_arc_with_pub_sub(
            session_name.as_str(),
            &publications,
            &subscriptions,
            AnyTableNameSubscribe::new_box(),
        ));
        self.processors.replace(processors);

        // Add the runtime environment
        let mut runtime_envs = self.runtime_envs.take().unwrap_or_default();
        runtime_envs.push(RuntimeEnv::new().with_name(runtime_env_name.as_str()));
        self.runtime_envs.replace(runtime_envs);

        Ok(self)
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
    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        None
    }
    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        None
    }
    fn make_state_tables(&self) -> Option<Vec<Table>> {
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
        if let Some(runtime_envs) = self.make_runtime_envs() {
            builder = builder.with_runtime_envs(runtime_envs);
        }
        if let Some(state_tables) = self.make_state_tables() {
            builder = builder.with_state(state_tables);
        }
        builder
    }
}

pub mod test_session_context_builder_agents {

    use phymes_core::{
        AllTableNamesSubscribe, BuildableTrait, BuilderTrait, SubscribeTrait, TableBuilderTrait,
        TablePublish, TableSubscribe,
        test_processor::ProcessorMock,
        test_session_context_builder::make_test_session_builder_tasks,
        test_task::{make_runtime_env, make_state_tables},
    };
    use phymes_data::{AvailableCandleOperators, DataConfig};

    use super::*;

    pub fn make_test_session_builder_agents() -> Result<SessionContextBuilder> {
        let processor_plans = vec![
            ProcessorMock::new_arc_with_pub_sub(
                "processor_1",
                &[TablePublish::Extend {
                    table_name: "state_1".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: "processor_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ProcessorMock::new_arc_with_pub_sub(
                "processor_2",
                &[TablePublish::Extend {
                    table_name: "state_2".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: "state_2".to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: "processor_2".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ProcessorMock::new_arc_with_pub_sub(
                "processor_3",
                &[TablePublish::Extend {
                    table_name: "state_3".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: "state_3".to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: "processor_3".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ProcessorMock::new_arc_with_pub_sub(
                "session_1",
                &[TablePublish::Extend {
                    table_name: "state_3".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "state_2".to_string(),
                    },
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "session_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
        ];
        let mut state = make_state_tables("state_1", "processor_1")?;
        state.extend(make_state_tables("state_2", "processor_2")?);
        state.extend(make_state_tables("state_3", "processor_3")?);

        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            operator: AvailableCandleOperators::JoinInner,
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Table::get_builder()
            .with_name("session_1")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        state.push(join_config_state);

        let builder = SessionContextBuilder::new()
            .with_tasks(make_test_session_builder_tasks())
            .with_processors(processor_plans)
            .with_name("session")
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(state)
            .with_diagnostics(true);
        Ok(builder)
    }
}

#[cfg(test)]
mod tests {

    use phymes_core::{
        AllTableNamesSubscribe, BuildableTrait, BuilderTrait, PubSubTrait, SubscribeTrait,
        TableBuilderTrait, TablePublish, TableSubscribe, TaskTrait,
        test_processor::ProcessorMock,
        test_session_context_builder::{
            make_test_session_builder_parallel_task, make_test_session_builder_tasks,
        },
        test_task::{make_runtime_env, make_state_tables},
    };
    use phymes_data::{AvailableCandleOperators, DataConfig};

    use super::*;

    #[test]
    fn test_build_with_tables_success() -> Result<()> {
        let session = test_session_context_builder_agents::make_test_session_builder_agents()?
            .build_with_tables()?;
        assert_eq!(session.get_states().len(), 13);
        assert_eq!(session.get_tasks().len(), 4);
        assert_eq!(session.get_name(), "session");
        assert_eq!(session.get_max_iter(), 25);
        assert!(session.get_diagnostics());
        Ok(())
    }

    #[test]
    fn test_build_with_tables_add_session_interface() -> Result<()> {
        let session = test_session_context_builder_agents::make_test_session_builder_agents()?
            .add_session_interface(Some(&["state_1"]))?
            .build_with_tables()?;
        assert_eq!(session.get_states().len(), 13);
        assert_eq!(session.get_tasks().len(), 5);
        assert_eq!(
            session.get_tasks().get("session").unwrap().get_name(),
            "session"
        );
        let test = session
            .get_tasks()
            .get("session")
            .unwrap()
            .get_processors()
            .iter()
            .map(|p| p.get_name())
            .collect::<Vec<_>>();
        assert_eq!(test, ["session"]);
        let test = session
            .get_tasks()
            .get("session")
            .unwrap()
            .get_runtime_env()
            .lock();
        assert_eq!(test.get_name(), "session-runtime_env");
        let test = session
            .get_tasks()
            .get("session")
            .unwrap()
            .get_subscriptions()
            .iter()
            .map(|p| p.get_table_name())
            .collect::<Vec<_>>();
        assert_eq!(test, ["state_1"]);
        let test = session
            .get_tasks()
            .get("session")
            .unwrap()
            .get_publications()
            .iter()
            .map(|p| p.get_table_name())
            .collect::<Vec<_>>();
        assert_eq!(test, ["state_1"]);
        assert_eq!(session.get_name(), "session");
        assert_eq!(session.get_max_iter(), 25);
        assert!(session.get_diagnostics());
        Ok(())
    }

    #[test]
    fn test_build_with_tables_missing_processor_configs_subjects() -> Result<()> {
        let mut state = make_state_tables("state_1", "config_1")?;
        state.extend(make_state_tables("state_2", "config_2")?);
        state.extend(make_state_tables("state_3", "config_3")?);
        let result = make_test_session_builder_parallel_task()
            .with_name("session")
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(state.clone())
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "A subscription with the same name as the processor (i.e., its config) is not provided for processors [\"processor_1\", \"processor_2\", \"processor_3\", \"session_1\"]."
            ),
        }

        // The default for the config subjects will also fail because of a mismatch in the data config tables
        let session = make_test_session_builder_parallel_task()
            .with_name("session")
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(state)
            .add_processor_subjects()?
            .build_with_tables()?;
        assert_eq!(session.get_states().len(), 16);
        assert_eq!(session.get_tasks().len(), 4);
        assert_eq!(session.get_name(), "session");
        assert_eq!(session.get_max_iter(), 25);
        assert!(!session.get_diagnostics());
        Ok(())
    }

    #[test]
    fn test_build_with_tables_missing_data_config_subjects() -> Result<()> {
        let processor_plans = vec![
            ProcessorMock::new_arc_with_pub_sub(
                "processor_1",
                &[TablePublish::Extend {
                    table_name: "state_1".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: "processor_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ProcessorMock::new_arc_with_pub_sub(
                "processor_2",
                &[TablePublish::Extend {
                    table_name: "state_2".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: "state_2".to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: "processor_2".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ProcessorMock::new_arc_with_pub_sub(
                "processor_3",
                &[TablePublish::Extend {
                    table_name: "state_3".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: "state_3".to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: "processor_3".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ProcessorMock::new_arc_with_pub_sub(
                "session_1",
                &[TablePublish::Extend {
                    table_name: "state_3".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "state_2".to_string(),
                    },
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "session_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
        ];
        let mut state = make_state_tables("state_1", "processor_1")?;
        state.extend(make_state_tables("state_2", "processor_2")?);
        state.extend(make_state_tables("state_3", "processor_3")?);

        let join_config = DataConfig {
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("missing_state".to_string()),
            operator: AvailableCandleOperators::JoinInner,
            ..Default::default()
        };
        let join_config_json = serde_json::to_vec(&join_config).unwrap();
        let join_config_state = Table::get_builder()
            .with_name("session_1")
            .with_json(&join_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        state.push(join_config_state);

        let result = SessionContextBuilder::new()
            .with_tasks(make_test_session_builder_tasks())
            .with_processors(processor_plans)
            .with_name("session")
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(state)
            .with_diagnostics(true)
            .build_with_tables();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "A subscriptions with the same names as the `DataConfig` lhs_name and rhs_name were not found for processors with lhs_name and rhs_name [(\"session_1\", [\"missing_state\"])]."
            ),
        }

        Ok(())
    }
}
