use std::sync::Arc;

use anyhow::{Result, anyhow};
use parking_lot::RwLock;
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait, ProcessorPlan, RuntimeEnv, StateMap, Table,
    TablePublication, TableSubscription, Task, TaskBuilderTrait, TaskMap, TaskPlan,
};
use phymes_diagnostics::{HashMap, HashSet};

use crate::SessionContext;
pub trait SessionContextBuilderTrait: BuilderTrait {
    fn with_processors(self, processors: Vec<ProcessorPlan>) -> Self;
    fn with_state(self, state: Vec<Table>) -> Self;
    fn with_runtime_envs(self, runtime_envs: Vec<RuntimeEnv>) -> Self;
    fn with_tasks(self, tasks: Vec<TaskPlan>) -> Self;
    fn with_max_iter(self, max_iter: usize) -> Self;
    fn with_diagnostics(self, diagnostics: bool) -> Self;
    /// Check that the [TaskPlan] and `name are given
    fn check_tasks(&self) -> Result<()>;
    /// Check that all [ProcessorTrait]s defined in the [TaskPlan] are accounted for
    fn check_processors(&self) -> Result<()>;
    /// Check that all [RuntimeEnv]s defined in the [TaskPlan] are accounted for
    fn check_runtime_envs(&self) -> Result<()>;
    /// Check that all subject [Table]s defined in the state and subscribed to by the [ProcessorTrait]s are accounted for
    fn check_state(&self) -> Result<()>;
}

#[derive(Default, PartialEq, Debug)]
pub struct SessionContextBuilder {
    pub name: Option<String>,
    pub processors: Option<Vec<ProcessorPlan>>,
    pub state: Option<Vec<Table>>,
    pub runtime_envs: Option<Vec<RuntimeEnv>>,
    pub tasks: Option<Vec<TaskPlan>>,
    pub max_iter: Option<usize>,
    pub diagnostics: Option<bool>,
}

type SessionContextInput = (
    String,
    TaskMap,
    StateMap,
    HashMap<String, Arc<RuntimeEnv>>,
    usize,
    bool,
);

impl SessionContextBuilder {
    // Get a list of subscriptions and publications for a specific task
    pub fn get_sub_pub_for_task(
        &self,
        task_name: &str,
    ) -> (Vec<&TableSubscription>, Vec<&TablePublication>) {
        // Get the processor name
        let processors = self
            .tasks
            .as_ref()
            .unwrap()
            .iter()
            .filter(|t| t.task_name.as_str() == task_name)
            .flat_map(|t| {
                t.processor_names
                    .iter()
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // Get the subscriptions and subjects of the entry processor
        // and also the other processors that are called by the entry processor
        let mut subscriptons_set = HashSet::new();
        let mut publications_set = HashSet::new();
        self.processors
            .as_ref()
            .unwrap()
            .iter()
            .filter(|p| processors.contains(&p.get_name()))
            .for_each(|p| {
                p.get_subscriptions().iter().for_each(|s| {
                    if s != &TableSubscription::None {
                        subscriptons_set.insert(s);
                    }
                });
                p.get_publications().iter().for_each(|s| {
                    if s != &TablePublication::None {
                        publications_set.insert(s);
                    }
                });
            });
        let subscriptions = subscriptons_set.into_iter().collect::<Vec<_>>();
        let publications = publications_set.into_iter().collect::<Vec<_>>();
        (subscriptions, publications)
    }

    /// Get all of the processors
    pub fn get_processor_names_from_tasks(&self) -> HashSet<String> {
        self.tasks
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|t| {
                t.processor_names
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<HashSet<_>>()
    }

    /// Get all of the subjects
    pub fn get_subject_names_from_processors(&self) -> HashSet<String> {
        self.processors
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|t| {
                let mut subjects = HashSet::new();
                t.get_publications().iter().for_each(|s| {
                    if !s.get_table_name().is_empty() {
                        subjects.insert(s.get_table_name().to_string());
                    }
                });
                t.get_subscriptions().iter().for_each(|s| {
                    if !s.get_table_name().is_empty() {
                        subjects.insert(s.get_table_name().to_string());
                    }
                });
                subjects
            })
            .collect::<HashSet<_>>()
    }

    /// Get all runtime environment names
    pub fn get_runtime_env_names(&self) -> HashSet<String> {
        self.tasks
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| t.runtime_env_name.to_string())
            .collect::<HashSet<_>>()
    }

    /// Get all of the processors for a particular task
    pub fn get_processor_names_for_task(&self, name: &str) -> Vec<String> {
        self.tasks
            .as_ref()
            .unwrap()
            .iter()
            .filter(|t| t.task_name.as_str() == name)
            .flat_map(|t| {
                t.processor_names
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    /// Build the [SessionContext] members
    pub fn build_inner(mut self) -> Result<SessionContextInput> {
        let runtime_env_map = self
            .runtime_envs
            .take()
            .unwrap()
            .into_iter()
            .map(|r| (r.get_name().to_string(), Arc::new(r)))
            .collect::<HashMap<String, Arc<RuntimeEnv>>>();

        let state_map = self
            .state
            .take()
            .unwrap()
            .into_iter()
            .map(|r| (r.get_name().to_string(), Arc::new(RwLock::new(r))))
            .collect::<HashMap<String, Arc<RwLock<Table>>>>();

        let task_map = self
            .tasks
            .as_ref()
            .unwrap()
            .iter()
            .map(|t| {
                let processor_names = self.get_processor_names_for_task(&t.task_name);
                let p = self
                    .processors
                    .as_ref()
                    .unwrap()
                    .iter()
                    .filter_map(|p| {
                        if processor_names.contains(&p.get_name().to_string()) {
                            Some(p.get_processor().clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let task = Task::get_builder()
                    .with_name(&t.task_name)
                    .with_runtime_env(Arc::clone(
                        runtime_env_map.get(t.runtime_env_name.as_str()).unwrap(),
                    ))
                    .with_processor(p)
                    .build()
                    .unwrap();
                (t.task_name.to_owned(), Arc::new(task))
            })
            .collect::<HashMap<_, _>>();

        let name = self.name.unwrap();
        let max_iter = self.max_iter.unwrap_or(25);
        Ok((
            name,
            task_map,
            state_map,
            runtime_env_map,
            max_iter,
            self.diagnostics.unwrap_or_default(),
        ))
    }

    /// Extend a session with another
    pub fn extend(mut self, other: SessionContextBuilder) -> Result<Self> {
        // Extend the state
        let other_state = if let Some(state) = self.state.as_ref() {
            if let Some(other) = other.state {
                let names = state.iter().map(|t| t.get_name()).collect::<HashSet<_>>();
                other.into_iter().filter(|t| !names.contains(t.get_name())).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if let Some(state) = self.state.as_mut() {
            state.extend(other_state);
        } else {
            if !other_state.is_empty() {
                self.state.replace(other_state);
            }
        }

        // Extend the processors
        let other_processors = if let Some(processors) = self.processors.as_ref() {
            if let Some(other) = other.processors {
                let names = processors.iter().map(|t| t.get_name()).collect::<HashSet<_>>();
                other.into_iter().filter(|t| !names.contains(t.get_name())).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if let Some(processors) = self.processors.as_mut() {
            processors.extend(other_processors);
        } else {
            if !other_processors.is_empty() {
                self.processors.replace(other_processors);
            }
        }
        
        // Extend the tasks
        let other_tasks = if let Some(tasks) = self.tasks.as_ref() {
            if let Some(other) = other.tasks {
                let names = tasks.iter().map(|t| &t.task_name).collect::<HashSet<_>>();
                other.into_iter().filter(|t| !names.contains(&t.task_name)).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if let Some(tasks) = self.tasks.as_mut() {
            tasks.extend(other_tasks);
        } else {
            if !other_tasks.is_empty() {
                self.tasks.replace(other_tasks);
            }
        }

        // Extend the runtime_envs
        let other_runtime_envs = if let Some(runtime_envs) = self.runtime_envs.as_ref() {
            if let Some(other) = other.runtime_envs {
                let names = runtime_envs.iter().map(|t| t.get_name()).collect::<HashSet<_>>();
                other.into_iter().filter(|t| !names.contains(t.get_name())).collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if let Some(runtime_envs) = self.runtime_envs.as_mut() {
            runtime_envs.extend(other_runtime_envs);
        } else {
            if !other_runtime_envs.is_empty() {
                self.runtime_envs.replace(other_runtime_envs);
            }
        }

        Ok(self)
    }
}

impl BuilderTrait for SessionContextBuilder {
    type T = SessionContext;
    fn new() -> Self {
        Self {
            name: None,
            processors: None,
            state: None,
            runtime_envs: None,
            tasks: None,
            max_iter: None,
            diagnostics: None,
        }
    }

    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn build(self) -> Result<Self::T> {
        // Check that we can build
        self.check_tasks()?;
        self.check_processors()?;
        self.check_runtime_envs()?;
        self.check_state()?;

        // build the tasks, state, metrics, and runtime objects
        let (name, tasks, state, runtime_envs, max_iter, diagnostics) = self.build_inner()?;

        // ready to build the session
        Ok(Self::T {
            name,
            tasks,
            state,
            runtime_envs,
            max_iter,
            diagnostics,
        })
    }
}

impl SessionContextBuilderTrait for SessionContextBuilder {
    fn with_processors(mut self, processors: Vec<ProcessorPlan>) -> Self {
        self.processors = Some(processors);
        self
    }
    fn with_state(mut self, state: Vec<Table>) -> Self {
        self.state = Some(state);
        self
    }
    fn with_runtime_envs(mut self, runtime_envs: Vec<RuntimeEnv>) -> Self {
        self.runtime_envs = Some(runtime_envs);
        self
    }
    fn with_tasks(mut self, tasks: Vec<TaskPlan>) -> Self {
        self.tasks = Some(tasks);
        self
    }
    fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }
    fn with_diagnostics(mut self, diagnostics: bool) -> Self {
        self.diagnostics = Some(diagnostics);
        self
    }
    fn check_tasks(&self) -> Result<()> {
        if self.name.is_none() {
            return Err(anyhow!(
                "Please give the session a name before attempting to build the session."
            ));
        }

        if self.tasks.is_none() {
            return Err(anyhow!(
                "Please add a plan before attempting to build the session."
            ));
        }
        Ok(())
    }
    fn check_processors(&self) -> Result<()> {
        if self.processors.is_none() {
            return Err(anyhow!(
                "Please add a processor before attempting to build the session."
            ));
        }

        let processor_names = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name().to_owned())
            .collect::<HashSet<_>>();
        let processor_names_task_plan = self.get_processor_names_from_tasks();
        if processor_names_task_plan != processor_names {
            let mut l = processor_names_task_plan.iter().collect::<Vec<_>>();
            l.sort();
            let mut r = processor_names.iter().collect::<Vec<_>>();
            r.sort();
            return Err(anyhow!(
                "Mismatch between provided processors {l:?} and plan processor names {r:?}."
            ));
        }

        Ok(())
    }
    fn check_runtime_envs(&self) -> Result<()> {
        if self.runtime_envs.is_none() {
            return Err(anyhow!(
                "Please add runtime environments before attempting to build the session."
            ));
        }

        let runtime_env_names = self
            .runtime_envs
            .as_ref()
            .unwrap()
            .iter()
            .map(|r| r.get_name().to_string())
            .collect::<HashSet<_>>();
        let runtime_env_names_task_plan = self.get_runtime_env_names();
        if runtime_env_names_task_plan != runtime_env_names {
            let mut l = runtime_env_names_task_plan.iter().collect::<Vec<_>>();
            l.sort();
            let mut r = runtime_env_names.iter().collect::<Vec<_>>();
            r.sort();
            return Err(anyhow!(
                "Mismatch between provided runtime environments {l:?} and plan runtime environment names {r:?}."
            ));
        }

        Ok(())
    }
    fn check_state(&self) -> Result<()> {
        if self.state.is_none() {
            return Err(anyhow!(
                "Please add state before attempting to build the session."
            ));
        }

        let state_names = self
            .state
            .as_ref()
            .unwrap()
            .iter()
            .map(|s| s.get_name().to_string())
            .collect::<HashSet<_>>();
        let state_names_task_plan = self.get_subject_names_from_processors();
        if state_names_task_plan != state_names {
            let mut l = state_names_task_plan.iter().collect::<Vec<_>>();
            l.sort();
            let mut r = state_names.iter().collect::<Vec<_>>();
            r.sort();
            return Err(anyhow!(
                "Mismatch between provided state {l:?} and plan subjects and subscription names {r:?}."
            ));
        }

        Ok(())
    }
}

/// Mock objects and functions for session context builer testing
pub mod test_session_context_builder {
    use phymes_core::{
        AvailableTableSubscribePolicies, ProcessorPlanBuilder,
        test_task::{make_config_tables, make_runtime_env, make_state_tables, make_state_tables_empty},
    };

    use crate::{AvailableProcessors, SessionContextBuilderAgentsTrait};

    use super::*;

    /// Tasks subscribe and publish to state_1, state_2, and state_3
    pub fn make_test_session_context_builder_parallel_tasks() -> Vec<TaskPlan> {
        vec![
            TaskPlan {
                task_name: "task_1".to_string(),
                runtime_env_name: "rt_1".to_string(),
                processor_names: vec!["processor_1".to_string()],
            },
            TaskPlan {
                task_name: "task_2".to_string(),
                runtime_env_name: "rt_1".to_string(),
                processor_names: vec!["processor_2".to_string()],
            },
            TaskPlan {
                task_name: "task_3".to_string(),
                runtime_env_name: "rt_1".to_string(),
                processor_names: vec!["processor_3".to_string()],
            },
            TaskPlan {
                task_name: "session_1".to_string(),
                runtime_env_name: "rt_1".to_string(),
                processor_names: vec!["session_1".to_string()],
            },
        ]
    }

    /// Tasks subscribe and publish to state_1
    pub fn make_test_session_context_builder_sequential_tasks() -> Vec<TaskPlan> {
        vec![
            TaskPlan {
                task_name: "task_1".to_string(),
                runtime_env_name: "rt_1".to_string(),
                processor_names: vec![
                    "processor_1".to_string(),
                    "processor_2".to_string(),
                    "processor_3".to_string(),
                ],
            },
            TaskPlan {
                task_name: "session_1".to_string(),
                runtime_env_name: "rt_1".to_string(),
                processor_names: vec!["session_1".to_string()],
            },
        ]
    }

    /// Tasks and processors subscribe and publish to state_1, state_2, and state_3
    pub fn make_test_session_context_builder_parallel_processors() -> SessionContextBuilder {
        let processor_plans = vec![
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_1"))
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: "processor_1".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_2"))
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_2".to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: "state_2".to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: "processor_2".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_3"))
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_3".to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: "state_3".to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: "processor_3".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorEcho.build_arc("session_1"))
                .with_publications(&[
                    TablePublication::Extend {
                        table_name: "state_1".to_string(),
                    },
                    TablePublication::Extend {
                        table_name: "state_2".to_string(),
                    },
                    TablePublication::Extend {
                        table_name: "state_3".to_string(),
                    },
                ])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: "state_2".to_string(),
                    },
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: "state_3".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
        ];

        // Build the session
        SessionContextBuilder::new()
            .with_tasks(make_test_session_context_builder_parallel_tasks())
            .with_processors(processor_plans)
    }

    /// Tasks and processors subscribe and publish to state_1
    pub fn make_test_session_context_builder_sequential_processors() -> SessionContextBuilder {
        let processor_plans = vec![
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_1"))
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: "processor_1".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_2"))
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: "processor_2".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_3"))
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: "processor_3".to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorEcho.build_arc("session_1"))
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_1".to_string(),
                }])
                .with_subscriptions(&[TableSubscription::OnUpdateLastRecordBatch {
                    table_name: "state_1".to_string(),
                }])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
        ];

        // Build the session
        SessionContextBuilder::new()
            .with_tasks(make_test_session_context_builder_sequential_tasks())
            .with_processors(processor_plans)
    }

    pub fn make_test_session_context_builder_parallel(
        name: &str,
        max_iter: usize,
    ) -> Result<SessionContextBuilder> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables("state_1", "processor_1")?;
        state.extend(make_state_tables("state_2", "processor_2")?);
        state.extend(make_state_tables("state_3", "processor_3")?);

        let builder = make_test_session_context_builder_parallel_processors()
            .with_name(name)
            .with_runtime_envs(runtime_envs)
            .with_state(state)
            .with_max_iter(max_iter)
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?;

        Ok(builder)
    }

    pub fn make_test_session_context_builder_parallel_empty(
        name: &str,
        max_iter: usize,
    ) -> Result<SessionContextBuilder> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables_empty("state_1", "processor_1")?;
        state.extend(make_state_tables_empty("state_2", "processor_2")?);
        state.extend(make_state_tables_empty("state_3", "processor_3")?);

        let builder = make_test_session_context_builder_parallel_processors()
            .with_name(name)
            .with_runtime_envs(runtime_envs)
            .with_state(state)
            .with_max_iter(max_iter)
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?;

        Ok(builder)
    }

    pub fn make_test_session_context_builder_sequential(
        name: &str,
        max_iter: usize,
    ) -> Result<SessionContextBuilder> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables("state_1", "processor_1")?;
        state.push(make_config_tables("processor_2")?);
        state.push(make_config_tables("processor_3")?);


        let builder = make_test_session_context_builder_sequential_processors()
            .with_name(name)
            .with_runtime_envs(runtime_envs)
            .with_state(state)
            .with_max_iter(max_iter)
            .with_diagnostics(true)
            .add_tasks_subscribe_publish()?;

        Ok(builder)
    }
}

#[cfg(test)]
mod tests {
    use crate::AvailableProcessors;

    use super::*;
    use phymes_core::{
        AvailableTableSubscribePolicies, ProcessorPlanBuilder, TableSubscription,
        test_task::{make_runtime_env, make_state_tables},
    };

    #[test]
    fn test_session_context_builder_get_task_sub_pub_with_input() {
        let plan = test_session_context_builder::make_test_session_context_builder_parallel_processors();
        let (subscriptions, publications) = plan.get_sub_pub_for_task("task_1");
        assert!(
            subscriptions.contains(&&TableSubscription::AlwaysFullTable {
                table_name: "processor_1".to_string()
            })
        );
        assert!(
            subscriptions.contains(&&TableSubscription::OnUpdateFullTable {
                table_name: "state_1".to_string()
            })
        );
        assert!(publications.contains(&&TablePublication::Extend {
            table_name: "state_1".to_string()
        }));
    }

    #[test]
    fn test_session_context_builder_get_processor_names() {
        let plan = test_session_context_builder::make_test_session_context_builder_parallel_processors();
        let names = plan.get_processor_names_from_tasks();
        assert!(names.contains("processor_1"));
        assert!(names.contains("processor_2"));
        assert!(names.contains("processor_3"));
        assert!(names.contains("session_1"));
    }

    #[test]
    fn test_session_context_builder_get_subject_names() {
        let plan = test_session_context_builder::make_test_session_context_builder_parallel_processors();
        let names = plan.get_subject_names_from_processors();
        assert!(names.contains("state_1"));
        assert!(names.contains("state_2"));
        assert!(names.contains("state_3"));
        assert!(names.contains("processor_1"));
        assert!(names.contains("processor_2"));
        assert!(names.contains("processor_3"));
    }

    #[test]
    fn test_session_context_builder_get_runtime_env_names() {
        let plan = test_session_context_builder::make_test_session_context_builder_parallel_processors();
        let names = plan.get_runtime_env_names();
        assert!(names.contains("rt_1"));
    }

    #[test]
    fn test_session_context_builder_get_processor_names_for_task() {
        let plan = test_session_context_builder::make_test_session_context_builder_parallel_processors();
        let names = plan.get_processor_names_for_task("task_1");
        assert_eq!(names, vec!["processor_1".to_string()]);
    }

    #[test]
    fn test_session_context_builder_extend_duplicate() -> Result<()> {
        let plan = test_session_context_builder::make_test_session_context_builder_parallel("session_1", 25)?;
        let plan = plan.extend(test_session_context_builder::make_test_session_context_builder_parallel("session_1", 25)?)?;
        assert_eq!(plan, test_session_context_builder::make_test_session_context_builder_parallel("session_1", 25)?);

        Ok(())
    }

    #[test]
    fn test_session_context_builder_extend_other() -> Result<()> {
        let task_plans = vec![
            TaskPlan {
                task_name: "task_4".to_string(),
                runtime_env_name: "rt_4".to_string(),
                processor_names: vec!["processor_4".to_string()],
            },
        ];
        let processor_plans = vec![
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_4"))
                .with_publications(&[TablePublication::Extend {
                    table_name: "state_4".to_string(),
                }])
                .with_subscriptions(&[TableSubscription::OnUpdateLastRecordBatch {
                    table_name: "state_4".to_string(),
                }])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()?,
        ];
        let runtime_envs = vec![make_runtime_env("rt_4")?];
        let mut state = make_state_tables("state_1", "processor_1")?;
        state.extend(make_state_tables("state_2", "processor_2")?);
        state.extend(make_state_tables("state_3", "processor_3")?);
        state.extend(make_state_tables("state_4", "processor_4")?);		
        let other_plan = SessionContextBuilder::new()
            .with_tasks(task_plans)
            .with_processors(processor_plans)
            .with_name("other")
            .with_runtime_envs(runtime_envs)
            .with_state(state)
            .with_max_iter(1)
            .with_diagnostics(false);
        let plan = test_session_context_builder::make_test_session_context_builder_parallel("session_1", 25)?;
        let plan = plan.extend(other_plan)?;
        assert_eq!(plan.name.unwrap(), "session_1");
        assert_eq!(plan.max_iter.unwrap(), 25);
        assert_eq!(plan.diagnostics.unwrap(), true);
        let names = plan.tasks.unwrap().into_iter().map(|t| t.task_name).collect::<Vec<_>>();
        assert_eq!(names, ["task_1", "task_2", "task_3", "session_1", "task_4"]);
        let names = plan.processors.unwrap().into_iter().map(|t| t.get_name().to_string()).collect::<Vec<_>>();
        assert_eq!(names, ["processor_1", "processor_2", "processor_3", "session_1", "processor_4"]);
        let names = plan.runtime_envs.unwrap().into_iter().map(|t| t.get_name().to_string()).collect::<Vec<_>>();
        assert_eq!(names, ["rt_1", "rt_4"]);
        let names = plan.state.unwrap().into_iter().map(|t| t.get_name().to_string()).collect::<Vec<_>>();
        assert_eq!(names, ["processor_1", "state_1", "processor_2", "state_2", "processor_3", "state_3", "processor_4", "state_4"]);

        Ok(())
    }

    #[test]
    fn test_session_context_builder_build_success() -> Result<()> {
        let session =
            test_session_context_builder::make_test_session_context_builder_parallel("session_1", 10)?.build()?;
        assert_eq!(session.get_states().len(), 6);
        assert_eq!(session.get_tasks().len(), 4);
        assert_eq!(session.get_name(), "session_1");
        assert_eq!(session.get_max_iter(), 10);
        assert!(session.get_diagnostics());
        Ok(())
    }

    #[test]
    fn test_session_context_builder_build_fail_missing_name() -> Result<()> {
        let result = SessionContextBuilder::new().build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Please give the session a name before attempting to build the session."
            ),
        }
        Ok(())
    }

    #[test]
    fn test_session_context_builder_build_fail_missing_plan() -> Result<()> {
        let result = SessionContextBuilder::new().with_name("session_1").build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Please add a plan before attempting to build the session."
            ),
        }
        Ok(())
    }

    #[test]
    fn test_session_context_builder_build_fail_missing_processor() -> Result<()> {
        // No tasks
        let result = SessionContextBuilder::new()
            .with_name("session_1")
            .with_tasks(test_session_context_builder::make_test_session_context_builder_parallel_tasks())
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Please add a processor before attempting to build the session."
            ),
        }

        // Missing tasks
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_1"))
                .with_publications(&[])
                .with_subscriptions(&[])
                .with_subscribe_policy(AvailableTableSubscribePolicies::default().build())
                .build()?,
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_2"))
                .with_publications(&[])
                .with_subscriptions(&[])
                .with_subscribe_policy(AvailableTableSubscribePolicies::default().build())
                .build()?,
        ];
        let result = SessionContextBuilder::new()
            .with_name("session_1")
            .with_tasks(test_session_context_builder::make_test_session_context_builder_parallel_tasks())
            .with_processors(processors)
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided processors [\"processor_1\", \"processor_2\", \"processor_3\", \"session_1\"] and plan processor names [\"processor_1\", \"processor_2\"]."
            ),
        }

        // Task not found in plan
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_1"))
                .with_publications(&[])
                .with_subscriptions(&[])
                .with_subscribe_policy(AvailableTableSubscribePolicies::default().build())
                .build()?,
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_2"))
                .with_publications(&[])
                .with_subscriptions(&[])
                .with_subscribe_policy(AvailableTableSubscribePolicies::default().build())
                .build()?,
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("processor_3"))
                .with_publications(&[])
                .with_subscriptions(&[])
                .with_subscribe_policy(AvailableTableSubscribePolicies::default().build())
                .build()?,
            ProcessorPlanBuilder::default()
                .with_processor(AvailableProcessors::ProcessorMock.build_arc("not_found"))
                .with_publications(&[])
                .with_subscriptions(&[])
                .with_subscribe_policy(AvailableTableSubscribePolicies::default().build())
                .build()?,
        ];
        let result = SessionContextBuilder::new()
            .with_name("session_1")
            .with_tasks(test_session_context_builder::make_test_session_context_builder_parallel_tasks())
            .with_processors(processors)
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided processors [\"processor_1\", \"processor_2\", \"processor_3\", \"session_1\"] and plan processor names [\"not_found\", \"processor_1\", \"processor_2\", \"processor_3\"]."
            ),
        }
        Ok(())
    }

    #[test]
    fn test_session_context_builder_build_fail_missing_runtime_env() -> Result<()> {
        // No runtime env
        let result = test_session_context_builder::make_test_session_context_builder_parallel_processors()
            .with_name("session_1")
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Please add runtime environments before attempting to build the session."
            ),
        }

        // Missing runtime env
        let result = test_session_context_builder::make_test_session_context_builder_parallel_processors()
            .with_name("session_1")
            .with_runtime_envs(Vec::new())
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided runtime environments [\"rt_1\"] and plan runtime environment names []."
            ),
        }

        // Runtime env not found in plan
        let result = test_session_context_builder::make_test_session_context_builder_parallel_processors()
            .with_name("session_1")
            .with_runtime_envs(vec![make_runtime_env("not_found")?])
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided runtime environments [\"rt_1\"] and plan runtime environment names [\"not_found\"]."
            ),
        }
        Ok(())
    }

    #[test]
    fn test_session_context_builder_build_fail_missing_state() -> Result<()> {
        // No state
        let result = test_session_context_builder::make_test_session_context_builder_parallel_processors()
            .with_name("session_1")
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Please add state before attempting to build the session."
            ),
        }

        // Missing state
        let result = test_session_context_builder::make_test_session_context_builder_parallel_processors()
            .with_name("session_1")
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(make_state_tables("state_1", "processor_1")?)
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided state [\"processor_1\", \"processor_2\", \"processor_3\", \"state_1\", \"state_2\", \"state_3\"] and plan subjects and subscription names [\"processor_1\", \"state_1\"]."
            ),
        }

        // State not found in plan
        let mut state = make_state_tables("state_1", "processor_1")?;
        state.extend(make_state_tables("state_2", "processor_2")?);
        state.extend(make_state_tables("not_found", "processor_3")?);
        let result = test_session_context_builder::make_test_session_context_builder_parallel_processors()
            .with_name("session_1")
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(state)
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided state [\"processor_1\", \"processor_2\", \"processor_3\", \"state_1\", \"state_2\", \"state_3\"] and plan subjects and subscription names [\"processor_1\", \"processor_2\", \"processor_3\", \"not_found\", \"state_1\", \"state_2\"]."
            ),
        }
        Ok(())
    }
}
