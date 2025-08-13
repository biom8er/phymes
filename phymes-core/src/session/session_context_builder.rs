use std::sync::Arc;

use anyhow::{Result, anyhow};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::{
    metrics::{ArrowTaskMetricsSet, HashMap, HashSet},
    table::{
        arrow_table::ArrowTable, arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::ArrowTableSubscribe,
    },
    task::{
        arrow_processor::ArrowProcessorTrait,
        arrow_task::{ArrowTask, ArrowTaskBuilderTrait},
    },
};

use super::{
    common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
    runtime_env::RuntimeEnv,
    session_context::SessionContext,
};

/// The plan for the tasks
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TaskPlan {
    /// The name of the task
    pub task_name: String,
    /// The entry processor name
    pub runtime_env_name: String,
    /// The name of processors to pre/post-process the messages stream
    pub processor_names: Vec<String>,
    // DM: to enable queue groups and automatic scaling
    //   The name of the queue group the task belongs to
    //   subscriptions are randomly assigned to a task in the queue group
    //   pub queue_gorup_name
}
pub trait SessionContextBuilderTrait: BuilderTrait {
    fn with_processors(self, processors: Vec<Arc<dyn ArrowProcessorTrait>>) -> Self;
    fn with_state(self, state: Vec<ArrowTable>) -> Self;
    fn with_metrics(self, metrics: ArrowTaskMetricsSet) -> Self;
    fn with_runtime_envs(self, runtime_envs: Vec<RuntimeEnv>) -> Self;
    fn with_tasks(self, tasks: Vec<TaskPlan>) -> Self;
    fn with_max_iter(self, max_iter: usize) -> Self;
}

#[derive(Default)]
pub struct SessionContextBuilder {
    pub name: Option<String>,
    pub processors: Option<Vec<Arc<dyn ArrowProcessorTrait>>>,
    pub state: Option<Vec<ArrowTable>>,
    pub metrics: Option<ArrowTaskMetricsSet>,
    pub runtime_envs: Option<Vec<RuntimeEnv>>,
    pub tasks: Option<Vec<TaskPlan>>,
    pub max_iter: Option<usize>,
}

impl SessionContextBuilder {
    // Get a list of subscriptions and publications for a specific task
    pub fn get_task_sub_pub(
        &self,
        task_name: &str,
    ) -> (Vec<&ArrowTableSubscribe>, Vec<&ArrowTablePublish>) {
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
                    if **s != ArrowTableSubscribe::None {
                        subscriptons_set.insert(s.to_owned());
                    }
                });
                p.get_publications().iter().for_each(|s| {
                    if **s != ArrowTablePublish::None {
                        publications_set.insert(s.to_owned());
                    }
                });
            });
        let subscriptions = subscriptons_set.into_iter().collect::<Vec<_>>();
        let publications = publications_set.into_iter().collect::<Vec<_>>();
        (subscriptions, publications)
    }

    /// Get all of the processors
    pub fn get_processor_names(&self) -> HashSet<String> {
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
    pub fn get_subject_names(&self) -> HashSet<String> {
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

    /// Make a mermaid.js flowchart of the session
    pub fn make_mermaid_flowchart(&self) -> Result<String> {
        // Check if there are members
        if self.tasks.is_none() {
            return Err(anyhow!("Add task plans before making the mermaid flowchart."));
        }
        if self.processors.is_none() {
            return Err(anyhow!("Add processors before making the mermaid flowchart."));
        }
        if self.runtime_envs.is_none() {
            return Err(anyhow!("Add runtime environments before making the mermaid flowchart."));
        }
        if self.state.is_none() {
            return Err(anyhow!("Add state subjects before making the mermaid flowchart."));
        }        

        // Entities with expanded shape/label attributes that will be appended to flowchart
        let mut processors_vec = Vec::new();
        for processor_name in self.get_processor_names() {
            processors_vec.push(format!("\t{processor_name}-processor@{{ shape: rect, label: {processor_name} }}"));
        }

        let mut subjects_vec = Vec::new();
        for subject_name in self.get_subject_names() {
            subjects_vec.push(format!("\t{subject_name}-subject@{{ shape: doc, label: {subject_name} }}"));
        }

        let mut runtime_envs_vec = Vec::new();
        for runtime_env_name in self.get_runtime_env_names() {
            runtime_envs_vec.push(format!("\t{runtime_env_name}-rt@{{ shape: subproc, label: {runtime_env_name} }}"));
        }

        // Subgraphs
        let mut tasks_vec = Vec::new();
        let mut subscriptions_vec = Vec::new();
        let mut publications_vec = Vec::new();
        let mut runtime_envs_to_tasks_vec = Vec::new();
        for task in self.tasks.as_ref().unwrap().iter() {
            tasks_vec.push(format!("\tsubgraph {}", task.task_name));
            runtime_envs_to_tasks_vec.push(format!("\t{}-rt --> {}", task.runtime_env_name, task.task_name));

            // Iterate through each processor
            for processor_name in task.processor_names.iter() {
                for processor in self.processors.as_ref().unwrap().iter() {
                    if processor_name == processor.get_name() {

                        // Subscriptions
                        let subscribe = processor.get_subscribe().get_name();
                        subscriptions_vec.push(format!("\t{processor_name}-subscribe@{{ shape: diamond, label: {subscribe} }}"));
                        for subscription in processor.get_subscriptions().iter() {
                            if subscription.is_update() {
                                tasks_vec.push(format!("\t\t{}-subject -. {} .-> {processor_name}-subscribe", subscription.get_table_name(), subscription.get_short_name()));
                            } else {
                                tasks_vec.push(format!("\t\t{}-subject -- {} --> {processor_name}-subscribe", subscription.get_table_name(), subscription.get_short_name()));
                            }                            
                        }
                        tasks_vec.push(format!("\t\t{processor_name}-subscribe --> {processor_name}-processor"));

                        // Publications
                        publications_vec.push(format!("\t{processor_name}-publish@{{ shape: fork }}"));
                        tasks_vec.push(format!("\t\t{processor_name}-processor --> {processor_name}-publish"));
                        for publication in processor.get_publications().iter() {
                            tasks_vec.push(format!("\t\t{processor_name}-publish -- {} --> {}-subject", publication.get_short_name(), publication.get_table_name()));
                        }

                        break;
                    }
                }                
            }
            tasks_vec.push(format!("\tend"));
        }

        // Create the final mermaid.js flowchart script
        let mut mermaid_js = vec!["flowchart TD".to_string()];
        mermaid_js.extend(tasks_vec);
        mermaid_js.extend(runtime_envs_to_tasks_vec);
        mermaid_js.extend(processors_vec);
        mermaid_js.extend(runtime_envs_vec);
        mermaid_js.extend(subjects_vec);
        mermaid_js.extend(publications_vec);
        mermaid_js.extend(subscriptions_vec);
        Ok(mermaid_js.join("\n"))
    }
}

impl BuilderTrait for SessionContextBuilder {
    type T = SessionContext;
    fn new() -> Self {
        Self {
            name: None,
            processors: None,
            state: None,
            metrics: None,
            runtime_envs: None,
            tasks: None,
            max_iter: None,
        }
    }
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    fn build(mut self) -> Result<Self::T> {
        if self.tasks.is_none() {
            return Err(anyhow!(
                "Please add a plan before attempting to build the session."
            ));
        }

        if self.processors.is_none() {
            return Err(anyhow!(
                "Please add a processor before attempting to build the session."
            ));
        }

        // Check that all processors are accounted for
        let processor_names = self
            .processors
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| p.get_name().to_owned())
            .collect::<HashSet<_>>();
        let processor_names_task_plan = self.get_processor_names();
        if processor_names_task_plan != processor_names {
            let mut l = processor_names_task_plan.iter().collect::<Vec<_>>();
            l.sort();
            let mut r = processor_names.iter().collect::<Vec<_>>();
            r.sort();
            return Err(anyhow!(
                "Mismatch between provided processors {:?} and plan processor names {:?}.",
                l,
                r
            ));
        }

        // Check that the runtime_env names are accounted for...
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
                "Mismatch between provided runtime environments {:?} and plan runtime environment names {:?}.",
                l,
                r
            ));
        }

        // ...then build
        let runtime_env_map = self
            .runtime_envs
            .take()
            .unwrap()
            .into_iter()
            .map(|r| (r.get_name().to_string(), Arc::new(Mutex::new(r))))
            .collect::<HashMap<String, Arc<Mutex<RuntimeEnv>>>>();

        // Check that the state names are accounted for...
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
        let state_names_task_plan = self.get_subject_names();
        if state_names_task_plan != state_names {
            let mut l = state_names_task_plan.iter().collect::<Vec<_>>();
            l.sort();
            let mut r = state_names.iter().collect::<Vec<_>>();
            r.sort();
            return Err(anyhow!(
                "Mismatch between provided state {:?} and plan subjects and subscription names {:?}.",
                l,
                r
            ));
        }

        // ...then build
        let state_map = self
            .state
            .take()
            .unwrap()
            .into_iter()
            .map(|r| (r.get_name().to_string(), Arc::new(RwLock::new(r))))
            .collect::<HashMap<String, Arc<RwLock<ArrowTable>>>>();

        // Check for metrics; if none, initialize with defaults
        let metrics = match self.metrics {
            Some(ref metrics) => metrics.clone(),
            None => ArrowTaskMetricsSet::new(),
        };

        // Build the tasks
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
                    .filter(|p| processor_names.contains(&p.get_name().to_string()))
                    .map(Arc::clone)
                    .collect::<Vec<_>>();
                let task = ArrowTask::get_builder()
                    .with_name(&t.task_name)
                    .with_metrics(metrics.clone())
                    .with_runtime_env(Arc::clone(
                        runtime_env_map.get(t.runtime_env_name.as_str()).unwrap(),
                    ))
                    .with_processor(p)
                    .build()
                    .unwrap();
                (t.task_name.to_owned(), Arc::new(task))
            })
            .collect::<HashMap<_, _>>();

        // ready to build the session
        Ok(Self::T {
            name: self.name.unwrap_or_default(),
            tasks: task_map,
            state: state_map,
            metrics,
            runtime_envs: runtime_env_map,
            max_iter: self.max_iter.unwrap_or(25),
        })
    }
}

impl SessionContextBuilderTrait for SessionContextBuilder {
    fn with_processors(mut self, processors: Vec<Arc<dyn ArrowProcessorTrait>>) -> Self {
        self.processors = Some(processors);
        self
    }
    fn with_state(mut self, state: Vec<ArrowTable>) -> Self {
        self.state = Some(state);
        self
    }
    fn with_metrics(mut self, metrics: ArrowTaskMetricsSet) -> Self {
        self.metrics = Some(metrics);
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
}

/// Mock objects and functions for session context builer testing
pub mod test_session_context_builder {
    use crate::{
        table::arrow_table_subscribe::{AllTableNamesSubscribe, SubscribeTrait},
        task::{
            arrow_processor::{ArrowProcessorEcho, test_processor::ArrowProcessorMock},
            arrow_task::test_task::{make_runtime_env, make_state_tables, make_state_tables_empty},
        },
    };

    use super::*;

    pub fn make_test_session_builder_tasks() -> Vec<TaskPlan> {
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

    /// Tasks subscribe and publish to state_1, state_2, and state_3
    pub fn make_test_session_builder_parallel_task() -> SessionContextBuilder {
        let processor_plans = vec![
            ArrowProcessorMock::new_arc_with_pub_sub(
                "processor_1",
                &[ArrowTablePublish::Extend {
                    table_name: "state_1".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: "config_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ArrowProcessorMock::new_arc_with_pub_sub(
                "processor_2",
                &[ArrowTablePublish::Extend {
                    table_name: "state_2".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: "state_2".to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: "config_2".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ArrowProcessorMock::new_arc_with_pub_sub(
                "processor_3",
                &[ArrowTablePublish::Extend {
                    table_name: "state_3".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: "state_3".to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: "config_3".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ArrowProcessorMock::new_arc_with_pub_sub(
                "session_1",
                &[
                    ArrowTablePublish::Extend {
                        table_name: "state_1".to_string(),
                    },
                    ArrowTablePublish::Extend {
                        table_name: "state_2".to_string(),
                    },
                    ArrowTablePublish::Extend {
                        table_name: "state_3".to_string(),
                    },
                ],
                &[
                    ArrowTableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "state_1".to_string(),
                    },
                    ArrowTableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "state_2".to_string(),
                    },
                    ArrowTableSubscribe::OnUpdateLastRecordBatch {
                        table_name: "state_3".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
        ];

        // Build the session
        SessionContextBuilder::new()
            .with_tasks(make_test_session_builder_tasks())
            .with_processors(processor_plans)
    }

    /// Tasks subscribe and publish to state_1
    pub fn make_test_session_builder_sequential_task() -> SessionContextBuilder {
        let processor_plans = vec![
            ArrowProcessorMock::new_arc_with_pub_sub(
                "processor_1",
                &[ArrowTablePublish::Extend {
                    table_name: "state_1".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: "config_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ArrowProcessorMock::new_arc_with_pub_sub(
                "processor_2",
                &[ArrowTablePublish::Extend {
                    table_name: "state_1".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: "config_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ArrowProcessorMock::new_arc_with_pub_sub(
                "processor_3",
                &[ArrowTablePublish::Extend {
                    table_name: "state_1".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: "state_1".to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: "config_1".to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ),
            ArrowProcessorMock::new_arc_with_pub_sub(
                "session_1",
                &[ArrowTablePublish::Extend {
                    table_name: "state_1".to_string(),
                }],
                &[ArrowTableSubscribe::OnUpdateLastRecordBatch {
                    table_name: "state_1".to_string(),
                }],
                AllTableNamesSubscribe::new_box(),
            ),
        ];

        // Build the session
        SessionContextBuilder::new()
            .with_tasks(make_test_session_builder_tasks())
            .with_processors(processor_plans)
    }

    pub fn make_test_processors() -> Vec<Arc<dyn ArrowProcessorTrait>> {
        vec![
            ArrowProcessorMock::new_arc("processor_1"),
            ArrowProcessorMock::new_arc("processor_2"),
            ArrowProcessorMock::new_arc("processor_3"),
            ArrowProcessorEcho::new_arc("session_1"),
        ]
    }

    pub fn make_test_session_context_parallel_task(
        name: &str,
        metrics: ArrowTaskMetricsSet,
        max_iter: usize,
    ) -> Result<SessionContext> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables("state_1", "config_1")?;
        state.extend(make_state_tables("state_2", "config_2")?);
        state.extend(make_state_tables("state_3", "config_3")?);

        make_test_session_builder_parallel_task()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_envs(runtime_envs)
            .with_state(state)
            .with_max_iter(max_iter)
            .build()
    }

    pub fn make_test_session_context_parallel_task_empty(
        name: &str,
        metrics: ArrowTaskMetricsSet,
        max_iter: usize,
    ) -> Result<SessionContext> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables_empty("state_1", "config_1")?;
        state.extend(make_state_tables_empty("state_2", "config_2")?);
        state.extend(make_state_tables_empty("state_3", "config_3")?);

        make_test_session_builder_parallel_task()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_envs(runtime_envs)
            .with_state(state)
            .with_max_iter(max_iter)
            .build()
    }

    pub fn make_test_session_context_sequential_task(
        name: &str,
        metrics: ArrowTaskMetricsSet,
        max_iter: usize,
    ) -> Result<SessionContext> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let state = make_state_tables("state_1", "config_1")?;

        make_test_session_builder_sequential_task()
            .with_name(name)
            .with_metrics(metrics)
            .with_runtime_envs(runtime_envs)
            .with_state(state)
            .with_max_iter(max_iter)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        table::arrow_table_subscribe::ArrowTableSubscribe,
        task::{
            arrow_processor::{ArrowProcessorTrait, test_processor::ArrowProcessorMock},
            arrow_task::test_task::{make_runtime_env, make_state_tables},
        },
    };

    #[test]
    fn test_get_task_sub_pub_with_input() {
        let plan = test_session_context_builder::make_test_session_builder_parallel_task();
        let (subscriptions, publications) = plan.get_task_sub_pub("task_1");
        assert!(
            subscriptions.contains(&&ArrowTableSubscribe::AlwaysFullTable {
                table_name: "config_1".to_string()
            })
        );
        assert!(
            subscriptions.contains(&&ArrowTableSubscribe::OnUpdateFullTable {
                table_name: "state_1".to_string()
            })
        );
        assert!(publications.contains(&&ArrowTablePublish::Extend {
            table_name: "state_1".to_string()
        }));
    }

    #[test]
    fn test_get_processor_names() {
        let plan = test_session_context_builder::make_test_session_builder_parallel_task();
        let names = plan.get_processor_names();
        assert!(names.contains("processor_1"));
        assert!(names.contains("processor_2"));
        assert!(names.contains("processor_3"));
        assert!(names.contains("session_1"));
    }

    #[test]
    fn test_get_subject_names() {
        let plan = test_session_context_builder::make_test_session_builder_parallel_task();
        let names = plan.get_subject_names();
        assert!(names.contains("state_1"));
        assert!(names.contains("state_2"));
        assert!(names.contains("state_3"));
        assert!(names.contains("config_1"));
        assert!(names.contains("config_2"));
        assert!(names.contains("config_3"));
    }

    #[test]
    fn test_get_runtime_env_names() {
        let plan = test_session_context_builder::make_test_session_builder_parallel_task();
        let names = plan.get_runtime_env_names();
        assert!(names.contains("rt_1"));
    }

    #[test]
    fn get_processor_names_for_task() {
        let plan = test_session_context_builder::make_test_session_builder_parallel_task();
        let names = plan.get_processor_names_for_task("task_1");
        assert_eq!(names, vec!["processor_1".to_string()]);
    }

    #[test]
    fn test_session_build_success() -> Result<()> {
        let metrics = ArrowTaskMetricsSet::new();
        let session = test_session_context_builder::make_test_session_context_parallel_task(
            "session_1",
            metrics,
            10,
        )?;
        assert_eq!(session.get_states().len(), 6);
        assert_eq!(session.get_tasks().len(), 4);
        assert_eq!(session.get_name(), "session_1");
        assert_eq!(session.get_max_iter(), 10);
        Ok(())
    }

    #[test]
    fn test_session_build_fail_missing_plan() -> Result<()> {
        let result = SessionContextBuilder::new().build();
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
    fn test_session_build_fail_missing_processor() -> Result<()> {
        // No tasks
        let result = SessionContextBuilder::new()
            .with_tasks(test_session_context_builder::make_test_session_builder_tasks())
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
            ArrowProcessorMock::new_arc("processor_1"),
            ArrowProcessorMock::new_arc("processor_2"),
        ];
        let result = SessionContextBuilder::new()
            .with_tasks(test_session_context_builder::make_test_session_builder_tasks())
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
            ArrowProcessorMock::new_arc("processor_1"),
            ArrowProcessorMock::new_arc("processor_2"),
            ArrowProcessorMock::new_arc("processor_3"),
            ArrowProcessorMock::new_arc("not_found"),
        ];
        let result = SessionContextBuilder::new()
            .with_tasks(test_session_context_builder::make_test_session_builder_tasks())
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
    fn test_session_build_fail_missing_runtime_env() -> Result<()> {
        // No runtime env
        let result =
            test_session_context_builder::make_test_session_builder_parallel_task().build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Please add runtime environments before attempting to build the session."
            ),
        }

        // Missing runtime env
        let result = test_session_context_builder::make_test_session_builder_parallel_task()
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
        let result = test_session_context_builder::make_test_session_builder_parallel_task()
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
    fn test_session_build_fail_missing_state() -> Result<()> {
        // No state
        let result = test_session_context_builder::make_test_session_builder_parallel_task()
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
        let result = test_session_context_builder::make_test_session_builder_parallel_task()
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(make_state_tables("state_1", "config_1")?)
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided state [\"config_1\", \"config_2\", \"config_3\", \"state_1\", \"state_2\", \"state_3\"] and plan subjects and subscription names [\"config_1\", \"state_1\"]."
            ),
        }

        // State not found in plan
        let mut state = make_state_tables("state_1", "config_1")?;
        state.extend(make_state_tables("state_2", "config_2")?);
        state.extend(make_state_tables("not_found", "config_3")?);
        let result = test_session_context_builder::make_test_session_builder_parallel_task()
            .with_runtime_envs(vec![make_runtime_env("rt_1")?])
            .with_state(state)
            .build();
        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => assert_eq!(
                e.to_string(),
                "Mismatch between provided state [\"config_1\", \"config_2\", \"config_3\", \"state_1\", \"state_2\", \"state_3\"] and plan subjects and subscription names [\"config_1\", \"config_2\", \"config_3\", \"not_found\", \"state_1\", \"state_2\"]."
            ),
        }
        Ok(())
    }

    #[test]
    fn test_make_mermaid_flowchart() -> Result<()> {
        // Init runtime env
        let runtime_envs = vec![make_runtime_env("rt_1")?];

        // Init state
        let mut state = make_state_tables("state_1", "config_1")?;
        state.extend(make_state_tables("state_2", "config_2")?);
        state.extend(make_state_tables("state_3", "config_3")?);

        // Make the builder
        let builder = test_session_context_builder::make_test_session_builder_parallel_task()
            .with_runtime_envs(runtime_envs)
            .with_state(state);

        // Test the flowchart
        let mermaid_js_str = builder.make_mermaid_flowchart()?;
        assert_eq!(mermaid_js_str, "flowchart TD\n\tsubgraph task_1\n\t\tstate_1-subject -. FullTable .-> processor_1-subscribe\n\t\tconfig_1-subject -- FullTable --> processor_1-subscribe\n\t\tprocessor_1-subscribe --> processor_1-processor\n\t\tprocessor_1-processor --> processor_1-publish\n\t\tprocessor_1-publish -- Extend --> state_1-subject\n\tend\n\tsubgraph task_2\n\t\tstate_2-subject -. FullTable .-> processor_2-subscribe\n\t\tconfig_2-subject -- FullTable --> processor_2-subscribe\n\t\tprocessor_2-subscribe --> processor_2-processor\n\t\tprocessor_2-processor --> processor_2-publish\n\t\tprocessor_2-publish -- Extend --> state_2-subject\n\tend\n\tsubgraph task_3\n\t\tstate_3-subject -. FullTable .-> processor_3-subscribe\n\t\tconfig_3-subject -- FullTable --> processor_3-subscribe\n\t\tprocessor_3-subscribe --> processor_3-processor\n\t\tprocessor_3-processor --> processor_3-publish\n\t\tprocessor_3-publish -- Extend --> state_3-subject\n\tend\n\tsubgraph session_1\n\t\tstate_1-subject -. LastRecordBatch .-> session_1-subscribe\n\t\tstate_2-subject -. LastRecordBatch .-> session_1-subscribe\n\t\tstate_3-subject -. LastRecordBatch .-> session_1-subscribe\n\t\tsession_1-subscribe --> session_1-processor\n\t\tsession_1-processor --> session_1-publish\n\t\tsession_1-publish -- Extend --> state_1-subject\n\t\tsession_1-publish -- Extend --> state_2-subject\n\t\tsession_1-publish -- Extend --> state_3-subject\n\tend\n\trt_1-rt --> task_1\n\trt_1-rt --> task_2\n\trt_1-rt --> task_3\n\trt_1-rt --> session_1\n\tprocessor_2-processor@{ shape: rect, label: processor_2 }\n\tprocessor_1-processor@{ shape: rect, label: processor_1 }\n\tsession_1-processor@{ shape: rect, label: session_1 }\n\tprocessor_3-processor@{ shape: rect, label: processor_3 }\n\trt_1-rt@{ shape: subproc, label: rt_1 }\n\tconfig_3-subject@{ shape: doc, label: config_3 }\n\tconfig_1-subject@{ shape: doc, label: config_1 }\n\tstate_1-subject@{ shape: doc, label: state_1 }\n\tstate_3-subject@{ shape: doc, label: state_3 }\n\tconfig_2-subject@{ shape: doc, label: config_2 }\n\tstate_2-subject@{ shape: doc, label: state_2 }\n\tprocessor_1-publish@{ shape: fork }\n\tprocessor_2-publish@{ shape: fork }\n\tprocessor_3-publish@{ shape: fork }\n\tsession_1-publish@{ shape: fork }\n\tprocessor_1-subscribe@{ shape: diamond, label: All }\n\tprocessor_2-subscribe@{ shape: diamond, label: All }\n\tprocessor_3-subscribe@{ shape: diamond, label: All }\n\tsession_1-subscribe@{ shape: diamond, label: All }".to_string());
        Ok(())
    }
}
