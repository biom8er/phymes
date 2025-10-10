use std::sync::Arc;

use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use phymes_core::{
    metrics::HashMap,
    session::{
        common_traits::{MappableTrait, StateMap, TaskMap},
        runtime_env::RuntimeEnv,
        session_context::SessionContext,
        session_context_builder::{SessionContextBuilder, SessionContextBuilderTrait, TaskPlan},
    },
    table::table_trait::Table,
    task::processor::ProcessorTrait,
};

use crate::session_traits::tabular::SessionContextBuilderTabularTrait;

type SessionContextInput = (
    String,
    TaskMap,
    StateMap,
    HashMap<String, Arc<Mutex<RuntimeEnv>>>,
    usize,
    Vec<Table>,
);

/// Trait extension for [SessionContextBuilderTrait] to facilitate building agentic workflows
pub trait SessionContextBuilderAgentsTrait {
    /// Build the [SessionContext] objects along with the [SessionContext] schema tables
    fn build_inner_with_tables(self) -> Result<SessionContextInput>;

    fn build_with_tables(self) -> Result<SessionContext>
    where
        Self: Sized,
    {
        // build the tasks, state, and runtime objects
        let (name, tasks, mut state, runtime_envs, max_iter, tables) =
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
        ))
    }
}

impl SessionContextBuilderAgentsTrait for SessionContextBuilder {
    fn build_inner_with_tables(self) -> Result<SessionContextInput> {
        let (tables, _state) = self.to_arrow_tables(false, true, true)?;
        let (name, tasks, state, runtime_envs, max_iter) = self.build_inner()?;
        Ok((name, tasks, state, runtime_envs, max_iter, tables))
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
