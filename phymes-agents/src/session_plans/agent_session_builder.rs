use std::sync::Arc;

use phymes_core::{
    metrics::ArrowTaskMetricsSet,
    session::{
        runtime_env::RuntimeEnv, session_context::SessionContext, session_context_builder::TaskPlan,
    },
    table::arrow_table::ArrowTable,
    task::arrow_processor::ArrowProcessorTrait,
};

use anyhow::Result;

pub trait AgentSessionBuilderTrait {
    fn make_task_plan(&self) -> Vec<TaskPlan>;
    fn make_processors(&self) -> Vec<Arc<dyn ArrowProcessorTrait>>;
    fn make_runtime_envs(&self) -> Result<Vec<RuntimeEnv>>;
    fn make_state_tables(&self) -> Result<Vec<ArrowTable>>;
    fn make_session_context(&self, metrics: ArrowTaskMetricsSet) -> Result<SessionContext>;
}

// pub mod agent_session_schemas {
//     use std::sync::Arc;
//     use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

//     /// Chat agent messages schema
//     pub struct Messages {
//         /// Can be either assistant, user, system, or tool
//         role: String,
//         content: String,
//     }
//     impl Messages {
//         /// Default schemas that are re-used throughout
//         pub fn get_schema() -> SchemaRef {
//             let role = Field::new("role", DataType::Utf8, false);
//             let content = Field::new("content", DataType::Utf8, false);
//             Arc::new(Schema::new(vec![role, content]))
//         }
//         pub fn get_role_name() -> &'static str {
//             "role"
//         }
//         pub fn get_content_name() -> &'static str {
//             "content"
//         }
//     }

//     pub struct ToolConfig {
//         /// Assumed to be JSON
//         values: String
//     }

//     pub struct Documents {
//         document_id: String,
//         chunk_id: u32,
//         text: String,
//         embeddings: Vec<Vec<f32>>,
//     }

//     pub struct Queries {
//         query_id: String,
//         text: String,
//         embeddings: Vec<Vec<f32>>,
//     }

// }
