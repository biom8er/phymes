use anyhow::Result;
use std::sync::Arc;

use super::agent_session_builder::AgentSessionBuilderTrait;
#[cfg(feature = "openai_api")]
use crate::openai_asset::chat_processor::OpenAIChatProcessor;
use crate::{
    candle_chat::{
        chat_config::CandleChatConfig, chat_processor::CandleChatProcessor,
        message_aggregator_processor::MessageAggregatorProcessor,
        message_parser_processor::MessageParserProcessor,
    },
    candle_ops::{
        ops_processor::CandleOpProcessor, ops_which::WhichCandleOps,
        summary_config::CandleOpsSummaryConfig, summary_processor::OpsSummaryProcessor,
    },
};
use phymes_core::{
    metrics::ArrowTaskMetricsSet,
    session::{
        common_traits::BuilderTrait,
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context::SessionContext,
        session_context_builder::{SessionContextBuilder, SessionContextBuilderTrait, TaskPlan},
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait},
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::ArrowTableSubscribe,
    },
    task::arrow_processor::{ArrowProcessorEcho, ArrowProcessorTrait},
};

use arrow::{
    array::{ArrayRef, Float32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

/// Tool agent node with human-in-the-loop
///
/// # Supersteps
/// 1. Tool call: session -> chat_task (message_aggregator, chat_processor, message_parser)
/// 2. Tool invoke: -> tool_task (config_processor, ops_processor, summary_processor)
///    or human-in-the-loop_task (config_processor, ops_processor)
/// 3. if tool_task: -> chat_task (message_aggregator, chat_processor, message_parser)
///    else if human-in-the-loop_task: End
/// 4. End
///
/// ## Chat Task
/// chat_task is composed of chained processors:
/// message_aggregator -> chat_processor -> message_parser
/// where message_aggregator combines multiple messages
///   i.e., from parallel tool calls,
/// chat_processor does the text generation using an LLM,
/// and message_parser parses the LLM structured output to
///   determine 1) the destination of the messsage(s) and
///   2) split the messages into seperate tool calls
///
/// ## Tool Task
/// tool_task is composed of a chained processor:
/// config_processor -> ops_processor -> summary_processor
/// where config_processor parses the tool call and
///   creates the ops_config
/// where ops_processor performs the computation
///   according to the ops_config,
/// and summary_processor formats computation result
///   into a message for the chat_task
pub struct ToolAgentSession<'a> {
    /// Text generation inference capabilities (i.e, the agent)
    pub chat_task_name: &'a str,
    pub chat_processor_name: &'a str, // also used as the config name
    pub chat_runtime_env_name: &'a str,
    /// Structured text generation inference parser
    pub message_parser_task_name: &'a str, // needed for openai api
    pub message_parser_processor_name: &'a str,
    pub message_aggregator_task_name: &'a str, // needed for openai api
    pub message_aggregator_processor_name: &'a str,
    pub message_runtime_env_name: &'a str,
    /// The tool node (one of the CandleOps i.e., sort op)
    pub tool_task_name: &'a str,
    pub tool_processor_name: &'a str,
    pub tool_runtime_env_name: &'a str,
    pub summary_processor_name: &'a str,
    pub hitl_task_name: &'a str,
    pub hitl_processor_name: &'a str,
    /// Session and state
    pub session_context_name: &'a str,
    pub state_messages_table_name: &'a str,
    pub state_tools_table_name: &'a str,
    pub state_scores_table_name: &'a str,
    pub chat_api_url: Option<&'a str>,
}

impl ToolAgentSession<'_> {
    pub fn make_tools_table(&self) -> Result<ArrowTable> {
        let tool_id: ArrayRef = Arc::new(StringArray::from(vec![
            WhichCandleOps::SortScoresAndIndices.get_name(),
            WhichCandleOps::HumanInTheLoops.get_name(),
        ]));
        let tool: ArrayRef = Arc::new(StringArray::from(vec![
            WhichCandleOps::SortScoresAndIndices.get_json_tool_schema(),
            WhichCandleOps::HumanInTheLoops.get_json_tool_schema(),
        ]));
        let batch = RecordBatch::try_from_iter(vec![("tool_id", tool_id), ("tool", tool)])?;
        ArrowTableBuilder::new()
            .with_name(self.state_tools_table_name)
            .with_record_batches(vec![batch])?
            .build()
    }
    pub fn make_scores_table(&self) -> Result<ArrowTable> {
        let lhs_ids: ArrayRef = Arc::new(StringArray::from(vec!["0", "1", "2"]));
        let scores: ArrayRef = Arc::new(Float32Array::from(vec![3.0, 2.0, 1.0]));
        let batch = RecordBatch::try_from_iter(vec![("lhs_pk", lhs_ids), ("score", scores)])?;
        ArrowTableBuilder::new()
            .with_name(self.state_scores_table_name)
            .with_record_batches(vec![batch])?
            .build()
    }
    pub fn make_messages_table(&self) -> Result<ArrowTable> {
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![role, content]));
        ArrowTableBuilder::new()
            .with_name(self.state_messages_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_message_aggregator_table(&self) -> Result<ArrowTable> {
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![role, content]));
        ArrowTableBuilder::new()
            .with_name(self.message_aggregator_task_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_message_parser_table(&self) -> Result<ArrowTable> {
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![role, content]));
        ArrowTableBuilder::new()
            .with_name(self.message_parser_task_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_tool_config_table(&self) -> Result<ArrowTable> {
        let values = Field::new("values", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![values]));
        ArrowTableBuilder::new()
            .with_name(self.tool_task_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_hitl_config_table(&self) -> Result<ArrowTable> {
        let values = Field::new("values", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![values]));
        ArrowTableBuilder::new()
            .with_name(self.hitl_task_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
}

impl AgentSessionBuilderTrait for ToolAgentSession<'_> {
    fn make_task_plan(&self) -> Vec<TaskPlan> {
        let mut tasks = Vec::new();

        // DM: `Reqwest` connections break prematurely in `OpenAIChatProcessor`
        //  when chained or nested within other streams
        // DM: another tool agent session publish/subscribe network needs to be
        //  made for openai_api access that breaks down the chat task into seperate
        //  tasks for each processor...
        if cfg!(not(feature = "candle")) {
            tasks.push(TaskPlan {
                task_name: self.message_aggregator_task_name.to_string(),
                runtime_env_name: self.chat_runtime_env_name.to_string(),
                processor_names: vec![self.message_aggregator_processor_name.to_string()],
            });
            tasks.push(TaskPlan {
                task_name: self.chat_task_name.to_string(),
                runtime_env_name: self.chat_runtime_env_name.to_string(),
                processor_names: vec![self.chat_processor_name.to_string()],
            });
            tasks.push(TaskPlan {
                task_name: self.message_parser_task_name.to_string(),
                runtime_env_name: self.chat_runtime_env_name.to_string(),
                processor_names: vec![self.message_parser_processor_name.to_string()],
            });
        } else {
            tasks.push(TaskPlan {
                task_name: self.chat_task_name.to_string(),
                runtime_env_name: self.chat_runtime_env_name.to_string(),
                processor_names: vec![
                    self.message_aggregator_processor_name.to_string(),
                    self.chat_processor_name.to_string(),
                    self.message_parser_processor_name.to_string(),
                ],
            });
        }
        tasks.push(TaskPlan {
            task_name: self.tool_task_name.to_string(),
            runtime_env_name: self.tool_runtime_env_name.to_string(),
            processor_names: vec![
                self.tool_processor_name.to_string(),
                self.summary_processor_name.to_string(),
            ],
        });
        tasks.push(TaskPlan {
            task_name: self.hitl_task_name.to_string(),
            runtime_env_name: "rt_default".to_string(),
            processor_names: vec![
                self.hitl_task_name.to_string(),
                self.summary_processor_name.to_string(),
            ],
        });
        tasks.push(TaskPlan {
            task_name: self.session_context_name.to_string(),
            runtime_env_name: "rt_default".to_string(),
            processor_names: vec![self.session_context_name.to_string()],
        });

        tasks
    }

    fn make_processors(&self) -> Vec<Arc<dyn ArrowProcessorTrait>> {
        // The order is the order in which the processors are called in the task
        let mut processors = Vec::new();

        if cfg!(not(feature = "candle")) {
            processors.push(MessageAggregatorProcessor::new_with_pub_sub_for(
                self.message_aggregator_processor_name,
                &[ArrowTablePublish::Replace {
                    table_name: self.state_messages_table_name.to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateLastRecordBatch {
                        table_name: self.message_aggregator_task_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysLastRecordBatch {
                        table_name: self.state_messages_table_name.to_string(),
                    },
                ],
                &[],
            ));
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIChatProcessor::new_with_pub_sub_for(
                self.chat_processor_name,
                &[ArrowTablePublish::ExtendChunks {
                    table_name: self.message_parser_task_name.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.state_messages_table_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.state_tools_table_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.chat_processor_name.to_string(),
                    },
                ],
                &[],
            ));
            processors.push(MessageParserProcessor::new_with_pub_sub_for(
                self.message_parser_processor_name,
                &[
                    ArrowTablePublish::ExtendChunks {
                        // The first publication is the default publish target
                        table_name: self.state_messages_table_name.to_string(),
                        col_name: "content".to_string(),
                    },
                    ArrowTablePublish::Extend {
                        table_name: self.tool_task_name.to_string(),
                    },
                    ArrowTablePublish::Extend {
                        table_name: self.hitl_task_name.to_string(),
                    },
                ],
                &[
                    ArrowTableSubscribe::OnUpdateLastRecordBatch {
                        table_name: self.message_parser_task_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.message_parser_processor_name.to_string(),
                    },
                ],
                &[],
            ));
        } else {
            processors.push(MessageAggregatorProcessor::new_with_pub_sub_for(
                self.message_aggregator_processor_name,
                &[ArrowTablePublish::ExtendChunks {
                    table_name: self.state_messages_table_name.to_string(),
                    col_name: "content".to_string(),
                }],
                &[ArrowTableSubscribe::OnUpdateFullTable {
                    table_name: self.message_aggregator_task_name.to_string(),
                }],
                &[],
            ));
            processors.push(CandleChatProcessor::new_with_pub_sub_for(
                self.chat_processor_name,
                &[ArrowTablePublish::ExtendChunks {
                    table_name: self.message_parser_task_name.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    ArrowTableSubscribe::AlwaysFullTable {
                        // We only want to trigger on aggregator table
                        table_name: self.state_messages_table_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.state_tools_table_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.chat_processor_name.to_string(),
                    },
                ],
                &[],
            ));
            processors.push(MessageParserProcessor::new_with_pub_sub_for(
                self.message_parser_processor_name,
                &[
                    ArrowTablePublish::ExtendChunks {
                        // The first publication is the default publish target
                        table_name: self.state_messages_table_name.to_string(),
                        col_name: "content".to_string(),
                    },
                    ArrowTablePublish::Extend {
                        table_name: self.tool_task_name.to_string(),
                    },
                    ArrowTablePublish::Extend {
                        table_name: self.hitl_task_name.to_string(),
                    },
                ],
                &[
                    ArrowTableSubscribe::AlwaysLastRecordBatch {
                        // We only want to trigger an update on aggregator table
                        table_name: self.message_parser_task_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.message_parser_processor_name.to_string(),
                    },
                ],
                &[],
            ));
        }
        processors.push(CandleOpProcessor::new_with_pub_sub_for(
            self.tool_processor_name,
            &[ArrowTablePublish::Replace {
                table_name: self.state_scores_table_name.to_string(),
            }],
            &[
                ArrowTableSubscribe::OnUpdateLastRecordBatch {
                    table_name: self.tool_task_name.to_string(),
                },
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: self.state_scores_table_name.to_string(),
                },
            ],
            &[self.summary_processor_name],
        ));
        processors.push(CandleOpProcessor::new_with_pub_sub_for(
            self.hitl_processor_name,
            &[ArrowTablePublish::Extend {
                table_name: self.state_messages_table_name.to_string(),
            }],
            &[ArrowTableSubscribe::OnUpdateLastRecordBatch {
                table_name: self.hitl_task_name.to_string(),
            }],
            &[],
        ));
        processors.push(OpsSummaryProcessor::new_with_pub_sub_for(
            self.summary_processor_name,
            &[ArrowTablePublish::Extend {
                table_name: self.message_aggregator_task_name.to_string(),
            }],
            &[ArrowTableSubscribe::AlwaysLastRecordBatch {
                table_name: self.summary_processor_name.to_string(),
            }],
            &[],
        ));
        processors.push(ArrowProcessorEcho::new_with_pub_sub_for(
            self.session_context_name,
            &[ArrowTablePublish::Extend {
                table_name: self.state_messages_table_name.to_string(),
            }],
            &[ArrowTableSubscribe::OnUpdateLastRecordBatch {
                table_name: self.state_messages_table_name.to_string(),
            }],
            &[],
        ));
        processors
    }

    fn make_runtime_envs(&self) -> Result<Vec<RuntimeEnv>> {
        Ok(vec![
            RuntimeEnv::new().with_name(self.chat_runtime_env_name),
            RuntimeEnv::new().with_name(self.tool_runtime_env_name),
            RuntimeEnv::new().with_name("rt_default"),
        ])
    }

    fn make_state_tables(&self) -> Result<Vec<ArrowTable>> {
        // Default chat config
        #[allow(unused_mut)]
        let mut candle_chat_config = CandleChatConfig {
            max_tokens: 1000,
            temperature: 0.8,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            // All files need to be local for WASM testing
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(
                crate::candle_assets::candle_which::WhichCandleAsset::SmolLM2_135MChat,
            ),
            ..Default::default()
        };

        // Add hf_hub if available
        #[cfg(feature = "hf_hub")]
        {
            candle_chat_config.candle_asset =
                Some(crate::candle_assets::candle_which::WhichCandleAsset::QwenV2p5_3bChat);
            candle_chat_config.openai_asset = None;
            candle_chat_config.weights_config_file = None;
            candle_chat_config.weights_file = None;
            candle_chat_config.tokenizer_file = None;
            candle_chat_config.tokenizer_config_file = None;
        }

        // Add openAI_api if available
        #[cfg(not(feature = "candle"))]
        {
            candle_chat_config.candle_asset = None;
            // DM: Bug in Llama model system template that requires it to only call tools instead of respond...
            // see update template <https://gist.github.com/K-Mistele/820d142b4dab50bd8ef0c7bbcad4515c>
            // see discussion when using vLLM <https://github.com/vllm-project/vllm/issues/9991>
            candle_chat_config.openai_asset =
                Some(crate::openai_asset::openai_which::WhichOpenAIAsset::MetaLlamaV3p2_1B);
            candle_chat_config.weights_config_file = None;
            candle_chat_config.weights_file = None;
            candle_chat_config.tokenizer_file = None;
            candle_chat_config.tokenizer_config_file = None;
            candle_chat_config.api_url = self.chat_api_url.map(|s| s.to_string());
        }

        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config)?;
        let candle_chat_state = ArrowTableBuilder::new()
            .with_name(self.chat_processor_name)
            .with_json(&candle_chat_config_json.clone(), 1)?
            .build()?;
        let candle_message_parser_state = ArrowTableBuilder::new()
            .with_name(self.message_parser_processor_name)
            .with_json(&candle_chat_config_json, 1)?
            .build()?;

        // Summary config
        let summary_config = CandleOpsSummaryConfig {
            ..Default::default()
        };
        let summary_config_json = serde_json::to_vec(&summary_config)?;
        let summary_state = ArrowTableBuilder::new()
            .with_name(self.summary_processor_name)
            .with_json(&summary_config_json, 1)?
            .build()?;

        Ok(vec![
            candle_chat_state,
            candle_message_parser_state,
            summary_state,
            self.make_scores_table()?,
            self.make_messages_table()?,
            self.make_message_aggregator_table()?,
            self.make_message_parser_table()?,
            self.make_tools_table()?,
            self.make_tool_config_table()?,
            self.make_hitl_config_table()?,
        ])
    }

    fn make_session_context(&self, metrics: ArrowTaskMetricsSet) -> Result<SessionContext> {
        SessionContextBuilder::new()
            .with_name(self.session_context_name)
            .with_tasks(self.make_task_plan())
            .with_metrics(metrics)
            .with_runtime_envs(self.make_runtime_envs()?)
            .with_state(self.make_state_tables()?)
            .with_processors(self.make_processors())
            .with_max_iter(8)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        metrics::HashMap,
        session::{
            common_traits::MappableTrait,
            session_context::{SessionStream, SessionStreamState},
        },
        table::{arrow_table::ArrowTableTrait, arrow_table_publish::ArrowTablePublish},
        task::arrow_message::{
            ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait,
            ArrowIncomingMessageTrait, ArrowMessageBuilderTrait,
        },
    };

    use super::*;
    use crate::candle_chat::message_history::MessageHistoryBuilderTraitExt;

    #[tokio::test(flavor = "current_thread")]
    async fn test_tool_agent_session() -> Result<()> {
        // initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // initialize the session
        let tool_agent_session = ToolAgentSession {
            session_context_name: "session_1",
            chat_processor_name: "chat_processor_1",
            chat_task_name: "chat_task_1",
            chat_runtime_env_name: "chat_rt_1",
            tool_task_name: WhichCandleOps::SortScoresAndIndices.get_name(),
            tool_processor_name: WhichCandleOps::SortScoresAndIndices.get_name(),
            tool_runtime_env_name: "tool_rt_1",
            summary_processor_name: "summary_processor_1",
            hitl_task_name: WhichCandleOps::HumanInTheLoops.get_name(),
            hitl_processor_name: WhichCandleOps::HumanInTheLoops.get_name(),
            message_parser_task_name: "message_parser_task_1",
            message_parser_processor_name: "message_parser_processor_1",
            message_aggregator_task_name: "message_aggregator_task_1",
            message_aggregator_processor_name: "message_aggregator_processor_1",
            message_runtime_env_name: "message_rt_1",
            state_messages_table_name: "messages",
            state_scores_table_name: "available_data_1",
            state_tools_table_name: "tools",
            chat_api_url: Some("http://0.0.0.0:8000/v1"),
        };
        let session_ctx = tool_agent_session.make_session_context(metrics.clone())?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        // ----- Query #1 -----

        // Make the system prompt and add the user query
        let message_builder = ArrowTableBuilder::new()
            .with_name(tool_agent_session.message_aggregator_task_name)
            // .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`. Do not call a function if you are able to answer the questions with information from previous tool_response",
                "user"
            )?;

        // Build the current message state
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(tool_agent_session.message_aggregator_task_name)
            .with_subject(tool_agent_session.message_aggregator_task_name)
            .with_publisher(tool_agent_session.session_context_name)
            .with_message(message_builder.clone().build()?)
            .with_update(&ArrowTablePublish::Extend {
                table_name: tool_agent_session.message_aggregator_task_name.to_string(),
            })
            .build()?;
        let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Run the session
        let session_stream =
            SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
                session_stream.try_collect().await?;

            println!(
                "Iters: {}",
                session_stream_state.try_read().unwrap().get_iter()
            );

            println!(
                "Messages: {:?}",
                session_stream_state
                    .try_read()
                    .unwrap()
                    .get_session_context()
                    .get_states()
                    .get(tool_agent_session.state_messages_table_name)
                    .unwrap()
            );

            println!(
                "Message Aggregator: {:?}",
                session_stream_state
                    .try_read()
                    .unwrap()
                    .get_session_context()
                    .get_states()
                    .get(tool_agent_session.message_aggregator_task_name)
                    .unwrap()
            );

            // Update the chat history with the response
            let json_data = response
                .last_mut()
                .unwrap()
                .remove(&format!(
                    "from_{}_on_{}",
                    tool_agent_session.session_context_name,
                    tool_agent_session.state_messages_table_name
                ))
                .unwrap()
                .get_message_own()
                .to_json_object()?;
            for row in &json_data {
                if row["role"] != "system" {
                    println!("{}: {}", row["role"], row["content"])
                }
            }

            for metric in metrics.clone_inner().iter() {
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == tool_agent_session.chat_processor_name
                {
                    assert!(metric.value().as_usize() > 0);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap()
                        == tool_agent_session.message_parser_processor_name
                {
                    assert!(metric.value().as_usize() > 0 || metric.value().as_usize() == 1);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == tool_agent_session.tool_processor_name
                {
                    assert_eq!(metric.value().as_usize(), 3);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == tool_agent_session.summary_processor_name
                {
                    assert_eq!(metric.value().as_usize(), 1);
                }
            }

            // DM: Bug in Llama model system template that requires it to only call tools instead of respond...
            if cfg!(feature = "candle") {
                assert_eq!(json_data.first().unwrap().get("role").unwrap(), "assistant");
            }
            assert!(json_data.first().unwrap().get("content").is_some());
        }

        Ok(())
    }
}
