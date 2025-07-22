use anyhow::Result;
use std::sync::Arc;

use super::agent_session_builder::AgentSessionBuilderTrait;
use crate::candle_chat::{chat_config::CandleChatConfig, chat_processor::CandleChatProcessor};
#[cfg(feature = "openai_api")]
use crate::openai_asset::chat_processor::OpenAIChatProcessor;
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

use arrow::datatypes::{DataType, Field, Schema};

pub struct ChatAgentSession<'a> {
    pub chat_task_name: &'a str,
    pub chat_processor_name: &'a str,
    pub runtime_env_name: &'a str,
    pub session_context_name: &'a str,
    pub chat_subscription_name: &'a str,
    pub chat_api_url: Option<&'a str>,
}

impl Default for ChatAgentSession<'_> {
    fn default() -> Self {
        ChatAgentSession {
            chat_task_name: "chat_task_1",
            chat_processor_name: "chat_processor_1",
            runtime_env_name: "rt_default_1",
            session_context_name: "session_context_1",
            chat_subscription_name: "messages",
            chat_api_url: None,
        }
    }
}

impl<'a> ChatAgentSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        ChatAgentSession {
            session_context_name,
            ..Default::default()
        }
    }
}

impl AgentSessionBuilderTrait for ChatAgentSession<'_> {
    fn make_task_plan(&self) -> Vec<TaskPlan> {
        vec![
            TaskPlan {
                task_name: self.chat_task_name.to_string(),
                runtime_env_name: self.runtime_env_name.to_string(),
                processor_names: vec![self.chat_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.session_context_name.to_string(),
                runtime_env_name: "rt_default".to_string(),
                processor_names: vec![self.session_context_name.to_string()],
            },
        ]
    }

    fn make_processors(&self) -> Vec<Arc<dyn ArrowProcessorTrait>> {
        let mut processors = Vec::new();
        // The order is the order in which the processors are called in the task
        if cfg!(not(feature = "candle")) {
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIChatProcessor::new_with_pub_sub_for(
                self.chat_processor_name,
                &[ArrowTablePublish::ExtendChunks {
                    table_name: self.chat_subscription_name.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.chat_subscription_name.to_string(),
                    },
                    ArrowTableSubscribe::None,
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.chat_processor_name.to_string(),
                    },
                ],
                &[],
            ));
        } else {
            processors.push(CandleChatProcessor::new_with_pub_sub_for(
                self.chat_processor_name,
                &[ArrowTablePublish::ExtendChunks {
                    table_name: self.chat_subscription_name.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.chat_subscription_name.to_string(),
                    },
                    ArrowTableSubscribe::None,
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.chat_processor_name.to_string(),
                    },
                ],
                &[],
            ));
        }
        processors.push(ArrowProcessorEcho::new_with_pub_sub_for(
            self.session_context_name,
            &[ArrowTablePublish::Extend {
                table_name: self.chat_subscription_name.to_string(),
            }],
            &[ArrowTableSubscribe::OnUpdateLastRecordBatch {
                table_name: self.chat_subscription_name.to_string(),
            }],
            &[],
        ));
        processors
    }

    fn make_runtime_envs(&self) -> Result<Vec<RuntimeEnv>> {
        Ok(vec![
            RuntimeEnv::new().with_name(self.runtime_env_name),
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
                Some(crate::candle_assets::candle_which::WhichCandleAsset::QwenV2p5_1p5bChat);
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
            candle_chat_config.openai_asset =
                Some(crate::openai_asset::openai_which::WhichOpenAIAsset::MetaLlamaV3p2_1B);
            candle_chat_config.weights_config_file = None;
            candle_chat_config.weights_file = None;
            candle_chat_config.tokenizer_file = None;
            candle_chat_config.tokenizer_config_file = None;
            candle_chat_config.api_url = self.chat_api_url.map(|s| s.to_string());
        }
        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config)?;
        let config = ArrowTableBuilder::new()
            .with_name(self.chat_processor_name)
            .with_json(&candle_chat_config_json, 1)?
            .build()?;

        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let timestamp = Field::new("timestamp", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![role, content, timestamp]));
        let messages = ArrowTableBuilder::new()
            .with_name(self.chat_subscription_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()?;
        Ok(vec![config, messages])
    }

    fn make_session_context(&self, metrics: ArrowTaskMetricsSet) -> Result<SessionContext> {
        SessionContextBuilder::new()
            .with_name(self.session_context_name)
            .with_tasks(self.make_task_plan())
            .with_metrics(metrics)
            .with_runtime_envs(self.make_runtime_envs()?)
            .with_state(self.make_state_tables()?)
            .with_processors(self.make_processors())
            .build()
    }
}

pub mod test_chat_agent_session {
    use parking_lot::{Mutex, RwLock};
    use phymes_core::{
        metrics::HashMap,
        session::{
            common_traits::{BuildableTrait, MappableTrait},
            session_context::{SessionStream, SessionStreamState},
        },
        table::arrow_table::ArrowTableTrait,
        task::arrow_message::{
            ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait,
            ArrowMessageBuilderTrait, ArrowOutgoingMessage, ArrowOutgoingMessageBuilderTrait,
            ArrowOutgoingMessageTrait,
        },
    };

    use super::*;

    use crate::candle_chat::message_history::MessageHistoryBuilderTraitExt;
    #[allow(unused_imports)]
    #[cfg(feature = "openai_api")]
    use crate::openai_asset::chat_processor::OpenAIChatProcessor;

    /// Run the chat processor with a given config and return the message history
    pub async fn bench_chat_processor(
        metrics: ArrowTaskMetricsSet,
        config: &CandleChatConfig,
        user_content: &str,
        name: &str,
    ) -> Result<ArrowTable> {
        // Named variables
        let messages = "messages";

        // State for the chat processor config
        let candle_chat_config_json = serde_json::to_vec(config)?;
        let candle_chat_config_table = ArrowTableBuilder::new()
            .with_name(name)
            .with_json(&candle_chat_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = ArrowTableBuilder::new()
            .with_name(messages)
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(user_content, "user")?;

        // Build the current message state
        let mut message = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&ArrowTablePublish::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            candle_chat_config_table.get_name().to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name(candle_chat_config_table.get_name())
                .with_publisher("")
                .with_subject(candle_chat_config_table.get_name())
                .with_update(&ArrowTablePublish::None)
                .with_message(candle_chat_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the chat task
        #[allow(unused_variables)]
        let chat_processor = CandleChatProcessor::new_with_pub_sub_for(
            name,
            &[ArrowTablePublish::ExtendChunks {
                table_name: messages.to_string(),
                col_name: "content".to_string(),
            }],
            &[
                ArrowTableSubscribe::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                ArrowTableSubscribe::None,
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: candle_chat_config_table.get_name().to_string(),
                },
            ],
            &[],
        );
        #[cfg(all(not(feature = "candle"), feature = "openai_api"))]
        let chat_processor = OpenAIChatProcessor::new_with_pub_sub_for(
            name,
            &[ArrowTablePublish::ExtendChunks {
                table_name: messages.to_string(),
                col_name: "content".to_string(),
            }],
            &[
                ArrowTableSubscribe::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                ArrowTableSubscribe::None,
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: candle_chat_config_table.get_name().to_string(),
                },
            ],
            &[],
        );
        let mut stream = chat_processor.process(
            message,
            metrics,
            Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt"))),
        )?;

        // Update the chat history with the response
        let (message_builder, _stream) = message_builder
            .append_chat_response_sendable_record_batch_stream(
                &mut stream.remove(messages).unwrap().get_message_own(),
                1000,
            )
            .await?;
        message_builder.clone().build()
    }

    /// Run the first query for the chat agent session and return the response
    pub fn bench_chat_agent_session_1<'a>(
        session_stream_state: Arc<RwLock<SessionStreamState>>,
        chat_agent_session: &ChatAgentSession<'a>,
        user_content: &str,
    ) -> SessionStream {
        // Make the system prompt and add the user query
        let message_builder = ArrowTableBuilder::new()
            .with_name(chat_agent_session.chat_subscription_name)
            .insert_system_template_str("You are a helpful assistant.")
            .unwrap()
            .append_new_user_query_str(user_content, "user")
            .unwrap();

        // Build the current message state
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(chat_agent_session.chat_subscription_name)
            .with_subject(chat_agent_session.chat_task_name)
            .with_publisher(chat_agent_session.session_context_name)
            .with_message(message_builder.build().unwrap())
            .with_update(&ArrowTablePublish::Extend {
                table_name: chat_agent_session.chat_subscription_name.to_string(),
            })
            .build()
            .unwrap();
        let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Run the session
        SessionStream::new(incoming_message_map, session_stream_state)
    }

    /// Run the second query for the chat agent session and return the response
    pub fn bench_chat_agent_session_2<'a>(
        session_stream_state: Arc<RwLock<SessionStreamState>>,
        chat_agent_session: &ChatAgentSession<'a>,
        user_content: &str,
    ) -> SessionStream {
        // Add a new query to the message history
        let message_builder = ArrowTableBuilder::new()
            .with_name(chat_agent_session.chat_subscription_name)
            .append_new_user_query_str(user_content, "user")
            .unwrap();

        // Build the incoming message state
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(chat_agent_session.chat_subscription_name)
            .with_subject(chat_agent_session.chat_task_name)
            .with_publisher(chat_agent_session.session_context_name)
            .with_message(message_builder.clone().build().unwrap())
            .with_update(&ArrowTablePublish::Extend {
                table_name: chat_agent_session.chat_subscription_name.to_string(),
            })
            .build()
            .unwrap();
        let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Run the session
        session_stream_state.try_write().unwrap().set_iter(0);
        SessionStream::new(incoming_message_map, session_stream_state)
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        metrics::HashMap,
        session::session_context::SessionStreamState,
        table::arrow_table::ArrowTableTrait,
        task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageTrait},
    };

    use super::*;
    use test_chat_agent_session::{bench_chat_agent_session_1, bench_chat_agent_session_2};

    #[tokio::test(flavor = "current_thread")]
    async fn test_chat_agent_session() -> Result<()> {
        // initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // initialize the session
        let chat_agent_session = ChatAgentSession::default();
        let session_ctx = chat_agent_session.make_session_context(metrics.clone())?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        // Skip actually running the session as it takes too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // ----- Query #1 -----
            let session_stream = bench_chat_agent_session_1(
                Arc::clone(&session_stream_state),
                &chat_agent_session,
                "Write a function to count prime numbers up to N.",
            );
            let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
                session_stream.try_collect().await?;

            // Update the chat history with the response
            let json_data = response
                .last_mut()
                .unwrap()
                .remove(&format!(
                    "from_{}_on_{}",
                    chat_agent_session.session_context_name,
                    chat_agent_session.chat_subscription_name
                ))
                .unwrap()
                .get_message_own()
                .to_json_object()?;
            for row in &json_data {
                if row["role"] != "system" {
                    println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
                }
            }

            for metric in metrics.clone_inner().iter() {
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == "chat_task_1"
                {
                    assert_eq!(metric.value().as_usize(), 2);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == "chat_processor_1"
                {
                    assert!(metric.value().as_usize() >= 1);
                }
            }

            assert_eq!(json_data.first().unwrap().get("role").unwrap(), "assistant");
            assert!(json_data.first().unwrap().get("content").is_some());

            // ----- Query #2 -----
            let session_stream = bench_chat_agent_session_2(
                Arc::clone(&session_stream_state),
                &chat_agent_session,
                "Please provide an example using the functions.",
            );
            let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
                session_stream.try_collect().await?;

            // Update the chat history with the response
            let json_data = response
                .first_mut()
                .unwrap()
                .remove(&format!(
                    "from_{}_on_{}",
                    chat_agent_session.session_context_name,
                    chat_agent_session.chat_subscription_name
                ))
                .unwrap()
                .get_message_own()
                .to_json_object()?;
            for row in &json_data {
                if row["role"] != "system" {
                    println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
                }
            }

            for metric in metrics.clone_inner().iter() {
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == "chat_task_1"
                    && metric.value().as_usize() != 2
                {
                    assert_eq!(metric.value().as_usize(), 4);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == "chat_processor_1"
                {
                    assert!(metric.value().as_usize() >= 1);
                }
            }

            assert_eq!(json_data.first().unwrap().get("role").unwrap(), "assistant");
            assert!(json_data.first().unwrap().get("content").is_some());
        }

        Ok(())
    }
}
