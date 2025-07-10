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
        let schema = Arc::new(Schema::new(vec![role, content]));
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
    async fn test_chat_agent_session() -> Result<()> {
        // initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // initialize the session
        let chat_agent_session = ChatAgentSession {
            session_context_name: "session_1",
            chat_processor_name: "chat_processor_1",
            chat_task_name: "chat_task_1",
            runtime_env_name: "rt_1",
            chat_subscription_name: "messages",
            chat_api_url: Some("http://0.0.0.0:8000/v1"),
        };
        let session_ctx = chat_agent_session.make_session_context(metrics.clone())?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        // ----- Query #1 -----

        // Make the system prompt and add the user query
        let message_builder = ArrowTableBuilder::new()
            .with_name(chat_agent_session.chat_subscription_name)
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(
                "Write a function to count prime numbers up to N.",
                "user",
            )?;

        // Build the current message state
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(chat_agent_session.chat_subscription_name)
            .with_subject(chat_agent_session.chat_task_name)
            .with_publisher(chat_agent_session.session_context_name)
            .with_message(message_builder.build()?)
            .with_update(&ArrowTablePublish::Extend {
                table_name: chat_agent_session.chat_subscription_name.to_string(),
            })
            .build()?;
        let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Run the session
        let session_stream =
            SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));

        // Skip actually running the session as it takes too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
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
                    println!("{}: {}", row["role"], row["content"])
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

            // Add a new query to the message history
            let message_builder = ArrowTableBuilder::new()
                .with_name(chat_agent_session.chat_subscription_name)
                .append_new_user_query_str(
                    "Please provide an example using the functions.",
                    "user",
                )?;

            // Build the incoming message state
            let incoming_message = ArrowIncomingMessageBuilder::new()
                .with_name(chat_agent_session.chat_subscription_name)
                .with_subject(chat_agent_session.chat_task_name)
                .with_publisher(chat_agent_session.session_context_name)
                .with_message(message_builder.clone().build()?)
                .with_update(&ArrowTablePublish::Extend {
                    table_name: chat_agent_session.chat_subscription_name.to_string(),
                })
                .build()?;
            let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
            incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

            // Run the session
            session_stream_state.try_write().unwrap().set_iter(0);
            let session_stream =
                SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));

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
                    println!("{}: {}", row["role"], row["content"])
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
