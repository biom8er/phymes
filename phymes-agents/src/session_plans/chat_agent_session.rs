use std::sync::Arc;

use phymes_core::{
    schemas::message_history::create_messages_schema,
    session::{
        common_traits::BuilderTrait,
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context_builder::TaskPlan,
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait},
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::{AllTableNamesSubscribe, ArrowTableSubscribe, SubscribeTrait},
    },
    task::arrow_processor::{ArrowProcessorEcho, ArrowProcessorTrait},
};
use phymes_ml::{
    candle_assets::available_candle_assets::AvailableCandleAssets,
    candle_chat::{chat_config::CandleChatConfig, chat_processor::CandleChatProcessor},
};
#[cfg(feature = "openai_api")]
use phymes_ml::{
    openai_asset::available_openai_assets::AvailableOpenAIAssets, openai_chat::chat_processor::OpenAIChatProcessor,
};

use crate::session_traits::agents::CustomAgentsBuilderTrait;

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

impl CustomAgentsBuilderTrait for ChatAgentSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        Some(vec![
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
        ])
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ArrowProcessorTrait>>> {
        let mut processors = Vec::new();
        // The order is the order in which the processors are called in the task
        if cfg!(not(feature = "candle")) {
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIChatProcessor::new_arc_with_pub_sub(
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
                AllTableNamesSubscribe::new_box(),
            ));
        } else {
            processors.push(CandleChatProcessor::new_arc_with_pub_sub(
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
                AllTableNamesSubscribe::new_box(),
            ));
        }
        processors.push(ArrowProcessorEcho::new_arc_with_pub_sub(
            self.session_context_name,
            &[ArrowTablePublish::Extend {
                table_name: self.chat_subscription_name.to_string(),
            }],
            &[ArrowTableSubscribe::OnUpdateLastRecordBatch {
                table_name: self.chat_subscription_name.to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        ));
        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name(self.runtime_env_name),
            RuntimeEnv::new().with_name("rt_default"),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<ArrowTable>> {
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
            candle_asset: Some(AvailableCandleAssets::SmolLM2_135MChat),
            ..Default::default()
        };

        // Add hf_hub if available
        #[cfg(feature = "hf_hub")]
        {
            candle_chat_config.candle_asset = Some(AvailableCandleAssets::QwenV2p5_1p5bChat);
            candle_chat_config.openai_asset = None;
            candle_chat_config.weights_config_file = None;
            candle_chat_config.weights_file = None;
            candle_chat_config.tokenizer_file = None;
            candle_chat_config.tokenizer_config_file = None;
        }

        // Add openAI_api if available
        #[cfg(all(feature = "openai_api", not(feature = "candle")))]
        {
            candle_chat_config.candle_asset = None;
            candle_chat_config.openai_asset = Some(WhichOpenAIAsset::MetaLlamaV3p2_1B);
            candle_chat_config.weights_config_file = None;
            candle_chat_config.weights_file = None;
            candle_chat_config.tokenizer_file = None;
            candle_chat_config.tokenizer_config_file = None;
            candle_chat_config.api_url = self.chat_api_url.map(|s| s.to_string());
        }
        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config).unwrap();
        let config = ArrowTableBuilder::new()
            .with_name(self.chat_processor_name)
            .with_json(&candle_chat_config_json, 1).unwrap()
            .build().unwrap();

        let messages = ArrowTableBuilder::new()
            .with_name(self.chat_subscription_name)
            .with_schema(create_messages_schema())
            .with_record_batches(Vec::new()).unwrap()
            .build().unwrap();
        Some(vec![config, messages])
    }
}

pub mod test_chat_agent_session {
    use parking_lot::RwLock;
    use phymes_core::{
        metrics::HashMap,
        session::{
            common_traits::MappableTrait,
            session_context::{SessionStream, SessionStreamState},
        },
        task::arrow_message::{
            ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait,
            ArrowMessageBuilderTrait,
        },
    };

    use super::*;

    use phymes_core::schemas::message_history::MessageHistoryBuilderTraitExt;

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
    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        metrics::{ArrowTaskMetricsSet, HashMap},
        session::{session_context::SessionStreamState, session_context_builder::SessionContextBuilderTrait},
        table::arrow_table::ArrowTableTrait,
        task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageTrait},
    };

    use crate::session_traits::agents::SessionContextBuilderAgentsTrait;

    use super::*;
    use test_chat_agent_session::{bench_chat_agent_session_1, bench_chat_agent_session_2};

    #[tokio::test(flavor = "current_thread")]
    async fn test_chat_agent_session() -> Result<()> {
        // initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // initialize the session
        let chat_agent_session = ChatAgentSession::default();
        let session_ctx = chat_agent_session.build()
            .with_metrics(metrics.clone())
            .with_name(chat_agent_session.session_context_name)
            .build_with_tables()?;
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
