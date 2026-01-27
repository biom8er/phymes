use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableTableSubscribePolicies, BuilderTrait,
    ProcessorPlan, ProcessorPlanBuilder, RuntimeEnv, RuntimeEnvTrait, Table, TableBuilder,
    TableBuilderTrait, TablePublication, TableSubscription, TaskPlan,
};
use phymes_data::{AvailableCandleOperators, DataConfig};
#[cfg(feature = "api")]
use phymes_ml::AvailableOpenAIAssets;
use phymes_ml::{AvailableCandleAssets, CandleChatConfig};

use crate::{AvailableInterfaceSubjects, AvailableProcessors, CustomAgentsBuilderTrait};

pub struct ChatAgentSession<'a> {
    /// Chat tasks
    pub chat_task_name: &'a str,
    pub chat_processor_name: &'a str,
    pub chat_runtime_env_name: &'a str,
    /// Aggregator for the chat task
    pub message_aggregator_task_1_name: &'a str,
    pub message_aggregator_processor_1_name: &'a str,
    /// Aggregator for the UI messages
    pub message_aggregator_task_2_name: &'a str,
    pub message_aggregator_processor_2_name: &'a str,
    pub message_aggregator_runtime_env_name: &'a str,
    /// Session and state
    pub session_context_name: &'a str,
    /// Other parameters
    pub chat_api_url: Option<&'a str>,
}

impl Default for ChatAgentSession<'_> {
    fn default() -> Self {
        ChatAgentSession {
            chat_task_name: "chat_task_1",
            message_aggregator_task_1_name: "message_aggregator_task_1",
            message_aggregator_processor_1_name: "message_aggregator_1",
            message_aggregator_task_2_name: "message_aggregator_task_2",
            message_aggregator_processor_2_name: "message_aggregator_2",
            message_aggregator_runtime_env_name: "message_aggregator_runtime_env_1",
            chat_processor_name: "chat_processor_1",
            chat_runtime_env_name: "chat_rt_1",
            session_context_name: "session_context_1",
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
        // DM: `Reqwest` connections break prematurely in `OpenAIChatProcessor`
        //  when chained or nested within other streams.
        let tasks = vec![
            TaskPlan {
                task_name: self.message_aggregator_task_1_name.to_string(),
                runtime_env_name: self.message_aggregator_runtime_env_name.to_string(),
                processor_names: vec![self.message_aggregator_processor_1_name.to_string()],
            },
            TaskPlan {
                task_name: self.message_aggregator_task_2_name.to_string(),
                runtime_env_name: self.message_aggregator_runtime_env_name.to_string(),
                processor_names: vec![self.message_aggregator_processor_2_name.to_string()],
            },
            TaskPlan {
                task_name: self.chat_task_name.to_string(),
                runtime_env_name: self.chat_runtime_env_name.to_string(),
                processor_names: vec![self.chat_processor_name.to_string()],
            },
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        // The order is the order in which the processors are called in the task
        let mut processors = Vec::new();

        processors.push(
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::MessageAggregatorProcessor
                        .build_arc(self.message_aggregator_processor_1_name),
                )
                .with_publications(&[TablePublication::Replace {
                    table_name: self.chat_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateFullTable {
                        table_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                    },
                    TableSubscription::AlwaysFullTable {
                        table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    },
                    TableSubscription::AlwaysLastRecordBatch {
                        table_name: self.message_aggregator_processor_1_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
        );

        processors.push(
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::MessageAggregatorProcessor
                        .build_arc(self.message_aggregator_processor_2_name),
                )
                .with_publications(&[TablePublication::Extend {
                    table_name: AvailableInterfaceSubjects::AggregatedMessages.to_string(),
                }])
                .with_subscriptions(&[
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                    },
                    TableSubscription::OnUpdateLastRecordBatch {
                        table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    },
                    TableSubscription::AlwaysLastRecordBatch {
                        table_name: self.message_aggregator_processor_2_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
        );

        if cfg!(not(feature = "candle")) {
            #[cfg(feature = "api")]
            processors.push(
                ProcessorPlanBuilder::default()
                    .with_processor(
                        AvailableProcessors::OpenAIChatProcessor
                            .build_arc(self.chat_processor_name),
                    )
                    .with_publications(&[TablePublication::ExtendChunks {
                        table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                        col_name: "content".to_string(),
                    }])
                    .with_subscriptions(&[
                        TableSubscription::OnUpdateFullTable {
                            table_name: self.chat_task_name.to_string(),
                        },
                        TableSubscription::AlwaysFullTable {
                            table_name: self.chat_processor_name.to_string(),
                        },
                    ])
                    .with_subscribe_policy(
                        AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                    )
                    .build()
                    .unwrap(),
            );
        } else {
            processors.push(
                ProcessorPlanBuilder::default()
                    .with_processor(
                        AvailableProcessors::CandleChatProcessor
                            .build_arc(self.chat_processor_name),
                    )
                    .with_publications(&[TablePublication::ExtendChunks {
                        table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                        col_name: "content".to_string(),
                    }])
                    .with_subscriptions(&[
                        TableSubscription::OnUpdateFullTable {
                            table_name: self.chat_task_name.to_string(),
                        },
                        TableSubscription::AlwaysFullTable {
                            table_name: self.chat_processor_name.to_string(),
                        },
                    ])
                    .with_subscribe_policy(
                        AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
                    )
                    .build()
                    .unwrap(),
            );
        }

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name(self.chat_runtime_env_name),
            RuntimeEnv::new().with_name(self.message_aggregator_runtime_env_name),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
        // Default chat config
        #[allow(unused_mut)]
        let mut candle_chat_config = CandleChatConfig {
            messages: self.chat_task_name.to_string(),
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
                "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-q4_k_m.gguf", // seems to be broken...
                // "{}/.cache/hf/models--HuggingFaceTB--SmolLM2-135M-Instruct/smollm2-135m-instruct-f16.gguf",
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
        #[cfg(all(feature = "api", not(feature = "candle")))]
        {
            candle_chat_config.candle_asset = None;
            candle_chat_config.openai_asset = Some(AvailableOpenAIAssets::MetaLlamaV3p2_1B);
            candle_chat_config.weights_config_file = None;
            candle_chat_config.weights_file = None;
            candle_chat_config.tokenizer_file = None;
            candle_chat_config.tokenizer_config_file = None;
            candle_chat_config.api_url = self.chat_api_url.map(|s| s.to_string());
        }
        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config).unwrap();
        let config = TableBuilder::new()
            .with_name(self.chat_processor_name)
            .with_json(&candle_chat_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Message aggregator config
        let aggregator_config = DataConfig {
            lhs_values: Some(vec!["timestamp".to_string()]),
            asc: Some(true),
            operator: AvailableCandleOperators::Sort,
            ..Default::default()
        };
        let aggregator_config_json = serde_json::to_vec(&aggregator_config).unwrap();
        let aggregator_1_state = TableBuilder::new()
            .with_name(self.message_aggregator_processor_1_name)
            .with_json(&aggregator_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();
        let aggregator_2_state = TableBuilder::new()
            .with_name(self.message_aggregator_processor_2_name)
            .with_json(&aggregator_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        Some(vec![
            config,
            aggregator_1_state,
            aggregator_2_state,
            AvailableSubjects::Messages
                .to_table(Some(self.chat_task_name), None)
                .unwrap(),
            AvailableInterfaceSubjects::UserMessages
                .to_table(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::AssistantMessages
                .to_table(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::AggregatedMessages
                .to_table(None, None)
                .unwrap(),
        ])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        BuildableTrait, ChatBuilderTraitExt, IPCMessage, MappableTrait, MessageBuilderTrait,
        MessageTrait, TableTrait,
    };
    use phymes_diagnostics::HashMap;

    use crate::{SessionContextBuilderAgentsTrait, SessionStream, create_message_map};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_chat_agent_session() -> Result<()> {
        // initialize the session
        let chat_agent_session = ChatAgentSession::default();
        let session_ctx = chat_agent_session
            .build()
            .with_name(chat_agent_session.session_context_name)
            .add_session_interface(None)?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(RwLock::new(session_ctx));

        // Skip actually running the session as it takes too long on the CPU
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // ----- Query #1 -----
            let chat = AvailableInterfaceSubjects::UserMessages
                .to_table_builder(None)
                .append_new_user_query_str(
                    "Write a function to count prime numbers up to N.",
                    "user",
                )?
                .build()?;
            let message = IPCMessage::get_builder()
                .with_message(chat.to_ipc_stream()?)
                .with_subject(chat.get_name())
                .with_update(&TablePublication::Extend {
                    table_name: chat.get_name().to_string(),
                })
                .with_publisher(chat_agent_session.session_context_name)
                .make_name()?
                .build()?;
            let incoming_message_map = create_message_map(vec![message]);
            let session_stream =
                SessionStream::new(incoming_message_map, Arc::clone(&session_ctx_arc));
            let mut response: Vec<HashMap<String, IPCMessage>> =
                session_stream.try_collect().await?;

            // Update the chat history with the response
            let bytes = response
                .iter_mut()
                .filter_map(|map| {
                    map.remove(&format!(
                        "from_{}_on_{}",
                        chat_agent_session.session_context_name,
                        AvailableInterfaceSubjects::AssistantMessages
                    ))
                    .map(|v| v.get_message_own())
                })
                .flatten()
                .collect::<Vec<_>>();
            let json_data = TableBuilder::new_from_ipc_stream(&bytes)?
                .with_name("")
                .build()?
                .to_json_object()?;
            for row in &json_data {
                if row["role"] != "system" {
                    println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
                }
            }

            // for metric in metrics.clone_inner().iter() {
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == "chat_task_1"
            //     {
            //         assert_eq!(metric.value().as_usize(), 2);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == "chat_processor_1"
            //     {
            //         assert!(metric.value().as_usize() >= 1);
            //     }
            // }

            assert_eq!(json_data.first().unwrap().get("role").unwrap(), "assistant");
            assert!(json_data.first().unwrap().get("content").is_some());

            // ----- Query #2 -----
            let chat = AvailableInterfaceSubjects::UserMessages
                .to_table_builder(None)
                .append_new_user_query_str(
                    "Please provide an example using the functions.",
                    "user",
                )?
                .build()?;
            let message = IPCMessage::get_builder()
                .with_message(chat.to_ipc_stream()?)
                .with_subject(chat.get_name())
                .with_update(&TablePublication::Extend {
                    table_name: chat.get_name().to_string(),
                })
                .with_publisher(chat_agent_session.session_context_name)
                .make_name()?
                .build()?;
            let incoming_message_map = create_message_map(vec![message]);
            let session_stream =
                SessionStream::new(incoming_message_map, Arc::clone(&session_ctx_arc));
            let mut response: Vec<HashMap<String, IPCMessage>> =
                session_stream.try_collect().await?;

            // Update the chat history with the response
            let bytes = response
                .iter_mut()
                .filter_map(|map| {
                    map.remove(&format!(
                        "from_{}_on_{}",
                        chat_agent_session.session_context_name,
                        AvailableInterfaceSubjects::AssistantMessages
                    ))
                    .map(|v| v.get_message_own())
                })
                .flatten()
                .collect::<Vec<_>>();
            let json_data = TableBuilder::new_from_ipc_stream(&bytes)?
                .with_name("")
                .build()?
                .to_json_object()?;
            for row in &json_data {
                if row["role"] != "system" {
                    println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
                }
            }

            // for metric in metrics.clone_inner().iter() {
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == "chat_task_1"
            //         && metric.value().as_usize() != 2
            //     {
            //         assert_eq!(metric.value().as_usize(), 4);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == "chat_processor_1"
            //     {
            //         assert!(metric.value().as_usize() >= 1);
            //     }
            // }

            assert_eq!(json_data.first().unwrap().get("role").unwrap(), "assistant");
            assert!(json_data.first().unwrap().get("content").is_some());
        }

        Ok(())
    }
}
