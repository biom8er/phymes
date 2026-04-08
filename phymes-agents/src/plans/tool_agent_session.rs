use std::sync::Arc;

use anyhow::Result;
use serde_json::json;

use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableSubscribeEvents, BuildableTrait,
    BuilderTrait, DataEncoding, DataFormat, ProcessorPlan, ProcessorPlanBuilder, Publication,
    RuntimeEnv, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait,
    Subscription, create_schema_from_fields, create_tools_record_batch,
};
use phymes_data::{
    AvailableOperators, AvailableJinja2Templates, DataCastOperator, DataColumnOperator,
    DataConfig, DataStreamManager, ToolTrait,
};
#[cfg(all(not(feature = "candle"), feature = "api"))]
use phymes_ml::AvailableOpenAIAssets;
use phymes_ml::{AvailableCandleAssets, CandleChatConfig};

use arrow::datatypes::{DataType, Field, Fields};

use crate::{AvailableInterfaceSubjects, AvailableProcessors, CustomAgentsBuilderTrait, TaskPlan};

/// Tool agent node with human-in-the-loop
pub struct ToolAgentSession<'a> {
    /// Text generation inference capabilities (i.e, the agent)
    pub chat_task_name: &'a str,
    pub chat_processor_name: &'a str, // also used as the config name
    pub chat_runtime_env_name: &'a str,
    /// Structured text generation inference parser
    pub message_parser_task_name: &'a str,
    pub message_parser_processor_name: &'a str,
    /// Message aggregators for the chat task and another for the UI
    pub message_aggregator_task_1_name: &'a str,
    pub message_aggregator_processor_1_name: &'a str,
    pub message_aggregator_task_2_name: &'a str,
    pub message_aggregator_processor_2_name: &'a str,
    pub message_aggregator_runtime_env_name: &'a str,
    /// Attachment aggregator for the tool task
    pub attachment_aggregator_task_name: &'a str,
    pub attachment_aggregator_processor_name: &'a str,
    pub attachment_aggregator_runtime_env_name: &'a str,
    /// Extract tabular data from the user attachments
    pub extract_tabular_data_task_name: &'a str,
    pub extract_tabular_data_processor_name: &'a str,
    /// The tool node (one of the CandleOps i.e., sort op)
    pub tool_task_name: &'a str,
    pub tool_processor_name: &'a str,
    pub tool_runtime_env_name: &'a str,
    /// Create the attachment for the user
    pub tool_attachment_task_name: &'a str,
    pub tool_attachment_processor_name: &'a str,
    pub tool_visualization_task_name: &'a str,
    pub tool_vis_renamecols_processor_name: &'a str,
    pub tool_vis_xychart_processor_name: &'a str,
    /// Summarize the tool node results for the chat node
    pub tool_summary_task_name: &'a str,
    pub tool_summary_processor_name: &'a str,
    /// Human in the loop tool node
    pub hitl_task_name: &'a str,
    pub hitl_processor_name: &'a str,
    pub hitl_summary_processor_name: &'a str,
    /// Session and state
    pub session_context_name: &'a str,
    pub state_tools_table_name: &'a str,
    pub state_scores_table_name: &'a str,
    pub chat_api_url: Option<&'a str>,
}

impl Default for ToolAgentSession<'_> {
    fn default() -> Self {
        ToolAgentSession {
            session_context_name: "session_context_1",
            chat_processor_name: "chat_processor_1",
            chat_task_name: "chat_task_1",
            chat_runtime_env_name: "chat_rt_1",
            extract_tabular_data_task_name: "extract_tabular_data_task_1",
            extract_tabular_data_processor_name: "extract_tabular_data_processor_1",
            // Needs to match the operator name
            tool_task_name: "Sort",
            // Needs to match the operator name
            tool_processor_name: "Sort",
            tool_runtime_env_name: "tool_rt_1",
            tool_attachment_task_name: "tool_attachment_task_1",
            tool_attachment_processor_name: "tool_attachment_processor_1",
            tool_visualization_task_name: "tool_visualization_task_1",
            tool_vis_renamecols_processor_name: "tool_vis_renamecols_processor_1",
            tool_vis_xychart_processor_name: "tool_visualization_processor_1",
            tool_summary_task_name: "tool_summary_task_1",
            tool_summary_processor_name: "tool_summary_processor_1",
            // Needs to match the operator name
            hitl_task_name: "HumanInTheLoop",
            // Needs to match the operator name
            hitl_processor_name: "HumanInTheLoop",
            hitl_summary_processor_name: "hitl_summary_processor_1",
            message_parser_task_name: "message_parser_task_1",
            message_parser_processor_name: "message_parser_processor_1",
            message_aggregator_task_1_name: "message_aggregator_task_1",
            message_aggregator_processor_1_name: "message_aggregator_processor_1",
            message_aggregator_task_2_name: "message_aggregator_task_2",
            message_aggregator_processor_2_name: "message_aggregator_processor_2",
            message_aggregator_runtime_env_name: "message_aggregator_rt_1",
            attachment_aggregator_task_name: "attachment_aggregator_task_1",
            attachment_aggregator_processor_name: "attachment_aggregator_processor_1",
            attachment_aggregator_runtime_env_name: "attachment_aggregator_rt_1",
            state_scores_table_name: "available_data_1",
            state_tools_table_name: "tools",
            chat_api_url: Some("http://0.0.0.0:8000/v1"),
        }
    }
}

impl<'a> ToolAgentSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        ToolAgentSession {
            session_context_name,
            ..Default::default()
        }
    }
    pub fn make_tools_table(&self) -> Result<Subject> {
        let tool_ids = vec![
            AvailableOperators::Sort.to_string(),
            AvailableOperators::HumanInTheLoop.to_string(),
        ];
        let tools = vec![
            AvailableOperators::Sort.to_json_tool_schema(),
            AvailableOperators::HumanInTheLoop.to_json_tool_schema(),
        ];
        let batch = create_tools_record_batch(tool_ids, tools)?;
        SubjectBuilder::new()
            .with_name(self.state_tools_table_name)
            .with_record_batches(vec![batch])?
            .build()
    }
}

impl CustomAgentsBuilderTrait for ToolAgentSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        Some(vec![
            // DM: `Reqwest` connections break prematurely in `OpenAIChatProcessor`
            //  when chained or nested within other streams
            // DM: another tool agent session publish/subscribe network needs to be
            //  made for api access that breaks down the chat task into seperate
            //  tasks for each processor...
            TaskPlan {
                task_name: self.message_aggregator_task_1_name.to_string(),
                processor_names: vec![self.message_aggregator_processor_1_name.to_string()],
            },
            TaskPlan {
                task_name: self.message_aggregator_task_2_name.to_string(),
                processor_names: vec![self.message_aggregator_processor_2_name.to_string()],
            },
            TaskPlan {
                task_name: self.attachment_aggregator_task_name.to_string(),
                processor_names: vec![self.attachment_aggregator_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.tool_visualization_task_name.to_string(),
                processor_names: vec![
                    self.tool_vis_renamecols_processor_name.to_string(),
                    self.tool_vis_xychart_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.chat_task_name.to_string(),
                processor_names: vec![self.chat_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.message_parser_task_name.to_string(),
                processor_names: vec![self.message_parser_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.extract_tabular_data_task_name.to_string(),
                processor_names: vec![self.extract_tabular_data_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.tool_task_name.to_string(),
                processor_names: vec![self.tool_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.tool_attachment_task_name.to_string(),
                processor_names: vec![self.tool_attachment_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.tool_summary_task_name.to_string(),
                processor_names: vec![self.tool_summary_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.hitl_task_name.to_string(),
                processor_names: vec![
                    self.hitl_processor_name.to_string(),
                    self.hitl_summary_processor_name.to_string(),
                ],
            },
        ])
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::AggregatorProcessor
                        .build_arc(self.message_aggregator_processor_1_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.chat_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    // Subscription::OnUpdateAllRecordBatches {
                    //     subject_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                    // },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                    },
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
                    },
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableSubjects::SessionErrors.to_string(),
                    },
                    // Subscription::OnUpdateAllRecordBatches {
                    //     subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    // },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.message_aggregator_processor_1_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    // AvailableTableSubscribePolicies::AnyTableNameSubscribe.build(),
                    AvailableSubscribeEvents::ChatContentSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::AggregatorProcessor
                        .build_arc(self.message_aggregator_processor_2_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: AvailableInterfaceSubjects::AggregatedMessages.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                    },
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.message_aggregator_processor_2_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AnySubjectNameSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::AggregatorProcessor
                        .build_arc(self.attachment_aggregator_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: AvailableInterfaceSubjects::AggregatedAttachments.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::UserCsv.to_string(),
                    },
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::AssistantCsv.to_string(),
                    },
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::AssistantScript.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.attachment_aggregator_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AnySubjectNameSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select.build_arc(self.tool_vis_renamecols_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: AvailableSubjects::MermaidXYChart.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.tool_summary_task_name.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.tool_vis_renamecols_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ApplyTemplate
                        .build_arc(self.tool_vis_xychart_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: AvailableInterfaceSubjects::AssistantScript.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: AvailableSubjects::MermaidXYChart.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.tool_vis_xychart_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            #[cfg(all(feature = "api", not(feature = "candle")))]
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::OpenAIChatProcessor.build_arc(self.chat_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.message_parser_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.chat_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_tools_table_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.chat_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            #[cfg(feature = "candle")]
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::CandleChatProcessor.build_arc(self.chat_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.message_parser_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.chat_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_tools_table_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.chat_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::MessageParserProcessor
                        .build_arc(self.message_parser_processor_name),
                )
                .with_publications(&[
                    Publication::Extend {
                        // The first publication is the default publish target
                        // subject_name: self.chat_task_name.to_string(),
                        subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    },
                    Publication::Extend {
                        subject_name: self.tool_task_name.to_string(),
                    },
                    Publication::Extend {
                        subject_name: self.hitl_task_name.to_string(),
                    },
                ])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.message_parser_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.message_parser_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ExtractTabular
                        .build_arc(self.extract_tabular_data_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.state_scores_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableInterfaceSubjects::UserCsv.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.extract_tabular_data_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::CandleDataProcessor.build_arc(self.tool_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.tool_summary_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: self.tool_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_scores_table_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::CandleDataProcessor.build_arc(self.hitl_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                }])
                .with_subscriptions(&[Subscription::OnUpdateLastRecordBatch {
                    subject_name: self.hitl_task_name.to_string(),
                }])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::PackTabular.build_arc(self.tool_attachment_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: AvailableInterfaceSubjects::AssistantCsv.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.tool_attachment_processor_name.to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.tool_summary_task_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::PackTabular.build_arc(self.tool_summary_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: AvailableInterfaceSubjects::ToolMessages.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.tool_summary_processor_name.to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.tool_summary_task_name.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::PackTabular.build_arc(self.hitl_summary_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.hitl_summary_processor_name.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    },
                ])
                .with_subscribe_policy(AvailableSubscribeEvents::AllSubjectNamesSubscribe.build())
                .build()
                .unwrap(),
        ];
        Some(processors)
    }

    fn make_runtime_env(&self) -> Option<Arc<RuntimeEnv>> {
        Some(
            RuntimeEnv::get_builder()
                .with_name(self.session_context_name)
                .build_arc()
                .unwrap(),
        )
    }

    fn make_subjects(&self) -> Option<Vec<SubjectPlan>> {
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
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--Qwen--Qwen2-0.5B-Instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(AvailableCandleAssets::QwenV2p5_1p5bChat),
            ..Default::default()
        };

        // Add hf_hub if available
        #[cfg(feature = "hf_hub")]
        {
            candle_chat_config.candle_asset = Some(AvailableCandleAssets::QwenV2p5_3bChat);
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
            // DM: Bug in Llama model system template that requires it to only call tools instead of respond...
            // see update template <https://gist.github.com/K-Mistele/820d142b4dab50bd8ef0c7bbcad4515c>
            // see discussion when using vLLM <https://github.com/vllm-project/vllm/issues/9991>
            candle_chat_config.openai_asset = Some(AvailableOpenAIAssets::MetaLlamaV3p2_1B);
            candle_chat_config.weights_config_file = None;
            candle_chat_config.weights_file = None;
            candle_chat_config.tokenizer_file = None;
            candle_chat_config.tokenizer_config_file = None;
            candle_chat_config.api_url = self.chat_api_url.map(|s| s.to_string());
        }

        // Message Parser config
        candle_chat_config.messages = self.message_parser_task_name.to_string();
        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config).unwrap();
        let candle_message_parser_state = SubjectBuilder::new()
            .with_name(self.message_parser_processor_name)
            .with_json(&candle_chat_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Chat config
        candle_chat_config.messages = self.chat_task_name.to_string();
        candle_chat_config.tools = Some(self.state_tools_table_name.to_string());
        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config).unwrap();
        let candle_chat_state = SubjectBuilder::new()
            .with_name(self.chat_processor_name)
            .with_json(&candle_chat_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Message aggregator config
        let aggregator_config = DataConfig {
            lhs_values: Some(vec!["timestamp".to_string()]),
            asc: Some(true),
            operator: AvailableOperators::Sort,
            ..Default::default()
        };
        let aggregator_config_json = serde_json::to_vec(&aggregator_config).unwrap();
        let aggregator_1_state = SubjectBuilder::new()
            .with_name(self.message_aggregator_processor_1_name)
            .with_json(&aggregator_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let aggregator_2_state = SubjectBuilder::new()
            .with_name(self.message_aggregator_processor_2_name)
            .with_json(&aggregator_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let aggregator_3_state = SubjectBuilder::new()
            .with_name(self.attachment_aggregator_processor_name)
            .with_json(&aggregator_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Extract tabular data config
        let extract_tabular_data_config = DataConfig {
            lhs_name: Some(AvailableInterfaceSubjects::UserCsv.to_string()),
            lhs_values: Some(vec!["bytes".to_string()]),
            format: Some(DataFormat::CsvDefault),
            schema: Some(AvailableSubjects::Empty), //DM: not used for CSV
            operator: AvailableOperators::ExtractTabular,
            encoding: Some(DataEncoding::None),
            ..Default::default()
        };
        let extract_tabular_data_config_json =
            serde_json::to_vec(&extract_tabular_data_config).unwrap();
        let extract_tabular_data_state = SubjectBuilder::new()
            .with_name(self.extract_tabular_data_processor_name)
            .with_json(&extract_tabular_data_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Select and cast config
        let vis_renamecols_config = DataConfig {
            lhs_name: Some(self.tool_summary_task_name.to_string()),
            lhs_values: Some(vec!["lhs_pk".to_string(), "score".to_string()]),
            rhs_values: Some(vec!["".to_string(), "".to_string()]),
            as_columns: Some(vec!["x".to_string(), "y".to_string()]),
            column_operators: Some(vec![DataColumnOperator::None, DataColumnOperator::None]),
            cast_operators: Some(vec![DataCastOperator::None, DataCastOperator::None]),
            cast_datatypes: Some(vec![
                DataType::Utf8.to_string(),
                DataType::Float32.to_string(),
            ]),
            cast_templates: Some(vec!["".to_string(), "".to_string()]),
            operator: AvailableOperators::Select,
            ..Default::default()
        };
        let vis_renamecols_config_json = serde_json::to_vec(&vis_renamecols_config).unwrap();
        let vis_renamecols_state = SubjectBuilder::new()
            .with_name(self.tool_vis_renamecols_processor_name)
            .with_json(&vis_renamecols_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Visualize tabular data config
        let vis_xychart_config = DataConfig {
            lhs_name: Some(AvailableSubjects::MermaidXYChart.to_string()),
            doc_template: Some(AvailableJinja2Templates::MermaidXYChartHTML),
            doc_name: Some(self.state_scores_table_name.to_string()),
            doc_input: Some(
                serde_json::to_string(&json!({
                "title": self.state_scores_table_name,
                "x_title": "lhs_pk",
                "y_title": "score"}))
                .unwrap(),
            ),
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::Html),
            schema: Some(AvailableSubjects::Attachments),
            operator: AvailableOperators::ApplyTemplate,
            ..Default::default()
        };
        let vis_xychart_config_json = serde_json::to_vec(&vis_xychart_config).unwrap();
        let vis_xychart_state = SubjectBuilder::new()
            .with_name(self.tool_vis_xychart_processor_name)
            .with_json(&vis_xychart_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Attachment config
        let attachment_config = DataConfig {
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::CsvDefault),
            schema: Some(AvailableSubjects::Attachments),
            cpu: false,
            lhs_stream: DataStreamManager::Accumulate,
            operator: AvailableOperators::PackTabular,
            lhs_name: Some(self.tool_summary_task_name.to_string()),
            doc_name: Some(self.tool_summary_task_name.to_string()),
            ..Default::default()
        };
        let attachmen_config_json = serde_json::to_vec(&attachment_config).unwrap();
        let attachmen_state = SubjectBuilder::new()
            .with_name(self.tool_attachment_processor_name)
            .with_json(&attachmen_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Summary config
        let summary_config = DataConfig {
            encoding: Some(DataEncoding::None),
            operator: AvailableOperators::PackTabular,
            format: Some(DataFormat::None),
            schema: Some(AvailableSubjects::Messages),
            cpu: false,
            lhs_stream: DataStreamManager::Accumulate,
            lhs_name: Some(self.tool_summary_task_name.to_string()),
            doc_name: Some(self.tool_summary_task_name.to_string()),
            ..Default::default()
        };
        let summary_config_json = serde_json::to_vec(&summary_config).unwrap();
        let summary_state_1 = SubjectBuilder::new()
            .with_name(self.tool_summary_processor_name)
            .with_json(&summary_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();
        let summary_config = DataConfig {
            encoding: Some(DataEncoding::None),
            operator: AvailableOperators::PackTabular,
            format: Some(DataFormat::None),
            schema: Some(AvailableSubjects::Messages),
            cpu: false,
            lhs_stream: DataStreamManager::Accumulate,
            lhs_name: Some(AvailableInterfaceSubjects::AssistantMessages.to_string()),
            doc_name: Some(AvailableInterfaceSubjects::AssistantMessages.to_string()),
            ..Default::default()
        };
        let summary_config_json = serde_json::to_vec(&summary_config).unwrap();
        let summary_state_2 = SubjectBuilder::new()
            .with_name(self.hitl_summary_processor_name)
            .with_json(&summary_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Scores and summary table schemas
        fn create_scores_fields() -> Fields {
            let fields_vec = vec![
                Field::new("lhs_pk", DataType::Utf8, false),
                Field::new("score", DataType::Float64, false),
            ];
            Fields::from(fields_vec)
        }
        let scores_table = Subject::get_builder()
            .with_name(self.state_scores_table_name)
            .with_schema(create_schema_from_fields(&create_scores_fields))
            .with_record_batches(Vec::new())
            .unwrap()
            .build()
            .unwrap();
        let tool_summary_table = Subject::get_builder()
            .with_name(self.tool_summary_task_name)
            .with_schema(create_schema_from_fields(&create_scores_fields))
            .with_record_batches(Vec::new())
            .unwrap()
            .build()
            .unwrap();

        let subjects = vec![
            candle_chat_state,
            candle_message_parser_state,
            aggregator_1_state,
            aggregator_2_state,
            aggregator_3_state,
            vis_renamecols_state,
            vis_xychart_state,
            extract_tabular_data_state,
            attachmen_state,
            summary_state_1,
            summary_state_2,
            scores_table,
            tool_summary_table,
            AvailableSubjects::MermaidXYChart
                .to_subject(None, None)
                .unwrap(),
            self.make_tools_table().unwrap(),
            AvailableInterfaceSubjects::AggregatedMessages
                .to_subject(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::UserMessages
                .to_subject(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::UserCsv
                .to_subject(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::AssistantMessages
                .to_subject(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::ToolMessages
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::Messages
                .to_subject(Some(self.chat_task_name), None)
                .unwrap(),
            AvailableSubjects::Messages
                .to_subject(Some(self.message_parser_task_name), None)
                .unwrap(),
            AvailableSubjects::Bytes
                .to_subject(Some(self.tool_task_name), None)
                .unwrap(),
            AvailableSubjects::Bytes
                .to_subject(Some(self.hitl_task_name), None)
                .unwrap(),
            AvailableInterfaceSubjects::AssistantCsv
                .to_subject(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::AggregatedAttachments
                .to_subject(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::AssistantScript
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::SessionErrors
                .to_subject(None, None)
                .unwrap(),
        ];
        let subject_plans = subjects
            .into_iter()
            .map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap())
            .collect::<Vec<_>>();
        Some(subject_plans)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::TryStreamExt;
    use phymes_core::{
        AttachmentBuilderTraitExt, ChatBuilderTraitExt, CsvFormat, IPCMessage, MappableTrait,
        MessageBuilderTrait, MessageTrait, SubjectTrait,
    };
    use phymes_data::test_extract_tabular_data::make_scores_table;
    use phymes_diagnostics::HashMap;

    use crate::{SessionContextBuilderAgentsTrait, SessionStream, create_message_map};

    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_tool_agent_session() -> Result<()> {
        // initialize the session
        let tool_agent_session = ToolAgentSession::default();
        let (session_ctx, session_messages) = tool_agent_session
            .build()
            .with_name(tool_agent_session.session_context_name)
            .add_session_interface(None)?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Make the tabular data
        let csv_format = CsvFormat::default();
        let tabular_data = make_scores_table()?;
        let bytes = tabular_data.to_csv(csv_format.delimiter, csv_format.header)?;

        // Wrap into the message
        let chat = AvailableInterfaceSubjects::UserMessages.to_subject_builder(None)
            .append_new_user_query_str("Sort a list of scores in ascending order. The lhs_name is `available_data_1`, the lhs_pk is `lhs_pk` and the lhs_values is `score`.", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&Publication::Extend {
                subject_name: chat.get_name().to_string(),
            })
            .with_publisher(tool_agent_session.session_context_name)
            .make_name()?
            .build()?;
        let blob = AvailableInterfaceSubjects::UserCsv
            .to_subject_builder(None)
            .with_attachment(None, Some("csv"), &bytes, None)?
            .build()?;
        let blob_message = IPCMessage::get_builder()
            .with_message(blob.to_ipc_stream()?)
            .with_subject(blob.get_name())
            .with_update(&Publication::Extend {
                subject_name: blob.get_name().to_string(),
            })
            .with_publisher(tool_agent_session.session_context_name)
            .make_name()?
            .build()?;
        let message_map = create_message_map(vec![chat_message, blob_message]);
        let _ = session_ctx_arc
            .update_subjects_from_messages(session_messages.unwrap_or_default(), 0)
            .await;

        // Avoid running with Candle without GPU acceleration
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
            let mut response: Vec<HashMap<String, IPCMessage>> =
                session_stream.try_collect().await?;

            // Update the chat history with the response
            let bytes = response
                .iter_mut()
                .filter_map(|map| {
                    map.remove(&format!(
                        "from_{}_on_{}",
                        tool_agent_session.session_context_name,
                        AvailableInterfaceSubjects::AssistantMessages
                    ))
                    .map(|v| v.get_message_own())
                })
                .flatten()
                .collect::<Vec<_>>();
            let json_data = SubjectBuilder::new_from_ipc_stream(&bytes)?
                .with_name("")
                .build()?
                .to_json_object()?;
            for row in &json_data {
                if row["role"] != "system" {
                    println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
                }
            }

            let bytes = response
                .iter_mut()
                .filter_map(|map| {
                    map.remove(&format!(
                        "from_{}_on_{}",
                        tool_agent_session.session_context_name,
                        AvailableInterfaceSubjects::AssistantCsv
                    ))
                    .map(|v| v.get_message_own())
                })
                .flatten()
                .collect::<Vec<_>>();
            let attachment_data = SubjectBuilder::new_from_ipc_stream(&bytes)?
                .with_name("")
                .build()?
                .to_json_object()?;
            for row in &attachment_data {
                let bytes = row["bytes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect::<Vec<u8>>();
                println!(
                    "attachment {}.{}: {}",
                    row["filename"],
                    row["extension"],
                    String::from_utf8_lossy(bytes.as_ref()).into_owned()
                )
            }

            // for metric in metrics.clone_inner().iter() {
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == tool_agent_session.chat_processor_name
            //     {
            //         assert!(metric.value().as_usize() > 0);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap()
            //             == tool_agent_session.message_parser_processor_name
            //     {
            //         assert!(metric.value().as_usize() > 0 || metric.value().as_usize() == 1);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == tool_agent_session.tool_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 3);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap()
            //             == tool_agent_session.tool_summary_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 1);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap()
            //             == tool_agent_session.tool_attachment_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 1);
            //     }
            // }

            // DM: Bug in Llama model system template that requires it to only call tools instead of respond...
            let roles = ["assistant", "tool"];
            assert!(
                roles.contains(
                    &json_data
                        .first()
                        .unwrap()
                        .get("role")
                        .unwrap()
                        .as_str()
                        .unwrap()
                )
            );
            assert!(json_data.first().unwrap().get("content").is_some());
            assert!(attachment_data.first().unwrap().get("bytes").is_some());
            assert!(attachment_data.first().unwrap().get("filename").is_some());
            assert!(attachment_data.first().unwrap().get("extension").is_some());
        }

        Ok(())
    }
}
