use std::sync::Arc;

use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, AvailableSubscribeEvents, BuildableTrait, BuilderTrait, DataEncoding, DataFormat, ProcessorPlan, ProcessorPlanBuilder, Publication, RuntimeEnv, Subject, SubjectBuilder, SubjectBuilderTrait, SubjectPlan, SubjectPlanBuilderTrait, Subscription, create_schema_from_fields
};
use phymes_data::{
    AvailableCandleOperators, DataCastOperator, DataColumnOperator, DataConfig,
    DataDistanceOperator, DataJoinOperator, DataStreamManager, LimitConfig,
};
#[cfg(feature = "api")]
use phymes_ml::AvailableOpenAIAssets;
use phymes_ml::{AvailableCandleAssets, CandleChatConfig, CandleEmbedConfig};

use arrow::datatypes::{DataType, Field, Fields, SchemaRef};

use crate::{AvailableInterfaceSubjects, AvailableProcessors, CustomAgentsBuilderTrait, TaskPlan};

/// Document Retrieval Augmented Generation (RAG) session plan.
///
/// # Notes
///
/// * The embedding size must be specified before which is determined by the size of
///   the hidden layer of the embedding model
pub struct DocumentRAGSession<'a> {
    /// Chat tasks
    pub chat_task_name: &'a str,
    pub chat_processor_name: &'a str,
    pub chat_runtime_env_name: &'a str,
    // DM: needed for openai api since we cannot chain streams
    pub message_aggregator_task_1_name: &'a str,
    pub message_aggregator_processor_1_name: &'a str,
    pub message_aggregator_task_2_name: &'a str,
    pub message_aggregator_processor_2_name: &'a str,
    /// Attachment aggregator for the tool task
    pub attachment_aggregator_task_name: &'a str,
    pub attachment_aggregator_processor_name: &'a str,
    pub attachment_aggregator_runtime_env_name: &'a str,
    /// Embed tasks
    pub message_to_query_task_name: &'a str,
    pub message_to_query_processor_name: &'a str,
    pub embed_query_task_name: &'a str,
    pub embed_documents_task_name: &'a str,
    pub embed_query_processor_name: &'a str,
    pub embed_documents_processor_name: &'a str,
    /// Extract PDF task
    pub extract_pdf_task_name: &'a str,
    pub extract_pdf_processor_name: &'a str,
    /// Chunk documents task
    pub document_chunk_task_name: &'a str,
    pub document_chunk_processor_name: &'a str,
    // DM: Two embed runtimes are needed for embedded Candle models due to edge cases
    //   where the mutexes are access simultaneously. Set the embed runtimes to
    //   the same name when using OpenAI API
    pub embed_documents_runtime_env_name: &'a str,
    pub embed_query_runtime_env_name: &'a str,
    /// Vector search tasks
    pub vector_search_task_name: &'a str,
    pub relative_similarity_processor_name: &'a str,
    pub sort_scores_processor_name: &'a str,
    pub join_chunks_processor_name: &'a str,
    pub top_k_select_processor_name: &'a str,
    pub top_k_limit_processor_name: &'a str,
    pub top_k_summary_processor_name: &'a str,
    pub vector_search_runtime_env_name: &'a str,
    /// Session and state
    pub session_context_name: &'a str,
    pub state_documents_table_name: &'a str,
    pub state_doc_embed_table_name: &'a str,
    pub state_q_embed_table_name: &'a str,
    pub state_top_k_select_docs_table_name: &'a str,
    pub state_top_k_limit_docs_table_name: &'a str,
    pub state_top_k_summary_docs_table_name: &'a str,
    pub state_scores_table_name: &'a str,
    pub state_scores_chunks_join_table_name: &'a str,
    /// Other parameters
    pub chat_api_url: Option<&'a str>,
    pub embed_api_url: Option<&'a str>,
}

impl Default for DocumentRAGSession<'_> {
    fn default() -> Self {
        Self {
            chat_task_name: "chat_task_1",
            message_aggregator_task_1_name: "message_aggregator_task_1",
            message_aggregator_processor_1_name: "message_aggregator_1",
            message_aggregator_task_2_name: "message_aggregator_task_2",
            message_aggregator_processor_2_name: "message_aggregator_2",
            attachment_aggregator_task_name: "attachment_aggregator_task_1",
            attachment_aggregator_processor_name: "attachment_aggregator_processor_1",
            attachment_aggregator_runtime_env_name: "attachment_aggregator_rt_1",
            chat_processor_name: "chat_processor_1",
            chat_runtime_env_name: "chat_rt_1",
            message_to_query_task_name: "message_to_query_task_1",
            message_to_query_processor_name: "message_to_query_processor_1",
            embed_query_task_name: "embed_query_task_1",
            embed_documents_task_name: "embed_documents_task_1",
            embed_query_processor_name: "embed_query_processor_1",
            embed_documents_processor_name: "embed_documents_processor_1",
            extract_pdf_task_name: "extract_pdf_task_1",
            extract_pdf_processor_name: "extract_pdf_processor_1",
            document_chunk_task_name: "chunk_documents_task_1",
            document_chunk_processor_name: "chunk_documents_processor_1",
            embed_documents_runtime_env_name: "embed_documents_rt_1",
            embed_query_runtime_env_name: "embed_query_rt_1", // "embed_documents_rt_1",
            vector_search_task_name: "vs_task_1",
            relative_similarity_processor_name: "rel_sim_processor_1",
            sort_scores_processor_name: "sort_scores_processor_1",
            join_chunks_processor_name: "join_scores_chunks_processor_1",
            top_k_select_processor_name: "top_k_select_processor_1",
            top_k_limit_processor_name: "top_k_limit_processor_1",
            top_k_summary_processor_name: "top_k_summary_processor_1",
            vector_search_runtime_env_name: "vs_rt_1",
            session_context_name: "session_context_1",
            state_documents_table_name: "documents",
            state_doc_embed_table_name: "doc_embeddings",
            state_q_embed_table_name: "q_embeddings",
            state_top_k_select_docs_table_name: "top_k_select",
            state_top_k_limit_docs_table_name: "top_k_limit",
            state_top_k_summary_docs_table_name: "top_k_summary",
            state_scores_table_name: "tmp_scores",
            state_scores_chunks_join_table_name: "tmp_scores_chunks_join",
            chat_api_url: None,
            embed_api_url: None,
        }
    }
}

impl<'a> DocumentRAGSession<'a> {
    pub fn new_with_session_name(session_context_name: &'a str) -> Self {
        DocumentRAGSession {
            session_context_name,
            ..Default::default()
        }
    }
}

impl CustomAgentsBuilderTrait for DocumentRAGSession<'_> {
    fn make_task_plans(&self) -> Option<Vec<TaskPlan>> {
        // DM: `Reqwest` connections break prematurely in `OpenAIChatProcessor`
        //  when chained or nested within other streams.
        let tasks = vec![
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
                task_name: self.message_to_query_task_name.to_string(),
                processor_names: vec![self.message_to_query_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.chat_task_name.to_string(),
                processor_names: vec![self.chat_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.extract_pdf_task_name.to_string(),
                processor_names: vec![
                    self.extract_pdf_processor_name.to_string(),
                    self.document_chunk_processor_name.to_string(),
                ],
            },
            TaskPlan {
                task_name: self.embed_documents_task_name.to_string(),
                processor_names: vec![self.embed_documents_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.embed_query_task_name.to_string(),
                processor_names: vec![self.embed_query_processor_name.to_string()],
            },
            TaskPlan {
                task_name: self.vector_search_task_name.to_string(),
                processor_names: vec![
                    self.relative_similarity_processor_name.to_string(),
                    self.sort_scores_processor_name.to_string(),
                    self.join_chunks_processor_name.to_string(),
                    self.top_k_select_processor_name.to_string(),
                    self.top_k_limit_processor_name.to_string(),
                    self.top_k_summary_processor_name.to_string(),
                ],
            },
        ];

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<ProcessorPlan>> {
        // The order is the order in which the processors are called in the task
        let processors = vec![
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::MessageAggregatorProcessor
                        .build_arc(self.message_aggregator_processor_1_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.chat_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                    },
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.state_top_k_summary_docs_table_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.message_aggregator_processor_1_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::MessageAggregatorProcessor
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
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::AttachmentAggregatorProcessor
                        .build_arc(self.attachment_aggregator_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: AvailableInterfaceSubjects::AggregatedAttachments.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.attachment_aggregator_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AnySubjectNameSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select.build_arc(self.message_to_query_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: AvailableInterfaceSubjects::UserQueries.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                    },
                    Subscription::AlwaysLastRecordBatch {
                        subject_name: self.message_to_query_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            #[cfg(all(feature = "api", not(feature = "candle")))]
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::OpenAIChatProcessor.build_arc(self.chat_processor_name),
                )
                .with_publications(&[Publication::ExtendChunks {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    col_name: "content".to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.chat_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.chat_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            #[cfg(feature = "candle")]
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::CandleChatProcessor.build_arc(self.chat_processor_name),
                )
                .with_publications(&[Publication::ExtendChunks {
                    subject_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    col_name: "content".to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateAllRecordBatches {
                        subject_name: self.chat_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.chat_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ExtractPDF.build_arc(self.extract_pdf_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: self.document_chunk_task_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::UserPdf.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.extract_pdf_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::ChunkDocuments
                        .build_arc(self.document_chunk_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: self.state_documents_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.document_chunk_task_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.document_chunk_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            #[cfg(all(feature = "api", not(feature = "candle")))]
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::OpenAIEmbedProcessor
                        .build_arc(self.embed_documents_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: self.state_doc_embed_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: self.state_documents_table_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.embed_documents_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            #[cfg(all(feature = "api", not(feature = "candle")))]
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::OpenAIEmbedProcessor
                        .build_arc(self.embed_query_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: self.state_q_embed_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::UserQueries.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.embed_query_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            #[cfg(feature = "candle")]
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::CandleEmbedProcessor
                        .build_arc(self.embed_documents_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: self.state_doc_embed_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: self.state_documents_table_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.embed_documents_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            #[cfg(feature = "candle")]
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::CandleEmbedProcessor
                        .build_arc(self.embed_query_processor_name),
                )
                .with_publications(&[Publication::Extend {
                    subject_name: self.state_q_embed_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: AvailableInterfaceSubjects::UserQueries.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.embed_query_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::VectorDistance
                        .build_arc(self.relative_similarity_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.state_scores_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_doc_embed_table_name.to_string(),
                    },
                    Subscription::OnUpdateLastRecordBatch {
                        subject_name: self.state_q_embed_table_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.relative_similarity_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Sort.build_arc(self.sort_scores_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.state_scores_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.sort_scores_processor_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_scores_table_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Join.build_arc(self.join_chunks_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.state_scores_chunks_join_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_documents_table_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_scores_table_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.join_chunks_processor_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::Select.build_arc(self.top_k_select_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.state_top_k_select_docs_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.top_k_select_processor_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_scores_chunks_join_table_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::LimitProcessor.build_arc(self.top_k_limit_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.state_top_k_limit_docs_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.top_k_limit_processor_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_top_k_select_docs_table_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
            ProcessorPlanBuilder::default()
                .with_processor(
                    AvailableProcessors::PackTabular.build_arc(self.top_k_summary_processor_name),
                )
                .with_publications(&[Publication::Replace {
                    subject_name: self.state_top_k_summary_docs_table_name.to_string(),
                }])
                .with_subscriptions(&[
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.top_k_summary_processor_name.to_string(),
                    },
                    Subscription::AlwaysAllRecordBatches {
                        subject_name: self.state_top_k_limit_docs_table_name.to_string(),
                    },
                ])
                .with_subscribe_policy(
                    AvailableSubscribeEvents::AllSubjectNamesSubscribe.build(),
                )
                .build()
                .unwrap(),
        ];

        Some(processors)
    }

    fn make_runtime_env(&self) -> Option<Arc<RuntimeEnv>> {
        Some(RuntimeEnv::get_builder().with_name(self.session_context_name).build_arc().unwrap())
    }

    fn make_subjects(&self) -> Option<Vec<SubjectPlan>> {
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
            candle_chat_config.openai_asset = Some(AvailableOpenAIAssets::MetaLlamaV3p2_1B);
            candle_chat_config.weights_config_file = None;
            candle_chat_config.weights_file = None;
            candle_chat_config.tokenizer_file = None;
            candle_chat_config.tokenizer_config_file = None;
            candle_chat_config.api_url = self.chat_api_url.map(|s| s.to_string());
        }

        let candle_chat_config_json = serde_json::to_vec(&candle_chat_config).unwrap();
        let candle_chat_state = SubjectBuilder::new()
            .with_name(self.chat_processor_name)
            .with_json(&candle_chat_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Default embed config
        #[allow(unused_mut)]
        let mut candle_embed_config = CandleEmbedConfig {
            documents: self.state_documents_table_name.to_string(),
            // All files need to be local for WASM testing
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                // "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/pytorch_model.bin",
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/all-minilm-l6-v2-q8_0.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(
                // WhichCandleAsset::BertEmbed,
                AvailableCandleAssets::QuantizedBertEmbed,
            ),
            // weights_config_file: Some(format!(
            //     "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/config.json",
            //     std::env::var("HOME").unwrap_or("".to_string())
            // )),
            // weights_file: Some(format!(
            //     "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf",
            //     std::env::var("HOME").unwrap_or("".to_string())
            // )),
            // tokenizer_file: Some(format!(
            //     "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer.json",
            //     std::env::var("HOME").unwrap_or("".to_string())
            // )),
            // tokenizer_config_file: Some(format!(
            //     "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer_config.json",
            //     std::env::var("HOME").unwrap_or("".to_string())
            // )),
            // candle_asset: Some(
            //     WhichCandleAsset::QwenV2_1p5bEmbed,
            // ),
            encoding_format: "float".to_string(),
            modality: "text".to_string(),
            input_type: "query".to_string(),
            ..Default::default()
        };

        // Add hf_hub if available
        #[cfg(feature = "hf_hub")]
        {
            candle_embed_config.weights_config_file = None;
            candle_embed_config.weights_file = None;
            candle_embed_config.tokenizer_file = None;
            candle_embed_config.tokenizer_config_file = None;
            candle_embed_config.candle_asset = Some(AvailableCandleAssets::QwenV2_1p5bEmbed);
        }

        // Add openAI_api if available
        #[cfg(all(feature = "api", not(feature = "candle")))]
        {
            candle_embed_config.candle_asset = None;
            candle_embed_config.openai_asset =
                Some(AvailableOpenAIAssets::NvidiaLlamaV3p2NvEmbedQA1BV2);
            candle_embed_config.weights_config_file = None;
            candle_embed_config.weights_file = None;
            candle_embed_config.tokenizer_file = None;
            candle_embed_config.tokenizer_config_file = None;
            candle_embed_config.api_url = self.embed_api_url.map(|s| s.to_string());
            candle_embed_config.input_type = "query".to_string();
        }
        let candle_embed_config_json = serde_json::to_vec(&candle_embed_config).unwrap();
        let candle_doc_embed_state = SubjectBuilder::new()
            .with_name(self.embed_documents_processor_name)
            .with_json(&candle_embed_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        candle_embed_config.documents = AvailableInterfaceSubjects::UserQueries.to_string();
        let candle_embed_config_json = serde_json::to_vec(&candle_embed_config).unwrap();
        let candle_query_embed_state = SubjectBuilder::new()
            .with_name(self.embed_query_processor_name)
            .with_json(&candle_embed_config_json, 1)
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
        let aggregator_1_state = SubjectBuilder::new()
            .with_name(self.message_aggregator_processor_1_name)
            .with_json(&aggregator_config_json.clone(), 1)
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

        // Select and cast config
        let message_to_query_config = DataConfig {
            lhs_name: Some(AvailableInterfaceSubjects::UserMessages.to_string()),
            lhs_values: Some(vec!["timestamp".to_string(),"content".to_string()]),
            rhs_values: Some(vec!["".to_string(),"".to_string()]),
            as_columns: Some(vec!["query_id".to_string(), "text".to_string()]),
            column_operators: Some(vec![DataColumnOperator::None, DataColumnOperator::None]),
            cast_operators: Some(vec![DataCastOperator::Cast, DataCastOperator::None]),
            cast_datatypes: Some(vec![DataType::Utf8.to_string(), DataType::Utf8.to_string()]),
            cast_templates: Some(vec!["".to_string(), "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {{ content }}".to_string()]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let message_to_query_config_json = serde_json::to_vec(&message_to_query_config).unwrap();
        let message_to_query_state = SubjectBuilder::new()
            .with_name(self.message_to_query_processor_name)
            .with_json(&message_to_query_config_json.clone(), 1)
            .unwrap()
            .build()
            .unwrap();

        // Extract pdf config
        let extract_pdf_config = DataConfig {
            lhs_name: Some(AvailableInterfaceSubjects::UserPdf.to_string()),
            lhs_pk: Some("filename".to_string()),
            lhs_values: Some(vec!["bytes".to_string()]),
            operator: AvailableCandleOperators::ExtractPDF,
            ..Default::default()
        };
        let extract_pdf_config_json = serde_json::to_vec(&extract_pdf_config).unwrap();
        let extract_pdf_state = SubjectBuilder::new()
            .with_name(self.extract_pdf_processor_name)
            .with_json(&extract_pdf_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Chunk documents config
        let chunk_document_config = DataConfig {
            lhs_name: Some(self.document_chunk_task_name.to_string()),
            lhs_pk: Some("document_id".to_string()),
            lhs_fk: Some("document_id".to_string()),
            lhs_values: Some(vec!["text".to_string()]),
            operator: AvailableCandleOperators::ChunkDocuments,
            ..Default::default()
        };
        let chunk_document_config_json = serde_json::to_vec(&chunk_document_config).unwrap();
        let chunk_document_state = SubjectBuilder::new()
            .with_name(self.document_chunk_processor_name)
            .with_json(&chunk_document_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Relative similarity config
        let rel_sim_config = DataConfig {
            lhs_name: Some(self.state_q_embed_table_name.to_string()),
            lhs_pk: Some("query_id".to_string()),
            lhs_fk: Some("query_id".to_string()),
            lhs_values: Some(vec!["embedding".to_string()]),
            rhs_name: Some(self.state_doc_embed_table_name.to_string()),
            rhs_pk: Some("chunk_id".to_string()),
            rhs_fk: Some("chunk_id".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
            operator: AvailableCandleOperators::VectorDistance,
            ..Default::default()
        };
        let rel_sim_config_json = serde_json::to_vec(&rel_sim_config).unwrap();
        let rel_sim_state = SubjectBuilder::new()
            .with_name(self.relative_similarity_processor_name)
            .with_json(&rel_sim_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Sort scores config
        let sort_scores_config = DataConfig {
            lhs_name: Some(self.state_scores_table_name.to_string()),
            lhs_pk: Some("chunk_id".to_string()),
            lhs_fk: Some("chunk_id".to_string()),
            lhs_values: Some(vec!["score".to_string()]),
            operator: AvailableCandleOperators::Sort,
            ..Default::default()
        };
        let sort_scores_config_json = serde_json::to_vec(&sort_scores_config).unwrap();
        let sort_scores_state = SubjectBuilder::new()
            .with_name(self.sort_scores_processor_name)
            .with_json(&sort_scores_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Join chunks scores config
        let join_chunks_config = DataConfig {
            lhs_name: Some(self.state_scores_table_name.to_string()),
            lhs_pk: Some("chunk_id".to_string()),
            lhs_fk: Some("chunk_id".to_string()),
            lhs_values: Some(vec!["score".to_string()]),
            rhs_name: Some(self.state_documents_table_name.to_string()),
            rhs_pk: Some("chunk_id".to_string()),
            rhs_fk: Some("chunk_id".to_string()),
            rhs_values: Some(vec!["text".to_string()]),
            operator: AvailableCandleOperators::Join,
            join_operators: Some(DataJoinOperator::Inner),
            ..Default::default()
        };
        let join_chunks_config_json = serde_json::to_vec(&join_chunks_config).unwrap();
        let join_chunks_state = SubjectBuilder::new()
            .with_name(self.join_chunks_processor_name)
            .with_json(&join_chunks_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Select top K config
        let top_k_select_config = DataConfig {
            lhs_name: Some(self.state_scores_chunks_join_table_name.to_string()),
            lhs_values: Some(vec!["text".to_string()]),
            rhs_values: Some(vec!["".to_string()]),
            as_columns: Some(vec!["".to_string()]),
            column_operators: Some(vec![DataColumnOperator::None]),
            cast_operators: Some(vec![DataCastOperator::None]),
            cast_datatypes: Some(vec![DataType::Utf8.to_string()]),
            cast_templates: Some(vec!["".to_string()]),
            operator: AvailableCandleOperators::Select,
            ..Default::default()
        };
        let top_k_select_config_json = serde_json::to_vec(&top_k_select_config).unwrap();
        let top_k_select_state = SubjectBuilder::new()
            .with_name(self.top_k_select_processor_name)
            .with_json(&top_k_select_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Limit top K config
        let top_k_limit_config = LimitConfig {
            fetch: 3,
            skip: Some(0),
        };
        let top_k_limit_config_json = serde_json::to_vec(&top_k_limit_config).unwrap();
        let top_k_limit_state = SubjectBuilder::new()
            .with_name(self.top_k_limit_processor_name)
            .with_json(&top_k_limit_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Scores and summary table schemas
        fn create_top_k_fields() -> Fields {
            let fields_vec = vec![Field::new("text", DataType::Utf8, false)];
            Fields::from(fields_vec)
        }
        let top_k_select_table = Subject::get_builder()
            .with_name(self.state_top_k_select_docs_table_name)
            .with_schema(create_schema_from_fields(&create_top_k_fields))
            .with_record_batches(Vec::new())
            .unwrap()
            .build()
            .unwrap();
        let top_k_limit_table = Subject::get_builder()
            .with_name(self.state_top_k_limit_docs_table_name)
            .with_schema(create_schema_from_fields(&create_top_k_fields))
            .with_record_batches(Vec::new())
            .unwrap()
            .build()
            .unwrap();

        // Summary top K
        let top_k_summary_config = DataConfig {
            lhs_name: Some(self.state_top_k_limit_docs_table_name.to_string()),
            doc_name: Some(self.state_top_k_limit_docs_table_name.to_string()),
            encoding: Some(DataEncoding::None),
            format: Some(DataFormat::None),
            schema: Some(AvailableSubjects::Messages),
            cpu: false,
            operator: AvailableCandleOperators::PackTabular,
            lhs_stream: DataStreamManager::Accumulate,
            ..Default::default()
        };
        let top_k_summary_config_json = serde_json::to_vec(&top_k_summary_config).unwrap();
        let top_k_summary_state = SubjectBuilder::new()
            .with_name(self.top_k_summary_processor_name)
            .with_json(&top_k_summary_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        let subjects = vec![
            candle_chat_state,
            candle_doc_embed_state,
            candle_query_embed_state,
            aggregator_1_state,
            aggregator_2_state,
            aggregator_3_state,
            message_to_query_state,
            extract_pdf_state,
            chunk_document_state,
            rel_sim_state,
            sort_scores_state,
            join_chunks_state,
            top_k_select_state,
            top_k_limit_state,
            top_k_summary_state,
            AvailableSubjects::Messages
                .to_subject(Some(self.chat_task_name), None)
                .unwrap(),
            AvailableInterfaceSubjects::AggregatedMessages
                .to_subject(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::UserMessages
                .to_subject(None, None)
                .unwrap(),
            AvailableInterfaceSubjects::AssistantMessages
                .to_subject(None, None)
                .unwrap(),
            top_k_select_table,
            top_k_limit_table,
            AvailableSubjects::Messages
                .to_subject(Some(self.state_top_k_summary_docs_table_name), None)
                .unwrap(),
            AvailableInterfaceSubjects::UserPdf
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::Documents
                .to_subject(Some(self.state_documents_table_name), None)
                .unwrap(),
            AvailableSubjects::Documents
                .to_subject(Some(self.document_chunk_task_name), None)
                .unwrap(),
            AvailableInterfaceSubjects::UserQueries
                .to_subject(None, None)
                .unwrap(),
            AvailableSubjects::DocumentEmbeddings
                .to_subject(Some(self.state_doc_embed_table_name), None)
                .unwrap(),
            AvailableSubjects::QueryEmbeddings
                .to_subject(Some(self.state_q_embed_table_name), None)
                .unwrap(),
            AvailableSubjects::EmbeddingScores
                .to_subject(Some(self.state_scores_table_name), None)
                .unwrap(),
            AvailableSubjects::JoinChunksScores
                .to_subject(Some(self.state_scores_chunks_join_table_name), None)
                .unwrap(),
            AvailableInterfaceSubjects::AggregatedAttachments
                .to_subject(None, None)
                .unwrap(),
        ];
        let subject_plans = subjects.into_iter().map(|s| SubjectPlan::get_builder().with_subject(s).build().unwrap()).collect::<Vec<_>>();
        Some(subject_plans)
    }
}

#[allow(dead_code)]
pub fn fields_in_schemas(lhs_schema: SchemaRef, rhs_schema: SchemaRef) -> Vec<String> {
    let mut found_fields = Vec::new();
    for lhs_field in lhs_schema.fields() {
        for rhs_field in rhs_schema.fields() {
            if lhs_field == rhs_field {
                found_fields.push(lhs_field.name().to_string());
                break;
            }
        }
    }
    found_fields
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::Result;
    use futures::TryStreamExt;
    use phymes_core::{
        AttachmentBuilderTraitExt, BuildableTrait, ChatBuilderTraitExt, IPCMessage, MappableTrait,
        MessageBuilderTrait, MessageTrait, SubjectTrait,
    };
    use phymes_data::make_pdf_document;
    use phymes_diagnostics::HashMap;

    use crate::{SessionContextBuilderAgentsTrait, SessionStream, create_message_map};

    use super::*;

    #[tokio::test]
    async fn test_doc_rag_session() -> Result<()> {
        // initialize the session
        let mut doc_rag_session = DocumentRAGSession::default();
        if cfg!(not(feature = "candle")) {
            doc_rag_session.chat_api_url = Some("http://0.0.0.0:8000/v1");
            doc_rag_session.embed_api_url = Some("http://0.0.0.0:8001/v1");
        }
        let (session_ctx, session_messages) = doc_rag_session
            .build()
            .with_name(doc_rag_session.session_context_name)
            .add_session_interface(None)?
            .add_next_tasks()?
            .add_next_supersteps()?
            .build_with_tables()?;
        let session_ctx_arc = Arc::new(session_ctx);

        // Create the document message
        let document_texts = &[
            "Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually results in protein folding into a specific 3D structure that determines its activity.\n\nA linear chain of amino acid residues is called a polypeptide. A protein contains at least one long polypeptide. Short polypeptides, containing less than 20–30 residues, are rarely considered to be proteins and are commonly called peptides. The individual amino acid residues are bonded together by peptide bonds and adjacent amino acid residues. The sequence of amino acid residues in a protein is defined by the sequence of a gene, which is encoded in the genetic code. In general, the genetic code specifies 20 standard amino acids; but in certain organisms the genetic code can include selenocysteine and—in certain archaea—pyrrolysine. Shortly after or even during synthesis, the residues in a protein are often chemically modified by post-translational modification, which alters the physical and chemical properties, folding, stability, activity, and ultimately, the function of the proteins. Some proteins have non-peptide groups attached, which can be called prosthetic groups or cofactors. Proteins can work together to achieve a particular function, and they often associate to form stable protein complexes.\n\nOnce formed, proteins only exist for a certain period and are then degraded and recycled by the cell's machinery through the process of protein turnover. A protein's lifespan is measured in terms of its half-life and covers a wide range. They can exist for minutes or years with an average lifespan of 1-2 days in mammalian cells. Abnormal or misfolded proteins are degraded more rapidly either due to being targeted for destruction or due to being unstable.\n\nLike other biological macromolecules such as polysaccharides and nucleic acids, proteins are essential parts of organisms and participate in virtually every process within cells. Many proteins are enzymes that catalyse biochemical reactions and are vital to metabolism. Some proteins have structural or mechanical functions, such as actin and myosin in muscle, and the cytoskeleton's scaffolding proteins that maintain cell shape. Other proteins are important in cell signaling, immune responses, cell adhesion, and the cell cycle. In animals, proteins are needed in the diet to provide the essential amino acids that cannot be synthesized. Digestion breaks the proteins down for metabolic use.",
            "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.\n\nThe two DNA strands are known as polynucleotides as they are composed of simpler monomeric units called nucleotides.[2][3] Each nucleotide is composed of one of four nitrogen-containing nucleobases (cytosine [C], guanine [G], adenine [A] or thymine [T]), a sugar called deoxyribose, and a phosphate group. The nucleotides are joined to one another in a chain by covalent bonds (known as the phosphodiester linkage) between the sugar of one nucleotide and the phosphate of the next, resulting in an alternating sugar-phosphate backbone. The nitrogenous bases of the two separate polynucleotide strands are bound together, according to base pairing rules (A with T and C with G), with hydrogen bonds to make double-stranded DNA. The complementary nitrogenous bases are divided into two groups, the single-ringed pyrimidines and the double-ringed purines. In DNA, the pyrimidines are thymine and cytosine; the purines are adenine and guanine.\n\nBoth strands of double-stranded DNA store the same biological information. This information is replicated when the two strands separate. A large part of DNA (more than 98% for humans) is non-coding, meaning that these sections do not serve as patterns for protein sequences. The two strands of DNA run in opposite directions to each other and are thus antiparallel. Attached to each sugar is one of four types of nucleobases (or bases). It is the sequence of these four nucleobases along the backbone that encodes genetic information. RNA strands are created using DNA strands as a template in a process called transcription, where DNA bases are exchanged for their corresponding bases except in the case of thymine (T), for which RNA substitutes uracil (U).[4] Under the genetic code, these RNA strands specify the sequence of amino acids within proteins in a process called translation.\n\nWithin eukaryotic cells, DNA is organized into long structures called chromosomes. Before typical cell division, these chromosomes are duplicated in the process of DNA replication, providing a complete set of chromosomes for each daughter cell. Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus as nuclear DNA, and some in the mitochondria as mitochondrial DNA or in chloroplasts as chloroplast DNA.[5] In contrast, prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in circular chromosomes. Within eukaryotic chromosomes, chromatin proteins, such as histones, compact and organize DNA. These compacting structures guide the interactions between DNA and other proteins, helping control which parts of the DNA are transcribed.",
            "Lipids are a broad group of organic compounds which include fats, waxes, sterols, fat-soluble vitamins (such as vitamins A, D, E and K), monoglycerides, diglycerides, phospholipids, and others. The functions of lipids include storing energy, signaling, and acting as structural components of cell membranes.[3][4] Lipids have applications in the cosmetic and food industries, and in nanotechnology.[5]\n\nLipids may be broadly defined as hydrophobic or amphiphilic small molecules; the amphiphilic nature of some lipids allows them to form structures such as vesicles, multilamellar/unilamellar liposomes, or membranes in an aqueous environment. Biological lipids originate entirely or in part from two distinct types of biochemical subunits or building-blocks: ketoacyl and isoprene groups.[3] Using this approach, lipids may be divided into eight categories: fatty acyls, glycerolipids, glycerophospholipids, sphingolipids, saccharolipids, and polyketides (derived from condensation of ketoacyl subunits); and sterol lipids and prenol lipids (derived from condensation of isoprene subunits).[3]\n\nAlthough the term lipid is sometimes used as a synonym for fats, fats are a subgroup of lipids called triglycerides. Lipids also encompass molecules such as fatty acids and their derivatives (including tri-, di-, monoglycerides, and phospholipids), as well as other sterol-containing metabolites such as cholesterol.[6] Although humans and other mammals use various biosynthetic pathways both to break down and to synthesize lipids, some essential lipids cannot be made this way and must be obtained from the diet.\n\n",
            "The cell is the basic structural and functional unit of all forms of life. Every cell consists of cytoplasm enclosed within a membrane; many cells contain organelles, each with a specific function. The term comes from the Latin word cellula meaning 'small room'. Most cells are only visible under a microscope. Cells emerged on Earth about 4 billion years ago. All cells are capable of replication, protein synthesis, and motility.\n\nCells are broadly categorized into two types: eukaryotic cells, which possess a nucleus, and prokaryotic cells, which lack a nucleus but have a nucleoid region. Prokaryotes are single-celled organisms such as bacteria, whereas eukaryotes can be either single-celled, such as amoebae, or multicellular, such as some algae, plants, animals, and fungi. Eukaryotic cells contain organelles including mitochondria, which provide energy for cell functions, chloroplasts, which in plants create sugars by photosynthesis, and ribosomes, which synthesise proteins.\n\nCells were discovered by Robert Hooke in 1665, who named them after their resemblance to cells inhabited by Christian monks in a monastery. Cell theory, developed in 1839 by Matthias Jakob Schleiden and Theodor Schwann, states that all organisms are composed of one or more cells, that cells are the fundamental unit of structure and function in all living organisms, and that all cells come from pre-existing cells.",
        ];
        let mut pdf = make_pdf_document(document_texts);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes)?;

        // Wrap into the message
        let chat = AvailableInterfaceSubjects::UserMessages
            .to_subject_builder(None)
            .append_new_user_query_str("What are the four molecules that compose DNA?", "user")?
            .build()?;
        let chat_message = IPCMessage::get_builder()
            .with_message(chat.to_ipc_stream()?)
            .with_subject(chat.get_name())
            .with_update(&Publication::Extend {
                subject_name: chat.get_name().to_string(),
            })
            .with_publisher(doc_rag_session.session_context_name)
            .make_name()?
            .build()?;
        let blob = AvailableInterfaceSubjects::UserPdf
            .to_subject_builder(None)
            .with_attachment(None, Some("pdf"), &bytes, None)?
            .build()?;
        let blob_message = IPCMessage::get_builder()
            .with_message(blob.to_ipc_stream()?)
            .with_subject(blob.get_name())
            .with_update(&Publication::Extend {
                subject_name: blob.get_name().to_string(),
            })
            .with_publisher(doc_rag_session.session_context_name)
            .make_name()?
            .build()?;

        // Skip actually running the session as it takes too long on the CPU
        //     until a smaller embedding model is supported (i.e., QuantBERT)
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // ----- Query #1 -----
            // Embed the documents
            let mut message_map = create_message_map(vec![blob_message]);
            message_map.extend(session_messages.unwrap_or_default());
            let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
            let _response: Vec<HashMap<String, IPCMessage>> = session_stream.try_collect().await?;

            // Embed the query and invoke a response
            let message_map = create_message_map(vec![chat_message]);
            let session_stream = SessionStream::new(message_map, Arc::clone(&session_ctx_arc));
            let mut response: Vec<HashMap<String, IPCMessage>> =
                session_stream.try_collect().await?;

            // Update the chat history with the response
            let bytes = response
                .iter_mut()
                .filter_map(|map| {
                    map.remove(&format!(
                        "from_{}_on_{}",
                        doc_rag_session.session_context_name,
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

            // for metric in metrics.clone_inner().iter() {
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == doc_rag_session.chat_processor_name
            //     {
            //         assert!(metric.value().as_usize() >= 1);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap()
            //             == doc_rag_session.embed_documents_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 21);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap()
            //             == doc_rag_session.document_chunk_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 21);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == doc_rag_session.embed_query_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 1);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap()
            //             == doc_rag_session.relative_similarity_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 21);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == doc_rag_session.sort_scores_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 21);
            //     }
            //     if metric.value().name() == "output_rows"
            //         && metric.span_name().as_ref().unwrap() == doc_rag_session.top_k_processor_name
            //     {
            //         assert_eq!(metric.value().as_usize(), 1);
            //     }
            // }

            assert_eq!(json_data.first().unwrap().get("role").unwrap(), "assistant");
            assert!(json_data.first().unwrap().get("content").is_some());
        }

        Ok(())
    }
}
