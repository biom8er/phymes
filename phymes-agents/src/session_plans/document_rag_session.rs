use std::sync::Arc;

use phymes_core::{
    schemas::available_subjects::{AvailableSubjects, AvailableSubjectsTrait},
    session::{
        common_traits::BuilderTrait,
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
        session_context_builder::TaskPlan,
    },
    table::{
        table::{Table, TableBuilder, TableBuilderTrait},
        table_publish::TablePublish,
        table_subscribe::{AllTableNamesSubscribe, TableSubscribe, SubscribeTrait},
    },
    task::processor::{ProcessorEcho, ProcessorTrait},
};
use phymes_data::{
    candle_data::{
        data_config::DataConfig, data_processor::CandleDataProcessor,
        summary_config::{DataSummaryConfig, DataSummaryFormat}, summary_processor::DataSummaryProcessor,
    },
    candle_operators::available_candle_operators::AvailableCandleOperators,
};
use phymes_ml::{
    candle_assets::available_candle_assets::AvailableCandleAssets,
    candle_chat::{
        chat_config::CandleChatConfig, chat_processor::CandleChatProcessor,
        message_aggregator_processor::MessageAggregatorProcessor,
    },
    candle_embed::{embed_config::CandleEmbedConfig, embed_processor::CandleEmbedProcessor},
};
#[cfg(feature = "openai_api")]
use phymes_ml::{
    openai_asset::available_openai_assets::AvailableOpenAIAssets,
    openai_chat::chat_processor::OpenAIChatProcessor,
    openai_embed::embed_processor::OpenAIEmbedProcessor,
};

use arrow::datatypes::SchemaRef;

use crate::{session_plans::available_interface_subjects::AvailableInterfaceSubjects, session_traits::agents::CustomAgentsBuilderTrait};

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
    /// Embed tasks
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
    pub top_k_processor_name: &'a str,
    pub vector_search_runtime_env_name: &'a str,
    /// Session and state
    pub session_context_name: &'a str,
    pub state_documents_table_name: &'a str,
    pub state_doc_embed_table_name: &'a str,
    pub state_q_embed_table_name: &'a str,
    pub state_top_k_docs_table_name: &'a str,
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
            chat_processor_name: "chat_processor_1",
            chat_runtime_env_name: "chat_rt_1",
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
            top_k_processor_name: "top_k_processor_1",
            vector_search_runtime_env_name: "vs_rt_1",
            session_context_name: "session_context_1",
            state_documents_table_name: "documents",
            state_doc_embed_table_name: "doc_embeddings",
            state_q_embed_table_name: "q_embeddings",
            state_top_k_docs_table_name: "top_k",
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
        let mut tasks = Vec::new();

        // DM: `Reqwest` connections break prematurely in `OpenAIChatProcessor`
        //  when chained or nested within other streams.
        tasks.push(TaskPlan {
            task_name: self.message_aggregator_task_1_name.to_string(),
            runtime_env_name: self.vector_search_runtime_env_name.to_string(),
            processor_names: vec![self.message_aggregator_processor_1_name.to_string()],
        });
        tasks.push(TaskPlan {
            task_name: self.message_aggregator_task_2_name.to_string(),
            runtime_env_name: self.vector_search_runtime_env_name.to_string(),
            processor_names: vec![self.message_aggregator_processor_2_name.to_string()],
        });
        tasks.push(TaskPlan {
            task_name: self.chat_task_name.to_string(),
            runtime_env_name: self.chat_runtime_env_name.to_string(),
            processor_names: vec![self.chat_processor_name.to_string()],
        });
        tasks.push(TaskPlan {
            task_name: self.extract_pdf_task_name.to_string(),
            runtime_env_name: "rt_default".to_string(),
            processor_names: vec![
                self.extract_pdf_processor_name.to_string(),
                self.document_chunk_processor_name.to_string(),
            ],
        });
        tasks.push(TaskPlan {
            task_name: self.embed_documents_task_name.to_string(),
            runtime_env_name: self.embed_documents_runtime_env_name.to_string(),
            processor_names: vec![self.embed_documents_processor_name.to_string()],
        });

        tasks.push(TaskPlan {
            task_name: self.embed_query_task_name.to_string(),
            runtime_env_name: self.embed_query_runtime_env_name.to_string(),
            processor_names: vec![self.embed_query_processor_name.to_string()],
        });
        tasks.push(TaskPlan {
            task_name: self.vector_search_task_name.to_string(),
            runtime_env_name: self.vector_search_runtime_env_name.to_string(),
            processor_names: vec![
                self.relative_similarity_processor_name.to_string(),
                self.sort_scores_processor_name.to_string(),
                self.join_chunks_processor_name.to_string(),
                self.top_k_processor_name.to_string(),
            ],
        });
        tasks.push(TaskPlan {
            task_name: self.session_context_name.to_string(),
            runtime_env_name: "rt_default".to_string(),
            processor_names: vec![self.session_context_name.to_string()],
        });

        Some(tasks)
    }

    fn make_processors(&self) -> Option<Vec<Arc<dyn ProcessorTrait>>> {
        // The order is the order in which the processors are called in the task
        let mut processors = Vec::new();

        processors.push(MessageAggregatorProcessor::new_arc_with_pub_sub(
            self.message_aggregator_processor_1_name,
            &[TablePublish::Replace {
                table_name: self.chat_task_name.to_string(),
            }],
            &[
                TableSubscribe::OnUpdateFullTable {
                    table_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                },
                TableSubscribe::OnUpdateFullTable {
                    table_name: self.state_top_k_docs_table_name.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                },
                TableSubscribe::AlwaysLastRecordBatch {
                    table_name: self.message_aggregator_processor_1_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));
        processors.push(MessageAggregatorProcessor::new_arc_with_pub_sub(
            self.message_aggregator_processor_2_name,
            &[TablePublish::Extend {
                table_name: AvailableInterfaceSubjects::AggregatedMessages.to_string(),
            }],
            &[
                TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                },
                TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                },
                TableSubscribe::AlwaysLastRecordBatch {
                    table_name: self.message_aggregator_processor_2_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));
        if cfg!(not(feature = "candle")) {
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIChatProcessor::new_arc_with_pub_sub(
                self.chat_processor_name,
                &[TablePublish::ExtendChunks {
                    table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.chat_task_name.to_string(),
                    },
                    TableSubscribe::None,
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.chat_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ));
        } else {
            processors.push(CandleChatProcessor::new_arc_with_pub_sub(
                self.chat_processor_name,
                &[TablePublish::ExtendChunks {
                    table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateFullTable {
                        table_name: self.chat_task_name.to_string(),
                    },
                    TableSubscribe::None,
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.chat_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ));
        }

        processors.push(CandleDataProcessor::new_arc_with_pub_sub(
            self.extract_pdf_processor_name,
            &[TablePublish::Extend {
                table_name: self.document_chunk_task_name.to_string(),
            }],
            &[
                TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: AvailableInterfaceSubjects::UserPdf.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: self.extract_pdf_processor_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));
        processors.push(CandleDataProcessor::new_arc_with_pub_sub(
            self.document_chunk_processor_name,
            &[TablePublish::Extend {
                table_name: self.state_documents_table_name.to_string(),
            }],
            &[
                TableSubscribe::AlwaysFullTable {
                    table_name: self.document_chunk_task_name.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: self.document_chunk_processor_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));

        if cfg!(not(feature = "candle")) {
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIEmbedProcessor::new_arc_with_pub_sub(
                self.embed_documents_processor_name,
                &[TablePublish::Extend {
                    table_name: self.state_doc_embed_table_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: self.state_documents_table_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.embed_documents_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ));
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIEmbedProcessor::new_arc_with_pub_sub(
                self.embed_query_processor_name,
                &[TablePublish::Extend {
                    table_name: self.state_q_embed_table_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: AvailableInterfaceSubjects::UserQueries.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.embed_query_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ));
        } else {
            processors.push(CandleEmbedProcessor::new_arc_with_pub_sub(
                self.embed_documents_processor_name,
                &[TablePublish::Extend {
                    table_name: self.state_doc_embed_table_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: self.state_documents_table_name.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.embed_documents_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ));
            processors.push(CandleEmbedProcessor::new_arc_with_pub_sub(
                self.embed_query_processor_name,
                &[TablePublish::Extend {
                    table_name: self.state_q_embed_table_name.to_string(),
                }],
                &[
                    TableSubscribe::OnUpdateLastRecordBatch {
                        table_name: AvailableInterfaceSubjects::UserQueries.to_string(),
                    },
                    TableSubscribe::AlwaysFullTable {
                        table_name: self.embed_query_processor_name.to_string(),
                    },
                ],
                AllTableNamesSubscribe::new_box(),
            ));
        }

        processors.push(CandleDataProcessor::new_arc_with_pub_sub(
            self.relative_similarity_processor_name,
            &[TablePublish::Replace {
                table_name: self.state_scores_table_name.to_string(),
            }],
            &[
                TableSubscribe::AlwaysFullTable {
                    table_name: self.state_doc_embed_table_name.to_string(),
                },
                TableSubscribe::OnUpdateLastRecordBatch {
                    table_name: self.state_q_embed_table_name.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: self.relative_similarity_processor_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));
        processors.push(CandleDataProcessor::new_arc_with_pub_sub(
            self.sort_scores_processor_name,
            &[TablePublish::Replace {
                table_name: self.state_scores_table_name.to_string(),
            }],
            &[
                TableSubscribe::AlwaysFullTable {
                    table_name: self.sort_scores_processor_name.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: self.state_scores_table_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));
        processors.push(CandleDataProcessor::new_arc_with_pub_sub(
            self.join_chunks_processor_name,
            &[TablePublish::Replace {
                table_name: self.state_scores_chunks_join_table_name.to_string(),
            }],
            &[
                TableSubscribe::AlwaysFullTable {
                    table_name: self.state_documents_table_name.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: self.state_scores_table_name.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: self.join_chunks_processor_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));
        processors.push(DataSummaryProcessor::new_arc_with_pub_sub(
            self.top_k_processor_name,
            &[TablePublish::Replace {
                table_name: self.state_top_k_docs_table_name.to_string(),
            }],
            &[
                TableSubscribe::AlwaysFullTable {
                    table_name: self.top_k_processor_name.to_string(),
                },
                TableSubscribe::AlwaysFullTable {
                    table_name: self
                        .state_scores_chunks_join_table_name.to_string(),
                },
            ],
            AllTableNamesSubscribe::new_box(),
        ));
        processors.push(ProcessorEcho::new_arc_with_pub_sub(
            self.session_context_name,
            &[
                TablePublish::Extend {
                    table_name: AvailableInterfaceSubjects::UserMessages.to_string(),
                },
                TablePublish::Extend {
                    table_name: self.state_documents_table_name.to_string(),
                },
                TablePublish::Extend {
                    table_name: AvailableInterfaceSubjects::UserQueries.to_string(),
                },
                TablePublish::Extend {
                    table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
                },
            ],
            &[TableSubscribe::OnUpdateLastRecordBatch {
                table_name: AvailableInterfaceSubjects::AssistantMessages.to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        ));

        Some(processors)
    }

    fn make_runtime_envs(&self) -> Option<Vec<RuntimeEnv>> {
        Some(vec![
            RuntimeEnv::new().with_name(self.chat_runtime_env_name),
            RuntimeEnv::new().with_name(self.embed_documents_runtime_env_name),
            RuntimeEnv::new().with_name(self.embed_query_runtime_env_name),
            RuntimeEnv::new().with_name(self.vector_search_runtime_env_name),
            RuntimeEnv::new().with_name("rt_default"),
        ])
    }

    fn make_state_tables(&self) -> Option<Vec<Table>> {
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
        #[cfg(all(feature = "openai_api", not(feature = "candle")))]
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
        let candle_chat_state = TableBuilder::new()
            .with_name(self.chat_processor_name)
            .with_json(&candle_chat_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Default embed config
        #[allow(unused_mut)]
        let mut candle_embed_config = CandleEmbedConfig {
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
        #[cfg(all(feature = "openai_api", not(feature = "candle")))]
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
        let candle_doc_embed_state = TableBuilder::new()
            .with_name(self.embed_documents_processor_name)
            .with_json(&candle_embed_config_json, 1)
            .unwrap()
            .build()
            .unwrap();
        let candle_query_embed_state = TableBuilder::new()
            .with_name(self.embed_query_processor_name)
            .with_json(&candle_embed_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Message aggregator config
        let aggregator_config = DataConfig {
            lhs_name: "".to_string(),
            lhs_pk: "".to_string(),
            lhs_fk: "".to_string(),
            lhs_values: "timestamp".to_string(),
            op_kwargs: Some("{\"asc\": true}".to_string()),
            operator: AvailableCandleOperators::SortColumnAndIndices,
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

        // Extract pdf config
        let extract_pdf_config = DataConfig {
            lhs_name: AvailableInterfaceSubjects::UserPdf.to_string(),
            lhs_pk: "filename".to_string(),
            lhs_values: "bytes".to_string(),
            operator: AvailableCandleOperators::ExtractPDFText,
            ..Default::default()
        };
        let extract_pdf_config_json = serde_json::to_vec(&extract_pdf_config).unwrap();
        let extract_pdf_state = TableBuilder::new()
            .with_name(self.extract_pdf_processor_name)
            .with_json(&extract_pdf_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Chunk documents config
        let chunk_document_config = DataConfig {
            lhs_name: self.document_chunk_task_name.to_string(),
            lhs_pk: "document_id".to_string(),
            lhs_fk: "document_id".to_string(),
            lhs_values: "text".to_string(),
            operator: AvailableCandleOperators::ChunkDocuments,
            ..Default::default()
        };
        let chunk_document_config_json = serde_json::to_vec(&chunk_document_config).unwrap();
        let chunk_document_state = TableBuilder::new()
            .with_name(self.document_chunk_processor_name)
            .with_json(&chunk_document_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Relative similarity config
        let rel_sim_config = DataConfig {
            lhs_name: self.state_q_embed_table_name.to_string(),
            lhs_pk: "query_id".to_string(),
            lhs_fk: "query_id".to_string(),
            lhs_values: "embedding".to_string(),
            rhs_name: Some(self.state_doc_embed_table_name.to_string()),
            rhs_pk: Some("chunk_id".to_string()),
            rhs_fk: Some("chunk_id".to_string()),
            rhs_values: Some("embedding".to_string()),
            operator: AvailableCandleOperators::RelativeSimilarityScore,
            ..Default::default()
        };
        let rel_sim_config_json = serde_json::to_vec(&rel_sim_config).unwrap();
        let rel_sim_state = TableBuilder::new()
            .with_name(self.relative_similarity_processor_name)
            .with_json(&rel_sim_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Sort scores config
        let sort_scores_config = DataConfig {
            lhs_name: self.state_scores_table_name.to_string(),
            lhs_pk: "chunk_id".to_string(),
            lhs_fk: "chunk_id".to_string(),
            lhs_values: "score".to_string(),
            operator: AvailableCandleOperators::SortColumnAndIndices,
            ..Default::default()
        };
        let sort_scores_config_json = serde_json::to_vec(&sort_scores_config).unwrap();
        let sort_scores_state = TableBuilder::new()
            .with_name(self.sort_scores_processor_name)
            .with_json(&sort_scores_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Join chunks scores config
        let join_chunks_config = DataConfig {
            lhs_name: self.state_scores_table_name.to_string(),
            lhs_pk: "chunk_id".to_string(),
            lhs_fk: "chunk_id".to_string(),
            lhs_values: "score".to_string(),
            rhs_name: Some(self.state_documents_table_name.to_string()),
            rhs_pk: Some("chunk_id".to_string()),
            rhs_fk: Some("chunk_id".to_string()),
            rhs_values: Some("text".to_string()),
            operator: AvailableCandleOperators::JoinInner,
            ..Default::default()
        };
        let join_chunks_config_json = serde_json::to_vec(&join_chunks_config).unwrap();
        let join_chunks_state = TableBuilder::new()
            .with_name(self.join_chunks_processor_name)
            .with_json(&join_chunks_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        // Summary config (to limit the number of documents)
        let top_k_config = DataSummaryConfig {
            col_names: Some(vec!["text".to_string()]),
            num_rows: Some(3),
            num_batches: Some(1),
            format: DataSummaryFormat::Message,
        };
        let top_k_config_json = serde_json::to_vec(&top_k_config).unwrap();
        let top_k_state = TableBuilder::new()
            .with_name(self.top_k_processor_name)
            .with_json(&top_k_config_json, 1)
            .unwrap()
            .build()
            .unwrap();

        Some(vec![
            candle_chat_state, 
            candle_doc_embed_state,
            candle_query_embed_state,
            aggregator_1_state,
            aggregator_2_state,
            extract_pdf_state,
            chunk_document_state,
            rel_sim_state,
            sort_scores_state,
            join_chunks_state,
            top_k_state,
            AvailableSubjects::Messages.to_table(Some(self.chat_task_name)).unwrap(),
            AvailableInterfaceSubjects::AggregatedMessages.to_table(None).unwrap(),
            AvailableInterfaceSubjects::UserMessages.to_table(None).unwrap(),
            AvailableInterfaceSubjects::AssistantMessages.to_table(None).unwrap(),
            AvailableSubjects::Messages.to_table(Some(self.state_top_k_docs_table_name)).unwrap(),
            AvailableInterfaceSubjects::UserPdf.to_table(None).unwrap(),
            AvailableSubjects::Documents.to_table(Some(self.state_documents_table_name)).unwrap(),
            AvailableSubjects::Documents.to_table(Some(self.document_chunk_task_name)).unwrap(),
            AvailableInterfaceSubjects::UserQueries.to_table(None).unwrap(),
            AvailableSubjects::DocumentEmbeddings.to_table(Some(self.state_doc_embed_table_name)).unwrap(), 
            AvailableSubjects::QueryEmbeddings.to_table(Some(self.state_q_embed_table_name)).unwrap(),       
            AvailableSubjects::EmbeddingScores.to_table(Some(self.state_scores_table_name)).unwrap(),         
            AvailableSubjects::JoinChunksScores.to_table(Some(self.state_scores_chunks_join_table_name)).unwrap(),
        ])
    }
}

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
    use anyhow::Result;
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        metrics::{ArrowTaskMetricsSet, HashMap}, schemas::available_subjects::create_timestamp_micros, session::{
            session_context::{SessionStream, SessionStreamState},
            session_context_builder::SessionContextBuilderTrait,
        }, table::table::TableTrait, task::message::{IPCMessage, ArrowIncomingMessageTrait}
    };
    use phymes_data::candle_operators::extract_pdf_text::make_pdf_document;

    use crate::{session_plans::available_interface_subjects::{create_incoming_message_map, AttachmentInterface, MessageInterface}, session_traits::agents::SessionContextBuilderAgentsTrait};

    use super::*;

    #[tokio::test]
    async fn test_doc_rag_session() -> Result<()> {
        // initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // initialize the session
        let mut doc_rag_session = DocumentRAGSession::default();
        if cfg!(feature = "hf_hub") {
        }
        if cfg!(not(feature = "candle")) {
            doc_rag_session.chat_api_url = Some("http://0.0.0.0:8000/v1");
            doc_rag_session.embed_api_url = Some("http://0.0.0.0:8001/v1");
        }
        let session_ctx = doc_rag_session
            .build()
            .with_metrics(metrics.clone())
            .with_name(doc_rag_session.session_context_name)
            .build_with_tables()?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

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
        
        // Wrap into the message/attachment interfaces
        let message_interface = MessageInterface { 
            role: "user".to_string(), 
            content: "What are the four molecules that compose DNA?".to_string(), 
            timestamp: create_timestamp_micros()
        };
        let attachment_interface = AttachmentInterface {
            filename: "wiki".to_string(),
            bytes,
            extension: ".pdf".to_string(),
            metadata: String::new(),
        };

        // Skip actually running the session as it takes too long on the CPU
        //     until a smaller embedding model is supported (i.e., QuantBERT)
        if cfg!(any(
            all(not(feature = "candle"), feature = "wsl"),
            all(not(feature = "candle"), feature = "wasip2"),
            feature = "gpu"
        )) {
            // ----- Query #1 -----
            // Embed the documents
            let incoming_message_map = create_incoming_message_map(vec![
                AvailableInterfaceSubjects::UserPdf.to_incoming_message(None, Some(vec![attachment_interface]), doc_rag_session.session_context_name)?,
            ]);
            let session_stream = SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));
            let _response: Vec<HashMap<String, IPCMessage>> =
                session_stream.try_collect().await?;

            // Embed the query and invoke a response
            let incoming_message_map = create_incoming_message_map(vec![
                AvailableInterfaceSubjects::UserMessages.to_incoming_message(Some(vec![message_interface.clone()]), None, doc_rag_session.session_context_name)?,
                AvailableInterfaceSubjects::UserQueries.to_incoming_message(Some(vec![message_interface]), None, doc_rag_session.session_context_name)?,
            ]);
            let session_stream = SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));
            let mut response: Vec<HashMap<String, IPCMessage>> =
                session_stream.try_collect().await?;

            // Update the chat history with the response
            let json_data = response
                .last_mut()
                .unwrap()
                .remove(&format!(
                    "from_{}_on_{}",
                    doc_rag_session.session_context_name,
                    AvailableInterfaceSubjects::AssistantMessages
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
                    && metric.task().as_ref().unwrap() == doc_rag_session.chat_processor_name
                {
                    assert!(metric.value().as_usize() >= 1);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap()
                        == doc_rag_session.embed_documents_processor_name
                {
                    assert_eq!(metric.value().as_usize(), 21);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap()
                        == doc_rag_session.document_chunk_processor_name
                {
                    assert_eq!(metric.value().as_usize(), 21);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == doc_rag_session.embed_query_processor_name
                {
                    assert_eq!(metric.value().as_usize(), 1);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap()
                        == doc_rag_session.relative_similarity_processor_name
                {
                    assert_eq!(metric.value().as_usize(), 21);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == doc_rag_session.sort_scores_processor_name
                {
                    assert_eq!(metric.value().as_usize(), 21);
                }
                if metric.value().name() == "output_rows"
                    && metric.task().as_ref().unwrap() == doc_rag_session.top_k_processor_name
                {
                    assert_eq!(metric.value().as_usize(), 1);
                }
            }

            assert_eq!(json_data.first().unwrap().get("role").unwrap(), "assistant");
            assert!(json_data.first().unwrap().get("content").is_some());

            // ----- Query #2 -----
            // Embed the next query and invoke another response
        }

        Ok(())
    }
}
