use anyhow::Result;
use std::sync::Arc;

#[cfg(feature = "openai_api")]
use crate::openai_asset::{
    chat_processor::OpenAIChatProcessor, embed_processor::OpenAIEmbedProcessor,
};
use crate::{
    candle_chat::{
        chat_config::CandleChatConfig, chat_processor::CandleChatProcessor,
        message_aggregator_processor::MessageAggregatorProcessor,
    },
    candle_embed::{embed_config::CandleEmbedConfig, embed_processor::CandleEmbedProcessor},
    candle_ops::{
        ops_config::CandleOpsConfig, ops_processor::CandleOpProcessor, ops_which::WhichCandleOps,
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

use super::agent_session_builder::AgentSessionBuilderTrait;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

/// Document RAG
///
/// # Supersteps
/// 1. Embed documents: session -> embed_task (chunk_processor, embed_processor)
///    ... and Embed queries: session -> embed_task (embed_processor)
///
/// 2. Vector search: -> vs_task (rel_sim_score_processor, sort_score_processor, summary_processor)
/// 3. Chat: session -> chat_task (chat_processor)
/// 4. End
///
/// # Notes
/// * The embedding size must be specified before which is determined by the size of
///   the hidden layer of the embedding model
pub struct DocumentRAGSession<'a> {
    /// Chat tasks
    pub chat_task_name: &'a str,
    // DM: needed for openai api since we cannot chain streams
    pub message_aggregator_task_name: &'a str,
    pub message_aggregator_processor_name: &'a str,
    pub chat_processor_name: &'a str,
    pub chat_runtime_env_name: &'a str,
    /// Embed tasks
    pub embed_query_task_name: &'a str,
    pub embed_documents_task_name: &'a str,
    pub embed_query_processor_name: &'a str,
    pub embed_documents_processor_name: &'a str,
    // DM: needed for openai api since we cannot chain streams
    pub document_chunk_task_name: &'a str,
    pub document_chunk_processor_1_name: &'a str,
    // DM: Two embed runtimes are needed for embedded Candle models due to edge cases
    //   where the mutexes are access simultaneously. Set the embed runtimes to
    //   the same name when using OpenAI API
    pub embed_documents_runtime_env_name: &'a str,
    pub embed_query_runtime_env_name: &'a str,
    /// Vector search tasks
    pub vector_search_task_name: &'a str,
    pub relative_similarity_processor_name: &'a str,
    pub sort_scores_processor_name: &'a str,
    // DM: Needed because the document chunks are not stored during the embed task
    //  for embedded Candle models. Set to the same name as `document_chunk_processor_1_name`
    //  when using OpenAI API
    pub document_chunk_processor_2_name: &'a str,
    pub join_chunks_processor_name: &'a str,
    pub top_k_processor_name: &'a str,
    pub vector_search_runtime_env_name: &'a str,
    /// Session and state
    pub session_context_name: &'a str,
    pub state_messages_table_name: &'a str,
    pub state_documents_table_name: &'a str,
    pub state_doc_embed_table_name: &'a str,
    pub state_queries_table_name: &'a str,
    pub state_q_embed_table_name: &'a str,
    pub state_top_k_docs_table_name: &'a str,
    pub state_scores_table_name: &'a str,
    pub state_scores_chunks_join_table_name: &'a str,
    /// Other parameters
    pub embed_length: usize,
    pub chat_api_url: Option<&'a str>,
    pub embed_api_url: Option<&'a str>,
}

impl DocumentRAGSession<'_> {
    pub fn make_messages_table(&self) -> Result<ArrowTable> {
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let timestamp = Field::new("timestamp", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![role, content, timestamp]));
        ArrowTableBuilder::new()
            .with_name(self.state_messages_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_message_aggregator_table(&self) -> Result<ArrowTable> {
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let timestamp = Field::new("timestamp", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![role, content, timestamp]));
        ArrowTableBuilder::new()
            .with_name(self.message_aggregator_task_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_documents_table(&self) -> Result<ArrowTable> {
        let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
        let document_id = Field::new("document_id", DataType::Utf8, false);
        let text = Field::new("text", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![chunk_id, document_id, text]));
        ArrowTableBuilder::new()
            .with_name(self.state_documents_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_document_chunk_table(&self) -> Result<ArrowTable> {
        let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
        let document_id = Field::new("document_id", DataType::Utf8, false);
        let text = Field::new("text", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![chunk_id, document_id, text]));
        ArrowTableBuilder::new()
            .with_name(self.document_chunk_task_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_doc_embed_table(&self) -> Result<ArrowTable> {
        let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
        let document_id = Field::new("document_id", DataType::Utf8, false);
        let list_data_type = DataType::FixedSizeList(
            Arc::new(Field::new_list_field(DataType::Float32, false)),
            self.embed_length.try_into().unwrap(),
        );
        let embeddings = Field::new("embeddings", list_data_type, false);
        let schema = Arc::new(Schema::new(vec![chunk_id, document_id, embeddings]));
        ArrowTableBuilder::new()
            .with_name(self.state_doc_embed_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_queries_table(&self) -> Result<ArrowTable> {
        let query_id = Field::new("query_id", DataType::Utf8, false);
        let text = Field::new("text", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![query_id, text]));
        ArrowTableBuilder::new()
            .with_name(self.state_queries_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_q_embed_table(&self) -> Result<ArrowTable> {
        let query_id = Field::new("query_id", DataType::Utf8, false);
        let list_data_type = DataType::FixedSizeList(
            Arc::new(Field::new_list_field(DataType::Float32, false)),
            self.embed_length.try_into().unwrap(),
        );
        let embeddings = Field::new("embeddings", list_data_type, false);
        let schema = Arc::new(Schema::new(vec![query_id, embeddings]));
        ArrowTableBuilder::new()
            .with_name(self.state_q_embed_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_scores_table(&self) -> Result<ArrowTable> {
        let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
        let query_id = Field::new("query_id", DataType::Utf8, false);
        let score = Field::new("score", DataType::Float32, false);
        let schema = Arc::new(Schema::new(vec![chunk_id, query_id, score]));
        ArrowTableBuilder::new()
            .with_name(self.state_scores_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_join_chunks_scores_table(&self) -> Result<ArrowTable> {
        let chunk_id = Field::new("chunk_id", DataType::Utf8, false);
        let query_id = Field::new("query_id", DataType::Utf8, false);
        let score = Field::new("score", DataType::Float32, false);
        let document_id = Field::new("document_id", DataType::Utf8, false);
        let text = Field::new("text", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![
            chunk_id,
            query_id,
            score,
            document_id,
            text,
        ]));
        ArrowTableBuilder::new()
            .with_name(self.state_scores_chunks_join_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
    pub fn make_top_k_docs_table(&self) -> Result<ArrowTable> {
        let role = Field::new("role", DataType::Utf8, false);
        let content = Field::new("content", DataType::Utf8, false);
        let timestamp = Field::new("timestamp", DataType::Utf8, false);
        let schema = Arc::new(Schema::new(vec![role, content, timestamp]));
        ArrowTableBuilder::new()
            .with_name(self.state_top_k_docs_table_name)
            .with_schema(schema)
            .with_record_batches(Vec::new())?
            .build()
    }
}

impl AgentSessionBuilderTrait for DocumentRAGSession<'_> {
    fn make_task_plan(&self) -> Vec<TaskPlan> {
        let mut tasks = Vec::new();

        // DM: `Reqwest` connections break prematurely in `OpenAIChatProcessor`
        //  when chained or nested within other streams
        // DM: another tool agent session publish/subscribe network needs to be
        //  made for openai_api access that breaks down the chat and document embed tasks into seperate
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
        } else {
            tasks.push(TaskPlan {
                task_name: self.chat_task_name.to_string(),
                runtime_env_name: self.chat_runtime_env_name.to_string(),
                processor_names: vec![
                    self.message_aggregator_processor_name.to_string(),
                    self.chat_processor_name.to_string(),
                ],
            });
        }

        if cfg!(not(feature = "candle")) {
            tasks.push(TaskPlan {
                task_name: self.document_chunk_task_name.to_string(),
                runtime_env_name: "rt_default".to_string(),
                processor_names: vec![self.document_chunk_processor_1_name.to_string()],
            });
            tasks.push(TaskPlan {
                task_name: self.embed_documents_task_name.to_string(),
                runtime_env_name: self.embed_documents_runtime_env_name.to_string(),
                processor_names: vec![self.embed_documents_processor_name.to_string()],
            });
        } else {
            tasks.push(TaskPlan {
                task_name: self.embed_documents_task_name.to_string(),
                runtime_env_name: self.embed_documents_runtime_env_name.to_string(),
                processor_names: vec![
                    self.document_chunk_processor_1_name.to_string(),
                    self.embed_documents_processor_name.to_string(),
                ],
            });
        }

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
                self.document_chunk_processor_2_name.to_string(),
                self.join_chunks_processor_name.to_string(),
                self.top_k_processor_name.to_string(),
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
                &[ArrowTablePublish::ExtendChunks {
                    table_name: self.state_messages_table_name.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.message_aggregator_task_name.to_string(),
                    },
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.state_top_k_docs_table_name.to_string(),
                    },
                ],
                &[],
            ));
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIChatProcessor::new_with_pub_sub_for(
                self.chat_processor_name,
                &[ArrowTablePublish::ExtendChunks {
                    table_name: self.state_messages_table_name.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.state_messages_table_name.to_string(),
                    },
                    ArrowTableSubscribe::None,
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.chat_processor_name.to_string(),
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
                &[
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.message_aggregator_task_name.to_string(),
                    },
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.state_top_k_docs_table_name.to_string(),
                    },
                ],
                &[],
            ));
            processors.push(CandleChatProcessor::new_with_pub_sub_for(
                self.chat_processor_name,
                &[ArrowTablePublish::ExtendChunks {
                    table_name: self.state_messages_table_name.to_string(),
                    col_name: "content".to_string(),
                }],
                &[
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.state_messages_table_name.to_string(),
                    },
                    ArrowTableSubscribe::None,
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.chat_processor_name.to_string(),
                    },
                ],
                &[],
            ));
        }

        processors.push(CandleOpProcessor::new_with_pub_sub_for(
            self.document_chunk_processor_1_name,
            &[ArrowTablePublish::Replace {
                table_name: self.document_chunk_task_name.to_string(),
            }],
            &[
                ArrowTableSubscribe::OnUpdateFullTable {
                    table_name: self.state_documents_table_name.to_string(),
                },
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: self.document_chunk_processor_1_name.to_string(),
                },
            ],
            &[self.embed_documents_processor_name],
        ));

        if cfg!(not(feature = "candle")) {
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIEmbedProcessor::new_with_pub_sub_for(
                self.embed_documents_processor_name,
                &[ArrowTablePublish::Replace {
                    table_name: self.state_doc_embed_table_name.to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.document_chunk_task_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.embed_documents_processor_name.to_string(),
                    },
                ],
                &[],
            ));
            #[cfg(feature = "openai_api")]
            processors.push(OpenAIEmbedProcessor::new_with_pub_sub_for(
                self.embed_query_processor_name,
                &[ArrowTablePublish::Extend {
                    table_name: self.state_q_embed_table_name.to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.state_queries_table_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.embed_query_processor_name.to_string(),
                    },
                ],
                &[],
            ));
        } else {
            processors.push(CandleEmbedProcessor::new_with_pub_sub_for(
                self.embed_documents_processor_name,
                &[ArrowTablePublish::Replace {
                    table_name: self.state_doc_embed_table_name.to_string(),
                }],
                &[
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.document_chunk_task_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.embed_documents_processor_name.to_string(),
                    },
                ],
                &[],
            ));
            processors.push(CandleEmbedProcessor::new_with_pub_sub_for(
                self.embed_query_processor_name,
                &[ArrowTablePublish::Extend {
                    table_name: self.state_q_embed_table_name.to_string(),
                }],
                &[
                    ArrowTableSubscribe::OnUpdateFullTable {
                        table_name: self.state_queries_table_name.to_string(),
                    },
                    ArrowTableSubscribe::AlwaysFullTable {
                        table_name: self.embed_query_processor_name.to_string(),
                    },
                ],
                &[],
            ));
        }

        processors.push(CandleOpProcessor::new_with_pub_sub_for(
            self.relative_similarity_processor_name,
            &[ArrowTablePublish::Replace {
                table_name: self.state_scores_table_name.to_string(),
            }],
            &[
                ArrowTableSubscribe::OnUpdateFullTable {
                    table_name: self.state_doc_embed_table_name.to_string(),
                },
                ArrowTableSubscribe::OnUpdateFullTable {
                    table_name: self.state_q_embed_table_name.to_string(),
                },
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: self.relative_similarity_processor_name.to_string(),
                },
            ],
            &[
                self.sort_scores_processor_name,
                self.top_k_processor_name,
                self.document_chunk_processor_2_name,
                self.join_chunks_processor_name,
                self.state_documents_table_name,
            ],
        ));
        processors.push(CandleOpProcessor::new_with_pub_sub_for(
            self.sort_scores_processor_name,
            &[ArrowTablePublish::Replace {
                table_name: self.state_scores_table_name.to_string(),
            }],
            &[ArrowTableSubscribe::AlwaysFullTable {
                table_name: self.sort_scores_processor_name.to_string(),
            }],
            &[
                self.top_k_processor_name,
                self.document_chunk_processor_2_name,
                self.join_chunks_processor_name,
                self.state_documents_table_name,
            ],
        ));
        processors.push(CandleOpProcessor::new_with_pub_sub_for(
            self.document_chunk_processor_2_name,
            &[ArrowTablePublish::Replace {
                table_name: self.state_documents_table_name.to_string(),
            }],
            &[
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: self.state_documents_table_name.to_string(),
                },
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: self.document_chunk_processor_2_name.to_string(),
                },
            ],
            &[
                self.join_chunks_processor_name,
                self.top_k_processor_name,
                self.state_scores_table_name,
            ],
        ));
        processors.push(CandleOpProcessor::new_with_pub_sub_for(
            self.join_chunks_processor_name,
            &[ArrowTablePublish::Replace {
                table_name: self.state_scores_chunks_join_table_name.to_string(),
            }],
            &[
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: self.state_documents_table_name.to_string(),
                },
                ArrowTableSubscribe::AlwaysFullTable {
                    table_name: self.join_chunks_processor_name.to_string().to_string(),
                },
            ],
            &[self.top_k_processor_name],
        ));
        processors.push(OpsSummaryProcessor::new_with_pub_sub_for(
            self.top_k_processor_name,
            &[ArrowTablePublish::Replace {
                table_name: self.state_top_k_docs_table_name.to_string(),
            }],
            &[ArrowTableSubscribe::AlwaysFullTable {
                table_name: self.top_k_processor_name.to_string().to_string(),
            }],
            &[],
        ));
        processors.push(ArrowProcessorEcho::new_with_pub_sub_for(
            self.session_context_name,
            &[
                ArrowTablePublish::Extend {
                    table_name: self.state_messages_table_name.to_string(),
                },
                ArrowTablePublish::Extend {
                    table_name: self.state_documents_table_name.to_string(),
                },
                ArrowTablePublish::Extend {
                    table_name: self.state_queries_table_name.to_string(),
                },
            ],
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
            RuntimeEnv::new().with_name(self.embed_documents_runtime_env_name),
            RuntimeEnv::new().with_name(self.embed_query_runtime_env_name),
            RuntimeEnv::new().with_name(self.vector_search_runtime_env_name),
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
        let candle_chat_state = ArrowTableBuilder::new()
            .with_name(self.chat_processor_name)
            .with_json(&candle_chat_config_json, 1)?
            .build()?;

        // Default embed config
        #[allow(unused_mut)]
        let mut candle_embed_config = CandleEmbedConfig {
            dimensions: Some(self.embed_length as i32),
            // All files need to be local for WASM testing
            weights_config_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            weights_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/gte-Qwen2-1.5B-instruct-Q4_K_M.gguf",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            tokenizer_config_file: Some(format!(
                "{}/.cache/hf/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/tokenizer_config.json",
                std::env::var("HOME").unwrap_or("".to_string())
            )),
            candle_asset: Some(
                crate::candle_assets::candle_which::WhichCandleAsset::QwenV2_1p5bEmbed,
            ),
            ..Default::default()
        };

        // Add hf_hub if available
        #[cfg(feature = "hf_hub")]
        {
            candle_embed_config.weights_config_file = None;
            candle_embed_config.weights_file = None;
            candle_embed_config.tokenizer_file = None;
            candle_embed_config.tokenizer_config_file = None;
        }

        // Add openAI_api if available
        #[cfg(not(feature = "candle"))]
        {
            candle_embed_config.candle_asset = None;
            candle_embed_config.openai_asset = Some(
                crate::openai_asset::openai_which::WhichOpenAIAsset::NvidiaLlamaV3p2NvEmbedQA1BV2,
            );
            candle_embed_config.weights_config_file = None;
            candle_embed_config.weights_file = None;
            candle_embed_config.tokenizer_file = None;
            candle_embed_config.tokenizer_config_file = None;
            candle_embed_config.api_url = self.embed_api_url.map(|s| s.to_string());
            candle_embed_config.input_type = "query".to_string();
        }
        let candle_embed_config_json = serde_json::to_vec(&candle_embed_config)?;
        let candle_doc_embed_state = ArrowTableBuilder::new()
            .with_name(self.embed_documents_processor_name)
            .with_json(&candle_embed_config_json, 1)?
            .build()?;
        let candle_query_embed_state = ArrowTableBuilder::new()
            .with_name(self.embed_query_processor_name)
            .with_json(&candle_embed_config_json, 1)?
            .build()?;

        // Chunk documents config
        let chunk_document_config = CandleOpsConfig {
            lhs_name: self.state_documents_table_name.to_string(),
            lhs_pk: "document_id".to_string(),
            lhs_fk: "document_id".to_string(),
            lhs_values: "text".to_string(),
            which: WhichCandleOps::ChunkDocuments,
            ..Default::default()
        };
        let chunk_document_config_json = serde_json::to_vec(&chunk_document_config)?;
        let chunk_document_1_state = ArrowTableBuilder::new()
            .with_name(self.document_chunk_processor_1_name)
            .with_json(&chunk_document_config_json, 1)?
            .build()?;
        let chunk_document_2_state = ArrowTableBuilder::new()
            .with_name(self.document_chunk_processor_2_name)
            .with_json(&chunk_document_config_json, 1)?
            .build()?;

        // Relative similarity config
        let rel_sim_config = CandleOpsConfig {
            lhs_name: self.state_q_embed_table_name.to_string(),
            lhs_pk: "query_id".to_string(),
            lhs_fk: "query_id".to_string(),
            lhs_values: "embeddings".to_string(),
            rhs_name: Some(self.state_doc_embed_table_name.to_string()),
            rhs_pk: Some("chunk_id".to_string()),
            rhs_fk: Some("chunk_id".to_string()),
            rhs_values: Some("embeddings".to_string()),
            which: WhichCandleOps::RelativeSimilarityScore,
            ..Default::default()
        };
        let rel_sim_config_json = serde_json::to_vec(&rel_sim_config)?;
        let rel_sim_state = ArrowTableBuilder::new()
            .with_name(self.relative_similarity_processor_name)
            .with_json(&rel_sim_config_json, 1)?
            .build()?;

        // Sort scores config
        let sort_scores_config = CandleOpsConfig {
            lhs_name: self.state_scores_table_name.to_string(),
            lhs_pk: "chunk_id".to_string(),
            lhs_fk: "chunk_id".to_string(),
            lhs_values: "score".to_string(),
            which: WhichCandleOps::SortScoresAndIndices,
            ..Default::default()
        };
        let sort_scores_config_json = serde_json::to_vec(&sort_scores_config)?;
        let sort_scores_state = ArrowTableBuilder::new()
            .with_name(self.sort_scores_processor_name)
            .with_json(&sort_scores_config_json, 1)?
            .build()?;

        // Join chunks scores config
        let join_chunks_config = CandleOpsConfig {
            lhs_name: self.state_scores_table_name.to_string(),
            lhs_pk: "chunk_id".to_string(),
            lhs_fk: "chunk_id".to_string(),
            lhs_values: "score".to_string(),
            rhs_name: Some(self.state_documents_table_name.to_string()),
            rhs_pk: Some("chunk_id".to_string()),
            rhs_fk: Some("chunk_id".to_string()),
            rhs_values: Some("text".to_string()),
            which: WhichCandleOps::JoinInner,
            ..Default::default()
        };
        let join_chunks_config_json = serde_json::to_vec(&join_chunks_config)?;
        let join_chunks_state = ArrowTableBuilder::new()
            .with_name(self.join_chunks_processor_name)
            .with_json(&join_chunks_config_json, 1)?
            .build()?;

        // Summary config (to limit the number of documents)
        let top_k_config = CandleOpsSummaryConfig {
            col_names: Some("[\"text\"]".to_string()),
            num_rows: Some(3),
            num_batches: Some(1),
        };
        let top_k_config_json = serde_json::to_vec(&top_k_config)?;
        let top_k_state = ArrowTableBuilder::new()
            .with_name(self.top_k_processor_name)
            .with_json(&top_k_config_json, 1)?
            .build()?;

        Ok(vec![
            candle_chat_state,
            candle_doc_embed_state,
            candle_query_embed_state,
            chunk_document_1_state,
            rel_sim_state,
            sort_scores_state,
            chunk_document_2_state,
            join_chunks_state,
            top_k_state,
            self.make_messages_table()?,
            self.make_message_aggregator_table()?,
            self.make_documents_table()?,
            self.make_document_chunk_table()?,
            self.make_queries_table()?,
            self.make_top_k_docs_table()?,
            self.make_doc_embed_table()?,
            self.make_q_embed_table()?,
            self.make_scores_table()?,
            self.make_join_chunks_scores_table()?,
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
            .build()
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
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use futures::TryStreamExt;
    use parking_lot::RwLock;
    use phymes_core::{
        metrics::HashMap,
        session::{
            common_traits::MappableTrait,
            session_context::{SessionStream, SessionStreamState},
        },
        table::arrow_table::ArrowTableTrait,
        task::arrow_message::{
            ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait,
            ArrowIncomingMessageTrait, ArrowMessageBuilderTrait,
        },
    };

    use super::*;
    use crate::candle_chat::message_history::MessageHistoryBuilderTraitExt;

    #[tokio::test]
    async fn test_doc_rag_session() -> Result<()> {
        // initialize the metrics
        let metrics = ArrowTaskMetricsSet::new();

        // initialize the session
        let mut doc_rag_session = DocumentRAGSession {
            // Chat tasks
            chat_task_name: "chat_task_1",
            message_aggregator_task_name: "message_aggregator_task_1",
            message_aggregator_processor_name: "message_aggregator_processor_1",
            chat_processor_name: "chat_processor_1",
            chat_runtime_env_name: "chat_rt_1",
            // Embed tasks
            embed_query_task_name: "embed_query_task_1",
            embed_documents_task_name: "embed_documents_task_1",
            embed_query_processor_name: "embed_query_processor_1",
            embed_documents_processor_name: "embed_documents_processor_1",
            document_chunk_task_name: "chunk_documents_task_1",
            document_chunk_processor_1_name: "chunk_documents_processor_1",
            embed_documents_runtime_env_name: "embed_documents_rt_1",
            embed_query_runtime_env_name: "embed_query_rt_1", // "embed_documents_rt_1",
            // Vector search tasks
            vector_search_task_name: "vs_task_1",
            relative_similarity_processor_name: "rel_sim_processor_1",
            sort_scores_processor_name: "sort_scores_processor_1",
            document_chunk_processor_2_name: "chunk_documents_processor_2", //"chunk_documents_processor_1",
            join_chunks_processor_name: "join_scores_chunks_processor_1",
            top_k_processor_name: "top_k_processor_1",
            vector_search_runtime_env_name: "vs_rt_1",
            // Session and state
            session_context_name: "session_1",
            state_messages_table_name: "messages",
            state_documents_table_name: "documents",
            state_doc_embed_table_name: "doc_embeddings",
            state_queries_table_name: "queries",
            state_q_embed_table_name: "q_embeddings",
            state_top_k_docs_table_name: "top_k",
            state_scores_table_name: "tmp_scores",
            state_scores_chunks_join_table_name: "tmp_scores_chunks_join",
            embed_length: 1536, // Hidden size for GTE Qwen2 1.5B
            chat_api_url: None,
            embed_api_url: None,
        };
        if cfg!(not(feature = "candle")) {
            doc_rag_session.embed_length = 384; // Smallest dimension for Llama
            doc_rag_session.chat_api_url = Some("http://0.0.0.0:8000/v1");
            doc_rag_session.embed_api_url = Some("http://0.0.0.0:8001/v1");
        }
        let session_ctx = doc_rag_session.make_session_context(metrics.clone())?;
        let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

        // ----- Query #1 -----

        // Create the document message
        let documents_vec = vec![
            "Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually results in protein folding into a specific 3D structure that determines its activity.\n\nA linear chain of amino acid residues is called a polypeptide. A protein contains at least one long polypeptide. Short polypeptides, containing less than 20–30 residues, are rarely considered to be proteins and are commonly called peptides. The individual amino acid residues are bonded together by peptide bonds and adjacent amino acid residues. The sequence of amino acid residues in a protein is defined by the sequence of a gene, which is encoded in the genetic code. In general, the genetic code specifies 20 standard amino acids; but in certain organisms the genetic code can include selenocysteine and—in certain archaea—pyrrolysine. Shortly after or even during synthesis, the residues in a protein are often chemically modified by post-translational modification, which alters the physical and chemical properties, folding, stability, activity, and ultimately, the function of the proteins. Some proteins have non-peptide groups attached, which can be called prosthetic groups or cofactors. Proteins can work together to achieve a particular function, and they often associate to form stable protein complexes.\n\nOnce formed, proteins only exist for a certain period and are then degraded and recycled by the cell's machinery through the process of protein turnover. A protein's lifespan is measured in terms of its half-life and covers a wide range. They can exist for minutes or years with an average lifespan of 1-2 days in mammalian cells. Abnormal or misfolded proteins are degraded more rapidly either due to being targeted for destruction or due to being unstable.\n\nLike other biological macromolecules such as polysaccharides and nucleic acids, proteins are essential parts of organisms and participate in virtually every process within cells. Many proteins are enzymes that catalyse biochemical reactions and are vital to metabolism. Some proteins have structural or mechanical functions, such as actin and myosin in muscle, and the cytoskeleton's scaffolding proteins that maintain cell shape. Other proteins are important in cell signaling, immune responses, cell adhesion, and the cell cycle. In animals, proteins are needed in the diet to provide the essential amino acids that cannot be synthesized. Digestion breaks the proteins down for metabolic use.",
            "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life.\n\nThe two DNA strands are known as polynucleotides as they are composed of simpler monomeric units called nucleotides.[2][3] Each nucleotide is composed of one of four nitrogen-containing nucleobases (cytosine [C], guanine [G], adenine [A] or thymine [T]), a sugar called deoxyribose, and a phosphate group. The nucleotides are joined to one another in a chain by covalent bonds (known as the phosphodiester linkage) between the sugar of one nucleotide and the phosphate of the next, resulting in an alternating sugar-phosphate backbone. The nitrogenous bases of the two separate polynucleotide strands are bound together, according to base pairing rules (A with T and C with G), with hydrogen bonds to make double-stranded DNA. The complementary nitrogenous bases are divided into two groups, the single-ringed pyrimidines and the double-ringed purines. In DNA, the pyrimidines are thymine and cytosine; the purines are adenine and guanine.\n\nBoth strands of double-stranded DNA store the same biological information. This information is replicated when the two strands separate. A large part of DNA (more than 98% for humans) is non-coding, meaning that these sections do not serve as patterns for protein sequences. The two strands of DNA run in opposite directions to each other and are thus antiparallel. Attached to each sugar is one of four types of nucleobases (or bases). It is the sequence of these four nucleobases along the backbone that encodes genetic information. RNA strands are created using DNA strands as a template in a process called transcription, where DNA bases are exchanged for their corresponding bases except in the case of thymine (T), for which RNA substitutes uracil (U).[4] Under the genetic code, these RNA strands specify the sequence of amino acids within proteins in a process called translation.\n\nWithin eukaryotic cells, DNA is organized into long structures called chromosomes. Before typical cell division, these chromosomes are duplicated in the process of DNA replication, providing a complete set of chromosomes for each daughter cell. Eukaryotic organisms (animals, plants, fungi and protists) store most of their DNA inside the cell nucleus as nuclear DNA, and some in the mitochondria as mitochondrial DNA or in chloroplasts as chloroplast DNA.[5] In contrast, prokaryotes (bacteria and archaea) store their DNA only in the cytoplasm, in circular chromosomes. Within eukaryotic chromosomes, chromatin proteins, such as histones, compact and organize DNA. These compacting structures guide the interactions between DNA and other proteins, helping control which parts of the DNA are transcribed.",
            "Lipids are a broad group of organic compounds which include fats, waxes, sterols, fat-soluble vitamins (such as vitamins A, D, E and K), monoglycerides, diglycerides, phospholipids, and others. The functions of lipids include storing energy, signaling, and acting as structural components of cell membranes.[3][4] Lipids have applications in the cosmetic and food industries, and in nanotechnology.[5]\n\nLipids may be broadly defined as hydrophobic or amphiphilic small molecules; the amphiphilic nature of some lipids allows them to form structures such as vesicles, multilamellar/unilamellar liposomes, or membranes in an aqueous environment. Biological lipids originate entirely or in part from two distinct types of biochemical subunits or building-blocks: ketoacyl and isoprene groups.[3] Using this approach, lipids may be divided into eight categories: fatty acyls, glycerolipids, glycerophospholipids, sphingolipids, saccharolipids, and polyketides (derived from condensation of ketoacyl subunits); and sterol lipids and prenol lipids (derived from condensation of isoprene subunits).[3]\n\nAlthough the term lipid is sometimes used as a synonym for fats, fats are a subgroup of lipids called triglycerides. Lipids also encompass molecules such as fatty acids and their derivatives (including tri-, di-, monoglycerides, and phospholipids), as well as other sterol-containing metabolites such as cholesterol.[6] Although humans and other mammals use various biosynthetic pathways both to break down and to synthesize lipids, some essential lipids cannot be made this way and must be obtained from the diet.\n\n",
            "The cell is the basic structural and functional unit of all forms of life. Every cell consists of cytoplasm enclosed within a membrane; many cells contain organelles, each with a specific function. The term comes from the Latin word cellula meaning 'small room'. Most cells are only visible under a microscope. Cells emerged on Earth about 4 billion years ago. All cells are capable of replication, protein synthesis, and motility.\n\nCells are broadly categorized into two types: eukaryotic cells, which possess a nucleus, and prokaryotic cells, which lack a nucleus but have a nucleoid region. Prokaryotes are single-celled organisms such as bacteria, whereas eukaryotes can be either single-celled, such as amoebae, or multicellular, such as some algae, plants, animals, and fungi. Eukaryotic cells contain organelles including mitochondria, which provide energy for cell functions, chloroplasts, which in plants create sugars by photosynthesis, and ribosomes, which synthesise proteins.\n\nCells were discovered by Robert Hooke in 1665, who named them after their resemblance to cells inhabited by Christian monks in a monastery. Cell theory, developed in 1839 by Matthias Jakob Schleiden and Theodor Schwann, states that all organisms are composed of one or more cells, that cells are the fundamental unit of structure and function in all living organisms, and that all cells come from pre-existing cells.",
        ];
        let documents_arr: ArrayRef = Arc::new(StringArray::from(documents_vec));
        let document_ids_vec = vec!["Proteins", "DNA", "Lipids", "Cells"];
        let document_id_arr: ArrayRef = Arc::new(StringArray::from(document_ids_vec));
        let chunk_id_vec = vec!["0", "0", "0", "0"];
        let chunk_id_arr: ArrayRef = Arc::new(StringArray::from(chunk_id_vec));

        let batch = RecordBatch::try_from_iter(vec![
            ("chunk_id", chunk_id_arr),
            ("document_id", document_id_arr),
            ("text", documents_arr),
        ])?;
        let table = ArrowTableBuilder::new()
            .with_name(doc_rag_session.state_documents_table_name)
            .with_record_batches(vec![batch])?
            .build()?;

        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(doc_rag_session.state_documents_table_name)
            .with_subject(doc_rag_session.state_documents_table_name)
            .with_publisher(doc_rag_session.session_context_name)
            .with_message(table)
            .with_update(&ArrowTablePublish::Extend {
                table_name: doc_rag_session.state_documents_table_name.to_string(),
            })
            .build()?;
        let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Create the query message
        let query_str = "What are the four molecules that compose DNA?";
        let mut query_vec = Vec::new();
        if cfg!(feature = "candle") {
            // DM: note that the prompt for the query is specific to Qwen!
            let query_embed_str = format!(
                "{}{}",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                query_str
            );
            query_vec.push(query_embed_str);
        } else {
            query_vec.push(query_str.to_string());
        }
        let query_arr: ArrayRef = Arc::new(StringArray::from(query_vec));
        let query_ids_vec = vec!["question_1"];
        let query_id_arr: ArrayRef = Arc::new(StringArray::from(query_ids_vec));

        let batch =
            RecordBatch::try_from_iter(vec![("query_id", query_id_arr), ("text", query_arr)])?;
        let table = ArrowTableBuilder::new()
            .with_name(doc_rag_session.state_queries_table_name)
            .with_record_batches(vec![batch])?
            .build()?;

        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(doc_rag_session.state_queries_table_name)
            .with_subject(doc_rag_session.state_queries_table_name)
            .with_publisher(doc_rag_session.session_context_name)
            .with_message(table)
            .with_update(&ArrowTablePublish::Extend {
                table_name: doc_rag_session.state_queries_table_name.to_string(),
            })
            .build()?;
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Make the system prompt and add the user query
        let message_builder = ArrowTableBuilder::new()
            .with_name(doc_rag_session.state_messages_table_name)
            .insert_system_template_str("You are a helpful assistant.")?
            .append_new_user_query_str(query_str, "user")?;

        // Build the current message state
        let incoming_message = ArrowIncomingMessageBuilder::new()
            .with_name(doc_rag_session.message_aggregator_task_name)
            .with_subject(doc_rag_session.message_aggregator_task_name)
            .with_publisher(doc_rag_session.session_context_name)
            .with_message(message_builder.clone().build()?)
            .with_update(&ArrowTablePublish::Extend {
                table_name: doc_rag_session.message_aggregator_task_name.to_string(),
            })
            .build()?;
        incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

        // Run the session
        let session_stream =
            SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));

        // Skip actually running the session as it takes too long on the CPU
        //     until a smaller embedding model is supported (i.e., QuantBERT)
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
                "Top K: {:?}",
                session_stream_state
                    .try_read()
                    .unwrap()
                    .get_session_context()
                    .get_states()
                    .get(doc_rag_session.state_top_k_docs_table_name)
                    .unwrap()
            );

            // Update the chat history with the response
            let json_data = response
                .last_mut()
                .unwrap()
                .remove(&format!(
                    "from_{}_on_{}",
                    doc_rag_session.session_context_name, doc_rag_session.state_messages_table_name
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
                        == doc_rag_session.document_chunk_processor_1_name
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
        }

        Ok(())
    }
}
