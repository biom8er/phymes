use std::{fmt::Display, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::datatypes::DataType;
use clap::ValueEnum;
use phymes_core::{
    AvailableSubjects, DataEncoding, DataFormat, DiffType, MappableTrait, ObjectStorageBackend, ProcessorBuilder, ProcessorEcho, ProcessorTrait, Subject, WorkspacePatchSubject, test_processor::{ProcessorError, ProcessorMock}
};
use phymes_data::{
    AggregatorProcessor, AvailableCandleOperators, AvailableJinja2Templates, CandleDataProcessor, CoalesceProcessor, DataAggregatorOperator, DataCastOperator, DataColumnOperator, DataComparatorOperator, DataComparatorPredicate, DataConfig, DataConfigTrait, DataDistanceOperator, DataJoinOperator, DataStreamManager, LimitConfig, LimitProcessor, ObjectStoreConfig, ObjectStoreOptsType, ObjectStoreProcessor, ToolTrait
};
#[cfg(feature = "api")]
use phymes_data::{
    CommandSandboxConfig, CommandSandboxEnvironments, CommandSandboxProcessor,
    CommandSandboxRunners, DataIOMethod, HTTPClientConfig, HTTPClientRequestProcessor,
    HTTPClientRequestSchemas, HTTPClientRequestType,
};
use phymes_ml::{
    AvailableCandleAssets, CandleChatConfig, CandleChatProcessor, CandleEmbedConfig,
    CandleEmbedProcessor, MessageParserProcessor, ToolCallConfig, ToolCallProcessor,
};
#[cfg(feature = "api")]
use phymes_ml::{AvailableOpenAIAssets, OpenAIChatProcessor, OpenAIEmbedProcessor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The available [ProcessorTrait]s
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableProcessors {
    #[value(name = "ProcessorError")]
    ProcessorError,
    #[value(name = "ProcessorMock")]
    ProcessorMock,
    #[value(name = "ProcessorEcho")]
    #[default]
    ProcessorEcho,
    #[value(name = "CandleDataProcessor")]
    CandleDataProcessor,
    #[value(name = "ApplyTemplate")]
    ApplyTemplate,
    #[value(name = "VectorDistance")]
    VectorDistance,
    #[value(name = "Sort")]
    Sort,
    #[value(name = "HumanInTheLoop")]
    HumanInTheLoop,
    #[value(name = "ChunkDocuments")]
    ChunkDocuments,
    #[value(name = "Join")]
    Join,
    #[value(name = "ExtractPDF")]
    ExtractPDF,
    #[value(name = "GroupBy")]
    GroupBy,
    #[value(name = "Filter")]
    Filter,
    #[value(name = "ExtractTabular")]
    ExtractTabular,
    #[value(name = "Select")]
    Select,
    #[value(name = "Pivot")]
    Pivot,
    #[value(name = "ExtractXML")]
    ExtractXML,
    #[value(name = "Melt")]
    Melt,
    #[value(name = "NormalizeTime")]
    NormalizeTime,
    #[value(name = "PackTabular")]
    PackTabular,
    #[value(name = "Patch")]
    Patch,
    #[value(name = "Diff")]
    Diff,
    #[value(name = "CoalesceProcessor")]
    CoalesceProcessor,
    #[value(name = "LimitProcessor")]
    LimitProcessor,
    #[value(name = "AggregatorProcessor")]
    AggregatorProcessor,
    #[value(name = "CandleChatProcessor")]
    CandleChatProcessor,
    #[value(name = "MessageParserProcessor")]
    MessageParserProcessor,
    #[value(name = "ToolCallProcessor")]
    ToolCallProcessor,
    #[value(name = "CandleEmbedProcessor")]
    CandleEmbedProcessor,
    #[value(name = "ObjectStoreProcessor")]
    ObjectStoreProcessor,
    #[cfg(feature = "api")]
    #[value(name = "HTTPClientRequestProcessor")]
    HTTPClientRequestProcessor,
    #[cfg(feature = "api")]
    #[value(name = "CommandSandboxProcessor")]
    CommandSandboxProcessor,
    #[cfg(feature = "api")]
    #[value(name = "OpenAIChatProcessor")]
    OpenAIChatProcessor,
    #[cfg(feature = "api")]
    #[value(name = "OpenAIEmbedProcessor")]
    OpenAIEmbedProcessor,
}

impl Display for AvailableProcessors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApplyTemplate => write!(f, "{}", AvailableCandleOperators::ApplyTemplate),
            Self::VectorDistance => write!(f, "{}", AvailableCandleOperators::VectorDistance),
            Self::Sort => {
                write!(f, "{}", AvailableCandleOperators::Sort)
            }
            Self::HumanInTheLoop => write!(f, "{}", AvailableCandleOperators::HumanInTheLoop),
            Self::ChunkDocuments => write!(f, "{}", AvailableCandleOperators::ChunkDocuments),
            Self::Join => write!(f, "{}", AvailableCandleOperators::Join),
            Self::ExtractPDF => write!(f, "{}", AvailableCandleOperators::ExtractPDF),
            Self::GroupBy => {
                write!(f, "{}", AvailableCandleOperators::GroupBy)
            }
            Self::Filter => {
                write!(f, "{}", AvailableCandleOperators::Filter)
            }
            Self::ExtractTabular => {
                write!(f, "{}", AvailableCandleOperators::ExtractTabular)
            }
            Self::Select => write!(f, "{}", AvailableCandleOperators::Select),
            Self::Pivot => write!(f, "{}", AvailableCandleOperators::Pivot),
            Self::ExtractXML => write!(f, "{}", AvailableCandleOperators::ExtractXML),
            Self::Melt => write!(f, "{}", AvailableCandleOperators::Melt),
            Self::NormalizeTime => write!(f, "{}", AvailableCandleOperators::NormalizeTime),
            Self::PackTabular => write!(f, "{}", AvailableCandleOperators::PackTabular),
            Self::Patch => write!(f, "{}", AvailableCandleOperators::Patch),
            Self::Diff => write!(f, "{}", AvailableCandleOperators::Diff),
            Self::ProcessorError => write!(f, "{}", ProcessorError::get_static_name()),
            Self::ProcessorMock => write!(f, "{}", ProcessorMock::get_static_name()),
            Self::ProcessorEcho => write!(f, "{}", ProcessorEcho::get_static_name()),
            Self::CandleDataProcessor => write!(f, "{}", CandleDataProcessor::get_static_name()),
            Self::CoalesceProcessor => write!(f, "{}", CoalesceProcessor::get_static_name()),
            Self::LimitProcessor => write!(f, "{}", LimitProcessor::get_static_name()),
            Self::AggregatorProcessor => {
                write!(f, "{}", AggregatorProcessor::get_static_name())
            }
            Self::CandleChatProcessor => write!(f, "{}", CandleChatProcessor::get_static_name()),
            Self::MessageParserProcessor => {
                write!(f, "{}", MessageParserProcessor::get_static_name())
            }
            Self::ToolCallProcessor => {
                write!(f, "{}", ToolCallProcessor::get_static_name())
            }
            Self::CandleEmbedProcessor => write!(f, "{}", CandleEmbedProcessor::get_static_name()),
            Self::ObjectStoreProcessor => write!(f, "{}", ObjectStoreProcessor::get_static_name()),
            #[cfg(feature = "api")]
            Self::HTTPClientRequestProcessor => {
                write!(f, "{}", HTTPClientRequestProcessor::get_static_name())
            }
            #[cfg(feature = "api")]
            Self::CommandSandboxProcessor => {
                write!(f, "{}", CommandSandboxProcessor::get_static_name())
            }
            #[cfg(feature = "api")]
            Self::OpenAIChatProcessor => write!(f, "{}", OpenAIChatProcessor::get_static_name()),
            #[cfg(feature = "api")]
            Self::OpenAIEmbedProcessor => write!(f, "{}", OpenAIEmbedProcessor::get_static_name()),
        }
    }
}

impl DataConfigTrait for AvailableProcessors {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error>
    where
        Self: Serialize,
    {
        match self {
            Self::ProcessorMock | Self::ProcessorError => {
                serde_json::to_vec(&DataConfig::default())
            } // Just for testing purposes...
            Self::ProcessorEcho => Ok(Vec::new()),
            Self::CandleDataProcessor => serde_json::to_vec(&DataConfig::default()),
            Self::ApplyTemplate => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                doc_template: Some(AvailableJinja2Templates::default()),
                doc_name: Some("doc_name".to_string()),
                doc_input: Some("{}".to_string()),
                encoding: Some(DataEncoding::default()),
                format: Some(DataFormat::Html),
                schema: Some(AvailableSubjects::default()),
                cpu: false,
                operator: AvailableCandleOperators::ApplyTemplate,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::VectorDistance => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_pk: Some("lhs_pk".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                rhs_name: Some("rhs_name".to_string()),
                rhs_pk: Some("rhs_pk".to_string()),
                rhs_values: Some(vec!["rhs_values".to_string()]),
                dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
                cpu: false,
                operator: AvailableCandleOperators::VectorDistance,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::Sort => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                asc: Some(true),
                cpu: false,
                operator: AvailableCandleOperators::Sort,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::HumanInTheLoop => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                cpu: false,
                operator: AvailableCandleOperators::HumanInTheLoop,
                ..Default::default()
            }),
            Self::ChunkDocuments => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_pk: Some("lhs_pk".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                chunk_size: Some(512),
                chunk_overlap: Some(64),
                cpu: false,
                operator: AvailableCandleOperators::ChunkDocuments,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::Join => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                rhs_name: Some("rhs_name".to_string()),
                lhs_pk: Some("lhs_pk".to_string()),
                rhs_pk: Some("rhs_pk".to_string()),
                lhs_fk: Some("lhs_fk".to_string()),
                rhs_fk: Some("rhs_fk".to_string()),
                cpu: false,
                operator: AvailableCandleOperators::Join,
                join_operators: Some(DataJoinOperator::Inner),
                lhs_stream: DataStreamManager::Accumulate,
                rhs_stream: Some(DataStreamManager::Accumulate),
                ..Default::default()
            }),
            Self::ExtractPDF => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_pk: Some("lhs_pk".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::ExtractPDF,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::GroupBy => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                agg_columns: Some(vec!["agg_columns".to_string()]),
                agg_operators: Some(vec![DataAggregatorOperator::Sum]),
                cpu: false,
                operator: AvailableCandleOperators::GroupBy,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::Filter => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                cmp_columns: Some(vec!["cmp_columns".to_string()]),
                cmp_operators: Some(vec![DataComparatorOperator::Equals]),
                cmp_predicate: Some(DataComparatorPredicate::All),
                cpu: false,
                operator: AvailableCandleOperators::Filter,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::ExtractTabular => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                encoding: Some(DataEncoding::default()),
                format: Some(DataFormat::CsvDefault),
                schema: Some(AvailableSubjects::default()),
                cpu: false,
                operator: AvailableCandleOperators::ExtractTabular,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::ExtractXML => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                format: Some(DataFormat::Owl),
                schema: Some(AvailableSubjects::default()),
                cpu: false,
                operator: AvailableCandleOperators::ExtractXML,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::Select => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                rhs_values: Some(vec!["rhs_values".to_string()]),
                as_columns: Some(vec!["as_columns".to_string()]),
                column_operators: Some(vec![DataColumnOperator::None]),
                cast_operators: Some(vec![DataCastOperator::None]),
                cast_datatypes: Some(vec![DataType::Utf8.to_string()]),
                cast_templates: Some(vec!["cast_template".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::Select,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::Pivot => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                agg_columns: Some(vec!["agg_columns".to_string()]),
                agg_operators: Some(vec![DataAggregatorOperator::Sum]),
                default_values: Some(vec!["0".to_string()]),
                pvt_columns: Some(vec!["pvt_columns".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::Pivot,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::Melt => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                pvt_columns: Some(vec!["pvt_columns".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::Melt,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::NormalizeTime => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::NormalizeTime,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::PackTabular => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                encoding: Some(DataEncoding::default()),
                format: Some(DataFormat::None),
                doc_name: Some("doc_name".to_string()),
                schema: Some(AvailableSubjects::default()),
                cpu: false,
                operator: AvailableCandleOperators::PackTabular,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::Patch => serde_json::to_vec(&DataConfig {
                lhs_name: Some("workspace".to_string()),
                rhs_name: Some("patches".to_string()),
                lhs_values: Some(vec!["content".to_string()]),
                rhs_values: Some(vec!["diff".to_string(), "operator".to_string()]),
                lhs_pk: Some("path".to_string()),
                rhs_pk: Some("filename".to_string()),
                doc_patch: Some(serde_json::to_string(&[WorkspacePatchSubject {
                    filename: "filename".to_string(),
                    diff: "@@ content\n+new content\n".to_string(),
                    operator: "Update".to_string(),
                }])?),
                cpu: false,
                operator: AvailableCandleOperators::Patch,
                lhs_stream: DataStreamManager::Accumulate,
                rhs_stream: Some(DataStreamManager::Accumulate),
                ..Default::default()
            }),
            Self::Diff => serde_json::to_vec(&DataConfig {
                lhs_name: Some("workspace_1".to_string()),
                rhs_name: Some("workspace_2".to_string()),
                lhs_values: Some(vec!["col_1".to_string()]),
                rhs_values: Some(vec!["col_1".to_string()]),
                lhs_pk: Some("pk".to_string()),
                rhs_pk: Some("pk".to_string()),
                diff: Some(DiffType::default()),
                cpu: false,
                operator: AvailableCandleOperators::Diff,
                lhs_stream: DataStreamManager::Accumulate,
                rhs_stream: Some(DataStreamManager::Accumulate),
                ..Default::default()
            }),
            Self::LimitProcessor => serde_json::to_vec(&LimitConfig {
                skip: Some(0),
                fetch: 100,
            }),
            Self::CoalesceProcessor => serde_json::to_vec(&LimitConfig {
                fetch: 100,
                ..Default::default()
            }),
            Self::AggregatorProcessor => serde_json::to_vec(&DataConfig {
                lhs_values: Some(vec!["timestamp".to_string()]),
                asc: Some(true),
                cpu: false,
                operator: AvailableCandleOperators::Sort,
                lhs_stream: DataStreamManager::Accumulate,
                ..Default::default()
            }),
            Self::CandleChatProcessor => serde_json::to_vec(&CandleChatConfig {
                messages: "messages".to_string(),
                tools: Some("tools".to_string()),
                max_tokens: 1000,
                temperature: 0.8,
                seed: 299792458,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
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
            }),
            Self::MessageParserProcessor => serde_json::to_vec(&CandleChatConfig {
                messages: "messages".to_string(),
                max_tokens: 1000,
                temperature: 0.8,
                seed: 299792458,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
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
            }),
            Self::ToolCallProcessor => serde_json::to_vec(&ToolCallConfig {
                subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
                subject_names: vec!["processor_1".to_string()],
                subscription_table_names: vec!["lhs_name".to_string()],
                ..Default::default()
            }),
            Self::CandleEmbedProcessor => serde_json::to_vec(&CandleEmbedConfig {
                documents: "documents".to_string(),
                encoding_format: "float".to_string(),
                modality: "text".to_string(),
                input_type: "query".to_string(),
                weights_config_file: Some(format!(
                    "{}/.cache/hf/models--sentence-transformers--all-MiniLM-L6-v2/config.json",
                    std::env::var("HOME").unwrap_or("".to_string())
                )),
                weights_file: Some(format!(
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
                candle_asset: Some(AvailableCandleAssets::QuantizedBertEmbed),
                ..Default::default()
            }),
            Self::ObjectStoreProcessor => serde_json::to_vec(&ObjectStoreConfig {
                timeout: 15,
                ops_type: ObjectStoreOptsType::default(),
                backend: ObjectStorageBackend::default(),
                bucket: Some("bucket".to_string()),
                backend_config: Some(Map::<String, Value>::new()),
                locations: Some(vec!["location".to_string()]),
                subject_name: Some("subject_name".to_string()),
                ..Default::default()
            }),
            #[cfg(feature = "api")]
            Self::HTTPClientRequestProcessor => serde_json::to_vec(&HTTPClientConfig {
                timeout: 5,
                request_type: HTTPClientRequestType::Get,
                user_agent_type: Some("rust-openalex-client/2.0".to_string()),
                base_url: "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?".to_string(),
                subject_name: Some("messages".to_string()),
                request_schema: HTTPClientRequestSchemas::Messages,
                json: Some("db=pubmed&retmode=json&retmax=5&mindate=2020&maxdate=2023".to_string()),
                ..Default::default()
            }),
            #[cfg(feature = "api")]
            Self::CommandSandboxProcessor => serde_json::to_vec(&CommandSandboxConfig {
                runner: CommandSandboxRunners::Docker,
                environment: CommandSandboxEnvironments::Bash,
                container_image: "alpine".to_string(),
                data_i: DataIOMethod::None,
                data_o: DataIOMethod::None,
                command: Some("echo".to_string()),
                timeout: 5,
                cli_args: Some(vec!["Hello from Docker!".to_string()]),
                subject_name: Some("subject_name".to_string()),
                workspace_name: Some("workspace".to_string()),
                ..Default::default()
            }),
            #[cfg(feature = "api")]
            Self::OpenAIChatProcessor => serde_json::to_vec(&CandleChatConfig {
                messages: "messages".to_string(),
                max_tokens: 1000,
                temperature: 0.8,
                seed: 299792458,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
                candle_asset: None,
                openai_asset: Some(AvailableOpenAIAssets::MetaLlamaV3p2_1B),
                weights_config_file: None,
                weights_file: None,
                tokenizer_file: None,
                tokenizer_config_file: None,
                api_url: Some("http://0.0.0.0:8000/v1".to_string()),
                ..Default::default()
            }),
            #[cfg(feature = "api")]
            Self::OpenAIEmbedProcessor => serde_json::to_vec(&CandleEmbedConfig {
                documents: "documents".to_string(),
                encoding_format: "float".to_string(),
                modality: "text".to_string(),
                input_type: "query".to_string(),
                openai_asset: Some(AvailableOpenAIAssets::NvidiaLlamaV3p2NvEmbedQA1BV2),
                api_url: Some("http://0.0.0.0:8001/v1".to_string()),
                candle_asset: None,
                ..Default::default()
            }),
        }
    }
    fn from_table(_table: &Subject) -> Result<Self>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl ToolTrait for AvailableProcessors {
    fn get_description(&self) -> String {
        match self {
            Self::ProcessorError => todo!(),
            Self::ProcessorEcho => todo!(),
            Self::ProcessorMock => todo!(),
            Self::CandleDataProcessor => todo!(),
            Self::ChunkDocuments => AvailableCandleOperators::ChunkDocuments.get_description(),
            Self::ExtractPDF => AvailableCandleOperators::ExtractPDF.get_description(),
            Self::ExtractTabular => AvailableCandleOperators::ExtractTabular.get_description(),
            Self::Filter => AvailableCandleOperators::Filter.get_description(),
            Self::GroupBy => AvailableCandleOperators::GroupBy.get_description(),
            Self::HumanInTheLoop => AvailableCandleOperators::HumanInTheLoop.get_description(),
            Self::Join => AvailableCandleOperators::Join.get_description(),
            Self::NormalizeTime => AvailableCandleOperators::NormalizeTime.get_description(),
            Self::Pivot => AvailableCandleOperators::Pivot.get_description(),
            Self::ExtractXML => AvailableCandleOperators::ExtractXML.get_description(),
            Self::Melt => AvailableCandleOperators::Melt.get_description(),
            Self::Select => AvailableCandleOperators::Select.get_description(),
            Self::Sort => AvailableCandleOperators::Sort.get_description(),
            Self::VectorDistance => AvailableCandleOperators::VectorDistance.get_description(),
            Self::ApplyTemplate => AvailableCandleOperators::ApplyTemplate.get_description(),
            Self::PackTabular => AvailableCandleOperators::PackTabular.get_description(),
            Self::Patch => AvailableCandleOperators::Patch.get_description(),
            Self::Diff => AvailableCandleOperators::Diff.get_description(),
            Self::AggregatorProcessor => todo!(),
            Self::CoalesceProcessor => todo!(),
            Self::LimitProcessor => todo!(),
            Self::CandleChatProcessor => todo!(),
            Self::MessageParserProcessor => todo!(),
            Self::ToolCallProcessor => todo!(),
            Self::CandleEmbedProcessor => todo!(),
            Self::ObjectStoreProcessor => todo!(),
            #[cfg(feature = "api")]
            Self::HTTPClientRequestProcessor => todo!(),
            #[cfg(feature = "api")]
            Self::CommandSandboxProcessor => todo!(),
            #[cfg(feature = "api")]
            Self::OpenAIChatProcessor => todo!(),
            #[cfg(feature = "api")]
            Self::OpenAIEmbedProcessor => todo!(),
        }
    }
    fn to_json_tool_schema(&self) -> String {
        match self {
            Self::ProcessorError => todo!(),
            Self::ProcessorEcho => todo!(),
            Self::ProcessorMock => todo!(),
            Self::CandleDataProcessor => todo!(),
            Self::ChunkDocuments => AvailableCandleOperators::ChunkDocuments.to_json_tool_schema(),
            Self::ExtractPDF => AvailableCandleOperators::ExtractPDF.to_json_tool_schema(),
            Self::ExtractTabular => AvailableCandleOperators::ExtractTabular.to_json_tool_schema(),
            Self::Filter => AvailableCandleOperators::Filter.to_json_tool_schema(),
            Self::GroupBy => AvailableCandleOperators::GroupBy.to_json_tool_schema(),
            Self::HumanInTheLoop => AvailableCandleOperators::HumanInTheLoop.to_json_tool_schema(),
            Self::Join => AvailableCandleOperators::Join.to_json_tool_schema(),
            Self::NormalizeTime => AvailableCandleOperators::NormalizeTime.to_json_tool_schema(),
            Self::Pivot => AvailableCandleOperators::Pivot.to_json_tool_schema(),
            Self::ExtractXML => AvailableCandleOperators::ExtractXML.to_json_tool_schema(),
            Self::Melt => AvailableCandleOperators::Melt.to_json_tool_schema(),
            Self::Select => AvailableCandleOperators::Select.to_json_tool_schema(),
            Self::Sort => AvailableCandleOperators::Sort.to_json_tool_schema(),
            Self::VectorDistance => AvailableCandleOperators::VectorDistance.to_json_tool_schema(),
            Self::ApplyTemplate => AvailableCandleOperators::ApplyTemplate.to_json_tool_schema(),
            Self::PackTabular => AvailableCandleOperators::PackTabular.to_json_tool_schema(),
            Self::Patch => AvailableCandleOperators::Patch.to_json_tool_schema(),
            Self::Diff => AvailableCandleOperators::Diff.to_json_tool_schema(),
            Self::AggregatorProcessor => todo!(),
            Self::CoalesceProcessor => todo!(),
            Self::LimitProcessor => todo!(),
            Self::CandleChatProcessor => todo!(),
            Self::MessageParserProcessor => todo!(),
            Self::ToolCallProcessor => todo!(),
            Self::CandleEmbedProcessor => todo!(),
            Self::ObjectStoreProcessor => todo!(),
            #[cfg(feature = "api")]
            Self::HTTPClientRequestProcessor => todo!(),
            #[cfg(feature = "api")]
            Self::CommandSandboxProcessor => todo!(),
            #[cfg(feature = "api")]
            Self::OpenAIChatProcessor => todo!(),
            #[cfg(feature = "api")]
            Self::OpenAIEmbedProcessor => todo!(),
        }
    }
}

impl AvailableProcessors {
    /// Get all available processor plans
    pub fn all_varient_names() -> Vec<String> {
        let processor_names = [
            AvailableProcessors::ProcessorError.to_string(),
            AvailableProcessors::ProcessorMock.to_string(),
            AvailableProcessors::ProcessorEcho.to_string(),
            AvailableProcessors::CandleDataProcessor.to_string(),
            AvailableProcessors::VectorDistance.to_string(),
            AvailableProcessors::ApplyTemplate.to_string(),
            AvailableProcessors::Sort.to_string(),
            AvailableProcessors::HumanInTheLoop.to_string(),
            AvailableProcessors::ChunkDocuments.to_string(),
            AvailableProcessors::Join.to_string(),
            AvailableProcessors::ExtractPDF.to_string(),
            AvailableProcessors::GroupBy.to_string(),
            AvailableProcessors::Filter.to_string(),
            AvailableProcessors::ExtractTabular.to_string(),
            AvailableProcessors::Select.to_string(),
            AvailableProcessors::Pivot.to_string(),
            AvailableProcessors::ExtractXML.to_string(),
            AvailableProcessors::Melt.to_string(),
            AvailableProcessors::NormalizeTime.to_string(),
            AvailableProcessors::PackTabular.to_string(),
            AvailableProcessors::Patch.to_string(),
            AvailableProcessors::Diff.to_string(),
            AvailableProcessors::CoalesceProcessor.to_string(),
            AvailableProcessors::LimitProcessor.to_string(),
            AvailableProcessors::AggregatorProcessor.to_string(),
            AvailableProcessors::CandleChatProcessor.to_string(),
            AvailableProcessors::MessageParserProcessor.to_string(),
            AvailableProcessors::ToolCallProcessor.to_string(),
            AvailableProcessors::CandleEmbedProcessor.to_string(),
            AvailableProcessors::ObjectStoreProcessor.to_string(),
            #[cfg(feature = "api")]
            AvailableProcessors::HTTPClientRequestProcessor.to_string(),
            #[cfg(feature = "api")]
            AvailableProcessors::CommandSandboxProcessor.to_string(),
            #[cfg(feature = "api")]
            AvailableProcessors::OpenAIChatProcessor.to_string(),
            #[cfg(feature = "api")]
            AvailableProcessors::OpenAIEmbedProcessor.to_string(),
        ];
        processor_names
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// New [AvailableProcessors] from a short name identifying the variant contained in a [String]
    pub fn from_str_fuzzy(line: &str) -> Result<Self> {
        if line.contains(&AvailableProcessors::ProcessorMock.to_string()) {
            Ok(AvailableProcessors::ProcessorMock)
        } else if line.contains(&AvailableProcessors::ProcessorError.to_string()) {
            Ok(AvailableProcessors::ProcessorError)
        } else if line.contains(&AvailableProcessors::ProcessorEcho.to_string()) {
            Ok(AvailableProcessors::ProcessorEcho)
        } else if line.contains(&AvailableProcessors::CandleDataProcessor.to_string()) {
            Ok(AvailableProcessors::CandleDataProcessor)
        } else if line.contains(&AvailableProcessors::Pivot.to_string()) {
            Ok(AvailableProcessors::Pivot)
        } else if line.contains(&AvailableProcessors::ExtractXML.to_string()) {
            Ok(AvailableProcessors::ExtractXML)
        } else if line.contains(&AvailableProcessors::Melt.to_string()) {
            Ok(AvailableProcessors::Melt)
        } else if line.contains(&AvailableProcessors::ApplyTemplate.to_string()) {
            Ok(AvailableProcessors::ApplyTemplate)
        } else if line.contains(&AvailableProcessors::VectorDistance.to_string()) {
            Ok(AvailableProcessors::VectorDistance)
        } else if line.contains(&AvailableProcessors::Sort.to_string()) {
            Ok(AvailableProcessors::Sort)
        } else if line.contains(&AvailableProcessors::HumanInTheLoop.to_string()) {
            Ok(AvailableProcessors::HumanInTheLoop)
        } else if line.contains(&AvailableProcessors::ChunkDocuments.to_string()) {
            Ok(AvailableProcessors::ChunkDocuments)
        } else if line.contains(&AvailableProcessors::Join.to_string()) {
            Ok(AvailableProcessors::Join)
        } else if line.contains(&AvailableProcessors::ExtractPDF.to_string()) {
            Ok(AvailableProcessors::ExtractPDF)
        } else if line.contains(&AvailableProcessors::GroupBy.to_string()) {
            Ok(AvailableProcessors::GroupBy)
        } else if line.contains(&AvailableProcessors::Filter.to_string()) {
            Ok(AvailableProcessors::Filter)
        } else if line.contains(&AvailableProcessors::ExtractTabular.to_string()) {
            Ok(AvailableProcessors::ExtractTabular)
        } else if line.contains(&AvailableProcessors::Select.to_string()) {
            Ok(AvailableProcessors::Select)
        } else if line.contains(&AvailableProcessors::NormalizeTime.to_string()) {
            Ok(AvailableProcessors::NormalizeTime)
        } else if line.contains(&AvailableProcessors::CandleDataProcessor.to_string()) {
            Ok(AvailableProcessors::CandleDataProcessor)
        } else if line.contains(&AvailableProcessors::PackTabular.to_string()) {
            Ok(AvailableProcessors::PackTabular)
        } else if line.contains(&AvailableProcessors::Patch.to_string()) {
            Ok(AvailableProcessors::Patch)
        } else if line.contains(&AvailableProcessors::Diff.to_string()) {
            Ok(AvailableProcessors::Diff)
        } else if line.contains(&AvailableProcessors::CoalesceProcessor.to_string()) {
            Ok(AvailableProcessors::CoalesceProcessor)
        } else if line.contains(&AvailableProcessors::LimitProcessor.to_string()) {
            Ok(AvailableProcessors::LimitProcessor)
        } else if line.contains(&AvailableProcessors::AggregatorProcessor.to_string()) {
            Ok(AvailableProcessors::AggregatorProcessor)
        } else if line.contains(&AvailableProcessors::CandleChatProcessor.to_string()) {
            Ok(AvailableProcessors::CandleChatProcessor)
        } else if line.contains(&AvailableProcessors::MessageParserProcessor.to_string()) {
            Ok(AvailableProcessors::MessageParserProcessor)
        } else if line.contains(&AvailableProcessors::ToolCallProcessor.to_string()) {
            Ok(AvailableProcessors::ToolCallProcessor)
        } else if line.contains(&AvailableProcessors::CandleEmbedProcessor.to_string()) {
            Ok(AvailableProcessors::CandleEmbedProcessor)
        } else if line.contains(&AvailableProcessors::ObjectStoreProcessor.to_string()) {
            Ok(AvailableProcessors::ObjectStoreProcessor)
        } else {
            #[cfg(feature = "api")]
            if line.contains(&AvailableProcessors::HTTPClientRequestProcessor.to_string()) {
                Ok(AvailableProcessors::HTTPClientRequestProcessor)
            } else if line.contains(&AvailableProcessors::CommandSandboxProcessor.to_string()) {
                Ok(AvailableProcessors::CommandSandboxProcessor)
            } else if line.contains(&AvailableProcessors::OpenAIChatProcessor.to_string()) {
                Ok(AvailableProcessors::OpenAIChatProcessor)
            } else if line.contains(&AvailableProcessors::OpenAIEmbedProcessor.to_string()) {
                Ok(AvailableProcessors::OpenAIEmbedProcessor)
            } else {
                Err(anyhow!(
                    "Processor not found in {line}. Available processors are {:?}.",
                    AvailableProcessors::all_varient_names()
                ))
            }
            #[cfg(not(feature = "api"))]
            Err(anyhow!(
                "Processor not found in {line}. Available processors are {:?}.",
                AvailableProcessors::all_varient_names()
            ))
        }
    }

    /// Build the [ProcessorTrait] object
    pub fn build_arc(self, name: &str) -> Arc<dyn ProcessorTrait> {
        match self {
            Self::ProcessorError => Arc::new(ProcessorError::new(name, self.to_string().as_str())),
            Self::ProcessorMock => Arc::new(ProcessorMock::new(name, self.to_string().as_str())),
            Self::ProcessorEcho => Arc::new(ProcessorEcho::new(name, self.to_string().as_str())),
            Self::CandleDataProcessor
            | Self::ChunkDocuments
            | Self::ExtractPDF
            | Self::ExtractTabular
            | Self::Filter
            | Self::GroupBy
            | Self::HumanInTheLoop
            | Self::Join
            | Self::NormalizeTime
            | Self::Pivot
            | Self::ExtractXML
            | Self::Melt
            | Self::Select
            | Self::Sort
            | Self::VectorDistance
            | Self::ApplyTemplate
            | Self::PackTabular
            | Self::Patch 
            | Self::Diff => {
                Arc::new(CandleDataProcessor::new(name, self.to_string().as_str()))
            }
            Self::CoalesceProcessor => {
                Arc::new(CoalesceProcessor::new(name, self.to_string().as_str()))
            }
            Self::LimitProcessor => Arc::new(LimitProcessor::new(name, self.to_string().as_str())),
            Self::AggregatorProcessor => {
                Arc::new(AggregatorProcessor::new(name, self.to_string().as_str()))
            }
            Self::CandleChatProcessor => {
                Arc::new(CandleChatProcessor::new(name, self.to_string().as_str()))
            }
            Self::MessageParserProcessor => {
                Arc::new(MessageParserProcessor::new(name, self.to_string().as_str()))
            }
            Self::ToolCallProcessor => {
                Arc::new(ToolCallProcessor::new(name, self.to_string().as_str()))
            }
            Self::CandleEmbedProcessor => {
                Arc::new(CandleEmbedProcessor::new(name, self.to_string().as_str()))
            }
            Self::ObjectStoreProcessor => {
                Arc::new(ObjectStoreProcessor::new(name, self.to_string().as_str()))
            }
            #[cfg(feature = "api")]
            Self::HTTPClientRequestProcessor => Arc::new(HTTPClientRequestProcessor::new(
                name,
                self.to_string().as_str(),
            )),
            #[cfg(feature = "api")]
            Self::CommandSandboxProcessor => Arc::new(CommandSandboxProcessor::new(
                name,
                self.to_string().as_str(),
            )),
            #[cfg(feature = "api")]
            Self::OpenAIChatProcessor => {
                Arc::new(OpenAIChatProcessor::new(name, self.to_string().as_str()))
            }
            #[cfg(feature = "api")]
            Self::OpenAIEmbedProcessor => {
                Arc::new(OpenAIEmbedProcessor::new(name, self.to_string().as_str()))
            }
        }
    }

    /// Build the [ProcessorTrait] object form the [ProcessorBuilder]
    pub fn build_with_builder(self, builder: ProcessorBuilder) -> Result<Arc<dyn ProcessorTrait>> {
        match self {
            Self::ProcessorError => builder.build_arc::<ProcessorError>(),
            Self::ProcessorMock => builder.build_arc::<ProcessorMock>(),
            Self::ProcessorEcho => builder.build_arc::<ProcessorEcho>(),
            Self::CandleDataProcessor
            | Self::ChunkDocuments
            | Self::ExtractPDF
            | Self::ExtractTabular
            | Self::Filter
            | Self::GroupBy
            | Self::HumanInTheLoop
            | Self::Join
            | Self::NormalizeTime
            | Self::Pivot
            | Self::ExtractXML
            | Self::Melt
            | Self::Select
            | Self::Sort
            | Self::VectorDistance
            | Self::ApplyTemplate
            | Self::PackTabular
            | Self::Patch 
            | Self::Diff => builder.build_arc::<CandleDataProcessor>(),
            Self::CoalesceProcessor => builder.build_arc::<CoalesceProcessor>(),
            Self::LimitProcessor => builder.build_arc::<LimitProcessor>(),
            Self::AggregatorProcessor => builder.build_arc::<AggregatorProcessor>(),
            Self::CandleChatProcessor => builder.build_arc::<CandleChatProcessor>(),
            Self::MessageParserProcessor => builder.build_arc::<MessageParserProcessor>(),
            Self::ToolCallProcessor => builder.build_arc::<ToolCallProcessor>(),
            Self::CandleEmbedProcessor => builder.build_arc::<CandleEmbedProcessor>(),
            Self::ObjectStoreProcessor => builder.build_arc::<ObjectStoreProcessor>(),
            #[cfg(feature = "api")]
            Self::HTTPClientRequestProcessor => builder.build_arc::<HTTPClientRequestProcessor>(),
            #[cfg(feature = "api")]
            Self::CommandSandboxProcessor => builder.build_arc::<CommandSandboxProcessor>(),
            #[cfg(feature = "api")]
            Self::OpenAIChatProcessor => builder.build_arc::<OpenAIChatProcessor>(),
            #[cfg(feature = "api")]
            Self::OpenAIEmbedProcessor => builder.build_arc::<OpenAIEmbedProcessor>(),
        }
    }

    /// Identify the [DataConfigTrait] object for the [ProcessorTrait] object
    pub fn config_type(&self) -> &str {
        match self {
            Self::ProcessorEcho => "",
            Self::ProcessorMock
            | Self::ProcessorError
            | Self::CandleDataProcessor
            | Self::ChunkDocuments
            | Self::ExtractPDF
            | Self::ExtractTabular
            | Self::Filter
            | Self::GroupBy
            | Self::HumanInTheLoop
            | Self::Join
            | Self::NormalizeTime
            | Self::Pivot
            | Self::ExtractXML
            | Self::Melt
            | Self::Select
            | Self::Sort
            | Self::VectorDistance
            | Self::ApplyTemplate
            | Self::AggregatorProcessor
            | Self::PackTabular
            | Self::Patch 
            | Self::Diff => "DataConfig",
            Self::CoalesceProcessor | Self::LimitProcessor => "LimitConfig",
            Self::ToolCallProcessor => "ToolCallConfig",
            Self::CandleChatProcessor | Self::MessageParserProcessor => "CandleChatConfig",
            Self::CandleEmbedProcessor => "CandleEmbedConfig",
            Self::ObjectStoreProcessor => "ObjectStoreConfig",
            #[cfg(feature = "api")]
            Self::HTTPClientRequestProcessor => "HTTPClientConfig",
            #[cfg(feature = "api")]
            Self::CommandSandboxProcessor => "CommandSandboxConfig",
            #[cfg(feature = "api")]
            Self::OpenAIChatProcessor => "CandleChatConfig",
            #[cfg(feature = "api")]
            Self::OpenAIEmbedProcessor => "CandleEmbedConfig",
        }
    }
}
