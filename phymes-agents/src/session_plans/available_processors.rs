use std::{fmt::Display, sync::Arc};

use anyhow::{Result, anyhow};
use arrow::datatypes::DataType;
use clap::ValueEnum;
use phymes_core::{
    DataFormat, MappableTrait, ProcessorBuilder, ProcessorEcho, ProcessorTrait, Table,
    TablePublication, TableSubscribePolicyTrait, TableSubscription, test_processor::ProcessorMock,
};
use phymes_data::{
    AttachmentAggregatorProcessor, AvailableCandleOperators, AvailableJinja2Templates,
    CandleDataProcessor, DataAggregatorOperator, DataCastOperator, DataComparatorOperator,
    DataComparatorPredicate, DataConfig, DataConfigTrait, DataDistanceOperator, DataStreamManager,
    DataSummaryConfig, DataSummaryProcessor, ToolTrait,
};
use phymes_ml::{
    AvailableCandleAssets, CandleChatConfig, CandleChatProcessor, CandleEmbedConfig,
    CandleEmbedProcessor, MessageAggregatorProcessor, MessageParserProcessor,
};
#[cfg(feature = "openai_api")]
use phymes_ml::{AvailableOpenAIAssets, OpenAIChatProcessor, OpenAIEmbedProcessor};
use serde::{Deserialize, Serialize};

/// The available [ProcessorTrait]s
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableProcessors {
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
    #[value(name = "SortColumnAndIndices")]
    SortColumnAndIndices,
    #[value(name = "HumanInTheLoop")]
    HumanInTheLoop,
    #[value(name = "ChunkDocuments")]
    ChunkDocuments,
    #[value(name = "JoinInner")]
    JoinInner,
    #[value(name = "ExtractPDFText")]
    ExtractPDFText,
    #[value(name = "GroupByAndAggregate")]
    GroupByAndAggregate,
    #[value(name = "FilterColumnsAndIndices")]
    FilterColumnsAndIndices,
    #[value(name = "ExtractTabularData")]
    ExtractTabularData,
    #[value(name = "SelectAndCast")]
    SelectAndCast,
    #[value(name = "Pivot")]
    Pivot,
    #[value(name = "NormalizeTime")]
    NormalizeTime,
    #[value(name = "DataSummaryProcessor")]
    DataSummaryProcessor,
    #[value(name = "AttachmentAggregatorProcessor")]
    AttachmentAggregatorProcessor,
    #[value(name = "CandleChatProcessor")]
    CandleChatProcessor,
    #[value(name = "MessageAggregatorProcessor")]
    MessageAggregatorProcessor,
    #[value(name = "MessageParserProcessor")]
    MessageParserProcessor,
    #[value(name = "CandleEmbedProcessor")]
    CandleEmbedProcessor,
    #[cfg(feature = "openai_api")]
    #[value(name = "OpenAIChatProcessor")]
    OpenAIChatProcessor,
    #[cfg(feature = "openai_api")]
    #[value(name = "OpenAIEmbedProcessor")]
    OpenAIEmbedProcessor,
}

impl Display for AvailableProcessors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApplyTemplate => write!(f, "{}", AvailableCandleOperators::ApplyTemplate),
            Self::VectorDistance => write!(f, "{}", AvailableCandleOperators::VectorDistance),
            Self::SortColumnAndIndices => {
                write!(f, "{}", AvailableCandleOperators::SortColumnAndIndices)
            }
            Self::HumanInTheLoop => write!(f, "{}", AvailableCandleOperators::HumanInTheLoop),
            Self::ChunkDocuments => write!(f, "{}", AvailableCandleOperators::ChunkDocuments),
            Self::JoinInner => write!(f, "{}", AvailableCandleOperators::JoinInner),
            Self::ExtractPDFText => write!(f, "{}", AvailableCandleOperators::ExtractPDFText),
            Self::GroupByAndAggregate => {
                write!(f, "{}", AvailableCandleOperators::GroupByAndAggregate)
            }
            Self::FilterColumnsAndIndices => {
                write!(f, "{}", AvailableCandleOperators::FilterColumnsAndIndices)
            }
            Self::ExtractTabularData => {
                write!(f, "{}", AvailableCandleOperators::ExtractTabularData)
            }
            Self::SelectAndCast => write!(f, "{}", AvailableCandleOperators::SelectAndCast),
            Self::Pivot => write!(f, "{}", AvailableCandleOperators::Pivot),
            Self::NormalizeTime => write!(f, "{}", AvailableCandleOperators::NormalizeTime),
            Self::ProcessorMock => write!(f, "{}", ProcessorMock::get_static_name()),
            Self::ProcessorEcho => write!(f, "{}", ProcessorEcho::get_static_name()),
            Self::CandleDataProcessor => write!(f, "{}", CandleDataProcessor::get_static_name()),
            Self::DataSummaryProcessor => write!(f, "{}", DataSummaryProcessor::get_static_name()),
            Self::AttachmentAggregatorProcessor => {
                write!(f, "{}", AttachmentAggregatorProcessor::get_static_name())
            }
            Self::CandleChatProcessor => write!(f, "{}", CandleChatProcessor::get_static_name()),
            Self::MessageAggregatorProcessor => {
                write!(f, "{}", MessageAggregatorProcessor::get_static_name())
            }
            Self::MessageParserProcessor => {
                write!(f, "{}", MessageParserProcessor::get_static_name())
            }
            Self::CandleEmbedProcessor => write!(f, "{}", CandleEmbedProcessor::get_static_name()),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => write!(f, "{}", OpenAIChatProcessor::get_static_name()),
            #[cfg(feature = "openai_api")]
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
            Self::ProcessorMock => serde_json::to_vec(&DataConfig::default()), // Just for testing purposes...
            Self::ProcessorEcho => Ok(Vec::new()),
            Self::CandleDataProcessor => serde_json::to_vec(&DataConfig::default()),
            Self::ApplyTemplate => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                doc_template: Some(AvailableJinja2Templates::default()),
                doc_name: Some("doc_name".to_string()),
                doc_input: Some("{}".to_string()),
                format: Some(DataFormat::Html),
                cpu: false,
                operator: AvailableCandleOperators::ApplyTemplate,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
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
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::SortColumnAndIndices => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                asc: Some(true),
                cpu: false,
                operator: AvailableCandleOperators::SortColumnAndIndices,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
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
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::JoinInner => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                rhs_name: Some("rhs_name".to_string()),
                lhs_pk: Some("lhs_pk".to_string()),
                rhs_pk: Some("rhs_pk".to_string()),
                lhs_fk: Some("lhs_fk".to_string()),
                rhs_fk: Some("rhs_fk".to_string()),
                cpu: false,
                operator: AvailableCandleOperators::JoinInner,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::ExtractPDFText => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_pk: Some("lhs_pk".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::ExtractPDFText,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::GroupByAndAggregate => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                agg_columns: Some(vec!["agg_columns".to_string()]),
                agg_operators: Some(vec![DataAggregatorOperator::Sum]),
                cpu: false,
                operator: AvailableCandleOperators::GroupByAndAggregate,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::FilterColumnsAndIndices => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                cmp_columns: Some(vec!["cmp_columns".to_string()]),
                cmp_operators: Some(vec![DataComparatorOperator::Equals]),
                cmp_predicate: Some(DataComparatorPredicate::All),
                cpu: false,
                operator: AvailableCandleOperators::FilterColumnsAndIndices,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::ExtractTabularData => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                format: Some(DataFormat::None),
                cpu: false,
                operator: AvailableCandleOperators::ExtractTabularData,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::SelectAndCast => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                as_columns: Some(vec!["as_columns".to_string()]),
                cast_operators: Some(vec![DataCastOperator::None]),
                cast_datatypes: Some(vec![DataType::Utf8.to_string()]),
                cast_templates: Some(vec!["cast_template".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::SelectAndCast,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
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
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::NormalizeTime => serde_json::to_vec(&DataConfig {
                lhs_name: Some("lhs_name".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::NormalizeTime,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::DataSummaryProcessor => serde_json::to_vec(&DataSummaryConfig {
                num_rows: Some(10),
                num_batches: Some(1),
                summary_format: DataFormat::None,
                ..Default::default()
            }),
            Self::AttachmentAggregatorProcessor => serde_json::to_vec(&DataConfig {
                lhs_values: Some(vec!["timestamp".to_string()]),
                asc: Some(true),
                cpu: false,
                operator: AvailableCandleOperators::SortColumnAndIndices,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::CandleChatProcessor => serde_json::to_vec(&CandleChatConfig {
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
            Self::MessageAggregatorProcessor => serde_json::to_vec(&DataConfig {
                lhs_values: Some(vec!["timestamp".to_string()]),
                asc: Some(true),
                cpu: false,
                operator: AvailableCandleOperators::SortColumnAndIndices,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::MessageParserProcessor => serde_json::to_vec(&CandleChatConfig {
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
            Self::CandleEmbedProcessor => serde_json::to_vec(&CandleEmbedConfig {
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
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => serde_json::to_vec(&CandleChatConfig {
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
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => serde_json::to_vec(&CandleEmbedConfig {
                openai_asset: Some(AvailableOpenAIAssets::NvidiaLlamaV3p2NvEmbedQA1BV2),
                api_url: Some("http://0.0.0.0:8001/v1".to_string()),
                input_type: "query".to_string(),
                candle_asset: None,
                encoding_format: "float".to_string(),
                modality: "text".to_string(),
                ..Default::default()
            }),
        }
    }
    fn from_table(_table: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

impl ToolTrait for AvailableProcessors {
    fn get_description(&self) -> String {
        match self {
            Self::ProcessorEcho => todo!(),
            Self::ProcessorMock => todo!(),
            Self::CandleDataProcessor => todo!(),
            Self::ChunkDocuments => AvailableCandleOperators::ChunkDocuments.get_description(),
            Self::ExtractPDFText => AvailableCandleOperators::ExtractPDFText.get_description(),
            Self::ExtractTabularData => {
                AvailableCandleOperators::ExtractTabularData.get_description()
            }
            Self::FilterColumnsAndIndices => {
                AvailableCandleOperators::FilterColumnsAndIndices.get_description()
            }
            Self::GroupByAndAggregate => {
                AvailableCandleOperators::GroupByAndAggregate.get_description()
            }
            Self::HumanInTheLoop => AvailableCandleOperators::HumanInTheLoop.get_description(),
            Self::JoinInner => AvailableCandleOperators::JoinInner.get_description(),
            Self::NormalizeTime => AvailableCandleOperators::NormalizeTime.get_description(),
            Self::Pivot => AvailableCandleOperators::Pivot.get_description(),
            Self::SelectAndCast => AvailableCandleOperators::SelectAndCast.get_description(),
            Self::SortColumnAndIndices => {
                AvailableCandleOperators::SortColumnAndIndices.get_description()
            }
            Self::VectorDistance => AvailableCandleOperators::VectorDistance.get_description(),
            Self::ApplyTemplate => AvailableCandleOperators::ApplyTemplate.get_description(),
            Self::AttachmentAggregatorProcessor => todo!(),
            Self::MessageAggregatorProcessor => todo!(),
            Self::DataSummaryProcessor => todo!(),
            Self::CandleChatProcessor => todo!(),
            Self::MessageParserProcessor => todo!(),
            Self::CandleEmbedProcessor => todo!(),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => todo!(),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => todo!(),
        }
    }
    fn to_json_tool_schema(&self) -> String {
        match self {
            Self::ProcessorEcho => todo!(),
            Self::ProcessorMock => todo!(),
            Self::CandleDataProcessor => todo!(),
            Self::ChunkDocuments => AvailableCandleOperators::ChunkDocuments.to_json_tool_schema(),
            Self::ExtractPDFText => AvailableCandleOperators::ExtractPDFText.to_json_tool_schema(),
            Self::ExtractTabularData => {
                AvailableCandleOperators::ExtractTabularData.to_json_tool_schema()
            }
            Self::FilterColumnsAndIndices => {
                AvailableCandleOperators::FilterColumnsAndIndices.to_json_tool_schema()
            }
            Self::GroupByAndAggregate => {
                AvailableCandleOperators::GroupByAndAggregate.to_json_tool_schema()
            }
            Self::HumanInTheLoop => AvailableCandleOperators::HumanInTheLoop.to_json_tool_schema(),
            Self::JoinInner => AvailableCandleOperators::JoinInner.to_json_tool_schema(),
            Self::NormalizeTime => AvailableCandleOperators::NormalizeTime.to_json_tool_schema(),
            Self::Pivot => AvailableCandleOperators::Pivot.to_json_tool_schema(),
            Self::SelectAndCast => AvailableCandleOperators::SelectAndCast.to_json_tool_schema(),
            Self::SortColumnAndIndices => {
                AvailableCandleOperators::SortColumnAndIndices.to_json_tool_schema()
            }
            Self::VectorDistance => AvailableCandleOperators::VectorDistance.to_json_tool_schema(),
            Self::ApplyTemplate => AvailableCandleOperators::ApplyTemplate.to_json_tool_schema(),
            Self::AttachmentAggregatorProcessor => todo!(),
            Self::MessageAggregatorProcessor => todo!(),
            Self::DataSummaryProcessor => todo!(),
            Self::CandleChatProcessor => todo!(),
            Self::MessageParserProcessor => todo!(),
            Self::CandleEmbedProcessor => todo!(),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => todo!(),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => todo!(),
        }
    }
}

impl AvailableProcessors {
    /// Get all available processor plans
    pub fn all_varient_names() -> Vec<String> {
        let processor_names = [
            AvailableProcessors::ProcessorMock.to_string(),
            AvailableProcessors::ProcessorEcho.to_string(),
            AvailableProcessors::CandleDataProcessor.to_string(),
            AvailableProcessors::VectorDistance.to_string(),
            AvailableProcessors::ApplyTemplate.to_string(),
            AvailableProcessors::SortColumnAndIndices.to_string(),
            AvailableProcessors::HumanInTheLoop.to_string(),
            AvailableProcessors::ChunkDocuments.to_string(),
            AvailableProcessors::JoinInner.to_string(),
            AvailableProcessors::ExtractPDFText.to_string(),
            AvailableProcessors::GroupByAndAggregate.to_string(),
            AvailableProcessors::FilterColumnsAndIndices.to_string(),
            AvailableProcessors::ExtractTabularData.to_string(),
            AvailableProcessors::SelectAndCast.to_string(),
            AvailableProcessors::Pivot.to_string(),
            AvailableProcessors::NormalizeTime.to_string(),
            AvailableProcessors::DataSummaryProcessor.to_string(),
            AvailableProcessors::AttachmentAggregatorProcessor.to_string(),
            AvailableProcessors::CandleChatProcessor.to_string(),
            AvailableProcessors::MessageAggregatorProcessor.to_string(),
            AvailableProcessors::MessageParserProcessor.to_string(),
            AvailableProcessors::CandleEmbedProcessor.to_string(),
            #[cfg(feature = "openai_api")]
            AvailableProcessors::OpenAIChatProcessor.to_string(),
            #[cfg(feature = "openai_api")]
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
        } else if line.contains(&AvailableProcessors::ProcessorEcho.to_string()) {
            Ok(AvailableProcessors::ProcessorEcho)
        } else if line.contains(&AvailableProcessors::CandleDataProcessor.to_string()) {
            Ok(AvailableProcessors::CandleDataProcessor)
        } else if line.contains(&AvailableProcessors::Pivot.to_string()) {
            Ok(AvailableProcessors::Pivot)
        } else if line.contains(&AvailableProcessors::ApplyTemplate.to_string()) {
            Ok(AvailableProcessors::ApplyTemplate)
        } else if line.contains(&AvailableProcessors::VectorDistance.to_string()) {
            Ok(AvailableProcessors::VectorDistance)
        } else if line.contains(&AvailableProcessors::SortColumnAndIndices.to_string()) {
            Ok(AvailableProcessors::SortColumnAndIndices)
        } else if line.contains(&AvailableProcessors::HumanInTheLoop.to_string()) {
            Ok(AvailableProcessors::HumanInTheLoop)
        } else if line.contains(&AvailableProcessors::ChunkDocuments.to_string()) {
            Ok(AvailableProcessors::ChunkDocuments)
        } else if line.contains(&AvailableProcessors::JoinInner.to_string()) {
            Ok(AvailableProcessors::JoinInner)
        } else if line.contains(&AvailableProcessors::ExtractPDFText.to_string()) {
            Ok(AvailableProcessors::ExtractPDFText)
        } else if line.contains(&AvailableProcessors::GroupByAndAggregate.to_string()) {
            Ok(AvailableProcessors::GroupByAndAggregate)
        } else if line.contains(&AvailableProcessors::FilterColumnsAndIndices.to_string()) {
            Ok(AvailableProcessors::FilterColumnsAndIndices)
        } else if line.contains(&AvailableProcessors::ExtractTabularData.to_string()) {
            Ok(AvailableProcessors::ExtractTabularData)
        } else if line.contains(&AvailableProcessors::SelectAndCast.to_string()) {
            Ok(AvailableProcessors::SelectAndCast)
        } else if line.contains(&AvailableProcessors::NormalizeTime.to_string()) {
            Ok(AvailableProcessors::NormalizeTime)
        } else if line.contains(&AvailableProcessors::CandleDataProcessor.to_string()) {
            Ok(AvailableProcessors::CandleDataProcessor)
        } else if line.contains(&AvailableProcessors::DataSummaryProcessor.to_string()) {
            Ok(AvailableProcessors::DataSummaryProcessor)
        } else if line.contains(&AvailableProcessors::AttachmentAggregatorProcessor.to_string()) {
            Ok(AvailableProcessors::AttachmentAggregatorProcessor)
        } else if line.contains(&AvailableProcessors::CandleChatProcessor.to_string()) {
            Ok(AvailableProcessors::CandleChatProcessor)
        } else if line.contains(&AvailableProcessors::MessageAggregatorProcessor.to_string()) {
            Ok(AvailableProcessors::MessageAggregatorProcessor)
        } else if line.contains(&AvailableProcessors::MessageParserProcessor.to_string()) {
            Ok(AvailableProcessors::MessageParserProcessor)
        } else if line.contains(&AvailableProcessors::CandleEmbedProcessor.to_string()) {
            Ok(AvailableProcessors::CandleEmbedProcessor)
        } else {
            #[cfg(feature = "openai_api")]
            if line.contains(&AvailableProcessors::OpenAIChatProcessor.to_string()) {
                Ok(AvailableProcessors::OpenAIChatProcessor)
            } else if line.contains(&AvailableProcessors::OpenAIEmbedProcessor.to_string()) {
                Ok(AvailableProcessors::OpenAIEmbedProcessor)
            } else {
                Err(anyhow!(
                    "Processor not found in {line}. Available processors are {:?}.",
                    AvailableProcessors::all_varient_names()
                ))
            }
            #[cfg(not(feature = "openai_api"))]
            Err(anyhow!(
                "Processor not found in {line}. Available processors are {:?}.",
                AvailableProcessors::all_varient_names()
            ))
        }
    }

    /// Build the [ProcessorTrait] object
    pub fn build_arc(
        self,
        name: &str,
        publications: &[TablePublication],
        subscriptions: &[TableSubscription],
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    ) -> Arc<dyn ProcessorTrait> {
        match self {
            Self::ProcessorMock => Arc::new(ProcessorMock::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            Self::ProcessorEcho => Arc::new(ProcessorEcho::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            Self::CandleDataProcessor
            | Self::ChunkDocuments
            | Self::ExtractPDFText
            | Self::ExtractTabularData
            | Self::FilterColumnsAndIndices
            | Self::GroupByAndAggregate
            | Self::HumanInTheLoop
            | Self::JoinInner
            | Self::NormalizeTime
            | Self::Pivot
            | Self::SelectAndCast
            | Self::SortColumnAndIndices
            | Self::VectorDistance
            | Self::ApplyTemplate => Arc::new(CandleDataProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            Self::DataSummaryProcessor => Arc::new(DataSummaryProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            Self::AttachmentAggregatorProcessor => Arc::new(AttachmentAggregatorProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            Self::CandleChatProcessor => Arc::new(CandleChatProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            Self::MessageAggregatorProcessor => Arc::new(MessageAggregatorProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            Self::MessageParserProcessor => Arc::new(MessageParserProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            Self::CandleEmbedProcessor => Arc::new(CandleEmbedProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => Arc::new(OpenAIChatProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => Arc::new(OpenAIEmbedProcessor::new(
                name,
                self.to_string().as_str(),
                publications,
                subscriptions,
                subscribe_policy,
            )),
        }
    }

    /// Build the [ProcessorTrait] object form the [ProcessorBuilder]
    pub fn build_with_builder(self, builder: ProcessorBuilder) -> Result<Arc<dyn ProcessorTrait>> {
        match self {
            Self::ProcessorMock => builder.build_arc::<ProcessorMock>(),
            Self::ProcessorEcho => builder.build_arc::<ProcessorMock>(),
            Self::CandleDataProcessor
            | Self::ChunkDocuments
            | Self::ExtractPDFText
            | Self::ExtractTabularData
            | Self::FilterColumnsAndIndices
            | Self::GroupByAndAggregate
            | Self::HumanInTheLoop
            | Self::JoinInner
            | Self::NormalizeTime
            | Self::Pivot
            | Self::SelectAndCast
            | Self::SortColumnAndIndices
            | Self::VectorDistance
            | Self::ApplyTemplate => builder.build_arc::<CandleDataProcessor>(),
            Self::DataSummaryProcessor => builder.build_arc::<DataSummaryProcessor>(),
            Self::AttachmentAggregatorProcessor => {
                builder.build_arc::<AttachmentAggregatorProcessor>()
            }
            Self::CandleChatProcessor => builder.build_arc::<CandleChatProcessor>(),
            Self::MessageAggregatorProcessor => builder.build_arc::<MessageAggregatorProcessor>(),
            Self::MessageParserProcessor => builder.build_arc::<MessageParserProcessor>(),
            Self::CandleEmbedProcessor => builder.build_arc::<CandleEmbedProcessor>(),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => builder.build_arc::<OpenAIChatProcessor>(),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => builder.build_arc::<OpenAIEmbedProcessor>(),
        }
    }

    /// Identify the [DataConfigTrait] object for the [ProcessorTrait] object
    pub fn config_type(&self) -> &str {
        match self {
            Self::ProcessorEcho => "",
            Self::ProcessorMock
            | Self::CandleDataProcessor
            | Self::ChunkDocuments
            | Self::ExtractPDFText
            | Self::ExtractTabularData
            | Self::FilterColumnsAndIndices
            | Self::GroupByAndAggregate
            | Self::HumanInTheLoop
            | Self::JoinInner
            | Self::NormalizeTime
            | Self::Pivot
            | Self::SelectAndCast
            | Self::SortColumnAndIndices
            | Self::VectorDistance
            | Self::ApplyTemplate
            | Self::AttachmentAggregatorProcessor
            | Self::MessageAggregatorProcessor => "DataConfig",
            Self::DataSummaryProcessor => "DataSummaryConfig",
            Self::CandleChatProcessor | Self::MessageParserProcessor => "CandleChatConfig",
            Self::CandleEmbedProcessor => "CandleEmbedConfig",
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => "CandleChatConfig",
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => "CandleEmbedConfig",
        }
    }
}
