use std::{fmt::Display, sync::Arc};

use anyhow::{anyhow, Result};
use arrow::datatypes::DataType;
use clap::ValueEnum;
use phymes_core::{
    DataFormat, ProcessorBuilder, ProcessorEcho, ProcessorTrait, Table, TablePublication, TableSubscribePolicyTrait, TableSubscription, test_processor::ProcessorMock
};
use phymes_data::{
    AttachmentAggregatorProcessor, AvailableCandleOperators, CandleDataProcessor, DataAggregatorOperator, DataCastOperator, DataComparatorOperator, DataComparatorPredicate, DataConfig, DataConfigTrait, DataDistanceOperator, DataStreamManager, DataSummaryConfig, DataSummaryProcessor, ToolTrait
};
use phymes_ml::{
    AvailableCandleAssets, CandleChatConfig, CandleChatProcessor, CandleEmbedConfig, CandleEmbedProcessor, MessageAggregatorProcessor, MessageParserProcessor
};
#[cfg(feature = "openai_api")]
use phymes_ml::{OpenAIChatProcessor, OpenAIEmbedProcessor};
use serde::{Deserialize, Serialize};

/// The available [ProcessorTrait]s
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableProcessors {
    #[value(name = "Mock")]
    Mock,
    #[value(name = "Echo")]
    #[default]
    Echo,
    #[value(name = "Data")]
    Data,
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
    #[value(name = "Summary")]
    Summary,
    #[value(name = "AttachmentAggregator")]
    AttachmentAggregator,
    #[value(name = "CandleChat")]
    CandleChat,
    #[value(name = "MessageAggregator")]
    MessageAggregator,
    #[value(name = "MessageParser")]
    MessageParser,
    #[value(name = "CandleEmbed")]
    CandleEmbed,
    #[cfg(feature = "openai_api")]
    #[value(name = "OpenAIChat")]
    OpenAIChat,
    #[cfg(feature = "openai_api")]
    #[value(name = "OpenAIEmbed")]
    OpenAIEmbed,
}

impl Display for AvailableProcessors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mock => write!(f, "Mock"),
            Self::Echo => write!(f, "Echo"),
            Self::Data => write!(f, "Data"),
            Self::VectorDistance => write!(f, "VectorDistance"),
            Self::SortColumnAndIndices => write!(f, "SortColumnAndIndices"),
            Self::HumanInTheLoop => write!(f, "HumanInTheLoop"),
            Self::ChunkDocuments => write!(f, "ChunkDocuments"),
            Self::JoinInner => write!(f, "JoinInner"),
            Self::ExtractPDFText => write!(f, "ExtractPDFText"),
            Self::GroupByAndAggregate => write!(f, "GroupByAndAggregate"),
            Self::FilterColumnsAndIndices => write!(f, "FilterColumnsAndIndices"),
            Self::ExtractTabularData => write!(f, "ExtractTabularData"),
            Self::SelectAndCast => write!(f, "SelectAndCast"),
            Self::Pivot => write!(f, "Pivot"),
            Self::NormalizeTime => write!(f, "NormalizeTime"),
            Self::Summary => write!(f, "Summary"),
            Self::AttachmentAggregator => write!(f, "AttachmentAggregator"),
            Self::CandleChat => write!(f, "CandleChat"),
            Self::MessageAggregator => write!(f, "MessageAggregator"),
            Self::MessageParser => write!(f, "MessageParser"),
            Self::CandleEmbed => write!(f, "CandleEmbed"),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChat => write!(f, "OpenAIChat"),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbed => write!(f, "OpenAIEmbed"),
        }
    }
}

impl DataConfigTrait for AvailableProcessors {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error>
        where
            Self: Serialize {        
        match self {
            Self::Mock => serde_json::to_vec(&DataConfig::default()), // Just for testing purposes...
            Self::Echo => Ok(Vec::new()),
            Self::Data => serde_json::to_vec(&DataConfig::default()),
            Self::VectorDistance => serde_json::to_vec(&DataConfig {
                lhs_pk: Some("lhs_pk".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                rhs_pk: Some("rhs_pk".to_string()),
                rhs_values: Some(vec!["rhs_values".to_string()]),
                dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
                cpu: false,
                operator: AvailableCandleOperators::VectorDistance,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::SortColumnAndIndices => serde_json::to_vec(&DataConfig {
                lhs_values: Some(vec!["lhs_values".to_string()]),
                asc: Some(true),
                cpu: false,
                operator: AvailableCandleOperators::SortColumnAndIndices,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::HumanInTheLoop => serde_json::to_vec(&DataConfig {
                cpu: false,
                operator: AvailableCandleOperators::HumanInTheLoop,
                ..Default::default()
            }),
            Self::ChunkDocuments => serde_json::to_vec(&DataConfig {
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
                lhs_pk: Some("lhs_pk".to_string()),
                lhs_values: Some(vec!["lhs_values".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::ExtractPDFText,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::GroupByAndAggregate => serde_json::to_vec(&DataConfig {
                lhs_values: Some(vec!["lhs_values".to_string()]),
                agg_columns: Some(vec!["agg_columns".to_string()]),
                agg_operators: Some(vec![DataAggregatorOperator::Sum]),
                cpu: false,
                operator: AvailableCandleOperators::GroupByAndAggregate,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::FilterColumnsAndIndices => serde_json::to_vec(&DataConfig {
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
                lhs_values: Some(vec!["lhs_values".to_string()]),
                format: Some(DataFormat::None),
                cpu: false,
                operator: AvailableCandleOperators::ExtractTabularData,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::SelectAndCast => serde_json::to_vec(&DataConfig {
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
                lhs_values: Some(vec!["lhs_values".to_string()]),
                cpu: false,
                operator: AvailableCandleOperators::NormalizeTime,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::Summary => serde_json::to_vec(&DataSummaryConfig {
                num_rows: Some(10),
                num_batches: Some(1),
                summary_format: DataFormat::None,
                ..Default::default()
            }),
            Self::AttachmentAggregator => serde_json::to_vec(&DataConfig {
                lhs_values: Some(vec!["timestamp".to_string()]),
                asc: Some(true),
                cpu: false,
                operator: AvailableCandleOperators::SortColumnAndIndices,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::CandleChat => serde_json::to_vec(&CandleChatConfig {
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
            Self::MessageAggregator => serde_json::to_vec(&DataConfig {
                lhs_values: Some(vec!["timestamp".to_string()]),
                asc: Some(true),
                cpu: false,
                operator: AvailableCandleOperators::SortColumnAndIndices,
                stream: DataStreamManager::AccumulateLHSAccumulateRHS,
                ..Default::default()
            }),
            Self::MessageParser => serde_json::to_vec(&CandleChatConfig {
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
            Self::CandleEmbed => serde_json::to_vec(&CandleEmbedConfig {
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
            Self::OpenAIChat => serde_json::to_vec(&CandleChatConfig {
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
            Self::OpenAIEmbed => serde_json::to_vec(&CandleEmbedConfig {
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
            Self: Sized {
        unimplemented!()
    }
}

impl ToolTrait for AvailableProcessors {
    fn get_description(&self) -> String {
        todo!()
    }
    fn to_json_tool_schema(&self) -> String {
        todo!()
    }
}

impl AvailableProcessors {
    /// Get all available processor plans
    pub fn get_all_processor_names() -> Vec<String> {
        let processor_names = [
            AvailableProcessors::Mock.to_string(),
            AvailableProcessors::Echo.to_string(),
            AvailableProcessors::Data.to_string(),
            AvailableProcessors::VectorDistance.to_string(),
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
            AvailableProcessors::Summary.to_string(),
            AvailableProcessors::AttachmentAggregator.to_string(),
            AvailableProcessors::CandleChat.to_string(),
            AvailableProcessors::MessageAggregator.to_string(),
            AvailableProcessors::MessageParser.to_string(),
            AvailableProcessors::CandleEmbed.to_string(),
            #[cfg(feature = "openai_api")]
            AvailableProcessors::OpenAIChat.to_string(),
            #[cfg(feature = "openai_api")]
            AvailableProcessors::OpenAIEmbed.to_string(),
        ];
        processor_names
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    pub fn from_str_fuzzy(line: &str) -> Result<Self> {
        if line.contains(&AvailableProcessors::Mock.to_string()) {
            Ok(AvailableProcessors::Mock)
        } else if line.contains(&AvailableProcessors::Echo.to_string()) {
            Ok(AvailableProcessors::Echo)
        } else if line.contains(&AvailableProcessors::Data.to_string()) {
            Ok(AvailableProcessors::Data)
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
        } else if line.contains(&AvailableProcessors::Data.to_string()) {
            Ok(AvailableProcessors::Data)
        } else if line.contains(&AvailableProcessors::Summary.to_string()) {
            Ok(AvailableProcessors::Summary)
        } else if line.contains(&AvailableProcessors::AttachmentAggregator.to_string()) {
            Ok(AvailableProcessors::AttachmentAggregator)
        } else if line.contains(&AvailableProcessors::CandleChat.to_string()) {
            Ok(AvailableProcessors::CandleChat)
        } else if line.contains(&AvailableProcessors::MessageAggregator.to_string()) {
            Ok(AvailableProcessors::MessageAggregator)
        } else if line.contains(&AvailableProcessors::MessageParser.to_string()) {
            Ok(AvailableProcessors::MessageParser)
        } else if line.contains(&AvailableProcessors::CandleEmbed.to_string()) {
            Ok(AvailableProcessors::CandleEmbed)
        } else {
            #[cfg(feature = "openai_api")]
            if line.contains(&AvailableProcessors::OpenAIChat.to_string()) {
                Ok(AvailableProcessors::OpenAIChat)
            } else if line.contains(&AvailableProcessors::OpenAIEmbed) {
                Ok(AvailableProcessors::OpenAIEmbed)
            } else {
                Err(anyhow!(
                    "Processor not found in {line}. Available processors are {:?}.", AvailableProcessors::get_all_processor_names()
                ))
            }
            #[cfg(not(feature = "openai_api"))]
            Err(anyhow!(
                "Processor not found in {line}. Available processors are {:?}.", AvailableProcessors::get_all_processor_names()
            ))
        }
    }

    pub fn build_arc_with_pub_sub(
        self,
        name: &str,
        publications: &[TablePublication],
        subscriptions: &[TableSubscription],
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    ) -> Arc<dyn ProcessorTrait> {
        match self {
            Self::Mock => {
                ProcessorMock::new_arc_with_pub_sub(name, publications, subscriptions, subscribe_policy)
            }
            Self::Echo => {
                ProcessorEcho::new_arc_with_pub_sub(name, publications, subscriptions, subscribe_policy)
            }
            Self::Data 
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
            | Self::VectorDistance => CandleDataProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe_policy,
            ),
            Self::Summary => DataSummaryProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe_policy,
            ),
            Self::AttachmentAggregator => {
                AttachmentAggregatorProcessor::new_arc_with_pub_sub(
                    name,
                    publications,
                    subscriptions,
                    subscribe_policy,
                )
            }
            Self::CandleChat => CandleChatProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe_policy,
            ),
            Self::MessageAggregator => MessageAggregatorProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe_policy,
            ),
            Self::MessageParser => MessageParserProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe_policy,
            ),
            Self::CandleEmbed => CandleEmbedProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe_policy,
            ),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChat => OpenAIChatProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe_policy,
            ),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbed => OpenAIEmbedProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe_policy,
            ),
        }
    }

    pub fn build_with_builder(self, builder: ProcessorBuilder) -> Result<Arc<dyn ProcessorTrait>> {
        let (name, publications, subscriptions, subscribe_policy) = builder.take()?;
        Ok(self.build_arc_with_pub_sub(&name, &publications, &subscriptions, subscribe_policy))
    }
}
