use std::{fmt::Display, sync::Arc};

use anyhow::Result;
use clap::ValueEnum;
use phymes_core::{
    test_processor::ProcessorMock, MappableTrait, ProcessorBuilder, ProcessorEcho, ProcessorTrait, SubscribeTrait, TablePublish, TableSubscribe
};
use phymes_data::{AttachmentAggregatorProcessor, CandleDataProcessor, DataSummaryProcessor};
use phymes_ml::{
    CandleChatProcessor, CandleEmbedProcessor, MessageAggregatorProcessor, MessageParserProcessor
};
#[cfg(feature = "openai_api")]
use phymes_ml::{OpenAIChatProcessor, OpenAIEmbedProcessor};
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

impl AvailableProcessors {
    /// Get all available processor plans
    pub fn get_all_processor_names() -> Vec<String> {
        let processor_names = [
            AvailableProcessors::ProcessorMock.to_string(),
            AvailableProcessors::ProcessorEcho.to_string(),
            AvailableProcessors::CandleDataProcessor.to_string(),
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

    /// Get the processor's corresponding config
    pub fn get_config(&self) -> AvailableProcessorConfigs {
        match self {
            Self::ProcessorMock => AvailableProcessorConfigs::None,
            Self::ProcessorEcho => AvailableProcessorConfigs::None,
            Self::CandleDataProcessor => AvailableProcessorConfigs::DataConfig,
            Self::DataSummaryProcessor => AvailableProcessorConfigs::DataSummaryConfig,
            Self::AttachmentAggregatorProcessor => AvailableProcessorConfigs::DataConfig,
            Self::CandleChatProcessor => AvailableProcessorConfigs::CandleChatConfig,
            Self::MessageAggregatorProcessor => AvailableProcessorConfigs::DataConfig,
            Self::MessageParserProcessor => AvailableProcessorConfigs::CandleChatConfig,
            Self::CandleEmbedProcessor => AvailableProcessorConfigs::CandleEmbedConfig,
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => AvailableProcessorConfigs::CandleChatConfig,
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => AvailableProcessorConfigs::CandleEmbedConfig,
        }
    }

    pub fn build_arc_with_pub_sub(
        self,
        name: &str,
        publications: &[TablePublish],
        subscriptions: &[TableSubscribe],
        subscribe: Box<dyn SubscribeTrait>,
    ) -> Arc<dyn ProcessorTrait> {
        match self {
            Self::ProcessorMock => {
                ProcessorMock::new_arc_with_pub_sub(name, publications, subscriptions, subscribe)
            }
            Self::ProcessorEcho => {
                ProcessorEcho::new_arc_with_pub_sub(name, publications, subscriptions, subscribe)
            }
            Self::CandleDataProcessor => CandleDataProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
            Self::DataSummaryProcessor => DataSummaryProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
            Self::AttachmentAggregatorProcessor => {
                AttachmentAggregatorProcessor::new_arc_with_pub_sub(
                    name,
                    publications,
                    subscriptions,
                    subscribe,
                )
            }
            Self::CandleChatProcessor => CandleChatProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
            Self::MessageAggregatorProcessor => MessageAggregatorProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
            Self::MessageParserProcessor => MessageParserProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
            Self::CandleEmbedProcessor => CandleEmbedProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => OpenAIChatProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => OpenAIEmbedProcessor::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
        }
    }

    pub fn build_with_builder(self, builder: ProcessorBuilder) -> Result<Arc<dyn ProcessorTrait>> {
        let (name, publications, subscriptions, subscribe) = builder.take()?;
        Ok(self.build_arc_with_pub_sub(&name, &publications, &subscriptions, subscribe))
    }
}



/// The available [ProcessorTrait] configs
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableProcessorConfigs {
    #[value(name = "None")]
    #[default]
    None,
    #[value(name = "CandleEmbedConfig")]
    CandleEmbedConfig,
    #[value(name = "CandleChatConfig")]
    CandleChatConfig,
    #[value(name = "DataSummaryConfig")]
    DataSummaryConfig,
    #[value(name = "DataConfig")]
    DataConfig,
}

impl Display for AvailableProcessorConfigs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::CandleEmbedConfig => write!(f, "CandleEmbedConfig"),
            Self::CandleChatConfig => write!(f, "CandleChatConfig"),
            Self::DataSummaryConfig => write!(f, "DataSummaryConfig"),
            Self::DataConfig => write!(f, "DataConfig"),
        }
    }
}

impl AvailableProcessorConfigs {
    /// convert JSON to a [Table]
    /// 
    /// let config = DataConfig { ..Default::default()};
    /// let config_json = serde_json::to_vec(&config)?;
    pub fn to_default_json(&self) -> Result<Vec<u8>> {        
        match self {
            Self::None => Ok(Vec::new()),
            Self::CandleEmbedConfig => Ok(Vec::new()),
            Self::CandleChatConfig => Ok(Vec::new()),
            Self::DataSummaryConfig => Ok(Vec::new()),
            Self::DataConfig => Ok(Vec::new()),
        }
    }
}