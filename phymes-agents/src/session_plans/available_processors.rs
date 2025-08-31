use std::{fmt::Display, sync::Arc};

use anyhow::Result;
use clap::ValueEnum;
use phymes_core::{
    session::common_traits::MappableTrait, table::{
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::{ArrowTableSubscribe, SubscribeTrait},
    }, task::arrow_processor::{
        test_processor::ArrowProcessorMock, ArrowProcessorBuilder, ArrowProcessorEcho, ArrowProcessorTrait
    }
};
use phymes_data::candle_data::{
    data_processor::CandleDataProcessor, summary_processor::DataSummaryProcessor,
};
use phymes_ml::{
    candle_chat::{
        chat_processor::CandleChatProcessor,
        message_aggregator_processor::MessageAggregatorProcessor,
        message_parser_processor::MessageParserProcessor,
    },
    candle_embed::embed_processor::CandleEmbedProcessor,
};
#[cfg(feature = "openai_api")]
use phymes_ml::{
    openai_chat::chat_processor::OpenAIChatProcessor,
    openai_embed::embed_processor::OpenAIEmbedProcessor,
};
use serde::{Deserialize, Serialize};

/// The available session plans
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableProcessors {
    #[value(name = "ArrowProcessorMock")]
    ArrowProcessorMock,
    #[value(name = "ArrowProcessorEcho")]
    #[default]
    ArrowProcessorEcho,
    #[value(name = "CandleDataProcessor")]
    CandleDataProcessor,
    #[value(name = "DataSummaryProcessor")]
    DataSummaryProcessor,
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
            Self::ArrowProcessorMock => write!(f, "{}", ArrowProcessorMock::get_static_name()),
            Self::ArrowProcessorEcho => write!(f, "{}", ArrowProcessorEcho::get_static_name()),
            Self::CandleDataProcessor => write!(f, "{}", CandleDataProcessor::get_static_name()),
            Self::DataSummaryProcessor => write!(f, "{}", DataSummaryProcessor::get_static_name()),
            Self::CandleChatProcessor => write!(f, "{}", CandleChatProcessor::get_static_name()),
            Self::MessageAggregatorProcessor => write!(f, "{}", MessageAggregatorProcessor::get_static_name()),
            Self::MessageParserProcessor => write!(f, "{}", MessageParserProcessor::get_static_name()),
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
            AvailableProcessors::ArrowProcessorMock.to_string(),
            AvailableProcessors::ArrowProcessorEcho.to_string(),
            AvailableProcessors::CandleDataProcessor.to_string(),
            AvailableProcessors::DataSummaryProcessor.to_string(),
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

    pub fn build_arc_with_pub_sub(
        self,
        name: &str,
        publications: &[ArrowTablePublish],
        subscriptions: &[ArrowTableSubscribe],
        subscribe: Box<dyn SubscribeTrait>,
    ) -> Arc<dyn ArrowProcessorTrait> {
        match self {
            Self::ArrowProcessorMock => ArrowProcessorMock::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
            Self::ArrowProcessorEcho => ArrowProcessorEcho::new_arc_with_pub_sub(
                name,
                publications,
                subscriptions,
                subscribe,
            ),
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

    pub fn build_with_builder(
        self,
        builder: ArrowProcessorBuilder,
    ) -> Result<Arc<dyn ArrowProcessorTrait>> {
        let (name, publications, subscriptions, subscribe) = builder.take()?;
        Ok(self.build_arc_with_pub_sub(&name, &publications, &subscriptions, subscribe))
    }
}
