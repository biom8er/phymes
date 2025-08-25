use std::sync::Arc;

use anyhow::Result;
use clap::ValueEnum;
use phymes_core::{
    session::common_traits::MappableTrait,
    table::{
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::{ArrowTableSubscribe, SubscribeTrait},
    },
    task::arrow_processor::{
        ArrowProcessorBuilder, ArrowProcessorEcho, ArrowProcessorTrait,
        test_processor::ArrowProcessorMock,
    },
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

impl MappableTrait for AvailableProcessors {
    fn get_name(&self) -> &str {
        match self {
            Self::ArrowProcessorMock => ArrowProcessorMock::get_static_name(),
            Self::ArrowProcessorEcho => ArrowProcessorEcho::get_static_name(),
            Self::CandleDataProcessor => CandleDataProcessor::get_static_name(),
            Self::DataSummaryProcessor => DataSummaryProcessor::get_static_name(),
            Self::CandleChatProcessor => CandleChatProcessor::get_static_name(),
            Self::MessageAggregatorProcessor => MessageAggregatorProcessor::get_static_name(),
            Self::MessageParserProcessor => MessageParserProcessor::get_static_name(),
            Self::CandleEmbedProcessor => CandleEmbedProcessor::get_static_name(),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => OpenAIChatProcessor::get_static_name(),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => OpenAIEmbedProcessor::get_static_name(),
        }
    }
}

impl AvailableProcessors {
    pub fn new_from_name(name: &str) -> Option<Self> {
        if name == ArrowProcessorMock::get_static_name() {
            Some(Self::ArrowProcessorMock)
        } else if name == ArrowProcessorEcho::get_static_name() {
            Some(Self::ArrowProcessorEcho)
        } else if name == CandleDataProcessor::get_static_name() {
            Some(Self::CandleChatProcessor)
        } else if name == DataSummaryProcessor::get_static_name() {
            Some(Self::DataSummaryProcessor)
        } else if name == CandleChatProcessor::get_static_name() {
            Some(Self::CandleChatProcessor)
        } else if name == MessageAggregatorProcessor::get_static_name() {
            Some(Self::MessageAggregatorProcessor)
        } else if name == MessageParserProcessor::get_static_name() {
            Some(Self::MessageParserProcessor)
        } else if name == CandleEmbedProcessor::get_static_name() {
            Some(Self::CandleEmbedProcessor)
        } else {
            #[cfg(feature = "openai_api")]
            if name == OpenAIChatProcessor::get_static_name() {
                Some(Self::OpenAIChatProcessor)
            } else if name == OpenAIEmbedProcessor::get_static_name() {
                Some(Self::OpenAIEmbedProcessor)
            } else {
                None
            }
            #[cfg(not(feature = "openai_api"))]
            None
        }
    }
    /// Get all available processor plans
    pub fn get_all_processor_names() -> Vec<String> {
        let processor_names = [
            AvailableProcessors::ArrowProcessorMock.get_name(),
            AvailableProcessors::ArrowProcessorEcho.get_name(),
            AvailableProcessors::CandleDataProcessor.get_name(),
            AvailableProcessors::DataSummaryProcessor.get_name(),
            AvailableProcessors::CandleChatProcessor.get_name(),
            AvailableProcessors::MessageAggregatorProcessor.get_name(),
            AvailableProcessors::MessageParserProcessor.get_name(),
            AvailableProcessors::CandleEmbedProcessor.get_name(),
            #[cfg(feature = "openai_api")]
            AvailableProcessors::OpenAIChatProcessor.get_name(),
            #[cfg(feature = "openai_api")]
            AvailableProcessors::OpenAIEmbedProcessor.get_name(),
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
