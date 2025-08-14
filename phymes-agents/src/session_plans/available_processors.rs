use std::sync::Arc;

use clap::ValueEnum;
use phymes_core::{session::common_traits::MappableTrait, table::{arrow_table_publish::ArrowTablePublish, arrow_table_subscribe::{ArrowTableSubscribe, SubscribeTrait}}, task::arrow_processor::{ArrowProcessorEcho, ArrowProcessorTrait}};
use phymes_data::candle_data::{data_processor::CandleDataProcessor, summary_processor::DataSummaryProcessor};
use phymes_ml::{candle_chat::{chat_processor::CandleChatProcessor, message_aggregator_processor::MessageAggregatorProcessor, message_parser_processor::MessageParserProcessor}, candle_embed::embed_processor::CandleEmbedProcessor};
#[cfg(feature = "openai_api")]
use phymes_ml::{openai_chat::chat_processor::OpenAIChatProcessor, openai_embed::embed_processor::OpenAIEmbedProcessor};
use serde::{Deserialize, Serialize};

/// The available session plans
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum AvailableProcessors {
    #[value(name = "ArrowProcessorEcho")]
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

impl Default for AvailableProcessors {
    fn default() -> Self {
        AvailableProcessors::ArrowProcessorEcho
    }
}

impl MappableTrait for AvailableProcessors {
    fn get_name(&self) -> &str {
        match self {
            Self::ArrowProcessorEcho => "ArrowProcessorEcho",
            Self::CandleDataProcessor => "CandleDataProcessor",
            Self::DataSummaryProcessor => "DataSummaryProcessor",
            Self::CandleChatProcessor => "CandleChatProcessor",
            Self::MessageAggregatorProcessor => "MessageAggregatorProcessor",
            Self::MessageParserProcessor => "MessageParserProcessor",
            Self::CandleEmbedProcessor => "CandleEmbedProcessor",
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => "OpenAIChatProcessor",
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => "OpenAIEmbedProcessor",
        }
    }
}

impl AvailableProcessors {
    pub fn build(self,
        name: &str,
        publications: &[ArrowTablePublish],
        subscriptions: &[ArrowTableSubscribe],
        subscribe: Box<dyn SubscribeTrait>
    ) -> Arc<dyn ArrowProcessorTrait> {
        match self {
            Self::ArrowProcessorEcho => ArrowProcessorEcho::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            Self::CandleDataProcessor => CandleDataProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            Self::DataSummaryProcessor => DataSummaryProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            Self::CandleChatProcessor => CandleChatProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            Self::MessageAggregatorProcessor => MessageAggregatorProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            Self::MessageParserProcessor => MessageParserProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            Self::CandleEmbedProcessor => CandleEmbedProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            #[cfg(feature = "openai_api")]
            Self::OpenAIChatProcessor => OpenAIChatProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            #[cfg(feature = "openai_api")]
            Self::OpenAIEmbedProcessor => OpenAIEmbedProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
        }
    }
}