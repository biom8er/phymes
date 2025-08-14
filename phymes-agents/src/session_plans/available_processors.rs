use phymes_core::task::arrow_processor::ArrowProcessorTrait;

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
    #[value(name = "OpenAIChatProcessor")]
    OpenAIChatProcessor,
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
            Self::CandleEmbedProcessor => "CandleEmbedProcessor",
            Self::OpenAIChatProcessor => "OpenAIChatProcessor",
            Self::OpenAIEmbedProcessor => "OpenAIEmbedProcessor",
        }
    }
}

impl AvailableProcessors {
    pub fn build(
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
            Self::CandleEmbedProcessor => CandleEmbedProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            Self::OpenAIChatProcessor => OpenAIChatProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
            Self::OpenAIEmbedProcessor => OpenAIEmbedProcessor::new_arc_with_pub_sub(name, publications, subscriptions, subscribe),
        }
    }
}