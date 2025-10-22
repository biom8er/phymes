mod data_config;
mod data_processor;
mod summary_config;
mod summary_processor;
mod tensor_service;
mod attachment_aggregator_processor;

pub use data_config::{DataStreamManager, DataAggregatorOperator, DataComparatorOperator, DataComparatorPredicate, DataDistanceOperator, DataCastOperator, DataConfig};
pub use data_processor::CandleDataProcessor;
#[allow(unused_imports)]
pub(crate) use data_processor::test_candle_ops_processor;
pub use summary_config::DataSummaryConfig;
pub use summary_processor::{DataSummaryProcessor, table_and_data_format_to_record_batch};
pub use tensor_service::CandleTensorService;
pub use attachment_aggregator_processor::{AttachmentAggregatorProcessor, AggregatorStream};