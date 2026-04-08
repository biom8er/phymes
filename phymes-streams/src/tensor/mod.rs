mod data_stream;
mod tensor_service;

pub use data_stream::{test_candle_ops, CandleDataStream};
pub use tensor_service::{CandleTensorService, TensorStreamTrait};