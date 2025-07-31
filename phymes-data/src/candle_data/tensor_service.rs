use candle_core::Device;
use phymes_core::session::common_traits::TensorProcessorTrait;

/// The actual asset struct
#[derive(Debug)]
pub struct CandleTensorService {
    /// The device for computation
    pub device: Device,
}

impl CandleTensorService {
    pub fn new(device: Device) -> CandleTensorService {
        CandleTensorService { device }
    }
}

impl TensorProcessorTrait for CandleTensorService {
    fn get_device(&self) -> &Device {
        &self.device
    }
}
