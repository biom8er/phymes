use candle_core::Device;
use phymes_core::session::common_traits::TensorProcessorTrait;

/// The actual asset struct
#[derive(Debug)]
pub struct CandleOpsService {
    /// The device for computation
    pub device: Device,
}

impl CandleOpsService {
    pub fn new(device: Device) -> CandleOpsService {
        CandleOpsService { device }
    }
}

impl TensorProcessorTrait for CandleOpsService {
    fn get_device(&self) -> &Device {
        &self.device
    }
}
