use std::fmt::Debug;

use candle_core::Device;

/// For services that process Tensors
pub trait TensorStreamTrait: Send + Sync + Debug {
    /// Device
    fn get_device(&self) -> &Device;
}

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

impl TensorStreamTrait for CandleTensorService {
    fn get_device(&self) -> &Device {
        &self.device
    }
}