use std::fmt::Debug;

use candle_core::{
    Device,
    utils::{cuda_is_available, metal_is_available},
};

/// From <https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs>
pub fn device(cpu: bool) -> candle_core::Result<Device> {
    if cpu {
        candle_core::Result::Ok(Device::Cpu)
    } else if cuda_is_available() {
        candle_core::Result::Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        candle_core::Result::Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // println!("Running on CPU, to run on GPU(metal), build this example with `--features metal`");
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            // println!("Running on CPU, to run on GPU, build this example with `--features gpu`");
        }
        candle_core::Result::Ok(Device::Cpu)
    }
}

/// For services that process Tensors
pub trait TensorProcessorTrait: Send + Sync + Debug {
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

impl TensorProcessorTrait for CandleTensorService {
    fn get_device(&self) -> &Device {
        &self.device
    }
}
