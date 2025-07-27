use candle_core::Device;
/// From https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs
use candle_core::utils::{cuda_is_available, metal_is_available};

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
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        candle_core::Result::Ok(Device::Cpu)
    }
}
