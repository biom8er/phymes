#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;
use phymes_subject::SubjectTrait;
use phymes_ml::CandleChatConfig;
use phymes_processor::bench_chat_processor::bench_chat_processor;

pub async fn run_main() -> Result<()> {
    // DM, todo!(): move to phymes-ml
    // println!(
    //     "avx: {}, neon: {}, simd128: {}, f16c: {}",
    //     candle_core::utils::with_avx(),
    //     candle_core::utils::with_neon(),
    //     candle_core::utils::with_simd128(),
    //     candle_core::utils::with_f16c()
    // );

    // Chat processor config
    let config = CandleChatConfig::parse();

    // Run the chat processor
    let message_history = bench_chat_processor(
        None,
        &config,
        "What are the four molecules that compose DNA?",
        "chat_processor",
    )
    .await?;
    let json_data = message_history.to_json_object()?;
    for row in json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    Ok(())
}
