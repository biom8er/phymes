#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;
use phymes_core::{metrics::ArrowTaskMetricsSet, table::arrow_table::ArrowTableTrait};

use phymes_agents::{
    candle_chat::chat_config::CandleChatConfig,
    session_plans::chat_agent_session::test_chat_agent_session::bench_chat_processor,
};

pub async fn run_main() -> Result<()> {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    // Metrics to compute time and rows
    let metrics = ArrowTaskMetricsSet::new();

    // Chat processor config
    let config = CandleChatConfig::parse();

    // Run the chat processor
    let message_history = bench_chat_processor(
        metrics.clone(),
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

    println!(
        "number of rows {}",
        metrics.clone_inner().output_rows().unwrap()
    );
    println!(
        "elasped compute {}",
        metrics.clone_inner().elapsed_compute().unwrap()
    );

    Ok(())
}
