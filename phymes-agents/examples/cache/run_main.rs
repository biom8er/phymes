use clap::Parser;
use serde::{Deserialize, Serialize};

use candle_core::DType;
use phymes_agents::candle_assets::{candle_which::WhichCandleAsset, device::device};

#[derive(Parser, Debug, Serialize, Deserialize, Default, Clone)]
#[command(author, version, about, long_about = None)]
pub struct WhichCandleAssetConfig {
    /// The model to use.
    #[arg(long, default_value = "Qwen-v2.5-0.5b-chat")]
    pub which: WhichCandleAsset,
}

pub async fn run_main() -> anyhow::Result<()> {
    // Build the candle asset
    let config = WhichCandleAssetConfig::parse();
    let device = device(true)?; // force CPU
    let _ = config
        .which
        .build(None, None, None, None, DType::F32, device)?;
    // all model assets will be placed at $HOME/.cache/huggingface/hub
    // e.g., /home/dmccloskey/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775a/tokenizer.json
    // e.g., /home/dmccloskey/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775a/tokenizer_config.json
    // e.g., /home/dmccloskey/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_0.gguf

    Ok(())
}
