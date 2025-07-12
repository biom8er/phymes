#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;
use phymes_core::{
    metrics::{ArrowTaskMetricsSet, HashMap},
    session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait},
        runtime_env::{RuntimeEnv, RuntimeEnvTrait},
    },
    table::{
        arrow_table::{ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait},
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::ArrowTableSubscribe,
    },
    task::arrow_message::{
        ArrowMessageBuilderTrait, ArrowOutgoingMessage, ArrowOutgoingMessageBuilderTrait,
        ArrowOutgoingMessageTrait,
    },
};

use phymes_agents::candle_chat::{
    chat_config::CandleChatConfig, chat_processor::CandleChatProcessor,
    message_history::MessageHistoryBuilderTraitExt,
};

#[allow(unused_imports)]
#[cfg(feature = "openai_api")]
use phymes_agents::openai_asset::chat_processor::OpenAIChatProcessor;

use parking_lot::Mutex;
use std::sync::Arc;

pub async fn run_main() -> anyhow::Result<()> {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let name = "CandleChatProcessor";
    let messages = "messages";

    // Metrics to compute time and rows
    let metrics = ArrowTaskMetricsSet::new();

    // State for the chat processor config
    let candle_chat_config = CandleChatConfig::parse();
    let candle_chat_config_json = serde_json::to_vec(&candle_chat_config)?;
    let candle_chat_config_table = ArrowTableBuilder::new()
        .with_name(name)
        .with_json(&candle_chat_config_json, 1)?
        .build()?;

    // Make the system prompt and add the user query
    let message_builder = ArrowTableBuilder::new()
        .with_name(messages)
        .insert_system_template_str("You are a helpful assistant.")?
        .append_new_user_query_str("What are the four molecules that compose DNA?", "user")?;

    // Build the current message state
    let mut message = HashMap::<String, ArrowOutgoingMessage>::new();
    let _ = message.insert(
        messages.to_string(),
        ArrowOutgoingMessage::get_builder()
            .with_name(messages)
            .with_publisher("")
            .with_subject(messages)
            .with_update(&ArrowTablePublish::None)
            .with_message(message_builder.clone().build()?.to_record_batch_stream())
            .build()?,
    );
    let _ = message.insert(
        candle_chat_config_table.get_name().to_string(),
        ArrowOutgoingMessage::get_builder()
            .with_name(candle_chat_config_table.get_name())
            .with_publisher("")
            .with_subject(candle_chat_config_table.get_name())
            .with_update(&ArrowTablePublish::None)
            .with_message(candle_chat_config_table.to_record_batch_stream())
            .build()?,
    );

    // Build the chat task
    #[allow(unused_variables)]
    let chat_processor = CandleChatProcessor::new_with_pub_sub_for(
        name,
        &[ArrowTablePublish::ExtendChunks {
            table_name: messages.to_string(),
            col_name: "content".to_string(),
        }],
        &[
            ArrowTableSubscribe::OnUpdateFullTable {
                table_name: messages.to_string(),
            },
            ArrowTableSubscribe::None,
            ArrowTableSubscribe::AlwaysFullTable {
                table_name: candle_chat_config_table.get_name().to_string(),
            },
        ],
        &[],
    );
    #[cfg(all(not(feature = "candle"), feature = "openai_api"))]
    let chat_processor = OpenAIChatProcessor::new_with_pub_sub_for(
        name,
        &[ArrowTablePublish::ExtendChunks {
            table_name: messages.to_string(),
            col_name: "content".to_string(),
        }],
        &[
            ArrowTableSubscribe::OnUpdateFullTable {
                table_name: messages.to_string(),
            },
            ArrowTableSubscribe::None,
            ArrowTableSubscribe::AlwaysFullTable {
                table_name: candle_chat_config_table.get_name().to_string(),
            },
        ],
        &[],
    );
    let mut stream = chat_processor.process(
        message,
        metrics.clone(),
        Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt"))),
    )?;

    // Update the chat history with the response
    let (message_builder, _stream) = message_builder
        .append_chat_response_sendable_record_batch_stream(
            &mut stream.remove(messages).unwrap().get_message_own(),
            1000,
        )
        .await?;
    let message_history = message_builder.clone().build()?;
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
