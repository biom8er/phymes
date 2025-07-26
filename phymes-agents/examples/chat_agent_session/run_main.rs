#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use std::sync::Arc;

use phymes_agents::session_plans::{
    agent_session_builder::AgentSessionBuilderTrait,
    chat_agent_session::{
        ChatAgentSession,
        test_chat_agent_session::{bench_chat_agent_session_1, bench_chat_agent_session_2},
    },
};
use phymes_core::{
    metrics::{ArrowTaskMetricsSet, HashMap},
    session::session_context::SessionStreamState,
    table::arrow_table::ArrowTableTrait,
    task::arrow_message::{ArrowIncomingMessage, ArrowIncomingMessageTrait},
};

pub async fn run_main() -> Result<()> {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    // initialize the metrics
    let metrics = ArrowTaskMetricsSet::new();

    // initialize the session
    let chat_agent_session = ChatAgentSession {
        session_context_name: "session_1",
        chat_processor_name: "chat_processor_1",
        chat_task_name: "chat_task_1",
        runtime_env_name: "rt_1",
        chat_subscription_name: "messages",
        chat_api_url: Some("http://0.0.0.0:8000/v1"),
    };
    let session_ctx = chat_agent_session.make_session_context(metrics.clone())?;
    let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_ctx)));

    // ----- Query #1 -----
    let session_stream = bench_chat_agent_session_1(
        Arc::clone(&session_stream_state),
        &chat_agent_session,
        "Write a function to count prime numbers up to N.",
    );
    let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
        session_stream.try_collect().await?;

    // Update the chat history with the response
    let json_data = response
        .last_mut()
        .unwrap()
        .remove(&format!(
            "from_{}_on_{}",
            chat_agent_session.session_context_name, chat_agent_session.chat_subscription_name
        ))
        .unwrap()
        .get_message_own()
        .to_json_object()?;
    for row in &json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    // ----- Query #2 -----
    let session_stream = bench_chat_agent_session_2(
        Arc::clone(&session_stream_state),
        &chat_agent_session,
        "Please provide an example using the functions.",
    );
    let mut response: Vec<HashMap<String, ArrowIncomingMessage>> =
        session_stream.try_collect().await?;

    // Update the chat history with the response
    let json_data = response
        .first_mut()
        .unwrap()
        .remove(&format!(
            "from_{}_on_{}",
            chat_agent_session.session_context_name, chat_agent_session.chat_subscription_name
        ))
        .unwrap()
        .get_message_own()
        .to_json_object()?;
    for row in &json_data {
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
