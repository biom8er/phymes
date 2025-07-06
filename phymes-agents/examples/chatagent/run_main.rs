#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use std::sync::Arc;

use phymes_agents::{
    candle_chat::message_history::MessageHistoryBuilderTraitExt,
    session_plans::{
        agent_session_builder::AgentSessionBuilderTrait, chat_agent_session::ChatAgentSession,
    },
};
use phymes_core::{
    metrics::{ArrowTaskMetricsSet, HashMap},
    session::{
        common_traits::{BuilderTrait, MappableTrait},
        session_context::{SessionStream, SessionStreamState},
    },
    table::{
        arrow_table::{ArrowTableBuilder, ArrowTableTrait},
        arrow_table_publish::ArrowTablePublish,
    },
    task::arrow_message::{
        ArrowIncomingMessage, ArrowIncomingMessageBuilder, ArrowIncomingMessageBuilderTrait,
        ArrowIncomingMessageTrait, ArrowMessageBuilderTrait,
    },
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

    // Make the system prompt and add the user query
    let message_builder = ArrowTableBuilder::new()
        .with_name(chat_agent_session.chat_subscription_name)
        .insert_system_template_str("You are a helpful assistant.")?
        .append_new_user_query_str("What is the tallest mountain in the world?", "user")?;

    // Build the current message state
    let incoming_message = ArrowIncomingMessageBuilder::new()
        .with_name(chat_agent_session.chat_subscription_name)
        .with_subject(chat_agent_session.chat_subscription_name)
        .with_publisher(chat_agent_session.session_context_name)
        .with_message(message_builder.build()?)
        .with_update(&ArrowTablePublish::Extend {
            table_name: chat_agent_session.chat_subscription_name.to_string(),
        })
        .build()?;
    let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
    incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

    // Run the session
    let session_stream =
        SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));

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
            println!("{}: {}", row["role"], row["content"])
        }
    }

    // ----- Query #2 -----

    // Add a new query to the message history
    let message_builder = ArrowTableBuilder::new()
        .with_name(chat_agent_session.chat_subscription_name)
        .append_new_user_query_str("What is the next tallest mountain in the world?", "user")?;

    // Build the incoming message state
    let incoming_message = ArrowIncomingMessageBuilder::new()
        .with_name(chat_agent_session.chat_subscription_name)
        .with_subject(chat_agent_session.chat_task_name)
        .with_publisher(chat_agent_session.session_context_name)
        .with_message(message_builder.clone().build()?)
        .with_update(&ArrowTablePublish::Extend {
            table_name: chat_agent_session.chat_subscription_name.to_string(),
        })
        .build()?;
    let mut incoming_message_map = HashMap::<String, ArrowIncomingMessage>::new();
    incoming_message_map.insert(incoming_message.get_name().to_string(), incoming_message);

    // Run the session
    session_stream_state.try_write().unwrap().set_iter(0);
    let session_stream =
        SessionStream::new(incoming_message_map, Arc::clone(&session_stream_state));

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
            println!("{}: {}", row["role"], row["content"])
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
