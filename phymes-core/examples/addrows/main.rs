use anyhow::Result;
use futures::TryStreamExt;
use parking_lot::RwLock;
use phymes_core::metrics::ArrowTaskMetricsSet;
use phymes_core::metrics::HashMap;
use phymes_core::session::common_traits::MappableTrait;
use phymes_core::session::session_context::SessionStream;
use phymes_core::session::session_context::SessionStreamState;
use phymes_core::session::session_context_builder::test_session_context_builder::make_test_session_context_sequential_task;
use phymes_core::table::arrow_table_publish::ArrowTablePublish;
use phymes_core::task::arrow_message::ArrowIncomingMessage;
use phymes_core::task::arrow_task::test_task::make_test_input_message;
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    #[cfg(not(target_family = "wasm"))]
    use tracing_chrome::ChromeLayerBuilder;
    #[cfg(not(target_family = "wasm"))]
    use tracing_subscriber::prelude::*;
    #[cfg(not(target_family = "wasm"))]
    let _guard = {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    };

    let metrics = ArrowTaskMetricsSet::new();
    let session_context =
        make_test_session_context_sequential_task("session_1", metrics.clone(), 4)?;
    let input = make_test_input_message(
        "task_1",
        "session_1",
        "state_1",
        "state_1",
        &ArrowTablePublish::Replace {
            table_name: "state_1".to_string(),
        },
    )?;
    let session_stream_state = Arc::new(RwLock::new(SessionStreamState::new(session_context)));
    let session_stream = SessionStream::new(input, session_stream_state.clone());
    let response: Vec<HashMap<String, ArrowIncomingMessage>> = session_stream.try_collect().await?;

    // check the response
    println!(
        "Response name {}",
        response
            .last()
            .unwrap()
            .get("from_session_1_on_state_1")
            .unwrap()
            .get_name()
    );

    // Check the metrics
    println!(
        "Output rows {}",
        metrics.clone_inner().output_rows().unwrap()
    );
    println!(
        "Elapsed compute {}",
        metrics.clone_inner().elapsed_compute().unwrap()
    );

    Ok(())
}
