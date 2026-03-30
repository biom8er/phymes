use anyhow::Result;
use clap::Parser;

// DM: need to add CLI support
#[cfg(all(feature = "wasip2", not(feature = "wsl")))]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    use bytes::Bytes;
    use futures::TryStreamExt;
    use futures_executor::block_on;
    use phymes_core::{
        BuildableTrait, BuilderTrait, ObjectStorageBackend, RuntimeEnv, RuntimeEnvBuilderTrait,
        make_store,
    };
    use phymes_server::{Serverless, ServerlessConfig, serverless_app};
    use std::sync::Arc;

    // parse the config
    let config = ServerlessConfig::parse();

    // create the runtime
    let runtime_env = if let Some(bucket) = config.object_store_bucket.as_ref() {
        let store = make_store(&ObjectStorageBackend::InMemory, Some(bucket), None)?;
        RuntimeEnv::get_builder()
            .with_name("Serverless App Runtime Environment")
            .with_object_store(store)
            .with_object_store_backend(&ObjectStorageBackend::InMemory)
            .with_object_store_bucket(bucket)
            .build_arc()?
    } else {
        Arc::new(RuntimeEnv::default())
    };

    // call the serverless application
    let mut serverless = Serverless::new(None, &runtime_env).await.unwrap();
    // DM: blocking on serverless_app hangs indefinitely...
    // let response = block_on(serverless_app(config, &mut serverless)).unwrap();
    let response = serverless_app(config, &mut serverless).await.unwrap();

    // parse the response
    let bytes: Vec<Bytes> = response.into_body().into_data_stream().try_collect().await?;
    // DM: blocking on the response results in an empty array for session_stream...
    // let bytes: Vec<Bytes> = block_on(response.into_body().into_data_stream().try_collect()).unwrap();

    println!("{bytes:?}");
    Ok(())
}

#[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
#[tokio::main]
async fn main() -> Result<()> {
    use phymes_server::{Server, ServerConfig};
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    // initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("{}=trace", env!("CARGO_CRATE_NAME")).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = ServerConfig::parse();
    Server::new(config).run().await?;

    Ok(())
}
