use anyhow::Result;
use clap::Parser;
use phymes_server::server;

// DM: need to add CLI support
#[cfg(feature = "wasip2")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    use bytes::Bytes;
    use futures::TryStreamExt;
    use server::{
        serverless_app::{Serverless, serverless_app},
        serverless_config::ServerlessConfig,
    };

    // parse the config
    let config = ServerlessConfig::parse();

    // call the serverless application
    let mut serverless = Serverless::new();
    let response = serverless_app(config, &mut serverless).await.unwrap();

    // Parse the response
    let bytes: Vec<Bytes> = response
        .into_body()
        .into_data_stream()
        .try_collect()
        .await
        .unwrap();
    println!("{bytes:?}");
    Ok(())
}

#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
#[tokio::main]
async fn main() -> Result<()> {
    use server::{server_app::Server, server_config::ServerConfig};
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
