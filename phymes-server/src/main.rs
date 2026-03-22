use anyhow::Result;
use clap::Parser;

// DM: need to add CLI support
#[cfg(feature = "wasip2")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    use bytes::Bytes;
    use futures::TryStreamExt;
    use futures_executor::block_on;
    use phymes_server::{Serverless, ServerlessConfig, serverless_app};

    // parse the config
    let config = ServerlessConfig::parse();

    // call the serverless application
    let mut serverless = Serverless::new(None).await.unwrap();
    // DM: blocking on serverless_app hangs indefinitely...
    // let response = block_on(serverless_app(config, &mut serverless)).unwrap();
    let response = serverless_app(config, &mut serverless).await.unwrap();

    // parse the response
    let bytes: Vec<Bytes> =
        block_on(response.into_body().into_data_stream().try_collect()).unwrap();

    println!("{bytes:?}");
    Ok(())
}

#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
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
