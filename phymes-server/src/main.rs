use anyhow::Result;
use clap::Parser;

pub mod handlers;
pub mod server;

// DM: need to add CLI support
#[cfg(feature = "wasip2")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    use crate::handlers::sign_in::basic_auth;
    use anyhow::anyhow;
    use bytes::Bytes;
    use futures::TryStreamExt;
    use futures_executor::block_on;
    use http::Request;
    use server::{serverless_app::Serverless, serverless_config::ServerlessConfig};

    // initialize the server
    let mut serverless = Serverless::new();

    // parse the config
    let config = ServerlessConfig::parse();

    // start building the request
    let url = format!("https://serverless/{}", config.route);
    let request_builder = Request::builder().method("POST").uri(url);

    let response = if let Some(credentials) = config.basic_auth {
        // Parse the credentials
        let mid = credentials.find(":");
        if mid.is_none() {
            return Err(anyhow!("Error: unable to parse the basic_auth."));
        }
        let (username, password) = credentials.split_at(mid.unwrap());
        let password = &password[1..];

        // Make the credentials with basic authorization
        let credentials = basic_auth(username, Some(password));

        // build the request
        let request: Request<String> = request_builder
            .header("Content-type", "text/plain; charset=utf-8")
            .header("Authorization", credentials)
            .body("".into())
            .unwrap();
        block_on(serverless.call(request))
    } else if let (Some(bearer), Some(data)) = (config.bearer_auth, config.data) {
        // Make the credentials for bearer authorization
        let bearer = format!("Bearer {bearer}");

        // build the request
        let request: Request<String> = request_builder
            .header("Content-type", "application/json")
            .header("Authorization", bearer)
            .body(data)
            .unwrap();
        block_on(serverless.call(request))
    } else {
        return Err(anyhow!(
            "Error: no basic_auth nor bearer_auth with data were provided."
        ));
    };

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
