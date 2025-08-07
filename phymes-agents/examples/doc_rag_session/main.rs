mod run_main;
use run_main::run_main;

#[cfg(not(target_family = "wasm"))]
#[tokio::main]
async fn main() {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::fmt;
    use tracing_subscriber::prelude::*;
    let _guard = {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("{}=trace", env!("CARGO_CRATE_NAME")).into()),
            )
            .with(chrome_layer)
            .with(fmt::Layer::default())
            .try_init()
            .unwrap();
        Some(guard)
    };

    if let Err(e) = run_main().await {
        println!("Failed to run Candle: {e:?}");
    }
}

#[cfg(target_family = "wasm")]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = run_main().await {
        println!("Failed to run Candle: {e:?}");
    }
}
