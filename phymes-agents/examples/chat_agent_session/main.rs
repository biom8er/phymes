mod run_main;
use run_main::run_main;

#[tokio::main(flavor = "current_thread")]
async fn main() {
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

    if let Err(e) = run_main().await {
        println!("Failed to run Candle: {e:?}");
    }
}
