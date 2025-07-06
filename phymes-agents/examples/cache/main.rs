mod run_main;
use run_main::run_main;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = run_main().await {
        println!("Failed to run Candle: {e:?}");
    }
}
