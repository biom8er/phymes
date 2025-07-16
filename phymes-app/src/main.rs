// Dioxus imports
use dioxus::prelude::*;

// UI components
mod state;
mod ui;
use ui::main_window::main_window;

// CSS
static MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    // DM: Uncomment for full stack
    // #[cfg(any(feature = "web", feature = "mobile", feature = "desktop"))]
    // dioxus::fullstack::prelude::server_fn::client::set_server_url("http://127.0.0.1:4000");
    #[cfg(any(feature = "web", feature = "mobile", feature = "desktop"))]
    dioxus::launch(app);

    #[cfg(feature = "server")]
    use clap::Parser;
    #[cfg(feature = "server")]
    use phymes_server::server::{server_app::Server, server_config::ServerConfig};
    #[cfg(feature = "server")]
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            // let config = ServerConfig::parse();
            let config = ServerConfig {
                assets_dir: "./public/".to_string(),
                address: "127.0.0.1:4000".to_string(),
            };
            Server::new(config).run().await.unwrap();
        });
}

fn app() -> Element {
    // render the UI
    rsx! {
        document::Link { rel: "stylesheet", href: MAIN_CSS },
        div {
            id: "container",
            main_window {}
        }
    }
}
