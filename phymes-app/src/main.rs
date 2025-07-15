// Dioxus imports
use dioxus::prelude::*;

// UI components
mod ui;
use ui::main_window::main_window;

// CSS
static MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    #[cfg(any(feature = "web", feature = "mobile", feature = "desktop"))]
    dioxus::launch(app);

    #[cfg(feature = "server")]
    {
        use clap::Parser;
        use phymes_server::server::{server_app::Server, server_config::ServerConfig};
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let config = ServerConfig::parse();
                Server::new(config).run().await?;
            }
        );
    }
}

fn app() -> Element {
    // render the UI
    rsx! {
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        div {
            id: "container",
            main_window {}
        }
    }
}
