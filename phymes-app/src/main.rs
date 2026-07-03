use dioxus::prelude::*;

mod state;
mod ui;
use ui::main_window_view;

fn main() {
    // DM: Uncomment for full stack
    // #[cfg(any(feature = "web", feature = "mobile", feature = "desktop"))]
    // dioxus::fullstack::prelude::server_fn::client::set_server_url("http://127.0.0.1:4000");
    #[cfg(any(feature = "web", feature = "mobile", feature = "desktop"))]
    dioxus::launch(app);

    #[cfg(feature = "server")]
    use clap::Parser;
    #[cfg(feature = "server")]
    use phymes_server::{Server, ServerConfig};
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
        document::Link { rel: "stylesheet", href: asset!("/assets/tailwind.css") },
        mermaid_js {},
        main_window_view {}
    }
}

fn mermaid_js() -> Element {
    #[cfg(feature = "mermaid_js_cdn")]
    rsx! {
        script { src: "https://unpkg.com/@panzoom/panzoom@4.6.0/dist/panzoom.min.js" }
        script { r#type: "module", src: asset!("/assets/mermaid.cdn.js")}
    }

    #[cfg(feature = "mermaid_js_embed")]
    rsx! {
        script { src: asset!("/assets/panzoom.min.js"), crossorigin: true }
        // DM: loading from file does not work...
        // script { r#type: "module", src: asset!("/assets/mermaid.embed.js"), crossorigin: true}
        script { src: asset!("/assets/mermaid.min.js"), crossorigin: true, onload: move |_| {
            document::eval(r#"
                mermaid.initialize({
                    theme: "dark",
                    startOnLoad: false,
                    maxTextSize: 50000,
                    maxEdges: 500,
                    securityLevel: "loose",
                    suppressErrorRendering: true
                });
                "#
            );
        } }
    }
}
