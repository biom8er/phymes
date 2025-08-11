// Dioxus imports
use dioxus::prelude::*;

// UI components
mod state;
mod ui;
use ui::main_window::main_window;

// CSS
static MAIN_CSS: Asset = asset!("/assets/main.css");
#[cfg(feature = "mermaid_js_embed")]
static MERMAID_JS: Asset = asset!("/assets/mermaid.min.js");
// static MERMAID_MJS: Asset = asset!("/assets/mermaid.esm.min.mjs");

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
        mermaid_js {},
        div {
            id: "container",
            main_window {}
        }
    }
}


fn mermaid_js() -> Element {
    
    #[cfg(feature = "mermaid_js_cdn")]
    rsx! {
        script { r#type: "module", crossorigin: true, onload: move |_| {
            document::eval(r#"
                import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs").then(({ default: mermaid }) => {
                    window.mermaid = mermaid;
                    mermaid.initialize({
                        theme: "dark",
                        startOnLoad: false,
                        securityLevel: "loose"
                    });
                });
                "#
            );
        }}
    }

    #[cfg(feature = "mermaid_js_embed")]
    rsx! {
        script { src: MERMAID_JS, crossorigin: true, onload: move |_| {
            document::eval(r#"
                mermaid.initialize({
                    theme: "dark",
                    startOnLoad: false,
                    securityLevel: "loose"
                });
                "#
            );
        } }
        // // DM: This code does not yet work for loading the mjs file
        // script { src: MERMAID_MJS, r#type: "module", crossorigin: true, onload: move |_| {
        //     let path = MERMAID_MJS.bundled().bundled_path();
        //     document::eval(format!(r#"
        //         import("{path}").then(({{ default: mermaid }}) => {{
        //             window.mermaid = mermaid;
        //             mermaid.initialize({{
        //                 theme: "dark",
        //                 startOnLoad: false,
        //                 securityLevel: "loose"
        //             }});
        //         }});
        //         "#).as_str()
        //     );
        // } }
    }
}