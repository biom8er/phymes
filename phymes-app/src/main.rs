// Dioxus imports
use dioxus::prelude::*;

// UI components
mod state;
mod ui;
use ui::main_window_view;

// CSS
// static MAIN_CSS: Asset = asset!("/assets/main.css");
static TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");
#[cfg(feature = "mermaid_js_embed")]
static MERMAID_JS: Asset = asset!("/assets/mermaid.min.js");
// static MERMAID_MJS: Asset = asset!("/assets/mermaid.esm.min.mjs");
#[cfg(feature = "mermaid_js_embed")]
static PANZOOM_JS: Asset = asset!("/assets/panzoom.min.js");

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
        // document::Link { rel: "stylesheet", href: MAIN_CSS },
        document::Link { rel: "stylesheet", href: TAILWIND_CSS },
        mermaid_js {},
        div {
            class: "w-screen h-screen bg-gray-900 text-white flex flex-col",
            main_window_view {}
        }
    }
}

fn mermaid_js() -> Element {
    #[cfg(feature = "mermaid_js_cdn")]
    rsx! {
        script { src: "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs", onload: move |_| {
            document::eval(r#"
                import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs").then(({ default: mermaid }) => {
                    window.mermaid = mermaid;
                    mermaid.initialize({
                        theme: "dark",
                        startOnLoad: false,
                        securityLevel: "loose",
                        suppressErrorRendering: true
                    });
                });
                "#
            );
        }}
        script { src: "https://unpkg.com/@panzoom/panzoom@4.6.0/dist/panzoom.min.js" }
    }

    #[cfg(feature = "mermaid_js_embed")]
    rsx! {
        script { src: MERMAID_JS, crossorigin: true, onload: move |_| {
            document::eval(r#"
                mermaid.initialize({
                    theme: "dark",
                    startOnLoad: false,
                    securityLevel: "loose",
                    suppressErrorRendering: true
                });
                "#
            );
        } }
        // // DM: This code does not yet work for loading the mjs file
        // script { src: MERMAID_MJS, crossorigin: true, onload: move |_| {
        //     let path = MERMAID_MJS.bundled().bundled_path();
        //     document::eval(format!(r#"
        //         import("{path}").then(({{ default: mermaid }}) => {{
        //             window.mermaid = mermaid;
        //             mermaid.initialize({{
        //                 theme: "dark",
        //                 startOnLoad: false,
        //                 securityLevel: "loose",
        //                 suppressErrorRendering: true
        //             }});
        //         }});
        //         "#).as_str()
        //     );
        // } }
        script { src: PANZOOM_JS, crossorigin: true }
    }
}
