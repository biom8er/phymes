// Dioxus imports
use dioxus::prelude::*;

#[cfg(feature = "mermaid_js")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "mermaid_js")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "mermaid_js")]
use wasm_bindgen_futures::spawn_local;

#[cfg(feature = "mermaid_js")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = mermaid)]
    fn parse(code: &str) -> bool;
    #[wasm_bindgen(js_namespace = mermaid)]
    fn run();
    #[wasm_bindgen(js_namespace = mermaid, catch)]
    async fn render(id: &str, code: &str) -> Result<JsValue, JsValue>;
}

#[cfg(feature = "mermaid_js")]
#[derive(Serialize, Deserialize)]
pub struct MermaidSvgObject {
    svg: String
}

#[cfg(feature = "mermaid_js")]
#[derive(Debug, Deserialize)]
struct JsError {
    message: String,
    name: Option<String>,
    stack: Option<String>,
}

use super::messaging::{messaging_interface_footer, messaging_interface_view};
use super::metrics::metrics_modal;
use super::settings::settings_modal;
use super::sign_in::sign_in_modal;
use super::subjects::subjects_modal;
use super::svg_icons::{
    database_icon_svg, help_icon_svg, logo_icon_svg, menu_icon_svg, message_icon_svg,
    person_icon_svg, settings_icon_svg, tools_icon_svg, top_speed_icon_svg,
};
use super::tasks::tasks_modal;

#[component]
pub fn title() -> Element {
    rsx! {
        h1 { "Biom8er agentic messaging" }
    }
}

pub enum HeaderMenu {
    Help,
    Account,
    Settings,
    Subjects,
    Tasks,
    Message,
    Metrics,
}

impl HeaderMenu {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Help => "Help",
            Self::Account => "Account",
            Self::Settings => "Settings",
            Self::Subjects => "Subjects",
            Self::Tasks => "Tasks",
            Self::Message => "Message",
            Self::Metrics => "Metrics",
        }
    }
}

#[component]
pub fn main_window() -> Element {
    let mut header_menu: Signal<HeaderMenu> = use_signal(|| HeaderMenu::Account);
    let mut navbar_toggle: Signal<bool> = use_signal(|| false);

    use_effect(move || {
        // Toggle the sidebar visibility
        let navbar_toggle = navbar_toggle.read();
        document::eval(
            format!(
                r#" var nav_toggle = {navbar_toggle};
            var elements = document.getElementsByClassName("sidebar");
            for (var i = 0; i < elements.length; i++) {{
                var x = elements[i];
                if (x.style.display === "none") {{
                    x.style.display = "block";
                }} else {{
                    x.style.display = "none";
                }}
            }}
            var elements = document.getElementsByClassName("messaging_list");
            for (var i = 0; i < elements.length; i++) {{
                var x = elements[i];
                if (x.style.marginLeft  === "0px") {{
                    x.style.display = "64px";
                }} else {{
                    x.style.display = "0px";
                }}
            }}"#
            )
            .as_str(),
        );
    });

    rsx! {
        main {
            id: "chat_main",
            header {
                div {
                    class: "navbar",
                    label {
                        class: "checkbtn",
                        r#for: "navbartoggle",
                        svg { dangerous_inner_html: menu_icon_svg() }
                    }
                    input {
                        r#type: "checkbox",
                        id: "navbartoggle",
                        onclick: move |_| {
                            let current = navbar_toggle.read().to_owned();
                            navbar_toggle.set(!current);
                        },
                    }
                }
                div {
                    class: "search",
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Help);
                        },
                        svg { dangerous_inner_html: help_icon_svg() }
                    }
                    a {
                        href: "https://github.com/biom8er/phymes",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        svg { dangerous_inner_html: logo_icon_svg() }
                    }
                    // form {
                    //     id: "search_form",
                    //     input {
                    //         r#type: "text",
                    //         placeholder: "search messages",
                    //     }
                    // }
                    // // DM: convert to buttons that actually do something
                    // button { svg { dangerous_inner_html: search_icon_svg() } }
                }
            }

            div {
                class: "sidebar",
                // DM: add tooltip for each of the icons
                // see https://www.w3schools.com/css/css_tooltip.asp
                button {
                    onclick: move |_| async move {
                        header_menu.set(HeaderMenu::Account);
                    },
                    svg { dangerous_inner_html: person_icon_svg() }
                }
                button {
                    onclick: move |_| async move {
                        header_menu.set(HeaderMenu::Settings);
                    },
                    svg { dangerous_inner_html: settings_icon_svg() }
                }
                button {
                    onclick: move |_| async move {
                        header_menu.set(HeaderMenu::Subjects);
                    },
                    svg { dangerous_inner_html: database_icon_svg() }
                }
                button {
                    onclick: move |_| async move {
                        header_menu.set(HeaderMenu::Tasks);
                    },
                    svg { dangerous_inner_html: tools_icon_svg() }
                }
                button {
                    onclick: move |_| async move {
                        header_menu.set(HeaderMenu::Message);
                    },
                    svg { dangerous_inner_html: message_icon_svg() }
                }
                button {
                    onclick: move |_| async move {
                        header_menu.set(HeaderMenu::Metrics);
                    },
                    svg { dangerous_inner_html: top_speed_icon_svg() }
                }
            }

            // DM: required because each component is its own type!
            if header_menu.read().as_str() == "Help" {
                about_text_modal {},
            } else if header_menu.read().as_str() == "Account" {
                sign_in_modal {},
            } else if header_menu.read().as_str() == "Settings" {
                settings_modal {},
            } else if header_menu.read().as_str() == "Subjects" {
                subjects_modal {},
            } else if header_menu.read().as_str() == "Tasks" {
                tasks_modal {},
            } else if header_menu.read().as_str() == "Message" {
                messaging_interface_view {},
                messaging_interface_footer {},
            }else if header_menu.read().as_str() == "Metrics" {
                metrics_modal {},
            }
        }
    }
}

#[component]
pub fn about_text_modal() -> Element {
    let mut diagram_code = use_signal(|| String::from(r#"graph TB
    a-->b"#));
    let mut rendered_html = use_signal(|| String::new());

    #[cfg(feature = "mermaid_js")]
    use_effect( move || {
        let code = diagram_code.read().clone();        
        spawn_local(async move {
            match render("graphDiv", &code).await{
                Ok(svg) => {
                    let obj_str = match serde_wasm_bindgen::from_value::<MermaidSvgObject>(svg) {
                        Ok(obj) => obj,
                        Err(err) => MermaidSvgObject { svg: err.to_string()},
                    };
                    let escaped_str = format!("'{0}'", obj_str.svg);
                    rendered_html.set(escaped_str);
                }
                Err(err) => {
                    let obj_str = match serde_wasm_bindgen::from_value::<JsError>(err) {
                        Ok(obj) => obj,
                        Err(err) => JsError { message: err.to_string(), name: None, stack: None },
                    };
                    rendered_html.set(obj_str.message);
                }
            }
        });
        // run(); // render all "mermaid" classes (cannot be dynamically updated)
    });

    #[cfg(not(any(feature = "plotly_embed_js", feature = "plotly_cdn_js")))]
    rsx! {
        div {
            class: "messaging_list",
            p { "Welcome to PHYMES by Biom🤖er" },
            textarea {
                rows: "10",
                cols: "40",
                value: "{diagram_code}",
                oninput: move |evt| diagram_code.set(evt.value()),
            }            
            div {
                id: "graphDiv",
                class: "mermaid",
                svg { dangerous_inner_html: rendered_html.read().to_string() },
                // "{diagram_code.read()}"
            }
            div {
                p { "{rendered_html.read()}" },
            }
        }
    }
}
