// Dioxus imports
use dioxus::prelude::*;

#[cfg(feature = "mermaid_js")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "mermaid_js")]
use phymes_core::session::session_context_builder::SessionContextBuilder;
#[cfg(feature = "mermaid_js")]
use phymes_agents::session_traits::mermaid_js::SessionContextBuilderMermaidTrait;

#[cfg(feature = "mermaid_js")]
#[derive(Debug, Deserialize, Serialize)]
struct MermaidJsObject {
    svg: Option<String>,
    error: Option<String>,
}

use super::messaging::{messaging_interface_footer, messaging_interface_view};
use super::metrics::metrics_modal;
use super::settings::{settings_interface_view, settings_interface_footer};
use super::sign_in::sign_in_modal;
use super::subjects::subjects_modal;
use super::svg_icons::{
    database_icon_svg, help_icon_svg, logo_icon_svg, menu_icon_svg, message_icon_svg,
    person_icon_svg, settings_icon_svg, top_speed_icon_svg,
};

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
                settings_interface_view {},
                settings_interface_footer {},
            } else if header_menu.read().as_str() == "Subjects" {
                subjects_modal {},
            } else if header_menu.read().as_str() == "Message" {
                messaging_interface_view {},
                messaging_interface_footer {},
            }else if header_menu.read().as_str() == "Metrics" {
                metrics_modal {},
            }
        }
    }
}

#[cfg(feature = "mermaid_js")]
pub fn render_mermaid_svg(diagram_code: Signal<String>, id: &str) -> Resource<(Option<String>,Option<String>,Option<String>)> {
    let div_id = id.to_string();
    let rendered_html: Resource<(Option<String>,Option<String>,Option<String>)> = use_resource(move || {
        let div_id = div_id.clone();
        async move {
            // Render the mermaid.js diagram
            let eval = document::eval(format!(r#"
                try {{
                    let code = await dioxus.recv();
                    const {{ svg }} = await mermaid.render("{div_id}", code);
                    return {{ svg: svg, error: null }};
                }} catch (error) {{
                    return {{ svg: null, error: error.message }};
                }}"#).as_str()
            );
            eval.send(diagram_code.read().to_string()).unwrap();
            let mermaid_js_object = match eval.await {
                Ok(res) => {
                    let res: MermaidJsObject = serde_json::from_value(res).unwrap();
                    res
                },
                Err(err) => {
                    tracing::error!("Mermaid.js err {err:?}");
                    MermaidJsObject { svg: None, error: Some(err.to_string())}
                }
            };

            // Build the preliminary session context
            let builder_error =  match SessionContextBuilder::from_mermaid_flowchart(&diagram_code.read().to_string()) {
                Ok(_res) => None,
                Err(err) => Some(err.to_string()),
            };

            (mermaid_js_object.svg, mermaid_js_object.error, builder_error)
        }
    });

    // add pan and zoom
    let div_id = id.to_string();
    use_effect(move || {
        let div_id = div_id.clone();
        let _ = rendered_html.read();
        document::eval(format!(r#"
            const container = document.getElementById("{div_id}");
            const svgElement = container.querySelector("svg");

            // Initialize Panzoom
            const panzoomInstance = Panzoom(svgElement, {{
                maxScale: 5,
                minScale: 0.5,
                step: 0.1,
            }});

            // Add mouse wheel zoom
            container.addEventListener("wheel", (event) => {{
                panzoomInstance.zoomWithWheel(event);
            }});
            "#).as_str()
        );
    });

    rendered_html
}

#[component]
pub fn about_text_modal() -> Element {
    let mut diagram_code = use_signal(|| String::from("graph TB\n\ta-->b"));
    let rendered_html = render_mermaid_svg(diagram_code, "graphDiv");
    
    let out = if let Some(result) = &*rendered_html.read() {
        match result {
            // Mermaid.js error
            (_, Some(error), None) => {
                rsx! {
                    div {
                        class: "messaging_list",
                        p { "Welcome to PHYMES by Biom🤖er" }
                        div {
                            class: "text_input",
                            form {
                                textarea {
                                    rows: "10",
                                    cols: "40",
                                    value: "{diagram_code}",
                                    oninput: move |evt| diagram_code.set(evt.value()),
                                }
                            }
                        }
                        p { "{error}" },
                    }            
                }
            }
            // SessionContextBuilder error
            (_, None, Some(error)) => {
                rsx! {
                    div {
                        class: "messaging_list",
                        p { "Welcome to PHYMES by Biom🤖er" }
                        div {
                            class: "text_input",
                            form {
                                textarea {
                                    rows: "10",
                                    cols: "40",
                                    value: "{diagram_code}",
                                    oninput: move |evt| diagram_code.set(evt.value()),
                                }
                            }
                        }
                        p { "{error}" },
                    }            
                }
            }
            // Mermaid.js and SessionContextBuilder error
            (_, Some(error_mjs), Some(error_ctxb)) => {
                rsx! {
                    div {
                        class: "messaging_list",
                        p { "Welcome to PHYMES by Biom🤖er" }
                        div {
                            class: "text_input",
                            form {
                                textarea {
                                    rows: "10",
                                    cols: "40",
                                    value: "{diagram_code}",
                                    oninput: move |evt| diagram_code.set(evt.value()),
                                }
                            }
                        }
                        p { "{error_mjs}" },
                        p { "{error_ctxb}" },
                    }            
                }
            }
            // Valid SVG with no errors
            (Some(svg), _, _) => {
                rsx! {
                    div {
                        class: "messaging_list",
                        p { "Welcome to PHYMES by Biom🤖er" }
                        div {
                            class: "text_input",
                            form {
                                textarea {
                                    rows: "10",
                                    cols: "40",
                                    value: "{diagram_code}",
                                    oninput: move |evt| diagram_code.set(evt.value()),
                                }
                            }
                        }
                        div {
                            id: "graphDiv",
                            class: "mermaid",
                            svg { dangerous_inner_html: svg.to_string() }
                        }
                    }
                }
            }
            // All other cases
            (_, _, _) => {
                rsx! {
                    div {
                        class: "messaging_list",
                        p { "Welcome to PHYMES by Biom🤖er" },
                        div {
                            class: "text_input",
                            form {
                                textarea {
                                    rows: "10",
                                    cols: "40",
                                    value: "{diagram_code}",
                                    oninput: move |evt| diagram_code.set(evt.value()),
                                }
                            }
                        }
                    }            
                }
            }
        }
    } else {
        rsx! {
            div {
                class: "messaging_list",
                p { "Welcome to PHYMES by Biom🤖er" },
                div {
                    class: "text_input",
                    form {
                        textarea {
                            rows: "10",
                            cols: "40",
                            value: "{diagram_code}",
                            oninput: move |evt| diagram_code.set(evt.value()),
                        }
                    }
                }
            }            
        }
    };
    out
}
