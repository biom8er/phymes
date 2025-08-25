// Dioxus imports
use dioxus::prelude::*;

use super::messaging::{messaging_interface_footer, messaging_interface_view};
use super::metrics::metrics_modal;
use super::settings::{settings_interface_footer, settings_interface_view};
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

/// About text view with information on using the application
///
/// # Notes
/// * all of the text should match /phymes-book/src/phymes-app/ui.md
#[component]
pub fn about_text_modal() -> Element {
    rsx! {
        div {
            class: "messaging_list",
            p { "Welcome to PHYMES by Biom🤖er" },
            ul {
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Help.as_str()}" }
                        svg { dangerous_inner_html: help_icon_svg() }
                        p { "(Hopefully 🤞) useful information for using PHYMES 😇. Please create an issue on GitHubp https://github.com/biom8er/phymes/issues if you run into problems." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "Menu" }
                        svg { dangerous_inner_html: menu_icon_svg() }
                        p { "Hide or show the menu items below." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Settings.as_str()}" }
                        svg { dangerous_inner_html: settings_icon_svg() }
                        p { "A list of session plans available to the account. Each session is like a different app with different functionality and state. Only one session can be activated at a time. A schematic of the session plan with all main components is rendered using mermaid.js. The mermaid.js script is provided in the footer, and can be modified to create new session plans." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Subjects.as_str()}" }
                        svg { dangerous_inner_html: database_icon_svg() }
                        p { "A list of subject associated with the active session plan. A table shows the schema of the subject tables along with the number if rows. The subject tables can be extended or replaced by uploading tables in comma deliminated CSV format with headers that match the subject. The subject tables can also be downloaded in comma deliminated CSV format. Note that all of the parameters for describing how processors process streaming messages are subject tables. Extending the subject tables for a processors parameters will update the processors parameters on the next run. Note that the message history is also a subject table. Extending the messages table is the equivalent of human in the loop." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Message.as_str()}" }
                        svg { dangerous_inner_html: message_icon_svg() }
                        p { "The message history for the active session plan. A chat interface is provided for users to publish messages to the messages subject and to receive subscriptions from the messages subject when the messages subject is updated." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Metrics.as_str()}" }
                        svg { dangerous_inner_html: top_speed_icon_svg() }
                        p { "A list of metrics associated with the active session plan. Metrics are tracked per processor. Baseline metrics for row count, and processor start, stop, and total time in nanoseconds are provided. The baseline metrics are visually represented using mermaid.js gantt charts. Note each row is approximately one token for text generation inference processors. Please submit a feature request issue if additional metrics are of interest." }
                    }
                }
            }
        }
    }
}
