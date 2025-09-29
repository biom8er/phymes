// Dioxus imports
use dioxus::prelude::*;

use crate::state::sign_in::{BUILDER, DEBUGGER};
use crate::ui::builds::builds_interface_footer;

use super::messaging::{messaging_interface_footer, messaging_interface_view};
use super::metrics::metrics_modal;
use super::apps::apps_interface_view;
use super::sign_in::sign_in_view;
use super::subjects::subjects_interface_view;
use super::svg_icons::{
    ms_database_icon_svg, aws_help_icon_svg, b8_logo_icon_svg, b8_menu_icon_svg, ms_message_icon_svg,
    ms_person_icon_svg, ms_apps_icon_svg, ms_top_speed_icon_svg, ms_tools_icon_svg
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
    Apps,
    Builds,
    Subjects,
    Message,
    Metrics,
}

impl HeaderMenu {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Help => "Help",
            Self::Account => "Account",
            Self::Apps => "Apps",
            Self::Builds => "Builds",
            Self::Subjects => "Subjects",
            Self::Message => "Message",
            Self::Metrics => "Metrics",
        }
    }
}

#[component]
pub fn main_window() -> Element {
    // View control signals
    let mut header_menu: Signal<HeaderMenu> = use_signal(|| HeaderMenu::Account);
    let mut navbar_toggle: Signal<bool> = use_signal(|| false);

    // Toggle the sidebar visibility
    use_effect(move || {
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
                        svg { dangerous_inner_html: b8_menu_icon_svg() }
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
                        svg { dangerous_inner_html: aws_help_icon_svg() }
                    }
                    a {
                        href: "https://github.com/biom8er/phymes",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        svg { dangerous_inner_html: b8_logo_icon_svg() }
                    }
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
                    svg { dangerous_inner_html: ms_person_icon_svg() }
                }
                if BUILDER() {
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Builds);
                        },
                        svg { dangerous_inner_html: ms_tools_icon_svg() }
                    }
                } else {
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Apps);
                        },
                        // svg { dangerous_inner_html: settings_icon_svg() }
                        svg { dangerous_inner_html: ms_apps_icon_svg() }
                    }
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Message);
                        },
                        svg { dangerous_inner_html: ms_message_icon_svg() }
                    }
                }
                if DEBUGGER() {
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Subjects);
                        },
                        svg { dangerous_inner_html: ms_database_icon_svg() }
                    }
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Metrics);
                        },
                        svg { dangerous_inner_html: ms_top_speed_icon_svg() }
                    }
                }                
            }

            // DM: required because each component is its own type!
            if header_menu.read().as_str() == "Help" {
                about_text_modal {},
            } else if header_menu.read().as_str() == "Account" {
                sign_in_view {},
            } else if header_menu.read().as_str() == "Builds" {
                apps_interface_view {},
                split_panel_drag_handle {}, 
                builds_interface_footer {},
            } else if header_menu.read().as_str() == "Apps" {
                apps_interface_view {},
            } else if header_menu.read().as_str() == "Subjects" {
                subjects_interface_view {},
            } else if header_menu.read().as_str() == "Message" {
                messaging_interface_view {},
                messaging_interface_footer {},
            }else if header_menu.read().as_str() == "Metrics" {
                metrics_modal {},
            }
        }
    }
}

/// Split panel vertical drag
/// 
/// # Notes
/// * this component is a work in progress...
/// * the JS listeners are necessary for the component to work
/// * the dioxus signals are also needed to trigger the JS code
/// * attempting to call the JS directly without the listeners results in bugs
///   See commented code for trying to resize with JS directly
#[component]
pub fn split_panel_drag_handle() -> Element {
    // let mut is_dragging: Signal<bool> = use_signal(|| false);
    // let mut y_coordinate: Signal<f64> = use_signal(|| 0.0 as f64);
    let mut js_trigger: Signal<bool> = use_signal(|| false);
    
    // use_effect(move || {
    //     // Resize the two window windows
    //     let y_coordinate = y_coordinate.read().to_owned();
    //     document::eval(format!(
    //         r#"const container = document.querySelector('#container');
    //         const topPane = document.querySelector('.messaging_list');
    //         const bottomPane = document.querySelector('.resizable_text_input');
    //         const dragHandle = document.querySelector('.drag-handle');

    //         const containerHeight = container.offsetHeight;
    //         const offsetY = {y_coordinate} - container.getBoundingClientRect().top;
    //         let percentHeight = (offsetY / containerHeight) * 100;

    //         // Clamp between 10% and 90% for usability
    //         percentHeight = Math.max(10, Math.min(90, percentHeight));
    //         percentHeightChange = 100 - percentHeight;

    //         topPane.style.height = `${{percentHeight}}%`;
    //         bottomPane.style.height = `${{percentHeightChange}}%`;"#).as_str()
    //     );
    // });
    use_effect(move || {
        // Resize the two window windows
        let _js_trigger = js_trigger.read().to_owned();
        document::eval(
            r#"const container = document.querySelector('#container');
            const topPane = document.querySelector('.messaging_list');
            const bottomPane = document.querySelector('.resizable_text_input');
            const dragHandle = document.querySelector('.drag-handle');

            let isDragging = false;

            dragHandle.addEventListener('mousedown', (e) => {
                isDragging = true;
                document.body.style.cursor = 'row-resize';
            });

            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;

                const containerHeight = container.offsetHeight;
                const offsetY = e.clientY - container.getBoundingClientRect().top;
                let percentHeight = (offsetY / containerHeight) * 100;

                // Clamp between 10% and 90% for usability
                percentHeight = Math.max(10, Math.min(90, percentHeight));
                percentHeightChange = 100 - percentHeight;

                topPane.style.height = `${percentHeight}%`;
                bottomPane.style.height = `${percentHeightChange}%`;
            });

            document.addEventListener('mouseup', () => {
                isDragging = false;
                document.body.style.cursor = 'default';
            });"#
        );
    });

    rsx! {
        div {
            class: "drag-handle",
            // onmousemove: move |event| {
            //     if is_dragging() {
            //         y_coordinate.set(event.client_coordinates().y);
            //     }
            // },
            // onmousedown: move |_| is_dragging.set(true),
            // onmouseup: move |_| is_dragging.set(false),
            onclick: move |_| {
                let current = js_trigger.read().to_owned();
                js_trigger.set(!current);
            },
            onmousedown: move |_| {
                let current = js_trigger.read().to_owned();
                js_trigger.set(!current);
            },
            onmouseup: move |_| {
                let current = js_trigger.read().to_owned();
                js_trigger.set(!current);
            },
            onmousemove: move |_| {
                let current = js_trigger.read().to_owned();
                js_trigger.set(!current);
            },
            onmouseenter: move |_| {
                let current = js_trigger.read().to_owned();
                js_trigger.set(!current);
            },
            onmouseleave: move |_| {
                let current = js_trigger.read().to_owned();
                js_trigger.set(!current);
            },
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
                        svg { dangerous_inner_html: aws_help_icon_svg() }
                        p { "(Hopefully 🤞) useful information for using PHYMES 😇. Please create an issue on GitHubp https://github.com/biom8er/phymes/issues if you run into problems." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "Menu" }
                        svg { dangerous_inner_html: b8_menu_icon_svg() }
                        p { "Hide or show the menu items below." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Builds.as_str()}" }
                        svg { dangerous_inner_html: ms_tools_icon_svg() }
                        p { "A list of session plans available to the account. Each session is like a different app with different functionality and state. Only one session can be activated at a time. A schematic of the session plan with all main components is rendered using mermaid.js. The mermaid.js script is provided in the footer, and can be modified to create new session plans." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Apps.as_str()}" }
                        svg { dangerous_inner_html: ms_apps_icon_svg() }
                        p { "A list of session plans available to the account. Each session is like a different app with different functionality and state. Only one session can be activated at a time. A schematic of the session plan with all main components is rendered using mermaid.js. The mermaid.js script is provided in the footer, and can be modified to create new session plans." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Message.as_str()}" }
                        svg { dangerous_inner_html: ms_message_icon_svg() }
                        p { "The message history for the active session plan. A chat interface is provided for users to publish messages to the messages subject and to receive subscriptions from the messages subject when the messages subject is updated." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Subjects.as_str()}" }
                        svg { dangerous_inner_html: ms_database_icon_svg() }
                        p { "A list of subject associated with the active session plan. A table shows the schema of the subject tables along with the number if rows. The subject tables can be extended or replaced by uploading tables in comma deliminated CSV format with headers that match the subject. The subject tables can also be downloaded in comma deliminated CSV format. Note that all of the parameters for describing how processors process streaming messages are subject tables. Extending the subject tables for a processors parameters will update the processors parameters on the next run. Note that the message history is also a subject table. Extending the messages table is the equivalent of human in the loop." }
                    }
                }
                li {
                    div {
                        class: "help_li_item",
                        h2 { "{HeaderMenu::Metrics.as_str()}" }
                        svg { dangerous_inner_html: ms_top_speed_icon_svg() }
                        p { "A list of metrics associated with the active session plan. Metrics are tracked per processor. Baseline metrics for row count, and processor start, stop, and total time in nanoseconds are provided. The baseline metrics are visually represented using mermaid.js gantt charts. Note each row is approximately one token for text generation inference processors. Please submit a feature request issue if additional metrics are of interest." }
                    }
                }
            }
        }
    }
}
