// Dioxus imports
use dioxus::prelude::*;

use super::settings_interface::settings_modal;
use super::subjects_interface::subjects_modal;
use super::messaging_interface::{messaging_interface_view, messaging_interface_footer};
use super::metrics_interface::metrics_modal;
use super::sign_in_interface::sign_in_modal;
use super::svg_icons::{
    database_icon_svg, help_icon_svg, message_icon_svg, person_icon_svg, settings_icon_svg,
    tools_icon_svg, top_speed_icon_svg,
};
use super::tasks_interface::tasks_modal;

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

    rsx! {
        main {
            id: "chat_main",
            header {
                div {
                    class: "search",
                    // form {
                    //     id: "search_form",
                    //     input {
                    //         r#type: "text",
                    //         placeholder: "search messages",
                    //     }
                    // }
                    // // DM: convert to buttons that actually do something
                    // button { svg { dangerous_inner_html: search_icon_svg() } }
                    // DM: convert to a responsive navbar that is vertical on desktop and horizontal on mobile
                    // see https://www.w3schools.com/howto/howto_css_sidebar_responsive.asp
                    // see https://www.w3schools.com/howto/howto_css_icon_bar.asp
                    // see https://www.w3schools.com/howto/howto_js_mobile_navbar.asp
                    // DM: add tooltip for each of the icons
                    // see https://www.w3schools.com/css/css_tooltip.asp
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Help);
                        },
                        svg { dangerous_inner_html: help_icon_svg() }
                    }
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
                div {
                    class: "logo",
                    h1 { id: "logo1", "bio" }
                    h1 { id: "logo2", "M🤖ER" }
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
    rsx! {
        div {
            class: "messaging_list",
            p { "Welcome to Biom8er messaging application!" },
        }
    }
}
