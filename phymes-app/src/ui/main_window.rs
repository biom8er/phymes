// Dioxus imports
use dioxus::prelude::*;

// Plotting imports
use plotly::{color::Rgb, image::ColorModel, Image, Plot};

use super::messaging_interface::{messaging_interface_footer, messaging_interface_view};
use super::metrics_interface::metrics_modal;
use super::settings_interface::settings_modal;
use super::sign_in_interface::sign_in_modal;
use super::subjects_interface::subjects_modal;
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
    use_future(move || async move {
        let w = Rgb::new(255, 255, 255);
        let b = Rgb::new(0, 0, 0);
        let r = Rgb::new(240, 8, 5);
        let db = Rgb::new(145, 67, 7);
        let lb = Rgb::new(251, 200, 129);
        let s = Rgb::new(153, 75, 10);
        let bl = Rgb::new(3, 111, 191);
        let y = Rgb::new(251, 250, 15);

        let pixels = vec![
            vec![b, b, b, b, r, r, r, r, r, b, b, b, b, b, b],
            vec![b, b, b, r, r, r, r, r, r, r, r, r, b, b, b],
            vec![b, b, b, db, db, db, lb, lb, b, lb, b, b, b, b, b],
            vec![b, b, db, lb, db, lb, lb, lb, w, lb, lb, lb, b, b, b],
            vec![b, b, db, lb, db, db, lb, lb, lb, db, lb, lb, lb, b, b],
            vec![b, b, db, db, lb, lb, lb, lb, db, db, db, db, b, b, b],
            vec![b, b, b, b, lb, lb, lb, lb, lb, lb, lb, b, b, b, b],
            vec![b, b, b, r, r, bl, r, r, r, b, b, b, b, b, b],
            vec![b, b, r, r, r, bl, r, r, bl, r, r, r, b, b, b],
            vec![b, r, r, r, r, bl, bl, bl, bl, r, r, r, r, b, b],
            vec![b, lb, lb, r, bl, y, bl, bl, y, bl, r, lb, lb, b, b],
            vec![b, lb, lb, lb, bl, bl, bl, bl, bl, bl, lb, lb, lb, b, b],
            vec![b, lb, lb, bl, bl, bl, bl, bl, bl, bl, bl, lb, lb, b, b],
            vec![b, b, b, bl, bl, bl, b, b, bl, bl, bl, b, b, b, b],
            vec![b, b, s, s, s, b, b, b, b, s, s, s, b, b, b],
            vec![b, s, s, s, s, b, b, b, b, b, s, s, s, s, b],
        ];
        let trace = Image::new(pixels).color_model(ColorModel::RGB);
        let layout = plotly::Layout::new()
            .paper_background_color(b)
            .x_axis(plotly::layout::Axis::new().show_tick_labels(false))
            .y_axis(plotly::layout::Axis::new().show_tick_labels(false));

        let mut plot = Plot::new();
        plot.add_trace(trace);
        plot.set_layout(layout);

        #[cfg(target_family = "wasm")]
        plotly::bindings::new_plot("plot-div", &plot).await;
    });

    rsx! {
        div {
            class: "messaging_list",
            p { "Welcome to Biom8er messaging application!" },
            div { id: "plot-div" }
        }
    }
}
