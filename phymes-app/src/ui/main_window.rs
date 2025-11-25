// Dioxus imports
use dioxus::prelude::*;

use crate::state::{
    svg_icons::{
        aws_help_icon_svg, b8_logo_icon_svg, ms_apps_icon_svg,
        ms_attachment_icon_svg, ms_database_icon_svg, ms_message_icon_svg, ms_person_icon_svg,
        ms_tools_icon_svg, ms_top_speed_icon_svg,
    },
    BUILDER, DEBUGGER,
};
use crate::ui::{
    apps_interface_view, attachments_interface_view, messaging_interface_view,
    metrics_interface_view, sign_in_view, subjects_interface_view,
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
    Messages,
    Attachments,
    Subjects,
    Metrics,
}

impl HeaderMenu {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Help => "Help",
            Self::Account => "Account",
            Self::Apps => "Apps",
            Self::Builds => "Builds",
            Self::Messages => "Messages",
            Self::Attachments => "Attachments",
            Self::Subjects => "Subjects",
            Self::Metrics => "Metrics",
        }
    }
}

#[component]
pub fn main_window_view() -> Element {
    // View control signals
    let mut header_menu: Signal<HeaderMenu> = use_signal(|| HeaderMenu::Account);

    rsx! {
        main {            
            class: "w-screen h-screen bg-gray-900 text-white flex flex-col sm:flex-row",

            // Responsive sidebar that is horizontal on mobile and vertical on large screens
            aside {
                class: "sm:w-[64px] w-full sm:h-full h-[64px] bg-gray-800 flex sm:flex-col flex-row items-center py-2 space-y-2",
                div {
                    class: "sm:w-auto w-[calc(100%-128px)] sm:h-[calc(100%-128px)] h-auto place-content-start",
                    // DM: add tooltip for each of the icons
                    // see https://www.w3schools.com/css/css_tooltip.asp
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Account);
                        },
                        class: "p-1 rounded hover:bg-gray-700 cursor-pointer",
                        svg { 
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: ms_person_icon_svg() 
                        }
                    }
                    if BUILDER() {
                        button {
                            onclick: move |_| async move {
                                header_menu.set(HeaderMenu::Builds);
                            },
                            class: "p-1 rounded hover:bg-gray-700 cursor-pointer",
                            svg { 
                                class: "max-w-[48px] max-h-[48px]",
                                dangerous_inner_html: ms_tools_icon_svg() 
                            }
                        }
                    } else {
                        button {
                            onclick: move |_| async move {
                                header_menu.set(HeaderMenu::Apps);
                            },
                            class: "p-1 rounded hover:bg-gray-700 cursor-pointer",
                            svg { 
                                class: "max-w-[48px] max-h-[48px]",
                                dangerous_inner_html: ms_apps_icon_svg() 
                            }
                        }
                        button {
                            onclick: move |_| async move {
                                header_menu.set(HeaderMenu::Messages);
                            },
                            class: "p-1 rounded hover:bg-gray-700 cursor-pointer",
                            svg { 
                                class: "max-w-[48px] max-h-[48px]",
                                dangerous_inner_html: ms_message_icon_svg() 
                            }
                        }
                        button {
                            onclick: move |_| async move {
                                header_menu.set(HeaderMenu::Attachments);
                            },
                            class: "p-1 rounded hover:bg-gray-700 cursor-pointer",
                            svg { 
                                class: "max-w-[48px] max-h-[48px]",
                                dangerous_inner_html: ms_attachment_icon_svg() 
                            }
                        }
                    }
                    if DEBUGGER() {
                        button {
                            onclick: move |_| async move {
                                header_menu.set(HeaderMenu::Subjects);
                            },
                            class: "p-1 rounded hover:bg-gray-700 cursor-pointer",
                            svg { 
                                class: "max-w-[48px] max-h-[48px]",
                                dangerous_inner_html: ms_database_icon_svg() 
                            }
                        }
                        button {
                            onclick: move |_| async move {
                                header_menu.set(HeaderMenu::Metrics);
                            },
                            class: "p-1 rounded hover:bg-gray-700 cursor-pointer",
                            svg { 
                                class: "max-w-[48px] max-h-[48px]",
                                dangerous_inner_html: ms_top_speed_icon_svg() 
                            }
                        }
                    }
                }
                div {
                    class: "sm:w-auto w-[128px] sm:h-[128px] h-auto place-content-end",
                    button {
                        onclick: move |_| async move {
                            header_menu.set(HeaderMenu::Help);
                        },
                        class: "p-1 rounded hover:bg-gray-700 cursor-pointer",
                        svg { 
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: aws_help_icon_svg() 
                        }
                    }
                    a {
                        href: "https://github.com/biom8er/phymes",
                        target: "_blank",
                        rel: "noopener noreferrer",
                        class: "inline-flex p-1 rounded hover:bg-gray-700 cursor-pointer",
                        svg { 
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: b8_logo_icon_svg() 
                        }
                    }
                }
            }

            // Main content area to the right of the sidebar
            div {
                class: "w-full sm:w-[calc(100%-64px)] h-[calc(100%-64px)] sm:h-full",
                if header_menu.read().as_str() == "Help" {
                    about_text_modal {},
                } else if header_menu.read().as_str() == "Account" {
                    sign_in_view {},
                } else if header_menu.read().as_str() == "Builds" {
                    apps_interface_view {},
                } else if header_menu.read().as_str() == "Apps" {
                    apps_interface_view {},
                } else if header_menu.read().as_str() == "Subjects" {
                    subjects_interface_view {},
                } else if header_menu.read().as_str() == "Messages" {
                    messaging_interface_view {},
                } else if header_menu.read().as_str() == "Attachments" {
                    attachments_interface_view {},
                } else if header_menu.read().as_str() == "Metrics" {
                    metrics_interface_view {},
                }
            }
        }
    }
}

/// Snap horizontal or vertical positions
#[derive(Clone, Copy, PartialEq)]
pub enum SnapPct {
    Pct0,
    Pct20,
    Pct50,
    Pct80,
    Pct100
}

impl SnapPct {
    pub fn to_f32(&self) -> f32 {
        match self {
            Self::Pct0 => 0.0,
            Self::Pct20 => 20.0,
            Self::Pct50 => 50.0,
            Self::Pct80 => 80.0,
            Self::Pct100 => 100.0,
        }
    }
    pub fn decrease(&self) -> Self {
        match self {
            Self::Pct0 => SnapPct::Pct0,
            Self::Pct20 => SnapPct::Pct0,
            Self::Pct50 => SnapPct::Pct20,
            Self::Pct80 => SnapPct::Pct50,
            Self::Pct100 => SnapPct::Pct80,
        }
    }
    pub fn increase(&self) -> Self {
        match self {
            Self::Pct0 => SnapPct::Pct20,
            Self::Pct20 => SnapPct::Pct50,
            Self::Pct50 => SnapPct::Pct80,
            Self::Pct80 => SnapPct::Pct100,
            Self::Pct100 => SnapPct::Pct100,
        }
    }
}

/// Split panel generic container
/// 
/// # Notes
/// * On large screens top = left and bottom = right
/// * It is not possible to dynamically change from horizontal to vertical without JS
#[component]
pub fn split_panel(
    top: Element,
    bottom: Element,
    #[props(default = SnapPct::Pct80)]
    initial_top_pct: SnapPct,
    #[props(default = true)]
    horizontal: bool,
) -> Element {
    let mut top_pct = use_signal(|| initial_top_pct);
    let mut is_dragging = use_signal(|| false);
    let mut start_y = use_signal(|| 0.0);
    let mut start_pct = use_signal(|| top_pct());

    let on_mouse_move = {
        move |evt: MouseEvent| {
            if !is_dragging() {
                return;
            }

            let dxy = if horizontal {
                evt.page_coordinates().y as f32 - start_y()
            } else {
                evt.page_coordinates().x as f32 - start_y()
            };

            // DM: since we either need to call external JS or use a UI-dependent library
            //  to get the container dimensions to calculate the relative position, 
            //  we instead implement a snap behavior that snaps the containers at 20, 50, and 80 percentages
            if dxy > 5.0 {
                let new_pct = top_pct().increase();
                top_pct.set(new_pct);

                // Stop drag
                is_dragging.set(false);
            } else if dxy < -5.0 {
                let new_pct = top_pct().decrease();
                top_pct.set(new_pct);

                // Stop drag
                is_dragging.set(false);
            }
        }
    };

    let on_mouse_up = {
        move |_evt: MouseEvent| {
            if is_dragging() {
                is_dragging.set(false);
            }
        }
    };

    let on_divider_mouse_down = {
        move |evt: MouseEvent| {
            is_dragging.set(true);
            if horizontal {
                start_y.set(evt.page_coordinates().y as f32);
            } else {
                start_y.set(evt.page_coordinates().x as f32);
            }
            start_pct.set(top_pct());
            evt.prevent_default();
        }
    };

    let (div_class, top_bottom_class, middle_class) = if horizontal {
        ("flex flex-col h-full w-full overflow-hidden", "w-full overflow-auto", "w-full h-2 bg-neutral-200 dark:bg-neutral-700 hover:bg-neutral-300 active:bg-neutral-400 cursor-row-resize")
    } else {
        ("flex flex-row h-full w-full overflow-hidden", "h-full overflow-auto", "h-full w-2 bg-neutral-200 dark:bg-neutral-700 hover:bg-neutral-300 active:bg-neutral-400 cursor-col-resize")
    };

    let height_or_width = if horizontal {
        "height"
    } else {
        "width"
    };
    let top_style = format!("{height_or_width}: {}%;", top_pct().to_f32());
    let bottom_style = format!("{height_or_width}: {}%;", 100.0 - top_pct().to_f32());

    rsx! {
        div {
            class: div_class,
            // Attach global listeners via onmousemove/onmouseup on parent
            onmousemove: on_mouse_move,
            onmouseup: on_mouse_up,

            div {
                class: top_bottom_class,
                style: "{top_style}",
                {top}
            }

            div {
                class: middle_class,
                onmousedown: on_divider_mouse_down,
            }

            div {
                class: top_bottom_class,
                style: "{bottom_style}",
                {bottom}
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
    let items = [
        HeaderMenu::Help.as_str(),
        HeaderMenu::Builds.as_str(),
        HeaderMenu::Apps.as_str(),
        HeaderMenu::Messages.as_str(),
        HeaderMenu::Attachments.as_str(),
        HeaderMenu::Subjects.as_str(),
        HeaderMenu::Metrics.as_str()

    ];
    let icons = [
        aws_help_icon_svg(),
        ms_tools_icon_svg(),
        ms_apps_icon_svg(),
        ms_message_icon_svg(),
        ms_attachment_icon_svg(),
        ms_database_icon_svg(),
        ms_top_speed_icon_svg()
    ];
    let descriptions = [
        "(Hopefully 🤞) useful information for using PHYMES 😇. Please create an issue on GitHub https://github.com/biom8er/phymes/issues if you run into problems.",
        "A list of undeployed session plans in the building phase. Each session is like a different app with different functionality and state. A schematic of the session plan with all main components is rendered using mermaid.js. The mermaid.js script is provided in the footer, and can be modified to create new session plans.",
        "A list of session plans available to the account. Each session is like a different app with different functionality and state. Only one session can be activated at a time.",
        "The message history for the active session plan. A chat interface is provided for users to publish messages to the messages subject and to receive subscriptions from the messages subject when the messages subject is updated.",
        "The attachment history for the active session plan. A file upload and download interface is provided for users to publish attachments to the attachments subject and to receive subscriptions from the attachments subjects when the attachments subjects are updated.",
        "A list of subject associated with the active session plan. A table shows the schema of the subject tables along with the number if rows. The subject tables can be extended or replaced by uploading tables in comma deliminated CSV format with headers that match the subject. The subject tables can also be downloaded in comma deliminated CSV format. Note that all of the parameters for describing how processors process streaming messages are subject tables. Extending the subject tables for a processors parameters will update the processors parameters on the next run. Note that the message history is also a subject table. Extending the messages table is the equivalent of human in the loop.",
        "The diagnostics for debugging and optimizing the active session plan. The diagnostic tools include logs, traces, and metrics. Traces track the flow of subject messages through tasks and processors. Events add context to enable building a comprehensive timeline of what happened, when it happened, and why it happened. Metrics focus on aggregating numerical data over time from events to provide an overview of system performance and resource utilization. Metrics are tracked per processor. Baseline metrics for row count, and processor start, stop, and total time in nanoseconds are provided. The baseline metrics are visually represented using mermaid.js gantt charts. Note each row is approximately one token for text generation inference processors. Please submit a feature request issue if additional metrics are of interest."
    ];
    let rows = items.into_iter()
        .zip(icons.into_iter())
        .zip(descriptions.into_iter())
        .map(|((a, b), c)| (a, b, c))
        .collect::<Vec<_>>();

    rsx! {
        div {
            class: "p-2 flex flex-col items-center w-auto h-auto overflow-hidden",
            p { 
                class: "text-lg",
                "Welcome to PHYMES by Biom🤖er" 
            },
            table { 
                class: "table-auto overflow-auto rounded bg-gray-800 text-gray-200",
                caption {  
                    class: "caption-top",
                    "Table 1.1: Menu items available in the sidebar."
                }
                thead {  
                    class: "bg-gray-700",
                    tr {  
                        th { "Item" }
                        th { "Icon" }
                        th { "Description" }
                    }
                }
                tbody { 
                    class: "table-auto text-gray-200",
                    {rows.into_iter().map(|(item, icon, description)| {
                        rsx! {
                            tr { 
                                class: "odd:bg-gray-800 even:bg-gray-900",
                                td { "{item}" }
                                td { svg { 
                                    class: "max-w-[48px] max-h-[48px]",
                                    dangerous_inner_html: icon
                                } }
                                td { "{description}" }
                            }
                        }
                    })}
                }
            }
        }
    }
}

#[component]
pub fn code_editor(initial_text: String) -> Element {
    let mut text = use_signal(|| initial_text);

    // Compute the line numbers for the gutter
    // DM: cannot use `lines` or there is a delay for new lines
    let line_count = text.read().split('\n').count().max(1);

    let on_input = move |evt: FormEvent| {
        text.set(evt.value());
    };

    // DM: The JS is not needed...
//     // Listener to synchronize scrolling between the gutter and code
//     use_effect(move || {
//         let _ = text.read();
//         document::eval(
//             format!(r#"const gutter = document.getElementById('gutter');
// const code = document.getElementById('code');
// code.addEventListener('scroll', () => {{
//     gutter.scrollTop = code.scrollTop;
// }});"#).as_str(),
//         );
//     });

    rsx! {
        div { 
            class: "w-full max-h-[128px] rounded-md shadow-sm py-2 p-2 snap-y overflow-auto grid grid-cols-[3rem_1fr] font-mono text-sm leading-6 snap-start",
            div { 
                id: "gutter",
                class: "h-full text-right flex flex-col whitespace-pre overflow-hidden",
                {(1..=line_count).map(|n| rsx! {
                    div { 
                        class: "px-2 text-gray-500 select-none", 
                        "{n}" 
                    }
                })}
            }

            textarea {
                id: "code",
                value: "{text}",
                oninput: on_input,
                class: "w-full h-full grow bg-gray-800 px-3 resize-none focus:outline-none whitespace-pre overflow-hidden"
            }
        }
    }
}