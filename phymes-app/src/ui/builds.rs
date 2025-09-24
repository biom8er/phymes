use dioxus::prelude::*;

use crate::{
    state::{
        apps::{
            sync_current_active_session_state, sync_current_session_mermaid_state, sync_is_flowchart_shown_state, SyncCurrentActiveSessionState, SyncCurrentSessionMermaidJSState, SyncIsFlowchartShownState, ACTIVE_SESSION_NAME, IS_FLOWCHART_SHOWN, SESSION_ER_DIAGRAM, SESSION_FLOWCHART_DIAGRAM
        }, builds::MERMAID_SESSION_CONTEXT_NAME, messaging::{clear_current_message_state, ClearCurrentMessageState}, sign_in::JWT
    },
    ui::{apps::get_non_duplicated_sorted_subjects, svg_icons::{column_arrow_right_icon_svg, deploy_icon_svg, edit_icon_svg, save_icon_svg, trash_icon_svg}},
};

/// View for the builds drop down menu
#[component]
pub fn builds_dropdown_view() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_active_session_state);
    use_coroutine(clear_current_message_state);
    use_coroutine(sync_is_flowchart_shown_state);
    let sync_current_active_session_state = use_coroutine_handle::<SyncCurrentActiveSessionState>();
    let clear_current_message_state = use_coroutine_handle::<ClearCurrentMessageState>();
    let sync_is_flowchart_shown_state = use_coroutine_handle::<SyncIsFlowchartShownState>();

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());

    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &MERMAID_SESSION_CONTEXT_NAME
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    rsx! {
        div {
            class: "dropdown_form",
            form {
                class: "dropdown_form_input",
                input {
                    r#type: "text",
                    placeholder: "search session",
                    value: "{subject_dropdown}",
                    onclick: move |_| show_subject_dropdown.set(true),
                    onfocusout: move |_| show_subject_dropdown.set(false),
                    oninput: move |evt| subject_dropdown.set(evt.value()),
                    onkeyup: move |_| {
                        subjects_filtered.set(subjects_vec().iter()
                            .filter(|s| !s.contains(subject_dropdown.read().as_str()))
                            .cloned()
                            .collect::<Vec<_>>());
                    }
                },
            },
            button {
                class: "dropdown_form_button",
                onclick: move |_evt| async move {
                    // Reset the dropdown
                    let active_session = subject_dropdown.try_read().unwrap().to_string();
                    subject_dropdown.set(String::new());

                    // Set the active session
                    sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: active_session.clone() });

                    // Reset the current session messaging
                    clear_current_message_state.send(ClearCurrentMessageState {});
                },
                svg { dangerous_inner_html: edit_icon_svg() },
            },
            button { 
                svg { dangerous_inner_html: trash_icon_svg() },
            },
            button { 
                onclick: move |_| async move {
                    let current = IS_FLOWCHART_SHOWN.read().to_owned();
                    sync_is_flowchart_shown_state.send( SyncIsFlowchartShownState { is_shown: !current} );
                },
                svg { dangerous_inner_html: column_arrow_right_icon_svg() },
            },
            button { 
                svg { dangerous_inner_html: deploy_icon_svg() },
            },
        }

        // Dynamic dropdown
        if show_subject_dropdown() {
            div {
                class: "dropdown_list",
                ul {
                    id: "builds_dropdown_list",
                    {subjects_vec().iter().filter(|s| ACTIVE_SESSION_NAME.read().to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
                        let sub = sub.clone();
                        rsx! {
                            li {
                                key: "{i}",
                                div {
                                    onmouseover: move |_evt| subject_dropdown.set(sub.clone()),
                                    p { "{sub}" },
                                }
                            }
                        }
                    })}
                }
            }
        }

        if !ACTIVE_SESSION_NAME().is_empty() {
            div {
                p { "Active session {ACTIVE_SESSION_NAME().to_string()}" },
            }
        }
    }
}

/// Diagram code editor
#[component]
pub fn builds_interface_footer() -> Element {
    use_coroutine(sync_current_session_mermaid_state);
    let diagram_code: Memo<String> = use_memo(move || {
        if IS_FLOWCHART_SHOWN() {
            SESSION_FLOWCHART_DIAGRAM.read().to_string()
        } else {
            SESSION_ER_DIAGRAM.read().to_string()
        }        
    });

    rsx! {
        if !JWT.read().is_empty() && !ACTIVE_SESSION_NAME.read().is_empty() {
            footer {
                class: "resizable_text_input",
                div {
                    class: "text_input",
                    form {
                        id: "diagram_code_form",
                        textarea {
                            value: "{diagram_code.to_string()}",
                            oninput: move |event| async move {
                                let sync_current_session_mermaid_state = use_coroutine_handle::<SyncCurrentSessionMermaidJSState>();
                                let current_session_mermaid_js = if IS_FLOWCHART_SHOWN() {
                                    SyncCurrentSessionMermaidJSState {
                                        flowchart_diagram: Some(event.value()),
                                        er_diagram: None,
                                    }
                                } else {
                                    SyncCurrentSessionMermaidJSState {
                                        flowchart_diagram: None,
                                        er_diagram: Some(event.value()),
                                    }
                                };
                                sync_current_session_mermaid_state.send(current_session_mermaid_js);
                            },
                        }
                    }
                }

                div {
                    class: "submit_button",
                    // This must be outside the form or it will be refreshed on each submit
                    button {
                        onclick: move |_| async move {
                            // TODO: create new session
                        },
                        if !diagram_code().is_empty() {
                            svg { dangerous_inner_html: save_icon_svg() }
                        }
                    }
                }
            }
        }
    }
}
