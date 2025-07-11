use std::collections::HashSet;

use dioxus::prelude::*;

use super::{
    settings_state::{
        sync_current_active_session_state, SyncCurrentActiveSessionState, ACTIVE_SESSION_NAME,
    },
    sign_in_state::{JWT, SESSION_NAMES},
    svg_icons::search_icon_svg,
    messaging_state::{clear_current_message_state, ClearCurrentMessageState},
};

/// Get a non duplicated list of sorted subject names
pub fn get_non_duplicated_sorted_subjects(subjects: &[&str]) -> Vec<String> {
    let subjects_set = subjects
        .iter()
        .map(|s| s.to_string())
        .collect::<HashSet<_>>();
    let mut subjects_vec = subjects_set.into_iter().collect::<Vec<_>>();
    subjects_vec.sort();
    subjects_vec
}

/// View for the per runtime settings
#[component]
pub fn settings_modal() -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_active_session_state);
    use_coroutine(clear_current_message_state);

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());
    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &SESSION_NAMES
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    #[allow(clippy::redundant_closure)]
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(|| Vec::new());

    rsx! {
        // Check for sign-in
        if JWT.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please sign-in before activating a session." },
            }
        } else if SESSION_NAMES.is_empty(){
            div {
                class: "messaging_list",
                p { "Waiting to retrieve available session plans..." },
            }
        } else  {
            div {
                class: "messaging_list",
                // Active session manager
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
                            let sync_current_active_session_state = use_coroutine_handle::<SyncCurrentActiveSessionState>();
                            sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: active_session.clone() });

                            // Reset the current session messaging
                            let clear_current_message_state = use_coroutine_handle::<ClearCurrentMessageState>();
                            clear_current_message_state.send(ClearCurrentMessageState {});
                        },
                        svg { dangerous_inner_html: search_icon_svg() },
                    },
                }

                // Dynamic dropdown
                if show_subject_dropdown() {
                    div {
                        class: "dropdown_list",
                        ul {
                            id: "sessions_dropdown_list",
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
    }
}
