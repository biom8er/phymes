// Dioxus imports
use dioxus::prelude::*;

// General imports
use futures::StreamExt;
use reqwest::{self, header::CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{self, Map, Value};

// mod imports
use super::{
    backend::{create_session_name, ADDR_BACKEND, GetSessionState},
    messaging_state::{
        clear_current_message_state, sync_current_message_content_state, sync_current_message_state, 
        ClearCurrentMessageState, SyncCurrentMessageContentState, SyncCurrentMessageState,
        CONTENT, INDEX, ROLE, TIMESTAMP},
    sign_in_state::{EMAIL, JWT},
    svg_icons::{assistant_icon_svg, send_icon_svg, user_icon_svg},
    settings_state::ACTIVE_SESSION_NAME,
};

#[derive(Serialize, Deserialize, Debug)]
struct DioxusMessage {
    content: String,
    session_name: String,
    subject_name: String,
}

/// View for messaging between the user and AI assistant
#[component]
pub fn messaging_interface_view() -> Element {
    // intialize state and coroutines
    use_coroutine(sync_current_message_state);
    use_coroutine(sync_current_message_content_state);
    use_coroutine(clear_current_message_state);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<GetSessionState> = use_memo(move || GetSessionState {
        session_name: create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()),
        subject_name: "".to_string(),
    });

    // Get the last 25 messages for the messages view
    let clear_current_message_state = use_coroutine_handle::<ClearCurrentMessageState>();
    let sync_current_message_state = use_coroutine_handle::<SyncCurrentMessageState>();
    let _ = use_resource(move || async move {
        let data_serialized = serde_json::to_string(&get_session_state()).unwrap();
        clear_current_message_state.send(ClearCurrentMessageState {});
        let addr = format!("{ADDR_BACKEND}/app/v1/subjects_info");
        match reqwest::Client::new()
            .post(addr)
            .bearer_auth(JWT().to_string())
            .header(CONTENT_TYPE, "application/json")
            .body(data_serialized)
            .send()
            .await
        {
            Ok(stream) => {
                let mut stream = stream.bytes_stream();
                while let Some(Ok(bytes)) = stream.next().await {
                    let json_str = String::from_utf8_lossy(bytes.as_ref()).into_owned();
                    let json_rows: Vec<Map<String, Value>> =
                        serde_json::from_str(json_str.as_str()).unwrap_or_else(|_err| {
                            // DM: find a better way to give feedback to the user
                            // content.write().push_str(format!("There was a error parsing SyncCurrentSubjectInfoState {err}.").as_str());
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        sync_current_message_state.send(SyncCurrentMessageState {
                            role: row
                                .get("role")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            content: row
                                .get("content")
                                .unwrap()
                                .as_str()
                                .unwrap()
                                .to_string(),
                            // DM: missing index and timestamp
                        });
                    }
                }
            }
            Err(_err) => (), //content.write().push_str(format!("There was a error getting subjects info {err}.").as_str()),
        }
    });

    // initialize the first message (if the are no messages for the session)
    let num_messages = ROLE.len();
    if num_messages == 0 {
        let sync_message = use_coroutine_handle::<SyncCurrentMessageState>();
        sync_message.send(SyncCurrentMessageState {
            role: "assistant".to_string(), 
            content: "Welcome to the Biom8er messaging interface. I am your assistant. Please ask any me a question 😊".to_string()
        });
    }

    // render the chat messages
    rsx! {
        // Check for sign-in
        if JWT.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please sign-in before messaging." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please activate a session before messaging." },
            }
        } else {
            ul {
                id: "messaging",
                class: "messaging_list",
                {(0..num_messages).map(|i| {
                    let role = ROLE.get(i).unwrap().to_string();
                    let index = INDEX.get(i).unwrap();
                    let timestamp = TIMESTAMP.get(i).unwrap().to_string();
                    let content = CONTENT.get(i).unwrap().to_string();
                    rsx! {
                        li {
                            key: "{index}",
                            class: "{role}", // either assistant or user
                            if role == *"assistant" {
                                div {
                                    class: "entete",
                                    svg { dangerous_inner_html: assistant_icon_svg() }
                                    h2 { "AI Assistant" }
                                    h3 { "{timestamp}" }
                                }
                            } else {
                                // TODO: change the color according to sign-in
                                // not signed-in: Red
                                // signed-in: White
                                div {
                                    class: "entete",
                                    h3 { "{timestamp}" }
                                    h2 { "User" }
                                    svg { dangerous_inner_html: user_icon_svg() }
                                }
                            }
                            div {
                                class: "message",
                                dangerous_inner_html: "<p>{content}</p>"
                            }
                        }
                    }
                })}
            }
        }
    }
}

/// View for messaging between the user and AI assistant
#[component]
pub fn messaging_interface_footer() -> Element {
    // intialize state and coroutines
    use_coroutine(sync_current_message_state);
    use_coroutine(sync_current_message_content_state);

    #[allow(clippy::redundant_closure)]
    let mut prompt = use_signal(|| String::new());

    // render the chat messages
    rsx! {
        // Check for sign-in
        if !JWT.read().is_empty() && !ACTIVE_SESSION_NAME.read().is_empty() {
            footer {
                div {
                    class: "text_input",
                    form {
                        id: "message_form",
                        textarea {
                            placeholder: "Type your message here...",
                            // Text input
                            value: "{prompt.to_string()}",
                            oninput: move |event| prompt.set(event.value()),
                        }
                    }
                }

                div {
                    class: "submit_button",
                    // This must be outside the form or it will be refreshed on each submit
                    button {
                        onclick: move |_| async move {
                            let sync_message = use_coroutine_handle::<SyncCurrentMessageState>();
                            let sync_message_content = use_coroutine_handle::<SyncCurrentMessageContentState>();
                            // signed in and ready to chat
                            sync_message.send(SyncCurrentMessageState {role: "user".to_string(), content: prompt.to_string()});

                            // create the message
                            let data = DioxusMessage {
                                content: prompt.to_string(),
                                session_name: create_session_name(EMAIL.read().as_str(), ACTIVE_SESSION_NAME.read().as_str()),
                                subject_name: "messages".to_string(),
                            };
                            prompt.write().clear();
                            let data_serialized = serde_json::to_string(&data).unwrap();
                            let addr = format!("{ADDR_BACKEND}/app/v1/chat");
                            sync_message.send(SyncCurrentMessageState {role: "assistant".to_string(), content: "Preparing response...".to_string()});
                            match reqwest::Client::new()
                                .post(addr)
                                .bearer_auth(JWT.to_string())
                                .header(CONTENT_TYPE, "application/json")
                                .body(data_serialized)
                                .send()
                                .await {
                                Ok(stream) => {
                                    sync_message_content.send(SyncCurrentMessageContentState {content: "".to_string(), replace_last: true});
                                    let mut stream = stream.bytes_stream();
                                    while let Some(Ok(bytes)) = stream.next().await {
                                        let json_str = String::from_utf8_lossy(bytes.as_ref()).into_owned();
                                        let json_rows: Vec<Map<String, Value>> = serde_json::from_str(json_str.trim_end_matches(char::from(0)))
                                            .unwrap_or_else(|e| {
                                                let mut m = Map::new();
                                                m.insert("content".to_string(), format!("Error: {e:?}").into());
                                                vec![m]
                                            });
                                        for row in json_rows.iter() {
                                            sync_message_content.send(SyncCurrentMessageContentState {
                                                content: row.get("content").unwrap().as_str().unwrap().to_string(),
                                                replace_last: false
                                            });
                                        }
                                    }
                                },
                                Err(e) => {
                                    sync_message_content.send(SyncCurrentMessageContentState {content: format!("Error: {e:?}"), replace_last: true});
                                }
                            }
                        },
                        svg { dangerous_inner_html: send_icon_svg() }
                    }
                }
            }
        }
    }
}