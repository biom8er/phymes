// Dioxus imports
use dioxus::prelude::*;

// General imports
use phymes_agents::AvailableInterfaceSubjects;
use phymes_diagnostics::{convert_timestamp_micros_to_str, create_timestamp_micros};
use serde_json::{self, Map, Value};

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

// Phymes imports
use phymes_core::{
    AvailableSubjectsTrait, BuildableTrait, BuilderTrait, ChatBuilderTraitExt, DataFormat,
    MappableTrait, MessageBuilderTrait, SessionInterfaceMessage, SessionInterfaceMessageBuilder,
    SessionInterfaceMessageBuilderTrait, TablePublication, TableTrait,
};
use phymes_server::create_session_name;

#[cfg(not(feature = "serverless"))]
use super::backend::ADDR_BACKEND;

#[cfg(not(feature = "serverless"))]
use futures::StreamExt;

#[cfg(feature = "serverless")]
use bytes::Bytes;
#[cfg(feature = "serverless")]
use futures::TryStreamExt;
#[cfg(feature = "serverless")]
use phymes_server::{serverless_app, Serverless, ServerlessConfig};

// mod imports
use crate::{
    state::{
        svg_icons::{
            aws_assistant_icon_svg, aws_user_icon_svg, b8_microphone_icon_svg, b8_send_icon_svg,
        },
        update_message_content_state, update_message_state, ACTIVE_SESSION_NAME, EMAIL, JWT,
    },
    ui::{attach_textfiles_input, main_window::split_panel},
};

/// View for messaging between the user and AI assistant
#[component]
pub fn messaging_interface_view() -> Element {
    // Global signals
    let messaging_roles = use_signal(Vec::<String>::new);
    let messaging_contents = use_signal(Vec::<String>::new);
    let messaging_indices = use_signal(Vec::<usize>::new);
    let messaging_timestamps = use_signal(Vec::<i64>::new);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || {
        SessionInterfaceMessage::get_builder()
            .with_session_name(&create_session_name(
                EMAIL().as_str(),
                ACTIVE_SESSION_NAME().as_str(),
            ))
            .with_format(&DataFormat::Bytes)
            .with_publisher(&create_session_name(
                EMAIL().as_str(),
                ACTIVE_SESSION_NAME().as_str(),
            ))
            .with_update(&TablePublication::None)
            .with_stream(false)
    });

    // Get the last 25 messages for the messages view
    let got_messages = use_memo(move || !messaging_roles().is_empty());
    use_resource(move || async move {
        // Prevent re-fetching messages if we already have some
        if got_messages() {
            return;
        }

        let data = get_session_state()
            .with_subject(
                AvailableInterfaceSubjects::AggregatedMessages
                    .to_string()
                    .as_str(),
            )
            .make_name()
            .unwrap()
            .build()
            .unwrap();
        let data_serialized = serde_json::to_string(&data).unwrap();
        let route = "/app/v1/get_state";

        #[cfg(not(feature = "serverless"))]
        let addr = format!("{ADDR_BACKEND}{route}");
        #[cfg(not(feature = "serverless"))]
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
                        serde_json::from_str(json_str.as_str()).unwrap_or_else(|err| {
                            tracing::error!("There was a error parsing messages {err}.");
                            Vec::new()
                        });
                    if json_rows.is_empty() {
                        // initialize the first message (if the are no messages for the session)

                        use phymes_diagnostics::create_timestamp_micros;
                        update_message_state(messaging_roles,
                            messaging_contents,
                            messaging_indices,
                            messaging_timestamps,
                            "assistant",
                            "Welcome to the Biom8er messaging interface. I am your assistant. Please ask any me a question 😊", 
                            create_timestamp_micros());
                    } else {
                        // append the messages to the state
                        for row in json_rows.iter() {
                            if row.get("role").is_some() {
                                update_message_state(
                                    messaging_roles,
                                    messaging_contents,
                                    messaging_indices,
                                    messaging_timestamps,
                                    row.get("role").unwrap().as_str().unwrap(),
                                    row.get("content").unwrap().as_str().unwrap(),
                                    row.get("timestamp").unwrap().as_i64().unwrap(),
                                );
                            }
                        }
                    }
                }
            }
            Err(err) => {
                use phymes_diagnostics::create_timestamp_micros;

                tracing::error!("{err:?}");

                // initialize the first message
                update_message_state(
                    messaging_roles,
                    messaging_contents,
                    messaging_indices,
                    messaging_timestamps,
                    "assistant",
                    "Messaging is not enabled for this app 😞.",
                    create_timestamp_micros(),
                );
            }
        }

        #[cfg(feature = "serverless")]
        let config = ServerlessConfig {
            route: route.to_string(),
            basic_auth: None,
            bearer_auth: Some(JWT().to_string()),
            data: Some(data_serialized),
        };
        #[cfg(feature = "serverless")]
        let mut serverless = Serverless::new(None);
        #[cfg(feature = "serverless")]
        match serverless_app(config, &mut serverless).await {
            Ok(response) => {
                let bytes: Vec<Bytes> = response
                    .into_body()
                    .into_data_stream()
                    .try_collect()
                    .await
                    .unwrap();
                for byte in bytes.iter() {
                    let json_rows: Vec<Map<String, Value>> =
                        serde_json::from_slice(byte).unwrap_or_else(|_err| Vec::new());
                    if json_rows.is_empty() {
                        // initialize the first message (if the are no messages for the session)
                        update_message_state(messaging_roles,
                            messaging_contents,
                            messaging_indices,
                            messaging_timestamps,
                            "assistant", 
                            "Welcome to the Biom8er messaging interface. I am your assistant. Please ask any me a question 😊", 
                            create_timestamp_micros());
                    } else {
                        // append the messages to the state
                        for row in json_rows.iter() {
                            if row.get("role").is_some() {
                                update_message_state(
                                    messaging_roles,
                                    messaging_contents,
                                    messaging_indices,
                                    messaging_timestamps,
                                    row.get("role").unwrap().as_str().unwrap(),
                                    row.get("content").unwrap().as_str().unwrap(),
                                    row.get("timestamp").unwrap().as_i64().unwrap(),
                                );
                            }
                        }
                    }
                }
            }
            Err(err) => {
                tracing::error!("{err:?}");

                // initialize the first message
                update_message_state(
                    messaging_roles,
                    messaging_contents,
                    messaging_indices,
                    messaging_timestamps,
                    "assistant",
                    "Messaging is not enabled for this app 😞.",
                    create_timestamp_micros(),
                );
            }
        }
    });

    // render the chat messages
    rsx! {
        // Check for sign-in
        if JWT.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please sign-in before messaging." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please activate a session before messaging." },
            }
        } else {
            split_panel {
                top: rsx! {
                    ul {
                        class: "p-2 flex flex-col list-none",
                        {(0..messaging_roles().len()).map(|i| {
                            let role = messaging_roles.get(i).unwrap();
                            let index = messaging_indices.get(i).unwrap();
                            let timestamp = convert_timestamp_micros_to_str(*messaging_timestamps.get(i).unwrap());
                            let content = messaging_contents.get(i).unwrap();
                            let li_style = if role.as_str() == "assistant" {
                                "flex flex-col flex-content-start gap-1 my-2"
                            } else {
                                "flex flex-col flex-content-end items-end gap-1 my-2"
                            };
                            rsx! {
                                li {
                                    key: "{index}",
                                    class: li_style,
                                    if role.as_str() == "assistant" {
                                        div {
                                            class: "flex items-center gap-2",
                                            svg {
                                                class: "max-w-[48px] max-h-[48px]",
                                                dangerous_inner_html: aws_assistant_icon_svg()
                                            }
                                            h2 {
                                                class: "font-bold",
                                                "AI Assistant"
                                            }
                                            h3 { "{timestamp}" }
                                        }
                                    } else {
                                        div {
                                            class: "flex items-center gap-2",
                                            h3 { "{timestamp}" }
                                            h2 {
                                                class: "font-bold",
                                                "User"
                                            }
                                            svg {
                                                class: "max-w-[48px] max-h-[48px]",
                                                dangerous_inner_html: aws_user_icon_svg()
                                            }
                                        }
                                    }
                                    div {
                                        class: "p-4 leading-6 max-w-[90%] rounded bg-gray-800",
                                        dangerous_inner_html: "{content}"
                                        // dangerous_inner_html: "<p>{content}</p>"
                                    }
                                }
                            }
                        })}
                    }
                },
                bottom: rsx! {
                    messaging_interface_footer { messaging_roles, messaging_contents, messaging_indices, messaging_timestamps }
                }
            }
        }
    }
}

#[component]
pub fn messaging_interface_footer(
    mut messaging_roles: Signal<Vec<String>>,
    mut messaging_contents: Signal<Vec<String>>,
    mut messaging_indices: Signal<Vec<usize>>,
    mut messaging_timestamps: Signal<Vec<i64>>,
) -> Element {
    let mut prompt = use_signal(String::new);

    rsx! {
        footer {
            class: "h-full grid grid-rows-[64px_1fr] grid-cols-[64px_1fr_64px] items-center p-2 gap-2",
            div {
                class: "row-span-1 col-span-1 row-start-1 col-start-1 p-1 hover:bg-gray-700 rounded bg-gray-800 cursor-pointer",
                attach_textfiles_input { except_files: use_signal(|| ".txt,.csv,.tsv,.js,.ts,.py,.java,.c,.cpp,.cs,.rb,.go,.rs,.json,.svg,.html".to_string()), content: prompt }
            }

            form {
                class: "w-full h-full flex row-span-2 col-span-1 row-start-1 col-start-2",
                textarea {
                    placeholder: "Type your message here...",
                    value: "{prompt.to_string()}",
                    oninput: move |event| prompt.set(event.value()),
                    class: "w-full h-full grow p-2 gap-2 rounded bg-gray-800 text-gray-200 resize-none overflow-auto focus:outline-none",
                }
            }

            div {
                class: "row-span-1 col-span-1 row-start-1 col-start-3",
                // This must be outside the form or it will be refreshed on each submit
                if prompt.read().is_empty() {
                    button {
                        class: "p-1 hover:bg-gray-700 rounded bg-gray-800 cursor-pointer",
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: b8_microphone_icon_svg()
                        }
                    }
                } else {
                    button {
                        class: "p-1 hover:bg-gray-700 rounded bg-gray-800 cursor-pointer",
                        onclick: move |_| async move {
                            // signed in and ready to chat
                            update_message_state(messaging_roles,
                                messaging_contents,
                                messaging_indices,
                                messaging_timestamps,
                                "user",
                                &prompt(),
                                create_timestamp_micros());

                            // let the user know that the response is being prepared
                            update_message_state(messaging_roles,
                                messaging_contents,
                                messaging_indices,
                                messaging_timestamps,
                                "assistant",
                                "Preparing response...",
                                create_timestamp_micros());

                            // create the message
                            let chat = AvailableInterfaceSubjects::UserMessages.to_table_builder(None)
                                .append_new_user_query_str(&prompt.read(), "user")
                                .unwrap()
                                .build()
                                .unwrap();
                            let data = SessionInterfaceMessage::get_builder()
                                .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                                .with_format(&DataFormat::Bytes)
                                .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                                .with_update(&TablePublication::Extend { table_name: AvailableInterfaceSubjects::UserMessages.to_string() })
                                .with_stream(false)
                                .with_subject(chat.get_name())
                                .with_message(chat.to_bytes().unwrap().to_vec())
                                .make_name()
                                .unwrap()
                                .build()
                                .unwrap();
                            prompt.write().clear();
                            let data_serialized = serde_json::to_string(&data).unwrap();
                            let route = "/app/v1/chat";

                            #[cfg(not(feature = "serverless"))]
                            let addr = format!("{ADDR_BACKEND}{route}");
                            #[cfg(not(feature = "serverless"))]
                            match reqwest::Client::new()
                                .post(addr)
                                .bearer_auth(JWT.to_string())
                                .header(CONTENT_TYPE, "application/json")
                                .body(data_serialized)
                                .send()
                                .await {
                                Ok(stream) => {
                                    update_message_content_state(messaging_contents, "", true);
                                    let mut stream = stream.bytes_stream();
                                    while let Some(Ok(bytes)) = stream.next().await {
                                        let json_str = String::from_utf8_lossy(bytes.as_ref()).into_owned();
                                        let json_rows: Vec<Map<String, Value>> = serde_json::from_str(json_str.trim_end_matches(char::from(0)))
                                            .unwrap_or_else(|e| {
                                                let mut m = Map::new();
                                                m.insert("content".to_string(), format!("{e:?} caused by {json_str}").into());
                                                vec![m]
                                            });
                                        for row in json_rows.iter() {
                                            if row.get("role").is_none() {
                                                tracing::error!("Message response does not have key role: {:?}", row);
                                            } else if row.get("role").unwrap().as_str().unwrap() == "assistant" {
                                                update_message_content_state(messaging_contents, row.get("content").unwrap().as_str().unwrap(), false);
                                            }
                                        }
                                    }
                                },
                                Err(e) => update_message_content_state(messaging_contents, e.to_string().as_str(), true),
                            }

                            #[cfg(feature = "serverless")]
                            let config = ServerlessConfig {
                                route: route.to_string(),
                                basic_auth: None,
                                bearer_auth: Some(JWT().to_string()),
                                data: Some(data_serialized),
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None);
                            #[cfg(feature = "serverless")]
                            match serverless_app(config, &mut serverless).await {
                                Ok(response) => {
                                    update_message_content_state(messaging_contents, "", true);
                                    let bytes: Vec<Bytes> = response
                                        .into_body()
                                        .into_data_stream()
                                        .try_collect()
                                        .await
                                        .unwrap();
                                    for byte in bytes.iter() {
                                        let json_rows: Vec<Map<String, Value>> = serde_json::from_slice(byte).unwrap_or_else(|e| {
                                            let mut m = Map::new();
                                            m.insert("content".to_string(), format!("Error: {e:?}").into());
                                            vec![m]
                                        });
                                        for row in json_rows.iter() {
                                            if row.get("role").is_none() {
                                                tracing::error!("Message response does not have key role: {:?}", row);
                                            } else if row.get("role").unwrap().as_str().unwrap() == "assistant" {
                                                update_message_content_state(messaging_contents, row.get("content").unwrap().as_str().unwrap(), false);
                                            }
                                        }
                                    }
                                },
                                Err(e) => update_message_content_state(messaging_contents, e.to_string().as_str(), true),
                            }
                        },
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: b8_send_icon_svg()
                        }
                    }
                }
            }
        }
    }
}
