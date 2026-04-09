// Dioxus imports
use dioxus::prelude::*;

use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_diagnostics::{convert_timestamp_micros_to_str, create_timestamp_micros};
use phymes_event::Publication;
use phymes_message::{
    MessageBuilderTrait, SessionInterfaceMessage, SessionInterfaceMessageBuilder,
    SessionInterfaceMessageBuilderTrait,
};
use phymes_schemas::{AvailableInterfaceSubjects, AvailableSubjectsTrait, DataFormat};
use phymes_server::create_session_name;
use phymes_streams::ChatBuilderTraitExt;

#[cfg(not(feature = "serverless"))]
use super::backend::ADDR_BACKEND;
#[cfg(not(feature = "serverless"))]
use futures::StreamExt;
#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

#[cfg(feature = "serverless")]
use crate::state::RUNTIME_ENV;
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
    let mut messaging_roles = use_signal(Vec::<String>::new);
    let mut messaging_contents = use_signal(Vec::<String>::new);
    let mut messaging_indices = use_signal(Vec::<usize>::new);
    let mut messaging_timestamps = use_signal(Vec::<i64>::new);

    // Update the index in a different scope
    let current_index: Memo<usize> = use_memo(move || {
        if messaging_indices.len() == 0 {
            0
        } else {
            *messaging_indices.last().unwrap()
        }
    });

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || {
        SessionInterfaceMessage::get_builder()
            .with_session_name(&create_session_name(
                EMAIL().as_str(),
                ACTIVE_SESSION_NAME().as_str(),
            ))
            .with_format(&DataFormat::Ipc)
            .with_publisher(&create_session_name(
                EMAIL().as_str(),
                ACTIVE_SESSION_NAME().as_str(),
            ))
            .with_update(&Publication::None)
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
                let mut bytes = Vec::new();
                while let Some(Ok(b)) = stream.next().await {
                    bytes.extend(b);
                }
                match SubjectBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        let combined = table
                            .get_column_as_vec_nonprimitive::<String>("role")
                            .unwrap()
                            .into_iter()
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("content")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_primitive::<i64>("timestamp")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .enumerate()
                            .filter_map(|(i, ((r, c), t))| {
                                if r.is_empty() {
                                    None
                                } else {
                                    let index = current_index() + i + 1;
                                    Some((r, c, t, index))
                                }
                            })
                            .collect::<Vec<_>>();
                        for (r, c, t, index) in combined {
                            messaging_roles.push(r);
                            messaging_contents.push(c);
                            messaging_timestamps.push(t);
                            messaging_indices.push(index);
                        }
                    }
                    Err(err) => {
                        tracing::error!("{err:?}");

                        // initialize the first message
                        update_message_state(messaging_roles,
                            messaging_contents,
                            messaging_indices,
                            messaging_timestamps,
                            "assistant",
                            "Welcome to the Biom8er messaging interface. I am your assistant. Please ask me a question 😊", 
                            create_timestamp_micros());
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

        #[cfg(feature = "serverless")]
        let config = ServerlessConfig {
            route: route.to_string(),
            basic_auth: None,
            bearer_auth: Some(JWT().to_string()),
            data: Some(data_serialized),
            object_store_backend: None,
            object_store_bucket: None,
            object_store_config: None,
        };
        #[cfg(feature = "serverless")]
        let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
        #[cfg(feature = "serverless")]
        match serverless_app(config, &mut serverless).await {
            Ok(response) => {
                let bytes: Vec<Bytes> = response
                    .into_body()
                    .into_data_stream()
                    .try_collect()
                    .await
                    .unwrap();
                let bytes = bytes.into_iter().flatten().collect::<Vec<_>>();
                match SubjectBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        let combined = table
                            .get_column_as_vec_nonprimitive::<String>("role")
                            .unwrap()
                            .into_iter()
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("content")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_primitive::<i64>("timestamp")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .enumerate()
                            .filter_map(|(i, ((r, c), t))| {
                                if r.is_empty() {
                                    None
                                } else {
                                    let index = current_index() + i + 1;
                                    Some((r, c, t, index))
                                }
                            })
                            .collect::<Vec<_>>();
                        for (r, c, t, index) in combined {
                            messaging_roles.push(r);
                            messaging_contents.push(c);
                            messaging_timestamps.push(t);
                            messaging_indices.push(index);
                        }
                    }
                    Err(err) => tracing::error!("{err:?}"),
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
                    div {
                        class: "h-full w-full overflow-auto",
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
                                            class: "p-4 leading-6 max-w-[90%] rounded bg-neutral-800",
                                            dangerous_inner_html: "{content}"
                                            // dangerous_inner_html: "<p>{content}</p>"
                                        }
                                    }
                                }
                            })}
                        }
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

    // Update the index in a different scope
    let current_index: Memo<usize> = use_memo(move || {
        if messaging_indices.len() == 0 {
            0
        } else {
            *messaging_indices.last().unwrap()
        }
    });

    // Check if the last message is assistant pending
    let assistent_pending: Memo<bool> = use_memo(move || {
        if let (Some(role), Some(contents)) = (
            messaging_roles.read().last(),
            messaging_contents.read().last(),
        ) {
            role.as_str() == "assistant" && contents.as_str() == "Preparing response..."
        } else {
            false
        }
    });

    rsx! {
        footer {
            class: "h-full grid grid-rows-[auto_1fr] grid-cols-[auto_1fr_auto] items-center p-2",
            div {
                class: "row-span-1 col-span-1 row-start-1 col-start-1 p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                attach_textfiles_input { except_files: use_signal(|| ".txt,.csv,.tsv,.js,.ts,.py,.java,.c,.cpp,.cs,.rb,.go,.rs,.json,.svg,.html".to_string()), content: prompt }
            }

            form {
                class: "w-full h-full flex row-span-2 col-span-1 row-start-1 col-start-2",
                textarea {
                    placeholder: "Type your message here...",
                    value: "{prompt.to_string()}",
                    oninput: move |event| prompt.set(event.value()),
                    class: "w-full h-full grow p-2 gap-2 rounded bg-neutral-800 text-gray-200 resize-none overflow-auto focus:outline-none",
                }
            }

            div {
                class: "row-span-1 col-span-1 row-start-1 col-start-3",
                // This must be outside the form or it will be refreshed on each submit
                if prompt.read().is_empty() {
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
                        svg {
                            class: "max-w-[48px] max-h-[48px]",
                            dangerous_inner_html: b8_microphone_icon_svg()
                        }
                    }
                } else {
                    button {
                        class: "p-2 hover:bg-neutral-700 rounded bg-neutral-800 cursor-pointer",
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
                            let chat = AvailableInterfaceSubjects::UserMessages.to_subject_builder(None)
                                .append_new_user_query_str(&prompt.read(), "user")
                                .unwrap()
                                .build()
                                .unwrap();
                            let data = SessionInterfaceMessage::get_builder()
                                .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                                .with_format(&DataFormat::Ipc)
                                .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
                                .with_update(&Publication::Extend { subject_name: AvailableInterfaceSubjects::UserMessages.to_string() })
                                .with_stream(false)
                                .with_subject(chat.get_name())
                                .with_message(chat.to_ipc_stream().unwrap())
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
                                    // Remove the last message
                                    if assistent_pending() {
                                        messaging_roles.write().pop();
                                        messaging_contents.write().pop();
                                        messaging_timestamps.write().pop();
                                        messaging_indices.write().pop();
                                    }

                                    // Collect the bytes
                                    let mut stream = stream.bytes_stream();
                                    let mut bytes = Vec::new();
                                    while let Some(Ok(b)) = stream.next().await {
                                        bytes.extend(b);
                                    }

                                    // Collect the batches
                                    match SubjectBuilder::from_ipc_stream_to_record_batches(&bytes) {
                                        Ok(builder) => {
                                            let batches = builder.into_iter()
                                                .filter(|batch| batch.schema() // DM: filtering out UserQuery
                                                    .fields()
                                                    .iter()
                                                    .map(|f| f.name())
                                                    .collect::<Vec<_>>()
                                                    .contains(&&"role".to_string()))
                                                .collect::<Vec<_>>();

                                            // Update the messages
                                            if !batches.is_empty() {
                                                let table = SubjectBuilder::new().with_record_batches(batches).unwrap().with_name("").build().unwrap();
                                                let combined = table.get_column_as_vec_nonprimitive::<String>("role").unwrap().into_iter()
                                                    .zip(table.get_column_as_vec_nonprimitive::<String>("content").unwrap().into_iter())
                                                    .zip(table.get_column_as_vec_primitive::<i64>("timestamp").unwrap().into_iter())
                                                    .enumerate()
                                                    .filter_map(|(i, ((r, c), t))| if r.is_empty() || r != "assistant" {
                                                        None
                                                    } else {
                                                        let index = current_index() + i + 1;
                                                        Some((r, c, t, index))
                                                    }).collect::<Vec<_>>();
                                                if combined.is_empty() {
                                                    update_message_state(messaging_roles,
                                                        messaging_contents,
                                                        messaging_indices,
                                                        messaging_timestamps,
                                                        "assistant",
                                                        "Session returned without a text message response.",
                                                        create_timestamp_micros());
                                                } else {
                                                    for (r, c, t, index) in combined {
                                                        messaging_roles.push(r);
                                                        messaging_contents.push(c);
                                                        messaging_timestamps.push(t);
                                                        messaging_indices.push(index);
                                                    }
                                                }
                                            } else {
                                                update_message_state(messaging_roles,
                                                    messaging_contents,
                                                    messaging_indices,
                                                    messaging_timestamps,
                                                    "assistant",
                                                    "Session returned without a text message response.",
                                                    create_timestamp_micros());
                                            }
                                        }
                                        Err(err) => {
                                            tracing::error!("{err:?}");
                                            update_message_state(messaging_roles,
                                                messaging_contents,
                                                messaging_indices,
                                                messaging_timestamps,
                                                "assistant",
                                                "Session returned without a text message response.",
                                                create_timestamp_micros());
                                        },
                                    }
                                },
                                Err(err) => update_message_content_state(messaging_contents, err.to_string().as_str(), true),
                            }

                            #[cfg(feature = "serverless")]
                            let config = ServerlessConfig {
                                route: route.to_string(),
                                basic_auth: None,
                                bearer_auth: Some(JWT().to_string()),
                                data: Some(data_serialized),
                                object_store_backend: None,
                                object_store_bucket: None,
                                object_store_config: None,
                            };
                            #[cfg(feature = "serverless")]
                            let mut serverless = Serverless::new(None, &RUNTIME_ENV).await.unwrap();
                            #[cfg(feature = "serverless")]
                            match serverless_app(config, &mut serverless).await {
                                Ok(response) => {
                                    let bytes: Vec<Bytes> = response
                                        .into_body()
                                        .into_data_stream()
                                        .try_collect()
                                        .await
                                        .unwrap();
                                    let bytes = bytes.into_iter().flatten().collect::<Vec<_>>();

                                    // Remove the last message
                                    if assistent_pending() {
                                        messaging_roles.write().pop();
                                        messaging_contents.write().pop();
                                        messaging_timestamps.write().pop();
                                        messaging_indices.write().pop();
                                    }

                                    // Collect the batches
                                    match SubjectBuilder::from_ipc_stream_to_record_batches(&bytes) {
                                        Ok(builder) => {
                                            let batches = builder.into_iter()
                                                .filter(|batch| batch.schema() // DM: filtering out UserQuery
                                                    .fields()
                                                    .iter()
                                                    .map(|f| f.name())
                                                    .collect::<Vec<_>>()
                                                    .contains(&&"role".to_string()))
                                                .collect::<Vec<_>>();

                                            // Update the messages
                                            if !batches.is_empty() {
                                                let table = SubjectBuilder::new().with_record_batches(batches).unwrap().with_name("").build().unwrap();
                                                let combined = table.get_column_as_vec_nonprimitive::<String>("role").unwrap().into_iter()
                                                    .zip(table.get_column_as_vec_nonprimitive::<String>("content").unwrap().into_iter())
                                                    .zip(table.get_column_as_vec_primitive::<i64>("timestamp").unwrap().into_iter())
                                                    .enumerate()
                                                    .filter_map(|(i, ((r, c), t))| if r.is_empty() {
                                                        None
                                                    } else {
                                                        let index = current_index() + i + 1;
                                                        Some((r, c, t, index))
                                                    }).collect::<Vec<_>>();
                                                for (r, c, t, index) in combined {
                                                    messaging_roles.push(r);
                                                    messaging_contents.push(c);
                                                    messaging_timestamps.push(t);
                                                    messaging_indices.push(index);
                                                }
                                            }
                                        }
                                        Err(err) => {
                                            tracing::error!("{err:?}");
                                            update_message_state(messaging_roles,
                                                messaging_contents,
                                                messaging_indices,
                                                messaging_timestamps,
                                                "assistant",
                                                "Please try again...",
                                                create_timestamp_micros());
                                        },
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
