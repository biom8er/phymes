// Dioxus imports
use dioxus::prelude::*;

// General imports
use futures::StreamExt;
use phymes_agents::session_plans::available_interface_subjects::AvailableInterfaceSubjects;
use serde_json::{self, Map, Value};

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

// Phymes imports
use phymes_core::{
    schemas::{available_subjects::{convert_timestamp_micros_to_str, create_timestamp_str, AvailableSubjectsTrait}, chat::ChatBuilderTraitExt}, session::{common_traits::{BuildableTrait, BuilderTrait, MappableTrait}, message::{SessionInterfaceMessage, SessionInterfaceMessageBuilderTrait}}, table::{data_format::DataFormat, table_trait::TableTrait, table_publish::TablePublish}, task::message::MessageBuilderTrait
};
use phymes_server::handlers::sign_in::create_session_name;

#[cfg(not(feature = "serverless"))]
use super::backend::ADDR_BACKEND;

#[cfg(feature = "serverless")]
use bytes::Bytes;
#[cfg(feature = "serverless")]
use futures::TryStreamExt;
#[cfg(feature = "serverless")]
use phymes_server::server::{
    serverless_app::{serverless_app, Serverless},
    serverless_config::ServerlessConfig,
};

// mod imports
use crate::{
    state::{
        apps::ACTIVE_SESSION_NAME, attachments::{
            clear_current_attachments_state, sync_current_attachments_state, ClearCurrentAttachmentsState, SyncCurrentAttachmentsState, ATTACHMENTS_CONTENT, ATTACHMENTS_EXTENSION, ATTACHMENTS_FILENAME, ATTACHMENTS_INDEX, ATTACHMENTS_ROLE, ATTACHMENTS_TIMESTAMP
        }, sign_in::{EMAIL, JWT}
    },
    ui::svg_icons::{aws_assistant_icon_svg, aws_table_icon_svg, aws_user_icon_svg, b8_microphone_icon_svg, b8_send_icon_svg, ms_arrow_download_icon_svg, ms_attachment_icon_svg, ms_code_icon_svg, ms_document_icon_svg, ms_image_icon_svg, ms_video_icon_svg},
};

pub fn extension_to_icon_svg(extension: &str) -> String {
    match extension.to_lowercase().as_str() {
        "pdf" => ms_document_icon_svg(),
        "mp3" | "wav" | "flac" | "aac" => b8_microphone_icon_svg(),
        "mp4" | "mov" | "avi" | "mkv" => ms_video_icon_svg(),
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" => ms_image_icon_svg(),
        "js" | "ts" | "py" | "java" | "c" | "cpp" | "cs" | "rb" | "go" | "rs" | "json" => ms_code_icon_svg(),
        "csv" | "tsv" => aws_table_icon_svg(),
        _ => ms_attachment_icon_svg(), // default icon for unknown file types
    }
}

/// View for attachments between the user and AI assistant
#[component]
pub fn attachments_interface_view() -> Element {
    // intialize state and coroutines
    use_coroutine(sync_current_attachments_state);
    use_coroutine(clear_current_attachments_state);
    let sync_current_attachments_state = use_coroutine_handle::<SyncCurrentAttachmentsState>();
    let clear_current_attachments_state = use_coroutine_handle::<ClearCurrentAttachmentsState>();

    // Get the last 25 attachments (without the actual blob content) for the attachments view
    let _ = use_resource(move || async move {
        clear_current_attachments_state.send(ClearCurrentAttachmentsState {});
        let data = SessionInterfaceMessage::get_builder()
            .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
            .with_format(&DataFormat::Bytes)
            .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
            .with_update(&TablePublish::None)
            .with_stream(false)
            .with_subject(AvailableInterfaceSubjects::AggregatedAttachments.to_string().as_str())
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
                            tracing::error!(
                                "There was a error parsing SyncCurrentAttachmentsState {err}."
                            );
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        if row.get("metadata").is_some() {
                            sync_current_attachments_state.send(SyncCurrentAttachmentsState {
                                role: row.get("metadata").unwrap().as_str().unwrap().to_string(),
                                content: "".to_string(),
                                timestamp: convert_timestamp_micros_to_str(
                                    row.get("timestamp").unwrap().as_i64().unwrap(),
                                ),
                                filename: row.get("filename").unwrap().as_str().unwrap().to_string(),
                                extension: row.get("extension").unwrap().as_str().unwrap().to_string(),
                            });
                        }
                    }
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }

        #[cfg(feature = "serverless")]
        let config = ServerlessConfig {
            route: route.to_string(),
            basic_auth: None,
            bearer_auth: Some(JWT().to_string()),
            data: Some(data_serialized),
        };
        #[cfg(feature = "serverless")]
        let mut serverless = Serverless::new();
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
                    for row in json_rows.iter() {
                        if row.get("metadata").is_some() {
                            sync_current_attachments_state.send(SyncCurrentAttachmentsState {
                                role: row.get("metadata").unwrap().as_str().unwrap().to_string(),
                                content: "".to_string(),
                                timestamp: convert_timestamp_micros_to_str(
                                    row.get("timestamp").unwrap().as_i64().unwrap(),
                                ),
                                filename: row.get("filename").unwrap().as_str().unwrap().to_string(),
                                extension: row.get("extension").unwrap().as_str().unwrap().to_string(),
                            });
                        }
                    }
                }
            }
            Err(_err) => (),
        }
    });

    // render the chat messages
    rsx! {
        // Check for sign-in
        if JWT.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please sign-in before attachments." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please activate a session before attachments." },
            }
        } else {
            ul {
                id: "attachments",
                class: "messaging_list",
                {(0..ATTACHMENTS_ROLE.len()).map(|i| {
                    let role = ATTACHMENTS_ROLE.get(i).unwrap().to_string();
                    let index = ATTACHMENTS_INDEX.get(i).unwrap();
                    let timestamp = ATTACHMENTS_TIMESTAMP.get(i).unwrap().to_string();
                    let content = ATTACHMENTS_CONTENT.get(i).unwrap().to_string();
                    let filename = ATTACHMENTS_FILENAME.get(i).unwrap().to_string();
                    let extension = ATTACHMENTS_EXTENSION.get(i).unwrap().to_string();
                    rsx! {
                        li {
                            key: "{index}",
                            class: "assistant", // we borrow the assistant class for styling
                            div {
                                class: "entete",
                                if role == *"assistant" {
                                    svg { dangerous_inner_html: aws_assistant_icon_svg() }
                                    h2 { "AI Assistant" }
                                } else {
                                    svg { dangerous_inner_html: aws_user_icon_svg() }
                                    h2 { "User" }
                                }
                                h3 { "{timestamp}" }
                                svg { dangerous_inner_html: extension_to_icon_svg(extension) }
                                if content.is_empty() {
                                    h3 { "{filename}.{extension}" },
                                    button {
                                        id: "submit_files",
                                        svg { dangerous_inner_html: ms_arrow_download_icon_svg() }
                                        // TODO: add support for downloading the file
                                        // TODO: check if there is content and replace with a link to download the content
                                    }
                                } else {
                                    // TODO: update when we have support for downloading the file
                                    // a {
                                    //     href: f.href.to_owned(),
                                    //     download: f.download.to_owned(),
                                    //     "{f.download.as_str()}"
                                    // },
                                }
                                
                            }
                        }
                    }
                })}
            }
        }
    }
}

/// View for attachments between the user and AI assistant
#[component]
pub fn attachments_interface_footer() -> Element {
    // intialize state and coroutines
    use_coroutine(sync_current_attachments_state);
    let sync_current_attachments_state = use_coroutine_handle::<SyncCurrentAttachmentsState>();

    #[allow(clippy::redundant_closure)]
    let mut prompt = use_signal(|| String::new());

    // render the chat messages
    rsx! {
        // Check for sign-in
        if !JWT.read().is_empty() && !ACTIVE_SESSION_NAME.read().is_empty() {
            footer {
                div {
                    class: "attach_button",
                    // This must be outside the form or it will be refreshed on each submit
                    button {
                        onclick: move |_| async move {
                            // TODO: add support for adding attachments through the attachments interface
                        },
                        svg { dangerous_inner_html: ms_attachment_icon_svg() }
                    }
                }

                div {
                    class: "files",
                    p { "Files to upload" },
                    ul {
                        id: "uploaded_subject_files",
                        class: "file_list",
                        {file_names.iter().enumerate().map(|(i, f)| {
                            rsx! {
                                li {
                                    key: "{i}",
                                    div {
                                        class: "files",                                        
                                        svg { dangerous_inner_html: extension_to_icon_svg(extension) }
                                        h3 { "{f}" },
                                        // div { class: "loader" },
                                    }
                                }
                            }
                        })}
                    }
                }

                div {
                    class: "submit_button",
                    // This must be outside the form or it will be refreshed on each submit
                    if prompt.read().is_empty() {
                        button {
                            svg { dangerous_inner_html: b8_microphone_icon_svg() }
                        }
                    } else {
                        button {
                            onclick: move |_| async move {
                                // signed in and ready to chat
                                sync_current_attachments_state.send(SyncCurrentAttachmentsState {
                                    role: "user".to_string(),
                                    content: prompt.to_string(),
                                    timestamp: create_timestamp_str()
                                });

                                // let the user know that the response is being prepared
                                sync_current_attachments_state.send(SyncCurrentAttachmentsState {
                                    role: "assistant".to_string(),
                                    content: "Preparing response...".to_string(),
                                    timestamp: create_timestamp_str()
                                });

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
                                    .with_update(&TablePublish::Extend { table_name: AvailableInterfaceSubjects::UserMessages.to_string() })
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
                                        sync_attachments_content.send(SyncCurrentMessageContentState {content: "".to_string(), replace_last: true});
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
                                                if row.get("role").unwrap().as_str().unwrap() == "assistant" {
                                                    sync_attachments_content.send(SyncCurrentMessageContentState {
                                                        content: row.get("content").unwrap().as_str().unwrap().to_string(),
                                                        replace_last: false
                                                    });
                                                }
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        sync_attachments_content.send(SyncCurrentMessageContentState {content: format!("{e:?}"), replace_last: true});
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
                                let mut serverless = Serverless::new();
                                #[cfg(feature = "serverless")]
                                match serverless_app(config, &mut serverless).await {
                                    Ok(response) => {
                                        sync_attachments_content.send(SyncCurrentMessageContentState {content: "".to_string(), replace_last: true});
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
                                                sync_attachments_content.send(SyncCurrentMessageContentState {
                                                    content: row.get("content").unwrap().as_str().unwrap().to_string(),
                                                    replace_last: false
                                                });
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        sync_attachments_content.send(SyncCurrentMessageContentState {content: format!("Error: {e:?}"), replace_last: true});
                                    }
                                }
                            },
                            svg { dangerous_inner_html: b8_send_icon_svg() }
                        }
                    }
                }
            }
        }
    }
}
