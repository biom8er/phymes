// Dioxus imports
use dioxus::prelude::*;

use phymes_agents::AvailableInterfaceSubjects;
use phymes_diagnostics::convert_timestamp_micros_to_str;
use serde_json::{self, Map, Value};

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

use phymes_core::{BuildableTrait, BuilderTrait, SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait, DataFormat, TablePublish, MessageBuilderTrait};
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
    state::{ACTIVE_SESSION_NAME, update_attachments_state, extension_and_file_to_data_href, extension_to_icon_svg, filename_and_extension_to_download, EMAIL, JWT,
        svg_icons::{aws_assistant_icon_svg, aws_user_icon_svg, fa_trash_icon_svg, ms_arrow_download_icon_svg}},
    ui::{attach_files_input, clear_upload_files_button, upload_files_button}
};

/// View for attachments between the user and AI assistant
#[component]
pub fn attachments_interface_view() -> Element {
    // Global signals
    let attachments_roles = use_signal(Vec::<String>::new);
    let mut attachments_contents = use_signal(Vec::<Option<Vec<u8>>>::new);
    let attachments_indices = use_signal(Vec::<usize>::new);
    let attachments_timestamps = use_signal(Vec::<i64>::new);
    let attachments_filenames = use_signal(Vec::<String>::new);
    let attachments_extensions = use_signal(Vec::<String>::new);

    // `get_session_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_session_state: Memo<SessionInterfaceMessageBuilder> = use_memo(move || SessionInterfaceMessage::get_builder()
        .with_session_name(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_format(&DataFormat::Bytes)
        .with_publisher(&create_session_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str()))
        .with_update(&TablePublish::None)
        .with_stream(false)
    );

    // Get the last 25 attachments (without the actual blob content) for the attachments view
    let got_attachments = use_memo(move || !attachments_roles().is_empty());
    let _ = use_resource(move || async move {
        // Prevent re-fetching attachments if we already have some
        if got_attachments() {
            return;
        }

        let data = get_session_state()
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
                    let json_rows: Vec<Map<String, Value>> =
                        serde_json::from_slice(&bytes).unwrap_or_else(|err| {
                            tracing::error!(
                                "There was a error parsing SyncCurrentAttachmentsState {err}."
                            );
                            Vec::new()
                        });
                    for row in json_rows.iter() {
                        let bytes: Vec<u8> = serde_json::from_value(row.get("bytes").unwrap().to_owned()).unwrap();
                        if row.get("metadata").is_some() {
                            update_attachments_state(attachments_roles, attachments_contents, attachments_indices, attachments_timestamps, attachments_filenames, attachments_extensions,
                                row.get("metadata").unwrap().as_str().unwrap(),
                                Some(bytes),
                                // None,
                                row.get("timestamp").unwrap().as_i64().unwrap(),
                                row.get("filename").unwrap().as_str().unwrap(),
                                row.get("extension").unwrap().as_str().unwrap(),
                            );
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
                    for row in json_rows.iter() {
                        if row.get("metadata").is_some() {
                            update_attachments_state(attachments_roles, attachments_contents, attachments_indices, attachments_timestamps, attachments_filenames, attachments_extensions,
                                row.get("metadata").unwrap().as_str().unwrap(),
                                None,
                                row.get("timestamp").unwrap().as_i64().unwrap(),
                                row.get("filename").unwrap().as_str().unwrap(),
                                row.get("extension").unwrap().as_str().unwrap(),
                            );
                        }
                    }
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }
    });

    rsx! {
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
                class: "messaging_list",
                {(0..attachments_roles.len()).map(|i| {
                    let role = attachments_roles.get(i).unwrap();
                    let index = attachments_indices.get(i).unwrap();
                    let timestamp = convert_timestamp_micros_to_str(*attachments_timestamps.get(i).unwrap());
                    let content = attachments_contents.get(i).unwrap();
                    let filename = attachments_filenames.get(i).unwrap();
                    let extension = attachments_extensions.get(i).unwrap();
                    rsx! {
                        li {
                            key: "{index}",
                            class: "assistant", // we borrow the assistant class for styling
                            div {
                                class: "entete",
                                if role.as_str() == "assistant" {
                                    svg { dangerous_inner_html: aws_assistant_icon_svg() }
                                    h2 { "AI Assistant" }
                                } else {
                                    svg { dangerous_inner_html: aws_user_icon_svg() }
                                    h2 { "User" }
                                }
                                h3 { "{timestamp}" }
                                svg { dangerous_inner_html: extension_to_icon_svg(&extension) }
                                if let Some(f) = content.as_ref() {
                                    a {
                                        href: extension_and_file_to_data_href(&extension, f).unwrap(),
                                        download: filename_and_extension_to_download(&filename, &extension),
                                        "{filename_and_extension_to_download(&filename, &extension)}"
                                    },
                                    button {
                                        onclick: move |_| async move {
                                            *attachments_contents.get_mut(i).unwrap() = None;
                                        },
                                        svg { dangerous_inner_html: fa_trash_icon_svg() }
                                    }
                                } else {
                                    h3 { "{filename}.{extension}" },
                                    button {
                                        svg { dangerous_inner_html: ms_arrow_download_icon_svg() }
                                        // TODO: download the attachment
                                    }
                                }                                
                            }
                        }
                    }
                })}
            }
            attachments_interface_footer { extend_input: use_signal(|| true), add_input: use_signal(|| false), except_files: use_signal(||".csv,.pdf,.json".to_string()), active_subject_name: None }
        }
    }
}

#[component]
pub fn attachments_interface_footer(extend_input: Signal<bool>, add_input: Signal<bool>, except_files: Signal<String>, active_subject_name: Option<Signal<String>>) -> Element {
    let files_uploaded = use_signal(Vec::<SessionInterfaceMessage>::new);
    let filenames_uploaded = use_signal(Vec::<String>::new);
    let extensions_uploaded = use_signal(Vec::<String>::new);
    
    let filenames = use_memo(move || {
        let mut filenames_vec = Vec::new();
        for i in 0..files_uploaded.len() {
            let download = filename_and_extension_to_download(&filenames_uploaded.get(i).unwrap(), &extensions_uploaded.get(i).unwrap());
            filenames_vec.push(download);
        }
        filenames_vec.join(", ")
    });

    rsx! {
        footer {
            div {
                class: "attach_button", 
                if extend_input() {              
                    attach_files_input { extend_publish: use_signal(|| true), except_files, active_subject_name, files_uploaded, filenames_uploaded, extensions_uploaded }
                } 
                if add_input() {
                    attach_files_input { extend_publish: use_signal(|| false), except_files, active_subject_name, files_uploaded, filenames_uploaded, extensions_uploaded }
                }
            }

            div {
                class: "text_input",
                form {
                    id: "message_form",
                    textarea {
                        placeholder: "Staged files",
                        value: "{filenames}",
                    }
                }
            }

            div {
                class: "submit_button",
                if !files_uploaded.read().is_empty() {
                    upload_files_button {files_uploaded, filenames_uploaded, extensions_uploaded}
                    clear_upload_files_button {files_uploaded, filenames_uploaded, extensions_uploaded}
                }
            }
        }
    }
}