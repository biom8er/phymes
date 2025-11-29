use dioxus::prelude::*;
use phymes_core::{
    AvailableSubjects, BuildableTrait, BuilderTrait, DataFormat, MessageBuilderTrait,
    SessionInterfaceMessage, SessionInterfaceMessageBuilder, SessionInterfaceMessageBuilderTrait,
    TablePublication, TableTrait, TableBuilder, TableBuilderTrait
};
use phymes_server::create_session_name;

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

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

use crate::{
    state::{
        get_non_duplicated_sorted_subjects, get_subject_num_rows_by_subject_name,
        get_subject_schema_col_type_by_subject_name, svg_icons::ms_search_icon_svg,
        ACTIVE_SESSION_NAME, EMAIL, JWT, SUBJECT_SCHEMA_HEADERS,
    },
    ui::{
        attachments_interface_footer, clear_download_files_button, download_files_button,
        download_files_list, main_window::split_panel,
    },
};

/// View to display the subject tables for the session
/// and to allow for easier upload by the user
#[component]
pub fn subjects_interface_view() -> Element {
    // Global signals
    let mut subject_schema_names = use_signal(Vec::<String>::new);
    let mut subject_schema_columns = use_signal(Vec::<String>::new);
    let mut subject_schema_types = use_signal(Vec::<String>::new);
    let mut subject_names = use_signal(Vec::<String>::new);
    let mut subject_num_rows = use_signal(Vec::<usize>::new);
    let active_subject_name = use_signal(String::new);
    let files_downloaded = use_signal(Vec::<Vec<u8>>::new);
    let filenames_downloaded = use_signal(Vec::<String>::new);
    let extensions_downloaded = use_signal(Vec::<String>::new);

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
            .with_update(&TablePublication::None)
            .with_stream(false)
    });

    // Get the active session schema for the subject view and
    // Get the active session row counts for the subject view
    // DM: these are combined into a single async block to prevent concurrent mutable borrows of the same user state
    use_resource(move || async move {
        // Get the active session schema for the subject view
        subject_schema_names.set(Vec::new());
        subject_schema_columns.set(Vec::new());
        subject_schema_types.set(Vec::new());
        let route = "/app/v1/get_state";
        let data_serialized = serde_json::to_string(
            &get_session_state()
                .with_subject(AvailableSubjects::SessionSubjects.to_string().as_str())
                .make_name()
                .unwrap()
                .build()
                .unwrap(),
        )
        .unwrap();

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
                match TableBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        subject_schema_names.set(table.get_column_as_vec_nonprimitive::<String>("subject_name").unwrap());
                        subject_schema_columns.set(table.get_column_as_vec_nonprimitive::<String>("column_name").unwrap());
                        subject_schema_types.set(table.get_column_as_vec_nonprimitive::<String>("type_name").unwrap());
                    },
                    Err(err) => {
                        tracing::error!("{err:?}");
                    },
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
                match TableBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        subject_schema_names.set(table.get_column_as_vec_nonprimitive::<String>("subject_name").unwrap());
                        subject_schema_columns.set(table.get_column_as_vec_nonprimitive::<String>("column_name").unwrap());
                        subject_schema_types.set(table.get_column_as_vec_nonprimitive::<String>("type_name").unwrap());
                    },
                    Err(err) => {
                        tracing::error!("{err:?}");
                    },
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }

        // Get the active session row counts for the subject view
        subject_names.set(Vec::new());
        subject_num_rows.set(Vec::new());
        let route = "/app/v1/get_state";
        let data_serialized = serde_json::to_string(
            &get_session_state()
                .with_subject(
                    AvailableSubjects::SessionSubjectsNumRows
                        .to_string()
                        .as_str(),
                )
                .make_name()
                .unwrap()
                .build()
                .unwrap(),
        )
        .unwrap();

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
                match TableBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        subject_names.set(table.get_column_as_vec_nonprimitive::<String>("subject_name").unwrap());
                        subject_num_rows.set(table.get_column_as_vec_primitive::<u64>("num_rows").unwrap().into_iter().map(|n| n as usize).collect::<Vec<_>>());
                    },
                    Err(err) => {
                        tracing::error!("{err:?}");
                    },
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
                match TableBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        subject_names.set(table.get_column_as_vec_nonprimitive::<String>("subject_name").unwrap());
                        subject_num_rows.set(table.get_column_as_vec_primitive::<u64>("num_rows").unwrap().into_iter().map(|n| n as usize).collect::<Vec<_>>());
                    },
                    Err(err) => {
                        tracing::error!("{err:?}");
                    },
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }
    });

    rsx! {
        if JWT.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please sign-in before searching subjects." },
            }
        } else if ACTIVE_SESSION_NAME.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Please activate a session before searching subjects." },
            }
        } else if subject_schema_names.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Waiting to retrieve session plan subject schemas..." },
            }
        } else {
            split_panel {
                top: rsx! {
                    div {
                        class: "p-2 overflow-auto flex flex-col items-center",
                        subjects_dropdown_menu { active_subject_name, subject_schema_names, files_downloaded, filenames_downloaded, extensions_downloaded },
                        subjects_schema_table { active_subject_name, subject_schema_names, subject_schema_columns, subject_schema_types, subject_names, subject_num_rows }

                        if !files_downloaded.read().is_empty() {
                            download_files_list {filenames_downloaded, files_downloaded, extensions_downloaded}
                        }
                    }
                },
                bottom: rsx! {
                    attachments_interface_footer { extend_input: use_signal(|| true), add_input: use_signal(|| true), except_files: use_signal(||".csv".to_string()), active_subject_name, subject_names: subject_schema_names }
                },
            }
        }
    }
}

#[component]
pub fn subjects_dropdown_menu(
    mut active_subject_name: Signal<String>,
    subject_schema_names: Signal<Vec<String>>,
    mut files_downloaded: Signal<Vec<Vec<u8>>>,
    mut filenames_downloaded: Signal<Vec<String>>,
    mut extensions_downloaded: Signal<Vec<String>>,
) -> Element {
    let mut show_subject_dropdown = use_signal(|| false);
    #[allow(clippy::redundant_closure)]
    let mut subject_dropdown = use_signal(|| String::new());

    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &subject_schema_names
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
            class: "p-2 rounded bg-neutral-800 grid grid-rows-[auto_1fr] grid-cols-[1fr_auto] md:max-w-3/4",
            form {
                class: "w-full h-full flex row-span-1 col-span-1 row-start-1 col-start-1",
                input {
                    class: "w-full h-full bg-neutral-700",
                    r#type: "text",
                    placeholder: "search subjects",
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

            // Dynamic dropdown of subjects
            if show_subject_dropdown() {
                div {
                    class: "p-2 rounded bg-neutral-800 list-none flex row-span-1 col-span-1 row-start-2 col-start-1",
                    ul {
                        {subjects_vec().iter().filter(|s| active_subject_name.to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
                            let sub = sub.clone();
                            rsx! {
                                li {
                                    class: "hover:bg-neutral-700 cursor-pointer",
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

            div {
                class: "row-span-1 col-span-1 row-start-1 col-start-2",
                button {
                    class: "p-1 rounded hover:bg-neutral-700 cursor-pointer flex-none",
                    onclick: move |_evt| {
                        active_subject_name.set(subject_dropdown.read().to_string());
                        subject_dropdown.set(String::new());
                    },
                    svg {
                        class: "max-w-[48px] max-h-[48px]",
                        dangerous_inner_html: ms_search_icon_svg()
                    },
                },
                if !active_subject_name().is_empty() {
                    download_files_button { data_format: use_signal(|| DataFormat::CsvDefault), active_subject_name, filenames_downloaded, files_downloaded, extensions_downloaded}
                }
                if !files_downloaded.read().is_empty() {
                    clear_download_files_button {files_downloaded, filenames_downloaded, extensions_downloaded}
                }
            }
        }
    }
}

#[component]
pub fn subjects_schema_table(
    active_subject_name: Signal<String>,
    subject_schema_names: Signal<Vec<String>>,
    subject_schema_columns: Signal<Vec<String>>,
    subject_schema_types: Signal<Vec<String>>,
    subject_names: Signal<Vec<String>>,
    subject_num_rows: Signal<Vec<usize>>,
) -> Element {
    let schema_columns_types = use_memo(move || {
        get_subject_schema_col_type_by_subject_name(
            active_subject_name.read().as_str(),
            &subject_schema_names
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            &subject_schema_columns
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            &subject_schema_types
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });

    let num_rows = use_memo(move || {
        get_subject_num_rows_by_subject_name(
            active_subject_name.read().as_str(),
            &subject_names
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            &subject_num_rows.read().iter().collect::<Vec<_>>(),
        )
    });

    rsx! {
        div {
            class: "output_table",
            table {
                class: "table-auto rounded bg-neutral-800 text-gray-200",
                if active_subject_name().is_empty() {
                    caption { "No subject selected." },
                } else if num_rows().is_empty() {
                    caption { "Schema for {active_subject_name.to_string()}." },
                } else {
                    caption { "{active_subject_name.to_string()}: {num_rows().first().unwrap()} rows." },
                }
                thead {
                    class: "bg-neutral-700",
                    tr {
                        {SUBJECT_SCHEMA_HEADERS.iter().map(|header| {
                            rsx! {
                                th { "{header}" }
                            }
                        })}
                    },
                }
                tbody {
                    class: "table-auto text-gray-200",
                    {(0..schema_columns_types().0.len()).map(|i| {
                        let subject_col = schema_columns_types().0.get(i).unwrap().to_string();
                        let subject_type = schema_columns_types().1.get(i).unwrap().to_string();
                        rsx! {
                            tr {
                                class: "odd:bg-neutral-800 even:bg-neutral-900",
                                td { "{subject_col}" },
                                td { "{subject_type}" },
                            }
                        }
                    })}
                }
            }
        }
    }
}
