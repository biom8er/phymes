use dioxus::prelude::*;

use phymes_event::Publication;
use phymes_message::{
    MessageBuilderTrait, NetworkInterfaceMessage, NetworkInterfaceMessageBuilder,
    NetworkInterfaceMessageBuilderTrait,
};
use phymes_schemas::{AvailableSubjects, DataFormat};
use phymes_server::{create_network_name, NetworkBuildResult};
use phymes_subject::{
    BuildableTrait, BuilderTrait, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_templates::AvailableNetworks;

use crate::{
    state::{
        filter_in_mermaid_diagrams_by_network_name, get_non_duplicated_sorted_subjects,
        svg_icons::{ms_chevron_circle_icon_svg, ms_search_icon_svg},
        sync_current_active_network_state, SyncCurrentActiveNetworkState, ACTIVE_SESSION_NAME,
        BUILDER, EMAIL, JWT, SESSION_NAMES,
    },
    ui::{
        builds::network_name_editor,
        builds_dropdown_view, diagram_code_editor,
        main_window::{split_panel, SnapPct},
    },
};

#[cfg(not(feature = "serverless"))]
use futures::StreamExt;

#[cfg(feature = "mermaid_js")]
use crate::state::MermaidJsObject;

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

#[cfg(not(feature = "serverless"))]
use super::backend::ADDR_BACKEND;

#[cfg(feature = "serverless")]
use crate::state::RUNTIME_ENV;
#[cfg(feature = "serverless")]
use bytes::Bytes;
#[cfg(feature = "serverless")]
use futures::TryStreamExt;
#[cfg(feature = "serverless")]
use phymes_server::{serverless_app, Serverless, ServerlessConfig};

/// View for the per runtime settings
#[component]
pub fn apps_interface_view() -> Element {
    // Intialize signals
    let is_flowchart_shown = use_signal(|| true);
    let active_network_name = use_signal(String::new);
    let mut active_flowchart_diagram = use_signal(String::new);
    let mut active_er_diagram = use_signal(String::new);

    let mut mermaid_network_names = use_signal(Vec::<String>::new);
    let mut mermaid_flowchart_diagrams = use_signal(Vec::<String>::new);
    let mut mermaid_er_diagrams = use_signal(Vec::<String>::new);
    let mut mermaid_timestamps = use_signal(Vec::<i64>::new);

    // `get_network_state` will update itself whenever EMAIL or ACTIVE_SESSION_NAME change
    let get_network_state: Memo<NetworkInterfaceMessageBuilder> = use_memo(move || {
        let network_name = if BUILDER() {
            // DM: this can be better optimized to prevent redundant API calls each time the active network is changed in Builder mode
            create_network_name(
                EMAIL().as_str(),
                AvailableNetworks::Builder.to_string().as_str(),
            )
        } else {
            create_network_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str())
        };
        NetworkInterfaceMessage::get_builder()
            .with_network_name(&network_name)
            .with_format(&DataFormat::Ipc)
            .with_publisher(&network_name)
            .with_update(&Publication::None)
            .with_stream(false)
    });

    // Get the mermaid.js diagrams for the network
    use_resource(move || async move {
        // clear the current mermaid state
        mermaid_network_names.set(Vec::new());
        mermaid_flowchart_diagrams.set(Vec::new());
        mermaid_er_diagrams.set(Vec::new());
        mermaid_timestamps.set(Vec::new());

        // get the mermaid state
        let route = "/app/v1/get_state";
        let subject = if BUILDER() {
            AvailableSubjects::BuilderMermaid.to_string()
        } else {
            AvailableSubjects::NetworkMermaid.to_string()
        };
        let data_serialized = serde_json::to_string(
            &get_network_state()
                .with_subject(subject.as_str())
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
                match SubjectBuilder::new_from_ipc_stream(&bytes) {
                    Ok(builder) => {
                        let table = builder.with_name("").build().unwrap();
                        let combined = table
                            .get_column_as_vec_nonprimitive::<String>("network_name")
                            .unwrap()
                            .into_iter()
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("flowchart_diagram")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("er_diagram")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_primitive::<i64>("timestamp")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .filter_map(|(((scn, fd), ed), t)| {
                                if scn.contains("__deleted__") {
                                    None
                                } else {
                                    Some((scn, fd, ed, t))
                                }
                            })
                            .collect::<Vec<_>>();
                        for (scn, fd, ed, t) in combined {
                            mermaid_network_names.push(scn);
                            mermaid_flowchart_diagrams.push(fd);
                            mermaid_er_diagrams.push(ed);
                            mermaid_timestamps.push(t);
                        }
                    }
                    Err(err) => tracing::error!("{err:?}"),
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
                            .get_column_as_vec_nonprimitive::<String>("network_name")
                            .unwrap()
                            .into_iter()
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("flowchart_diagram")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_nonprimitive::<String>("er_diagram")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .zip(
                                table
                                    .get_column_as_vec_primitive::<i64>("timestamp")
                                    .unwrap()
                                    .into_iter(),
                            )
                            .filter_map(|(((scn, fd), ed), t)| {
                                if scn.contains("__deleted__") {
                                    None
                                } else {
                                    Some((scn, fd, ed, t))
                                }
                            })
                            .collect::<Vec<_>>();
                        for (scn, fd, ed, t) in combined {
                            mermaid_network_names.push(scn);
                            mermaid_flowchart_diagrams.push(fd);
                            mermaid_er_diagrams.push(ed);
                            mermaid_timestamps.push(t);
                        }
                    }
                    Err(err) => tracing::error!("{err:?}"),
                }
            }
            Err(err) => tracing::error!("{err:?}"),
        }
    });

    // Filter the mermaid.js diagrams for the network
    let filtered_diagrams = use_memo(move || {
        // The builder network names do not contain the email
        let network_name = if BUILDER() {
            active_network_name()
        } else {
            create_network_name(EMAIL().as_str(), ACTIVE_SESSION_NAME().as_str())
        };

        // Filter in the active diagrams
        let (_network_names, flowchart_diagrams, er_diagrams, timestamps) =
            filter_in_mermaid_diagrams_by_network_name(
                &network_name,
                &mermaid_network_names
                    .read()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                &mermaid_flowchart_diagrams
                    .read()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                &mermaid_er_diagrams
                    .read()
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                &mermaid_timestamps(),
            );

        // Sort by timestamp
        let mut combined = flowchart_diagrams
            .into_iter()
            .zip(er_diagrams.into_iter())
            .zip(timestamps.into_iter())
            .map(|((a, b), c)| (a, b, c))
            .collect::<Vec<_>>();
        combined.sort_by(|a, b| a.2.cmp(&b.2));

        // last is most recent
        match combined.last() {
            Some(diagrams) => (Some(diagrams.0.to_owned()), Some(diagrams.1.to_owned())),
            None => (None, None),
        }
    });

    // Update the active mermaid.js diagrams for the network
    use_resource(move || async move {
        if let Some(diagram) = filtered_diagrams().0 {
            active_flowchart_diagram.set(diagram.to_string());
        }
        if let Some(diagram) = filtered_diagrams().1 {
            active_er_diagram.set(diagram.to_string());
        }
    });

    let diagram_code: Memo<NetworkBuildResult> = use_memo(move || {
        // Get the active diagram code
        let diagram_code = if is_flowchart_shown() {
            active_flowchart_diagram.read().to_string()
        } else {
            active_er_diagram.read().to_string()
        };

        NetworkBuildResult::new(Some(&diagram_code), None)
    });

    // Track when the diagram code changes
    let is_saved = use_signal(|| true);

    // Build errors that may have occured
    let build_errors = use_signal(String::new);

    // Panel direction
    let horizontal = use_signal(|| false);

    rsx! {
        if JWT.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Please sign-in before activating a network." },
            }
        } else if SESSION_NAMES.read().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "Waiting to retrieve available network plans..." },
            }
        } else {
            if BUILDER() {
                split_panel {
                    top: rsx! {
                        div {
                            class: "h-full w-full p-2 flex flex-col items-center",
                            builds_dropdown_view { is_flowchart_shown, active_network_name, active_flowchart_diagram, active_er_diagram, mermaid_network_names, mermaid_flowchart_diagrams, mermaid_er_diagrams, mermaid_timestamps, is_saved, build_errors }
                            diagram_code_editor { is_flowchart_shown, active_network_name, active_flowchart_diagram, active_er_diagram, is_saved }
                        }
                    },
                    bottom: rsx! {
                        div {
                            class: "h-full w-full p-2 flex flex-col items-center",
                            network_name_editor { active_network_name }
                            mermaid_view { diagram_code, build_errors }
                        }
                    },
                    initial_top_pct: SnapPct::Pct50,
                    horizontal: horizontal,
                }
            } else {
                div {
                    class: "h-full w-full p-2 flex flex-col items-center",
                    apps_dropdown_view { is_flowchart_shown }
                    if !active_network_name().is_empty() {
                        p {
                            class: "w-full rounded p-2 items-center text-center text-gray-200 bg-neutral-800",
                            "{active_network_name}"
                        }
                    }
                    mermaid_view { diagram_code, build_errors }
                }
            }
        }
    }
}

/// View for the per runtime settings
#[component]
pub fn apps_dropdown_view(mut is_flowchart_shown: Signal<bool>) -> Element {
    // Intialize state and coroutines
    use_coroutine(sync_current_active_network_state);
    let sync_current_active_network_state = use_coroutine_handle::<SyncCurrentActiveNetworkState>();

    // Dropdown signals
    let mut show_subject_dropdown = use_signal(|| false);
    let mut subject_dropdown = use_signal(String::new);
    let subjects_vec = use_memo(move || {
        get_non_duplicated_sorted_subjects(
            &SESSION_NAMES
                .read()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
    });
    let mut subjects_filtered: Signal<Vec<String>> = use_signal(Vec::new);

    rsx! {
        div {
            // input + 2 buttons of 64 px by 64 px
            class: "p-2 rounded bg-neutral-800 grid grid-rows-[auto_1fr] grid-cols-[1fr_auto] w-full sm:max-w-3/4 md:max-w-1/2",
            form {
                class: "w-full h-full flex row-span-1 col-span-1 row-start-1 col-start-1",
                input {
                    class: "w-full h-full bg-neutral-700",
                    r#type: "text",
                    placeholder: "search apps",
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

            // Dynamic dropdown
            if show_subject_dropdown() {
                div {
                    class: "p-2 rounded bg-neutral-800 list-none flex row-span-1 col-span-1 row-start-2 col-start-1",
                    ul {
                        {subjects_vec().iter().filter(|s| ACTIVE_SESSION_NAME.read().to_string()!=**s && !subjects_filtered.read().contains(*s)).enumerate().map(|(i, sub)|  {
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
                    class: "p-2 rounded hover:bg-neutral-700 cursor-pointer flex-none",
                    onclick: move |_evt| async move {
                        // Reset the dropdown
                        let active_network = subject_dropdown.try_read().unwrap().to_string();
                        subject_dropdown.set(String::new());

                        // Set the active network
                        sync_current_active_network_state.send(SyncCurrentActiveNetworkState { name: active_network.clone() });
                    },
                    svg {
                        class: "max-w-[24px] max-h-[24px]",
                        dangerous_inner_html: ms_search_icon_svg()
                    },
                },

                if !ACTIVE_SESSION_NAME().is_empty() {
                    button {
                        class: "p-2 rounded hover:bg-neutral-700 cursor-pointer flex-none",
                        onclick: move |_| async move {
                            let current = is_flowchart_shown.read().to_owned();
                            is_flowchart_shown.set(!current);
                        },
                        svg {
                            class: "max-w-[24px] max-h-[24px]",
                            dangerous_inner_html: ms_chevron_circle_icon_svg()
                        },
                    },
                }
            }
        }

        if !ACTIVE_SESSION_NAME().is_empty() {
            div {
                class: "p-2 flex flex-col items-center",
                p { "{ACTIVE_SESSION_NAME().to_string()}" },
            }
        }
    }
}

/// View to visualize mermaid.js diagrams
///
/// # Notes
/// * mermaid.js is used to render SVG diagrams
/// * errors when creating the SVG are also shown
#[component]
pub fn mermaid_view(
    diagram_code: Memo<NetworkBuildResult>,
    mut build_errors: Signal<String>,
) -> Element {
    let mut diagram_svg = use_signal(String::new);
    let mut error_mjs = use_signal(String::new);
    let id = use_signal(|| "graphDiv".to_string());
    // Temporary DOM elemented created by Mermaid.js breaks Dioxus
    // when the actual SVG target ID is used...
    let id_decoy = use_signal(|| "GraphDiv".to_string());

    // Render the mermaid.js diagram
    use_resource(move || async move {
        let eval = document::eval(
            format!(
                r#"
        try {{
            let code = await dioxus.recv();
            const {{ svg }} = await mermaid.render("{id_decoy}", code);
            return {{ svg: svg, error: null }};
        }} catch (error) {{
            return {{ svg: null, error: error.message }};
        }}"#
            )
            .as_str(),
        );
        eval.send(diagram_code().diagram.take().unwrap()).unwrap();
        let mermaid_js_object = match eval.await {
            Ok(res) => {
                let res: MermaidJsObject = serde_json::from_value(res).unwrap();
                res
            }
            Err(err) => {
                tracing::error!("Mermaid.js err {err:?}");
                MermaidJsObject {
                    svg: None,
                    error: Some(err.to_string()),
                }
            }
        };

        // Update the signals
        if mermaid_js_object.error.is_none() {
            diagram_svg.set(mermaid_js_object.svg.unwrap_or_default());
        }
        error_mjs.set(mermaid_js_object.error.unwrap_or_default());
    });

    // Add pan and zoom to the svg
    use_effect(move || {
        let _ = diagram_svg(); // needed to trigger the effect
        document::eval(
            format!(
                r#"
            const container = document.getElementById("{id}");
            const svgElement = container.querySelector("svg");

            // Initialize Panzoom
            const panzoomInstance = Panzoom(svgElement, {{
                maxScale: 100,
                minScale: 0.1,
                step: 0.1,
            }});

            // Add mouse wheel zoom
            container.addEventListener("wheel", (event) => {{
                panzoomInstance.zoomWithWheel(event);
            }});
            "#
            )
            .as_str(),
        );
    });

    rsx! {
        if !build_errors().is_empty() {
            div {
                class: "rounded p-2 items-center bg-neutral-700",
                p {
                    class: "text-gray-200 overflow-auto",
                    "{build_errors}"
                },
            }
        }
        if !error_mjs().is_empty() {
            div {
                class: "rounded p-2 items-center text-gray-200 bg-neutral-700",
                p {
                    class: "text-gray-200 overflow-auto",
                    "{error_mjs}"
                },
            }
        }
        if let Some(error_ctxb) = diagram_code().error {
            div {
                class: "rounded p-2 items-center bg-neutral-700",
                p {
                    class: "text-gray-200 overflow-auto",
                    "{error_ctxb}"
                },
            }
        }
        if !diagram_svg().is_empty() {
            mermaid_div { diagram_svg, id }
        }
    }
}

#[component]
pub fn mermaid_div(diagram_svg: Signal<String>, id: Signal<String>) -> Element {
    rsx! {
        div {
            id: id(),
            class: "w-full h-full",
            svg { dangerous_inner_html: diagram_svg() }
        }
    }
}
