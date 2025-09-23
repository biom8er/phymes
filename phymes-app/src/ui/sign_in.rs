use crate::state::{
    apps::{sync_current_active_session_state, SyncCurrentActiveSessionState},
    sign_in::{clear_jwt_state, sync_builder_state, sync_debugger_state, sync_jwt_state, ClearJWTState, SyncBuilderState, SyncDebuggerState, SyncJWTState, BUILDER, DEBUGGER, EMAIL, JWT},
};
use dioxus::prelude::*;

#[cfg(not(feature = "serverless"))]
use reqwest::{self, header::CONTENT_TYPE};

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

/// View for the user to sign-in
#[component]
pub fn sign_in_view() -> Element {
    // Memo to safetly determine sign-in
    let is_signed_in = use_memo(move || {
        !JWT.read().is_empty()
    });

    rsx! {
        div {
            class: "messaging_list",
            if is_signed_in() {
                sign_out_form {}
                application_mode {}
            } else {
                sign_in_form {}
            }
        }
    }
}

/// View for the user to sign-in
#[component]
pub fn sign_in_form() -> Element {
    // Sign-in signals
    #[allow(clippy::redundant_closure)]
    let mut email = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut password = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut content = use_signal(|| String::new());

    // intialize state and coroutines
    use_coroutine(sync_jwt_state);
    use_coroutine(sync_current_active_session_state);

    // DM: Refactor the login to include a registration and forgot password
    //  1. enter email
    //  2. if email is not found in the server, Register new password
    //  3. if email is found in the server, enter existing password
    //  4. if password does not match existing password, provide message and try again
    //  5. if password is forgotten, send a reset password link to the registered email address
    //  6. After clicking on reset password link, a password reset page is provided
    //  7. Send follow-up email notifying the user that their password was reset
    rsx! {
        form {
            class: "sign_in_form",
            div {
                label { "Email" }
                input {
                    r#type: "email",
                    placeholder: "email",
                    oninput: move |event| email.set(event.value()),
                }
                label { "Password" }
                input {
                    r#type: "password",
                    placeholder: "password",
                    oninput: move |event| password.set(event.value()),
                }
                // label { "Remember me" }
                // input {
                //     r#type: "checkbox",
                //     checked: "checked",
                // }
            }
        }
        button {
            onclick: move |_| async move {
                let sync_jwt = use_coroutine_handle::<SyncJWTState>();
                let route = "/app/v1/sign_in";

                #[cfg(not(feature = "serverless"))]
                let addr = format!("{ADDR_BACKEND}{route}");
                #[cfg(not(feature = "serverless"))]
                match reqwest::Client::new()
                    .post(addr)
                    .basic_auth(email, Some(password))
                    .header(CONTENT_TYPE, "text/plain; charset=utf-8")
                    .send()
                    .await {
                    Ok(response) => match response.json::<SyncJWTState>()
                        .await {
                            Ok(jwt_json) => {
                                // Set the active session
                                let sync_current_active_session_state = use_coroutine_handle::<SyncCurrentActiveSessionState>();
                                sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: jwt_json.session_plans.first().unwrap().to_string() });

                                // Set the sign-in credentials
                                sync_jwt.send(jwt_json);
                                
                                // Clear the signals
                                content.write().clear();
                                email.write().clear();
                                password.write().clear();

                            }
                            Err(err) => {
                                let msg = format!("There was a problem with Authentication {err:?}. Let's try again.");
                                content.write().push_str(msg.as_str());
                            }
                        },
                    Err(err) =>  {
                        let msg = format!("There was a problem with Authentication {err:?}. Let's try again.");
                        content.write().push_str(msg.as_str());
                    }
                }

                #[cfg(feature = "serverless")]
                let config = ServerlessConfig {
                    route: route.to_string(),
                    basic_auth: Some(format!("{email}:{password}")),
                    bearer_auth: None,
                    data: None,
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
                        let jwt_json: SyncJWTState = serde_json::from_slice(bytes.first().unwrap()).unwrap();

                        // Set the active session
                        let sync_current_active_session_state = use_coroutine_handle::<SyncCurrentActiveSessionState>();
                        sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: jwt_json.session_plans.first().unwrap().to_string() });

                        // Set the sign-in credentials
                        sync_jwt.send(jwt_json);
                                
                        // Clear the signals
                        content.write().clear();
                        email.write().clear();
                        password.write().clear();
                    }
                    Err(err) =>  {
                        let msg = format!("There was a problem with Authentication {err:?}. Let's try again.");
                        content.write().push_str(msg.as_str());
                    }
                }
            },
            "sign-in"
        }
        button {
            onclick: move |_| async move {
                // TODO
            },
            "forgot password"
        }
        p { "{content.to_string()}" }          
    }
}

/// View for the user to sign-out
#[component]
pub fn sign_out_form() -> Element {
    use_coroutine(clear_jwt_state);

    rsx! {
        p { "Signed in as {EMAIL.read().to_string()}." },
        button {
            onclick: move |_| async move {
                let clear_jwt_state = use_coroutine_handle::<ClearJWTState>();
                clear_jwt_state.send(ClearJWTState {});
            },
            "sign-out"
        },
    }
}

/// View for the user to change the mode of the application
#[component]
pub fn application_mode() -> Element {
    // intialize state and coroutines
    use_coroutine(sync_builder_state);
    use_coroutine(sync_debugger_state);

    let builder = use_memo(move || {
        if BUILDER() {
            "disable builder mode"
        } else {
            "enable builder mode"
        }
    });
    let debugger = use_memo(move || {
        if DEBUGGER() {
            "disable debugger mode"
        } else {
            "enable debugger mode"
        }
    });

    rsx! {
        p { "Application modes" }
        button {
            onclick: move |_evt| async move {                    
                let sync_builder_state = use_coroutine_handle::<SyncBuilderState>();
                sync_builder_state.send(SyncBuilderState { show: !BUILDER()});
            },
            "{builder}"
        },
        button {
            onclick: move |_evt| async move {
                let sync_debugger_state = use_coroutine_handle::<SyncDebuggerState>();
                sync_debugger_state.send(SyncDebuggerState { show: !DEBUGGER()});
            },
            "{debugger}"
        }
    }
}