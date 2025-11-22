use crate::state::{
    clear_jwt_state, clear_session_names_state, sync_builder_state,
    sync_current_active_session_state, sync_debugger_state, sync_jwt_state,
    sync_session_names_state, ClearJWTState, ClearSessionNamesState, SignInState, SyncBuilderState,
    SyncCurrentActiveSessionState, SyncDebuggerState, SyncJWTState, SyncSessionNamesState, BUILDER,
    DEBUGGER, EMAIL, JWT,
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
use phymes_server::{serverless_app, Serverless, ServerlessConfig};

/// View for the user to sign-in
#[component]
pub fn sign_in_view() -> Element {
    rsx! {
        div {
            class: "ml-16 p-2 list-none h-[77%] overflow-auto",
            if !JWT.read().is_empty() {
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
    let sync_jwt = use_coroutine_handle::<SyncJWTState>();
    use_coroutine(sync_session_names_state);
    let sync_session_names = use_coroutine_handle::<SyncSessionNamesState>();
    use_coroutine(sync_current_active_session_state);
    let sync_current_active_session_state = use_coroutine_handle::<SyncCurrentActiveSessionState>();

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
            class: "max-w-xl mx-auto w-11/12 p-4 rounded bg-gray-800",
            div {
                class: "flex flex-col gap-2",
                label { "Email" }
                input {
                    r#type: "email",
                    placeholder: "email",
                    oninput: move |event| email.set(event.value()),
                    class: "w-full p-2 rounded bg-gray-700 text-white",
                }
                label { "Password" }
                input {
                    r#type: "password",
                    placeholder: "password",
                    oninput: move |event| password.set(event.value()),
                    class: "w-full p-2 rounded bg-gray-700 text-white",
                }
                // label { "Remember me" }
                // input {
                //     r#type: "checkbox",
                //     checked: "checked",
                // }
            }
        }
        button {
            class: "block mx-auto mt-4 px-4 py-2 rounded bg-gray-800 text-white",
            onclick: move |_| async move {
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
                    Ok(response) => match response.json::<SignInState>()
                        .await {
                            Ok(jwt_json) => {
                                // Set the active session
                                sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: jwt_json.session_names.session_plans.first().unwrap().to_string() });

                                // Set the session names
                                sync_session_names.send(SyncSessionNamesState { session_plans: jwt_json.session_names.session_plans });

                                // Set the sign-in credentials
                                sync_jwt.send(SyncJWTState { jwt: jwt_json.jwt.jwt, email: jwt_json.jwt.email });

                                // Clear the signals
                                content.write().clear();
                                email.write().clear();
                                password.write().clear();

                            }
                            Err(err) => {
                                let msg = format!("There was a problem with Authentication \n{err:?}.\nLet's try again.");
                                content.write().push_str(msg.as_str());
                            }
                        },
                    Err(err) =>  {
                        let msg = format!("There was a problem with Authentication \n{err:?}.\nLet's try again.");
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
                        let jwt_json: SignInState = serde_json::from_slice(bytes.first().unwrap()).unwrap();

                        // Set the active session
                        sync_current_active_session_state.send(SyncCurrentActiveSessionState { name: jwt_json.session_names.session_plans.first().unwrap().to_string() });

                        // Set the session names
                        sync_session_names.send(SyncSessionNamesState { session_plans: jwt_json.session_names.session_plans });

                        // Set the sign-in credentials
                        sync_jwt.send(SyncJWTState { jwt: jwt_json.jwt.jwt, email: jwt_json.jwt.email });

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
            class: "block mx-auto mt-2 px-3 py-1 rounded bg-gray-800 text-white",
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
    let clear_jwt_state = use_coroutine_handle::<ClearJWTState>();
    use_coroutine(clear_session_names_state);
    let clear_session_names_state = use_coroutine_handle::<ClearSessionNamesState>();

    rsx! {
        p { "Signed in as {EMAIL.read().to_string()}." },
        button {
            class: "inline-block ml-4 px-3 py-1 rounded bg-gray-800 text-white",
            onclick: move |_| async move {
                clear_jwt_state.send(ClearJWTState {});
                clear_session_names_state.send(ClearSessionNamesState {});
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
    let sync_builder_state = use_coroutine_handle::<SyncBuilderState>();
    use_coroutine(sync_debugger_state);
    let sync_debugger_state = use_coroutine_handle::<SyncDebuggerState>();

    // Determine the button text
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
            class: "inline-block mr-2 px-3 py-1 rounded bg-gray-800 text-white",
            onclick: move |_evt| async move {
                sync_builder_state.send(SyncBuilderState { show: !BUILDER()});
                if BUILDER() {
                    // If we are enabling builder mode, disable debugger mode
                    sync_debugger_state.send(SyncDebuggerState { show: false });
                }
            },
            "{builder}"
        },
        button {
            class: "inline-block px-3 py-1 rounded bg-gray-800 text-white",
            onclick: move |_evt| async move {
                sync_debugger_state.send(SyncDebuggerState { show: !DEBUGGER()});
            },
            // If we are enabling builder mode, disable debugger mode
            if !BUILDER() {
                "{debugger}"
            }
        }
    }
}
