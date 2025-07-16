use crate::state::sign_in::{sync_jwt_state, SyncJWTState, EMAIL};
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
pub fn sign_in_modal() -> Element {
    // Sign-in signals
    #[allow(clippy::redundant_closure)]
    let mut email = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut password = use_signal(|| String::new());
    #[allow(clippy::redundant_closure)]
    let mut content = use_signal(|| String::new());

    // intialize state and coroutines
    use_coroutine(sync_jwt_state);

    rsx! {
        // Sign-in modal
        if !EMAIL.read().is_empty() {
            div {
                class: "messaging_list",
                p { "Signed in as {EMAIL.read().to_string()}." },
            }
        } else {
            // DM: Refactor the login to include a registration and forgot password
            //  1. enter email
            //  2. if email is not found in the server, Register new password
            //  3. if email is found in the server, enter existing password
            //  4. if password does not match existing password, provide message and try again
            //  5. if password is forgotten, send a reset password link to the registered email address
            //  6. After clicking on reset password link, a password reset page is provided
            //  7. Send follow-up email notifying the user that their password was reset
            div {
                class: "messaging_list",
                form {
                    div {
                        class: "container",
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
                                        sync_jwt.send(jwt_json);
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
                                sync_jwt.send(jwt_json);
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
    }
}
