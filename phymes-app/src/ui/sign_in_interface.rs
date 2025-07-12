use dioxus::prelude::*;

use crate::ui::sign_in_state::{sync_jwt_state, SyncJWTState, EMAIL};

use super::backend::ADDR_BACKEND;

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
                        let addr = format!("{ADDR_BACKEND}/app/v1/sign_in");
                        match reqwest::Client::new()
                            .post(addr)
                            .basic_auth(email, Some(password))
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
