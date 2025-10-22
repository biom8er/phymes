use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignInState {
    #[serde(flatten)]
    pub jwt: SyncJWTState,
    #[serde(flatten)]
    pub session_names: SyncSessionNamesState,
}

pub static JWT: GlobalSignal<String> = Signal::global(String::new);
pub static EMAIL: GlobalSignal<String> = Signal::global(String::new);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncJWTState {
    pub jwt: String,
    pub email: String,
}

pub async fn sync_jwt_state(mut rx: UnboundedReceiver<SyncJWTState>) {
    while let Some(updated_state) = rx.next().await {
        (*JWT.write()).clear();
        (*EMAIL.write()).clear();
        (*JWT.write()).push_str(updated_state.jwt.as_str());
        (*EMAIL.write()).push_str(updated_state.email.as_str());
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearJWTState {}

pub async fn clear_jwt_state(mut rx: UnboundedReceiver<ClearJWTState>) {
    while let Some(_updated_state) = rx.next().await {
        (*JWT.write()).clear();
        (*EMAIL.write()).clear();
    }
}

pub static SESSION_NAMES: GlobalSignal<Vec<String>> = Signal::global(Vec::new);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncSessionNamesState {
    pub session_plans: Vec<String>,
}

pub async fn sync_session_names_state(mut rx: UnboundedReceiver<SyncSessionNamesState>) {
    while let Some(updated_state) = rx.next().await {
        (*SESSION_NAMES.write()).clear();
        (*SESSION_NAMES.write()).extend(updated_state.session_plans);
    }
    (*SESSION_NAMES.write()).sort();
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ClearSessionNamesState {}

pub async fn clear_session_names_state(mut rx: UnboundedReceiver<ClearSessionNamesState>) {
    while let Some(_updated_state) = rx.next().await {
        (*SESSION_NAMES.write()).clear();
    }
}

pub static BUILDER: GlobalSignal<bool> = Signal::global(|| false);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncBuilderState {
    pub show: bool,
}

pub async fn sync_builder_state(mut rx: UnboundedReceiver<SyncBuilderState>) {
    while let Some(updated_state) = rx.next().await {
        (*BUILDER.write()) = updated_state.show;
    }
}

pub static DEBUGGER: GlobalSignal<bool> = Signal::global(|| false);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncDebuggerState {
    pub show: bool,
}

pub async fn sync_debugger_state(mut rx: UnboundedReceiver<SyncDebuggerState>) {
    while let Some(updated_state) = rx.next().await {
        (*DEBUGGER.write()) = updated_state.show;
    }
}
