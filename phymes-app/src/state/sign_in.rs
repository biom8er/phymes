use dioxus::prelude::*;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

// Sign-in State
#[allow(clippy::redundant_closure)]
pub static JWT: GlobalSignal<String> = Signal::global(|| String::new());
#[allow(clippy::redundant_closure)]
pub static EMAIL: GlobalSignal<String> = Signal::global(|| String::new());
#[allow(clippy::redundant_closure)]
pub static SESSION_NAMES: GlobalSignal<Vec<String>> = Signal::global(|| Vec::new());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyncJWTState {
    pub jwt: String,
    pub email: String,
    pub session_plans: Vec<String>,
}

pub async fn sync_jwt_state(mut rx: UnboundedReceiver<SyncJWTState>) {
    while let Some(updated_state) = rx.next().await {
        (*SESSION_NAMES.write()).clear();
        (*JWT.write()).clear();
        (*EMAIL.write()).clear();
        (*SESSION_NAMES.write()).extend(updated_state.session_plans);
        (*JWT.write()).push_str(updated_state.jwt.as_str());
        (*EMAIL.write()).push_str(updated_state.email.as_str());
    }
    (*SESSION_NAMES.write()).sort();
}
