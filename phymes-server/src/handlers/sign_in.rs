// Server related imports
use axum::{
    body::Body,
    extract::{Json, Request, State},
    http::{Response, StatusCode},
    middleware::Next,
    response::IntoResponse,
};

// Authentication imports
use axum_extra::{
    TypedHeader,
    headers::{
        Authorization,
        authorization::{Basic, Bearer},
    },
};

use bcrypt::{DEFAULT_COST, hash, verify};
use chrono::{Duration, Utc};
use jsonwebtoken::{DecodingKey, EncodingKey, Header, TokenData, Validation, decode, encode};

// General imports
use crate::{
    handlers::json_error::{ErrorToResponse, JsonError},
    server::server_state::ServerState,
};
use http::HeaderValue;
use serde::{Deserialize, Serialize};
use serde_json::json;

// Agent imports
use phymes_agents::session_plans::available_session_plans::AvailableSessionPlans;

/// From <https://github.com/seanmonstar/reqwest/blob/v0.12.22/src/util.rs#L4>
pub fn basic_auth<U, P>(username: U, password: Option<P>) -> HeaderValue
where
    U: std::fmt::Display,
    P: std::fmt::Display,
{
    use base64::prelude::BASE64_STANDARD;
    use base64::write::EncoderWriter;
    use std::io::Write;

    let mut buf = b"Basic ".to_vec();
    {
        let mut encoder = EncoderWriter::new(&mut buf, &BASE64_STANDARD);
        let _ = write!(encoder, "{username}:");
        if let Some(password) = password {
            let _ = write!(encoder, "{password}");
        }
    }
    let mut header = HeaderValue::from_bytes(&buf).expect("base64 is always valid HeaderValue");
    header.set_sensitive(true);
    header
}

#[derive(Serialize, Deserialize)]
pub struct Cliams {
    pub exp: usize,
    pub iat: usize,
    pub email: String,
}

pub fn verify_password(password: &str, hash: &str) -> Result<bool, bcrypt::BcryptError> {
    verify(password, hash)
}

pub fn hash_password(password: &str) -> Result<String, bcrypt::BcryptError> {
    let hash = hash(password, DEFAULT_COST)?;
    Ok(hash)
}

pub fn encode_jwt(email: String) -> Result<String, StatusCode> {
    let secret: String = "randomstring".to_string();

    let now = Utc::now();
    let expire: chrono::TimeDelta = Duration::hours(24);
    let exp: usize = (now + expire).timestamp() as usize;
    let iat: usize = now.timestamp() as usize;

    let claim = Cliams { iat, exp, email };

    encode(
        &Header::default(),
        &claim,
        &EncodingKey::from_secret(secret.as_ref()),
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

pub fn decode_jwt(jwt: String) -> Result<TokenData<Cliams>, StatusCode> {
    let secret = "randomstring".to_string();

    let result: Result<TokenData<Cliams>, StatusCode> = decode(
        &jwt,
        &DecodingKey::from_secret(secret.as_ref()),
        &Validation::default(),
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
    result
}

#[derive(Clone)]
pub struct CurrentUser {
    pub email: String,
    pub first_name: String,
    pub last_name: String,
    pub password_hash: String,
}

/// Create the session name by combining the user ID
/// with the session plan
pub fn create_session_name(email: &str, session_plan: &str) -> String {
    let session_name = format!("{email}{session_plan}");
    session_name
}

/// authorization middleware
pub async fn authorize(
    TypedHeader(Authorization(bearer)): TypedHeader<Authorization<Bearer>>,
    mut req: Request,
    next: Next,
) -> Result<Response<Body>, impl IntoResponse> {
    // Authentication
    let token = bearer.token();
    let token_data = match decode_jwt(token.to_string()) {
        Ok(data) => data,
        Err(_) => {
            return Err(JsonError::new("Unable to decode token".to_string())
                .to_response(StatusCode::UNAUTHORIZED));
        }
    };

    // Fetch the user details from the database
    let current_user = match test_sign_in_handler::retrieve_user_by_email(&token_data.claims.email)
    {
        Some(user) => user,
        None => {
            return Err(JsonError::new("You are not an authorized user".to_string())
                .to_response(StatusCode::UNAUTHORIZED));
        }
    };

    req.extensions_mut().insert(current_user);
    Ok(next.run(req).await)
}

/// sign in endpoint
pub async fn sign_in(
    TypedHeader(Authorization(creds)): TypedHeader<Authorization<Basic>>,
    State(mut state): State<ServerState>,
) -> impl IntoResponse {
    // Retrieve user from the database
    let user = match test_sign_in_handler::retrieve_user_by_email(creds.username()) {
        Some(user) => user,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "Unauthorized"})),
            );
        } // User not found
    };

    // Compare the password
    match verify_password(creds.password(), &user.password_hash) {
        Ok(result) => {
            if !result {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "Failed to generate token"})),
                ); // Wrong password
            }
        }
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Failed to generate token"})),
            );
        }
    }

    // Generate JWT
    let Ok(jwt) = encode_jwt(creds.username().to_string()) else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to generate token"})),
        );
    };

    // Add user state if it does not exist already
    let (session_plans, _session_names) = if !state.check_email_in_state(creds.username()) {
        if let Err(_e) = state.read_state_by_email(
            &format!("{}/.cache", std::env::var("HOME").unwrap_or("".to_string())),
            creds.username(),
        ) {
            match state.create_session_names_by_email(creds.username()) {
                Some(session_plans) => session_plans,
                None => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(
                            json!({"error": "Failed to find session plans for user {creds.username()}"}),
                        ),
                    );
                }
            }
        } else {
            match state.get_session_names_by_email(creds.username()) {
                Some(session_plans) => session_plans,
                None => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(
                            json!({"error": "Failed to find session plans for user {creds.username()}"}),
                        ),
                    );
                }
            }
        }
    } else {
        match state.get_session_names_by_email(creds.username()) {
            Some(session_plans) => session_plans,
            None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        json!({"error": "Failed to find session plans for user {creds.username()}"}),
                    ),
                );
            }
        }
    };

    // Return the token
    (
        StatusCode::OK,
        Json(json!({"jwt": jwt, "email": user.email, "session_plans": session_plans})),
    )
}

pub mod test_sign_in_handler {
    use super::*;

    // DM: this should be migrated to persistent storage
    pub fn retrieve_user_by_email(email: &str) -> Option<CurrentUser> {
        let password_hash = hash_password(email).unwrap();
        let current_user: CurrentUser = CurrentUser {
            email: "myemail@gmail.com".to_string(),
            first_name: "Eze".to_string(),
            last_name: "Sunday".to_string(),
            password_hash,
        };
        Some(current_user)
    }

    // DM: this should be migrated to persistent storage
    pub fn retrieve_session_plans_by_email(_email: &str) -> Option<Vec<String>> {
        Some(AvailableSessionPlans::get_all_session_plan_names())
    }
}
