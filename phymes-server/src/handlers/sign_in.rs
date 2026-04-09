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
    state::{ServerState, UserState},
};
#[cfg(feature = "wasip2")]
use http::HeaderValue;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// From <https://github.com/seanmonstar/reqwest/blob/v0.12.22/src/util.rs#L4>
#[cfg(feature = "wasip2")]
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
struct Claims {
    pub exp: usize,
    pub iat: usize,
    pub email: String,
}

fn verify_password(password: &str, hash: &str) -> Result<bool, bcrypt::BcryptError> {
    verify(password, hash)
}

#[allow(unused)]
fn hash_password(password: &str) -> Result<String, bcrypt::BcryptError> {
    let hash = hash(password, DEFAULT_COST)?;
    Ok(hash)
}

fn encode_jwt(email: String) -> Result<String, StatusCode> {
    let secret: String = "randomstring".to_string();

    let now = Utc::now();
    let expire: chrono::TimeDelta = Duration::hours(24);
    let exp: usize = (now + expire).timestamp() as usize;
    let iat: usize = now.timestamp() as usize;

    let claim = Claims { iat, exp, email };

    encode(
        &Header::default(),
        &claim,
        &EncodingKey::from_secret(secret.as_ref()),
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn decode_jwt(jwt: String) -> Result<TokenData<Claims>, StatusCode> {
    let secret = "randomstring".to_string();

    let result: Result<TokenData<Claims>, StatusCode> = decode(
        &jwt,
        &DecodingKey::from_secret(secret.as_ref()),
        &Validation::default(),
    )
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
    result
}

/// Remove all non alphanumeric characters
fn remove_nonalphanumeric(email: &str) -> String {
    let mut input = String::from(email);
    input.retain(|c| c.is_alphanumeric() || c.is_whitespace());
    input
}

/// Create the session name by combining the user ID
/// with the session plan
pub fn create_session_name(email: &str, session_plan: &str) -> String {
    let sanitized_email = remove_nonalphanumeric(email);
    let session_name = format!("{sanitized_email}{session_plan}");
    session_name
}

/// authorization middleware
pub async fn authorize(
    TypedHeader(Authorization(bearer)): TypedHeader<Authorization<Bearer>>,
    State(state): State<UserState>,
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

    // Retrieve user from the database
    let (user_info, user_networks) =
        match state.get_user_by_email(&token_data.claims.email).await {
            Ok((user_info, user_networks)) => (user_info, user_networks),
            Err(err) => {
                return Err(
                    JsonError::new(err.to_string()).to_response(StatusCode::INTERNAL_SERVER_ERROR)
                );
            }
        };
    if user_info.is_empty() {
        return Err(JsonError::new("You are not an authorized user".to_string())
            .to_response(StatusCode::UNAUTHORIZED));
    }
    if user_networks.is_empty() {
        return Err(JsonError::new(
            "Failed to find session plans for user {token_data.claims.email}".to_string(),
        )
        .to_response(StatusCode::UNAUTHORIZED));
    }

    req.extensions_mut()
        .insert((token_data.claims.email.to_owned(), user_networks));
    Ok(next.run(req).await)
}

/// sign in endpoint
#[axum::debug_handler]
pub async fn sign_in(
    TypedHeader(Authorization(creds)): TypedHeader<Authorization<Basic>>,
    State((state, _)): State<(UserState, ServerState)>,
) -> impl IntoResponse {
    // Retrieve user from the database
    let (user_info, user_networks) = match state.get_user_by_email(creds.username()).await {
        Ok((user_info, user_networks)) => (user_info, user_networks),
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("{err}")})),
            );
        }
    };

    // Check that the user exists and has session plans
    if user_info.is_empty() {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "Unauthorized"})),
        );
    }
    if user_networks.is_empty() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Failed to find session plans for user {creds.username()}"})),
        );
    }

    // Compare the password
    match verify_password(creds.password(), &user_info.first().unwrap().password_hash) {
        Ok(result) => {
            if !result {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": "Wrong password"})),
                ); // Wrong password
            }
        }
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "Failed to verify the password"})),
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

    // Return the sign-in confirmation
    let session_plans = user_networks
        .iter()
        .map(|ctx| ctx.network_name.to_string())
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(
            json!({"jwt": jwt, "email": creds.username().to_string(), "session_plans": session_plans}),
        ),
    )
}
