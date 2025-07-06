pub mod json_error;
pub mod session_info;
pub mod session_state;
pub mod session_stream;
pub mod sign_in;

// DM: We only want to run one server so all of the tests are consolidated here
// DM: Refactor to test each of the handlers individually
// see testing with axum <https://github.com/tokio-rs/axum/blob/main/examples/testing/src/main.rs>
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
#[cfg(test)]
mod tests {
    use crate::{
        handlers::session_stream::test_chat_handler::{StreamBytesInput, StreamBytesOutput},
        server::server_state::ServerState,
    };

    use super::*;
    use anyhow::Result;
    use axum::{Router, middleware, routing::post};
    use bytes::Bytes;
    use futures::TryStreamExt;
    use reqwest::{self, header::CONTENT_TYPE};
    use serde::{Deserialize, Serialize};
    use session_stream::test_chat_handler::stream_bytes;
    use sign_in::{authorize, sign_in};

    #[derive(Serialize, Deserialize, Debug)]
    struct TestJWT {
        jwt: String,
        email: String,
        session_plans: Vec<String>,
    }

    #[tokio::test]
    async fn test_stream_bytes() -> Result<()> {
        // setup the app
        let state = ServerState::new();
        let app = Router::new()
            .route("/app/v1/login", post(sign_in))
            .route(
                "/app/v1/chat",
                post(stream_bytes).layer(middleware::from_fn(authorize)),
            )
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:8000")
            .await
            .unwrap();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Login
        let response = reqwest::Client::new()
            .post("http://127.0.0.1:8000/app/v1/login")
            .basic_auth("myemail@gmail.com", Some("myemail@gmail.com"))
            .send()
            .await?
            .json::<TestJWT>()
            .await?;

        assert!(
            response
                .jwt
                .contains("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9")
        );
        assert_eq!(response.email, "myemail@gmail.com");
        assert_eq!(response.session_plans, ["Chat", "DocChat", "ToolChat"]);

        // Request some dummy data stream
        let data = StreamBytesInput {
            num_bytes: 3,
            greeting: "hello".to_string(),
        };
        let data_serialized = serde_json::to_string(&data)?;

        let stream = reqwest::Client::new()
            .post("http://127.0.0.1:8000/app/v1/chat")
            .bearer_auth(response.jwt.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(data_serialized)
            .send()
            .await?
            .bytes_stream();

        let bytes: Vec<Bytes> = stream.try_collect().await?;
        let outputs: Vec<String> = bytes
            .into_iter()
            .map(|bytes| {
                let o: StreamBytesOutput =
                    serde_json::from_str(std::str::from_utf8(&bytes).unwrap()).unwrap();
                o.message
            })
            .collect();

        assert_eq!(
            outputs,
            vec![
                "hello".to_string(),
                "hello".to_string(),
                "hello".to_string()
            ]
        );

        Ok(())
    }
}
