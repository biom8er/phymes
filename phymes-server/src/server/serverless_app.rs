// Server related imports
use axum::{Router, response::Response};

use http::Request;
use tower_service::Service;

// From lib
use super::server_app::AppBuilder;

/// Stateful implementation of the router to enable
/// continuous calls to the router
#[derive(Default)]
pub struct Serverless {
    router: Router,
}

impl Serverless {
    pub fn new() -> Self {
        Self {
            router: AppBuilder::new().build(),
        }
    }

    pub async fn call(&mut self, request: Request<String>) -> Response {
        self.router.call(request).await.unwrap()
    }
}

/// Based on https://github.com/tokio-rs/axum/blob/main/examples/simple-router-wasm/Cargo.toml
#[cfg(test)]
mod tests {
    use axum::{response::Html, routing::get};
    use bytes::Bytes;
    use futures::TryStreamExt;
    use futures_executor::block_on;
    use serde_json::{Map, Value};

    use crate::handlers::sign_in::{basic_auth, create_session_name};

    use super::*;

    async fn index() -> Html<&'static str> {
        Html("<h1>Hello, World!</h1>")
    }

    #[allow(clippy::let_and_return)]
    async fn app_test(request: Request<String>) -> Response {
        let mut router = Router::new().route("/api/", get(index));
        let response = router.call(request).await.unwrap();
        response
    }

    /// Example serverless to make sure dependencies are correct
    #[tokio::test]
    async fn test_app_0_nostd() {
        let request: Request<String> = Request::builder()
            .uri("https://serverless.example/api/")
            .body("Some Body Data".into())
            .unwrap();

        let response: Response = block_on(app_test(request));
        assert_eq!(200, response.status());
    }

    #[tokio::test]
    async fn test_serverless_stream_nostd() {
        // Check sign_in
        let mut server = Serverless::new();

        // Make the credentials with basic authorization
        let credentials = basic_auth("myemail@gmail.com", Some("myemail@gmail.com"));

        // Make the sign_in request
        let request: Request<String> = Request::builder()
            .method("POST")
            .uri("http://127.0.0.1:8000/app/v1/sign_in")
            .header("Content-type", "text/plain; charset=utf-8")
            .header("Authorization", credentials)
            .body("".into())
            .unwrap();
        let response: Response = block_on(server.call(request));
        assert_eq!(200, response.status());

        // Parse the sign_in request results
        let bytes: Vec<Bytes> = response
            .into_body()
            .into_data_stream()
            .try_collect()
            .await
            .unwrap();
        let values: serde_json::Value = serde_json::from_slice(bytes.first().unwrap()).unwrap();

        // Test subjects_info

        // Extract out the JWT token
        let token = values.get("jwt").unwrap().as_str().unwrap();
        let bearer = format!("Bearer {token}");

        // Create the session state JSON value
        let session_name =
            create_session_name(values.get("email").unwrap().as_str().unwrap(), "Chat");
        let mut map = Map::new();
        map.insert(
            "session_name".to_string(),
            Value::String(session_name.clone()),
        );
        map.insert("subject_name".to_string(), Value::String("".to_string()));
        let data = serde_json::to_string(&Value::Object(map)).unwrap();

        // Make the request for the subjects_info
        let request: Request<String> = Request::builder()
            .method("POST")
            .uri("http://127.0.0.1:8000/app/v1/subjects_info")
            .header("Content-type", "application/json")
            .header("Authorization", bearer.as_str())
            .body(data)
            .unwrap();
        let response: Response = block_on(server.call(request));
        assert_eq!(200, response.status());

        // Parse the response for the subjects_info
        let bytes: Vec<Bytes> = response
            .into_body()
            .into_data_stream()
            .try_collect()
            .await
            .unwrap();
        let _values: serde_json::Value = serde_json::from_slice(bytes.first().unwrap()).unwrap();

        // DM: this takes a while to run on WASM...
        // // Test session_stream

        // // Create the session state JSON value
        // let mut map = Map::new();
        // map.insert("session_name".to_string(), Value::String(session_name));
        // map.insert(
        //     "content".to_string(),
        //     Value::String("What is the world's tallest mountain?".to_string()),
        // );
        // map.insert(
        //     "subject_name".to_string(),
        //     Value::String("messages".to_string()),
        // );
        // let data = serde_json::to_string(&Value::Object(map)).unwrap();

        // // Make the request for the chat
        // let request: Request<String> = Request::builder()
        //     .method("POST")
        //     .uri("http://127.0.0.1:8000/app/v1/chat")
        //     .header("Content-type", "application/json")
        //     .header("Authorization", bearer.as_str())
        //     .body(data)
        //     .unwrap();
        // let response: Response = block_on(server.call(request));
        // assert_eq!(200, response.status());

        // // Parse the response for the chat
        // let bytes: Vec<Bytes> = response
        //     .into_body()
        //     .into_data_stream()
        //     .try_collect()
        //     .await
        //     .unwrap();
        // let values: serde_json::Value = serde_json::from_slice(bytes.first().unwrap()).unwrap();
        // println!("{values:?}");
    }

    #[tokio::test]
    async fn test_serverless_cli_nostd() {
        // Check sign_in
        let mut server = Serverless::new();

        // Make the credentials with basic authorization
        let credentials = basic_auth("myemail@gmail.com", Some("myemail@gmail.com"));

        // Make the sign_in request
        let request: Request<String> = Request::builder()
            .method("POST")
            .uri("http://127.0.0.1:8000/app/v1/sign_in")
            .header("Content-type", "text/plain; charset=utf-8")
            .header("Authorization", credentials)
            .body("".into())
            .unwrap();
        let response: Response = block_on(server.call(request));
        assert_eq!(200, response.status());

        // Parse the sign_in request results
        let bytes: Vec<Bytes> = response
            .into_body()
            .into_data_stream()
            .try_collect()
            .await
            .unwrap();
        let values: serde_json::Value = serde_json::from_slice(bytes.first().unwrap()).unwrap();

        // Test subjects_info
        let mut server = Serverless::new();

        // Extract out the JWT token
        let token = values.get("jwt").unwrap().as_str().unwrap();
        let bearer = format!("Bearer {token}");

        // Create the session state JSON value
        let session_name =
            create_session_name(values.get("email").unwrap().as_str().unwrap(), "Chat");
        let mut map = Map::new();
        map.insert(
            "session_name".to_string(),
            Value::String(session_name.clone()),
        );
        map.insert("subject_name".to_string(), Value::String("".to_string()));
        let data = serde_json::to_string(&Value::Object(map)).unwrap();

        // Make the request for the subjects_info
        let request: Request<String> = Request::builder()
            .method("POST")
            .uri("http://127.0.0.1:8000/app/v1/subjects_info")
            .header("Content-type", "application/json")
            .header("Authorization", bearer.as_str())
            .body(data)
            .unwrap();
        let response: Response = block_on(server.call(request));
        assert_eq!(200, response.status());

        // Parse the response for the subjects_info
        let bytes: Vec<Bytes> = response
            .into_body()
            .into_data_stream()
            .try_collect()
            .await
            .unwrap();
        let _values: serde_json::Value = serde_json::from_slice(bytes.first().unwrap()).unwrap();
    }
}
