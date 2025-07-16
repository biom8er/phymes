// Server related imports
use anyhow::{Result, anyhow};
use axum::{Router, response::Response};
use http::Request;
use tower_service::Service;

// From lib
use super::{server_app::AppBuilder, serverless_config::ServerlessConfig};
use crate::handlers::sign_in::basic_auth;

/// Stateful implementation of the router to enable
/// continuous calls to the router
#[derive(Default, Clone)]
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

/// Wrapper for calling the serverless application
pub async fn serverless_app(
    config: ServerlessConfig,
    serverless: &mut Serverless,
) -> Result<Response> {
    // // initialize the server
    // let mut serverless = Serverless::new();

    // start building the request
    let url = format!("https://serverless/{}", config.route);
    let request_builder = Request::builder().method("POST").uri(url);

    let response = if let Some(credentials) = config.basic_auth {
        // Parse the credentials
        let mid = credentials.find(":");
        if mid.is_none() {
            return Err(anyhow!("Error: unable to parse the basic_auth."));
        }
        let (username, password) = credentials.split_at(mid.unwrap());
        let password = &password[1..];

        // Make the credentials with basic authorization
        let credentials = basic_auth(username, Some(password));

        // build the request
        let request: Request<String> = request_builder
            .header("Content-type", "text/plain; charset=utf-8")
            .header("Authorization", credentials)
            .body("".into())
            .unwrap();
        serverless.call(request).await
    } else if let (Some(bearer), Some(data)) = (config.bearer_auth, config.data) {
        // Make the credentials for bearer authorization
        let bearer = format!("Bearer {bearer}");

        // build the request
        let request: Request<String> = request_builder
            .header("Content-type", "application/json")
            .header("Authorization", bearer)
            .body(data)
            .unwrap();
        serverless.call(request).await
    } else {
        return Err(anyhow!(
            "Error: no basic_auth nor bearer_auth with data were provided."
        ));
    };

    Ok(response)
}

#[cfg(test)]
mod tests {
    use axum::{response::Html, routing::get};
    use bytes::Bytes;
    use futures::TryStreamExt;
    use futures_executor::block_on;
    use phymes_core::table::arrow_table_publish::ArrowTablePublish;
    use serde_json::{Map, Value};

    use crate::handlers::{
        session_info::{SessionResponse, SessionResponseFormat},
        sign_in::{basic_auth, create_session_name},
    };

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
    async fn test_serverless_nostd() {
        let request: Request<String> = Request::builder()
            .uri("https://serverless.example/api/")
            .body("Some Body Data".into())
            .unwrap();

        let response: Response = block_on(app_test(request));
        assert_eq!(200, response.status());
    }

    #[tokio::test]
    async fn test_serverless_call() {
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
        let response: Response = server.call(request).await;
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
        let session_response = SessionResponse {
            session_name: session_name.clone(),
            subject_name: "".to_string(),
            format: SessionResponseFormat::Bytes,
            publish: ArrowTablePublish::None,
            content: "".to_string(),
            metadata: "".to_string(),
            stream: false,
        };
        let data = serde_json::to_string(&session_response).unwrap();

        // Make the request for the subjects_info
        let request: Request<String> = Request::builder()
            .method("POST")
            .uri("http://127.0.0.1:8000/app/v1/subjects_info")
            .header("Content-type", "application/json")
            .header("Authorization", bearer.as_str())
            .body(data)
            .unwrap();
        let response: Response = server.call(request).await;
        assert_eq!(200, response.status());

        // Parse the response for the subjects_info
        let bytes: Vec<Bytes> = response
            .into_body()
            .into_data_stream()
            .try_collect()
            .await
            .unwrap();
        let _values: serde_json::Value = serde_json::from_slice(bytes.first().unwrap()).unwrap();

        // DM: omitted to reduce test time
        // // Test session_stream
        // let mut server = Serverless::new();

        // // Create the session state JSON value
        // let session_response = SessionResponse {
        //     session_name: session_name.clone(),
        //     subject_name: "messages".to_string(),
        //     format: SessionResponseFormat::Bytes,
        //     publish: ArrowTablePublish::None,
        //     content: "What is the world's tallest mountain?".to_string(),
        //     metadata: "".to_string(),
        //     stream: true
        // };
        // let data = serde_json::to_string(&session_response).unwrap();

        // // Make the request for the chat
        // let request: Request<String> = Request::builder()
        //     .method("POST")
        //     .uri("http://127.0.0.1:8000/app/v1/chat")
        //     .header("Content-type", "application/json")
        //     .header("Authorization", bearer.as_str())
        //     .body(data)
        //     .unwrap();
        // let response: Response = server.call(request).await;
        // assert_eq!(200, response.status());

        // // Parse the response for the chat
        // let bytes: Vec<Bytes> = response
        //     .into_body()
        //     .into_data_stream()
        //     .try_collect()
        //     .await
        //     .unwrap();
        // let values: Vec<Map<String, Value>> = serde_json::from_slice(bytes.first().unwrap()).unwrap();
        // println!("{values:?}");
    }

    #[tokio::test]
    async fn test_serverless_app() {
        let mut serverless = Serverless::new();

        // Sign in using serverless_app
        let config = ServerlessConfig {
            route: "app/v1/sign_in".to_string(),
            basic_auth: Some("myemail@gmail.com:myemail@gmail.com".to_string()),
            bearer_auth: None,
            data: None,
        };
        let response = serverless_app(config, &mut serverless).await.unwrap();
        assert_eq!(200, response.status());

        // Parse the sign_in request results
        let bytes: Vec<Bytes> = response
            .into_body()
            .into_data_stream()
            .try_collect()
            .await
            .unwrap();
        let values: serde_json::Value = serde_json::from_slice(bytes.first().unwrap()).unwrap();

        // Test subjects_info using serverless_app
        let token = values.get("jwt").unwrap().as_str().unwrap();
        let bearer = token.to_string();
        let session_name =
            create_session_name(values.get("email").unwrap().as_str().unwrap(), "Chat");
        let session_response = SessionResponse {
            session_name: session_name.clone(),
            subject_name: "".to_string(),
            format: SessionResponseFormat::Bytes,
            publish: ArrowTablePublish::None,
            content: "".to_string(),
            metadata: "".to_string(),
            stream: false,
        };
        let data = serde_json::to_string(&session_response).unwrap();

        let config = ServerlessConfig {
            route: "app/v1/subjects_info".to_string(),
            basic_auth: None,
            bearer_auth: Some(bearer.clone()),
            data: Some(data),
        };
        let response = serverless_app(config, &mut serverless).await.unwrap();
        assert_eq!(200, response.status());

        let bytes: Vec<Bytes> = response
            .into_body()
            .into_data_stream()
            .try_collect()
            .await
            .unwrap();
        let _values: serde_json::Value = serde_json::from_slice(bytes.first().unwrap()).unwrap();

        // Test session_stream using serverless_app
        let session_response = SessionResponse {
            session_name: session_name.clone(),
            subject_name: "messages".to_string(),
            format: SessionResponseFormat::Bytes,
            publish: ArrowTablePublish::None,
            content: "What is the world's tallest mountain?".to_string(),
            metadata: "".to_string(),
            stream: true,
        };
        let data = serde_json::to_string(&session_response).unwrap();

        let config = ServerlessConfig {
            route: "app/v1/chat".to_string(),
            basic_auth: None,
            bearer_auth: Some(bearer),
            data: Some(data),
        };
        let response = serverless_app(config, &mut serverless).await.unwrap();
        assert_eq!(200, response.status());

        let bytes: Vec<Bytes> = response
            .into_body()
            .into_data_stream()
            .try_collect()
            .await
            .unwrap();
        let values: Vec<Map<String, Value>> =
            serde_json::from_slice(bytes.first().unwrap()).unwrap();
        println!("{values:?}");
    }
}
