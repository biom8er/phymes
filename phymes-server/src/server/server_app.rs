// Server related imports
#[allow(unused_imports)]
use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{self, Method},
    middleware,
    routing::{get_service, post},
};
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    limit::RequestBodyLimitLayer,
    services::ServeDir,
    trace::TraceLayer,
};

// General imports
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
use crate::server::server_config::ServerConfig;
#[allow(unused_imports)]
use anyhow::Result;
#[allow(unused_imports)]
use parking_lot::RwLock;
#[allow(unused_imports)]
use std::sync::Arc;
#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
use tokio::net::TcpListener;

// From lib
use crate::{
    handlers::{
        authorize, session_build, session_diagnostics, session_get_state, session_put_state,
        session_stream, sign_in,
    },
    state::{ServerState, UserState},
};

#[derive(Default)]
pub struct AppBuilder {
    pub app: Router,
}

impl AppBuilder {
    pub fn new(user_session_context_name: Option<&str>) -> Self {
        // Application state
        let user_state = UserState::new(user_session_context_name);
        let server_state = ServerState::new();

        // Router
        let app: Router = Router::new()
            .route("/app/v1/sign_in", post(sign_in))
            .route(
                "/app/v1/chat",
                post(session_stream).layer(middleware::from_fn_with_state(
                    user_state.clone(),
                    authorize,
                )),
            )
            .route(
                "/app/v1/put_state",
                post(session_put_state).layer(middleware::from_fn_with_state(
                    user_state.clone(),
                    authorize,
                )),
            )
            .route(
                "/app/v1/get_state",
                post(session_get_state).layer(middleware::from_fn_with_state(
                    user_state.clone(),
                    authorize,
                )),
            )
            .route(
                "/app/v1/build",
                post(session_build).layer(middleware::from_fn_with_state(
                    user_state.clone(),
                    authorize,
                )),
            )
            .route(
                "/app/v1/diagnostics",
                post(session_diagnostics).layer(middleware::from_fn_with_state(
                    user_state.clone(),
                    authorize,
                )),
            )
            .with_state((user_state.clone(), server_state));
        Self { app }
    }

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    fn with_fallback(self, dir: &str) -> Self {
        Self {
            app: self.app.fallback(get_service(ServeDir::new(dir))),
        }
    }

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    fn with_trace_layer(self) -> Self {
        Self {
            app: self.app.layer(TraceLayer::new_for_http()),
        }
    }

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    fn with_cors_layer(self) -> Self {
        // CORS
        let cors_layer = if cfg!(debug_assertions) {
            CorsLayer::permissive()
        } else {
            let allow_origin = AllowOrigin::any();
            CorsLayer::new()
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([http::header::CONTENT_TYPE])
                .allow_origin(allow_origin)
        };
        Self {
            app: self.app.layer(cors_layer),
        }
    }

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    fn with_max_body_limit(self) -> Self {
        use axum::extract::DefaultBodyLimit;
        let limit = RequestBodyLimitLayer::new(1024 * 1024 * 1000); // 1000 MB
        Self {
            app: self.app.layer(DefaultBodyLimit::disable()).layer(limit),
        }
    }

    pub fn build(self) -> Router {
        self.app
    }
}

#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
pub struct Server {
    /// Server configuration
    config: Arc<RwLock<ServerConfig>>,
}

#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
impl Server {
    /// Create a new server from a configuration
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
        }
    }

    /// Run the server
    pub async fn run(&self) -> Result<()> {
        // initialize the front-end
        let frontend = async {
            let app: Router = AppBuilder::new(None)
                .with_fallback(self.config.try_read().unwrap().assets_dir.as_str())
                .with_trace_layer()
                .with_cors_layer()
                .with_max_body_limit()
                .build();

            let address = self.config.try_read().unwrap().address.clone();
            Self::serve(app, address).await;
        };

        tokio::join!(frontend);
        Ok(())
    }

    async fn serve(app: Router, addr: String) {
        tracing::debug!("listening on {}", addr);
        let listener = TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    }
}
