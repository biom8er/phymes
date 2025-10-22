mod server_app;
mod server_config;
#[cfg(feature = "wasip2")]
mod serverless_app;
#[cfg(feature = "wasip2")]
mod serverless_config;

#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
pub use server_app::Server;
pub use server_app::AppBuilder;
pub use server_config::ServerConfig;
#[cfg(feature = "wasip2")]
pub use serverless_app::{Serverless, serverless_app};
#[cfg(feature = "wasip2")]
pub use serverless_config::ServerlessConfig;