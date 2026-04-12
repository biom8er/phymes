#[cfg(feature = "api")]
mod open_alex_network_builder_mermaid;
#[cfg(feature = "api")]
mod open_alex_network_builder;

#[cfg(feature = "api")]
pub use open_alex_network_builder_mermaid::OpenAlexNetworkBuilderMermaid;

#[cfg(feature = "api")]
pub use open_alex_network_builder::OpenAlexNetworkBuilder;