mod network_trait;
mod network_builder;
mod network_builder_apps;
mod network_builder_mermaid;
mod network_builder_tabular;

pub use network_trait::Network;
pub use network_builder::{
    NetworkBuilder, NetworkBuilderTrait, test_network_builder,
};
pub use network_builder_apps::{
    NetworkBuilderCustomTrait, NetworkBuilderAppsTrait, test_network_builder_apps,
};
pub use network_builder_mermaid::{
    NetworkBuilderMermaid, NetworkBuilderMermaidTrait,
};
pub use network_builder_tabular::NetworkBuilderTabularTrait;
