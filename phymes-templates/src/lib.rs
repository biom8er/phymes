mod dynamic;
mod pipelines;

#[cfg(feature = "api")]
pub use dynamic::GetContentNetwork;
pub use dynamic::{
    GenerateTextNetwork, PatchWorkspaceNetwork, ToolCallNetwork, ToolResponseNetwork,
};

#[cfg(feature = "api")]
pub use pipelines::ExecuteWorkspaceNetwork;
pub use pipelines::{
    EmbedTextNetwork, ExtractOntologyNetwork, ExtractPDFNetwork, MeltStudyDataNetwork, RetrieveTextNetwork, SyncContentNetwork,
};
