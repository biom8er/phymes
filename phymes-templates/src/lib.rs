mod dynamic;
mod pipelines;

#[cfg(feature = "api")]
pub use dynamic::GetContentNetwork;
pub use dynamic::{
    DynamicTaskNetwork, GenerateTextNetwork, PatchWorkspaceNetwork, ToolCallNetwork, TaskResponseNetwork,
};

#[cfg(feature = "api")]
pub use pipelines::ExecuteWorkspaceNetwork;
pub use pipelines::{
    EmbedTextNetwork, ExtractOntologyNetwork, ExtractPDFNetwork, MeltStudyDataNetwork, RetrieveTextNetwork, SyncContentNetwork,
};
