mod plans;

#[cfg(feature = "api")]
pub use plans::ExecuteWorkspaceNetwork;
#[cfg(feature = "api")]
pub use plans::GetContentNetwork;
pub use plans::{
    EmbedTextNetwork, ExtractOntologyNetwork, ExtractPDFNetwork, GenerateTextNetwork,
    MeltStudyDataNetwork, PatchWorkspaceNetwork, RetrieveTextNetwork, SyncContentNetwork,
    ToolCallNetwork, ToolResponseNetwork,
};
