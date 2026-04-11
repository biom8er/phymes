mod dynamic;
mod pipelines;

#[cfg(feature = "api")]
pub use dynamic::ExecuteWorkspaceNetwork;
pub use dynamic::{
    DynamicNetworkBuilderTrait, DynamicTaskNetworkBuilder, DynamicTaskNetworkNames, GenerateTextNetworkBuilder, InvokeTaskNetworkBuilder, TaskResponseNetworkBuilder,
    GetJsonNetworkBuilderDynamicWOSubject, GetJsonNetworkBuilderDynamicWSubject, GetJsonNetworkBuilderStaticWSubject, GetPdfNetworkBuilderDynamicWOSubject, GetPdfNetworkBuilderDynamicWSubject, GetPdfNetworkBuilderStaticWSubject,
    PatchWorkspaceNetworkBuilderDynamicWOSubject, PatchWorkspaceNetworkBuilderDynamicWSubject, PatchWorkspaceNetworkBuilderStaticWSubject
};

pub use pipelines::{
    EmbedTextNetworkBuilder, ExtractOntologyNetworkBuilder, ExtractPDFNetworkBuilder, MeltStudyDataNetworkBuilder, RetrieveTextNetworkBuilder, SyncContentNetworkBuilder,
};
