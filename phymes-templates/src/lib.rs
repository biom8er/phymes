mod composites;
mod diagnostics;
mod dynamic;
mod pipelines;

#[cfg(feature = "api")]
pub use composites::{GenerateCodeNetworkBuilder, OpenAlexNetworkBuilder};
pub use diagnostics::{
    default_diagnostic_subjects, extended_diagnostic_subjects, write_diagnostic_subjects_to_csv,
};
pub use dynamic::{
    DynamicNetworkBuilderTrait, DynamicTaskNetworkBuilder, DynamicTaskNetworkNames,
    GenerateTextNetworkBuilder, InvokeTaskNetworkBuilder,
    PatchWorkspaceNetworkBuilderDynamicWOSubject, PatchWorkspaceNetworkBuilderDynamicWSubject,
    PatchWorkspaceNetworkBuilderStaticWSubject, TaskResponseNetworkBuilder,
};
#[cfg(feature = "api")]
pub use dynamic::{
    ExecuteWorkspaceNetwork, GetJsonNetworkBuilderDynamicWOSubject,
    GetJsonNetworkBuilderDynamicWSubject, GetJsonNetworkBuilderStaticWSubject,
    GetObjectNetworkBuilderDynamicWOSubject, GetObjectNetworkBuilderDynamicWSubject,
    GetObjectNetworkBuilderStaticWSubject, GetPdfNetworkBuilderDynamicWOSubject,
    GetPdfNetworkBuilderDynamicWSubject, GetPdfNetworkBuilderStaticWSubject,
};

pub use pipelines::{
    EmbedTextNetworkBuilder, ExtractOntologyNetworkBuilder, ExtractPDFNetworkBuilder,
    MeltStudyDataNetworkBuilder, RetrieveTextNetworkBuilder, SyncContentNetworkBuilder,
};
