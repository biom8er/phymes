mod app;
mod composites;
mod diagnostics;
mod dynamic;
mod pipelines;

pub use app::{
    AvailableNetworks, MermaidNetworkBuilder, DiagnosticNetworkBuilder, UserNetwork, make_example_mermaid_table,
};
pub use composites::{RetrievalAugmentedGenerationPDFNetworkBuilder, RetrieveTextPDFNetworkBuilder, TabularDataOperatorNetworkBuilder};
#[cfg(feature = "api")]
pub use composites::{GenerateCodeNetworkBuilder, OpenAlexNetworkBuilder};
pub use diagnostics::{
    default_diagnostic_subjects, extended_diagnostic_subjects, write_diagnostic_subjects_to_csv,
};
pub use dynamic::{
    DiffWorkspaceNetworkBuilderDynamicWOSubject, DiffWorkspaceNetworkBuilderStaticWSubject, GenerateTextNetworkBuilder, 
    PatchWorkspaceNetworkBuilderDynamicWOSubject, PatchWorkspaceNetworkBuilderDynamicWSubject, PatchWorkspaceNetworkBuilderStaticWSubject,
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
    AttachmentsNetworkBuilder, EmbedTextNetworkBuilder, ExtractOntologyNetworkBuilder, ExtractPDFNetworkBuilder,
    MeltStudyDataNetworkBuilder, RetrieveTextNetworkBuilder, SyncContentNetworkBuilder,
};
