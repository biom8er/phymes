mod app;
mod composites;
mod dynamic;
mod pipelines;

pub use app::{
    AvailableNetworks, DiagnosticNetworkBuilder, MermaidNetworkBuilder, UserNetwork,
    make_example_mermaid_table,
};
#[cfg(feature = "api")]
pub use composites::{GenerateCodeNetworkBuilder, OpenAlexNetworkBuilder};
pub use composites::{
    RetrievalAugmentedGenerationPDFNetworkBuilder, RetrieveTextPDFNetworkBuilder,
    TabularDataOperatorNetworkBuilder,
};
pub use dynamic::{
    DiffWorkspaceNetworkBuilderDynamicWOSubject, DiffWorkspaceNetworkBuilderStaticWSubject,
    GenerateTextNetworkBuilder, PatchWorkspaceNetworkBuilderDynamicWOSubject,
    PatchWorkspaceNetworkBuilderDynamicWSubject, PatchWorkspaceNetworkBuilderStaticWSubject,
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
    AttachmentsNetworkBuilder, EmbedTextNetworkBuilder, ExtractOntologyNetworkBuilder,
    ExtractPDFNetworkBuilder, MeltStudyDataNetworkBuilder, RetrieveTextNetworkBuilder,
    SyncContentNetworkBuilder,
};
