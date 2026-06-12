#[cfg(feature = "api")]
mod execute_workspace_network_builder;
mod generate_text_network_builder;
#[cfg(feature = "api")]
mod get_json_network_builder;
#[cfg(feature = "api")]
mod get_object_network_builder;
#[cfg(feature = "api")]
mod get_pdf_network_builder;
mod patch_workspace_network_builder;

#[cfg(feature = "api")]
pub use execute_workspace_network_builder::ExecuteWorkspaceNetwork;
pub use generate_text_network_builder::GenerateTextNetworkBuilder;
#[cfg(feature = "api")]
pub use get_json_network_builder::{
    GetJsonNetworkBuilderDynamicWOSubject, GetJsonNetworkBuilderDynamicWSubject,
    GetJsonNetworkBuilderStaticWSubject,
};
#[cfg(feature = "api")]
pub use get_object_network_builder::{
    GetObjectNetworkBuilderDynamicWOSubject, GetObjectNetworkBuilderDynamicWSubject,
    GetObjectNetworkBuilderStaticWSubject,
};
#[cfg(feature = "api")]
pub use get_pdf_network_builder::{
    GetPdfNetworkBuilderDynamicWOSubject, GetPdfNetworkBuilderDynamicWSubject,
    GetPdfNetworkBuilderStaticWSubject,
};
pub use patch_workspace_network_builder::{
    PatchWorkspaceNetworkBuilderDynamicWOSubject, PatchWorkspaceNetworkBuilderDynamicWSubject,
    PatchWorkspaceNetworkBuilderStaticWSubject,
};
