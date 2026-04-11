mod dynamic_network_builder;
mod dynamic_task_network_builder;
#[cfg(feature = "api")]
mod execute_workspace_network_builder;
mod generate_text_network_builder;
#[cfg(feature = "api")]
mod get_json_network_builder;
#[cfg(feature = "api")]
mod get_object_network_builder;
#[cfg(feature = "api")]
mod get_pdf_network_builder;
mod invoke_task_network_builder;
mod patch_workspace_network_builder;
mod task_response_network_builder;

pub use dynamic_network_builder::DynamicNetworkBuilderTrait;
pub use dynamic_task_network_builder::{DynamicTaskNetworkBuilder, DynamicTaskNetworkNames};
#[cfg(feature = "api")]
pub use execute_workspace_network_builder::ExecuteWorkspaceNetwork;
pub use generate_text_network_builder::GenerateTextNetworkBuilder;
#[cfg(feature = "api")]
pub use get_json_network_builder::{
    GetJsonNetworkBuilderDynamicWOSubject, GetJsonNetworkBuilderDynamicWSubject,
    GetJsonNetworkBuilderStaticWSubject,
};
#[cfg(feature = "api")]
pub use get_pdf_network_builder::{
    GetPdfNetworkBuilderDynamicWOSubject, GetPdfNetworkBuilderDynamicWSubject,
    GetPdfNetworkBuilderStaticWSubject,
};
#[cfg(feature = "api")]
pub use get_object_network_builder::{
    GetObjectNetworkBuilderDynamicWOSubject, GetObjectNetworkBuilderDynamicWSubject,
    GetObjectNetworkBuilderStaticWSubject,
};
pub use invoke_task_network_builder::InvokeTaskNetworkBuilder;
pub use patch_workspace_network_builder::{
    PatchWorkspaceNetworkBuilderDynamicWOSubject, PatchWorkspaceNetworkBuilderDynamicWSubject,
    PatchWorkspaceNetworkBuilderStaticWSubject,
};
pub use task_response_network_builder::TaskResponseNetworkBuilder;
