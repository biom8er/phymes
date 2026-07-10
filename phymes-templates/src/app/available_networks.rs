use std::{fmt::Display, sync::Arc};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use phymes_message::IPCMessageMap;
use phymes_network::{
    DynamicTaskNetworkNames, Network, NetworkBuilder, NetworkBuilderAppsTrait,
    NetworkBuilderCustomTrait, NetworkBuilderMermaidTrait, NetworkBuilderTrait,
};
use phymes_subject::{BuildableTrait, BuilderTrait, RuntimeEnv, RuntimeEnvBuilderTrait};
use serde::{Deserialize, Serialize};

use crate::{
    GenerateTextNetworkBuilder, MermaidNetworkBuilder, RetrievalAugmentedGenerationPDFNetworkBuilder, TabularDataOperatorNetworkBuilder, UserNetwork,
};
#[cfg(feature = "api")]
use crate::GenerateCodeNetworkBuilder;

/// The available network plans
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum AvailableNetworks {
    #[value(name = "GenerateText")]
    GenerateText,
    #[value(name = "RAGTextPDF")]
    RAGTextPDF,
    #[value(name = "TabularDataOps")]
    TabularDataOps,
    #[cfg(feature = "api")]
    #[value(name = "GenerateCode")]
    GenerateCode,
    #[value(name = "Builder")]
    Builder,
    #[value(name = "Users")]
    Users,
}

impl Display for AvailableNetworks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GenerateText => write!(f, "GenerateText"),
            Self::RAGTextPDF => write!(f, "RAGTextPDF"),
            Self::TabularDataOps => write!(f, "TabularDataOps"),
            #[cfg(feature = "api")]
            Self::GenerateCode => write!(f, "GenerateCode"),
            Self::Builder => write!(f, "Builder"),
            Self::Users => write!(f, "Users"),
        }
    }
}

impl AvailableNetworks {
    /// Get all available network plans
    pub fn get_all_network_plan_names() -> Vec<String> {
        let network_plans = ["GenerateText", "RAGTextPDF", "TabularDataOps", 
            #[cfg(feature = "api")]
            "GenerateCode", 
            "Builder"
        ];
        network_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get all available network plans
    pub fn get_deployable_network_plan_names() -> Vec<String> {
        let network_plans = ["GenerateText", "RAGTextPDF", "TabularDataOps", 
            #[cfg(feature = "api")]
            "GenerateCode"
        ];
        network_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get the network stream state
    pub fn get_network_builder(&self, network_name: &str) -> NetworkBuilder {
        // Initialize the network context builder
        match self {
            Self::GenerateText => {
                let generate_text_network = GenerateTextNetworkBuilder::default();
                NetworkBuilder::from_mermaid_flowchart(
                    &generate_text_network.as_mermaid_flowchart(),
                    false,
                )
                .unwrap()
                .with_subjects_from_mermaid_erdiagram(
                    &generate_text_network.as_mermaid_erdiagram(),
                    false,
                    true,
                )
                .unwrap()
                .with_name(network_name)
                .add_processor_subjects()
                .unwrap()
            }
            Self::RAGTextPDF => RetrievalAugmentedGenerationPDFNetworkBuilder::default()
                .inner
                .take()
                .unwrap()
                // DM: will be overwritten anyway
                .with_runtime_env(
                    RuntimeEnv::get_builder()
                        .with_name(
                            DynamicTaskNetworkNames::RuntimeEnv(network_name)
                                .to_string()
                                .as_str(),
                        )
                        .with_max_steps(100)
                        .build_arc()
                        .unwrap(),
                )
                .with_name(network_name)
                .add_processor_subjects()
                .unwrap(),
            Self::TabularDataOps => TabularDataOperatorNetworkBuilder::default()
                .inner
                .take()
                .unwrap()
                // DM: will be overwritten anyway
                .with_runtime_env(
                    RuntimeEnv::get_builder()
                        .with_name(
                            DynamicTaskNetworkNames::RuntimeEnv(network_name)
                                .to_string()
                                .as_str(),
                        )
                        .with_max_steps(100)
                        .build_arc()
                        .unwrap(),
                )
                .with_name(network_name)
                .add_processor_subjects()
                .unwrap(),
            #[cfg(feature = "api")]
            Self::GenerateCode => {
                let generate_code_network_builder = GenerateCodeNetworkBuilder::default()
                    .inner
                    .take()
                    .unwrap();

                // Extend with execute_workspace_network
                let execute_workspace_network = ExecuteWorkspaceNetwork::new(
                    "execute_workspace_network_py",
                    None,
                    Some(subject_name_i),
                    subject_name_o,
                    &CommandSandboxEnvironments::Python,
                );
                let network_builder = NetworkBuilder::from_mermaid_flowchart(
                    &execute_workspace_network.as_mermaid_flowchart(),
                    false,
                )
                .unwrap()
                .with_subjects_from_mermaid_erdiagram(
                    &execute_workspace_network.as_mermaid_erdiagram().unwrap(),
                    false,
                    true,
                )
                .unwrap()
                .with_name(execute_workspace_network.network_name);
                let generate_code_network_builder = generate_code_network_builder.extend(network_builder).unwrap();
                
                generate_code_network_builder.with_runtime_env(
                    RuntimeEnv::get_builder()
                        .with_name(
                            DynamicTaskNetworkNames::RuntimeEnv(network_name)
                                .to_string()
                                .as_str(),
                        )
                        .with_max_steps(100)
                        .build_arc()
                        .unwrap(),
                )
                .with_name(network_name)
                .add_processor_subjects()
                .unwrap()
            },
            Self::Builder => MermaidNetworkBuilder::new_with_network_name(network_name).build(),
            Self::Users => UserNetwork::new_with_network_name(network_name).build(),
        }
    }

    /// Get the network stream state by name
    pub fn get_network_builder_by_name(
        network_plan_name: &str,
        network_name: &str,
    ) -> Result<NetworkBuilder> {
        if network_plan_name == Self::GenerateText.to_string() {
            Ok(Self::GenerateText.get_network_builder(network_name))
        } else if network_plan_name == Self::RAGTextPDF.to_string() {
            Ok(Self::RAGTextPDF.get_network_builder(network_name))
        } else if network_plan_name == Self::TabularDataOps.to_string() {
            Ok(Self::TabularDataOps.get_network_builder(network_name))
        } else if network_plan_name == Self::Builder.to_string() {
            Ok(Self::Builder.get_network_builder(network_name))
        } else if network_plan_name == Self::Users.to_string() {
            Ok(Self::Users.get_network_builder(network_name))
        } else {
            #[cfg(feature = "api")]
            if network_plan_name == Self::GenerateCode.to_string() {
                Ok(Self::GenerateCode.get_network_builder(network_name))
            } else {
                Err(anyhow!(
                    "Plan name {network_plan_name} was not found in the available network plans."
                ))
            }
            #[cfg(not(feature = "api"))]
            Err(anyhow!(
                "Plan name {network_plan_name} was not found in the available network plans."
            ))
        }
    }

    /// Get the network stream state
    pub fn get_network_stream_state(
        &self,
        network_name: &str,
        runtime_env: &Arc<RuntimeEnv>,
    ) -> (Arc<Network>, Option<IPCMessageMap>) {
        // Initialize the network
        let builder = self.get_network_builder(network_name);
        let (network, message) = builder
            .with_name(network_name)
            .with_runtime_env(Arc::clone(runtime_env))
            .with_diagnostics(true)
            .add_network_interface(None)
            .unwrap()
            .add_next_tasks()
            .unwrap()
            .add_next_supersteps()
            .unwrap()
            .build_with_tables()
            .unwrap();
        (Arc::new(network), message)
    }

    /// Get the network stream state by name
    pub fn get_network_stream_state_by_name(
        network_plan_name: &str,
        network_name: &str,
        runtime_env: &Arc<RuntimeEnv>,
    ) -> Result<(Arc<Network>, Option<IPCMessageMap>)> {
        if network_plan_name == Self::GenerateText.to_string() {
            Ok(Self::GenerateText.get_network_stream_state(network_name, runtime_env))
        } else if network_plan_name == Self::RAGTextPDF.to_string() {
            Ok(Self::RAGTextPDF.get_network_stream_state(network_name, runtime_env))
        } else if network_plan_name == Self::TabularDataOps.to_string() {
            Ok(Self::TabularDataOps.get_network_stream_state(network_name, runtime_env))
        } else if network_plan_name == Self::Builder.to_string() {
            Ok(Self::Builder.get_network_stream_state(network_name, runtime_env))
        } else if network_plan_name == Self::Users.to_string() {
            Ok(Self::Users.get_network_stream_state(network_name, runtime_env))
        } else {
            #[cfg(feature = "api")]
            if network_plan_name == Self::GenerateCode.to_string() {
                Ok(Self::GenerateCode.get_network_stream_state(network_name, runtime_env))
            } else {
                Err(anyhow!(
                    "Plan name {network_plan_name} was not found in the available network plans."
                ))
            }
            #[cfg(not(feature = "api"))]
            Err(anyhow!(
                "Plan name {network_plan_name} was not found in the available network plans."
            ))
        }
    }
}
