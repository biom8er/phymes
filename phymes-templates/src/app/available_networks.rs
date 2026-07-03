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

/// The available session plans
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum AvailableNetworks {
    #[value(name = "GenerateText")]
    GenerateText,
    #[value(name = "RAGTextPDF")]
    RAGTextPDF,
    #[value(name = "TabularDataOps")]
    TabularDataOps,
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
            Self::Builder => write!(f, "Builder"),
            Self::Users => write!(f, "Users"),
        }
    }
}

impl AvailableNetworks {
    /// Get all available session plans
    pub fn get_all_session_plan_names() -> Vec<String> {
        let session_plans = ["GenerateText", "RAGTextPDF", "TabularDataOps", "Builder"];
        session_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get all available session plans
    pub fn get_deployable_session_plan_names() -> Vec<String> {
        let session_plans = ["GenerateText", "RAGTextPDF", "TabularDataOps"];
        session_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get the session stream state
    pub fn get_network_builder(&self, session_name: &str) -> NetworkBuilder {
        // Initialize the session context builder
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
                .with_name(session_name)
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
                            DynamicTaskNetworkNames::RuntimeEnv(session_name)
                                .to_string()
                                .as_str(),
                        )
                        .with_max_steps(100)
                        .build_arc()
                        .unwrap(),
                )
                .with_name(session_name)
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
                            DynamicTaskNetworkNames::RuntimeEnv(session_name)
                                .to_string()
                                .as_str(),
                        )
                        .with_max_steps(100)
                        .build_arc()
                        .unwrap(),
                )
                .with_name(session_name)
                .add_processor_subjects()
                .unwrap(),
            Self::Builder => MermaidNetworkBuilder::new_with_network_name(session_name).build(),
            Self::Users => UserNetwork::new_with_network_name(session_name).build(),
        }
    }

    /// Get the session stream state by name
    pub fn get_network_builder_by_name(
        session_plan_name: &str,
        session_name: &str,
    ) -> Result<NetworkBuilder> {
        if session_plan_name == Self::GenerateText.to_string() {
            Ok(Self::GenerateText.get_network_builder(session_name))
        } else if session_plan_name == Self::RAGTextPDF.to_string() {
            Ok(Self::RAGTextPDF.get_network_builder(session_name))
        } else if session_plan_name == Self::TabularDataOps.to_string() {
            Ok(Self::TabularDataOps.get_network_builder(session_name))
        } else if session_plan_name == Self::Builder.to_string() {
            Ok(Self::Builder.get_network_builder(session_name))
        } else if session_plan_name == Self::Users.to_string() {
            Ok(Self::Users.get_network_builder(session_name))
        } else {
            Err(anyhow!(
                "Plan name {session_plan_name} was not found in the available session plans."
            ))
        }
    }

    /// Get the session stream state
    pub fn get_network_stream_state(
        &self,
        session_name: &str,
        runtime_env: &Arc<RuntimeEnv>,
    ) -> (Arc<Network>, Option<IPCMessageMap>) {
        // Initialize the session
        let builder = self.get_network_builder(session_name);
        let (network, message) = builder
            .with_name(session_name)
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

    /// Get the session stream state by name
    pub fn get_network_stream_state_by_name(
        session_plan_name: &str,
        session_name: &str,
        runtime_env: &Arc<RuntimeEnv>,
    ) -> Result<(Arc<Network>, Option<IPCMessageMap>)> {
        if session_plan_name == Self::GenerateText.to_string() {
            Ok(Self::GenerateText.get_network_stream_state(session_name, runtime_env))
        } else if session_plan_name == Self::RAGTextPDF.to_string() {
            Ok(Self::RAGTextPDF.get_network_stream_state(session_name, runtime_env))
        } else if session_plan_name == Self::TabularDataOps.to_string() {
            Ok(Self::TabularDataOps.get_network_stream_state(session_name, runtime_env))
        } else if session_plan_name == Self::Builder.to_string() {
            Ok(Self::Builder.get_network_stream_state(session_name, runtime_env))
        } else if session_plan_name == Self::Users.to_string() {
            Ok(Self::Users.get_network_stream_state(session_name, runtime_env))
        } else {
            Err(anyhow!(
                "Plan name {session_plan_name} was not found in the available session plans."
            ))
        }
    }
}
