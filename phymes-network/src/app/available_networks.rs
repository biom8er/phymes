use std::{fmt::Display, sync::Arc};

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use phymes_subject::{BuilderTrait, RuntimeEnv};
use phymes_message::IPCMessageMap;
use serde::{Deserialize, Serialize};

use crate::{
    BuilderNetwork, ChatAgentNetwork, CustomAgentsBuilderTrait, DocumentRAGNetwork, Network,
    NetworkBuilder, NetworkBuilderAgentsTrait, NetworkBuilderTrait,
    ToolAgentNetwork, UserNetwork,
};

/// The available session plans
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum AvailableNetworks {
    #[value(name = "Chat")]
    Chat,
    #[value(name = "DocChat")]
    DocChat,
    #[value(name = "ToolChat")]
    ToolChat,
    #[value(name = "Builder")]
    Builder,
    #[value(name = "Users")]
    Users,
}

impl Display for AvailableNetworks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "Chat"),
            Self::DocChat => write!(f, "DocChat"),
            Self::ToolChat => write!(f, "ToolChat"),
            Self::Builder => write!(f, "Builder"),
            Self::Users => write!(f, "Users"),
        }
    }
}

impl AvailableNetworks {
    /// Get all available session plans
    pub fn get_all_session_plan_names() -> Vec<String> {
        let session_plans = ["Chat", "DocChat", "ToolChat", "Builder"];
        // let session_plans = ["Chat", "Builder"];
        session_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get all available session plans
    pub fn get_deployable_session_plan_names() -> Vec<String> {
        let session_plans = ["Chat", "DocChat", "ToolChat"];
        session_plans
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    }

    /// Get the session stream state
    pub fn get_network_builder(&self, session_name: &str) -> NetworkBuilder {
        // Initialize the session context builder
        match self {
            Self::Chat => ChatAgentNetwork::new_with_network_name(session_name).build(),
            Self::DocChat => DocumentRAGNetwork::new_with_network_name(session_name).build(),
            Self::ToolChat => ToolAgentNetwork::new_with_network_name(session_name).build(),
            Self::Builder => BuilderNetwork::new_with_network_name(session_name).build(),
            Self::Users => UserNetwork::new_with_network_name(session_name).build(),
        }
    }

    /// Get the session stream state by name
    pub fn get_network_builder_by_name(
        session_plan_name: &str,
        session_name: &str,
    ) -> Result<NetworkBuilder> {
        if session_plan_name == Self::Chat.to_string() {
            Ok(Self::Chat.get_network_builder(session_name))
        } else if session_plan_name == Self::DocChat.to_string() {
            Ok(Self::DocChat.get_network_builder(session_name))
        } else if session_plan_name == Self::ToolChat.to_string() {
            Ok(Self::ToolChat.get_network_builder(session_name))
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
            .add_session_interface(None)
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
        if session_plan_name == Self::Chat.to_string() {
            Ok(Self::Chat.get_network_stream_state(session_name, runtime_env))
        } else if session_plan_name == Self::DocChat.to_string() {
            Ok(Self::DocChat.get_network_stream_state(session_name, runtime_env))
        } else if session_plan_name == Self::ToolChat.to_string() {
            Ok(Self::ToolChat.get_network_stream_state(session_name, runtime_env))
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
