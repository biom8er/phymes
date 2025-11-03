use std::fmt::Display;

use anyhow::{anyhow, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::{MappableTrait, TableSubscribePolicyTrait, table::{AllTableNamesSubscribe, AllTableSchemasSubscribe, AlwaysSubscribe, AnyTableNameSubscribe, AnyTableSchemaSubscribe, ChatContentSubscribe}};

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableTableSubscribePolicies {
    #[value(name = "AlwaysSubscribe")]
    AlwaysSubscribe,
    #[value(name = "AnyTableNameSubscribe")]
    AnyTableNameSubscribe,
    #[default]
    #[value(name = "AllTableNamesSubscribe")]
    AllTableNamesSubscribe,
    #[value(name = "AnyTableSchemaSubscribe")]
    AnyTableSchemaSubscribe,
    #[value(name = "AllTableSchemasSubscribe")]
    AllTableSchemasSubscribe,
    #[value(name = "ChatContentSubscribe")]
    ChatContentSubscribe,
}

impl AvailableTableSubscribePolicies {
    pub fn build(self) -> Box<impl TableSubscribePolicyTrait> {
        match self {
            Self::AlwaysSubscribe => AlwaysSubscribe::new_box(),
            Self::AnyTableNameSubscribe => AnyTableNameSubscribe::new_box(),
            Self::AllTableNamesSubscribe => AllTableNamesSubscribe::new_box(),
            Self::AnyTableSchemaSubscribe => AnyTableSchemaSubscribe::new_box(),
            Self::AllTableSchemasSubscribe => AllTableSchemasSubscribe::new_box(),
            Self::ChatContentSubscribe => ChatContentSubscribe::new_box(),
        }
    }
    /// Convert a [String] to a [TableSubscribePolicyTrait]
    ///   by checking if the [String] contains the [TableSubscribePolicyTrait] name
    pub fn from_str_fuzzy(policy: &str) -> Result<Self> {
        let subscribe = if policy.contains(AllTableSchemasSubscribe::get_static_name()) {
            AvailableTableSubscribePolicies::AllTableSchemasSubscribe
        } else if policy.contains(AnyTableSchemaSubscribe::get_static_name()) {
            AvailableTableSubscribePolicies::AnyTableSchemaSubscribe
        } else if policy.contains(AllTableNamesSubscribe::get_static_name()) {
            AvailableTableSubscribePolicies::AllTableNamesSubscribe
        } else if policy.contains(AnyTableNameSubscribe::get_static_name()) {
            AvailableTableSubscribePolicies::AnyTableNameSubscribe
        } else if policy.contains(AlwaysSubscribe::get_static_name()) {
            AvailableTableSubscribePolicies::AlwaysSubscribe
        } else if policy.contains(ChatContentSubscribe::get_static_name()) {
            AvailableTableSubscribePolicies::ChatContentSubscribe
        } else {
            return Err(anyhow!("Subscribe policy {policy} was not recognized."));
        };
        Ok(subscribe)
    }
}

impl Display for AvailableTableSubscribePolicies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlwaysSubscribe => write!(f, "{}", AlwaysSubscribe::get_static_name()),
            Self::AnyTableNameSubscribe => write!(f, "{}", AnyTableNameSubscribe::get_static_name()),
            Self::AllTableNamesSubscribe => write!(f, "{}", AllTableNamesSubscribe::get_static_name()),
            Self::AnyTableSchemaSubscribe => write!(f, "{}", AnyTableSchemaSubscribe::get_static_name()),
            Self::AllTableSchemasSubscribe => write!(f, "{}", AllTableSchemasSubscribe::get_static_name()),
            Self::ChatContentSubscribe => write!(f, "{}", ChatContentSubscribe::get_static_name()),
        }
    }
}