use std::fmt::Display;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::{
    MappableTrait, TableChangedSinceLastRunUpdate, TableExistsUpdate, TableHasBatchesUpdate,
    TableUpdatePolicyTrait,
};

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableTableUpdatePolicies {
    #[value(name = "TableHasBatchesUpdate")]
    TableHasBatchesUpdate,
    #[default]
    #[value(name = "TableChangedSinceLastRunUpdate")]
    TableChangedSinceLastRunUpdate,
    #[value(name = "TableExistsUpdate")]
    TableExistsUpdate,
}

impl AvailableTableUpdatePolicies {
    pub fn build(self) -> Box<dyn TableUpdatePolicyTrait> {
        match self {
            Self::TableHasBatchesUpdate => TableHasBatchesUpdate::new_box(),
            Self::TableChangedSinceLastRunUpdate => TableChangedSinceLastRunUpdate::new_box(),
            Self::TableExistsUpdate => TableExistsUpdate::new_box(),
        }
    }
}

impl Display for AvailableTableUpdatePolicies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TableHasBatchesUpdate => {
                write!(f, "{}", TableHasBatchesUpdate::get_static_name())
            }
            Self::TableChangedSinceLastRunUpdate => {
                write!(f, "{}", TableChangedSinceLastRunUpdate::get_static_name())
            }
            Self::TableExistsUpdate => {
                write!(f, "{}", TableExistsUpdate::get_static_name())
            }
        }
    }
}
