use std::fmt::Display;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::{
    MappableTrait, SubjectChangedSinceLastRunUpdate, SubjectExistsUpdate, SubjectHasBatchesUpdate,
    UpdatePolicyTrait,
};

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableUpdatePolicies {
    #[value(name = "SubjectHasBatchesUpdate")]
    SubjectHasBatchesUpdate,
    #[default]
    #[value(name = "SubjectChangedSinceLastRunUpdate")]
    SubjectChangedSinceLastRunUpdate,
    #[value(name = "SubjectExistsUpdate")]
    SubjectExistsUpdate,
}

impl AvailableUpdatePolicies {
    pub fn build(self) -> Box<dyn UpdatePolicyTrait> {
        match self {
            Self::SubjectHasBatchesUpdate => SubjectHasBatchesUpdate::new_box(),
            Self::SubjectChangedSinceLastRunUpdate => SubjectChangedSinceLastRunUpdate::new_box(),
            Self::SubjectExistsUpdate => SubjectExistsUpdate::new_box(),
        }
    }
}

impl Display for AvailableUpdatePolicies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SubjectHasBatchesUpdate => {
                write!(f, "{}", SubjectHasBatchesUpdate::get_static_name())
            }
            Self::SubjectChangedSinceLastRunUpdate => {
                write!(f, "{}", SubjectChangedSinceLastRunUpdate::get_static_name())
            }
            Self::SubjectExistsUpdate => {
                write!(f, "{}", SubjectExistsUpdate::get_static_name())
            }
        }
    }
}
