use std::fmt::Display;

use clap::ValueEnum;
use phymes_core::MappableTrait;
use serde::{Deserialize, Serialize};

use crate::{
    SubjectChangedSinceLastRunUpdate, SubjectExistsUpdate, SubjectHasBatchesUpdate,
    UpdateEventTrait,
};

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableUpdateEvents {
    #[value(name = "SubjectHasBatchesUpdate")]
    SubjectHasBatchesUpdate,
    #[default]
    #[value(name = "SubjectChangedSinceLastRunUpdate")]
    SubjectChangedSinceLastRunUpdate,
    #[value(name = "SubjectExistsUpdate")]
    SubjectExistsUpdate,
}

impl AvailableUpdateEvents {
    pub fn build(self) -> Box<dyn UpdateEventTrait> {
        match self {
            Self::SubjectHasBatchesUpdate => SubjectHasBatchesUpdate::new_box(),
            Self::SubjectChangedSinceLastRunUpdate => SubjectChangedSinceLastRunUpdate::new_box(),
            Self::SubjectExistsUpdate => SubjectExistsUpdate::new_box(),
        }
    }
}

impl Display for AvailableUpdateEvents {
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
