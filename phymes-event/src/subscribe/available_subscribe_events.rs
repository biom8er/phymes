use std::fmt::Display;

use anyhow::{Result, anyhow};
use clap::ValueEnum;
use phymes_subject::MappableTrait;
use serde::{Deserialize, Serialize};

use crate::subscribe::{
    AllSubjectNamesSubscribe, AllSubjectSchemasSubscribe, AlwaysSubscribe,
    AnySubjectSchemaSubscribe, AnySubscribeNameSubscribe, TextGenerationSubscribe,
    SubscribeEventTrait,
};

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableSubscribeEvents {
    #[value(name = "AlwaysSubscribe")]
    AlwaysSubscribe,
    #[value(name = "AnySubjectNameSubscribe")]
    AnySubjectNameSubscribe,
    #[default]
    #[value(name = "AllSubjectNamesSubscribe")]
    AllSubjectNamesSubscribe,
    #[value(name = "AnySubjectSchemaSubscribe")]
    AnySubjectSchemaSubscribe,
    #[value(name = "AllSubjectSchemasSubscribe")]
    AllSubjectSchemasSubscribe,
    #[value(name = "TextGenerationSubscribe")]
    TextGenerationSubscribe,
}

impl AvailableSubscribeEvents {
    pub fn build(self) -> Box<dyn SubscribeEventTrait> {
        match self {
            Self::AlwaysSubscribe => AlwaysSubscribe::new_box(),
            Self::AnySubjectNameSubscribe => AnySubscribeNameSubscribe::new_box(),
            Self::AllSubjectNamesSubscribe => AllSubjectNamesSubscribe::new_box(),
            Self::AnySubjectSchemaSubscribe => AnySubjectSchemaSubscribe::new_box(),
            Self::AllSubjectSchemasSubscribe => AllSubjectSchemasSubscribe::new_box(),
            Self::TextGenerationSubscribe => TextGenerationSubscribe::new_box(),
        }
    }
    /// Convert a [String] to a [SubscribeEventTrait]
    ///   by checking if the [String] contains the [SubscribeEventTrait] name
    pub fn from_str_fuzzy(policy: &str) -> Result<Self> {
        let subscribe = if policy.contains(AllSubjectSchemasSubscribe::get_static_name()) {
            AvailableSubscribeEvents::AllSubjectSchemasSubscribe
        } else if policy.contains(AnySubjectSchemaSubscribe::get_static_name()) {
            AvailableSubscribeEvents::AnySubjectSchemaSubscribe
        } else if policy.contains(AllSubjectNamesSubscribe::get_static_name()) {
            AvailableSubscribeEvents::AllSubjectNamesSubscribe
        } else if policy.contains(AnySubscribeNameSubscribe::get_static_name()) {
            AvailableSubscribeEvents::AnySubjectNameSubscribe
        } else if policy.contains(AlwaysSubscribe::get_static_name()) {
            AvailableSubscribeEvents::AlwaysSubscribe
        } else if policy.contains(TextGenerationSubscribe::get_static_name()) {
            AvailableSubscribeEvents::TextGenerationSubscribe
        } else {
            return Err(anyhow!("Subscribe policy {policy} was not recognized."));
        };
        Ok(subscribe)
    }
}

impl Display for AvailableSubscribeEvents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlwaysSubscribe => write!(f, "{}", AlwaysSubscribe::get_static_name()),
            Self::AnySubjectNameSubscribe => {
                write!(f, "{}", AnySubscribeNameSubscribe::get_static_name())
            }
            Self::AllSubjectNamesSubscribe => {
                write!(f, "{}", AllSubjectNamesSubscribe::get_static_name())
            }
            Self::AnySubjectSchemaSubscribe => {
                write!(f, "{}", AnySubjectSchemaSubscribe::get_static_name())
            }
            Self::AllSubjectSchemasSubscribe => {
                write!(f, "{}", AllSubjectSchemasSubscribe::get_static_name())
            }
            Self::TextGenerationSubscribe => write!(f, "{}", TextGenerationSubscribe::get_static_name()),
        }
    }
}
