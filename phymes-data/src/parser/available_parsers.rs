use std::fmt::Display;

use clap::ValueEnum;
use phymes_subject::MappableTrait;
use serde::{Deserialize, Serialize};

#[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
use crate::parser::code_splitter::{CodeSplitter, CountMode};
use crate::parser::{
    parser_trait::NodeParserTrait, sentence::SentenceSplitter, token_text::TokenTextSplitter,
};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableParsers {
    #[value(name = "TokenTextSplitter")]
    TokenTextSplitter,
    #[default]
    #[value(name = "SentenceSplitter")]
    SentenceSplitter,
    #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
    #[value(name = "PythonSplitter")]
    PythonSplitter,
    #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
    #[value(name = "RustSplitter")]
    RustSplitter,
    #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
    #[value(name = "HtmlSplitter")]
    HtmlSplitter,
}

impl Display for AvailableParsers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenTextSplitter => write!(f, "{}", TokenTextSplitter::get_static_name()),
            Self::SentenceSplitter => write!(f, "{}", SentenceSplitter::get_static_name()),
            #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
            Self::PythonSplitter => write!(f, "{}", CodeSplitter::get_static_name()),
            #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
            Self::RustSplitter => write!(f, "{}", CodeSplitter::get_static_name()),
            #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
            Self::HtmlSplitter => write!(f, "{}", CodeSplitter::get_static_name()),
        }
    }
}

impl AvailableParsers {
    pub fn build(&self) -> Box<dyn NodeParserTrait> {
        match self {
            Self::TokenTextSplitter => {
                Box::new(TokenTextSplitter::default()) as Box<dyn NodeParserTrait>
            }
            Self::SentenceSplitter => {
                Box::new(SentenceSplitter::default()) as Box<dyn NodeParserTrait>
            }
            #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
            Self::PythonSplitter => Box::new(CodeSplitter::new(
                "python",
                40,
                15,
                1500,
                CountMode::Char,
                512,
                None,
                None,
            )) as Box<dyn NodeParserTrait>,
            #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
            Self::RustSplitter => Box::new(CodeSplitter::new(
                "rust",
                40,
                15,
                1500,
                CountMode::Char,
                512,
                None,
                None,
            )) as Box<dyn NodeParserTrait>,
            #[cfg(all(not(target_family = "wasm"), feature = "wsl"))]
            Self::HtmlSplitter => Box::new(CodeSplitter::new(
                "html",
                40,
                15,
                1500,
                CountMode::Char,
                512,
                None,
                None,
            )) as Box<dyn NodeParserTrait>,
        }
    }
}
