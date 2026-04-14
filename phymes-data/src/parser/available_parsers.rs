use std::fmt::Display;

use clap::ValueEnum;
use phymes_subject::MappableTrait;
use serde::{Deserialize, Serialize};

use crate::{parser::{code_splitter::{CodeSplitter, CountMode}, parser_trait::NodeParserTrait, sentence::SentenceSplitter, token_text::TokenTextSplitter}};

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableParsers {
    #[value(name = "TokenTextSplitter")]
    TokenTextSplitter,
    #[default]
    #[value(name = "SentenceSplitter")]
    SentenceSplitter,
    #[value(name = "PythonSplitter")]
    PythonSplitter,
    #[value(name = "RustSplitter")]
    RustSplitter,
    #[value(name = "HtmlSplitter")]
    HtmlSplitter,
}

impl Display for AvailableParsers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenTextSplitter => write!(f, "{}", TokenTextSplitter::get_static_name()),
            Self::SentenceSplitter => write!(f, "{}", SentenceSplitter::get_static_name()),
            Self::PythonSplitter => write!(f, "{}", CodeSplitter::get_static_name()),
            Self::RustSplitter => write!(f, "{}", CodeSplitter::get_static_name()),
            Self::HtmlSplitter => write!(f, "{}", CodeSplitter::get_static_name()),
        }
    }
}

impl AvailableParsers {
    pub fn build(&self) -> Box<dyn NodeParserTrait> {
        match self {
            Self::TokenTextSplitter => Box::new(TokenTextSplitter::default()) as Box<dyn NodeParserTrait>,
            Self::SentenceSplitter => Box::new(SentenceSplitter::default()) as Box<dyn NodeParserTrait>,
            Self::PythonSplitter => Box::new(CodeSplitter::new("python", 40, 15, 1500, CountMode::Char, 512, None, None)) as Box<dyn NodeParserTrait>,
            Self::RustSplitter => Box::new(CodeSplitter::new("rust", 40, 15, 1500, CountMode::Char, 512, None, None)) as Box<dyn NodeParserTrait>,
            Self::HtmlSplitter => Box::new(CodeSplitter::new("html", 40, 15, 1500, CountMode::Char, 512, None, None)) as Box<dyn NodeParserTrait>,
        }
    }
}