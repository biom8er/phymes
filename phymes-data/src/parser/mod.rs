//! use node_parser::create_parser;
//! 
//! fn main() {
//!     if let Some(parser) = create_parser("html") {
//!         let nodes = parser.parse("<p>Hello world. How are you?</p>");
//!         for node in nodes {
//!             println!("Chunk: {}", node.content);
//!         }
//!     } else {
//!         eprintln!("Unknown parser name");
//!     }
//! }

mod interface;
mod code_splitter;
mod sentence;
mod token_text;

use interface::NodeParser;
use code_splitter::{CodeSplitter, CountMode};
use sentence::{SentenceSplitter, TextNode, Document};
use token_text::TokenTextSplitter;

/// Factory function to create a parser by name.
/// Returns a boxed trait object implementing `NodeParser`.
pub fn create_parser(name: &str) -> Option<Box<dyn NodeParser>> {
    match name.to_lowercase().as_str() {
        "codesplitter" | "code" => {
            Some(Box::new(CodeSplitter::new("python", 40, 15, 1500, CountMode::Char, 512, None, None)) as Box<dyn NodeParser>)
        }
        "sentencesplitter" | "sentence" => {
            Some(Box::new(SentenceSplitter::new(512, 64, " ", "\n\n\n", None, None)) as Box<dyn NodeParser>)
        }
        "tokentextsplitter" | "token" => {
            Some(Box::new(TokenTextSplitter::new(512, 64, " ", None, true, None)) as Box<dyn NodeParser>)
        }
        _ => None,
    }
}

fn default_tokenizer(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creates_code_parser() {
        let parser = create_parser("code").unwrap();
        assert_eq!(parser.class_name(), "CodeSplitter");
    }

    #[test]
    fn test_factory_handles_unknown() {
        assert!(create_parser("unknown").is_none());
    }
}
