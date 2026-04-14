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

mod available_parsers;
mod parser_trait;
mod code_splitter;
mod sentence;
mod token_text;

pub use available_parsers::AvailableParsers;
use parser_trait::{NodeParserTrait, TextParserTrait};
use code_splitter::{CodeSplitter, CountMode};
use sentence::{SentenceSplitter, TextNode, Document, Split};
use token_text::TokenTextSplitter;

fn default_tokenizer(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_string()).collect()
}
