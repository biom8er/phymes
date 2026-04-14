use crate::parser::{TextNode, sentence::Document};

pub trait NodeParserTrait {
    fn parse(&self, text: &str) -> Vec<TextNode>;
    fn parse_with_metadata(&self, text: &str, metadata: &str) -> Vec<TextNode>;
    fn class_name(&self) -> &'static str;
}

pub trait TextParserTrait {
    fn split_text(&self, text: &str) -> Vec<String>;
    fn merge<T>(&self, splits: Vec<T>) -> Vec<String>;
    fn split_text_metadata_aware(&self, text: &str, metadata_str: &str) -> Vec<String>;
    fn get_nodes_from_documents(&self, docs: &[Document]) -> Vec<TextNode>;
}
