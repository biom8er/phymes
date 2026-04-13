use crate::parser::TextNode;

pub trait NodeParser {
    fn parse(&self, text: &str) -> Vec<TextNode>;
    fn parse_with_metadata(&self, text: &str, metadata: &str) -> Vec<TextNode>;
    fn class_name(&self) -> &'static str;
}
