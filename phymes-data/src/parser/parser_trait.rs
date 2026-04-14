use crate::parser::{TextNode, sentence::{Document, Split}};

pub trait NodeParserTrait {
    fn parse(&self, text: &str) -> Vec<TextNode>;
    fn parse_with_metadata(&self, text: &str, metadata: &str) -> Vec<TextNode>;
}

pub trait TextParserTrait {
    fn split(&self, text: &str, chunk_size: usize) -> Vec<Split>;
    fn split_text(&self, text: &str) -> Vec<String>;
    fn merge(&self, splits: Vec<Split>, chunk_size: usize) -> Vec<String>;
    fn split_text_metadata_aware(&self, text: &str, metadata_str: &str) -> Vec<String>;
    fn token_size(&self, text: &str) -> usize;
    fn get_nodes_from_documents(&self, docs: &[Document]) -> Vec<TextNode>;
    fn split_text_with_chunk_size(&self, text: &str, chunk_size: usize) -> Vec<String> {
        if text.is_empty() {
            return vec![String::new()];
        }
        let splits = self.split(text, chunk_size);
        self.merge(splits, chunk_size)
    }
    fn split_texts(&self, texts: &[String]) -> Vec<String> {
        texts.iter().flat_map(|t| self.split_text(t)).collect()
    }
    fn split_texts_metadata_aware(
        &self,
        texts: &[String],
        metadata: &[String],
    ) -> Vec<String> {
        texts
            .iter()
            .zip(metadata.iter())
            .flat_map(|(t, m)| self.split_text_metadata_aware(t, m))
            .collect()
    }
}
