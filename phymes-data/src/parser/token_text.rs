// src/node_parser/token_text.rs
use std::sync::Arc;
use phymes_subject::MappableTrait;

use crate::parser::{Document, NodeParserTrait, TextNode, TokenizerType, default_tokenizer, parser_trait::TextParserTrait, sentence::Split};

#[derive(Clone)]
pub struct TokenTextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separator: String,
    backup_separators: Vec<String>,
    keep_whitespaces: bool,
    tokenizer: TokenizerType,
}

impl TokenTextSplitter {
    pub fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        separator: &str,
        backup_separators: Option<Vec<String>>,
        keep_whitespaces: bool,
        tokenizer: Option<TokenizerType>,
    ) -> Self {
        if chunk_overlap > chunk_size {
            panic!("chunk_overlap must be smaller than chunk_size");
        }

        let tokenizer = tokenizer.unwrap_or_else(|| Arc::new(default_tokenizer));
        let backup_separators = backup_separators.unwrap_or_else(|| vec!["\n".to_string()]);

        Self {
            chunk_size,
            chunk_overlap,
            separator: separator.to_string(),
            backup_separators,
            keep_whitespaces,
            tokenizer,
        }
    }
}

impl Default for TokenTextSplitter {
    fn default() -> Self {
        let tokenizer = Arc::new(default_tokenizer);
        Self { chunk_size: 512, chunk_overlap: 64, separator: " ".to_string(), backup_separators: vec!["\n".to_string()], keep_whitespaces: false, tokenizer }
    }
}

impl TextParserTrait for TokenTextSplitter {
    fn split(&self, text: &str, chunk_size: usize) -> Vec<Split> {
        let token_size = self.token_size(text);
        if token_size <= chunk_size {
            return vec![Split {
                text: text.to_string(),
                is_sentence: false,
                token_size,
            }];
        }

        let mut splits = Vec::new();
        for sep in std::iter::once(&self.separator).chain(self.backup_separators.iter()) {
            splits = text.split(sep).map(|s| [s, sep].join("")).collect();
            if splits.len() > 1 {
                break;
            }
        }

        let mut result = Vec::new();

        for s in splits {
            let size = self.token_size(&s);
            if size <= chunk_size {
                result.push(Split {
                    text: s,
                    is_sentence: false,
                    token_size: size,
                });
            } else {
                result.extend(self.split(&s, chunk_size));
            }
        }

        result
    }

    fn split_text(&self, text: &str) -> Vec<String> {
        self.split_text_with_chunk_size(text, self.chunk_size)
    }

    fn merge(&self, splits: Vec<Split>, chunk_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut cur_chunk = Vec::<String>::new();
        let mut cur_len = 0usize;

        for split in splits.into_iter().map(|s| s.text) {
            let split_len = self.token_size(&split);
            if cur_len + split_len > chunk_size {
                let chunk = if self.keep_whitespaces {
                    cur_chunk.join("")
                } else {
                    cur_chunk.join("").trim().to_string()
                };
                if !chunk.is_empty() {
                    chunks.push(chunk);
                }

                while cur_len > self.chunk_overlap || cur_len + split_len > chunk_size {
                    let first = cur_chunk.remove(0);
                    cur_len -= self.token_size(&first);
                }
            }

            cur_chunk.push(split);
            cur_len += split_len;
        }

        let chunk = if self.keep_whitespaces {
            cur_chunk.join("")
        } else {
            cur_chunk.join("").trim().to_string()
        };
        if !chunk.is_empty() {
            chunks.push(chunk);
        }

        chunks
    }

    fn split_text_metadata_aware(&self, text: &str, metadata_str: &str) -> Vec<String> {
        let metadata_len = (self.tokenizer)(metadata_str).len() + 2; // reserve for metadata format
        let effective_chunk_size = self.chunk_size.saturating_sub(metadata_len);
        if effective_chunk_size == 0 {
            panic!("Metadata length exceeds chunk size");
        }
        self.split_text_with_chunk_size(text, effective_chunk_size)
    }

    fn token_size(&self, text: &str) -> usize {
        (self.tokenizer)(text).len()
    }

    fn get_nodes_from_documents(&self, docs: &[Document]) -> Vec<TextNode> {
        let mut nodes = Vec::new();
        for doc in docs {
            let chunks = self.split_text(&doc.text);
            let mut offset = 0;
            for chunk in chunks {
                let start = offset;
                let end = start + chunk.len();
                nodes.push(TextNode {
                    content: chunk.clone(),
                    start_char_idx: start,
                    end_char_idx: end,
                });
                offset = end;
            }
        }
        nodes
    }
}

impl NodeParserTrait for TokenTextSplitter {
    fn parse(&self, text: &str) -> Vec<TextNode> {
        self.get_nodes_from_documents(&[Document { text: text.to_string(), metadata: None }])
    }

    fn parse_with_metadata(&self, text: &str, metadata: &str) -> Vec<TextNode> {
        let chunks = self.split_text_metadata_aware(text, metadata);
        let mut offset = 0;
        chunks
            .into_iter()
            .map(|chunk| {
                let start = offset;
                let end = start + chunk.len();
                offset = end;
                TextNode {
                    content: chunk,
                    start_char_idx: start,
                    end_char_idx: end,
                }
            })
            .collect()
    }
}

impl MappableTrait for TokenTextSplitter {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_token() {
        let splitter = TokenTextSplitter::new(1, 0, " ", None, false, None);
        let chunks = splitter.split_text("foo bar");
        assert_eq!(chunks, vec!["foo", "bar"]);

        let splitter = TokenTextSplitter::new(2, 1, " ", None, false, None);
        let chunks = splitter.split_text("foo bar hello world");
        assert_eq!(chunks, vec!["foo bar", "bar hello", "hello world"]);
    }

    #[test]
    fn test_start_end_char_idx() {
        let doc = Document { text: "foo bar hello world baz bbq".to_string(), metadata: None };
        let splitter = TokenTextSplitter::new(3, 1, " ", None, false, None);
        let nodes = splitter.get_nodes_from_documents(&[doc]);
        for node in nodes {
            assert_eq!(node.end_char_idx - node.start_char_idx, node.content.len());
        }
    }

    #[test]
    fn test_split_with_metadata() {
        let metadata = "word ".repeat(50);
        let splitter = TokenTextSplitter::new(100, 0, " ", None, false, None);
        let text = "foo bar hello world baz bbq".repeat(10);
        let chunks = splitter.split_text_metadata_aware(&text, &metadata);
        assert!(!chunks.is_empty());
    }
}
