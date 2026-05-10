use phymes_subject::MappableTrait;
// src/text_splitter/sentence.rs
use regex::Regex;
use std::sync::Arc;

use crate::parser::{TokenizerType, default_tokenizer, parser_trait::{NodeParserTrait, TextParserTrait}};

#[derive(Clone)]
pub struct SentenceSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separator: String,
    paragraph_separator: String,
    secondary_chunking_regex: Option<Regex>,
    tokenizer: TokenizerType,
}

#[derive(Clone)]
pub struct Split {
    pub text: String,
    pub is_sentence: bool,
    pub token_size: usize,
}

#[derive(Clone)]
pub struct TextNode {
    pub content: String,
    pub start_char_idx: usize,
    pub end_char_idx: usize,
}

#[derive(Clone)]
pub struct Document {
    pub text: String,
    pub metadata: Option<String>,
}

impl SentenceSplitter {
    pub fn new(
        chunk_size: usize,
        chunk_overlap: usize,
        separator: &str,
        paragraph_separator: &str,
        secondary_chunking_regex: Option<&str>,
        tokenizer: Option<TokenizerType>,
    ) -> Self {
        if chunk_overlap > chunk_size {
            panic!("chunk_overlap must be smaller than chunk_size");
        }

        let regex = secondary_chunking_regex.map(|r| Regex::new(r).unwrap());
        let tokenizer = tokenizer.unwrap_or_else(|| Arc::new(default_tokenizer));

        Self {
            chunk_size,
            chunk_overlap,
            separator: separator.to_string(),
            paragraph_separator: paragraph_separator.to_string(),
            secondary_chunking_regex: regex,
            tokenizer,
        }
    }

    fn get_splits_by_fns(&self, text: &str) -> (Vec<String>, bool) {
        let paragraph_splits: Vec<String> = text.split(&self.paragraph_separator).map(|s| s.to_string()).collect();
        if paragraph_splits.len() > 1 {
            return (paragraph_splits, true);
        }

        if let Some(regex) = &self.secondary_chunking_regex {
            let mut splits = Vec::new();
            for cap in regex.find_iter(text) {
                splits.push(cap.as_str().to_string());
            }
            if splits.len() > 1 {
                return (splits, false);
            }
        }

        let word_splits: Vec<String> = text.split(&self.separator).map(|s| [s, &self.separator].join("")).collect();
        (word_splits, false)
    }
}

fn join_text(chunks: &[(String, usize)]) -> String {
    chunks.iter().map(|(t, _)| t).cloned().collect::<Vec<_>>().join("")
}

impl TextParserTrait for SentenceSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![String::new()];
        }

        let splits = self.split(text, self.chunk_size);
        self.merge(splits, self.chunk_size)
    }

    fn merge(&self, splits: Vec<Split>, chunk_size: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut cur_chunk = Vec::new();
        let mut cur_len = 0usize;

        let mut new_chunk = true;
        let mut idx = 0;

        while idx < splits.len() {
            let s = &splits[idx];
            if s.token_size > chunk_size {
                panic!("Single token exceeded chunk size");
            }

            if cur_len + s.token_size > chunk_size && !new_chunk {
                chunks.push(join_text(&cur_chunk));
                let last_chunk = cur_chunk.clone();
                cur_chunk.clear();
                cur_len = 0;
                new_chunk = true;

                // add overlap
                for (text, len) in last_chunk.iter().rev() {
                    if cur_len + len <= self.chunk_overlap {
                        cur_chunk.insert(0, (text.clone(), *len));
                        cur_len += len;
                    } else {
                        break;
                    }
                }
            } else {
                if new_chunk && cur_len + s.token_size > chunk_size {
                    while !cur_chunk.is_empty() && cur_len + s.token_size > chunk_size {
                        let (_, len) = cur_chunk.remove(0);
                        cur_len -= len;
                    }
                }

                if s.is_sentence || cur_len + s.token_size <= chunk_size || new_chunk {
                    cur_chunk.push((s.text.clone(), s.token_size));
                    cur_len += s.token_size;
                    idx += 1;
                    new_chunk = false;
                } else {
                    chunks.push(join_text(&cur_chunk));
                    cur_chunk.clear();
                    cur_len = 0;
                    new_chunk = true;
                }
            }
        }

        if !cur_chunk.is_empty() {
            chunks.push(join_text(&cur_chunk));
        }

        chunks
            .into_iter()
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect()
    }

    fn split_text_metadata_aware(&self, text: &str, metadata_str: &str) -> Vec<String> {
        let metadata_len = (self.tokenizer)(metadata_str).len();
        let effective_chunk_size = self.chunk_size.saturating_sub(metadata_len);

        if effective_chunk_size == 0 {
            panic!("Metadata length exceeds chunk size");
        }

        self.split_text_with_chunk_size(text, effective_chunk_size)
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
    
    fn split(&self, text: &str, chunk_size: usize) -> Vec<Split> {
        let token_size = self.token_size(text);
        if token_size <= chunk_size {
            return vec![Split {
                text: text.to_string(),
                is_sentence: true,
                token_size,
            }];
        }

        let (splits, is_sentence) = self.get_splits_by_fns(text);
        let mut result = Vec::new();

        for s in splits {
            let size = self.token_size(&s);
            if size <= chunk_size {
                result.push(Split {
                    text: s,
                    is_sentence,
                    token_size: size,
                });
            } else {
                result.extend(self.split(&s, chunk_size));
            }
        }

        result
    }

    fn token_size(&self, text: &str) -> usize {
        (self.tokenizer)(text).len()
    }
}

impl Default for SentenceSplitter {
    fn default() -> Self {
        let regex = Regex::new("[^,.;。？！]+[,.;。？！]?|[,.;。？！]").unwrap();
        let tokenizer = Arc::new(default_tokenizer);
        Self { chunk_size: 512, chunk_overlap: 0, separator: " ".to_string(), paragraph_separator: "\n\n\n".to_string(), secondary_chunking_regex: Some(regex), tokenizer }
    }
}

impl NodeParserTrait for SentenceSplitter {
    fn parse(&self, text: &str) -> Vec<TextNode> {
        self.get_nodes_from_documents(&[Document { text: text.to_string(), metadata: None }])
    }

    fn parse_with_metadata(&self, text: &str, metadata: &str) -> Vec<TextNode> {
        let doc = Document {
            text: text.to_string(),
            metadata: Some(metadata.to_string()),
        };
        self.get_nodes_from_documents(&[doc])
    }
}

impl MappableTrait for SentenceSplitter {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paragraphs() {
        let splitter = SentenceSplitter::new(20, 0, " ", "\n\n\n", Some("[^,.;。？！]+[,.;。？！]?|[,.;。？！]"), None);
        let text = format!("{}{}{}", "foo ".repeat(15), "\n\n\n", "bar ".repeat(15));
        let chunks = splitter.split_text(&text);
        assert_eq!(chunks[0].trim(), "foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo");
        assert_eq!(chunks[1].trim(), "bar bar bar bar bar bar bar bar bar bar bar bar bar bar bar");
    }

    #[test]
    fn test_sentences() {
        let splitter = SentenceSplitter::new(20, 0, " ", "\n\n\n", Some("[^,.;。？！]+[,.;。？！]?|[,.;。？！]"), None);
        let text = format!("{}{}{}", "foo ".repeat(15), ". ", "bar ".repeat(15));
        let chunks = splitter.split_text(&text);
        dbg!(&chunks);
        assert_eq!(chunks[0], "foo foo foo foo foo foo foo foo foo foo foo foo foo foo foo .");
        assert_eq!(chunks[1], "bar bar bar bar bar bar bar bar bar bar bar bar bar bar bar");
    }

    #[test]
    fn test_overlap() {
        let splitter = SentenceSplitter::new(15, 10, " ", "\n\n\n", Some("[^,.;。？！]+[,.;。？！]?|[,.;。？！]"), None);
        let chunks = splitter.split_text("Hello! How are you? I am fine. And you?");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks.first().unwrap(),  "Hello! How are you? I am fine. And you?");
    }
    #[test]
    fn test_start_end_char_idx() {
        let doc = Document {
            text: "foo ".repeat(15) + "\n\n\n" + &"bar ".repeat(15),
            metadata: None,
        };
        let splitter = SentenceSplitter::new(2, 1, " ", "\n\n\n", Some("[^,.;。？！]+[,.;。？！]?|[,.;。？！]"), None);
        let nodes = splitter.get_nodes_from_documents(&[doc]);
        for node in nodes {
            assert_eq!(node.end_char_idx - node.start_char_idx, node.content.len());
        }
    }

    #[test]
    fn test_split_with_metadata() {
        let chunk_size = 100;
        let metadata_str = "word ".repeat(50);
        let tokenizer = Arc::new(|s: &str| s.split_whitespace().map(|x| x.to_string()).collect());
        let splitter = SentenceSplitter::new(chunk_size, 0, " ", "\n\n\n", None, Some(tokenizer));

        let text = "foo ".repeat(200);
        let chunks = splitter.split_text_metadata_aware(&text, &metadata_str);
        for chunk in chunks {
            let combined = format!("{chunk}{metadata_str}");
            let token_count = (splitter.tokenizer)(&combined).len();
            assert!(token_count <= chunk_size);
        }
    }

    #[test]
    fn test_split_texts_multiple() {
        let splitter = SentenceSplitter::new(20, 0, " ", "\n\n\n", None, None);
        let text1 = "foo ".repeat(15) + "\n\n\n" + &"bar ".repeat(15);
        let text2 = "bar ".repeat(15) + "\n\n\n" + &"foo ".repeat(15);
        let texts = vec![text1, text2];
        let chunks = splitter.split_texts(&texts);
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn test_no_overflow_with_chinese_text_and_metadata() {
        // Chinese text sample similar to the Python test
        let text = "你所描述的情况可能与身体健康有关，尤其是与压力、疲劳和动机相关的身体状态。长时间的工作压力和疲劳可能导致身体功能下降，包括记忆力、注意力和决策能力。此外，焦虑和压力可能会影响你的情绪状态和工作表现，从而形成一个恶性循环。";
        let metadata = "教育、文化、学习、人才、成长、创造、未来、资源、关注、才华和潜力";

        let doc = Document {
            text: text.to_string(),
            metadata: Some(metadata.to_string()),
        };

        let splitter = SentenceSplitter::new(512, 64, " ", "\n\n\n", None, None);
        let nodes = splitter.get_nodes_from_documents(&[doc]);

        for (i, node) in nodes.iter().enumerate() {
            let content_length = (splitter.tokenizer)(&node.content).len();
            assert!(
                content_length <= 512,
                "Node {i} has {content_length} tokens, exceeds chunk_size of 512",
            );
        }
    }

    #[test]
    fn test_overlap_edge_case() {
        let splitter = SentenceSplitter::new(10, 5, " ", "\n\n\n", None, None);
        let chunks = splitter.split_text("Hello! How are you? I am fine. And you?");
        assert_eq!(chunks.len(), 1);

        let chunks2 = splitter.split_text(
            "Hello! How are you? I am fine. And you? This is a slightly longer sentence.",
        );
        assert_eq!(chunks2.len(), 2);
        assert_eq!(chunks2, [
            "Hello! How are you? I am fine. And you? This",
            "am fine. And you? This is a slightly longer sentence.",
        ]);
    }
}
