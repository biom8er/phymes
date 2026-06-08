// src/node_parser/code_splitter.rs
use crate::parser::{Document, NodeParserTrait, TextNode, TokenizerType, default_tokenizer};
use phymes_subject::MappableTrait;
use std::sync::Arc;
use tree_sitter::Parser;
pub struct CodeSplitter {
    language: String,
    chunk_lines: usize,
    chunk_lines_overlap: usize,
    max_chars: usize,
    count_mode: CountMode,
    max_tokens: usize,
    tokenizer: TokenizerType,
    parser: Parser,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CountMode {
    Char,
    Token,
}

impl CodeSplitter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        language: &str,
        chunk_lines: usize,
        chunk_lines_overlap: usize,
        max_chars: usize,
        count_mode: CountMode,
        max_tokens: usize,
        tokenizer: Option<TokenizerType>,
        parser: Option<Parser>,
    ) -> Self {
        let tokenizer = tokenizer.unwrap_or_else(|| Arc::new(default_tokenizer));
        let parser = parser.unwrap_or_else(|| {
            let mut p = Parser::new();
            let language_obj = tree_sitter_language(language)
                .unwrap_or_else(|| panic!("Unsupported language: {language}"));
            p.set_language(&language_obj).unwrap();
            p
        });

        Self {
            language: language.to_string(),
            chunk_lines,
            chunk_lines_overlap,
            max_chars,
            count_mode,
            max_tokens,
            tokenizer,
            parser,
        }
    }

    pub fn from_defaults(language: &str, count_mode: CountMode) -> Self {
        Self::new(language, 40, 15, 1500, count_mode, 512, None, None)
    }

    pub fn split_text(&mut self, text: &str) -> Vec<String> {
        let tree = self.parser.parse(text, None).expect("Failed to parse code");
        let text_bytes = text.as_bytes();
        let chunks = self.chunk_node(tree.root_node(), text_bytes);
        chunks.into_iter().map(|s| s.trim().to_string()).collect()
    }

    fn chunk_node(&self, node: tree_sitter::Node, text_bytes: &[u8]) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let max_size = match self.count_mode {
            CountMode::Char => self.max_chars,
            CountMode::Token => self.max_tokens,
        };

        for child in node.children(&mut node.walk()) {
            let child_text = &text_bytes[child.start_byte()..child.end_byte()];
            let child_str = String::from_utf8_lossy(child_text);
            let child_size = match self.count_mode {
                CountMode::Char => child_str.len(),
                CountMode::Token => (self.tokenizer)(&child_str).len(),
            };

            if child_size > max_size {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }
                chunks.extend(self.chunk_node(child, text_bytes));
            } else {
                let new_chunk = format!("{current_chunk}{child_str}");
                let new_size = match self.count_mode {
                    CountMode::Char => new_chunk.len(),
                    CountMode::Token => (self.tokenizer)(&new_chunk).len(),
                };

                if new_size > max_size {
                    if !current_chunk.is_empty() {
                        chunks.push(current_chunk.clone());
                    }
                    current_chunk = child_str.to_string();
                } else {
                    current_chunk.push_str(&child_str);
                }
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }
        chunks
    }

    pub fn split_text_metadata_aware(&mut self, text: &str, metadata_str: &str) -> Vec<String> {
        let metadata_len = (self.tokenizer)(metadata_str).len();
        let effective_limit = match self.count_mode {
            CountMode::Char => self.max_chars.saturating_sub(metadata_len),
            CountMode::Token => self.max_tokens.saturating_sub(metadata_len),
        };

        if effective_limit == 0 {
            panic!("Metadata length exceeds chunk size");
        }

        let tree = self.parser.parse(text, None).expect("Failed to parse code");
        let text_bytes = text.as_bytes();
        let chunks = self.chunk_node(tree.root_node(), text_bytes);
        chunks
            .into_iter()
            .filter(|c| !c.trim().is_empty())
            .map(|c| c.trim().to_string())
            .collect()
    }

    pub fn get_nodes_from_documents(&mut self, docs: &[Document]) -> Vec<TextNode> {
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

fn tree_sitter_language(lang: &str) -> Option<tree_sitter::Language> {
    match lang.to_lowercase().as_str() {
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "html" => Some(tree_sitter_html::LANGUAGE.into()),
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        _ => None,
    }
}

impl NodeParserTrait for CodeSplitter {
    fn parse(&self, text: &str) -> Vec<TextNode> {
        let mut splitter = CodeSplitter::new(
            &self.language,
            self.chunk_lines,
            self.chunk_lines_overlap,
            self.max_chars,
            self.count_mode,
            self.max_tokens,
            Some(self.tokenizer.clone()),
            None,
        );
        splitter.get_nodes_from_documents(&[Document {
            text: text.to_string(),
            metadata: None,
        }])
    }

    fn parse_with_metadata(&self, text: &str, metadata: &str) -> Vec<TextNode> {
        let mut splitter = CodeSplitter::new(
            &self.language,
            self.chunk_lines,
            self.chunk_lines_overlap,
            self.max_chars,
            self.count_mode,
            self.max_tokens,
            Some(self.tokenizer.clone()),
            None,
        );
        let chunks = splitter.split_text_metadata_aware(text, metadata);
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

impl MappableTrait for CodeSplitter {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_code_splitter() {
        let mut splitter = CodeSplitter::new("python", 4, 1, 30, CountMode::Char, 512, None, None);
        let text = "def foo():\n    print(\"bar\")\n\ndef baz():\n    print(\"bbq\")";
        let chunks = splitter.split_text(text);
        assert!(chunks[0].starts_with("def foo():"));
        assert!(chunks[1].starts_with("def baz():"));
    }

    #[test]
    fn test_token_mode() {
        let tokenizer = Arc::new(|s: &str| s.split_whitespace().map(|x| x.to_string()).collect());
        let mut splitter = CodeSplitter::new(
            "python",
            4,
            1,
            30,
            CountMode::Token,
            5,
            Some(tokenizer),
            None,
        );
        let text = "def foo():\n    print(\"bar\")\n    print(\"another line\")\n\ndef baz():\n    print(\"bbq\")";
        let chunks = splitter.split_text(text);
        assert!(chunks[0].starts_with("def foo():"));
        assert!(chunks[1].starts_with("def baz():"));
    }

    #[test]
    fn test_invalid_language() {
        let result = std::panic::catch_unwind(|| {
            CodeSplitter::new("invalid_lang", 4, 1, 30, CountMode::Char, 512, None, None);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_token_mode_node_parsing() {
        let text = r#"
def complex_function():
    # This is a comment
    variable_with_very_long_name = "some string value"
    another_variable = variable_with_very_long_name.upper()
    return another_variable
"#;

        let doc = Document {
            text: text.to_string(),
            metadata: None,
        };

        let mut splitter =
            CodeSplitter::new("python", 40, 15, 1500, CountMode::Token, 20, None, None);
        let nodes = splitter.get_nodes_from_documents(&[doc]);
        assert!(!nodes.is_empty());
        assert!(nodes[0].content.contains("def complex_function():"));
    }

    #[test]
    fn test_metadata_aware_split() {
        let mut splitter =
            CodeSplitter::new("python", 40, 15, 1500, CountMode::Token, 25, None, None);
        let text = "def example_function():\n    result = calculate_something()\n    return result";
        let metadata = "author: test_user, repo: example_repo";
        let chunks = splitter.split_text_metadata_aware(text, metadata);
        assert!(!chunks.is_empty());
        assert!(chunks[0].contains("def example_function():"));
    }
}
