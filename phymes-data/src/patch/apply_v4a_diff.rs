use anyhow::{Error, Result, anyhow};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyDiffMode {
    Default,
    Create,
}

impl std::str::FromStr for ApplyDiffMode {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "default" => Ok(ApplyDiffMode::Default),
            "create" => Ok(ApplyDiffMode::Create),
            _ => Err(anyhow!("`{s}` is not a recognized ApplyDiffMode variant.")),
        }
    }
}

#[derive(Debug)]
pub enum ApplyDiffError {
    InvalidLine(String),
    InvalidAddFileLine(String),
    InvalidContext {
        cursor: usize,
        context: String,
        eof: bool,
    },
    OverlappingChunk {
        orig_index: usize,
        cursor: usize,
    },
    ChunkOutOfBounds {
        orig_index: usize,
        len: usize,
    },
    EmptySection {
        index: usize,
        next_line: String,
    },
    Other(String),
}

impl fmt::Display for ApplyDiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ApplyDiffError::*;
        match self {
            InvalidLine(line) => write!(f, "Invalid Line:\n{line}"),
            InvalidAddFileLine(line) => write!(f, "Invalid Add File Line: {line}"),
            InvalidContext {
                cursor,
                context,
                eof,
            } => {
                if *eof {
                    write!(f, "Invalid EOF Context {cursor}:\n{context}")
                } else {
                    write!(f, "Invalid Context {cursor}:\n{context}")
                }
            }
            OverlappingChunk { orig_index, cursor } => write!(
                f,
                "applyDiff: overlapping chunk at {orig_index} (cursor {cursor})"
            ),
            ChunkOutOfBounds { orig_index, len } => write!(
                f,
                "applyDiff: chunk.origIndex {orig_index} > input length {len}"
            ),
            EmptySection { index, next_line } => {
                write!(f, "Nothing in this section - index={index} {next_line}")
            }
            Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ApplyDiffError {}

#[derive(Debug, Clone)]
struct Chunk {
    orig_index: usize,
    del_lines: Vec<String>,
    ins_lines: Vec<String>,
}

#[derive(Debug)]
struct ParserState {
    lines: Vec<String>,
    index: usize,
    fuzz: i32,
}

#[derive(Debug)]
struct ParsedUpdateDiff {
    chunks: Vec<Chunk>,
}

#[derive(Debug)]
struct ReadSectionResult {
    next_context: Vec<String>,
    section_chunks: Vec<Chunk>,
    end_index: usize,
    eof: bool,
}

const END_PATCH: &str = "*** End Patch";
const END_FILE: &str = "*** End of File";

const SECTION_TERMINATORS: &[&str] = &[
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
];

const END_SECTION_MARKERS: &[&str] = &[
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
    END_FILE,
];

pub fn apply_v4a_diff(
    input: &str,
    diff: &str,
    mode: ApplyDiffMode,
) -> Result<String, ApplyDiffError> {
    let newline = detect_newline(input, diff, mode);
    let diff_lines = normalize_diff_lines(diff);

    if matches!(mode, ApplyDiffMode::Create) {
        return parse_create_diff(&diff_lines, &newline);
    }

    let normalized_input = normalize_text_newlines(input);
    let parsed = parse_update_diff(&diff_lines, &normalized_input)?;
    let result = apply_chunks(&normalized_input, &parsed.chunks, &newline)?;
    Ok(result)
}

fn normalize_diff_lines(diff: &str) -> Vec<String> {
    let mut lines: Vec<String> = diff
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect();

    if let Some(last) = lines.last()
        && last.is_empty() {
            lines.pop();
        }
    lines
}

fn detect_newline_from_text(text: &str) -> &str {
    if text.contains("\r\n") { "\r\n" } else { "\n" }
}

fn detect_newline(input: &str, diff: &str, mode: ApplyDiffMode) -> String {
    match mode {
        ApplyDiffMode::Default if input.contains('\n') => detect_newline_from_text(input).into(),
        _ => detect_newline_from_text(diff).into(),
    }
}

fn normalize_text_newlines(text: &str) -> String {
    text.replace("\r\n", "\n")
}

fn is_done(state: &ParserState, prefixes: &[&str]) -> bool {
    if state.index >= state.lines.len() {
        return true;
    }
    let line = &state.lines[state.index];
    prefixes.iter().any(|p| line.starts_with(p))
}

fn read_str(state: &mut ParserState, prefix: &str) -> String {
    if state.index >= state.lines.len() {
        return String::new();
    }
    let current = &state.lines[state.index];
    if let Some(s) = current.strip_prefix(prefix) {
        state.index += 1;
        s.to_string()
    } else {
        String::new()
    }
}

fn parse_create_diff(lines: &[String], newline: &str) -> Result<String, ApplyDiffError> {
    let mut parser = ParserState {
        lines: {
            let mut v = lines.to_vec();
            v.push(END_PATCH.to_string());
            v
        },
        index: 0,
        fuzz: 0,
    };

    let mut output: Vec<String> = Vec::new();

    while !is_done(&parser, SECTION_TERMINATORS) {
        if parser.index >= parser.lines.len() {
            break;
        }
        let line = parser.lines[parser.index].clone();
        parser.index += 1;

        if !line.starts_with('+') {
            return Err(ApplyDiffError::InvalidAddFileLine(line));
        }
        output.push(line[1..].to_string());
    }

    Ok(output.join(newline))
}

fn parse_update_diff(lines: &[String], input: &str) -> Result<ParsedUpdateDiff, ApplyDiffError> {
    let mut parser = ParserState {
        lines: {
            let mut v = lines.to_vec();
            v.push(END_PATCH.to_string());
            v
        },
        index: 0,
        fuzz: 0,
    };

    let input_lines: Vec<String> = input.split('\n').map(|s| s.to_string()).collect();
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cursor: usize = 0;

    while !is_done(&parser, END_SECTION_MARKERS) {
        let anchor = read_str(&mut parser, "@@ ");
        let has_bare_anchor = anchor.is_empty()
            && parser.index < parser.lines.len()
            && parser.lines[parser.index] == "@@";

        if has_bare_anchor {
            parser.index += 1;
        }

        if anchor.is_empty() && !has_bare_anchor && cursor != 0 {
            let current_line = if parser.index < parser.lines.len() {
                parser.lines[parser.index].clone()
            } else {
                String::new()
            };
            return Err(ApplyDiffError::InvalidLine(current_line));
        }

        if !anchor.trim().is_empty() {
            cursor = advance_cursor_to_anchor(&anchor, &input_lines, cursor, &mut parser);
        }

        let section = read_section(&parser.lines, parser.index)?;
        let find_result = find_context(&input_lines, &section.next_context, cursor, section.eof);

        if find_result.new_index == usize::MAX {
            let ctx_text = section.next_context.join("\n");
            return Err(ApplyDiffError::InvalidContext {
                cursor,
                context: ctx_text,
                eof: section.eof,
            });
        }

        cursor = find_result.new_index + section.next_context.len();
        parser.fuzz += find_result.fuzz;

        parser.index = section.end_index;

        for ch in section.section_chunks {
            chunks.push(Chunk {
                orig_index: ch.orig_index + find_result.new_index,
                del_lines: ch.del_lines,
                ins_lines: ch.ins_lines,
            });
        }
    }

    Ok(ParsedUpdateDiff { chunks })
}

fn advance_cursor_to_anchor(
    anchor: &str,
    input_lines: &[String],
    mut cursor: usize,
    parser: &mut ParserState,
) -> usize {
    let mut found = false;

    for (i, line) in input_lines.iter().enumerate().skip(cursor) {
        if line == anchor {
            cursor = i + 1;
            found = true;
            break;
        }
    }

    if !found {
        for (i, line) in input_lines.iter().enumerate().skip(cursor) {
            if line.trim() == anchor.trim() {
                cursor = i + 1;
                parser.fuzz += 1;
                break;
            }
        }
    }

    cursor
}

fn read_section(lines: &[String], start_index: usize) -> Result<ReadSectionResult, ApplyDiffError> {
    let mut context: Vec<String> = Vec::new();
    let mut del_lines: Vec<String> = Vec::new();
    let mut ins_lines: Vec<String> = Vec::new();
    let mut section_chunks: Vec<Chunk> = Vec::new();

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Keep,
        Add,
        Delete,
    }

    let mut mode = Mode::Keep;
    let mut index = start_index;
    let orig_index = start_index;

    while index < lines.len() {
        let raw = &lines[index];

        if raw.starts_with("@@")
            || raw.starts_with(END_PATCH)
            || raw.starts_with("*** Update File:")
            || raw.starts_with("*** Delete File:")
            || raw.starts_with("*** Add File:")
            || raw.starts_with(END_FILE)
        {
            break;
        }

        if raw == "***" {
            break;
        }

        if raw.starts_with("***") {
            return Err(ApplyDiffError::InvalidLine(raw.clone()));
        }

        index += 1;

        let last_mode = mode;

        let line = if raw.is_empty() {
            " ".to_string()
        } else {
            raw.clone()
        };
        let prefix = line.chars().next().unwrap_or(' ');

        mode = match prefix {
            '+' => Mode::Add,
            '-' => Mode::Delete,
            ' ' => Mode::Keep,
            _ => return Err(ApplyDiffError::InvalidLine(line)),
        };

        let line_content = line[1..].to_string();
        let switching_to_context = mode == Mode::Keep && last_mode != mode;

        if switching_to_context && (!del_lines.is_empty() || !ins_lines.is_empty()) {
            section_chunks.push(Chunk {
                orig_index: context.len().saturating_sub(del_lines.len()),
                del_lines: del_lines.clone(),
                ins_lines: ins_lines.clone(),
            });
            del_lines.clear();
            ins_lines.clear();
        }

        match mode {
            Mode::Delete => {
                del_lines.push(line_content.clone());
                context.push(line_content);
            }
            Mode::Add => {
                ins_lines.push(line_content);
            }
            Mode::Keep => {
                context.push(line_content);
            }
        }
    }

    if !del_lines.is_empty() || !ins_lines.is_empty() {
        section_chunks.push(Chunk {
            orig_index: context.len().saturating_sub(del_lines.len()),
            del_lines,
            ins_lines,
        });
    }

    if index < lines.len() && lines[index] == END_FILE {
        return Ok(ReadSectionResult {
            next_context: context,
            section_chunks,
            end_index: index + 1,
            eof: true,
        });
    }

    if index == orig_index {
        let next_line = if index < lines.len() {
            lines[index].clone()
        } else {
            String::new()
        };
        return Err(ApplyDiffError::EmptySection { index, next_line });
    }

    Ok(ReadSectionResult {
        next_context: context,
        section_chunks,
        end_index: index,
        eof: false,
    })
}

#[derive(Debug)]
struct ContextMatch {
    new_index: usize,
    fuzz: i32,
}

fn find_context(lines: &[String], context: &[String], start: usize, eof: bool) -> ContextMatch {
    if eof {
        let end_start = lines.len().saturating_sub(context.len());
        let end_match = find_context_core(lines, context, end_start);
        if end_match.new_index != usize::MAX {
            return end_match;
        }
        let fallback = find_context_core(lines, context, start);
        return ContextMatch {
            new_index: fallback.new_index,
            fuzz: fallback.fuzz + 10_000,
        };
    }

    find_context_core(lines, context, start)
}

fn find_context_core(lines: &[String], context: &[String], start: usize) -> ContextMatch {
    if context.is_empty() {
        return ContextMatch {
            new_index: start,
            fuzz: 0,
        };
    }

    for i in start..lines.len() {
        if equals_slice(lines, context, i, |v| v.to_string()) {
            return ContextMatch {
                new_index: i,
                fuzz: 0,
            };
        }
    }

    for i in start..lines.len() {
        if equals_slice(lines, context, i, |v| v.trim_end().to_string()) {
            return ContextMatch {
                new_index: i,
                fuzz: 1,
            };
        }
    }

    for i in start..lines.len() {
        if equals_slice(lines, context, i, |v| v.trim().to_string()) {
            return ContextMatch {
                new_index: i,
                fuzz: 100,
            };
        }
    }

    ContextMatch {
        new_index: usize::MAX,
        fuzz: 0,
    }
}

fn equals_slice<F>(source: &[String], target: &[String], start: usize, map_fn: F) -> bool
where
    F: Fn(&str) -> String,
{
    if start + target.len() > source.len() {
        return false;
    }

    for (offset, target_value) in target.iter().enumerate() {
        if map_fn(&source[start + offset]) != map_fn(target_value) {
            return false;
        }
    }

    true
}

fn apply_chunks(input: &str, chunks: &[Chunk], newline: &str) -> Result<String, ApplyDiffError> {
    let orig_lines: Vec<String> = input.split('\n').map(|s| s.to_string()).collect();
    let mut dest_lines: Vec<String> = Vec::new();
    let mut cursor: usize = 0;

    for chunk in chunks {
        if chunk.orig_index > orig_lines.len() {
            return Err(ApplyDiffError::ChunkOutOfBounds {
                orig_index: chunk.orig_index,
                len: orig_lines.len(),
            });
        }

        if cursor > chunk.orig_index {
            return Err(ApplyDiffError::OverlappingChunk {
                orig_index: chunk.orig_index,
                cursor,
            });
        }

        dest_lines.extend_from_slice(&orig_lines[cursor..chunk.orig_index]);
        cursor = chunk.orig_index;

        if !chunk.ins_lines.is_empty() {
            dest_lines.extend(chunk.ins_lines.iter().cloned());
        }

        cursor += chunk.del_lines.len();
    }

    dest_lines.extend_from_slice(&orig_lines[cursor..]);
    Ok(dest_lines.join(newline))
}

pub fn apply_v4a_patch(original: &str, diff: &str, create: bool) -> Result<String> {
    let mode = if create {
        ApplyDiffMode::Create
    } else {
        ApplyDiffMode::Default
    };

    apply_v4a_diff(original, diff, mode).map_err(|e| anyhow!("{e:?}"))
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn normalize(s: &str) -> String {
        s.replace("\r\n", "\n")
    }

    #[test]
    fn test_apply_v4a_diff_with_floating_hunk_adds_lines() {
        let original = "a\nb\n";
        // Floating hunk: no explicit context, just a bare anchor and additions.
        let diff = "@@\n+X\n+Y\n*** End Patch\n";
        let result = apply_v4a_diff(original, diff, ApplyDiffMode::Default).unwrap();
        // Our implementation appends the inserted lines at the start (cursor 0).
        assert_eq!(normalize(&result), "X\nY\na\nb\n");
    }

    #[test]
    fn test_apply_v4a_diff_with_empty_input_and_crlf_diff_preserves_crlf() {
        let original = "";
        let diff = "+hello\r\n+world\r\n*** End Patch\r\n";
        let result = apply_v4a_diff(original, diff, ApplyDiffMode::Create).unwrap();
        assert!(result.contains("\r\n"));
        assert_eq!(result, "hello\r\nworld");
    }

    #[test]
    fn test_apply_v4a_diff_create_mode_requires_plus_prefix() {
        let diff = "hello\n";
        let err = apply_v4a_diff("", diff, ApplyDiffMode::Create).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Invalid Add File Line"));
    }

    #[test]
    fn test_apply_v4a_diff_create_mode_preserves_trailing_newline() {
        let diff = "+hello\n+world\n";
        let result = apply_v4a_diff("", diff, ApplyDiffMode::Create).unwrap();
        // No extra newline added/removed beyond what diff implies.
        assert_eq!(normalize(&result), "hello\nworld");
    }

    #[test]
    fn test_apply_v4a_diff_applies_contextual_replacement() {
        let original = "header\nline1\nline2\nline3\n";
        let diff = "@@ header\n line1\n-line2\n+LINE2\n line3\n";
        let result = apply_v4a_diff(original, diff, ApplyDiffMode::Default).unwrap();
        assert_eq!(normalize(&result), "header\nline1\nLINE2\nline3\n");
    }

    #[test]
    fn test_apply_v4a_diff_raises_on_context_mismatch() {
        let original = "a\nb\nc\n";
        let diff = "@@\n x\n-y\n z\n";
        let err = apply_v4a_diff(original, diff, ApplyDiffMode::Default).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Invalid Context") || msg.contains("Invalid EOF Context"));
    }

    #[test]
    fn test_apply_v4a_diff_with_crlf_input_and_lf_diff_preserves_crlf() {
        let original = "a\r\nb\r\nc\r\n";
        let diff = "@@\n a\n-b\n+B\n c\n";
        let result = apply_v4a_diff(original, diff, ApplyDiffMode::Default).unwrap();
        assert!(result.contains("\r\n"));
        assert_eq!(result, "a\r\nB\r\nc\r\n");
    }

    #[test]
    fn test_apply_v4a_diff_with_lf_input_and_crlf_diff_preserves_lf() {
        let original = "a\nb\nc\n";
        let diff = "@@\r\n a\r\n-b\r\n+B\r\n c\r\n";
        let result = apply_v4a_diff(original, diff, ApplyDiffMode::Default).unwrap();
        // Input is LF, so we keep LF even if diff uses CRLF.
        assert!(!result.contains("\r\n"));
        assert_eq!(normalize(&result), "a\nB\nc\n");
    }

    #[test]
    fn test_apply_v4a_diff_with_crlf_input_and_crlf_diff_preserves_crlf() {
        let original = "a\r\nb\r\nc\r\n";
        let diff = "@@\r\n a\r\n-b\r\n+B\r\n c\r\n";
        let result = apply_v4a_diff(original, diff, ApplyDiffMode::Default).unwrap();
        assert!(result.contains("\r\n"));
        assert_eq!(result, "a\r\nB\r\nc\r\n");
    }

    #[test]
    fn test_apply_v4a_diff_create_mode_preserves_crlf_newlines() {
        let original = "";
        let diff = "+a\r\n+b\r\n+c\r\n";
        let result = apply_v4a_diff(original, diff, ApplyDiffMode::Create).unwrap();
        assert!(result.contains("\r\n"));
        assert_eq!(result, "a\r\nb\r\nc");
    }
}
