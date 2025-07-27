use new_string_template::template::Template;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

/// Update with Message struct from TGI
#[derive(Default, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    role: String,
    content: String, // Needs to include multiple content
}

impl Message {
    pub fn new(role: String, content: String) -> Self {
        Message { role, content }
    }
}

pub type Messages = Vec<Message>;

// See https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py#L208
// `get_json_schema` for deriving a JSON schema for a arbitrary function or object
// JSON type is derived from serde_json and represented as a string input
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl Tool {
    pub fn new(name: String, description: String, parameters: serde_json::Value) -> Self {
        Tool {
            name,
            description,
            parameters,
        }
    }
}

impl Default for Tool {
    fn default() -> Self {
        let name = "multiply".to_string();
        let description = "A function that multiplies two numbers".to_string();
        let parameters = json!({
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number to multiply"},
                "y": {"type": "number", "description": "The second number to multiply"}
            },
            "required": ["x", "y"]
        });
        Tool::new(name, description, parameters)
    }
}

pub type Tools = Vec<Tool>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    title: String,
    text: String,
}

impl Document {
    pub fn new(title: String, text: String) -> Self {
        Document { title, text }
    }
}

impl Default for Document {
    fn default() -> Self {
        let title = "Title".to_string();
        let text = "The super informative document".to_string();
        Document::new(title, text)
    }
}

pub type Documents = Vec<Document>;

pub const SYSTEM_TEMPLATE: &str = r#"You are an expert AI assistant. You are given a question, a set of possible functions/tools, and a set of possible documents. 
Based on the question, you may need to make one or more function/tool calls to achieve the purpose.
If none of the functions/tools can be used, point it out. 
If the given question lacks the parameters required by the function/tool, also point it out.

You have access to the following tools:
<tools>{tools}</tools>

Given the context information and not prior knowledge, answer the question and provide citations from the documents.
If none of the documents are required to answer the question, point it out. 

You have access to the following documents:
<documents>{documents}</documents>"#;

/**
Returns an updated Messages
    See <https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/instructions_function_calling.md>
    for inspiration of the function

# Arguments

* `query` - String query from the user role
* `system_prompt` - Optional Template system prompt with variables for "tools" and "documents"
* `tools` - Optional set of Tools that can be called
* `documents` - Optional set of Documents that can be used
* `history` - Optional history of Messages to include

*/
pub fn prepare_messages(
    query: &String,
    system_prompt: Option<&str>,
    tools: &Option<Tools>,
    documents: &Option<Documents>,
    history: &Option<Messages>,
) -> anyhow::Result<Messages> {
    let mut tools_documents_map: HashMap<&str, String> = HashMap::new();
    match tools {
        Some(t) => tools_documents_map.insert("tools", serde_json::to_string(&t)?),
        None => tools_documents_map.insert("tools", "[]".to_string()),
    };

    match documents {
        Some(d) => tools_documents_map.insert("documents", serde_json::to_string(&d)?),
        None => tools_documents_map.insert("documents", "[]".to_string()),
    };

    let system_content = match system_prompt {
        Some(p) => Template::new(p),
        None => Template::new(SYSTEM_TEMPLATE),
    };

    let messages: Messages = match history {
        Some(h) => {
            let mut messages = h.clone();
            messages.push(Message::new("user".to_string(), query.to_string()));
            messages
        }
        None => {
            let messages: Messages = vec![
                Message::new(
                    "system".to_string(),
                    system_content.render_nofail(&tools_documents_map),
                ),
                Message::new("user".to_string(), query.to_string()),
            ];
            messages
        }
    };

    Ok(messages)
}

pub const CHAT_TEMPLATE: &str = "<|im_start|>{role}\n{content}<|im_end|>\n";
pub const GENERATION_PROMPT: &str = "<|im_start|>assistant\n";

/**
Returns a new prompt String

# Notes

  See <https://github.com/huggingface/transformers/blob/main/docs/source/en/chat_templating.md> and
  see <https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/tokenization_utils_base.py#L152>
  for inspiration of the function

# Arguments

* `Messages` - Messages that have been prepared using `prepare_messages`
* `chat_template` - Optional string literal for the chat prompt
* `add_generation_prompt` - Optional String
  If this is set, a prompt with the token(s) that indicate
  the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
  Note that this argument will be passed to the chat template, and so it must be supported in the
  template for this argument to have any effect.
* `continue_final_message` - Optional String
  If this is set, the chat will be formatted so that the final
  message in the chat is open-ended, without any EOS tokens. The model will continue this message
  rather than starting a new one. This allows you to "prefill" part of
  the model's response for it. Cannot be used at the same time as `add_generation_prompt`.

*/
pub fn create_prompt_chat(
    conversation: Messages,
    chat_template: Option<&str>,
    add_generation_prompt: &Option<String>,
    continue_final_message: &Option<String>,
) -> anyhow::Result<String> {
    let prompt = match chat_template {
        Some(p) => Template::new(p),
        None => Template::new(CHAT_TEMPLATE),
    };

    let generation_prompt = match add_generation_prompt {
        Some(agp) => agp,
        None => match continue_final_message {
            Some(cfm) => cfm,
            None => &GENERATION_PROMPT.to_string(),
        },
    };

    let mut prompt_str = String::new();
    for message in conversation {
        let mut map = HashMap::new();
        map.insert("role", message.role);
        map.insert("content", message.content);
        let mstr = prompt.clone().render_nofail(&map);
        prompt_str.push_str(mstr.as_str());
    }
    prompt_str.push_str(generation_prompt);

    Ok(prompt_str)
}
