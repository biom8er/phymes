use anyhow::Result;
use tracing::instrument;

use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;

/// Raise a exception (custom function) used in the chat templates
#[allow(dead_code)]
pub(crate) fn raise_exception(err_text: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ArrowTableScript {
    template: Template<'static, 'static>,
}

/// Convert from arrow table to arrow script input
///
/// # Note
///
/// Not all Jinja2 templates can be copy-n-pasted
///   e.g., <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json>
///   The problem appears to be with quotations
///   i.e., \" as the opening quotation does not work and instead ' or " should be used
///   Recommend testing problematic templates using the Minijinja playground
///   <https://github.com/mitsuhiko/minijinja>
impl ArrowTableScript {
    /// Write record batches to a script described by a template
    /// calls `to_json_object` to populate the template
    #[instrument(level = "trace")]
    pub fn new_from_template(template: String) -> Self {
        let mut env = Box::new(Environment::new());

        // enable things like .strip() or .capitalize()
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        let template_str = template.into_boxed_str();
        env.add_function("raise_exception", raise_exception);

        // leaking env and template_str as read-only, static resources for performance.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .unwrap();

        Self { template }
    }

    /// Apply the template to a json object
    /// The record_batch should be attached to the json_data
    pub fn apply_template(&self, json_data: &serde_json::Value) -> Result<String> {
        let rendered_template = self.template.render(json_data)?;
        Ok(rendered_template)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::arrow_table::{ArrowTableTrait, test_table::make_test_table_chat};

    #[test]
    fn test_to_from_script_with_template() -> Result<()> {
        let test_table = make_test_table_chat("test_table_chat")?;

        // Write data to script template
        let template = r#"
        {% for message in messages %}
            {% if message['role'] == 'system' %}
                {% if message['content']%}
                    {{'### System:\n' + message['content']+'\n\n'}}
                {% endif %}
            {% elif message['role'] == 'user' %}
                {{'### User:\n' + message['content']+'\n\n'}}
            {% elif message['role'] == 'assistant' %}
                {{'### Assistant:\n'  + message['content']}}
            {% endif %}
            {% if loop.last and add_generation_prompt %}
                {{ '### Assistant:\n' }}
            {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let template = template
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let chat_template_inputs = serde_json::json!({
            "messages": test_table.to_json_object()?,
            "bos_token": Some("[BOS]"),
            "eos_token": Some("[EOS]"),
            "add_generation_prompt": true,
        });

        let script_string =
            ArrowTableScript::new_from_template(template).apply_template(&chat_template_inputs)?;

        assert_eq!(
            script_string,
            "### User:\nHi!\n\n### Assistant:\nHello how can I help?### User:\nWhat is Deep Learning?\n\n### Assistant:\nmagic!### Assistant:\n"
        );

        Ok(())
    }
}
