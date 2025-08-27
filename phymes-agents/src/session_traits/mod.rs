use phymes_core::metrics::HashMap;

pub mod agents;
pub mod mermaid_js;
pub mod tabular;

pub enum BuilderJourney {
    Start,
    Name(String),
    Plan((String, String)),
    Entry(HashMap<String, String>),
    End,    
}