pub mod agents;
pub mod mermaid_js;
pub mod tabular;

pub enum BuilderJourney {
    Start,
    Name(String),
    Plan((String, String)),
    End,    
}