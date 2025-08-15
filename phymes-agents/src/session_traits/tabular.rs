use anyhow::Result;
use phymes_core::{metrics::HashMap, table::arrow_table::ArrowTable};

static RESERVED_TABLE_NAMES: &[&str] = &["METRICS", "TASKS", "PROCESSORS", "SUBJECTS", "RUNTIME_ENVIRONMENTS"];
pub trait SessionContextBuilderTabularTrait {
    /// Convert the session into tables
    /// 
    /// # Notes
    /// 
    /// * All subjects that are a part of the state are included
    /// * Additional meta tables describing the SessionContext schema are included
    /// * Mermaid_js scripts are also included
    /// 
    /// # Arguments
    /// 
    /// * `include_subjects` - whether to include the subject data or not
    /// * `include_mermaid` - whether to include the mermaid flowchart and erDiagrams or not
    /// 
    /// # Returns
    /// 
    /// * `HashMap<String,ArrowTable` with the SessionContext in tabular format
    fn to_arrow_tables(&self, include_subjects: bool, include_mermaid: bool) -> Result<HashMap<String,ArrowTable>>;

    /// Create the session from tables
    /// 
    /// # Notes
    /// 
    /// * Minimally, the meta tables describing the SessionContext schema must be included
    /// * Optionally, the subject tables will be populated with data if the state tables are included
    /// * Mermaid_js scripts are ignored
    /// 
    /// # Arguments
    /// 
    /// * `tables>` - HashMap of [ArrowTable]s describing the [SessionContext] schema with
    ///   optional subject tables with the actual data
    fn from_arrow_tables(tables: HashMap<String,ArrowTable>) -> Result<Self> where Self: Sized;
}