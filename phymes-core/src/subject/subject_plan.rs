

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SubjectPlan {
    pub location: String,
    pub bucket: String,
    pub metadata: Map<String, Value>,
    pub schema: Vec<u8>,
    pub last_modified: i64,
}