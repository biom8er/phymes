use serde::{Deserialize, Serialize};

/// Indexing method used by that index
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize)]
pub enum IndexType {
    /// B-Tree index (commonly default for many databases).
    BTree,
    /// Hash index.
    Hash,
    /// Generalized Inverted Index (GIN).
    GIN,
    /// Generalized Search Tree (GiST) index.
    GiST,
    /// Space-partitioned GiST (SPGiST) index.
    SPGiST,
    /// Block Range Index (BRIN).
    BRIN,
    /// Bloom filter based index.
    Bloom,
    /// Users may define their own index types, which would
    /// not be covered by the above variants.
    Custom(String),
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::BTree => write!(f, "BTREE"),
            Self::Hash => write!(f, "HASH"),
            Self::GIN => write!(f, "GIN"),
            Self::GiST => write!(f, "GIST"),
            Self::SPGiST => write!(f, "SPGIST"),
            Self::BRIN => write!(f, "BRIN"),
            Self::Bloom => write!(f, "BLOOM"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}