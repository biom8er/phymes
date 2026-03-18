use arrow::array::ArrowPrimitiveType;

use crate::subject::indices_schema_builders::{BRINIndexBuilder, BTreeIndexBuilder, GINIndexBuilder, GiSTIndexBuilder, HashIndexBuilder, SPGiSTIndexBuilder};

/// B‑Tree Index
/// 
/// B-Trees store sorted keys in a balanced tree with internal nodes and leaf nodes. Ideal for equality and range queries.
pub struct BTreeIndex<K, V> {
    pub root: BTreeNode<K, V>,
}

pub enum BTreeNode<K, V> {
    Internal {
        id: u64,
        keys: Vec<K>,
        children: Vec<u64>,
    },
    Leaf {
        id: u64,
        keys: Vec<K>,
        values: Vec<V>,
        next_leaf: Option<u64>,
    },
}

impl<K, V> BTreeNode<K, V>
where
    K: ArrowPrimitiveType<Native = K>,
    V: ArrowPrimitiveType<Native = V>,
{
    pub fn push_into(
        &self,
        builder: &mut BTreeIndexBuilder<K, V>,
    ) {
        match self {
            Self::Internal { id, keys, children } => {
                builder.append_node(
                    *id,
                    false,
                    keys,
                    None,
                    Some(children),
                    None,
                );
            }
            Self::Leaf { id, keys, values, next_leaf } => {
                builder.append_node(
                    *id,
                    true,
                    keys,
                    Some(values.as_slice()),
                    None,
                    *next_leaf,
                );
            }
        }
    }
}

/// Hash Index
/// 
/// Stores hash buckets mapping hash values to tuples. Good for equality comparisons.
pub struct HashIndex<K, V> {
    pub buckets: Vec<HashEntry<K, V>>,
}

pub struct HashEntry<K, V> {
    pub bucket_id: u64,
    pub hash: u64,
    pub key: K,
    pub value: V,
}

impl<K, V> HashEntry<K, V>
where
    K: ArrowPrimitiveType<Native = K>,
    V: ArrowPrimitiveType<Native = V>,
{
    pub fn push_into(self, builder: &mut HashIndexBuilder<K, V>) {
        builder.append_entry(self.bucket_id, self.hash, self.key, self.value);
    }
}

/// GiST (Generalized Search Tree)
/// 
/// A balanced tree for extensible indexing (e.g., geometric, full‑text). Nodes store “bounding” predicates.
pub struct GiSTIndex<P, T> {
    pub root: GiSTEntry<P, T>,
}
pub enum GiSTEntry<P, T> {
    Internal {
        id: u64,
        predicate: P,
        child_id: u64,
    },
    Leaf {
        id: u64,
        predicate: P,
        tuple: T,
    },
}

impl<P, T> GiSTEntry<P, T>
where
    P: ArrowPrimitiveType<Native = P> + Clone,
    T: ArrowPrimitiveType<Native = T> + Clone,
{
    pub fn push_into(&self, builder: &mut GiSTIndexBuilder<P, T>) {
        match self {
            Self::Internal { id, predicate, child_id } => {
                builder.append_internal(*id, predicate.clone(), *child_id);
            }
            Self::Leaf { id, predicate, tuple } => {
                builder.append_leaf(*id, predicate.clone(), tuple.clone());
            }
        }
    }
}

/// SP‑GiST (Space‑Partitioned GiST)
/// 
/// Stores data in partitioned tries, quadtrees, or kd‑trees. Nodes route based on partitioning rules.
pub struct SPGiSTIndex<K, V> {
    pub root: SPGiSTNode<K, V>,
}
pub enum SPGiSTNode<K, V> {
    Inner {
        id: u64,
        label: K,
        children: Vec<u64>,
    },
    Leaf {
        id: u64,
        key: K,
        value: V,
    },
}

impl<K, V> SPGiSTNode<K, V>
where
    K: ArrowPrimitiveType<Native = K> + Clone,
    V: ArrowPrimitiveType<Native = V> + Clone,
{
    pub fn push_into(&self, builder: &mut SPGiSTIndexBuilder<K, V>) {
        match self {
            Self::Inner { id, label, children } => {
                builder.append_inner(*id, label.clone(), children);
            }
            Self::Leaf { id, key, value } => {
                builder.append_leaf(*id, key.clone(), value.clone());
            }
        }
    }
}

/// GIN (Generalized Inverted Index)
/// 
/// Ideal for arrays, JSONB, full‑text. Maps keys → posting lists of TIDs.
pub struct GINIndex<K> {
    pub entries: Vec<GINPosting<K>>,
}

pub struct GINPosting<K> {
    pub key: K,
    /// TIDs
    pub postings: Vec<u64>,
}

impl<K> GINPosting<K>
where
    K: ArrowPrimitiveType<Native = K> + Clone,
{
    pub fn push_into(&self, builder: &mut GINIndexBuilder<K>) {
        builder.append_entry(self.key.clone(), &self.postings);
    }
}

/// BRIN (Block Range Index)
///
/// Stores summaries (min/max, bloom, etc.) for physical block ranges. Very compact.
pub struct BRINIndex<S> {
    pub ranges: Vec<BRINRange<S>>,
}

pub struct BRINRange<S> {
    pub block_start: u64,
    pub block_end: u64,
    pub summary: S, // e.g., min/max, bloom, etc.
}

impl<S> BRINRange<S>
where
    S: ArrowPrimitiveType<Native = S> + Clone,
{
    pub fn push_into(&self, builder: &mut BRINIndexBuilder<S>) {
        builder.append_range(self.block_start, self.block_end, self.summary.clone());
    }
}