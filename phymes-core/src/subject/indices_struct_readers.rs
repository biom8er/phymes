use arrow::array::{
    Array, ArrowPrimitiveType, BooleanArray, ListArray, PrimitiveArray, RecordBatch, UInt64Array
};

use crate::subject::indices_structs::{BRINRange, BTreeNode, GINPosting, GiSTEntry, HashEntry, SPGiSTNode};

pub struct BTreeIndexReader<'a, K: ArrowPrimitiveType, V: ArrowPrimitiveType> {
    node_id: &'a UInt64Array,
    is_leaf: &'a BooleanArray,
    keys: &'a ListArray,
    values: &'a ListArray,
    children: &'a ListArray,
    next_leaf: &'a UInt64Array,
    phantom_k: std::marker::PhantomData<K>,
    phantom_v: std::marker::PhantomData<V>,
}

impl<'a, K, V> BTreeIndexReader<'a, K, V>
where
    K: ArrowPrimitiveType,
    V: ArrowPrimitiveType,
{
    pub fn new(batch: &'a RecordBatch) -> Self {
        Self {
            node_id: batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap(),
            is_leaf: batch.column(1).as_any().downcast_ref::<BooleanArray>().unwrap(),
            keys: batch.column(2).as_any().downcast_ref::<ListArray>().unwrap(),
            values: batch.column(3).as_any().downcast_ref::<ListArray>().unwrap(),
            children: batch.column(4).as_any().downcast_ref::<ListArray>().unwrap(),
            next_leaf: batch.column(5).as_any().downcast_ref::<UInt64Array>().unwrap(),
            phantom_k: std::marker::PhantomData,
            phantom_v: std::marker::PhantomData,
        }
    }

    pub fn get(&self, row: usize) -> BTreeNode<K::Native, V::Native> {
        let id = self.node_id.value(row);
        let leaf = self.is_leaf.value(row);

        let keys_arr = self.keys.value(row);
        let keys = keys_arr
            .as_any()
            .downcast_ref::<PrimitiveArray<K>>()
            .unwrap()
            .values()
            .to_vec();

        if leaf {
            let values_arr = self.values.value(row);
            let values = values_arr
                .as_any()
                .downcast_ref::<PrimitiveArray<V>>()
                .unwrap()
                .values()
                .to_vec();

            let next_leaf = if self.next_leaf.is_null(row) {
                None
            } else {
                Some(self.next_leaf.value(row))
            };

            BTreeNode::Leaf { id, keys, values, next_leaf }
        } else {
            let children_arr = self.children.value(row);
            let children = children_arr
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .to_vec();

            BTreeNode::Internal { id, keys, children }
        }
    }
}

pub struct HashIndexReader<'a, K: ArrowPrimitiveType, V: ArrowPrimitiveType> {
    bucket_id: &'a UInt64Array,
    hash: &'a UInt64Array,
    key: &'a PrimitiveArray<K>,
    value: &'a PrimitiveArray<V>,
}

impl<'a, K, V> HashIndexReader<'a, K, V>
where
    K: ArrowPrimitiveType,
    V: ArrowPrimitiveType,
{
    pub fn new(batch: &'a RecordBatch) -> Self {
        Self {
            bucket_id: batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap(),
            hash: batch.column(1).as_any().downcast_ref::<UInt64Array>().unwrap(),
            key: batch.column(2).as_any().downcast_ref::<PrimitiveArray<K>>().unwrap(),
            value: batch.column(3).as_any().downcast_ref::<PrimitiveArray<V>>().unwrap(),
        }
    }

    pub fn get(&self, row: usize) -> HashEntry<K::Native, V::Native> {
        HashEntry {
            bucket_id: self.bucket_id.value(row),
            hash: self.hash.value(row),
            key: self.key.value(row),
            value: self.value.value(row),
        }
    }
}

pub struct GiSTIndexReader<'a, P: ArrowPrimitiveType, T: ArrowPrimitiveType> {
    node_id: &'a UInt64Array,
    is_leaf: &'a BooleanArray,
    predicate: &'a PrimitiveArray<P>,
    child_id: &'a UInt64Array,
    tuple: &'a PrimitiveArray<T>,
}

impl<'a, P, T> GiSTIndexReader<'a, P, T>
where
    P: ArrowPrimitiveType,
    T: ArrowPrimitiveType,
{
    pub fn new(batch: &'a RecordBatch) -> Self {
        Self {
            node_id: batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap(),
            is_leaf: batch.column(1).as_any().downcast_ref::<BooleanArray>().unwrap(),
            predicate: batch.column(2).as_any().downcast_ref::<PrimitiveArray<P>>().unwrap(),
            child_id: batch.column(3).as_any().downcast_ref::<UInt64Array>().unwrap(),
            tuple: batch.column(4).as_any().downcast_ref::<PrimitiveArray<T>>().unwrap(),
        }
    }

    pub fn get(&self, row: usize) -> GiSTEntry<P::Native, T::Native> {
        let id = self.node_id.value(row);
        let pred = self.predicate.value(row);

        if self.is_leaf.value(row) {
            GiSTEntry::Leaf {
                id,
                predicate: pred,
                tuple: self.tuple.value(row),
            }
        } else {
            GiSTEntry::Internal {
                id,
                predicate: pred,
                child_id: self.child_id.value(row),
            }
        }
    }
}

pub struct SPGiSTIndexReader<'a, K: ArrowPrimitiveType, V: ArrowPrimitiveType> {
    node_id: &'a UInt64Array,
    is_leaf: &'a BooleanArray,
    label: &'a PrimitiveArray<K>,
    child_ids: &'a ListArray,
    key: &'a PrimitiveArray<K>,
    value: &'a PrimitiveArray<V>,
}

impl<'a, K, V> SPGiSTIndexReader<'a, K, V>
where
    K: ArrowPrimitiveType,
    V: ArrowPrimitiveType,
{
    pub fn new(batch: &'a RecordBatch) -> Self {
        Self {
            node_id: batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap(),
            is_leaf: batch.column(1).as_any().downcast_ref::<BooleanArray>().unwrap(),
            label: batch.column(2).as_any().downcast_ref::<PrimitiveArray<K>>().unwrap(),
            child_ids: batch.column(3).as_any().downcast_ref::<ListArray>().unwrap(),
            key: batch.column(4).as_any().downcast_ref::<PrimitiveArray<K>>().unwrap(),
            value: batch.column(5).as_any().downcast_ref::<PrimitiveArray<V>>().unwrap(),
        }
    }

    pub fn get(&self, row: usize) -> SPGiSTNode<K::Native, V::Native> {
        let id = self.node_id.value(row);

        if self.is_leaf.value(row) {
            SPGiSTNode::Leaf {
                id,
                key: self.key.value(row),
                value: self.value.value(row),
            }
        } else {
            let children_arr = self.child_ids.value(row);
            let children = children_arr
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .to_vec();

            SPGiSTNode::Inner {
                id,
                label: self.label.value(row),
                children,
            }
        }
    }
}

pub struct GINIndexReader<'a, K: ArrowPrimitiveType> {
    key: &'a PrimitiveArray<K>,
    posting_list: &'a ListArray,
}

impl<'a, K> GINIndexReader<'a, K>
where
    K: ArrowPrimitiveType,
{
    pub fn new(batch: &'a RecordBatch) -> Self {
        Self {
            key: batch.column(0).as_any().downcast_ref::<PrimitiveArray<K>>().unwrap(),
            posting_list: batch.column(1).as_any().downcast_ref::<ListArray>().unwrap(),
        }
    }

    pub fn get(&self, row: usize) -> GINPosting<K::Native> {
        let postings_arr = self.posting_list.value(row);
        let postings = postings_arr
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();

        GINPosting {
            key: self.key.value(row),
            postings,
        }
    }
}

pub struct BRINIndexReader<'a, S: ArrowPrimitiveType> {
    block_start: &'a UInt64Array,
    block_end: &'a UInt64Array,
    summary: &'a PrimitiveArray<S>,
}

impl<'a, S> BRINIndexReader<'a, S>
where
    S: ArrowPrimitiveType,
{
    pub fn new(batch: &'a RecordBatch) -> Self {
        Self {
            block_start: batch.column(0).as_any().downcast_ref::<UInt64Array>().unwrap(),
            block_end: batch.column(1).as_any().downcast_ref::<UInt64Array>().unwrap(),
            summary: batch.column(2).as_any().downcast_ref::<PrimitiveArray<S>>().unwrap(),
        }
    }

    pub fn get(&self, row: usize) -> BRINRange<S::Native> {
        BRINRange {
            block_start: self.block_start.value(row),
            block_end: self.block_end.value(row),
            summary: self.summary.value(row),
        }
    }
}