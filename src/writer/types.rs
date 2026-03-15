use crate::datatype::Datatype;
use crate::writer::DatasetNode;
use crate::writer::GroupNode;

/// A filter to apply in a chunked dataset's filter pipeline.
#[derive(Debug, Clone)]
pub enum ChunkFilter {
    /// Deflate (zlib) compression with a given level (0-9).
    Deflate(u32),
    /// Shuffle filter — reorders bytes for better compression.
    Shuffle,
    /// Fletcher32 checksum appended to each chunk.
    Fletcher32,
}

/// Storage layout for a dataset.
#[derive(Debug, Clone, Default)]
pub enum StorageLayout {
    /// Data stored in a contiguous block after the object header.
    #[default]
    Contiguous,
    /// Data stored inline in the object header (small datasets only).
    Compact,
    /// Data stored in fixed-size chunks.
    Chunked {
        chunk_dims: Vec<u64>,
        filters: Vec<ChunkFilter>,
    },
}

pub(crate) enum ChildNode {
    Group(GroupNode),
    Dataset(DatasetNode),
}

pub(crate) struct AttrData {
    pub name: String,
    pub datatype: Datatype,
    pub shape: Vec<u64>,
    pub value: Vec<u8>,
}
