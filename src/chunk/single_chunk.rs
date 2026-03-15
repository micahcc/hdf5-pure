use crate::chunk::entry::ChunkEntry;
use crate::error::Result;

/// Single chunk: there's exactly one chunk covering the entire dataset.
pub(crate) fn read_single_chunk_entries(
    address: u64,
    chunk_byte_size: u64,
    filtered_size: Option<u64>,
    filter_mask: Option<u32>,
    ndims: usize,
) -> Result<Vec<ChunkEntry>> {
    Ok(vec![ChunkEntry {
        address,
        filtered_size: filtered_size.unwrap_or(chunk_byte_size),
        filter_mask: filter_mask.unwrap_or(0),
        scaled: vec![0; ndims],
    }])
}
