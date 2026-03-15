use crate::chunk::entry::ChunkEntry;
use crate::chunk::helpers::linear_to_scaled;
use crate::error::Result;

/// Implicit index: chunks are stored contiguously, no index structure.
/// Chunk at scaled coordinates is at: address + linear_index * chunk_byte_size.
pub(crate) fn read_implicit_chunk_entries(
    address: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    chunk_byte_size: u64,
) -> Result<Vec<ChunkEntry>> {
    let ndims = dataset_dims.len();
    let chunks_per_dim: Vec<u64> = (0..ndims)
        .map(|i| dataset_dims[i].div_ceil(chunk_dims[i] as u64))
        .collect();

    let total_chunks: u64 = chunks_per_dim.iter().product();
    let mut entries = Vec::with_capacity(total_chunks as usize);

    for linear in 0..total_chunks {
        let scaled = linear_to_scaled(linear, &chunks_per_dim);
        entries.push(ChunkEntry {
            address: address + linear * chunk_byte_size,
            filtered_size: chunk_byte_size,
            filter_mask: 0,
            scaled,
        });
    }

    Ok(entries)
}
