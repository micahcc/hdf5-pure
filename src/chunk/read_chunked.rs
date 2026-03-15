use crate::chunk::btree_v1::read_btree_v1_entries;
use crate::chunk::btree_v2::read_btree_v2_chunk_entries;
use crate::chunk::extensible_array::read_extensible_array_entries;
use crate::chunk::fixed_array::read_fixed_array_entries;
use crate::chunk::helpers::copy_chunk_to_output;
use crate::chunk::implicit::read_implicit_chunk_entries;
use crate::chunk::single_chunk::read_single_chunk_entries;
use crate::error::Error;
use crate::error::Result;
use crate::filters::FilterPipeline;
use crate::io::ReadAt;
use crate::layout::ChunkIndexType;
use crate::layout::DataLayout;

/// Read a chunked dataset, assembling all chunks into a contiguous buffer.
///
/// `chunk_dims`: the chunk dimensions (element counts per axis).
/// `dataset_dims`: the dataset dimensions.
/// `element_size`: size of one element in bytes.
/// `layout`: the parsed DataLayout::Chunked.
/// `filters`: optional filter pipeline.
/// `max_dims`: optional max dimensions from dataspace (needed for B-tree v2 chunk index).
pub fn read_chunked<R: ReadAt + ?Sized>(
    reader: &R,
    layout: &DataLayout,
    dataset_dims: &[u64],
    element_size: u32,
    filters: &Option<FilterPipeline>,
    size_of_offsets: u8,
    size_of_lengths: u8,
    max_dims: Option<&[u64]>,
) -> Result<Vec<u8>> {
    let (
        chunk_dims,
        address,
        chunk_index_type,
        _chunk_flags,
        layout_version,
        single_filtered_size,
        single_filter_mask,
    ) = match layout {
        DataLayout::Chunked {
            chunk_dims,
            address,
            chunk_index_type,
            chunk_flags,
            layout_version,
            single_chunk_filtered_size,
            single_chunk_filter_mask,
            ..
        } => (
            chunk_dims,
            *address,
            chunk_index_type,
            *chunk_flags,
            *layout_version,
            *single_chunk_filtered_size,
            *single_chunk_filter_mask,
        ),
        _ => {
            return Err(Error::InvalidLayout {
                msg: "expected chunked layout".into(),
            });
        }
    };

    let ndims = dataset_dims.len();

    // Both layout v3 and v4 store dimensionality = rank + 1 for chunked
    // datasets, with the last dimension being the element size in bytes.
    // Strip the extra dimension so we work with just the dataset-rank dimensions.
    let actual_chunk_dims: Vec<u32>;
    let chunk_dims = if chunk_dims.len() == ndims + 1 {
        actual_chunk_dims = chunk_dims[..ndims].to_vec();
        &actual_chunk_dims
    } else {
        chunk_dims
    };

    // Compute the uncompressed chunk size in bytes
    let chunk_elems: u64 = chunk_dims.iter().map(|&d| d as u64).product();
    let chunk_byte_size = chunk_elems * element_size as u64;

    // Compute total output size
    let total_elems: u64 = dataset_dims.iter().product();
    let total_size = total_elems * element_size as u64;
    let mut output = vec![0u8; total_size as usize];

    // If no data has been allocated, return the zeroed output buffer
    if address == u64::MAX {
        return Ok(output);
    }

    // Determine chunk index type and collect entries
    let entries = if layout_version == 3 {
        // Layout v3 always uses B-tree v1
        let v3_ndims = ndims + 1; // dimensionality in v3 = rank + 1
        read_btree_v1_entries(
            reader,
            address,
            dataset_dims,
            chunk_dims,
            v3_ndims,
            size_of_offsets,
        )?
    } else {
        let idx_type = chunk_index_type.unwrap_or(ChunkIndexType::SingleChunk);
        match idx_type {
            ChunkIndexType::SingleChunk => read_single_chunk_entries(
                address,
                chunk_byte_size,
                single_filtered_size,
                single_filter_mask,
                ndims,
            )?,
            ChunkIndexType::Implicit => {
                read_implicit_chunk_entries(address, dataset_dims, chunk_dims, chunk_byte_size)?
            }
            ChunkIndexType::FixedArray => read_fixed_array_entries(
                reader,
                address,
                dataset_dims,
                chunk_dims,
                filters.is_some(),
                size_of_offsets,
                size_of_lengths,
            )?,
            ChunkIndexType::ExtensibleArray => read_extensible_array_entries(
                reader,
                address,
                dataset_dims,
                chunk_dims,
                filters.is_some(),
                size_of_offsets,
                size_of_lengths,
            )?,
            ChunkIndexType::BTreeV2 => read_btree_v2_chunk_entries(
                reader,
                address,
                dataset_dims,
                chunk_dims,
                element_size,
                filters.is_some(),
                size_of_offsets,
                size_of_lengths,
                max_dims,
            )?,
        }
    };

    // Read each chunk and place it into the output buffer
    for entry in &entries {
        if entry.address == u64::MAX {
            // Chunk not allocated — leave as zeros (fill value)
            continue;
        }

        // Read the raw (possibly compressed) chunk data.
        // For unfiltered entries, filtered_size may be 0 — use chunk_byte_size instead.
        let read_size = if entry.filtered_size > 0 {
            entry.filtered_size as usize
        } else {
            chunk_byte_size as usize
        };
        let mut chunk_data = vec![0u8; read_size];
        reader
            .read_exact_at(entry.address, &mut chunk_data)
            .map_err(Error::Io)?;

        // Apply filter pipeline in reverse (decompress)
        if let Some(pipeline) = filters {
            if entry.filter_mask == 0 {
                // All filters applied
                chunk_data = pipeline.decompress(chunk_data)?;
            }
            // TODO: handle partial filter masks
        }

        // Place chunk data into the output buffer at the correct position
        copy_chunk_to_output(
            &chunk_data,
            &entry.scaled,
            chunk_dims,
            dataset_dims,
            element_size as usize,
            &mut output,
        );
    }

    Ok(output)
}
