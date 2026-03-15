use crate::chunk::btree_v1::read_btree_v1_entries;
use crate::chunk::btree_v2::read_btree_v2_chunk_entries;
use crate::chunk::extensible_array::read_extensible_array_entries;
use crate::chunk::fixed_array::read_fixed_array_entries;
use crate::chunk::implicit::read_implicit_chunk_entries;
use crate::chunk::single_chunk::read_single_chunk_entries;
use crate::error::Error;
use crate::error::Result;
use crate::filters::FilterPipeline;
use crate::io::ReadAt;
use crate::layout::ChunkIndexType;
use crate::layout::DataLayout;

/// Read a hyperslab from a chunked dataset, only reading overlapping chunks.
pub fn read_chunked_slice<R: ReadAt + ?Sized>(
    reader: &R,
    layout: &DataLayout,
    dataset_dims: &[u64],
    element_size: u32,
    filters: &Option<FilterPipeline>,
    size_of_offsets: u8,
    size_of_lengths: u8,
    max_dims: Option<&[u64]>,
    start: &[u64],
    count: &[u64],
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
    let actual_chunk_dims: Vec<u32>;
    let chunk_dims = if chunk_dims.len() == ndims + 1 {
        actual_chunk_dims = chunk_dims[..ndims].to_vec();
        &actual_chunk_dims
    } else {
        chunk_dims
    };

    let chunk_elems: u64 = chunk_dims.iter().map(|&d| d as u64).product();
    let chunk_byte_size = chunk_elems * element_size as u64;
    let elem = element_size as usize;

    // Output buffer for the selection
    let out_elems: u64 = count.iter().product();
    let mut output = vec![0u8; (out_elems * element_size as u64) as usize];

    // Compute output strides (row-major, in bytes)
    let mut out_strides = vec![elem; ndims];
    for i in (0..ndims - 1).rev() {
        out_strides[i] = out_strides[i + 1] * count[i + 1] as usize;
    }

    // If no data has been allocated, return the zeroed output buffer
    if address == u64::MAX {
        return Ok(output);
    }

    // Collect chunk entries (reuse the same logic as read_chunked)
    let entries = if layout_version == 3 {
        let v3_ndims = ndims + 1;
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

    // For each chunk, check if it overlaps the selection
    for entry in &entries {
        if entry.address == u64::MAX {
            continue;
        }

        // Chunk covers [chunk_start[i], chunk_end[i]) in each dimension
        let mut overlaps = true;
        for i in 0..ndims {
            let cs = entry.scaled[i] * chunk_dims[i] as u64;
            let ce = (cs + chunk_dims[i] as u64).min(dataset_dims[i]);
            let ss = start[i];
            let se = start[i] + count[i];
            if cs >= se || ce <= ss {
                overlaps = false;
                break;
            }
        }
        if !overlaps {
            continue;
        }

        // Read and decompress chunk
        let read_size = if entry.filtered_size > 0 {
            entry.filtered_size as usize
        } else {
            chunk_byte_size as usize
        };
        let mut chunk_data = vec![0u8; read_size];
        reader
            .read_exact_at(entry.address, &mut chunk_data)
            .map_err(Error::Io)?;
        if let Some(pipeline) = filters {
            if entry.filter_mask == 0 {
                chunk_data = pipeline.decompress(chunk_data)?;
            }
        }

        // Compute chunk strides
        let mut ch_strides = vec![elem; ndims];
        for i in (0..ndims - 1).rev() {
            ch_strides[i] = ch_strides[i + 1] * chunk_dims[i + 1] as usize;
        }

        // Compute overlap region in dataset coordinates
        let mut ov_start = vec![0u64; ndims];
        let mut ov_count = vec![0u64; ndims];
        for i in 0..ndims {
            let cs = entry.scaled[i] * chunk_dims[i] as u64;
            let ce = (cs + chunk_dims[i] as u64).min(dataset_dims[i]);
            ov_start[i] = start[i].max(cs);
            let ov_end = (start[i] + count[i]).min(ce);
            ov_count[i] = ov_end - ov_start[i];
        }

        // Copy the overlap region from chunk to output, row by row
        let inner_count = ov_count[ndims - 1] as usize * elem;
        let nrows: usize = ov_count[..ndims - 1]
            .iter()
            .map(|&c| c as usize)
            .product::<usize>()
            .max(1);

        for row in 0..nrows {
            let mut remaining = row;
            let mut src_off = 0usize;
            let mut dst_off = 0usize;

            for i in 0..ndims - 1 {
                let rows_below: usize = ov_count[i + 1..ndims - 1]
                    .iter()
                    .map(|&c| c as usize)
                    .product::<usize>()
                    .max(1);
                let idx = remaining / rows_below;
                remaining %= rows_below;
                // Source offset within chunk
                src_off += (ov_start[i] - entry.scaled[i] * chunk_dims[i] as u64) as usize
                    * ch_strides[i]
                    + idx * ch_strides[i];
                // Dest offset within output
                dst_off +=
                    (ov_start[i] - start[i]) as usize * out_strides[i] + idx * out_strides[i];
            }
            src_off += (ov_start[ndims - 1]
                - entry.scaled[ndims - 1] * chunk_dims[ndims - 1] as u64)
                as usize
                * elem;
            dst_off += (ov_start[ndims - 1] - start[ndims - 1]) as usize * elem;

            if src_off + inner_count <= chunk_data.len() && dst_off + inner_count <= output.len() {
                output[dst_off..dst_off + inner_count]
                    .copy_from_slice(&chunk_data[src_off..src_off + inner_count]);
            }
        }
    }

    Ok(output)
}
