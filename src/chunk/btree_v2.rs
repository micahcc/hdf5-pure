use crate::chunk::entry::ChunkEntry;
use crate::error::Result;
use crate::io::ReadAt;

/// Read chunk entries from a B-tree v2 chunk index (layout v4, index type 5).
///
/// B-tree v2 type 10 records (non-filtered): address + scaled_offsets[ndims]
/// B-tree v2 type 11 records (filtered): address + nbytes + filter_mask + scaled_offsets[ndims]
///
/// Scaled offsets are always 8 bytes per dimension (UINT64DECODE).
pub(crate) fn read_btree_v2_chunk_entries<R: ReadAt + ?Sized>(
    reader: &R,
    bt2_addr: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    element_size: u32,
    has_filters: bool,
    size_of_offsets: u8,
    size_of_lengths: u8,
    _max_dims: Option<&[u64]>,
) -> Result<Vec<ChunkEntry>> {
    use crate::btree2::BTree2Header;
    use crate::btree2::{self};

    let ndims = dataset_dims.len();

    // Compute chunk_size_len for filtered records using HDF5 formula:
    // chunk_size_len = 1 + (floor(log2(chunk_size)) + 8) / 8, capped at 8
    // (H5Dbtree2.c H5D_BT2_COMPUTE_CHUNK_SIZE_LEN)
    let chunk_byte_size: u64 =
        chunk_dims.iter().map(|&d| d as u64).product::<u64>() * element_size as u64;
    let chunk_size_len = if has_filters {
        if chunk_byte_size == 0 {
            1
        } else {
            let log2 = 63 - chunk_byte_size.leading_zeros() as usize;
            (1 + (log2 + 8) / 8).min(8)
        }
    } else {
        0
    };

    // Parse the B-tree v2 header
    let bt2 = BTree2Header::parse(reader, bt2_addr, size_of_offsets, size_of_lengths)?;

    let o = size_of_offsets as usize;
    let mut entries = Vec::new();

    btree2::iterate_records(reader, &bt2, size_of_offsets, |record| {
        let data = &record.data;
        let mut pos = 0;

        // Read chunk address first
        let address = read_var_uint_slice(data, pos, o);
        pos += o;

        // For filtered records: nbytes + filter_mask come BEFORE scaled offsets
        // (H5Dbtree2.c H5D__bt2_filt_decode)
        let (filtered_size, filter_mask) = if has_filters {
            let nbytes = read_var_uint_slice(data, pos, chunk_size_len);
            pos += chunk_size_len;
            let fmask =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            (nbytes, fmask)
        } else {
            (0, 0)
        };

        // Read scaled offsets — fixed 8 bytes (UINT64DECODE) per dimension
        let mut scaled = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let val = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            scaled.push(val);
            pos += 8;
        }

        entries.push(ChunkEntry {
            address,
            filtered_size,
            filter_mask,
            scaled,
        });
        Ok(())
    })?;

    Ok(entries)
}

fn read_var_uint_slice(data: &[u8], offset: usize, size: usize) -> u64 {
    let mut result = 0u64;
    for i in 0..size.min(8) {
        if offset + i < data.len() {
            result |= (data[offset + i] as u64) << (i * 8);
        }
    }
    result
}
