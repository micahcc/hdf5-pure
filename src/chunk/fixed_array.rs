use crate::checksum;
use crate::chunk::entry::ChunkEntry;
use crate::chunk::helpers::linear_to_scaled;
use crate::chunk::helpers::read_var_le;
use crate::error::Error;
use crate::error::Result;
use crate::io::Le;
use crate::io::ReadAt;

/// Fixed Array header magic: `FAHD`
const FAHD_MAGIC: [u8; 4] = *b"FAHD";
/// Fixed Array data block magic: `FADB`
const FADB_MAGIC: [u8; 4] = *b"FADB";

/// Read chunk entries from a Fixed Array index.
///
/// Fixed Array layout:
/// ```text
/// Header ("FAHD"):
///   magic(4) + version(1) + client_id(1) + entry_size(1) + max_dblk_page_nelmts_bits(1)
///   + nelmts(sizeof_lengths) + data_block_addr(sizeof_offsets) + checksum(4)
///
/// Data Block ("FADB"):
///   magic(4) + version(1) + client_id(1) + header_addr(sizeof_offsets)
///   + [optional page info if paged]
///   + entries(nelmts * entry_size)
///   + checksum(4)
/// ```
pub(crate) fn read_fixed_array_entries<R: ReadAt + ?Sized>(
    reader: &R,
    header_addr: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    has_filters: bool,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<ChunkEntry>> {
    let o = size_of_offsets as u64;
    let l = size_of_lengths as u64;

    // Parse Fixed Array header
    let mut magic = [0u8; 4];
    reader
        .read_exact_at(header_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != FAHD_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!("expected FAHD magic at {:#x}, got {:?}", header_addr, magic),
        });
    }

    let _version = Le::read_u8(reader, header_addr + 4).map_err(Error::Io)?;
    let _client_id = Le::read_u8(reader, header_addr + 5).map_err(Error::Io)?;
    let entry_size = Le::read_u8(reader, header_addr + 6).map_err(Error::Io)?;
    let _max_dblk_page_bits = Le::read_u8(reader, header_addr + 7).map_err(Error::Io)?;
    let nelmts = Le::read_length(reader, header_addr + 8, size_of_lengths).map_err(Error::Io)?;
    let dblk_addr =
        Le::read_offset(reader, header_addr + 8 + l, size_of_offsets).map_err(Error::Io)?;

    // Verify header checksum
    let hdr_size = (8 + l + o) as usize;
    let mut hdr_data = vec![0u8; hdr_size];
    reader
        .read_exact_at(header_addr, &mut hdr_data)
        .map_err(Error::Io)?;
    let stored_cksum = Le::read_u32(reader, header_addr + hdr_size as u64).map_err(Error::Io)?;
    let computed = checksum::lookup3(&hdr_data);
    if computed != stored_cksum {
        return Err(Error::ChecksumMismatch {
            expected: stored_cksum,
            actual: computed,
        });
    }

    if dblk_addr == u64::MAX {
        // No data block — all chunks are unallocated
        return Ok(Vec::new());
    }

    // Parse Fixed Array data block
    let mut db_magic = [0u8; 4];
    reader
        .read_exact_at(dblk_addr, &mut db_magic)
        .map_err(Error::Io)?;
    if db_magic != FADB_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected FADB magic at {:#x}, got {:?}",
                dblk_addr, db_magic
            ),
        });
    }

    // Data block prefix: magic(4) + version(1) + client_id(1) + hdr_addr(O)
    let db_prefix = 4 + 1 + 1 + o as usize;

    // TODO: handle paged data blocks (when max_dblk_page_bits > 0 and nelmts is large)
    // For now, assume non-paged (all entries in one contiguous block)

    // Read all entries
    let entries_start = dblk_addr + db_prefix as u64;
    let es = entry_size as usize;
    let ndims = dataset_dims.len();

    let chunks_per_dim: Vec<u64> = (0..ndims)
        .map(|i| dataset_dims[i].div_ceil(chunk_dims[i] as u64))
        .collect();

    let mut entries = Vec::with_capacity(nelmts as usize);

    for i in 0..nelmts {
        let entry_off = entries_start + i * es as u64;

        if has_filters {
            // Filtered entry: address(O) + chunk_nbytes(entry_size - O - 4) + filter_mask(4)
            let addr = Le::read_offset(reader, entry_off, size_of_offsets).map_err(Error::Io)?;
            let nbytes_size = es - size_of_offsets as usize - 4;
            let nbytes = read_var_le(reader, entry_off + o, nbytes_size)?;
            let fmask =
                Le::read_u32(reader, entry_off + o + nbytes_size as u64).map_err(Error::Io)?;

            let scaled = linear_to_scaled(i, &chunks_per_dim);
            entries.push(ChunkEntry {
                address: addr,
                filtered_size: nbytes,
                filter_mask: fmask,
                scaled,
            });
        } else {
            // Non-filtered entry: just address(O)
            let addr = Le::read_offset(reader, entry_off, size_of_offsets).map_err(Error::Io)?;
            let _chunk_byte_size: u64 = chunk_dims.iter().map(|&d| d as u64).product::<u64>()
                * (es as u64 / size_of_offsets as u64).max(1); // approximate

            let scaled = linear_to_scaled(i, &chunks_per_dim);
            // For non-filtered, the on-disk size equals the uncompressed chunk size.
            // We don't know element_size here, but the caller will figure it out
            // from the entry_size vs offset_size.
            entries.push(ChunkEntry {
                address: addr,
                filtered_size: 0, // will be set by caller
                filter_mask: 0,
                scaled,
            });
        }
    }

    Ok(entries)
}
