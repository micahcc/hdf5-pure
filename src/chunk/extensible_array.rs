use crate::chunk::entry::ChunkEntry;
use crate::chunk::helpers::linear_to_scaled;
use crate::error::Error;
use crate::error::Result;
use crate::io::Le;
use crate::io::ReadAt;

/// Extensible Array header magic: `EAHD`
const EAHD_MAGIC: [u8; 4] = *b"EAHD";
/// Extensible Array index block magic: `EAIB`
const EAIB_MAGIC: [u8; 4] = *b"EAIB";
/// Extensible Array super block magic: `EASB`
const EASB_MAGIC: [u8; 4] = *b"EASB";
/// Extensible Array data block magic: `EADB`
const EADB_MAGIC: [u8; 4] = *b"EADB";

/// Read chunk entries from an Extensible Array index.
///
/// The extensible array stores chunk addresses in a tiered structure:
/// 1. First `idx_blk_elmts` elements are stored directly in the index block
/// 2. Additional elements in data blocks addressed from the index block
/// 3. Further elements in data blocks addressed via super blocks
pub(crate) fn read_extensible_array_entries<R: ReadAt + ?Sized>(
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

    // Parse EA header
    let mut magic = [0u8; 4];
    reader
        .read_exact_at(header_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != EAHD_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!("expected EAHD magic at {:#x}, got {:?}", header_addr, magic),
        });
    }

    let _version = Le::read_u8(reader, header_addr + 4).map_err(Error::Io)?;
    let _client_id = Le::read_u8(reader, header_addr + 5).map_err(Error::Io)?;
    let elmt_size = Le::read_u8(reader, header_addr + 6).map_err(Error::Io)? as usize;
    let max_nelmts_bits = Le::read_u8(reader, header_addr + 7).map_err(Error::Io)?;
    let idx_blk_elmts = Le::read_u8(reader, header_addr + 8).map_err(Error::Io)? as u64;
    let data_blk_min_elmts = Le::read_u8(reader, header_addr + 9).map_err(Error::Io)? as u64;
    let sup_blk_min_data_ptrs = Le::read_u8(reader, header_addr + 10).map_err(Error::Io)? as u64;
    let max_dblk_page_nelmts_bits = Le::read_u8(reader, header_addr + 11).map_err(Error::Io)?;

    // 6 stats (each sizeof_lengths)
    let stats_start = header_addr + 12;
    let _nsuper_blks = Le::read_length(reader, stats_start, size_of_lengths).map_err(Error::Io)?;
    let _super_blk_size =
        Le::read_length(reader, stats_start + l, size_of_lengths).map_err(Error::Io)?;
    let _ndata_blks =
        Le::read_length(reader, stats_start + 2 * l, size_of_lengths).map_err(Error::Io)?;
    let _data_blk_size =
        Le::read_length(reader, stats_start + 3 * l, size_of_lengths).map_err(Error::Io)?;
    let max_idx_set =
        Le::read_length(reader, stats_start + 4 * l, size_of_lengths).map_err(Error::Io)?;

    let idx_blk_addr =
        Le::read_offset(reader, stats_start + 6 * l, size_of_offsets).map_err(Error::Io)?;

    if idx_blk_addr == u64::MAX || max_idx_set == 0 {
        return Ok(Vec::new());
    }

    // Parse index block
    let mut ib_magic = [0u8; 4];
    reader
        .read_exact_at(idx_blk_addr, &mut ib_magic)
        .map_err(Error::Io)?;
    if ib_magic != EAIB_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected EAIB magic at {:#x}, got {:?}",
                idx_blk_addr, ib_magic
            ),
        });
    }

    // Index block prefix: magic(4) + version(1) + client_id(1) + hdr_addr(O)
    let ib_prefix = 4 + 1 + 1 + o as usize;
    let elmts_start = idx_blk_addr + ib_prefix as u64;

    let ndims = dataset_dims.len();
    let chunks_per_dim: Vec<u64> = (0..ndims)
        .map(|i| dataset_dims[i].div_ceil(chunk_dims[i] as u64))
        .collect();

    let mut entries = Vec::new();
    let es = elmt_size;

    // Read elements directly stored in the index block
    let direct_count = idx_blk_elmts.min(max_idx_set);
    for i in 0..direct_count {
        let entry = read_ea_element(
            reader,
            elmts_start + i * es as u64,
            i,
            has_filters,
            size_of_offsets,
            es,
            &chunks_per_dim,
        )?;
        entries.push(entry);
    }

    if max_idx_set <= idx_blk_elmts {
        return Ok(entries);
    }

    // Data block addresses follow the direct elements in the index block
    let dblk_addrs_start = elmts_start + idx_blk_elmts * es as u64;
    let ndblk_addrs = 2 * (sup_blk_min_data_ptrs - 1);

    // Data blocks directly addressed from index block each have data_blk_min_elmts elements
    let mut global_idx = idx_blk_elmts;
    let mut dblk_nelmts = data_blk_min_elmts;

    // The first sup_blk_min_data_ptrs data blocks have data_blk_min_elmts elements each,
    // the next sup_blk_min_data_ptrs have 2*data_blk_min_elmts, etc.
    // But only ndblk_addrs total are in the index block.
    for d in 0..ndblk_addrs {
        if global_idx >= max_idx_set {
            break;
        }
        // Data block size doubles every sup_blk_min_data_ptrs blocks
        if d > 0 && d == sup_blk_min_data_ptrs {
            dblk_nelmts *= 2;
        }

        let dblk_addr_off = dblk_addrs_start + d * o;
        let dblk_addr =
            Le::read_offset(reader, dblk_addr_off, size_of_offsets).map_err(Error::Io)?;

        if dblk_addr == u64::MAX {
            // Data block not allocated — skip these entries
            global_idx += dblk_nelmts;
            continue;
        }

        read_ea_data_block_entries(
            reader,
            dblk_addr,
            dblk_nelmts,
            global_idx,
            max_idx_set,
            has_filters,
            size_of_offsets,
            es,
            &chunks_per_dim,
            &mut entries,
            max_nelmts_bits,
            max_dblk_page_nelmts_bits,
        )?;
        global_idx += dblk_nelmts;
    }

    if global_idx >= max_idx_set {
        return Ok(entries);
    }

    // Super block addresses follow the data block addresses
    let sblk_addrs_start = dblk_addrs_start + ndblk_addrs * o;
    let mut sblk_idx = 0u64;

    while global_idx < max_idx_set {
        let sblk_addr_off = sblk_addrs_start + sblk_idx * o;
        let sblk_addr =
            Le::read_offset(reader, sblk_addr_off, size_of_offsets).map_err(Error::Io)?;

        if sblk_addr == u64::MAX {
            break;
        }

        // Parse super block
        let mut sb_magic = [0u8; 4];
        reader
            .read_exact_at(sblk_addr, &mut sb_magic)
            .map_err(Error::Io)?;
        if sb_magic != EASB_MAGIC {
            return Err(Error::InvalidLayout {
                msg: format!(
                    "expected EASB magic at {:#x}, got {:?}",
                    sblk_addr, sb_magic
                ),
            });
        }

        // Super block: magic(4) + version(1) + client_id(1) + hdr_addr(O) + arr_off(varies)
        let arr_off_size = (max_nelmts_bits as u64).div_ceil(8).max(1);
        let sb_prefix = 4 + 1 + 1 + o + arr_off_size;

        // Number of data blocks in this super block: doubles with each super block pair
        let sblk_ndblks = sup_blk_min_data_ptrs * (1u64 << (sblk_idx / 2));
        // Elements per data block in this super block
        let sblk_dblk_nelmts = data_blk_min_elmts * (1u64 << (sblk_idx.div_ceil(2)));

        let sb_dblk_addrs_start = sblk_addr + sb_prefix;

        for d in 0..sblk_ndblks {
            if global_idx >= max_idx_set {
                break;
            }
            let db_addr = Le::read_offset(reader, sb_dblk_addrs_start + d * o, size_of_offsets)
                .map_err(Error::Io)?;

            if db_addr == u64::MAX {
                global_idx += sblk_dblk_nelmts;
                continue;
            }

            read_ea_data_block_entries(
                reader,
                db_addr,
                sblk_dblk_nelmts,
                global_idx,
                max_idx_set,
                has_filters,
                size_of_offsets,
                es,
                &chunks_per_dim,
                &mut entries,
                max_nelmts_bits,
                max_dblk_page_nelmts_bits,
            )?;
            global_idx += sblk_dblk_nelmts;
        }

        sblk_idx += 1;
    }

    Ok(entries)
}

/// Read a single element record from an extensible/fixed array.
fn read_ea_element<R: ReadAt + ?Sized>(
    reader: &R,
    offset: u64,
    idx: u64,
    has_filters: bool,
    size_of_offsets: u8,
    elmt_size: usize,
    chunks_per_dim: &[u64],
) -> Result<ChunkEntry> {
    let o = size_of_offsets as u64;
    let addr = Le::read_offset(reader, offset, size_of_offsets).map_err(Error::Io)?;

    let (filtered_size, filter_mask) = if has_filters {
        let nbytes_size = elmt_size - size_of_offsets as usize - 4;
        let nbytes = crate::chunk::helpers::read_var_le(reader, offset + o, nbytes_size)?;
        let fmask = Le::read_u32(reader, offset + o + nbytes_size as u64).map_err(Error::Io)?;
        (nbytes, fmask)
    } else {
        (0, 0) // unfiltered: size determined by caller
    };

    let scaled = linear_to_scaled(idx, chunks_per_dim);
    Ok(ChunkEntry {
        address: addr,
        filtered_size,
        filter_mask,
        scaled,
    })
}

/// Read entries from an EA data block.
fn read_ea_data_block_entries<R: ReadAt + ?Sized>(
    reader: &R,
    dblk_addr: u64,
    dblk_nelmts: u64,
    global_start_idx: u64,
    max_idx_set: u64,
    has_filters: bool,
    size_of_offsets: u8,
    elmt_size: usize,
    chunks_per_dim: &[u64],
    entries: &mut Vec<ChunkEntry>,
    max_nelmts_bits: u8,
    max_dblk_page_nelmts_bits: u8,
) -> Result<()> {
    let o = size_of_offsets as u64;

    let mut db_magic = [0u8; 4];
    reader
        .read_exact_at(dblk_addr, &mut db_magic)
        .map_err(Error::Io)?;
    if db_magic != EADB_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected EADB magic at {:#x}, got {:?}",
                dblk_addr, db_magic
            ),
        });
    }

    // Data block prefix: magic(4) + version(1) + client_id(1) + hdr_addr(O) + arr_off(varies)
    let arr_off_size = (max_nelmts_bits as u64).div_ceil(8).max(1);
    let prefix_size = 4 + 1 + 1 + o + arr_off_size;
    let elmts_start = dblk_addr + prefix_size;

    // Paging: when max_dblk_page_nelmts_bits > 0 and the data block has more
    // elements than one page, the block is divided into pages with a 4-byte
    // checksum after each page's elements.
    let page_nelmts =
        if max_dblk_page_nelmts_bits > 0 && dblk_nelmts > (1u64 << max_dblk_page_nelmts_bits) {
            1u64 << max_dblk_page_nelmts_bits
        } else {
            0 // no paging
        };

    let count = dblk_nelmts.min(max_idx_set.saturating_sub(global_start_idx));
    for i in 0..count {
        let offset = if page_nelmts > 0 {
            let page_idx = i / page_nelmts;
            let idx_in_page = i % page_nelmts;
            let page_data_bytes = page_nelmts * elmt_size as u64;
            // Each page: elements + 4-byte checksum
            elmts_start + page_idx * (page_data_bytes + 4) + idx_in_page * elmt_size as u64
        } else {
            elmts_start + i * elmt_size as u64
        };

        let entry = read_ea_element(
            reader,
            offset,
            global_start_idx + i,
            has_filters,
            size_of_offsets,
            elmt_size,
            chunks_per_dim,
        )?;
        entries.push(entry);
    }

    Ok(())
}
