use crate::checksum;
use crate::error::Result;
use crate::superblock::UNDEF_ADDR;
use crate::writer::chunk_util::enumerate_chunks;
use crate::writer::encode::SIZE_OF_OFFSETS;
use crate::writer::encode::limit_enc_size_u64;

/// Write a variable-length little-endian integer.
fn write_var_le(buf: &mut Vec<u8>, value: u64, nbytes: usize) {
    for i in 0..nbytes {
        buf.push((value >> (i * 8)) as u8);
    }
}

/// Write a single EA element (chunk address, optionally with filter info).
fn write_ea_element(buf: &mut Vec<u8>, addr: u64, size: u64, has_filters: bool, elem_size: u8) {
    buf.extend_from_slice(&addr.to_le_bytes());
    if has_filters {
        let size_enc = elem_size as usize - SIZE_OF_OFFSETS as usize - 4;
        write_var_le(buf, size, size_enc);
        buf.extend_from_slice(&0u32.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Fixed Array
// ---------------------------------------------------------------------------

pub(crate) fn write_fixed_array_index(
    buf: &mut Vec<u8>,
    chunk_addrs: &[u64],
    chunk_sizes: &[u64],
    has_filters: bool,
    raw_chunk_bytes: u64,
) -> Result<()> {
    let nelmts = chunk_addrs.len() as u64;
    let client_id: u8 = if has_filters { 1 } else { 0 };

    let chunk_size_enc_bytes = if has_filters {
        let max_size = chunk_sizes.iter().copied().max().unwrap_or(raw_chunk_bytes);
        limit_enc_size_u64(max_size.max(raw_chunk_bytes)) as u8
    } else {
        0
    };

    let elem_size: u8 = if has_filters {
        SIZE_OF_OFFSETS + chunk_size_enc_bytes + 4
    } else {
        SIZE_OF_OFFSETS
    };

    // Write header
    let hdr_addr = buf.len() as u64;
    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"FAHD");
    hdr.push(0);
    hdr.push(client_id);
    hdr.push(elem_size);
    hdr.push(0);
    hdr.extend_from_slice(&nelmts.to_le_bytes());

    let hdr_before_dblk_addr = hdr.len();
    hdr.extend_from_slice(&0u64.to_le_bytes());

    let cksum = checksum::lookup3(&hdr);
    hdr.extend_from_slice(&cksum.to_le_bytes());
    let hdr_size = hdr.len();

    let dblk_addr = hdr_addr + hdr_size as u64;
    hdr[hdr_before_dblk_addr..hdr_before_dblk_addr + 8].copy_from_slice(&dblk_addr.to_le_bytes());
    let cksum = checksum::lookup3(&hdr[..hdr.len() - 4]);
    let hdr_len = hdr.len();
    hdr[hdr_len - 4..].copy_from_slice(&cksum.to_le_bytes());

    buf.extend_from_slice(&hdr);

    // Write data block
    let mut dblk = Vec::new();
    dblk.extend_from_slice(b"FADB");
    dblk.push(0);
    dblk.push(client_id);
    dblk.extend_from_slice(&hdr_addr.to_le_bytes());

    for i in 0..chunk_addrs.len() {
        dblk.extend_from_slice(&chunk_addrs[i].to_le_bytes());
        if has_filters {
            let size = chunk_sizes[i];
            match chunk_size_enc_bytes {
                1 => dblk.push(size as u8),
                2 => dblk.extend_from_slice(&(size as u16).to_le_bytes()),
                4 => dblk.extend_from_slice(&(size as u32).to_le_bytes()),
                8 => dblk.extend_from_slice(&size.to_le_bytes()),
                _ => {
                    for b in 0..chunk_size_enc_bytes {
                        dblk.push((size >> (b as u64 * 8)) as u8);
                    }
                }
            }
            dblk.extend_from_slice(&0u32.to_le_bytes());
        }
    }

    let cksum = checksum::lookup3(&dblk);
    dblk.extend_from_slice(&cksum.to_le_bytes());

    buf.extend_from_slice(&dblk);

    Ok(())
}

// ---------------------------------------------------------------------------
// Extensible Array
// ---------------------------------------------------------------------------

pub(crate) fn write_extensible_array_index(
    buf: &mut Vec<u8>,
    chunk_addrs: &[u64],
    chunk_sizes: &[u64],
    has_filters: bool,
    nchunks: usize,
) -> Result<()> {
    let client_id: u8 = if has_filters { 1 } else { 0 };

    let elem_size: u8 = if has_filters {
        let max_size = chunk_sizes.iter().copied().max().unwrap_or(0);
        let size_enc = limit_enc_size_u64(max_size) as u8;
        SIZE_OF_OFFSETS + size_enc + 4
    } else {
        SIZE_OF_OFFSETS
    };

    let idx_blk_elmts: u8 = 4;
    let data_blk_min_elmts: u8 = 1;
    let sup_blk_min_data_ptrs: u8 = 2;
    let max_nelmts_bits: u8 = 32;
    let max_dblk_page_nelmts_bits: u8 = 0;

    let max_idx_set = nchunks as u64;

    let ndblk_addrs: u64 = if nchunks as u64 > idx_blk_elmts as u64 {
        2 * (sup_blk_min_data_ptrs as u64 - 1)
    } else {
        0
    };

    let mut capacity = idx_blk_elmts as u64;
    let mut dblk_nelmts = data_blk_min_elmts as u64;
    for d in 0..ndblk_addrs {
        if d > 0 && d == sup_blk_min_data_ptrs as u64 {
            dblk_nelmts *= 2;
        }
        capacity += dblk_nelmts;
    }

    let mut nsblk_addrs: u64 = 0;
    if nchunks as u64 > capacity {
        let mut remaining = nchunks as u64 - capacity;
        let mut sblk_idx = 0u64;
        while remaining > 0 {
            let sblk_ndblks = sup_blk_min_data_ptrs as u64 * (1u64 << (sblk_idx / 2));
            let sblk_dblk_nelmts = data_blk_min_elmts as u64 * (1u64 << (sblk_idx.div_ceil(2)));
            let sblk_capacity = sblk_ndblks * sblk_dblk_nelmts;
            remaining = remaining.saturating_sub(sblk_capacity);
            nsblk_addrs += 1;
            sblk_idx += 1;
        }
    }

    let chunks_in_idx_blk = (idx_blk_elmts as u64).min(max_idx_set);
    let chunks_remaining = max_idx_set.saturating_sub(chunks_in_idx_blk);

    let mut dblk_plan: Vec<u64> = Vec::new();
    {
        let mut remaining = chunks_remaining;
        let mut dn = data_blk_min_elmts as u64;
        for d in 0..ndblk_addrs {
            if remaining == 0 {
                break;
            }
            if d > 0 && d == sup_blk_min_data_ptrs as u64 {
                dn *= 2;
            }
            let count = dn.min(remaining);
            dblk_plan.push(count);
            remaining -= count;
        }
    }

    struct SblkPlan {
        dblk_counts: Vec<u64>,
    }
    let mut sblk_plans: Vec<SblkPlan> = Vec::new();
    {
        let mut global_idx = chunks_in_idx_blk;
        for plan in &dblk_plan {
            global_idx += plan;
        }
        let mut sblk_idx = 0u64;
        while global_idx < max_idx_set && sblk_idx < nsblk_addrs {
            let sblk_ndblks = sup_blk_min_data_ptrs as u64 * (1u64 << (sblk_idx / 2));
            let sblk_dblk_nelmts = data_blk_min_elmts as u64 * (1u64 << (sblk_idx.div_ceil(2)));
            let mut counts = Vec::new();
            for _ in 0..sblk_ndblks {
                if global_idx >= max_idx_set {
                    break;
                }
                let count = sblk_dblk_nelmts.min(max_idx_set - global_idx);
                counts.push(count);
                global_idx += count;
            }
            sblk_plans.push(SblkPlan {
                dblk_counts: counts,
            });
            sblk_idx += 1;
        }
    }

    let o = SIZE_OF_OFFSETS as usize;

    // --- Write EAHD header ---
    let hdr_addr = buf.len() as u64;
    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"EAHD");
    hdr.push(0);
    hdr.push(client_id);
    hdr.push(elem_size);
    hdr.push(max_nelmts_bits);
    hdr.push(idx_blk_elmts);
    hdr.push(data_blk_min_elmts);
    hdr.push(sup_blk_min_data_ptrs);
    hdr.push(max_dblk_page_nelmts_bits);

    let nsuper_blks = sblk_plans.len() as u64;
    let ndata_blks = dblk_plan.len() as u64
        + sblk_plans
            .iter()
            .map(|s| s.dblk_counts.len() as u64)
            .sum::<u64>();
    hdr.extend_from_slice(&nsuper_blks.to_le_bytes());
    hdr.extend_from_slice(&0u64.to_le_bytes());
    hdr.extend_from_slice(&ndata_blks.to_le_bytes());
    hdr.extend_from_slice(&0u64.to_le_bytes());
    hdr.extend_from_slice(&max_idx_set.to_le_bytes());
    let num_elements = max_idx_set;
    hdr.extend_from_slice(&num_elements.to_le_bytes());

    let idx_blk_addr_offset = hdr.len();
    hdr.extend_from_slice(&0u64.to_le_bytes());

    hdr.extend_from_slice(&0u32.to_le_bytes());
    let hdr_size = hdr.len();

    let ib_prefix = 4 + 1 + 1 + o;
    let direct_elems_bytes = idx_blk_elmts as usize * elem_size as usize;
    let dblk_addr_slots = ndblk_addrs as usize;
    let sblk_addr_slots = nsblk_addrs as usize;
    let ib_body = direct_elems_bytes + (dblk_addr_slots + sblk_addr_slots) * o;
    let ib_size = ib_prefix + ib_body + 4;

    let ib_addr = hdr_addr + hdr_size as u64;

    hdr[idx_blk_addr_offset..idx_blk_addr_offset + 8].copy_from_slice(&ib_addr.to_le_bytes());
    let hdr_cksum = checksum::lookup3(&hdr[..hdr.len() - 4]);
    let hdr_len = hdr.len();
    hdr[hdr_len - 4..].copy_from_slice(&hdr_cksum.to_le_bytes());

    buf.extend_from_slice(&hdr);

    // --- Compute data block and super block addresses ---
    let mut dblk_addrs_from_ib: Vec<u64> = Vec::new();
    let arr_off_size = (max_nelmts_bits as u64).div_ceil(8).max(1) as usize;
    let mut next_addr = ib_addr + ib_size as u64;
    for &count in &dblk_plan {
        dblk_addrs_from_ib.push(next_addr);
        let db_size = 4 + 1 + 1 + o + arr_off_size + (count as usize * elem_size as usize) + 4;
        next_addr += db_size as u64;
    }

    let mut sblk_addrs: Vec<u64> = Vec::new();
    let mut sblk_dblk_addrs: Vec<Vec<u64>> = Vec::new();
    for (si, splan) in sblk_plans.iter().enumerate() {
        sblk_addrs.push(next_addr);
        let sblk_ndblks = sup_blk_min_data_ptrs as u64 * (1u64 << (si as u64 / 2));
        let sb_size = 4 + 1 + 1 + o + arr_off_size + (sblk_ndblks as usize * o) + 4;
        next_addr += sb_size as u64;

        let mut db_addrs = Vec::new();
        for &count in &splan.dblk_counts {
            db_addrs.push(next_addr);
            let db_size = 4 + 1 + 1 + o + arr_off_size + (count as usize * elem_size as usize) + 4;
            next_addr += db_size as u64;
        }
        sblk_dblk_addrs.push(db_addrs);
    }

    // --- Write index block ---
    let mut ib = Vec::new();
    ib.extend_from_slice(b"EAIB");
    ib.push(0);
    ib.push(client_id);
    ib.extend_from_slice(&hdr_addr.to_le_bytes());

    let mut chunk_idx: usize = 0;
    for _ in 0..idx_blk_elmts {
        if chunk_idx < nchunks {
            write_ea_element(
                &mut ib,
                chunk_addrs[chunk_idx],
                chunk_sizes[chunk_idx],
                has_filters,
                elem_size,
            );
            chunk_idx += 1;
        } else {
            ib.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
            if has_filters {
                ib.extend(std::iter::repeat_n(0u8, elem_size as usize - o));
            }
        }
    }

    for i in 0..dblk_addr_slots {
        if i < dblk_addrs_from_ib.len() {
            ib.extend_from_slice(&dblk_addrs_from_ib[i].to_le_bytes());
        } else {
            ib.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        }
    }

    for i in 0..sblk_addr_slots {
        if i < sblk_addrs.len() {
            ib.extend_from_slice(&sblk_addrs[i].to_le_bytes());
        } else {
            ib.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        }
    }

    let ib_cksum = checksum::lookup3(&ib);
    ib.extend_from_slice(&ib_cksum.to_le_bytes());
    debug_assert_eq!(ib.len(), ib_size);

    buf.extend_from_slice(&ib);

    // --- Write data blocks from index block ---
    let mut global_idx = idx_blk_elmts as u64;
    for (di, &count) in dblk_plan.iter().enumerate() {
        let mut db = Vec::new();
        db.extend_from_slice(b"EADB");
        db.push(0);
        db.push(client_id);
        db.extend_from_slice(&hdr_addr.to_le_bytes());
        write_var_le(&mut db, global_idx, arr_off_size);

        for _ in 0..count as usize {
            if chunk_idx < nchunks {
                write_ea_element(
                    &mut db,
                    chunk_addrs[chunk_idx],
                    chunk_sizes[chunk_idx],
                    has_filters,
                    elem_size,
                );
                chunk_idx += 1;
            } else {
                db.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
                if has_filters {
                    db.extend(std::iter::repeat_n(0u8, elem_size as usize - o));
                }
            }
        }
        global_idx += count;

        let db_cksum = checksum::lookup3(&db);
        db.extend_from_slice(&db_cksum.to_le_bytes());
        debug_assert_eq!(buf.len() as u64, dblk_addrs_from_ib[di]);
        buf.extend_from_slice(&db);
    }

    // --- Write super blocks and their data blocks ---
    for (si, splan) in sblk_plans.iter().enumerate() {
        let sblk_ndblks = sup_blk_min_data_ptrs as u64 * (1u64 << (si as u64 / 2));

        let mut sb = Vec::new();
        sb.extend_from_slice(b"EASB");
        sb.push(0);
        sb.push(client_id);
        sb.extend_from_slice(&hdr_addr.to_le_bytes());
        write_var_le(&mut sb, global_idx, arr_off_size);

        for di in 0..sblk_ndblks as usize {
            if di < sblk_dblk_addrs[si].len() {
                sb.extend_from_slice(&sblk_dblk_addrs[si][di].to_le_bytes());
            } else {
                sb.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
            }
        }

        let sb_cksum = checksum::lookup3(&sb);
        sb.extend_from_slice(&sb_cksum.to_le_bytes());
        buf.extend_from_slice(&sb);

        for &count in splan.dblk_counts.iter() {
            let mut db = Vec::new();
            db.extend_from_slice(b"EADB");
            db.push(0);
            db.push(client_id);
            db.extend_from_slice(&hdr_addr.to_le_bytes());
            write_var_le(&mut db, global_idx, arr_off_size);

            for _ in 0..count as usize {
                if chunk_idx < nchunks {
                    write_ea_element(
                        &mut db,
                        chunk_addrs[chunk_idx],
                        chunk_sizes[chunk_idx],
                        has_filters,
                        elem_size,
                    );
                    chunk_idx += 1;
                } else {
                    db.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
                    if has_filters {
                        db.extend(std::iter::repeat_n(0u8, elem_size as usize - o));
                    }
                }
            }
            global_idx += count;

            let db_cksum = checksum::lookup3(&db);
            db.extend_from_slice(&db_cksum.to_le_bytes());
            buf.extend_from_slice(&db);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// B-tree v1
// ---------------------------------------------------------------------------

pub(crate) fn write_btree_v1_chunk_index(
    buf: &mut Vec<u8>,
    chunk_addrs: &[u64],
    chunk_sizes: &[u64],
    chunk_coords_list: &[Vec<u64>],
    chunk_dims: &[u64],
    element_size: u32,
    ndims: usize,
) -> Result<()> {
    let nchunks = chunk_addrs.len();
    let v3_ndims = ndims + 1;

    buf.extend_from_slice(b"TREE");
    buf.push(1);
    buf.push(0);
    buf.extend_from_slice(&(nchunks as u16).to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());

    for i in 0..nchunks {
        buf.extend_from_slice(&(chunk_sizes[i] as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());

        for d in 0..ndims {
            let elem_offset = chunk_coords_list[i][d] * chunk_dims[d];
            buf.extend_from_slice(&elem_offset.to_le_bytes());
        }
        buf.extend_from_slice(&0u64.to_le_bytes());

        buf.extend_from_slice(&chunk_addrs[i].to_le_bytes());
    }

    let raw_chunk_bytes = chunk_dims.iter().product::<u64>() * element_size as u64;
    buf.extend_from_slice(&(raw_chunk_bytes as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    for _ in 0..v3_ndims {
        buf.extend_from_slice(&u64::MAX.to_le_bytes());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// B-tree v2
// ---------------------------------------------------------------------------

pub(crate) fn write_btree_v2_chunk_index(
    buf: &mut Vec<u8>,
    chunk_addrs: &[u64],
    chunk_sizes: &[u64],
    has_filters: bool,
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: u32,
) -> Result<()> {
    let nchunks = chunk_addrs.len();
    let ndims = shape.len();
    let o = SIZE_OF_OFFSETS as usize;

    let chunk_byte_size: u64 = chunk_dims.iter().product::<u64>() * element_size as u64;
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

    let record_size = o + chunk_size_len + (if has_filters { 4 } else { 0 }) + 8 * ndims;
    let bt2_type: u8 = if has_filters { 11 } else { 10 };

    let node_size: u32 = 4096;
    let split_percent: u8 = 98;
    let merge_percent: u8 = 40;

    let mut records: Vec<Vec<u8>> = Vec::with_capacity(nchunks);
    let chunk_coords_list = enumerate_chunks(shape, chunk_dims);
    for (i, coords) in chunk_coords_list.iter().enumerate() {
        let mut rec = Vec::with_capacity(record_size);
        rec.extend_from_slice(&chunk_addrs[i].to_le_bytes());
        if has_filters {
            write_var_le(&mut rec, chunk_sizes[i], chunk_size_len);
            rec.extend_from_slice(&0u32.to_le_bytes());
        }
        for coord in &coords[..ndims] {
            rec.extend_from_slice(&coord.to_le_bytes());
        }
        debug_assert_eq!(rec.len(), record_size);
        records.push(rec);
    }

    // Write BTHD header
    let hdr_addr = buf.len() as u64;
    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"BTHD");
    hdr.push(0);
    hdr.push(bt2_type);
    hdr.extend_from_slice(&node_size.to_le_bytes());
    hdr.extend_from_slice(&(record_size as u16).to_le_bytes());
    hdr.extend_from_slice(&0u16.to_le_bytes());
    hdr.push(split_percent);
    hdr.push(merge_percent);

    let root_addr_offset = hdr.len();
    hdr.extend_from_slice(&0u64.to_le_bytes());
    hdr.extend_from_slice(&(nchunks as u16).to_le_bytes());
    hdr.extend_from_slice(&(nchunks as u64).to_le_bytes());

    hdr.extend_from_slice(&0u32.to_le_bytes());
    let hdr_size = hdr.len();

    let leaf_addr = hdr_addr + hdr_size as u64;
    hdr[root_addr_offset..root_addr_offset + 8].copy_from_slice(&leaf_addr.to_le_bytes());

    let hdr_cksum = checksum::lookup3(&hdr[..hdr.len() - 4]);
    let hdr_len = hdr.len();
    hdr[hdr_len - 4..].copy_from_slice(&hdr_cksum.to_le_bytes());

    buf.extend_from_slice(&hdr);

    // Write BTLF leaf node
    let mut leaf = Vec::new();
    leaf.extend_from_slice(b"BTLF");
    leaf.push(0);
    leaf.push(bt2_type);

    for rec in &records {
        leaf.extend_from_slice(rec);
    }

    let leaf_cksum = checksum::lookup3(&leaf);
    leaf.extend_from_slice(&leaf_cksum.to_le_bytes());

    buf.extend_from_slice(&leaf);

    Ok(())
}
