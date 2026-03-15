use crate::chunk::entry::ChunkEntry;
use crate::error::Error;
use crate::error::Result;
use crate::io::Le;
use crate::io::ReadAt;

/// B-tree v1 signature: `TREE`
const TREE_MAGIC: [u8; 4] = *b"TREE";

/// Read chunk entries from a B-tree v1 index (layout version 3).
///
/// B-tree v1 node layout:
/// ```text
/// Signature: "TREE" (4 bytes)
/// Node type: 1 byte (1 = chunked raw data)
/// Node level: 1 byte (0 = leaf, >0 = internal)
/// Entries used: 2 bytes (u16 LE)
/// Left sibling: sizeof_offsets bytes
/// Right sibling: sizeof_offsets bytes
/// Then interleaved: Key[0] Child[0] Key[1] Child[1] ... Key[N-1] Child[N-1] Key[N]
///
/// Each key (type 1):
///   chunk_size: 4 bytes (u32) — on-disk (filtered) size
///   filter_mask: 4 bytes (u32)
///   offset[v3_ndims]: v3_ndims * sizeof_offsets bytes — element-coordinate offsets
/// Each child pointer: sizeof_offsets bytes
/// ```
pub(crate) fn read_btree_v1_entries<R: ReadAt + ?Sized>(
    reader: &R,
    node_addr: u64,
    dataset_dims: &[u64],
    chunk_dims: &[u32],
    v3_ndims: usize,
    size_of_offsets: u8,
) -> Result<Vec<ChunkEntry>> {
    let o = size_of_offsets as usize;

    // Read and validate signature
    let mut magic = [0u8; 4];
    reader
        .read_exact_at(node_addr, &mut magic)
        .map_err(Error::Io)?;
    if magic != TREE_MAGIC {
        return Err(Error::InvalidLayout {
            msg: format!("expected TREE magic at {:#x}, got {:?}", node_addr, magic),
        });
    }

    let node_type = Le::read_u8(reader, node_addr + 4).map_err(Error::Io)?;
    if node_type != 1 {
        return Err(Error::InvalidLayout {
            msg: format!(
                "expected B-tree v1 node type 1 (chunked), got {}",
                node_type
            ),
        });
    }
    let node_level = Le::read_u8(reader, node_addr + 5).map_err(Error::Io)?;
    let entries_used = Le::read_u16(reader, node_addr + 6).map_err(Error::Io)? as usize;

    // Skip sibling pointers
    let keys_start = node_addr + 8 + 2 * o as u64;

    // Key size: chunk_size(4) + filter_mask(4) + offsets(v3_ndims * o)
    let key_size = 4 + 4 + v3_ndims * o;
    // Stride between consecutive keys: key_size + child_pointer(o)
    let entry_stride = key_size + o;

    let rank = dataset_dims.len();

    if node_level == 0 {
        // Leaf node: child pointers are chunk data addresses
        let mut entries = Vec::with_capacity(entries_used);
        for i in 0..entries_used {
            let key_off = keys_start + (i as u64) * entry_stride as u64;
            let child_off = key_off + key_size as u64;

            let chunk_size = Le::read_u32(reader, key_off).map_err(Error::Io)?;
            let filter_mask = Le::read_u32(reader, key_off + 4).map_err(Error::Io)?;

            // Read the chunk offsets (element coordinates)
            let mut offsets = Vec::with_capacity(rank);
            for d in 0..rank {
                let off = Le::read_offset(reader, key_off + 8 + (d * o) as u64, size_of_offsets)
                    .map_err(Error::Io)?;
                offsets.push(off);
            }

            let chunk_addr =
                Le::read_offset(reader, child_off, size_of_offsets).map_err(Error::Io)?;

            // Convert element-coordinate offsets to scaled chunk coordinates
            let scaled: Vec<u64> = (0..rank)
                .map(|d| offsets[d] / chunk_dims[d] as u64)
                .collect();

            entries.push(ChunkEntry {
                address: chunk_addr,
                filtered_size: chunk_size as u64,
                filter_mask,
                scaled,
            });
        }
        Ok(entries)
    } else {
        // Internal node: child pointers are addresses of child B-tree nodes
        let mut entries = Vec::new();
        for i in 0..entries_used {
            let key_off = keys_start + (i as u64) * entry_stride as u64;
            let child_off = key_off + key_size as u64;

            let child_addr =
                Le::read_offset(reader, child_off, size_of_offsets).map_err(Error::Io)?;

            if child_addr != u64::MAX {
                let child_entries = read_btree_v1_entries(
                    reader,
                    child_addr,
                    dataset_dims,
                    chunk_dims,
                    v3_ndims,
                    size_of_offsets,
                )?;
                entries.extend(child_entries);
            }
        }
        Ok(entries)
    }
}
