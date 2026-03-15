/// Parse a B-tree v2 "type 5" record (link name for new-style groups).
///
/// Layout: hash (4 bytes LE u32) + heap ID (7 bytes, variable based on heap ID length).
pub fn parse_link_name_record(data: &[u8], heap_id_len: usize) -> Option<(u32, Vec<u8>)> {
    if data.len() < 4 + heap_id_len {
        return None;
    }
    let hash = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let heap_id = data[4..4 + heap_id_len].to_vec();
    Some((hash, heap_id))
}

/// Parse a B-tree v2 "type 6" record (link creation order for new-style groups).
///
/// Layout: creation_order (8 bytes LE u64) + heap ID (heap_id_len bytes).
pub fn parse_link_creation_order_record(data: &[u8], heap_id_len: usize) -> Option<(u64, Vec<u8>)> {
    if data.len() < 8 + heap_id_len {
        return None;
    }
    let creation_order = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]);
    let heap_id = data[8..8 + heap_id_len].to_vec();
    Some((creation_order, heap_id))
}

/// Parse a B-tree v2 "type 8" record (attribute name for dense attribute storage).
///
/// Layout: heap_id (heap_id_len bytes) + flags (1 byte) + creation_order (4 bytes) + hash (4 bytes).
pub fn parse_attribute_name_record(data: &[u8], heap_id_len: usize) -> Option<Vec<u8>> {
    if data.len() < heap_id_len + 1 + 4 + 4 {
        return None;
    }
    let heap_id = data[..heap_id_len].to_vec();
    // flags at heap_id_len, creation_order at heap_id_len+1, hash at heap_id_len+5
    // We only need the heap_id to look up the attribute message
    Some(heap_id)
}

/// Parse a B-tree v2 "type 9" record (attribute creation order).
///
/// Layout: heap_id (heap_id_len bytes) + flags (1 byte) + creation_order (4 bytes).
pub fn parse_attribute_creation_order_record(
    data: &[u8],
    heap_id_len: usize,
) -> Option<(u32, Vec<u8>)> {
    if data.len() < heap_id_len + 1 + 4 {
        return None;
    }
    let heap_id = data[..heap_id_len].to_vec();
    // flags at heap_id_len
    let co_off = heap_id_len + 1;
    let creation_order = u32::from_le_bytes([
        data[co_off],
        data[co_off + 1],
        data[co_off + 2],
        data[co_off + 3],
    ]);
    Some((creation_order, heap_id))
}
