use crate::superblock::UNDEF_ADDR;
use crate::writer::encode::SIZE_OF_LENGTHS;
use crate::writer::encode::SIZE_OF_OFFSETS;

/// Build a global heap collection (GCOL) containing all vlen elements.
///
/// Objects are numbered 1..N. Each object has:
///   index(2) + refcount(2) + reserved(4) + size(L) + data + padding-to-8
///
/// The collection ends with a free-space marker (index 0, size 0).
pub(crate) fn build_global_heap_collection(elements: &[Vec<u8>]) -> Vec<u8> {
    let sl = SIZE_OF_LENGTHS as usize;
    let obj_hdr = 8 + sl;

    let mut objects_size = 0usize;
    for elem in elements {
        let padded = (elem.len() + 7) & !7;
        objects_size += obj_hdr + padded;
    }

    let free_marker_size = obj_hdr;
    let header_size = 8 + sl;
    let collection_size = header_size + objects_size + free_marker_size;

    let mut buf = Vec::with_capacity(collection_size);

    buf.extend_from_slice(b"GCOL");
    buf.push(1);
    buf.extend_from_slice(&[0u8; 3]);
    buf.extend_from_slice(&(collection_size as u64).to_le_bytes());

    for (i, elem) in elements.iter().enumerate() {
        buf.extend_from_slice(&((i + 1) as u16).to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&(elem.len() as u64).to_le_bytes());
        buf.extend_from_slice(elem);
        let padding = ((elem.len() + 7) & !7) - elem.len();
        buf.extend_from_slice(&vec![0u8; padding]);
    }

    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]);
    buf.extend_from_slice(&0u64.to_le_bytes());

    debug_assert_eq!(buf.len(), collection_size);
    buf
}

/// Build vlen heap ID data (the contiguous dataset payload).
///
/// Each element is: seq_len(4) + collection_addr(O) + object_index(4).
pub(crate) fn build_vlen_heap_ids(elements: &[Vec<u8>], gcol_addr: u64) -> Vec<u8> {
    let heap_id_size = 4 + SIZE_OF_OFFSETS as usize + 4;
    let mut buf = Vec::with_capacity(elements.len() * heap_id_size);

    for (i, elem) in elements.iter().enumerate() {
        if elem.is_empty() {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
        } else {
            buf.extend_from_slice(&(elem.len() as u32).to_le_bytes());
            buf.extend_from_slice(&gcol_addr.to_le_bytes());
            buf.extend_from_slice(&((i + 1) as u32).to_le_bytes());
        }
    }

    buf
}
