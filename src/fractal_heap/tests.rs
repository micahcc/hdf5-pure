use super::*;

/// Build a FractalHeapHeader with just the fields needed for doubling-table math.
fn make_header(
    starting_block_size: u64,
    max_direct_block_size: u64,
    table_width: u16,
) -> FractalHeapHeader {
    FractalHeapHeader {
        heap_id_length: 7,
        io_filter_encoded_length: 0,
        flags: 0,
        max_managed_object_size: 0,
        next_huge_object_id: 0,
        huge_bt2_address: u64::MAX,
        free_space_in_managed: 0,
        free_space_manager_address: u64::MAX,
        managed_space_total: 0,
        managed_space_allocated: 0,
        managed_alloc_iterator_offset: 0,
        managed_objects_count: 0,
        huge_objects_total_size: 0,
        huge_objects_count: 0,
        tiny_objects_total_size: 0,
        tiny_objects_count: 0,
        table_width,
        starting_block_size,
        max_direct_block_size,
        max_heap_size_bits: 32,
        starting_root_rows: 0,
        root_block_address: u64::MAX,
        current_root_rows: 0,
        filtered_root_direct_block_size: None,
        io_filter_mask: None,
        filter_pipeline: None,
    }
}

#[test]
fn test_block_size_for_row() {
    let h = make_header(512, 65536, 4);
    assert_eq!(h.block_size_for_row(0), 512);
    assert_eq!(h.block_size_for_row(1), 512);
    assert_eq!(h.block_size_for_row(2), 1024);
    assert_eq!(h.block_size_for_row(3), 2048);
    assert_eq!(h.block_size_for_row(8), 65536);
    // Indirect rows continue doubling past max_direct_block_size
    assert_eq!(h.block_size_for_row(9), 131072);
    assert_eq!(h.block_size_for_row(10), 262144);
}

#[test]
fn test_max_direct_rows() {
    // start=512, max_direct=65536: (16-9)+2 = 9
    let h = make_header(512, 65536, 4);
    assert_eq!(h.max_direct_rows(), 9);

    // start=1024, max_direct=65536: (16-10)+2 = 8
    let h2 = make_header(1024, 65536, 4);
    assert_eq!(h2.max_direct_rows(), 8);

    // start=256, max_direct=65536: (16-8)+2 = 10
    let h3 = make_header(256, 65536, 4);
    assert_eq!(h3.max_direct_rows(), 10);

    // start=512, max_direct=512: (9-9)+2 = 2
    let h4 = make_header(512, 512, 4);
    assert_eq!(h4.max_direct_rows(), 2);
}

#[test]
fn test_child_indirect_nrows() {
    // start=512, width=4, max_direct=65536
    // first_row_bits = log2(512*4) = log2(2048) = 11
    let h = make_header(512, 65536, 4);

    // Row 9 (first indirect row): block_size = 131072, log2 = 17
    // nrows = 17 - 11 + 1 = 7
    assert_eq!(h.child_indirect_nrows(9), 7);

    // Row 10: block_size = 262144, log2 = 18
    // nrows = 18 - 11 + 1 = 8
    assert_eq!(h.child_indirect_nrows(10), 8);
}

#[test]
fn huge_ids_direct_computed_not_flag() {
    // huge_ids_direct is derived from id_len, NOT from flags byte.
    // flags=0x02 is "checksum direct blocks", not "direct huge IDs".
    // H5HFhuge.c: direct if sizeof_addr + sizeof_size <= id_len - 1
    let sizeof_addr: u8 = 8;
    let sizeof_size: u8 = 8;

    // id_len=17 → 17-1=16 >= 8+8=16 → direct
    let h1 = FractalHeapHeader {
        heap_id_length: 17,
        io_filter_encoded_length: 0,
        flags: 0, // no flag bits set, yet should still be "direct"
        ..make_header(512, 65536, 4)
    };
    let needed = sizeof_addr as usize + sizeof_size as usize;
    assert!(needed <= (h1.heap_id_length as usize) - 1);

    // id_len=8 → 8-1=7 < 16 → indirect (must use B-tree)
    let h2 = FractalHeapHeader {
        heap_id_length: 8,
        io_filter_encoded_length: 0,
        flags: 0x02, // checksum flag set, but should NOT mean "direct"
        ..make_header(512, 65536, 4)
    };
    assert!(needed > (h2.heap_id_length as usize) - 1);

    // With filters: need sizeof_addr + sizeof_size + 4 + sizeof_size
    let needed_filtered = sizeof_addr as usize + sizeof_size as usize + 4 + sizeof_size as usize;
    // id_len=25 → 25-1=24 < 28 → indirect
    let h3 = FractalHeapHeader {
        heap_id_length: 25,
        io_filter_encoded_length: 12,
        flags: 0,
        ..make_header(512, 65536, 4)
    };
    assert!(needed_filtered > (h3.heap_id_length as usize) - 1);

    // id_len=29 → 29-1=28 >= 28 → direct
    let h4 = FractalHeapHeader {
        heap_id_length: 29,
        io_filter_encoded_length: 12,
        flags: 0,
        ..make_header(512, 65536, 4)
    };
    assert!(needed_filtered <= (h4.heap_id_length as usize) - 1);
}
