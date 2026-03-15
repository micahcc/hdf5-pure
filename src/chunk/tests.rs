use super::helpers::copy_chunk_to_output;
use super::helpers::linear_to_scaled;
use super::implicit::read_implicit_chunk_entries;
use super::single_chunk::read_single_chunk_entries;

// ── linear_to_scaled ──

#[test]
fn linear_to_scaled_1d() {
    // 10 chunks in one dimension
    let cpd = vec![10];
    assert_eq!(linear_to_scaled(0, &cpd), vec![0]);
    assert_eq!(linear_to_scaled(5, &cpd), vec![5]);
    assert_eq!(linear_to_scaled(9, &cpd), vec![9]);
}

#[test]
fn linear_to_scaled_2d() {
    // 3x4 grid of chunks (row-major: last dim varies fastest)
    let cpd = vec![3, 4];
    assert_eq!(linear_to_scaled(0, &cpd), vec![0, 0]);
    assert_eq!(linear_to_scaled(1, &cpd), vec![0, 1]);
    assert_eq!(linear_to_scaled(3, &cpd), vec![0, 3]);
    assert_eq!(linear_to_scaled(4, &cpd), vec![1, 0]);
    assert_eq!(linear_to_scaled(11, &cpd), vec![2, 3]);
}

#[test]
fn linear_to_scaled_3d() {
    let cpd = vec![2, 3, 4];
    assert_eq!(linear_to_scaled(0, &cpd), vec![0, 0, 0]);
    assert_eq!(linear_to_scaled(1, &cpd), vec![0, 0, 1]);
    assert_eq!(linear_to_scaled(4, &cpd), vec![0, 1, 0]);
    assert_eq!(linear_to_scaled(12, &cpd), vec![1, 0, 0]);
    assert_eq!(linear_to_scaled(23, &cpd), vec![1, 2, 3]);
}

// ── copy_chunk_to_output ──

#[test]
fn copy_chunk_1d_full() {
    // 1D dataset [8], chunk size 4, element size 2 (i16)
    let chunk_dims = [4u32];
    let dataset_dims = [8u64];
    let elem = 2;
    let mut output = vec![0u8; 16];

    // Chunk at scaled=[0]: copy 8 bytes to offset 0
    let chunk0 = vec![1, 0, 2, 0, 3, 0, 4, 0];
    copy_chunk_to_output(&chunk0, &[0], &chunk_dims, &dataset_dims, elem, &mut output);

    // Chunk at scaled=[1]: copy 8 bytes to offset 8
    let chunk1 = vec![5, 0, 6, 0, 7, 0, 8, 0];
    copy_chunk_to_output(&chunk1, &[1], &chunk_dims, &dataset_dims, elem, &mut output);

    assert_eq!(output, vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]);
}

#[test]
fn copy_chunk_1d_edge() {
    // 1D dataset [6], chunk size 4, element size 1 → 2 chunks, second is edge
    let chunk_dims = [4u32];
    let dataset_dims = [6u64];
    let elem = 1;
    let mut output = vec![0u8; 6];

    // Chunk 0: full, 4 bytes
    copy_chunk_to_output(
        &[10, 20, 30, 40],
        &[0],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );
    // Chunk 1: edge, only 2 of 4 elements matter, but chunk data is still 4 bytes
    copy_chunk_to_output(
        &[50, 60, 0, 0],
        &[1],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );

    assert_eq!(output, vec![10, 20, 30, 40, 50, 60]);
}

#[test]
fn copy_chunk_2d_full() {
    // 2D dataset [4,6], chunks [2,3], element size 1
    // 2x2 grid of chunks, each 2x3 = 6 bytes
    //
    // Desired output (row-major):
    //   Row 0: [ 1, 2, 3, 4, 5, 6]
    //   Row 1: [ 7, 8, 9,10,11,12]
    //   Row 2: [13,14,15,16,17,18]
    //   Row 3: [19,20,21,22,23,24]
    //
    // Each chunk stores its local region in chunk row-major order.
    let chunk_dims = [2u32, 3];
    let dataset_dims = [4u64, 6];
    let elem = 1;
    let mut output = vec![0u8; 24]; // 4*6

    // Chunk (0,0): rows 0-1, cols 0-2 → [1,2,3, 7,8,9]
    copy_chunk_to_output(
        &[1, 2, 3, 7, 8, 9],
        &[0, 0],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );

    // Chunk (0,1): rows 0-1, cols 3-5 → [4,5,6, 10,11,12]
    copy_chunk_to_output(
        &[4, 5, 6, 10, 11, 12],
        &[0, 1],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );

    // Chunk (1,0): rows 2-3, cols 0-2 → [13,14,15, 19,20,21]
    copy_chunk_to_output(
        &[13, 14, 15, 19, 20, 21],
        &[1, 0],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );

    // Chunk (1,1): rows 2-3, cols 3-5 → [16,17,18, 22,23,24]
    copy_chunk_to_output(
        &[16, 17, 18, 22, 23, 24],
        &[1, 1],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );

    // Expected: row-major [1..24]
    let expected: Vec<u8> = (1..=24).collect();
    assert_eq!(output, expected);
}

#[test]
fn copy_chunk_2d_edge() {
    // 2D dataset [3,5], chunks [2,3], element size 1
    // Grid: 2x2 chunks, but edges are partial
    //
    // Desired output (row-major):
    //   Row 0: [ 1, 2, 3, 4, 5]
    //   Row 1: [ 6, 7, 8, 9,10]
    //   Row 2: [11,12,13,14,15]
    //
    // Chunk (0,0): 2x3 full. Rows 0-1, cols 0-2.
    //   Chunk data: [1,2,3, 6,7,8]
    // Chunk (0,1): 2x2 edge (3 cols in chunk, only 2 used). Rows 0-1, cols 3-4.
    //   Chunk data: [4,5,0, 9,10,0]
    // Chunk (1,0): 1x3 edge (2 rows in chunk, only 1 used). Row 2, cols 0-2.
    //   Chunk data: [11,12,13, 0,0,0]
    // Chunk (1,1): 1x2 edge. Row 2, cols 3-4.
    //   Chunk data: [14,15,0, 0,0,0]
    let chunk_dims = [2u32, 3];
    let dataset_dims = [3u64, 5];
    let elem = 1;
    let mut output = vec![0u8; 15]; // 3*5

    copy_chunk_to_output(
        &[1, 2, 3, 6, 7, 8],
        &[0, 0],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );
    copy_chunk_to_output(
        &[4, 5, 0, 9, 10, 0],
        &[0, 1],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );
    copy_chunk_to_output(
        &[11, 12, 13, 0, 0, 0],
        &[1, 0],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );
    copy_chunk_to_output(
        &[14, 15, 0, 0, 0, 0],
        &[1, 1],
        &chunk_dims,
        &dataset_dims,
        elem,
        &mut output,
    );

    let expected: Vec<u8> = (1..=15).collect();
    assert_eq!(output, expected);
}

// ── read_implicit_chunk_entries ──

#[test]
fn implicit_entries_1d() {
    let entries = read_implicit_chunk_entries(
        1000,  // base address
        &[12], // dataset dims
        &[4],  // chunk dims
        16,    // chunk byte size (4 elements * 4 bytes)
    )
    .unwrap();

    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].address, 1000);
    assert_eq!(entries[0].scaled, vec![0]);
    assert_eq!(entries[1].address, 1016);
    assert_eq!(entries[1].scaled, vec![1]);
    assert_eq!(entries[2].address, 1032);
    assert_eq!(entries[2].scaled, vec![2]);
}

#[test]
fn implicit_entries_2d() {
    let entries = read_implicit_chunk_entries(
        0,        // base address
        &[10, 6], // dataset dims
        &[5, 3],  // chunk dims
        60,       // 5*3*4 bytes
    )
    .unwrap();

    assert_eq!(entries.len(), 4); // 2x2 chunks
    assert_eq!(entries[0].scaled, vec![0, 0]);
    assert_eq!(entries[0].address, 0);
    assert_eq!(entries[1].scaled, vec![0, 1]);
    assert_eq!(entries[1].address, 60);
    assert_eq!(entries[2].scaled, vec![1, 0]);
    assert_eq!(entries[2].address, 120);
    assert_eq!(entries[3].scaled, vec![1, 1]);
    assert_eq!(entries[3].address, 180);
}

#[test]
fn implicit_entries_edge_chunks() {
    // Dataset [7], chunk [4] → 2 chunks (one is partial)
    let entries = read_implicit_chunk_entries(100, &[7], &[4], 16).unwrap();

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].scaled, vec![0]);
    assert_eq!(entries[1].scaled, vec![1]);
}

// ── single chunk entries ──

#[test]
fn single_chunk_unfiltered() {
    let entries = read_single_chunk_entries(0x100, 80, None, None, 2).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].address, 0x100);
    assert_eq!(entries[0].filtered_size, 80);
    assert_eq!(entries[0].filter_mask, 0);
    assert_eq!(entries[0].scaled, vec![0, 0]);
}

#[test]
fn single_chunk_filtered() {
    let entries = read_single_chunk_entries(0x200, 80, Some(42), Some(0x01), 1).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].filtered_size, 42);
    assert_eq!(entries[0].filter_mask, 0x01);
}
