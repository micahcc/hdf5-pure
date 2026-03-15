# hdf5-reader TODO

Pure Rust HDF5 reader targeting superblock v2 and v3 files. WASM-compatible.

## Phase 1: Core Infrastructure
- [x] Crate skeleton, error types, I/O abstraction (`ReadAt` trait)
- [x] Superblock v2/v3 parsing (signature, version, offsets, base addr, root group addr, checksum)
- [x] Checksum validation (Jenkins lookup3 hash)
- [x] `File::open()` / `File::from_bytes()` entry points

## Phase 2: Object Headers
- [x] Object header v2 prefix (magic `OHDR`, version, flags, timestamps, chunk0 size)
- [x] Object header v2 message decoding loop (type, size, flags per message)
- [x] Continuation message (type 0x0010) — follow to next header chunk
- [x] Header chunk parsing (magic `OCHK`, messages, checksum)
- [x] Key message types:
  - [x] Dataspace (0x0001): version 1+2, rank, dimensions, max dimensions
  - [x] Datatype (0x0003): fixed-point, float, string, opaque, bitfield, reference, time
  - [x] Data Layout (0x0008): compact, contiguous, chunked (v3 + v4)
  - [x] Filter Pipeline (0x000B): v1 + v2, all built-in filter IDs
  - [x] Attribute (0x000C): v1/v2/v3, name, datatype, dataspace, value
  - [x] Link (0x0006): hard, soft, external
  - [x] Link Info (0x0002): fractal heap addr, B-tree v2 addrs
  - [x] Object Header Continuation (0x0010)
  - [x] Fill Value (0x0005): version, space/write times, defined flag, value
  - [x] Attribute Info (0x0015): dense attribute storage via fractal heap
  - [x] Group Info (0x000A)
  - [x] Fill Value Old (0x0004)

## Phase 3: B-tree v2
- [x] B-tree v2 header parsing (magic `BTHD`, version, type, node/record size, depth, root addr)
- [x] Leaf node parsing (magic `BTLF`, records)
- [x] Internal node parsing (magic `BTIN`, records + child pointers)
- [x] Tree iteration (walk all records in sorted order)
- [x] Record type 5: Link Name for indexed groups (hash + heap ID)
- [x] Record type 6: Creation Order for indexed groups
- [x] Record type 8: Attribute Name for indexed attributes
- [x] Record type 9: Attribute Creation Order
- [x] Record types 10/11: Chunked dataset indexing

## Phase 4: Fractal Heap
- [x] Fractal heap header parsing (magic `FRHP`, all fields + doubling table info)
- [x] Direct block reading (magic `FHDB`)
- [x] Indirect block traversal (magic `FHIB`)
- [x] Managed object lookup from heap ID
- [x] Tiny object decoding (inline in heap ID)
- [x] Row/column calculations for block addressing
- [x] Huge object reading (B-tree v2 lookup)
- [x] Filtered direct block handling

## Phase 4b: Global Heap
- [x] Global heap collection parsing (magic `GCOL`)
- [x] Object lookup by index
- [x] Variable-length element resolution (heap ID → data)

## Phase 5: Datatype Decoding (full)
- [x] Fixed-point (integer): byte order, sign, bit offset, precision
- [x] Floating-point: byte order, mantissa/exponent layout, bias
- [x] String: padding type, character set
- [x] Opaque: tag string
- [x] Bitfield: byte order, bit offset, precision
- [x] Reference: object / dataset region
- [x] Time: bit precision
- [x] Compound: member names, offsets, types (recursive)
- [x] Enumeration: base type, member names, values
- [x] Array: dimensions, element type
- [x] Variable-length: element type, padding, character set
- [x] Complex (HDF5 2.0): base floating-point type

## Phase 6: Data Reading
- [x] Contiguous layout: read raw bytes from file offset
- [x] Compact layout: read raw bytes from object header
- [x] Chunked layout with B-tree v1 index (layout v3)
- [x] Chunked layout with B-tree v2 index
- [x] Chunked layout with extensible array index (data blocks + super blocks + paged data blocks)
- [x] Chunked layout with fixed array index
- [x] Single chunk optimization
- [x] Filter pipeline application on chunked read
- [x] Hyperslab / partial reads (`read_slice`)
- [x] Type conversion on read (endian swap via `read_native`)

## Phase 7: Filter Pipeline
- [x] Pipeline framework (ordered filter application)
- [x] Deflate (zlib) decompression via `flate2`
- [x] Shuffle (byte de-interleaving)
- [x] Fletcher32 checksum verification
- [x] N-bit filter
- [x] Scale-offset filter
- [-] SZIP decompression — **won't implement**. HDF5's SZIP format is not standard CCSDS 121.0;
      no pure Rust decoder exists. Would require native libaec FFI, breaking WASM compatibility.

## Phase 8: Navigation / Public API
- [x] `File::open()` — from filesystem path
- [x] `File::from_bytes()` — from in-memory buffer
- [x] `File::root_group()` — navigate to root group
- [x] `Group::members()` — list all child link names
- [x] `Group::find_link()` / `Group::group()` / `Group::dataset()`
- [x] `Dataset::datatype()`, `Dataset::dataspace()`, `Dataset::shape()`
- [x] `Dataset::read_raw()` — entire dataset as bytes (contiguous + compact)
- [x] `Dataset::attributes()` / `Group::attributes()`
- [x] `File::open_path("/group1/subgroup/dataset")`
- [x] `Dataset::read_vlen()` — read variable-length dataset elements
- [x] `Dataset::read_vlen_strings()` — read vlen strings as `Vec<String>`
- [x] `Dataset::read_slice()` — read hyperslab
- [x] Shared/committed datatype resolution (shared message records v1/v2/v3)
- [x] Shared datatypes/dataspaces in attributes (attribute flags bits 0/1)
- [x] Empty/unallocated chunked datasets (return zeros when address is UNDEF)

## Phase 8b: Bug Fixes (audit against C library)

Sources in
```
~/sources/hdf5
```

### Critical — B-tree v2
- [x] Internal node layout: records and children are NOT interleaved; all records come first, then all child pointers (H5B2cache.c:655-682)
- [x] Internal node `total_rec_bytes`: must use structural `cum_max_nrec_size` from node_info array, not `bytes_needed(total_records)`
- [x] Type 10/11 chunk records: scaled offsets are fixed 8-byte UINT64DECODE per dimension, not variable-width (H5Dbtree2.c:410-429)
- [x] `chunk_size_len` formula: C uses `1 + ((log2(chunk_size) + 8) / 8)` capped at 8, not `bytes_needed_for(size)` (H5Dbtree2.c:48-62)

### Critical — Fractal heap
- [x] `max_direct_rows` off by one: C uses `(max_direct_bits - start_bits) + 2`, we use `+ 1` (H5HFdtable.c:94)
- [x] `block_size_for_row` must NOT cap at `max_direct_block_size` — indirect rows continue doubling (H5HFdtable.c:110-119)
- [x] Child indirect block `nrows`: C uses `(log2(row_block_size[row]) - first_row_bits) + 1` (H5HFdtable.c:237-251)

### High — Object header / global heap
- [x] Nil message (type 0) should be skipped with its size field, not break the message loop (object_header/mod.rs:208-215)
- [x] Global heap free-space entry (index 0) should skip over its data, not break the object loop (global_heap.rs:82-85)

### High — Datatype parsing
- [x] Compound v1/v2 name padding: should pad name length to 8-byte multiple from name start, not align absolute position (datatype.rs:376)
- [x] Compound v3 offset width: missing 3-byte case for sizes 65536–16777215 (datatype.rs:398-404)
- [x] `encoded_size` for VarLen/Enum/Array returns wrong values — affects compound types containing these

### Medium
- [x] Fractal heap huge object `directly_accessed` flag may read wrong bit
- [x] VAX byte order mapping may be inverted for floating-point types

## Phase 9: WASM & Portability
- [x] `ReadAt` impl for `&[u8]` / `Vec<u8>` (in-memory buffer)
- [x] `ReadAt` impl for `std::fs::File` (native)
- [ ] Async `ReadAt` variant for HTTP range requests
- [ ] Build and test on `wasm32-unknown-unknown`
- [ ] Build and test on `wasm32-wasip1`

## Phase 10: Testing & Validation
- [x] Fixture generator (C program using libhdf5)
- [x] Fixture files: superblock v2 + v3, contiguous, compact, nested groups, attributes
- [x] Integration tests: superblock, metadata, contiguous read, compact read, navigation, path lookup, attributes
- [x] Fixture files for chunked + compressed datasets
- [x] Fixture files for compound, enum, array types
- [x] Fletcher32 checksum filter fixture + integration test
- [x] Fixture files for vlen string + vlen sequence types
- [x] Fill value fixture + integration test
- [x] Dense attributes fixture + integration test
- [x] B-tree v2 chunk index fixture + integration test
- [x] Big-endian fixture + integration test (type conversion)
- [x] Hyperslab read integration tests (contiguous, compact, chunked)
- [x] Committed datatype fixture + integration test
- [x] Extensible array data blocks fixture + integration test (25 chunks)
- [x] Shared attribute type fixture + integration test
- [x] Empty/unallocated chunked dataset fixture + integration test
- [ ] Large file (>4GB) fixture
- [ ] SWMR file fixture (superblock v3 specific)
- [ ] Fuzz testing on malformed inputs
