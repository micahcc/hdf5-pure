# hdf5-io

Pure Rust HDF5 reader and writer. No C dependencies -- works everywhere
including WASM.

## Reading

Open any HDF5 1.8+ file (superblock v2/v3) and navigate its contents:

```rust
use hdf5_io::File;

let file = File::open("data.h5")?;
let root = file.root_group()?;

// List members, open groups/datasets
for name in root.members()? {
    println!("{name}");
}

let ds = root.dataset("temperature")?;
let shape = ds.shape()?;
let raw = ds.read_raw()?;         // full dataset as bytes
let slice = ds.read_slice(         // hyperslab selection
    &[0, 0], &[10, 20]
)?;
let native = ds.read_native()?;   // byte-swapped to native order
```

### What works (reading)

| Feature                                                                                                                        | Status |
| ------------------------------------------------------------------------------------------------------------------------------ | ------ |
| Superblock v2 and v3                                                                                                           | Full   |
| Groups (new-style link messages + dense/fractal heap)                                                                          | Full   |
| Datasets: contiguous, compact, chunked                                                                                         | Full   |
| Chunk indexes: B-tree v1, B-tree v2, extensible array, fixed array, single chunk, implicit                                     | Full   |
| All HDF5 datatype classes: fixed-point, float, string, compound, enum, array, vlen, opaque, bitfield, reference, time, complex | Full   |
| Variable-length data (strings and sequences via global heap)                                                                   | Full   |
| Hyperslab / slice reads (chunked and contiguous)                                                                               | Full   |
| Byte-order swapping (`read_native`)                                                                                            | Full   |
| Attributes (compact and dense storage, name and creation order)                                                                | Full   |
| Fill values (v1 and v2 messages)                                                                                               | Full   |
| Shared object header messages                                                                                                  | Full   |
| Filter pipelines: deflate/zlib, shuffle, fletcher32, nbit, scaleoffset, LZF                                                    | Full   |
| Hard links, soft links, external links                                                                                         | Parsed |
| Committed (named) datatypes                                                                                                    | Full   |
| Creation order tracking (groups and attributes)                                                                                | Full   |

### Not yet supported (reading)

- **Superblock v0/v1** -- files created with HDF5 < 1.8 or with
  `libver_bounds=(earliest, ...)` cannot be opened
- **Old-style groups** (symbol table / B-tree v1 group storage) -- only
  new-style (link message) groups are supported
- **SZIP decompression** -- returns an error; deflate/shuffle/fletcher32/nbit/scaleoffset/LZF all work
- **Virtual datasets** -- layout is recognized but data cannot be read
- **External file links** -- parsed but not followed (no cross-file I/O)
- **Object/region references** -- the reference type is parsed but
  dereferencing is not implemented
- **Attribute references** (HDF5 1.12+ unified reference API)

## Writing

Build an HDF5 file in memory and write it out:

```rust
use hdf5_io::writer::{FileWriter, WriteOptions};
use hdf5_io::Datatype;

let mut w = FileWriter::new();
let root = w.root_mut();

// Simple dataset
let data: Vec<u8> = (0..100i32).flat_map(|x| x.to_le_bytes()).collect();
root.add_dataset("numbers", Datatype::native_i32(), &[10, 10], data);

// Chunked + compressed
let big: Vec<u8> = vec![0u8; 4000];
root.add_dataset("chunked", Datatype::native_f32(), &[1000], big)
    .set_chunked(&[100], vec![hdf5_io::writer::ChunkFilter::Deflate(6)]);

// Groups and attributes
let grp = root.add_group("results");
grp.add_attribute("version", Datatype::native_i32(), &[], 1i32.to_le_bytes().to_vec());

w.write_to_file("output.h5")?;
```

### What works (writing)

| Feature                                                             | Status |
| ------------------------------------------------------------------- | ------ |
| Superblock v2 and v3                                                | Full   |
| Groups (nested, unlimited depth)                                    | Full   |
| Datasets: contiguous and chunked                                    | Full   |
| Compact storage                                                     | Full   |
| Chunk filters: deflate, shuffle, fletcher32, scaleoffset, nbit, LZF | Full   |
| Variable-length strings and sequences                               | Full   |
| Attributes on groups and datasets                                   | Full   |
| Dense attribute storage (fractal heap + B-tree v2)                  | Full   |
| Dense link storage                                                  | Full   |
| Committed (named) datatypes                                         | Full   |
| Fill values                                                         | Full   |
| Creation order tracking                                             | Full   |
| C library byte-compatible output (`hdf5lib_compat` mode)            | Full   |

### Not yet supported (writing)

- **Streaming / append** -- files are built entirely in memory then
  serialized; no incremental or in-place writes
- **Virtual datasets**
- **SZIP compression**
- **External file storage**
- **Soft links and external links** -- only hard links are written
- **Object/region references**

## Included binaries

- `h5info` -- print the structure of an HDF5 file (groups, datasets,
  attributes, shapes, types)
- `h5json` -- dump an HDF5 file as JSON (requires the `json` feature)

## Features

| Cargo feature | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| `json`        | Enables the `h5json` binary and `serde_json` dependency             |
| `system-zlib` | Links system zlib instead of the bundled miniz (via `flate2/zlib`). |
|               | This is primarily for byte-for-byte write compatbility.             |

## Still needed

Key features that are not yet implemented but would be required for
broader compatibility:

1. **Superblock v0/v1 reading** -- many existing HDF5 files use older
   superblock versions; this is the biggest gap for reading arbitrary
   files from the wild
2. **Old-style (symbol table) group reading** -- closely related to v0/v1
   support; files created with `H5Gopen` / pre-1.8 APIs use B-tree v1 +
   symbol table entries for groups
3. **SZIP filter** -- required for some scientific datasets (NASA, weather
   data)
4. **Streaming / incremental writes** -- building the entire file in
   memory limits output size; an append-oriented writer would enable
   large file creation
5. **Virtual dataset reading** -- needed for files that aggregate data
   across multiple source datasets
6. **Object and region reference dereferencing** -- commonly used in
   HDF5 files to cross-link datasets
7. **Soft/external link traversal on read** -- follow symbolic and
   cross-file links during navigation
8. **Soft/external link writing** -- create symbolic links between groups
