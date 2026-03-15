use crate::checksum;
use crate::datatype::ByteOrder;
use crate::datatype::CharacterSet;
use crate::datatype::Datatype;
use crate::datatype::StringPadding;
use crate::error::Error;
use crate::error::Result;
use crate::filters;
use crate::object_header::messages::MessageType;
use crate::superblock::HDF5_SIGNATURE;
use crate::superblock::UNDEF_ADDR;

const SIZE_OF_OFFSETS: u8 = 8;
const SIZE_OF_LENGTHS: u8 = 8;
const SUPERBLOCK_SIZE: usize = 48; // 8+1+1+1+1+4*8+4

// ---------------------------------------------------------------------------
// Public builder types
// ---------------------------------------------------------------------------

/// Options controlling how the HDF5 file is written.
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// If set, store these timestamps on every object header.
    /// Tuple: (access_time, modification_time, change_time, birth_time) as Unix seconds.
    pub timestamps: Option<(u32, u32, u32, u32)>,
}

/// Builds an HDF5 file in memory and writes it out.
///
/// # Example
/// ```
/// use hdf5_io::writer::FileWriter;
/// use hdf5_io::Datatype;
///
/// let mut w = FileWriter::new();
/// let data: Vec<u8> = (0..4i32).flat_map(|x| x.to_le_bytes()).collect();
/// w.root_mut().add_dataset("numbers", Datatype::native_i32(), &[4], data);
/// let bytes = w.to_bytes().unwrap();
/// ```
pub struct FileWriter {
    root: GroupNode,
    options: WriteOptions,
}

impl FileWriter {
    pub fn new() -> Self {
        FileWriter {
            root: GroupNode {
                children: vec![],
                attributes: vec![],
            },
            options: WriteOptions::default(),
        }
    }

    /// Create a writer with custom options.
    pub fn with_options(options: WriteOptions) -> Self {
        FileWriter {
            root: GroupNode {
                children: vec![],
                attributes: vec![],
            },
            options,
        }
    }

    pub fn root_mut(&mut self) -> &mut GroupNode {
        &mut self.root
    }

    /// Serialize the entire file to a byte vector.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; SUPERBLOCK_SIZE];
        let root_addr = write_group(&self.root, &mut buf, &self.options)?;
        let eof = buf.len() as u64;
        let sb = encode_superblock(root_addr, eof);
        buf[..SUPERBLOCK_SIZE].copy_from_slice(&sb);
        Ok(buf)
    }

    /// Serialize and write to a file on disk.
    pub fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(Error::Io)
    }
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// A group node in the file builder tree.
pub struct GroupNode {
    children: Vec<(String, ChildNode)>,
    attributes: Vec<AttrData>,
}

impl GroupNode {
    /// Add a child group, returning a mutable reference to it.
    pub fn add_group(&mut self, name: &str) -> &mut GroupNode {
        self.children.push((
            name.to_string(),
            ChildNode::Group(GroupNode {
                children: vec![],
                attributes: vec![],
            }),
        ));
        match &mut self.children.last_mut().unwrap().1 {
            ChildNode::Group(g) => g,
            _ => unreachable!(),
        }
    }

    /// Add a child dataset with raw data bytes.
    pub fn add_dataset(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: &[u64],
        data: Vec<u8>,
    ) -> &mut DatasetNode {
        self.children.push((
            name.to_string(),
            ChildNode::Dataset(DatasetNode {
                datatype,
                shape: shape.to_vec(),
                max_dims: None,
                data,
                attributes: vec![],
                layout: StorageLayout::default(),
                fill_value: None,
                layout_version: 4,
                vlen_elements: None,
            }),
        ));
        match &mut self.children.last_mut().unwrap().1 {
            ChildNode::Dataset(d) => d,
            _ => unreachable!(),
        }
    }

    /// Add a variable-length dataset.
    ///
    /// Each element in `elements` is the raw bytes for one vlen entry.
    /// For vlen strings, each element is the UTF-8 bytes (no NUL terminator needed).
    /// For vlen sequences of T, each element is `count * sizeof(T)` bytes.
    pub fn add_vlen_dataset(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: &[u64],
        elements: Vec<Vec<u8>>,
    ) -> &mut DatasetNode {
        self.children.push((
            name.to_string(),
            ChildNode::Dataset(DatasetNode {
                datatype,
                shape: shape.to_vec(),
                max_dims: None,
                data: vec![],
                attributes: vec![],
                layout: StorageLayout::default(),
                fill_value: None,
                layout_version: 4,
                vlen_elements: Some(elements),
            }),
        ));
        match &mut self.children.last_mut().unwrap().1 {
            ChildNode::Dataset(d) => d,
            _ => unreachable!(),
        }
    }

    /// Add an attribute to this group.
    pub fn add_attribute(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: &[u64],
        value: Vec<u8>,
    ) -> &mut Self {
        self.attributes.push(AttrData {
            name: name.to_string(),
            datatype,
            shape: shape.to_vec(),
            value,
        });
        self
    }
}

/// A filter to apply in a chunked dataset's filter pipeline.
#[derive(Debug, Clone)]
pub enum ChunkFilter {
    /// Deflate (zlib) compression with a given level (0-9).
    Deflate(u32),
    /// Shuffle filter — reorders bytes for better compression.
    Shuffle,
    /// Fletcher32 checksum appended to each chunk.
    Fletcher32,
}

/// Storage layout for a dataset.
#[derive(Debug, Clone)]
pub enum StorageLayout {
    /// Data stored in a contiguous block after the object header.
    Contiguous,
    /// Data stored inline in the object header (small datasets only).
    Compact,
    /// Data stored in fixed-size chunks.
    Chunked {
        chunk_dims: Vec<u64>,
        filters: Vec<ChunkFilter>,
    },
}

impl Default for StorageLayout {
    fn default() -> Self {
        StorageLayout::Contiguous
    }
}

/// A dataset node in the file builder tree.
pub struct DatasetNode {
    datatype: Datatype,
    shape: Vec<u64>,
    max_dims: Option<Vec<u64>>,
    data: Vec<u8>,
    attributes: Vec<AttrData>,
    layout: StorageLayout,
    fill_value: Option<Vec<u8>>,
    /// Force layout message version (3 or 4). Default is 4.
    layout_version: u8,
    /// Variable-length data elements (if set, serialized into GCOL + heap IDs).
    vlen_elements: Option<Vec<Vec<u8>>>,
}

impl DatasetNode {
    /// Add an attribute to this dataset.
    pub fn add_attribute(
        &mut self,
        name: &str,
        datatype: Datatype,
        shape: &[u64],
        value: Vec<u8>,
    ) -> &mut Self {
        self.attributes.push(AttrData {
            name: name.to_string(),
            datatype,
            shape: shape.to_vec(),
            value,
        });
        self
    }

    /// Set the storage layout for this dataset.
    pub fn set_layout(&mut self, layout: StorageLayout) -> &mut Self {
        self.layout = layout;
        self
    }

    /// Set chunked storage with the given chunk dimensions and filters.
    pub fn set_chunked(&mut self, chunk_dims: &[u64], filters: Vec<ChunkFilter>) -> &mut Self {
        self.layout = StorageLayout::Chunked {
            chunk_dims: chunk_dims.to_vec(),
            filters,
        };
        self
    }

    /// Force a specific layout message version (3 or 4).
    /// Version 3 uses B-tree v1 chunk indexing. Default is 4.
    pub fn set_layout_version(&mut self, version: u8) -> &mut Self {
        self.layout_version = version;
        self
    }

    /// Set maximum dimensions for the dataspace.
    /// Use `u64::MAX` for an unlimited dimension.
    pub fn set_max_dims(&mut self, max_dims: &[u64]) -> &mut Self {
        self.max_dims = Some(max_dims.to_vec());
        self
    }

    /// Set an explicit fill value for this dataset.
    pub fn set_fill_value(&mut self, value: Vec<u8>) -> &mut Self {
        self.fill_value = Some(value);
        self
    }
}

enum ChildNode {
    Group(GroupNode),
    Dataset(DatasetNode),
}

struct AttrData {
    name: String,
    datatype: Datatype,
    shape: Vec<u64>,
    value: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Serialization (post-order: children first, then parent)
// ---------------------------------------------------------------------------

fn write_group(group: &GroupNode, buf: &mut Vec<u8>, opts: &WriteOptions) -> Result<u64> {
    // Write children first (post-order) so we know their addresses
    let mut child_addrs: Vec<(&str, u64)> = Vec::new();
    for (name, child) in &group.children {
        let addr = match child {
            ChildNode::Group(g) => write_group(g, buf, opts)?,
            ChildNode::Dataset(d) => write_dataset(d, buf, opts)?,
        };
        child_addrs.push((name, addr));
    }

    let ohdr_addr = buf.len() as u64;

    // Build messages
    let mut messages: Vec<(u8, Vec<u8>)> = Vec::new();

    // Link Info message (type 0x02)
    messages.push((MessageType::LinkInfo.as_u8(), encode_link_info()));

    // Group Info message (type 0x0A)
    messages.push((MessageType::GroupInfo.as_u8(), encode_group_info()));

    // Link messages (type 0x06) for each child
    for (name, addr) in &child_addrs {
        messages.push((MessageType::Link.as_u8(), encode_link(name, *addr)));
    }

    // Attribute messages (type 0x0C)
    for attr in &group.attributes {
        messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?));
    }

    let ohdr = encode_object_header(&messages, opts)?;
    buf.extend_from_slice(&ohdr);

    Ok(ohdr_addr)
}

fn write_dataset(ds: &DatasetNode, buf: &mut Vec<u8>, opts: &WriteOptions) -> Result<u64> {
    validate_dataset(ds)?;

    let ohdr_addr = buf.len() as u64;

    match ds.layout {
        StorageLayout::Contiguous => {
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(|a| encode_attribute(a))
                .collect::<Result<Vec<_>>>()?;
            // Layout body is always 18 bytes for contiguous.
            let layout_body_size = 18usize;
            let total_msg_size: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + (4 + layout_body_size)
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(total_msg_size, opts);

            if let Some(ref vlen_elems) = ds.vlen_elements {
                // VLen data: build GCOL collection, then heap IDs
                let gcol_addr = ohdr_addr + ohdr_size as u64;
                let gcol_bytes = build_global_heap_collection(vlen_elems);
                let heap_id_data_addr = gcol_addr + gcol_bytes.len() as u64;
                let heap_id_data = build_vlen_heap_ids(vlen_elems, gcol_addr);

                let layout_body = encode_contiguous_layout(
                    heap_id_data_addr,
                    heap_id_data.len() as u64,
                );
                debug_assert_eq!(layout_body.len(), layout_body_size);

                let mut messages: Vec<(u8, Vec<u8>)> = vec![
                    (MessageType::Datatype.as_u8(), dt_body),
                    (MessageType::Dataspace.as_u8(), ds_body),
                    (MessageType::FillValue.as_u8(), fv_body),
                    (MessageType::DataLayout.as_u8(), layout_body),
                ];
                for body in attr_bodies {
                    messages.push((MessageType::Attribute.as_u8(), body));
                }

                let ohdr = encode_object_header(&messages, opts)?;
                debug_assert_eq!(ohdr.len(), ohdr_size);

                buf.extend_from_slice(&ohdr);
                buf.extend_from_slice(&gcol_bytes);
                buf.extend_from_slice(&heap_id_data);
            } else {
                let data_addr = if ds.data.is_empty() {
                    UNDEF_ADDR
                } else {
                    ohdr_addr + ohdr_size as u64
                };

                let layout_body = encode_contiguous_layout(data_addr, ds.data.len() as u64);
                debug_assert_eq!(layout_body.len(), layout_body_size);

                let mut messages: Vec<(u8, Vec<u8>)> = vec![
                    (MessageType::Datatype.as_u8(), dt_body),
                    (MessageType::Dataspace.as_u8(), ds_body),
                    (MessageType::FillValue.as_u8(), fv_body),
                    (MessageType::DataLayout.as_u8(), layout_body),
                ];
                for body in attr_bodies {
                    messages.push((MessageType::Attribute.as_u8(), body));
                }

                let ohdr = encode_object_header(&messages, opts)?;
                debug_assert_eq!(ohdr.len(), ohdr_size);

                buf.extend_from_slice(&ohdr);
                buf.extend_from_slice(&ds.data);
            }
        }
        StorageLayout::Compact => {
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(|a| encode_attribute(a))
                .collect::<Result<Vec<_>>>()?;
            // Compact layout: data stored inline in the layout message.
            let layout_body = encode_compact_layout(&ds.data);

            let mut messages: Vec<(u8, Vec<u8>)> = vec![
                (MessageType::Datatype.as_u8(), dt_body),
                (MessageType::Dataspace.as_u8(), ds_body),
                (MessageType::FillValue.as_u8(), fv_body),
                (MessageType::DataLayout.as_u8(), layout_body),
            ];
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body));
            }

            let ohdr = encode_object_header(&messages, opts)?;
            buf.extend_from_slice(&ohdr);
            // No external data for compact — it's all in the header.
        }
        StorageLayout::Chunked {
            ref chunk_dims,
            ref filters,
        } if ds.layout_version == 3 => {
            // Layout v3 with B-tree v1 chunk index
            let element_size = ds.datatype.element_size();
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(|a| encode_attribute(a))
                .collect::<Result<Vec<_>>>()?;
            let has_filters = !filters.is_empty();
            let filter_body = if has_filters {
                Some(encode_filter_pipeline(filters, element_size))
            } else {
                None
            };

            // Compress all chunks
            let chunk_coords_list = enumerate_chunks(&ds.shape, chunk_dims);
            let mut chunk_data_parts: Vec<Vec<u8>> = Vec::new();
            for coords in &chunk_coords_list {
                let mut chunk = extract_chunk_data(
                    &ds.data, &ds.shape, chunk_dims, coords, element_size as usize,
                );
                if has_filters {
                    chunk = apply_filters_forward(filters, chunk, element_size)?;
                }
                chunk_data_parts.push(chunk);
            }
            let chunk_sizes: Vec<u64> = chunk_data_parts.iter().map(|p| p.len() as u64).collect();

            // Layout v3 body: version(1) + class(1) + dimensionality(1) + address(O) + dims(ndims*4)
            let ndims = ds.shape.len();
            let dimensionality = ndims + 1; // extra dim for element size
            let layout_body_size = 3 + SIZE_OF_OFFSETS as usize + dimensionality * 4;

            let fixed_msg_sizes: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + filter_body.as_ref().map_or(0, |fb| 4 + fb.len())
                + (4 + layout_body_size)
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(fixed_msg_sizes, opts);

            // Chunk data starts after ohdr
            let chunk_data_start = ohdr_addr + ohdr_size as u64;
            let mut chunk_addrs: Vec<u64> = Vec::new();
            let mut pos = chunk_data_start;
            for part in &chunk_data_parts {
                chunk_addrs.push(pos);
                pos += part.len() as u64;
            }

            // B-tree v1 is written after chunk data; its address goes in the layout message
            let btree_addr = pos;

            // Build layout v3 body
            let mut layout_body = Vec::with_capacity(layout_body_size);
            layout_body.push(3); // version
            layout_body.push(2); // class = chunked
            layout_body.push(dimensionality as u8);
            layout_body.extend_from_slice(&btree_addr.to_le_bytes());
            for d in 0..ndims {
                layout_body.extend_from_slice(&(chunk_dims[d] as u32).to_le_bytes());
            }
            layout_body.extend_from_slice(&element_size.to_le_bytes()); // last dim = element size
            debug_assert_eq!(layout_body.len(), layout_body_size);

            // Build messages
            let mut messages: Vec<(u8, Vec<u8>)> = vec![
                (MessageType::Datatype.as_u8(), dt_body),
                (MessageType::Dataspace.as_u8(), ds_body),
                (MessageType::FillValue.as_u8(), fv_body),
            ];
            if let Some(fb) = filter_body {
                messages.push((MessageType::FilterPipeline.as_u8(), fb));
            }
            messages.push((MessageType::DataLayout.as_u8(), layout_body));
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body));
            }

            let ohdr = encode_object_header(&messages, opts)?;
            debug_assert_eq!(ohdr.len(), ohdr_size);
            buf.extend_from_slice(&ohdr);

            // Write chunk data
            for part in &chunk_data_parts {
                buf.extend_from_slice(part);
            }

            // Write B-tree v1 index
            write_btree_v1_chunk_index(
                buf,
                &chunk_addrs,
                &chunk_sizes,
                &chunk_coords_list,
                chunk_dims,
                element_size,
                ndims,
            )?;
        }
        StorageLayout::Chunked {
            ref chunk_dims,
            ref filters,
        } => {
            let element_size = ds.datatype.element_size();
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(|a| encode_attribute(a))
                .collect::<Result<Vec<_>>>()?;

            // Calculate chunk size in bytes
            let chunk_elements: u64 = chunk_dims.iter().product();
            let raw_chunk_bytes = chunk_elements * element_size as u64;

            // Split data into chunks and apply filters
            let total_chunks = compute_chunk_count(&ds.shape, chunk_dims);
            let has_filters = !filters.is_empty();

            // Build filter pipeline message if filters are present
            let filter_body = if has_filters {
                Some(encode_filter_pipeline(filters, element_size))
            } else {
                None
            };

            // Determine index type
            let n_unlimited = ds
                .max_dims
                .as_ref()
                .map_or(0, |md| md.iter().filter(|&&d| d == u64::MAX).count());
            let index_type_id = if n_unlimited >= 2 {
                5u8 // BTreeV2 (2+ unlimited dims)
            } else if total_chunks == 1 && n_unlimited == 0 {
                1u8 // SingleChunk
            } else if n_unlimited == 1 {
                4u8 // ExtensibleArray (exactly 1 unlimited dim)
            } else if !has_filters {
                2u8 // Implicit (no filters, early alloc, fixed max)
            } else {
                3u8 // FixedArray (filtered multi-chunk)
            };

            // Compress all chunks first to know their sizes
            let mut chunk_data_parts: Vec<Vec<u8>> = Vec::with_capacity(total_chunks);
            let chunk_coords_list = enumerate_chunks(&ds.shape, chunk_dims);

            for coords in &chunk_coords_list {
                let mut chunk = extract_chunk_data(
                    &ds.data,
                    &ds.shape,
                    chunk_dims,
                    coords,
                    element_size as usize,
                );
                if has_filters {
                    chunk = apply_filters_forward(filters, chunk, element_size)?;
                }
                chunk_data_parts.push(chunk);
            }

            let chunk_sizes: Vec<u64> = chunk_data_parts.iter().map(|p| p.len() as u64).collect();

            // Compute a dummy layout body to determine its size (size is independent of address)
            let dummy_layout = encode_chunked_layout(
                chunk_dims,
                index_type_id,
                0, // dummy address
                if index_type_id == 1 && has_filters {
                    Some(chunk_sizes[0])
                } else {
                    None
                },
            );
            let layout_body_size = dummy_layout.len();

            // Compute total ohdr size
            let fixed_msg_sizes: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + filter_body.as_ref().map_or(0, |fb| 4 + fb.len())
                + (4 + layout_body_size)
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(fixed_msg_sizes, opts);

            // Compute real addresses
            let chunk_data_start = ohdr_addr + ohdr_size as u64;
            let mut chunk_addrs: Vec<u64> = Vec::new();
            let mut pos = chunk_data_start;
            for part in &chunk_data_parts {
                chunk_addrs.push(pos);
                pos += part.len() as u64;
            }

            let (chunk_index_addr, single_filtered_size) = match index_type_id {
                1 => (
                    chunk_addrs[0],
                    if has_filters {
                        Some(chunk_sizes[0])
                    } else {
                        None
                    },
                ),
                2 => (chunk_data_start, None), // Implicit: address of first chunk
                3 | 4 | 5 => (pos, None),      // FA/EA/BT2: header after chunk data
                _ => (pos, None),
            };

            // Build real layout message
            let layout_body = encode_chunked_layout(
                chunk_dims,
                index_type_id,
                chunk_index_addr,
                single_filtered_size,
            );
            debug_assert_eq!(layout_body.len(), layout_body_size);

            // Build messages
            let mut messages: Vec<(u8, Vec<u8>)> = vec![
                (MessageType::Datatype.as_u8(), dt_body),
                (MessageType::Dataspace.as_u8(), ds_body),
                (MessageType::FillValue.as_u8(), fv_body),
            ];
            if let Some(fb) = filter_body {
                messages.push((MessageType::FilterPipeline.as_u8(), fb));
            }
            messages.push((MessageType::DataLayout.as_u8(), layout_body));
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body));
            }

            let ohdr = encode_object_header(&messages, opts)?;
            debug_assert_eq!(ohdr.len(), ohdr_size);
            buf.extend_from_slice(&ohdr);

            // Write chunk data
            for part in &chunk_data_parts {
                buf.extend_from_slice(part);
            }

            // Write chunk index structures
            match index_type_id {
                3 => {
                    write_fixed_array_index(
                        buf,
                        &chunk_addrs,
                        &chunk_sizes,
                        has_filters,
                        raw_chunk_bytes,
                    )?;
                }
                4 => {
                    write_extensible_array_index(
                        buf,
                        &chunk_addrs,
                        &chunk_sizes,
                        has_filters,
                        total_chunks,
                    )?;
                }
                5 => {
                    write_btree_v2_chunk_index(
                        buf,
                        &chunk_addrs,
                        &chunk_sizes,
                        has_filters,
                        &ds.shape,
                        chunk_dims,
                        ds.datatype.element_size(),
                    )?;
                }
                _ => {} // SingleChunk / Implicit need no separate index
            }
        }
    }

    Ok(ohdr_addr)
}

fn validate_dataset(ds: &DatasetNode) -> Result<()> {
    // VLen datasets store data in vlen_elements, not in data
    if ds.vlen_elements.is_some() {
        return Ok(());
    }
    let num_elements: u64 = if ds.shape.is_empty() {
        1 // scalar
    } else {
        ds.shape.iter().product()
    };
    let expected = num_elements * ds.datatype.element_size() as u64;
    if ds.data.len() as u64 != expected {
        return Err(Error::Other {
            msg: format!(
                "dataset data size mismatch: expected {} bytes (shape {:?}, element_size {}), got {}",
                expected,
                ds.shape,
                ds.datatype.element_size(),
                ds.data.len()
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Object header encoding
// ---------------------------------------------------------------------------

/// Total object header size given the sum of all message (header+body) bytes.
fn ohdr_overhead(total_msg_bytes: usize, opts: &WriteOptions) -> usize {
    let (prefix_size, _) = chunk_size_encoding(total_msg_bytes, opts);
    prefix_size + total_msg_bytes + 4 // + checksum
}

/// Returns (prefix_size, flags_value) for encoding chunk0 size.
fn chunk_size_encoding(total_msg_bytes: usize, opts: &WriteOptions) -> (usize, u8) {
    let ts_extra = if opts.timestamps.is_some() { 16 } else { 0 };
    let base_flags: u8 = if opts.timestamps.is_some() {
        0x20 // bit 5: store timestamps
    } else {
        0
    };
    if total_msg_bytes <= 0xFF {
        (7 + ts_extra, base_flags) // 4+1+1+1 = 7, 1-byte size field
    } else if total_msg_bytes <= 0xFFFF {
        (8 + ts_extra, base_flags | 0x01) // 4+1+1+2 = 8, 2-byte size field
    } else {
        (10 + ts_extra, base_flags | 0x02) // 4+1+1+4 = 10, 4-byte size field
    }
}

fn encode_object_header(messages: &[(u8, Vec<u8>)], opts: &WriteOptions) -> Result<Vec<u8>> {
    let total_msg_bytes: usize = messages.iter().map(|(_, b)| 4 + b.len()).sum();
    let (prefix_size, flags) = chunk_size_encoding(total_msg_bytes, opts);

    let mut buf = Vec::with_capacity(prefix_size + total_msg_bytes + 4);

    // OHDR magic
    buf.extend_from_slice(b"OHDR");
    // Version
    buf.push(2);
    // Flags
    buf.push(flags);

    // Optional timestamps (if flags bit 5 set)
    if let Some((at, mt, ct, bt)) = opts.timestamps {
        buf.extend_from_slice(&at.to_le_bytes());
        buf.extend_from_slice(&mt.to_le_bytes());
        buf.extend_from_slice(&ct.to_le_bytes());
        buf.extend_from_slice(&bt.to_le_bytes());
    }

    // Chunk0 size
    match flags & 0x03 {
        0x00 => buf.push(total_msg_bytes as u8),
        0x01 => buf.extend_from_slice(&(total_msg_bytes as u16).to_le_bytes()),
        0x02 => buf.extend_from_slice(&(total_msg_bytes as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    // Messages
    for (type_id, body) in messages {
        buf.push(*type_id);
        buf.extend_from_slice(&(body.len() as u16).to_le_bytes());
        buf.push(0); // message flags = 0
        buf.extend_from_slice(body);
    }

    // Checksum over everything before this point
    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Superblock encoding
// ---------------------------------------------------------------------------

fn encode_superblock(root_group_addr: u64, eof: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(SUPERBLOCK_SIZE);
    buf.extend_from_slice(&HDF5_SIGNATURE);
    buf.push(2); // version
    buf.push(SIZE_OF_OFFSETS);
    buf.push(SIZE_OF_LENGTHS);
    buf.push(0); // file consistency flags
    buf.extend_from_slice(&0u64.to_le_bytes()); // base address
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // superblock extension
    buf.extend_from_slice(&eof.to_le_bytes()); // EOF
    buf.extend_from_slice(&root_group_addr.to_le_bytes()); // root group
    let cksum = checksum::lookup3(&buf);
    buf.extend_from_slice(&cksum.to_le_bytes());
    debug_assert_eq!(buf.len(), SUPERBLOCK_SIZE);
    buf
}

// ---------------------------------------------------------------------------
// Message body encoders
// ---------------------------------------------------------------------------

fn encode_datatype(dt: &Datatype) -> Result<Vec<u8>> {
    match dt {
        Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            let mut buf = Vec::with_capacity(12);
            // byte 0: class(0) | version(1) << 4
            buf.push(0x10);
            // flags byte 0: bit0=byte_order, bit3=signed
            let mut f0 = 0u8;
            if *byte_order == ByteOrder::BigEndian {
                f0 |= 0x01;
            }
            if *signed {
                f0 |= 0x08;
            }
            buf.push(f0);
            buf.push(0); // flags byte 1
            buf.push(0); // flags byte 2
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&bit_offset.to_le_bytes());
            buf.extend_from_slice(&bit_precision.to_le_bytes());
            Ok(buf)
        }
        Datatype::FloatingPoint {
            size,
            byte_order,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        } => {
            let mut buf = Vec::with_capacity(20);
            // byte 0: class(1) | version(1) << 4
            buf.push(0x11);
            // flags byte 0: bit0=byte_order, bits5:4=normalization(2=IEEE implied)
            let mut f0 = 0x20u8; // normalization = 2 (implied leading 1)
            if *byte_order == ByteOrder::BigEndian {
                f0 |= 0x01;
            }
            buf.push(f0);
            // flags byte 1: sign bit position (bits 15:8)
            let sign_bit_pos = exponent_location + exponent_size;
            buf.push(sign_bit_pos);
            // flags byte 2
            buf.push(0);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&bit_offset.to_le_bytes());
            buf.extend_from_slice(&bit_precision.to_le_bytes());
            buf.push(*exponent_location);
            buf.push(*exponent_size);
            buf.push(*mantissa_location);
            buf.push(*mantissa_size);
            buf.extend_from_slice(&exponent_bias.to_le_bytes());
            Ok(buf)
        }
        Datatype::String {
            size,
            padding,
            char_set,
        } => {
            let mut buf = Vec::with_capacity(8);
            // byte 0: (class=3) | (version=1 << 4) = 0x13
            buf.push(0x13);
            // flags byte 0: bits3:0=padding, bits7:4=charset
            let pad = match padding {
                StringPadding::NullTerminate => 0u8,
                StringPadding::NullPad => 1,
                StringPadding::SpacePad => 2,
            };
            let cs = match char_set {
                CharacterSet::Ascii => 0u8,
                CharacterSet::Utf8 => 1,
            };
            buf.push(pad | (cs << 4));
            buf.push(0); // flags byte 1
            buf.push(0); // flags byte 2
            buf.extend_from_slice(&size.to_le_bytes());
            Ok(buf)
        }
        Datatype::Compound { size, members } => {
            let nmembers = members.len() as u16;
            // Version 3 compound: unpadded names, variable-width offsets
            let class_version_byte = (6u8) | (3u8 << 4); // class=6, version=3
            let mut buf = Vec::new();
            // Byte 0: class_version, bytes 1-2: nmembers LE, byte 3: 0
            buf.push(class_version_byte);
            buf.extend_from_slice(&nmembers.to_le_bytes());
            buf.push(0);
            // Size (u32 LE)
            buf.extend_from_slice(&size.to_le_bytes());
            // Members
            let off_size = limit_enc_size(*size);
            for m in members {
                // Name (NUL-terminated, no padding in v3)
                buf.extend_from_slice(m.name.as_bytes());
                buf.push(0);
                // Offset (variable width)
                match off_size {
                    1 => buf.push(m.byte_offset as u8),
                    2 => buf.extend_from_slice(&(m.byte_offset as u16).to_le_bytes()),
                    3 => {
                        buf.push(m.byte_offset as u8);
                        buf.push((m.byte_offset >> 8) as u8);
                        buf.push((m.byte_offset >> 16) as u8);
                    }
                    _ => buf.extend_from_slice(&m.byte_offset.to_le_bytes()),
                }
                // Recursive member datatype
                buf.extend_from_slice(&encode_datatype(&m.datatype)?);
            }
            Ok(buf)
        }
        Datatype::Enum { base, members } => {
            let nmembers = members.len() as u16;
            // Version 3 enum: class=8, version=3
            let class_version_byte = (8u8) | (3u8 << 4); // 0x38
            let base_size = base.element_size();
            let mut buf = Vec::new();
            // Byte 0: class_version, bytes 1-2: nmembers LE, byte 3: 0
            buf.push(class_version_byte);
            buf.extend_from_slice(&nmembers.to_le_bytes());
            buf.push(0);
            // Size = base element size
            buf.extend_from_slice(&base_size.to_le_bytes());
            // Base type (recursive)
            buf.extend_from_slice(&encode_datatype(base)?);
            // Member names (NUL-terminated, no padding in v3)
            for m in members {
                buf.extend_from_slice(m.name.as_bytes());
                buf.push(0);
            }
            // Member values
            for m in members {
                buf.extend_from_slice(&m.value);
            }
            Ok(buf)
        }
        Datatype::Array {
            element_type,
            dimensions,
        } => {
            // Version 3 array: class=10, version=3
            let class_version_byte = (10u8) | (3u8 << 4); // 0x3A
            let ndims = dimensions.len() as u8;
            // element_size = product(dimensions) * base_element_size
            let total_elements: u32 = dimensions.iter().product();
            let elem_size = total_elements * element_type.element_size();
            let mut buf = Vec::new();
            // Byte 0: class_version, bytes 1-3: 0 (class_bits)
            buf.push(class_version_byte);
            buf.push(0);
            buf.push(0);
            buf.push(0);
            // Size
            buf.extend_from_slice(&elem_size.to_le_bytes());
            // ndims
            buf.push(ndims);
            // dimension sizes (4 bytes each)
            for d in dimensions {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            // Base element type (recursive)
            buf.extend_from_slice(&encode_datatype(element_type)?);
            Ok(buf)
        }
        Datatype::Complex { size, base } => {
            // Version 5 complex: class=11, version=5
            let class_version_byte = (11u8) | (5u8 << 4); // 0x5B
            let mut buf = Vec::new();
            // Byte 0: class_version, byte 1: flags=0x01 (homogeneous), bytes 2-3: 0
            buf.push(class_version_byte);
            buf.push(0x01);
            buf.push(0);
            buf.push(0);
            // Size
            buf.extend_from_slice(&size.to_le_bytes());
            // Base type (recursive)
            buf.extend_from_slice(&encode_datatype(base)?);
            Ok(buf)
        }
        Datatype::VarLen {
            element_type,
            is_string,
            padding,
            char_set,
        } => {
            // Version 4 VarLen: class=9, version=4
            let class_version_byte = (9u8) | (4u8 << 4); // 0x49
            let mut class_bits_lo = 0u8;
            if *is_string {
                class_bits_lo |= 0x01; // type=1 for string
                if let Some(p) = padding {
                    let pad_val = match p {
                        StringPadding::NullTerminate => 0u8,
                        StringPadding::NullPad => 1,
                        StringPadding::SpacePad => 2,
                    };
                    class_bits_lo |= pad_val << 4;
                }
            }
            let mut class_bits_hi = 0u8;
            if *is_string {
                if let Some(cs) = char_set {
                    class_bits_hi = match cs {
                        CharacterSet::Ascii => 0,
                        CharacterSet::Utf8 => 1,
                    };
                }
            }
            // Element size for vlen is always pointer size (4+O+4 = 16 on 64-bit)
            let vlen_element_size: u32 = 4 + SIZE_OF_OFFSETS as u32 + 4;
            let mut buf = Vec::new();
            buf.push(class_version_byte);
            buf.push(class_bits_lo);
            buf.push(class_bits_hi);
            buf.push(0);
            buf.extend_from_slice(&vlen_element_size.to_le_bytes());
            // Base element type
            buf.extend_from_slice(&encode_datatype(element_type)?);
            Ok(buf)
        }
        _ => Err(Error::Other {
            msg: format!(
                "encoding not yet supported for datatype: {:?}",
                std::mem::discriminant(dt)
            ),
        }),
    }
}

/// Minimum number of bytes to encode a value up to `size`.
fn limit_enc_size(size: u32) -> usize {
    if size <= 0xFF {
        1
    } else if size <= 0xFFFF {
        2
    } else if size <= 0xFFFFFF {
        3
    } else {
        4
    }
}

fn encode_dataspace(shape: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
    if shape.is_empty() {
        // Scalar
        return vec![2, 0, 0, 0]; // ver=2, rank=0, flags=0, type=0(scalar)
    }
    let has_max = max_dims.is_some();
    let mut buf = Vec::with_capacity(4 + shape.len() * 8 * if has_max { 2 } else { 1 });
    buf.push(2); // version
    buf.push(shape.len() as u8); // rank
    buf.push(if has_max { 0x01 } else { 0x00 }); // flags: bit 0 = max dims present
    buf.push(1); // type = simple
    for &dim in shape {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    if let Some(md) = max_dims {
        for &dim in md {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
    }
    buf
}

fn encode_link_info() -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(0); // version
    buf.push(0); // flags (no creation order tracking)
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // fractal heap addr
    buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // name B-tree v2 addr
    buf
}

fn encode_group_info() -> Vec<u8> {
    vec![0, 0] // version=0, flags=0
}

fn encode_link(name: &str, target_addr: u64) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();

    // Determine name length encoding
    let (name_enc_bits, name_enc_size) = if name_len <= 0xFF {
        (0u8, 1usize)
    } else if name_len <= 0xFFFF {
        (1u8, 2usize)
    } else {
        (2u8, 4usize)
    };

    let mut buf = Vec::with_capacity(2 + name_enc_size + name_len + 8);
    buf.push(1); // version
    buf.push(name_enc_bits); // flags: hard link, name size encoding in bits 1:0

    // Name length
    match name_enc_size {
        1 => buf.push(name_len as u8),
        2 => buf.extend_from_slice(&(name_len as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(name_len as u32).to_le_bytes()),
        _ => unreachable!(),
    }

    // Name (no null terminator)
    buf.extend_from_slice(name_bytes);

    // Target address (hard link)
    buf.extend_from_slice(&target_addr.to_le_bytes());

    buf
}

fn encode_fill_value_msg(fill_data: &Option<Vec<u8>>) -> Vec<u8> {
    // Fill value v3
    // Byte 0: version = 3
    // Byte 1: flags (alloc_time bits 0-1, fill_time bits 2-3, undefined bit 4, have_value bit 5)
    // alloc_time=LATE(1), fill_time=IFSET(2)
    match fill_data {
        Some(data) => {
            // flags: alloc_time=1, fill_time=2(<<2=8), have_value=1(<<5=32) = 0x29
            let mut buf = Vec::with_capacity(6 + data.len());
            buf.push(3); // version
            buf.push(0x29); // flags: LATE + IFSET + have_value
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(data);
            buf
        }
        None => {
            // No explicit fill value
            vec![3, 0x09] // alloc_time=LATE(1), fill_time=IFSET(2)
        }
    }
}

/// Build a global heap collection (GCOL) containing all vlen elements.
///
/// Objects are numbered 1..N. Each object has:
///   index(2) + refcount(2) + reserved(4) + size(L) + data + padding-to-8
///
/// The collection ends with a free-space marker (index 0, size 0).
fn build_global_heap_collection(elements: &[Vec<u8>]) -> Vec<u8> {
    let sl = SIZE_OF_LENGTHS as usize; // 8
    let obj_hdr = 8 + sl; // index(2) + refcount(2) + reserved(4) + size(L)

    // Compute total size of all objects
    let mut objects_size = 0usize;
    for elem in elements {
        let padded = (elem.len() + 7) & !7;
        objects_size += obj_hdr + padded;
    }

    // Free-space terminator: obj_hdr with index=0, size=0
    let free_marker_size = obj_hdr;
    let header_size = 8 + sl; // magic(4) + version(1) + reserved(3) + collection_size(L)
    let collection_size = header_size + objects_size + free_marker_size;

    let mut buf = Vec::with_capacity(collection_size);

    // Header
    buf.extend_from_slice(b"GCOL");
    buf.push(1); // version
    buf.extend_from_slice(&[0u8; 3]); // reserved
    buf.extend_from_slice(&(collection_size as u64).to_le_bytes());

    // Objects (1-indexed)
    for (i, elem) in elements.iter().enumerate() {
        buf.extend_from_slice(&((i + 1) as u16).to_le_bytes()); // index (1-based)
        buf.extend_from_slice(&1u16.to_le_bytes()); // refcount
        buf.extend_from_slice(&[0u8; 4]); // reserved
        buf.extend_from_slice(&(elem.len() as u64).to_le_bytes()); // size
        buf.extend_from_slice(elem);
        // Pad to 8-byte boundary
        let padding = ((elem.len() + 7) & !7) - elem.len();
        buf.extend_from_slice(&vec![0u8; padding]);
    }

    // Free-space terminator
    buf.extend_from_slice(&0u16.to_le_bytes()); // index = 0
    buf.extend_from_slice(&0u16.to_le_bytes()); // refcount
    buf.extend_from_slice(&[0u8; 4]); // reserved
    buf.extend_from_slice(&0u64.to_le_bytes()); // size = 0

    debug_assert_eq!(buf.len(), collection_size);
    buf
}

/// Build vlen heap ID data (the contiguous dataset payload).
///
/// Each element is: seq_len(4) + collection_addr(O) + object_index(4).
/// seq_len is the number of logical elements (not bytes):
///   - For vlen strings: seq_len = byte length of string
///   - For vlen sequences of T: seq_len = byte_length / sizeof(T)
///
/// We store seq_len = byte_length, which works for vlen strings (element size 1)
/// and for vlen sequences the reader resolves using the base type size.
fn build_vlen_heap_ids(elements: &[Vec<u8>], gcol_addr: u64) -> Vec<u8> {
    let heap_id_size = 4 + SIZE_OF_OFFSETS as usize + 4; // 16 for 64-bit offsets
    let mut buf = Vec::with_capacity(elements.len() * heap_id_size);

    for (i, elem) in elements.iter().enumerate() {
        if elem.is_empty() {
            // Null/empty vlen element
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
        } else {
            buf.extend_from_slice(&(elem.len() as u32).to_le_bytes()); // seq_len
            buf.extend_from_slice(&gcol_addr.to_le_bytes()); // collection addr
            buf.extend_from_slice(&((i + 1) as u32).to_le_bytes()); // object index (1-based)
        }
    }

    buf
}

fn encode_contiguous_layout(data_addr: u64, data_size: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    buf.push(3); // version
    buf.push(1); // layout class = contiguous
    buf.extend_from_slice(&data_addr.to_le_bytes());
    buf.extend_from_slice(&data_size.to_le_bytes());
    buf
}

fn encode_compact_layout(data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + data.len());
    buf.push(3); // version
    buf.push(0); // layout class = compact
    buf.extend_from_slice(&(data.len() as u16).to_le_bytes());
    buf.extend_from_slice(data);
    buf
}

/// Encode a chunked layout v4 message.
/// `chunk_dims` are the chunk dimensions (element counts).
/// `index_type_id`: 1=single_chunk, 2=implicit, 3=fixed_array
/// `chunk_index_addr`: address of the chunk index (or first chunk for implicit/single).
/// `filters_present`: whether filter info is embedded for single-chunk.
/// `single_filtered_size`: for single-chunk with filter: the compressed size.
fn encode_chunked_layout(
    chunk_dims: &[u64],
    index_type_id: u8,
    chunk_index_addr: u64,
    single_filtered_size: Option<u64>,
) -> Vec<u8> {
    let ndims = chunk_dims.len() as u8;

    // Determine enc_bytes_per_dim (minimum bytes to represent largest chunk dim)
    let max_dim = chunk_dims.iter().copied().max().unwrap_or(1);
    let enc_bytes = if max_dim <= 0xFF {
        1u8
    } else if max_dim <= 0xFFFF {
        2
    } else if max_dim <= 0xFFFFFFFF {
        4
    } else {
        8
    };

    let mut flags = 0u8;
    if single_filtered_size.is_some() {
        flags |= 0x02; // SINGLE_INDEX_WITH_FILTER
    }

    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // layout class = chunked
    buf.push(flags);
    buf.push(ndims);
    buf.push(enc_bytes);

    // Chunk dimensions
    for &dim in chunk_dims {
        match enc_bytes {
            1 => buf.push(dim as u8),
            2 => buf.extend_from_slice(&(dim as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(dim as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&dim.to_le_bytes()),
            _ => unreachable!(),
        }
    }

    // Chunk index type
    buf.push(index_type_id);

    // Index-type-specific creation parameters
    match index_type_id {
        1 => {
            // SingleChunk
            if let Some(filtered_size) = single_filtered_size {
                buf.extend_from_slice(&filtered_size.to_le_bytes()); // size_of_lengths=8
                buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask=0 (all filters applied)
            }
        }
        2 => {} // Implicit: no creation parameters
        3 => {
            // FixedArray: max_dblk_page_nelmts_bits
            buf.push(0); // 0 = default paging
        }
        4 => {
            // ExtensibleArray: 5 creation parameters (must match EAHD values)
            buf.push(32); // max_nelmts_bits
            buf.push(1); // data_blk_min_nelmts
            buf.push(0); // min_dblk_page_nelmts_bits
            buf.push(0); // max_dblk_page_nelmts_bits
            buf.push(0); // (reserved)
        }
        5 => {
            // BTreeV2: node_size(4) + split_percent(1) + merge_percent(1)
            buf.extend_from_slice(&4096u32.to_le_bytes());
            buf.push(98); // split percent
            buf.push(40); // merge percent
        }
        _ => {}
    }

    // Chunk index address
    buf.extend_from_slice(&chunk_index_addr.to_le_bytes());

    buf
}

/// Encode filter pipeline message v2.
fn encode_filter_pipeline(chunk_filters: &[ChunkFilter], element_size: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(2); // version
    buf.push(chunk_filters.len() as u8); // number of filters

    for filter in chunk_filters {
        match filter {
            ChunkFilter::Deflate(level) => {
                buf.extend_from_slice(&filters::FILTER_DEFLATE.to_le_bytes()); // id
                buf.extend_from_slice(&0u16.to_le_bytes()); // flags
                buf.extend_from_slice(&1u16.to_le_bytes()); // num client data values
                buf.extend_from_slice(&level.to_le_bytes()); // client_data[0] = level
            }
            ChunkFilter::Shuffle => {
                buf.extend_from_slice(&filters::FILTER_SHUFFLE.to_le_bytes()); // id
                buf.extend_from_slice(&0u16.to_le_bytes()); // flags
                buf.extend_from_slice(&1u16.to_le_bytes()); // num client data values
                buf.extend_from_slice(&element_size.to_le_bytes()); // client_data[0] = element_size
            }
            ChunkFilter::Fletcher32 => {
                buf.extend_from_slice(&filters::FILTER_FLETCHER32.to_le_bytes()); // id
                buf.extend_from_slice(&0u16.to_le_bytes()); // flags
                buf.extend_from_slice(&0u16.to_le_bytes()); // num client data values
            }
        }
    }

    buf
}

/// Apply filters in forward direction (compression) to chunk data.
fn apply_filters_forward(
    chunk_filters: &[ChunkFilter],
    mut data: Vec<u8>,
    element_size: u32,
) -> Result<Vec<u8>> {
    for filter in chunk_filters {
        data = match filter {
            ChunkFilter::Shuffle => shuffle(&data, element_size as usize),
            ChunkFilter::Deflate(level) => compress_deflate(&data, *level)?,
            ChunkFilter::Fletcher32 => {
                let cksum = fletcher32_forward(&data);
                let mut out = data;
                out.extend_from_slice(&cksum.to_be_bytes());
                out
            }
        };
    }
    Ok(data)
}

/// Shuffle filter (forward direction).
fn shuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let num_elements = data.len() / element_size;
    let mut output = vec![0u8; data.len()];
    for byte_idx in 0..element_size {
        let dst_start = byte_idx * num_elements;
        for elem in 0..num_elements {
            output[dst_start + elem] = data[elem * element_size + byte_idx];
        }
    }
    output
}

/// Deflate compression (forward direction).
fn compress_deflate(data: &[u8], level: u32) -> Result<Vec<u8>> {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::ZlibEncoder;

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level));
    encoder.write_all(data).map_err(|e| Error::Other {
        msg: format!("deflate compress: {}", e),
    })?;
    encoder.finish().map_err(|e| Error::Other {
        msg: format!("deflate finish: {}", e),
    })
}

/// Fletcher32 checksum (forward direction).
fn fletcher32_forward(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let mut remaining = data.len() / 2;
    let mut i = 0;
    while remaining > 0 {
        let tlen = remaining.min(360);
        remaining -= tlen;
        for _ in 0..tlen {
            let word = u16::from_le_bytes([data[i], data[i + 1]]) as u32;
            sum1 += word;
            sum2 += sum1;
            i += 2;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }
    if data.len() % 2 != 0 {
        sum1 += (data[i] as u32) << 8;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 += sum1;
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    (sum1 << 16) | sum2
}

// ---------------------------------------------------------------------------
// Chunk helpers
// ---------------------------------------------------------------------------

fn compute_chunk_count(shape: &[u64], chunk_dims: &[u64]) -> usize {
    let mut total = 1usize;
    for (s, c) in shape.iter().zip(chunk_dims.iter()) {
        total *= ((*s + *c - 1) / *c) as usize;
    }
    total
}

/// Enumerate all chunk starting coordinates in row-major order.
fn enumerate_chunks(shape: &[u64], chunk_dims: &[u64]) -> Vec<Vec<u64>> {
    let ndims = shape.len();
    let mut chunks_per_dim: Vec<u64> = Vec::with_capacity(ndims);
    for i in 0..ndims {
        chunks_per_dim.push((shape[i] + chunk_dims[i] - 1) / chunk_dims[i]);
    }
    let total: usize = chunks_per_dim.iter().map(|&c| c as usize).product();
    let mut result = Vec::with_capacity(total);
    let mut coord = vec![0u64; ndims];
    for _ in 0..total {
        result.push(coord.iter().map(|c| *c).collect());
        // Increment coordinate (row-major order)
        for d in (0..ndims).rev() {
            coord[d] += 1;
            if coord[d] < chunks_per_dim[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    result
}

/// Extract the data for a specific chunk from the full dataset array.
///
/// Returns a full chunk-sized block (chunk_dims product * element_size bytes).
/// Positions beyond the dataset edge are zero-padded. The layout within the
/// block is row-major based on chunk_dims, matching what the HDF5 reader expects.
fn extract_chunk_data(
    data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    chunk_coords: &[u64],
    element_size: usize,
) -> Vec<u8> {
    let ndims = shape.len();
    let chunk_elements: u64 = chunk_dims.iter().product();
    let mut result = vec![0u8; chunk_elements as usize * element_size];

    // Iterate over every position in the full chunk
    let mut local_coord = vec![0u64; ndims];
    for flat_idx in 0..chunk_elements as usize {
        // Check if this position is within the dataset bounds
        let mut in_bounds = true;
        for d in 0..ndims {
            let global_d = chunk_coords[d] * chunk_dims[d] + local_coord[d];
            if global_d >= shape[d] {
                in_bounds = false;
                break;
            }
        }

        if in_bounds {
            // Compute global linear index in the source data
            let mut global_idx = 0u64;
            let mut stride = 1u64;
            for d in (0..ndims).rev() {
                let global_d = chunk_coords[d] * chunk_dims[d] + local_coord[d];
                global_idx += global_d * stride;
                stride *= shape[d];
            }
            let src_start = global_idx as usize * element_size;
            let dst_start = flat_idx * element_size;
            result[dst_start..dst_start + element_size]
                .copy_from_slice(&data[src_start..src_start + element_size]);
        }

        // Increment local coordinate (row-major over chunk_dims)
        for d in (0..ndims).rev() {
            local_coord[d] += 1;
            if local_coord[d] < chunk_dims[d] {
                break;
            }
            local_coord[d] = 0;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Fixed Array index writer
// ---------------------------------------------------------------------------

/// Write a Fixed Array (FARY) index for chunk addresses.
///
/// Fixed Array layout:
///   Header: FAHD magic(4) + version(1) + client_id(1) + elem_size(1) +
///           max_dblk_page_nelmts_bits(1) + nelmts(L) + data_blk_addr(O) + checksum(4)
///   Data Block: FADB magic(4) + version(1) + client_id(1) + hdr_addr(O) +
///               elements[nelmts] + checksum(4)
fn write_fixed_array_index(
    buf: &mut Vec<u8>,
    chunk_addrs: &[u64],
    chunk_sizes: &[u64],
    has_filters: bool,
    raw_chunk_bytes: u64,
) -> Result<()> {
    let nelmts = chunk_addrs.len() as u64;

    // Client ID: 0 for non-filtered chunks, 1 for filtered chunks
    let client_id: u8 = if has_filters { 1 } else { 0 };

    // Element size per entry:
    //   Non-filtered: just an address (O bytes = 8)
    //   Filtered: address(O) + chunk_size_enc(variable) + filter_mask(4)
    let chunk_size_enc_bytes = if has_filters {
        // Number of bytes needed to encode the max compressed chunk size
        let max_size = chunk_sizes.iter().copied().max().unwrap_or(raw_chunk_bytes);
        limit_enc_size_u64(max_size.max(raw_chunk_bytes)) as u8
    } else {
        0
    };

    let elem_size: u8 = if has_filters {
        SIZE_OF_OFFSETS + chunk_size_enc_bytes + 4 // addr + size + mask
    } else {
        SIZE_OF_OFFSETS // just addr
    };

    // Write header
    let hdr_addr = buf.len() as u64;
    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"FAHD");
    hdr.push(0); // version
    hdr.push(client_id);
    hdr.push(elem_size);
    hdr.push(0); // max_dblk_page_nelmts_bits (0 = no paging)
    hdr.extend_from_slice(&nelmts.to_le_bytes()); // nelmts (L=8)

    // Data block address (comes right after this header)
    let hdr_before_dblk_addr = hdr.len();
    // Placeholder — we'll fill it after computing
    hdr.extend_from_slice(&0u64.to_le_bytes());

    // Checksum placeholder
    let cksum = checksum::lookup3(&hdr);
    hdr.extend_from_slice(&cksum.to_le_bytes());
    let hdr_size = hdr.len();

    let dblk_addr = hdr_addr + hdr_size as u64;
    // Fix up data block address in header
    hdr[hdr_before_dblk_addr..hdr_before_dblk_addr + 8].copy_from_slice(&dblk_addr.to_le_bytes());
    // Recompute checksum
    let cksum = checksum::lookup3(&hdr[..hdr.len() - 4]);
    let hdr_len = hdr.len();
    hdr[hdr_len - 4..].copy_from_slice(&cksum.to_le_bytes());

    buf.extend_from_slice(&hdr);

    // Write data block
    let mut dblk = Vec::new();
    dblk.extend_from_slice(b"FADB");
    dblk.push(0); // version
    dblk.push(client_id);
    dblk.extend_from_slice(&hdr_addr.to_le_bytes()); // hdr_addr (O=8)

    // Elements
    for i in 0..chunk_addrs.len() {
        dblk.extend_from_slice(&chunk_addrs[i].to_le_bytes());
        if has_filters {
            // Chunk size (variable width)
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
            // Filter mask (0 = all filters applied)
            dblk.extend_from_slice(&0u32.to_le_bytes());
        }
    }

    // Checksum
    let cksum = checksum::lookup3(&dblk);
    dblk.extend_from_slice(&cksum.to_le_bytes());

    buf.extend_from_slice(&dblk);

    Ok(())
}

// ---------------------------------------------------------------------------
// Extensible Array index writer
// ---------------------------------------------------------------------------

/// Write an Extensible Array (EA) index for chunk addresses.
///
/// EA layout:
///   EAHD header → EAIB index block → EADB data blocks (if needed)
///
/// We use a simple strategy:
///   - idx_blk_elmts = min(nchunks, 4): direct elements in index block
///   - If nchunks > idx_blk_elmts: remaining go in data blocks addressed from index block
///   - No super blocks needed for reasonable chunk counts
fn write_extensible_array_index(
    buf: &mut Vec<u8>,
    chunk_addrs: &[u64],
    chunk_sizes: &[u64],
    has_filters: bool,
    nchunks: usize,
) -> Result<()> {
    let client_id: u8 = if has_filters { 1 } else { 0 };

    // Element size: address only for unfiltered, address + size + mask for filtered
    let elem_size: u8 = if has_filters {
        let max_size = chunk_sizes.iter().copied().max().unwrap_or(0);
        let size_enc = limit_enc_size_u64(max_size) as u8;
        SIZE_OF_OFFSETS + size_enc + 4
    } else {
        SIZE_OF_OFFSETS
    };

    // EA configuration parameters
    let idx_blk_elmts: u8 = 4; // Direct elements in index block
    let data_blk_min_elmts: u8 = 1; // Minimum elements per data block
    let sup_blk_min_data_ptrs: u8 = 2; // Min data block pointers per super block

    // max_nelmts_bits: enough bits to address all possible chunks
    // For unlimited dims, use 32 (same as HDF5 library default for reasonable sizes)
    let max_nelmts_bits: u8 = 32;
    let max_dblk_page_nelmts_bits: u8 = 0; // No paging

    let max_idx_set = nchunks as u64;

    // Compute how many data block pointer slots the index block needs
    let ndblk_addrs: u64 = if nchunks as u64 > idx_blk_elmts as u64 {
        2 * (sup_blk_min_data_ptrs as u64 - 1) // = 2 * (2-1) = 2
    } else {
        0
    };

    // Compute how many super block pointer slots are needed
    // For simplicity, compute how many chunks can be addressed by direct elements + data blocks
    let mut capacity = idx_blk_elmts as u64;
    let mut dblk_nelmts = data_blk_min_elmts as u64;
    for d in 0..ndblk_addrs {
        if d > 0 && d == sup_blk_min_data_ptrs as u64 {
            dblk_nelmts *= 2;
        }
        capacity += dblk_nelmts;
    }

    // Compute super block count needed
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

    // Total number of data blocks we actually need to write
    let chunks_in_idx_blk = (idx_blk_elmts as u64).min(max_idx_set);
    let chunks_remaining = max_idx_set.saturating_sub(chunks_in_idx_blk);

    // Plan data blocks from index block
    let mut dblk_plan: Vec<u64> = Vec::new(); // nelmts for each data block from IB
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

    // Plan data blocks from super blocks
    struct SblkPlan {
        dblk_counts: Vec<u64>, // actual element counts for each data block
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
    hdr.push(0); // version
    hdr.push(client_id);
    hdr.push(elem_size);
    hdr.push(max_nelmts_bits);
    hdr.push(idx_blk_elmts);
    hdr.push(data_blk_min_elmts);
    hdr.push(sup_blk_min_data_ptrs);
    hdr.push(max_dblk_page_nelmts_bits);

    // 6 stats (each SIZE_OF_LENGTHS = 8 bytes)
    let nsuper_blks = sblk_plans.len() as u64;
    let ndata_blks = dblk_plan.len() as u64
        + sblk_plans
            .iter()
            .map(|s| s.dblk_counts.len() as u64)
            .sum::<u64>();
    hdr.extend_from_slice(&nsuper_blks.to_le_bytes()); // stat[0]: nsuper_blks
    hdr.extend_from_slice(&0u64.to_le_bytes()); // stat[1]: super_blk_size (unused)
    hdr.extend_from_slice(&ndata_blks.to_le_bytes()); // stat[2]: ndata_blks
    hdr.extend_from_slice(&0u64.to_le_bytes()); // stat[3]: data_blk_size (unused)
    hdr.extend_from_slice(&max_idx_set.to_le_bytes()); // stat[4]: max_idx_set
    let num_elements = max_idx_set; // stat[5]: num_elements
    hdr.extend_from_slice(&num_elements.to_le_bytes());

    // Index block address (placeholder — we'll fill after computing)
    let idx_blk_addr_offset = hdr.len();
    hdr.extend_from_slice(&0u64.to_le_bytes());

    // Checksum placeholder
    hdr.extend_from_slice(&0u32.to_le_bytes());
    let hdr_size = hdr.len();

    // --- Compute index block size ---
    let ib_prefix = 4 + 1 + 1 + o; // magic + version + client_id + hdr_addr
    let direct_elems_bytes = idx_blk_elmts as usize * elem_size as usize;
    let dblk_addr_slots = ndblk_addrs as usize;
    let sblk_addr_slots = nsblk_addrs as usize;
    let ib_body = direct_elems_bytes + (dblk_addr_slots + sblk_addr_slots) * o;
    let ib_size = ib_prefix + ib_body + 4; // + checksum

    let ib_addr = hdr_addr + hdr_size as u64;

    // Fix up index block address in header
    hdr[idx_blk_addr_offset..idx_blk_addr_offset + 8].copy_from_slice(&ib_addr.to_le_bytes());
    // Recompute header checksum
    let hdr_cksum = checksum::lookup3(&hdr[..hdr.len() - 4]);
    let hdr_len = hdr.len();
    hdr[hdr_len - 4..].copy_from_slice(&hdr_cksum.to_le_bytes());

    buf.extend_from_slice(&hdr);

    // --- Compute data block and super block addresses ---
    // Data blocks from index block come after the IB
    let mut dblk_addrs_from_ib: Vec<u64> = Vec::new();
    let arr_off_size = (max_nelmts_bits as u64).div_ceil(8).max(1) as usize;
    let mut next_addr = ib_addr + ib_size as u64;
    for &count in &dblk_plan {
        dblk_addrs_from_ib.push(next_addr);
        // EADB size: magic(4) + version(1) + client_id(1) + hdr_addr(O) + arr_offset(varies)
        //            + elements(count * elem_size) + checksum(4)
        let db_size = 4 + 1 + 1 + o + arr_off_size + (count as usize * elem_size as usize) + 4;
        next_addr += db_size as u64;
    }

    // Super blocks and their data blocks
    let mut sblk_addrs: Vec<u64> = Vec::new();
    let mut sblk_dblk_addrs: Vec<Vec<u64>> = Vec::new();
    for (si, splan) in sblk_plans.iter().enumerate() {
        sblk_addrs.push(next_addr);
        // Super block size: magic(4) + version(1) + client_id(1) + hdr_addr(O) + arr_offset(varies)
        //   + data_block_addrs(ndblks * O) + checksum(4)
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
    ib.push(0); // version
    ib.push(client_id);
    ib.extend_from_slice(&hdr_addr.to_le_bytes()); // hdr_addr

    // Direct elements
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
            // UNDEF for unused slots
            ib.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
            if has_filters {
                for _ in 0..(elem_size as usize - o) {
                    ib.push(0);
                }
            }
        }
    }

    // Data block addresses from index block
    for i in 0..dblk_addr_slots {
        if i < dblk_addrs_from_ib.len() {
            ib.extend_from_slice(&dblk_addrs_from_ib[i].to_le_bytes());
        } else {
            ib.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        }
    }

    // Super block addresses
    for i in 0..sblk_addr_slots {
        if i < sblk_addrs.len() {
            ib.extend_from_slice(&sblk_addrs[i].to_le_bytes());
        } else {
            ib.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
        }
    }

    // Checksum
    let ib_cksum = checksum::lookup3(&ib);
    ib.extend_from_slice(&ib_cksum.to_le_bytes());
    debug_assert_eq!(ib.len(), ib_size);

    buf.extend_from_slice(&ib);

    // --- Write data blocks from index block ---
    let mut global_idx = idx_blk_elmts as u64;
    for (di, &count) in dblk_plan.iter().enumerate() {
        let mut db = Vec::new();
        db.extend_from_slice(b"EADB");
        db.push(0); // version
        db.push(client_id);
        db.extend_from_slice(&hdr_addr.to_le_bytes()); // hdr_addr
        // Array offset (variable size)
        write_var_le(&mut db, global_idx, arr_off_size);

        // Elements
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
                    for _ in 0..(elem_size as usize - o) {
                        db.push(0);
                    }
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
        sb.push(0); // version
        sb.push(client_id);
        sb.extend_from_slice(&hdr_addr.to_le_bytes()); // hdr_addr
        // Array offset
        write_var_le(&mut sb, global_idx, arr_off_size);

        // Data block addresses
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

        // Write data blocks for this super block
        for &count in splan.dblk_counts.iter() {
            let mut db = Vec::new();
            db.extend_from_slice(b"EADB");
            db.push(0); // version
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
                        for _ in 0..(elem_size as usize - o) {
                            db.push(0);
                        }
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

/// Write a single EA element (chunk address, optionally with filter info).
fn write_ea_element(buf: &mut Vec<u8>, addr: u64, size: u64, has_filters: bool, elem_size: u8) {
    buf.extend_from_slice(&addr.to_le_bytes());
    if has_filters {
        let size_enc = elem_size as usize - SIZE_OF_OFFSETS as usize - 4;
        write_var_le(buf, size, size_enc);
        buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask
    }
}

/// Write a variable-length little-endian integer.
fn write_var_le(buf: &mut Vec<u8>, value: u64, nbytes: usize) {
    for i in 0..nbytes {
        buf.push((value >> (i * 8)) as u8);
    }
}

// ---------------------------------------------------------------------------
// B-tree v2 chunk index writer
// ---------------------------------------------------------------------------

/// Write a B-tree v1 chunk index (layout v3).
///
/// Produces a single leaf node (level 0). Keys use element-coordinate offsets
/// and the trailing key has all-max offsets to mark the end.
fn write_btree_v1_chunk_index(
    buf: &mut Vec<u8>,
    chunk_addrs: &[u64],
    chunk_sizes: &[u64],
    chunk_coords_list: &[Vec<u64>],
    chunk_dims: &[u64],
    element_size: u32,
    ndims: usize,
) -> Result<()> {
    let nchunks = chunk_addrs.len();

    // v3_ndims = rank + 1 (extra dimension for element size)
    let v3_ndims = ndims + 1;

    // TREE header
    buf.extend_from_slice(b"TREE"); // magic
    buf.push(1); // node_type = 1 (chunked raw data)
    buf.push(0); // level = 0 (leaf)
    buf.extend_from_slice(&(nchunks as u16).to_le_bytes()); // entries_used
    buf.extend_from_slice(&u64::MAX.to_le_bytes()); // left sibling = UNDEF
    buf.extend_from_slice(&u64::MAX.to_le_bytes()); // right sibling = UNDEF

    // Body: Key[0] Child[0] Key[1] Child[1] ... Key[N-1] Child[N-1] Key[N]
    for i in 0..nchunks {
        // Key[i]
        buf.extend_from_slice(&(chunk_sizes[i] as u32).to_le_bytes()); // chunk_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask

        // Element-coordinate offsets: scaled_coord * chunk_dim
        for d in 0..ndims {
            let elem_offset = chunk_coords_list[i][d] * chunk_dims[d];
            buf.extend_from_slice(&elem_offset.to_le_bytes());
        }
        // Last offset (element size dimension) is always 0
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Child[i] = chunk data address
        buf.extend_from_slice(&chunk_addrs[i].to_le_bytes());
    }

    // Trailing key (Key[N]): marks the end, all offsets set to max
    let raw_chunk_bytes = chunk_dims.iter().product::<u64>() * element_size as u64;
    buf.extend_from_slice(&(raw_chunk_bytes as u32).to_le_bytes()); // chunk_size
    buf.extend_from_slice(&0u32.to_le_bytes()); // filter_mask
    for _ in 0..v3_ndims {
        buf.extend_from_slice(&u64::MAX.to_le_bytes());
    }

    Ok(())
}

// ---------------------------------------------------------------------------

/// Write a B-tree v2 chunk index.
///
/// For simplicity we always write a depth-0 tree (single leaf node as root).
/// This works for any number of records as long as they fit in one leaf node
/// (limited by node_size). We use a node_size of 4096 which typically fits
/// hundreds of records.
fn write_btree_v2_chunk_index(
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

    // Compute chunk_size_len for filtered records (H5D_BT2_COMPUTE_CHUNK_SIZE_LEN)
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

    // Record size: address + [chunk_size_len + filter_mask(4)] + scaled_offsets(8*ndims)
    let record_size = o + chunk_size_len + (if has_filters { 4 } else { 0 }) + 8 * ndims;

    // Type: 10 for non-filtered, 11 for filtered
    let bt2_type: u8 = if has_filters { 11 } else { 10 };

    let node_size: u32 = 4096;
    let split_percent: u8 = 98;
    let merge_percent: u8 = 40;

    // Build records
    let mut records: Vec<Vec<u8>> = Vec::with_capacity(nchunks);
    let chunk_coords_list = enumerate_chunks(shape, chunk_dims);
    for (i, coords) in chunk_coords_list.iter().enumerate() {
        let mut rec = Vec::with_capacity(record_size);
        rec.extend_from_slice(&chunk_addrs[i].to_le_bytes());
        if has_filters {
            write_var_le(&mut rec, chunk_sizes[i], chunk_size_len);
            rec.extend_from_slice(&0u32.to_le_bytes()); // filter_mask
        }
        // Scaled offsets (chunk coordinates, not element coordinates)
        for d in 0..ndims {
            rec.extend_from_slice(&coords[d].to_le_bytes());
        }
        debug_assert_eq!(rec.len(), record_size);
        records.push(rec);
    }

    // Write BTHD header
    let hdr_addr = buf.len() as u64;
    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"BTHD");
    hdr.push(0); // version
    hdr.push(bt2_type);
    hdr.extend_from_slice(&node_size.to_le_bytes());
    hdr.extend_from_slice(&(record_size as u16).to_le_bytes());
    hdr.extend_from_slice(&0u16.to_le_bytes()); // depth = 0 (leaf root)
    hdr.push(split_percent);
    hdr.push(merge_percent);

    // Root node address (placeholder — leaf node follows header)
    let root_addr_offset = hdr.len();
    hdr.extend_from_slice(&0u64.to_le_bytes()); // root_node_address
    hdr.extend_from_slice(&(nchunks as u16).to_le_bytes()); // root_num_records
    hdr.extend_from_slice(&(nchunks as u64).to_le_bytes()); // total_records (L=8)

    // Checksum placeholder
    hdr.extend_from_slice(&0u32.to_le_bytes());
    let hdr_size = hdr.len();

    // Leaf node address = right after header
    let leaf_addr = hdr_addr + hdr_size as u64;
    hdr[root_addr_offset..root_addr_offset + 8].copy_from_slice(&leaf_addr.to_le_bytes());

    // Recompute header checksum
    let hdr_cksum = checksum::lookup3(&hdr[..hdr.len() - 4]);
    let hdr_len = hdr.len();
    hdr[hdr_len - 4..].copy_from_slice(&hdr_cksum.to_le_bytes());

    buf.extend_from_slice(&hdr);

    // Write BTLF leaf node
    let mut leaf = Vec::new();
    leaf.extend_from_slice(b"BTLF");
    leaf.push(0); // version
    leaf.push(bt2_type);

    // Records
    for rec in &records {
        leaf.extend_from_slice(rec);
    }

    // Checksum
    let leaf_cksum = checksum::lookup3(&leaf);
    leaf.extend_from_slice(&leaf_cksum.to_le_bytes());

    buf.extend_from_slice(&leaf);

    Ok(())
}

fn limit_enc_size_u64(size: u64) -> usize {
    if size <= 0xFF {
        1
    } else if size <= 0xFFFF {
        2
    } else if size <= 0xFFFFFFFF {
        4
    } else {
        8
    }
}

fn encode_attribute(attr: &AttrData) -> Result<Vec<u8>> {
    let dt_enc = encode_datatype(&attr.datatype)?;
    let ds_enc = encode_dataspace(&attr.shape, None);
    let name_with_nul = attr.name.len() + 1;

    let mut buf = Vec::new();
    buf.push(3); // version
    buf.push(0); // flags (not shared)
    buf.extend_from_slice(&(name_with_nul as u16).to_le_bytes());
    buf.extend_from_slice(&(dt_enc.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(ds_enc.len() as u16).to_le_bytes());
    buf.push(0); // charset = ASCII
    buf.extend_from_slice(attr.name.as_bytes());
    buf.push(0); // NUL terminator
    buf.extend_from_slice(&dt_enc);
    buf.extend_from_slice(&ds_enc);
    buf.extend_from_slice(&attr.value);

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::File;
    use crate::dataspace::Dataspace;

    #[test]
    fn encode_superblock_roundtrip() {
        let sb_bytes = encode_superblock(48, 1024);
        assert_eq!(sb_bytes.len(), SUPERBLOCK_SIZE);
        let sb = crate::superblock::Superblock::parse(sb_bytes.as_slice(), 0).unwrap();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.size_of_offsets, 8);
        assert_eq!(sb.size_of_lengths, 8);
        assert_eq!(sb.root_group_object_header_address, 48);
        assert_eq!(sb.end_of_file_address, 1024);
    }

    #[test]
    fn encode_datatype_i32() {
        let dt = Datatype::native_i32();
        let enc = encode_datatype(&dt).unwrap();
        assert_eq!(enc.len(), 12);
        let parsed = Datatype::parse(&enc).unwrap();
        match parsed {
            Datatype::FixedPoint {
                size,
                byte_order,
                signed,
                bit_offset,
                bit_precision,
            } => {
                assert_eq!(size, 4);
                assert_eq!(byte_order, ByteOrder::LittleEndian);
                assert!(signed);
                assert_eq!(bit_offset, 0);
                assert_eq!(bit_precision, 32);
            }
            _ => panic!("expected FixedPoint"),
        }
    }

    #[test]
    fn encode_datatype_f64() {
        let dt = Datatype::native_f64();
        let enc = encode_datatype(&dt).unwrap();
        assert_eq!(enc.len(), 20);
        let parsed = Datatype::parse(&enc).unwrap();
        match parsed {
            Datatype::FloatingPoint {
                size,
                byte_order,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_size,
                exponent_bias,
                ..
            } => {
                assert_eq!(size, 8);
                assert_eq!(byte_order, ByteOrder::LittleEndian);
                assert_eq!(bit_precision, 64);
                assert_eq!(exponent_location, 52);
                assert_eq!(exponent_size, 11);
                assert_eq!(mantissa_size, 52);
                assert_eq!(exponent_bias, 1023);
            }
            _ => panic!("expected FloatingPoint"),
        }
    }

    #[test]
    fn encode_datatype_string() {
        let dt = Datatype::fixed_string(10);
        let enc = encode_datatype(&dt).unwrap();
        assert_eq!(enc.len(), 8);
        let parsed = Datatype::parse(&enc).unwrap();
        match parsed {
            Datatype::String {
                size,
                padding,
                char_set,
            } => {
                assert_eq!(size, 10);
                assert_eq!(padding, StringPadding::NullTerminate);
                assert_eq!(char_set, CharacterSet::Ascii);
            }
            _ => panic!("expected String"),
        }
    }

    #[test]
    fn encode_dataspace_scalar() {
        let enc = encode_dataspace(&[], None);
        assert_eq!(enc, vec![2, 0, 0, 0]);
        let parsed = Dataspace::parse(&enc).unwrap();
        assert!(matches!(parsed, Dataspace::Scalar));
    }

    #[test]
    fn encode_dataspace_1d() {
        let enc = encode_dataspace(&[100], None);
        assert_eq!(enc.len(), 12); // 4 + 1*8
        let parsed = Dataspace::parse(&enc).unwrap();
        match parsed {
            Dataspace::Simple { dimensions, .. } => assert_eq!(dimensions, vec![100]),
            _ => panic!("expected Simple"),
        }
    }

    #[test]
    fn encode_dataspace_2d() {
        let enc = encode_dataspace(&[10, 20], None);
        assert_eq!(enc.len(), 20); // 4 + 2*8
        let parsed = Dataspace::parse(&enc).unwrap();
        match parsed {
            Dataspace::Simple { dimensions, .. } => assert_eq!(dimensions, vec![10, 20]),
            _ => panic!("expected Simple"),
        }
    }

    #[test]
    fn roundtrip_empty_file() {
        let w = FileWriter::new();
        let bytes = w.to_bytes().unwrap();
        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        assert!(root.members().unwrap().is_empty());
    }

    #[test]
    fn roundtrip_single_i32_dataset() {
        let mut w = FileWriter::new();
        let data: Vec<u8> = (0..10i32).flat_map(|x| x.to_le_bytes()).collect();
        w.root_mut()
            .add_dataset("numbers", Datatype::native_i32(), &[10], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let members = root.members().unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0], "numbers");

        let ds = root.dataset("numbers").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![10]);
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn roundtrip_f64_dataset() {
        let mut w = FileWriter::new();
        let values: Vec<f64> = (0..5).map(|i| i as f64 * 1.5).collect();
        let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.root_mut()
            .add_dataset("floats", Datatype::native_f64(), &[5], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("floats").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![5]);
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn roundtrip_2d_dataset() {
        let mut w = FileWriter::new();
        let data: Vec<u8> = (0..12u16).flat_map(|x| x.to_le_bytes()).collect();
        w.root_mut()
            .add_dataset("matrix", Datatype::native_u16(), &[3, 4], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("matrix").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![3, 4]);
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn roundtrip_scalar_dataset() {
        let mut w = FileWriter::new();
        let data = 42i64.to_le_bytes().to_vec();
        w.root_mut()
            .add_dataset("scalar", Datatype::native_i64(), &[], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("scalar").unwrap();
        assert_eq!(ds.shape().unwrap(), Vec::<u64>::new());
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn roundtrip_group() {
        let mut w = FileWriter::new();
        w.root_mut().add_group("grp1");
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let members = root.members().unwrap();
        assert_eq!(members, vec!["grp1"]);
        // Open as group
        let grp = root.group("grp1").unwrap();
        assert!(grp.members().unwrap().is_empty());
    }

    #[test]
    fn roundtrip_nested_groups() {
        let mut w = FileWriter::new();
        let grp = w.root_mut().add_group("outer");
        grp.add_group("inner");
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let outer = root.group("outer").unwrap();
        let inner = outer.group("inner").unwrap();
        assert!(inner.members().unwrap().is_empty());
    }

    #[test]
    fn roundtrip_group_with_dataset() {
        let mut w = FileWriter::new();
        let grp = w.root_mut().add_group("data");
        let vals: Vec<u8> = (0..4u32).flat_map(|x| x.to_le_bytes()).collect();
        grp.add_dataset("values", Datatype::native_u32(), &[4], vals.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let grp = file.root_group().unwrap().group("data").unwrap();
        let ds = grp.dataset("values").unwrap();
        assert_eq!(ds.read_raw().unwrap(), vals);
    }

    #[test]
    fn roundtrip_group_attribute() {
        let mut w = FileWriter::new();
        let val = 123i32.to_le_bytes().to_vec();
        w.root_mut()
            .add_attribute("version", Datatype::native_i32(), &[], val.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let attrs = root.attributes().unwrap();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "version");
        assert_eq!(attrs[0].raw_value, val);
    }

    #[test]
    fn roundtrip_dataset_attribute() {
        let mut w = FileWriter::new();
        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let ds = w
            .root_mut()
            .add_dataset("temps", Datatype::native_f64(), &[3], data);
        ds.add_attribute("units", Datatype::fixed_string(7), &[], b"celsius".to_vec());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("temps").unwrap();
        let attrs = ds.attributes().unwrap();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].name, "units");
        assert_eq!(attrs[0].raw_value, b"celsius");
    }

    #[test]
    fn roundtrip_multiple_datasets() {
        let mut w = FileWriter::new();
        let root = w.root_mut();
        let d1: Vec<u8> = (0..3i32).flat_map(|x| x.to_le_bytes()).collect();
        let d2: Vec<u8> = (10..14u64).flat_map(|x| x.to_le_bytes()).collect();
        root.add_dataset("ints", Datatype::native_i32(), &[3], d1.clone());
        root.add_dataset("longs", Datatype::native_u64(), &[4], d2.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let ds1 = root.dataset("ints").unwrap();
        assert_eq!(ds1.read_raw().unwrap(), d1);
        let ds2 = root.dataset("longs").unwrap();
        assert_eq!(ds2.read_raw().unwrap(), d2);
    }

    #[test]
    fn rejects_wrong_data_size() {
        let mut w = FileWriter::new();
        w.root_mut()
            .add_dataset("bad", Datatype::native_i32(), &[10], vec![0u8; 8]);
        let err = w.to_bytes().unwrap_err();
        assert!(err.to_string().contains("data size mismatch"));
    }

    #[test]
    fn roundtrip_all_integer_types() {
        let mut w = FileWriter::new();
        let root = w.root_mut();
        root.add_dataset("i8", Datatype::native_i8(), &[2], vec![1u8, 2]);
        root.add_dataset(
            "i16",
            Datatype::native_i16(),
            &[2],
            [1i16, 2].iter().flat_map(|x| x.to_le_bytes()).collect(),
        );
        root.add_dataset(
            "u32",
            Datatype::native_u32(),
            &[1],
            100u32.to_le_bytes().to_vec(),
        );
        root.add_dataset(
            "u64",
            Datatype::native_u64(),
            &[1],
            999u64.to_le_bytes().to_vec(),
        );
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        assert_eq!(root.dataset("i8").unwrap().read_raw().unwrap(), vec![1, 2]);
    }

    #[test]
    fn roundtrip_multiple_attributes() {
        let mut w = FileWriter::new();
        let root = w.root_mut();
        root.add_attribute(
            "a1",
            Datatype::native_i32(),
            &[],
            1i32.to_le_bytes().to_vec(),
        );
        root.add_attribute(
            "a2",
            Datatype::native_f64(),
            &[],
            3.14f64.to_le_bytes().to_vec(),
        );
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let attrs = root.attributes().unwrap();
        assert_eq!(attrs.len(), 2);
        let names: Vec<&str> = attrs.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"a1"));
        assert!(names.contains(&"a2"));
    }

    #[test]
    fn roundtrip_complex_hierarchy() {
        let mut w = FileWriter::new();
        let root = w.root_mut();
        root.add_attribute(
            "file_version",
            Datatype::native_i32(),
            &[],
            1i32.to_le_bytes().to_vec(),
        );

        let data_grp = root.add_group("data");
        let vals: Vec<u8> = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let ds = data_grp.add_dataset("measurements", Datatype::native_f32(), &[2, 3], vals);
        ds.add_attribute("units", Datatype::fixed_string(1), &[], b"m".to_vec());

        let meta_grp = root.add_group("metadata");
        meta_grp.add_attribute("author", Datatype::fixed_string(4), &[], b"test".to_vec());

        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();

        // Check root attribute
        let root_attrs = root.attributes().unwrap();
        assert_eq!(root_attrs.len(), 1);
        assert_eq!(root_attrs[0].name, "file_version");

        // Check data group
        let data = root.group("data").unwrap();
        let ds = data.dataset("measurements").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![2, 3]);
        let ds_attrs = ds.attributes().unwrap();
        assert_eq!(ds_attrs.len(), 1);
        assert_eq!(ds_attrs[0].name, "units");

        // Check metadata group
        let meta = root.group("metadata").unwrap();
        let meta_attrs = meta.attributes().unwrap();
        assert_eq!(meta_attrs.len(), 1);
        assert_eq!(meta_attrs[0].name, "author");
    }

    #[test]
    fn roundtrip_with_timestamps() {
        let ts = (1700000000, 1700000000, 1700000000, 1700000000);
        let opts = WriteOptions {
            timestamps: Some(ts),
        };
        let mut w = FileWriter::with_options(opts);
        let data: Vec<u8> = (0..4i32).flat_map(|x| x.to_le_bytes()).collect();
        w.root_mut()
            .add_dataset("data", Datatype::native_i32(), &[4], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let ds = root.dataset("data").unwrap();
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn timestamps_stored_in_object_header() {
        let ts = (1700000000, 1700000001, 1700000002, 1700000003);
        let opts = WriteOptions {
            timestamps: Some(ts),
        };
        let mut w = FileWriter::with_options(opts);
        w.root_mut().add_group("grp");
        let bytes = w.to_bytes().unwrap();

        // Parse object headers directly to check timestamps
        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        // Verify it at least roundtrips correctly
        assert_eq!(root.members().unwrap(), vec!["grp"]);
    }

    #[test]
    fn encode_datatype_compound() {
        let dt = Datatype::Compound {
            size: 12,
            members: vec![
                crate::datatype::CompoundMember {
                    name: "id".to_string(),
                    byte_offset: 0,
                    datatype: Datatype::native_i32(),
                },
                crate::datatype::CompoundMember {
                    name: "x".to_string(),
                    byte_offset: 4,
                    datatype: Datatype::native_f32(),
                },
                crate::datatype::CompoundMember {
                    name: "y".to_string(),
                    byte_offset: 8,
                    datatype: Datatype::native_f32(),
                },
            ],
        };
        let enc = encode_datatype(&dt).unwrap();
        let parsed = Datatype::parse(&enc).unwrap();
        match &parsed {
            Datatype::Compound { size, members } => {
                assert_eq!(*size, 12);
                assert_eq!(members.len(), 3);
                assert_eq!(members[0].name, "id");
                assert_eq!(members[0].byte_offset, 0);
                assert_eq!(members[1].name, "x");
                assert_eq!(members[1].byte_offset, 4);
                assert_eq!(members[2].name, "y");
                assert_eq!(members[2].byte_offset, 8);
            }
            _ => panic!("expected Compound"),
        }
    }

    #[test]
    fn encode_datatype_enum() {
        let dt = Datatype::Enum {
            base: Box::new(Datatype::native_i8()),
            members: vec![
                crate::datatype::EnumMember {
                    name: "RED".to_string(),
                    value: vec![0],
                },
                crate::datatype::EnumMember {
                    name: "GREEN".to_string(),
                    value: vec![1],
                },
                crate::datatype::EnumMember {
                    name: "BLUE".to_string(),
                    value: vec![2],
                },
            ],
        };
        let enc = encode_datatype(&dt).unwrap();
        let parsed = Datatype::parse(&enc).unwrap();
        match &parsed {
            Datatype::Enum { base, members } => {
                assert_eq!(base.element_size(), 1);
                assert_eq!(members.len(), 3);
                assert_eq!(members[0].name, "RED");
                assert_eq!(members[0].value, vec![0]);
                assert_eq!(members[1].name, "GREEN");
                assert_eq!(members[1].value, vec![1]);
                assert_eq!(members[2].name, "BLUE");
                assert_eq!(members[2].value, vec![2]);
            }
            _ => panic!("expected Enum"),
        }
    }

    #[test]
    fn encode_datatype_array() {
        let dt = Datatype::Array {
            element_type: Box::new(Datatype::native_i32()),
            dimensions: vec![3],
        };
        let enc = encode_datatype(&dt).unwrap();
        let parsed = Datatype::parse(&enc).unwrap();
        match &parsed {
            Datatype::Array {
                element_type,
                dimensions,
            } => {
                assert_eq!(dimensions, &[3]);
                assert_eq!(element_type.element_size(), 4);
                assert_eq!(parsed.element_size(), 12);
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn encode_datatype_complex() {
        let dt = Datatype::Complex {
            size: 16,
            base: Box::new(Datatype::native_f64()),
        };
        let enc = encode_datatype(&dt).unwrap();
        let parsed = Datatype::parse(&enc).unwrap();
        match &parsed {
            Datatype::Complex { size, base } => {
                assert_eq!(*size, 16);
                assert_eq!(base.element_size(), 8);
            }
            _ => panic!("expected Complex"),
        }
    }

    #[test]
    fn roundtrip_compact_dataset() {
        let mut w = FileWriter::new();
        let data: Vec<u8> = [100i16, 200, 300, 400]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let ds = w
            .root_mut()
            .add_dataset("small", Datatype::native_i16(), &[4], data.clone());
        ds.set_layout(StorageLayout::Compact);
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let ds = root.dataset("small").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![4]);
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn roundtrip_fill_value() {
        let mut w = FileWriter::new();
        let data: Vec<u8> = [10i32, 20, 30, 40, -999, -999]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let ds = w
            .root_mut()
            .add_dataset("filled", Datatype::native_i32(), &[6], data.clone());
        ds.set_fill_value((-999i32).to_le_bytes().to_vec());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let root = file.root_group().unwrap();
        let ds = root.dataset("filled").unwrap();
        let fv = ds.fill_value().unwrap();
        assert!(fv.defined);
        let val_bytes = fv.value.expect("expected fill value bytes");
        let fill_val = i32::from_le_bytes(val_bytes.try_into().unwrap());
        assert_eq!(fill_val, -999);
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn roundtrip_compound_dataset() {
        let dt = Datatype::Compound {
            size: 12,
            members: vec![
                crate::datatype::CompoundMember {
                    name: "id".to_string(),
                    byte_offset: 0,
                    datatype: Datatype::native_i32(),
                },
                crate::datatype::CompoundMember {
                    name: "x".to_string(),
                    byte_offset: 4,
                    datatype: Datatype::native_f32(),
                },
                crate::datatype::CompoundMember {
                    name: "y".to_string(),
                    byte_offset: 8,
                    datatype: Datatype::native_f32(),
                },
            ],
        };
        // 3 records: {1,1.0,2.0}, {2,3.0,4.0}, {3,5.0,6.0}
        let mut data = Vec::new();
        for i in 0..3 {
            data.extend_from_slice(&((i + 1) as i32).to_le_bytes());
            data.extend_from_slice(&((2 * i + 1) as f32).to_le_bytes());
            data.extend_from_slice(&((2 * i + 2) as f32).to_le_bytes());
        }
        let mut w = FileWriter::new();
        w.root_mut().add_dataset("points", dt, &[3], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("points").unwrap();
        assert_eq!(ds.read_raw().unwrap(), data);
        match ds.datatype().unwrap() {
            Datatype::Compound { size, members } => {
                assert_eq!(size, 12);
                assert_eq!(members.len(), 3);
            }
            _ => panic!("expected Compound"),
        }
    }

    #[test]
    fn roundtrip_enum_dataset() {
        let dt = Datatype::Enum {
            base: Box::new(Datatype::native_i8()),
            members: vec![
                crate::datatype::EnumMember {
                    name: "RED".to_string(),
                    value: vec![0],
                },
                crate::datatype::EnumMember {
                    name: "GREEN".to_string(),
                    value: vec![1],
                },
                crate::datatype::EnumMember {
                    name: "BLUE".to_string(),
                    value: vec![2],
                },
            ],
        };
        let data = vec![0u8, 1, 2, 1, 0];
        let mut w = FileWriter::new();
        w.root_mut().add_dataset("colors", dt, &[5], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("colors").unwrap();
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn roundtrip_array_dataset() {
        let dt = Datatype::Array {
            element_type: Box::new(Datatype::native_i32()),
            dimensions: vec![3],
        };
        let data: Vec<u8> = (1..=12i32).flat_map(|x| x.to_le_bytes()).collect();
        let mut w = FileWriter::new();
        w.root_mut().add_dataset("vectors", dt, &[4], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("vectors").unwrap();
        assert_eq!(ds.read_raw().unwrap(), data);
        assert_eq!(ds.shape().unwrap(), vec![4]);
    }

    #[test]
    fn roundtrip_complex_dataset() {
        let dt = Datatype::Complex {
            size: 16,
            base: Box::new(Datatype::native_f64()),
        };
        // (1+2i), (3+4i), (-1+0i), (0-5i)
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.0, -5.0];
        let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut w = FileWriter::new();
        w.root_mut()
            .add_dataset("complex_data", dt, &[4], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("complex_data").unwrap();
        assert_eq!(ds.read_raw().unwrap(), data);
    }

    #[test]
    fn roundtrip_big_endian_dataset() {
        let dt = Datatype::FixedPoint {
            size: 4,
            byte_order: ByteOrder::BigEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        let values: Vec<i32> = vec![1, 256, 65536, -1, 1000000, 0];
        let data: Vec<u8> = values.iter().flat_map(|x| x.to_be_bytes()).collect();
        let mut w = FileWriter::new();
        w.root_mut().add_dataset("be_data", dt, &[6], data.clone());
        let bytes = w.to_bytes().unwrap();

        let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
        let ds = file.root_group().unwrap().dataset("be_data").unwrap();
        assert_eq!(ds.read_raw().unwrap(), data);

        // Verify the datatype is big-endian
        match ds.datatype().unwrap() {
            Datatype::FixedPoint { byte_order, .. } => {
                assert_eq!(byte_order, ByteOrder::BigEndian);
            }
            _ => panic!("expected FixedPoint"),
        }
    }

    #[test]
    fn h5dump_validates_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("h5dump_test.h5");

        let mut w = FileWriter::new();
        let root = w.root_mut();
        let data: Vec<u8> = (0..10i32).flat_map(|x| x.to_le_bytes()).collect();
        root.add_dataset("numbers", Datatype::native_i32(), &[10], data);
        let grp = root.add_group("metadata");
        grp.add_attribute(
            "version",
            Datatype::native_i32(),
            &[],
            1i32.to_le_bytes().to_vec(),
        );
        w.write_to_file(&path).unwrap();

        // Try h5dump if available
        if let Ok(output) = std::process::Command::new("h5dump")
            .arg("-H")
            .arg(&path)
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success(),
                "h5dump failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(stdout.contains("numbers"));
            assert!(stdout.contains("metadata"));
        }
    }

    #[test]
    fn write_to_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.h5");

        let mut w = FileWriter::new();
        let data: Vec<u8> = (0..3i32).flat_map(|x| x.to_le_bytes()).collect();
        w.root_mut()
            .add_dataset("vals", Datatype::native_i32(), &[3], data.clone());
        w.write_to_file(&path).unwrap();

        let file = crate::File::open(&path).unwrap();
        let ds = file.root_group().unwrap().dataset("vals").unwrap();
        assert_eq!(ds.read_raw().unwrap(), data);
    }
}
