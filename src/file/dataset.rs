use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::Error;
use crate::error::Result;
use crate::file::attribute::Attribute;
use crate::file::attribute::parse_attributes;
use crate::file::attribute::parse_attributes_by_creation_order;
use crate::file::fill_value::FillValue;
use crate::file::hdf5_file::File;
use crate::filters::FilterPipeline;
use crate::io::ReadAt;
use crate::layout::DataLayout;
use crate::object_header::ObjectHeader;
use crate::object_header::messages::MessageType;

/// A dataset in the HDF5 file.
pub struct Dataset<'a, R: ReadAt + ?Sized> {
    pub(crate) file: &'a File<R>,
    #[allow(dead_code)]
    pub(crate) address: u64,
    pub(crate) header: ObjectHeader,
}

impl<'a, R: ReadAt + ?Sized> Dataset<'a, R> {
    /// The dataset's datatype.
    pub fn datatype(&self) -> Result<Datatype> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Datatype {
                if msg.is_shared() {
                    let resolved = self
                        .file
                        .resolve_shared_message(&msg.data, MessageType::Datatype)?;
                    return Datatype::parse(&resolved);
                }
                return Datatype::parse(&msg.data);
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no datatype message".into(),
        })
    }

    /// The dataset's dataspace (shape information).
    pub fn dataspace(&self) -> Result<Dataspace> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::Dataspace {
                if msg.is_shared() {
                    let resolved = self
                        .file
                        .resolve_shared_message(&msg.data, MessageType::Dataspace)?;
                    return Dataspace::parse(&resolved);
                }
                return Dataspace::parse(&msg.data);
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no dataspace message".into(),
        })
    }

    /// The dataset's shape (convenience wrapper).
    pub fn shape(&self) -> Result<Vec<u64>> {
        Ok(self.dataspace()?.shape().to_vec())
    }

    /// The data layout (contiguous, chunked, compact).
    pub fn layout(&self) -> Result<DataLayout> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::DataLayout {
                return DataLayout::parse(
                    &msg.data,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                );
            }
        }
        Err(Error::InvalidObjectHeader {
            msg: "dataset has no layout message".into(),
        })
    }

    /// The filter pipeline, if any.
    pub fn filters(&self) -> Result<Option<FilterPipeline>> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::FilterPipeline {
                let data = if msg.is_shared() {
                    self.file
                        .resolve_shared_message(&msg.data, MessageType::FilterPipeline)?
                } else {
                    msg.data.clone()
                };
                return Ok(Some(FilterPipeline::parse(&data)?));
            }
        }
        Ok(None)
    }

    /// The fill value for this dataset, if defined.
    pub fn fill_value(&self) -> Result<FillValue> {
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::FillValue {
                let data = if msg.is_shared() {
                    self.file
                        .resolve_shared_message(&msg.data, MessageType::FillValue)?
                } else {
                    msg.data.clone()
                };
                return FillValue::parse(&data);
            }
        }
        // Fallback: check for Fill Value Old (0x0004)
        for msg in &self.header.messages {
            if msg.msg_type == MessageType::FillValueOld {
                return FillValue::parse_old(&msg.data);
            }
        }
        Ok(FillValue {
            defined: false,
            value: None,
        })
    }

    /// Read all attributes on this dataset.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        parse_attributes(&self.header, self.file)
    }

    /// Read all attributes on this dataset in creation order.
    ///
    /// Falls back to name order if creation order tracking is not available.
    pub fn attributes_by_creation_order(&self) -> Result<Vec<Attribute>> {
        parse_attributes_by_creation_order(&self.header, self.file)
    }

    /// Read the entire dataset as raw bytes.
    ///
    /// Returns the uncompressed, un-filtered data. The caller is responsible
    /// for interpreting the bytes according to `datatype()`.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        let layout = self.layout()?;
        let filters = self.filters()?;

        match layout {
            DataLayout::Compact { data } => {
                if let Some(pipeline) = filters {
                    pipeline.decompress(data)
                } else {
                    Ok(data)
                }
            }
            DataLayout::Contiguous { address, size } => {
                if address == u64::MAX {
                    // No data allocated — return fill value or zeros
                    let dtype = self.datatype()?;
                    let dspace = self.dataspace()?;
                    let total_size = dspace.num_elements() * dtype.element_size() as u64;
                    return Ok(vec![0u8; total_size as usize]);
                }
                let mut data = vec![0u8; size as usize];
                self.file
                    .reader
                    .read_exact_at(address, &mut data)
                    .map_err(Error::Io)?;
                if let Some(pipeline) = filters {
                    pipeline.decompress(data)
                } else {
                    Ok(data)
                }
            }
            DataLayout::Chunked { .. } => {
                let dtype = self.datatype()?;
                let dspace = self.dataspace()?;
                let dataset_dims = dspace.shape();
                let element_size = dtype.element_size();
                let max_dims = dspace.max_dimensions();
                crate::chunk::read_chunked(
                    &*self.file.reader,
                    &layout,
                    dataset_dims,
                    element_size,
                    &filters,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                    max_dims,
                )
            }
            DataLayout::Virtual { .. } => Err(Error::Other {
                msg: "virtual dataset reading not supported".into(),
            }),
        }
    }

    /// Read a hyperslab (rectangular sub-region) of the dataset.
    ///
    /// `start`: the starting index in each dimension.
    /// `count`: the number of elements to read in each dimension.
    ///
    /// Returns a flat byte buffer in row-major order containing only the
    /// selected elements. The caller interprets bytes according to `datatype()`.
    pub fn read_slice(&self, start: &[u64], count: &[u64]) -> Result<Vec<u8>> {
        let dspace = self.dataspace()?;
        let dataset_dims = dspace.shape();
        let ndims = dataset_dims.len();

        if start.len() != ndims || count.len() != ndims {
            return Err(Error::Other {
                msg: format!(
                    "selection rank ({}/{}) doesn't match dataset rank ({})",
                    start.len(),
                    count.len(),
                    ndims
                ),
            });
        }

        for i in 0..ndims {
            if start[i] + count[i] > dataset_dims[i] {
                return Err(Error::Other {
                    msg: format!(
                        "selection [{}, {}) exceeds dimension {} size {}",
                        start[i],
                        start[i] + count[i],
                        i,
                        dataset_dims[i]
                    ),
                });
            }
        }

        let dtype = self.datatype()?;
        let element_size = dtype.element_size() as usize;
        let layout = self.layout()?;
        let filters = self.filters()?;

        match layout {
            DataLayout::Compact { data } => {
                let raw = if let Some(pipeline) = &filters {
                    pipeline.decompress(data)?
                } else {
                    data
                };
                Ok(extract_hyperslab(
                    &raw,
                    dataset_dims,
                    start,
                    count,
                    element_size,
                ))
            }
            DataLayout::Contiguous { address, size } => {
                let raw = if address == u64::MAX {
                    vec![0u8; size as usize]
                } else {
                    let mut buf = vec![0u8; size as usize];
                    self.file
                        .reader
                        .read_exact_at(address, &mut buf)
                        .map_err(Error::Io)?;
                    if let Some(pipeline) = &filters {
                        pipeline.decompress(buf)?
                    } else {
                        buf
                    }
                };
                Ok(extract_hyperslab(
                    &raw,
                    dataset_dims,
                    start,
                    count,
                    element_size,
                ))
            }
            DataLayout::Chunked { .. } => {
                let max_dims = dspace.max_dimensions();
                crate::chunk::read_chunked_slice(
                    &*self.file.reader,
                    &layout,
                    dataset_dims,
                    element_size as u32,
                    &filters,
                    self.file.size_of_offsets(),
                    self.file.size_of_lengths(),
                    max_dims,
                    start,
                    count,
                )
            }
            DataLayout::Virtual { .. } => Err(Error::Other {
                msg: "virtual dataset reading not supported".into(),
            }),
        }
    }

    /// Read a variable-length dataset, resolving global heap references.
    ///
    /// Returns one `Vec<u8>` per element, containing the resolved vlen data.
    /// For vlen strings, each entry is the raw string bytes (use
    /// `read_vlen_strings()` for convenience).
    pub fn read_vlen(&self) -> Result<Vec<Vec<u8>>> {
        let raw = self.read_raw()?;
        let dspace = self.dataspace()?;
        let num_elements = dspace.num_elements() as usize;

        crate::global_heap::resolve_vlen_elements(
            &*self.file.reader,
            &raw,
            num_elements,
            self.file.size_of_offsets(),
            self.file.size_of_lengths(),
        )
    }

    /// Read a variable-length string dataset, returning Rust strings.
    ///
    /// Each element is converted from raw bytes to a `String`, stripping
    /// any null padding.
    pub fn read_vlen_strings(&self) -> Result<Vec<String>> {
        let vlen_data = self.read_vlen()?;
        Ok(vlen_data
            .into_iter()
            .map(|bytes| {
                String::from_utf8_lossy(&bytes)
                    .trim_end_matches('\0')
                    .to_string()
            })
            .collect())
    }

    /// Read the dataset and convert to native byte order if necessary.
    ///
    /// If the on-disk data is big-endian on a little-endian platform (or vice
    /// versa), each element is byte-swapped in place. This only applies to
    /// fixed-point and floating-point types. Other types are returned as-is.
    pub fn read_native(&self) -> Result<Vec<u8>> {
        let mut data = self.read_raw()?;
        let dt = self.datatype()?;
        swap_to_native(&dt, &mut data);
        Ok(data)
    }
}

/// Extract a hyperslab from a flat row-major buffer.
fn extract_hyperslab(
    data: &[u8],
    dims: &[u64],
    start: &[u64],
    count: &[u64],
    element_size: usize,
) -> Vec<u8> {
    let ndims = dims.len();
    let out_elems: usize = count.iter().map(|&c| c as usize).product();
    let mut output = vec![0u8; out_elems * element_size];

    if ndims == 0 || out_elems == 0 {
        return output;
    }

    // Compute source strides (row-major, in bytes)
    let mut src_strides = vec![element_size; ndims];
    for i in (0..ndims - 1).rev() {
        src_strides[i] = src_strides[i + 1] * dims[i + 1] as usize;
    }

    // Number of contiguous rows (innermost dim)
    let inner_count = count[ndims - 1] as usize * element_size;
    let nrows: usize = count[..ndims - 1]
        .iter()
        .map(|&c| c as usize)
        .product::<usize>()
        .max(1);

    for row in 0..nrows {
        let mut remaining = row;
        let mut src_off = 0usize;
        let dst_off = row * inner_count;

        for i in 0..ndims - 1 {
            let rows_below: usize = count[i + 1..ndims - 1]
                .iter()
                .map(|&c| c as usize)
                .product::<usize>()
                .max(1);
            let idx = remaining / rows_below;
            remaining %= rows_below;
            src_off += (start[i] as usize + idx) * src_strides[i];
        }
        src_off += start[ndims - 1] as usize * element_size;

        if src_off + inner_count <= data.len() && dst_off + inner_count <= output.len() {
            output[dst_off..dst_off + inner_count]
                .copy_from_slice(&data[src_off..src_off + inner_count]);
        }
    }

    output
}

/// Byte-swap data in place if the datatype's byte order differs from native.
fn swap_to_native(dt: &Datatype, data: &mut [u8]) {
    use crate::datatype::ByteOrder;

    let (elem_size, order) = match dt {
        Datatype::FixedPoint {
            size, byte_order, ..
        } => (*size as usize, *byte_order),
        Datatype::FloatingPoint {
            size, byte_order, ..
        } => (*size as usize, *byte_order),
        Datatype::Complex { base, .. } => {
            // Swap each component (real, imaginary) using the base float type
            if let Datatype::FloatingPoint {
                size: base_size, ..
            } = base.as_ref()
            {
                let bs = *base_size as usize;
                for chunk in data.chunks_exact_mut(bs) {
                    // Delegate to the base type for each component
                    swap_to_native(base, chunk);
                }
            }
            return;
        }
        _ => return,
    };

    let needs_swap = if cfg!(target_endian = "little") {
        order == ByteOrder::BigEndian
    } else {
        order == ByteOrder::LittleEndian
    };

    if !needs_swap || elem_size <= 1 {
        return;
    }

    for chunk in data.chunks_exact_mut(elem_size) {
        chunk.reverse();
    }
}
