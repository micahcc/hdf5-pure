use crate::error::Error;
use crate::error::Result;
use crate::object_header::messages::MessageType;
use crate::superblock::UNDEF_ADDR;
use crate::writer::chunk_index::write_btree_v1_chunk_index;
use crate::writer::chunk_index::write_btree_v2_chunk_index;
use crate::writer::chunk_index::write_extensible_array_index;
use crate::writer::chunk_index::write_fixed_array_index;
use crate::writer::chunk_util::compute_chunk_count;
use crate::writer::chunk_util::enumerate_chunks;
use crate::writer::chunk_util::extract_chunk_data;
use crate::writer::dataset_node::DatasetNode;
use crate::writer::encode::OhdrMsg;
use crate::writer::encode::SIZE_OF_OFFSETS;
use crate::writer::encode::encode_attribute;
use crate::writer::encode::encode_attribute_info;
use crate::writer::encode::encode_chunked_layout;
use crate::writer::encode::encode_compact_layout;
use crate::writer::encode::encode_contiguous_layout;
use crate::writer::encode::encode_dataspace;
use crate::writer::encode::encode_datatype;
use crate::writer::encode::encode_fill_value_msg;
use crate::writer::encode::encode_filter_pipeline;
use crate::writer::encode::encode_group_info;
use crate::writer::encode::encode_link;
use crate::writer::encode::encode_link_info;
use crate::writer::encode::encode_object_header;
use crate::writer::encode::ohdr_overhead;
use crate::writer::file_writer::WriteOptions;
use crate::writer::gcol::build_global_heap_collection;
use crate::writer::gcol::build_vlen_heap_ids;
use crate::writer::group_node::GroupNode;
use crate::writer::types::ChildNode;
use crate::writer::types::StorageLayout;
use crate::writer::write_filters::apply_filters_forward;

pub(crate) fn write_group(
    group: &GroupNode,
    buf: &mut Vec<u8>,
    opts: &WriteOptions,
) -> Result<u64> {
    let mut child_addrs: Vec<(&str, u64)> = Vec::new();
    for (name, child) in &group.children {
        let addr = match child {
            ChildNode::Group(g) => write_group(g, buf, opts)?,
            ChildNode::Dataset(d) => write_dataset(d, buf, opts)?,
        };
        child_addrs.push((name, addr));
    }

    let ohdr_addr = buf.len() as u64;

    let mut messages: Vec<OhdrMsg> = Vec::new();
    messages.push((MessageType::LinkInfo.as_u8(), encode_link_info(), 0));
    let group_info_flags: u8 = if opts.hdf5lib_compat { 0x01 } else { 0x00 };
    messages.push((
        MessageType::GroupInfo.as_u8(),
        encode_group_info(),
        group_info_flags,
    ));
    for (name, addr) in &child_addrs {
        messages.push((MessageType::Link.as_u8(), encode_link(name, *addr), 0));
    }
    for attr in &group.attributes {
        messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?, 0));
    }

    let ohdr = encode_object_header(&messages, opts, None)?;
    buf.extend_from_slice(&ohdr);

    Ok(ohdr_addr)
}

fn write_dataset(ds: &DatasetNode, buf: &mut Vec<u8>, opts: &WriteOptions) -> Result<u64> {
    validate_dataset(ds)?;

    let ohdr_addr = buf.len() as u64;
    let compat = opts.hdf5lib_compat;

    match ds.layout {
        StorageLayout::Contiguous => {
            let dt_body = encode_datatype(&ds.datatype)?;

            // In compat mode, always include max_dims (set to current dims if not specified)
            let effective_max_dims: Option<Vec<u64>> = if compat {
                Some(ds.max_dims.clone().unwrap_or_else(|| ds.shape.clone()))
            } else {
                ds.max_dims.clone()
            };
            let ds_body = encode_dataspace(&ds.shape, effective_max_dims.as_deref());

            let fv_body = encode_fill_value_msg(&ds.fill_value, compat);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(encode_attribute)
                .collect::<Result<Vec<_>>>()?;

            // Message flags
            let dt_flags: u8 = if compat { 0x01 } else { 0x00 };
            let fv_flags: u8 = if compat { 0x01 } else { 0x00 };

            let layout_body_size = 18usize;

            // Compute total message size for OHDR overhead calculation
            let attr_info_size = if compat { 4 + 18 } else { 0 };
            let total_msg_size: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + (4 + layout_body_size)
                + attr_info_size
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(total_msg_size, opts);

            if let Some(ref vlen_elems) = ds.vlen_elements {
                let gcol_addr = ohdr_addr + ohdr_size as u64;
                let gcol_bytes = build_global_heap_collection(vlen_elems);
                let heap_id_data_addr = gcol_addr + gcol_bytes.len() as u64;
                let heap_id_data = build_vlen_heap_ids(vlen_elems, gcol_addr);

                let layout_body =
                    encode_contiguous_layout(heap_id_data_addr, heap_id_data.len() as u64);
                debug_assert_eq!(layout_body.len(), layout_body_size);

                let messages = build_dataset_messages(
                    dt_body,
                    ds_body,
                    fv_body,
                    layout_body,
                    &attr_bodies,
                    dt_flags,
                    fv_flags,
                    compat,
                )?;

                let ohdr = encode_object_header(&messages, opts, None)?;
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

                let messages = build_dataset_messages(
                    dt_body,
                    ds_body,
                    fv_body,
                    layout_body,
                    &attr_bodies,
                    dt_flags,
                    fv_flags,
                    compat,
                )?;

                let ohdr = encode_object_header(&messages, opts, None)?;
                debug_assert_eq!(ohdr.len(), ohdr_size);

                buf.extend_from_slice(&ohdr);
                buf.extend_from_slice(&ds.data);
            }
        }
        StorageLayout::Compact => {
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value, compat);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(encode_attribute)
                .collect::<Result<Vec<_>>>()?;
            let layout_body = encode_compact_layout(&ds.data);

            let dt_flags: u8 = if compat { 0x01 } else { 0x00 };
            let fv_flags: u8 = if compat { 0x01 } else { 0x00 };

            let messages = build_dataset_messages(
                dt_body,
                ds_body,
                fv_body,
                layout_body,
                &attr_bodies,
                dt_flags,
                fv_flags,
                compat,
            )?;

            let ohdr = encode_object_header(&messages, opts, None)?;
            buf.extend_from_slice(&ohdr);
        }
        StorageLayout::Chunked {
            ref chunk_dims,
            ref filters,
        } if ds.layout_version == 3 => {
            let element_size = ds.datatype.element_size();
            let dt_body = encode_datatype(&ds.datatype)?;
            let ds_body = encode_dataspace(&ds.shape, ds.max_dims.as_deref());
            let fv_body = encode_fill_value_msg(&ds.fill_value, compat);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(encode_attribute)
                .collect::<Result<Vec<_>>>()?;
            let has_filters = !filters.is_empty();
            let filter_body = if has_filters {
                Some(encode_filter_pipeline(filters, element_size))
            } else {
                None
            };

            let chunk_coords_list = enumerate_chunks(&ds.shape, chunk_dims);
            let mut chunk_data_parts: Vec<Vec<u8>> = Vec::new();
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

            let ndims = ds.shape.len();
            let dimensionality = ndims + 1;
            let layout_body_size = 3 + SIZE_OF_OFFSETS as usize + dimensionality * 4;

            let fixed_msg_sizes: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + filter_body.as_ref().map_or(0, |fb| 4 + fb.len())
                + (4 + layout_body_size)
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(fixed_msg_sizes, opts);

            let chunk_data_start = ohdr_addr + ohdr_size as u64;
            let mut chunk_addrs: Vec<u64> = Vec::new();
            let mut pos = chunk_data_start;
            for part in &chunk_data_parts {
                chunk_addrs.push(pos);
                pos += part.len() as u64;
            }

            let btree_addr = pos;

            let mut layout_body = Vec::with_capacity(layout_body_size);
            layout_body.push(3);
            layout_body.push(2);
            layout_body.push(dimensionality as u8);
            layout_body.extend_from_slice(&btree_addr.to_le_bytes());
            for dim in &chunk_dims[..ndims] {
                layout_body.extend_from_slice(&(*dim as u32).to_le_bytes());
            }
            layout_body.extend_from_slice(&element_size.to_le_bytes());
            debug_assert_eq!(layout_body.len(), layout_body_size);

            let mut messages: Vec<OhdrMsg> = vec![
                (MessageType::Datatype.as_u8(), dt_body, 0),
                (MessageType::Dataspace.as_u8(), ds_body, 0),
                (MessageType::FillValue.as_u8(), fv_body, 0),
            ];
            if let Some(fb) = filter_body {
                messages.push((MessageType::FilterPipeline.as_u8(), fb, 0));
            }
            messages.push((MessageType::DataLayout.as_u8(), layout_body, 0));
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body, 0));
            }

            let ohdr = encode_object_header(&messages, opts, None)?;
            debug_assert_eq!(ohdr.len(), ohdr_size);
            buf.extend_from_slice(&ohdr);

            for part in &chunk_data_parts {
                buf.extend_from_slice(part);
            }

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
            let fv_body = encode_fill_value_msg(&ds.fill_value, compat);
            let attr_bodies: Vec<Vec<u8>> = ds
                .attributes
                .iter()
                .map(encode_attribute)
                .collect::<Result<Vec<_>>>()?;

            let chunk_elements: u64 = chunk_dims.iter().product();
            let raw_chunk_bytes = chunk_elements * element_size as u64;

            let total_chunks = compute_chunk_count(&ds.shape, chunk_dims);
            let has_filters = !filters.is_empty();

            let filter_body = if has_filters {
                Some(encode_filter_pipeline(filters, element_size))
            } else {
                None
            };

            let n_unlimited = ds
                .max_dims
                .as_ref()
                .map_or(0, |md| md.iter().filter(|&&d| d == u64::MAX).count());
            let index_type_id = if n_unlimited >= 2 {
                5u8
            } else if total_chunks == 1 && n_unlimited == 0 {
                1u8
            } else if n_unlimited == 1 {
                4u8
            } else if !has_filters {
                2u8
            } else {
                3u8
            };

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

            let dummy_layout = encode_chunked_layout(
                chunk_dims,
                index_type_id,
                0,
                if index_type_id == 1 && has_filters {
                    Some(chunk_sizes[0])
                } else {
                    None
                },
            );
            let layout_body_size = dummy_layout.len();

            let fixed_msg_sizes: usize = [&dt_body, &ds_body, &fv_body]
                .iter()
                .map(|b| 4 + b.len())
                .sum::<usize>()
                + filter_body.as_ref().map_or(0, |fb| 4 + fb.len())
                + (4 + layout_body_size)
                + attr_bodies.iter().map(|b| 4 + b.len()).sum::<usize>();
            let ohdr_size = ohdr_overhead(fixed_msg_sizes, opts);

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
                2 => (chunk_data_start, None),
                3..=5 => (pos, None),
                _ => (pos, None),
            };

            let layout_body = encode_chunked_layout(
                chunk_dims,
                index_type_id,
                chunk_index_addr,
                single_filtered_size,
            );
            debug_assert_eq!(layout_body.len(), layout_body_size);

            let mut messages: Vec<OhdrMsg> = vec![
                (MessageType::Datatype.as_u8(), dt_body, 0),
                (MessageType::Dataspace.as_u8(), ds_body, 0),
                (MessageType::FillValue.as_u8(), fv_body, 0),
            ];
            if let Some(fb) = filter_body {
                messages.push((MessageType::FilterPipeline.as_u8(), fb, 0));
            }
            messages.push((MessageType::DataLayout.as_u8(), layout_body, 0));
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body, 0));
            }

            let ohdr = encode_object_header(&messages, opts, None)?;
            debug_assert_eq!(ohdr.len(), ohdr_size);
            buf.extend_from_slice(&ohdr);

            for part in &chunk_data_parts {
                buf.extend_from_slice(part);
            }

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
                _ => {}
            }
        }
    }

    Ok(ohdr_addr)
}

/// Build the dataset message list in the correct order.
///
/// In compat mode: Dataspace, Datatype, FillValue, DataLayout, AttributeInfo, Attributes.
/// In normal mode: Datatype, Dataspace, FillValue, DataLayout, Attributes.
fn build_dataset_messages(
    dt_body: Vec<u8>,
    ds_body: Vec<u8>,
    fv_body: Vec<u8>,
    layout_body: Vec<u8>,
    attr_bodies: &[Vec<u8>],
    dt_flags: u8,
    fv_flags: u8,
    compat: bool,
) -> Result<Vec<OhdrMsg>> {
    let mut messages: Vec<OhdrMsg> = if compat {
        vec![
            (MessageType::Dataspace.as_u8(), ds_body, 0),
            (MessageType::Datatype.as_u8(), dt_body, dt_flags),
            (MessageType::FillValue.as_u8(), fv_body, fv_flags),
            (MessageType::DataLayout.as_u8(), layout_body, 0),
            (
                MessageType::AttributeInfo.as_u8(),
                encode_attribute_info(),
                0x04,
            ),
        ]
    } else {
        vec![
            (MessageType::Datatype.as_u8(), dt_body, 0),
            (MessageType::Dataspace.as_u8(), ds_body, 0),
            (MessageType::FillValue.as_u8(), fv_body, 0),
            (MessageType::DataLayout.as_u8(), layout_body, 0),
        ]
    };
    for body in attr_bodies {
        messages.push((MessageType::Attribute.as_u8(), body.clone(), 0));
    }
    Ok(messages)
}

fn validate_dataset(ds: &DatasetNode) -> Result<()> {
    if ds.vlen_elements.is_some() {
        return Ok(());
    }
    let num_elements: u64 = if ds.shape.is_empty() {
        1
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
