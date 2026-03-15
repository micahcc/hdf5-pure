use crate::error::Result;
use crate::object_header::messages::MessageType;
use crate::superblock::UNDEF_ADDR;
use crate::writer::dataset_node::DatasetNode;
use crate::writer::encode::OhdrMsg;
use crate::writer::encode::SUPERBLOCK_SIZE;
use crate::writer::encode::compat_dataset_chunk_size;
use crate::writer::encode::compat_group_chunk_size;
use crate::writer::encode::encode_attribute;
use crate::writer::encode::encode_attribute_info;
use crate::writer::encode::encode_contiguous_layout;
use crate::writer::encode::encode_dataspace;
use crate::writer::encode::encode_datatype;
use crate::writer::encode::encode_fill_value_msg;
use crate::writer::encode::encode_group_info;
use crate::writer::encode::encode_link;
use crate::writer::encode::encode_link_info;
use crate::writer::encode::encode_object_header;
use crate::writer::encode::encode_superblock;
use crate::writer::encode::ohdr_overhead;
use crate::writer::file_writer::WriteOptions;
use crate::writer::group_node::GroupNode;
use crate::writer::types::ChildNode;
use crate::writer::types::StorageLayout;

/// Two-pass compat serialization: parent-first ordering + metadata block alignment.
///
/// Layout: [superblock] [root_group_ohdr] [child_ohdrs...] [padding] [data_blocks...]
pub(crate) fn write_tree_compat(
    root: &GroupNode,
    opts: &WriteOptions,
    meta_block_size: usize,
) -> Result<Vec<u8>> {
    // Phase 1: Flatten tree into objects, compute sizes with dummy addresses.
    let mut objects: Vec<ObjectInfo> = Vec::new();
    flatten_tree(root, opts, &mut objects)?;

    // Phase 2: Assign metadata addresses (parent-first, starting after superblock).
    let mut meta_pos = SUPERBLOCK_SIZE;
    for obj in &mut objects {
        obj.meta_addr = meta_pos as u64;
        meta_pos += obj.ohdr_size;
    }

    // Phase 3: Compute data addresses (after metadata block).
    let data_start = meta_block_size.max(meta_pos);
    let mut data_pos = data_start;
    for obj in &mut objects {
        if !obj.data.is_empty() {
            obj.data_addr = data_pos as u64;
            data_pos += obj.data.len();
        }
    }

    let eof = data_pos;

    // Phase 4: Re-encode with real addresses.
    // We need to update:
    // - Group OHDRs: link messages point to child meta_addrs
    // - Dataset OHDRs: layout message points to data_addr
    let mut buf = vec![0u8; eof];

    // Write superblock (root group is at objects[0].meta_addr).
    let root_addr = objects[0].meta_addr;
    let sb = encode_superblock(root_addr, eof as u64);
    buf[..SUPERBLOCK_SIZE].copy_from_slice(&sb);

    // Re-encode each object with correct addresses.
    for i in 0..objects.len() {
        let meta_addr = objects[i].meta_addr;
        let data_addr = objects[i].data_addr;
        let data = objects[i].data.clone();

        let ohdr_bytes = encode_object_final(&objects, i, opts)?;
        let start = meta_addr as usize;
        buf[start..start + ohdr_bytes.len()].copy_from_slice(&ohdr_bytes);

        // Write data block.
        if !data.is_empty() {
            let ds = data_addr as usize;
            buf[ds..ds + data.len()].copy_from_slice(&data);
        }
    }

    // Truncate if data_pos < buf.len() (shouldn't happen, but be safe).
    buf.truncate(eof);
    Ok(buf)
}

struct ObjectInfo {
    kind: ObjectKind,
    /// Byte size of the encoded OHDR.
    ohdr_size: usize,
    /// Address assigned in metadata region.
    meta_addr: u64,
    /// Raw data bytes (for contiguous datasets).
    data: Vec<u8>,
    /// Address assigned for data (after metadata block).
    data_addr: u64,
    /// Child indices in the objects vec (for groups).
    child_indices: Vec<(String, usize)>,
    /// Target OHDR chunk size for NIL padding.
    target_chunk_size: Option<usize>,
}

enum ObjectKind {
    Group(GroupRef),
    Dataset(DatasetRef),
}

struct GroupRef {
    attributes: Vec<crate::writer::types::AttrData>,
}

struct DatasetRef {
    datatype: crate::datatype::Datatype,
    shape: Vec<u64>,
    max_dims: Option<Vec<u64>>,
    fill_value: Option<Vec<u8>>,
    attributes: Vec<crate::writer::types::AttrData>,
    vlen_elements: Option<Vec<Vec<u8>>>,
}

fn flatten_tree(
    group: &GroupNode,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    let my_index = objects.len();

    // Reserve slot for this group.
    objects.push(ObjectInfo {
        kind: ObjectKind::Group(GroupRef {
            attributes: group.attributes.iter().map(clone_attr).collect(),
        }),
        ohdr_size: 0,
        meta_addr: 0,
        data: vec![],
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: None,
    });

    // Recursively flatten children.
    let mut child_indices = Vec::new();
    for (name, child) in &group.children {
        let child_idx = match child {
            ChildNode::Group(g) => flatten_tree(g, opts, objects)?,
            ChildNode::Dataset(d) => flatten_dataset(d, opts, objects)?,
        };
        child_indices.push((name.clone(), child_idx));
    }

    // Compute group OHDR size using dummy child addresses.
    let messages = build_group_messages_dummy(&child_indices, &group.attributes, opts)?;
    let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();
    let target = compat_group_chunk_size(4, 8); // C library defaults: est_num_entries=4, est_name_len=8
    let target_chunk = if target > real_msg_bytes {
        Some(target)
    } else {
        None
    };
    let chunk_for_overhead = target_chunk.unwrap_or(real_msg_bytes);
    let ohdr_size = ohdr_overhead(chunk_for_overhead, opts);

    objects[my_index].child_indices = child_indices;
    objects[my_index].ohdr_size = ohdr_size;
    objects[my_index].target_chunk_size = target_chunk;

    Ok(my_index)
}

fn flatten_dataset(
    ds: &DatasetNode,
    opts: &WriteOptions,
    objects: &mut Vec<ObjectInfo>,
) -> Result<usize> {
    let idx = objects.len();

    // Compute raw data.
    let raw_data = match ds.layout {
        StorageLayout::Contiguous => {
            if ds.vlen_elements.is_some() {
                // For vlen, data is gcol + heap IDs — we'll handle this specially.
                vec![]
            } else {
                ds.data.clone()
            }
        }
        _ => vec![], // Chunked/compact data embedded in OHDR or handled differently
    };

    // Compute dataset OHDR size using dummy data address.
    let messages = build_dataset_messages_dummy(ds, opts, 0xDEAD_BEEF_0000_0000)?;
    let real_msg_bytes: usize = messages.iter().map(|(_, b, _)| 4 + b.len()).sum();
    let target = compat_dataset_chunk_size(real_msg_bytes);
    let target_chunk = if target > real_msg_bytes {
        Some(target)
    } else {
        None
    };
    let chunk_for_overhead = target_chunk.unwrap_or(real_msg_bytes);
    let ohdr_size = ohdr_overhead(chunk_for_overhead, opts);

    objects.push(ObjectInfo {
        kind: ObjectKind::Dataset(DatasetRef {
            datatype: ds.datatype.clone(),
            shape: ds.shape.clone(),
            max_dims: ds.max_dims.clone(),
            fill_value: ds.fill_value.clone(),
            attributes: ds.attributes.iter().map(clone_attr).collect(),
            vlen_elements: ds.vlen_elements.clone(),
        }),
        ohdr_size,
        meta_addr: 0,
        data: raw_data,
        data_addr: 0,
        child_indices: vec![],
        target_chunk_size: target_chunk,
    });

    Ok(idx)
}

fn encode_object_final(
    objects: &[ObjectInfo],
    index: usize,
    opts: &WriteOptions,
) -> Result<Vec<u8>> {
    let obj = &objects[index];
    let target_chunk = obj.target_chunk_size;

    match &obj.kind {
        ObjectKind::Group(g) => {
            let mut messages: Vec<OhdrMsg> = Vec::new();
            messages.push((MessageType::LinkInfo.as_u8(), encode_link_info(), 0));
            messages.push((
                MessageType::GroupInfo.as_u8(),
                encode_group_info(),
                0x01, // constant flag
            ));
            for (name, child_idx) in &obj.child_indices {
                let child_addr = objects[*child_idx].meta_addr;
                messages.push((MessageType::Link.as_u8(), encode_link(name, child_addr), 0));
            }
            for attr in &g.attributes {
                messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?, 0));
            }

            encode_object_header(&messages, opts, target_chunk)
        }
        ObjectKind::Dataset(d) => {
            let data_addr = if obj.data.is_empty() && d.vlen_elements.is_none() {
                UNDEF_ADDR
            } else if d.vlen_elements.is_some() {
                // For vlen datasets, we need to compute gcol + heap IDs inline.
                // For now, this isn't supported in compat mode.
                // Fall back to a dummy address.
                UNDEF_ADDR
            } else {
                obj.data_addr
            };

            let dt_body = encode_datatype(&d.datatype)?;

            // Always include max_dims in compat mode.
            let effective_max = Some(d.max_dims.clone().unwrap_or_else(|| d.shape.clone()));
            let ds_body = encode_dataspace(&d.shape, effective_max.as_deref());

            let fv_body = encode_fill_value_msg(&d.fill_value, true);

            let layout_body = encode_contiguous_layout(data_addr, obj.data.len() as u64);

            let attr_bodies: Vec<Vec<u8>> = d
                .attributes
                .iter()
                .map(encode_attribute)
                .collect::<Result<Vec<_>>>()?;

            let mut messages: Vec<OhdrMsg> = vec![
                (MessageType::Dataspace.as_u8(), ds_body, 0),
                (MessageType::Datatype.as_u8(), dt_body, 0x01),
                (MessageType::FillValue.as_u8(), fv_body, 0x01),
                (MessageType::DataLayout.as_u8(), layout_body, 0),
                (
                    MessageType::AttributeInfo.as_u8(),
                    encode_attribute_info(),
                    0x04,
                ),
            ];
            for body in attr_bodies {
                messages.push((MessageType::Attribute.as_u8(), body, 0));
            }

            encode_object_header(&messages, opts, target_chunk)
        }
    }
}

fn build_group_messages_dummy(
    child_indices: &[(String, usize)],
    attributes: &[crate::writer::types::AttrData],
    _opts: &WriteOptions,
) -> Result<Vec<OhdrMsg>> {
    let mut messages: Vec<OhdrMsg> = Vec::new();
    messages.push((MessageType::LinkInfo.as_u8(), encode_link_info(), 0));
    messages.push((MessageType::GroupInfo.as_u8(), encode_group_info(), 0x01));
    for (name, _) in child_indices {
        messages.push((MessageType::Link.as_u8(), encode_link(name, 0xDEAD_0000), 0));
    }
    for attr in attributes {
        messages.push((MessageType::Attribute.as_u8(), encode_attribute(attr)?, 0));
    }
    Ok(messages)
}

fn build_dataset_messages_dummy(
    ds: &DatasetNode,
    _opts: &WriteOptions,
    dummy_data_addr: u64,
) -> Result<Vec<OhdrMsg>> {
    let dt_body = encode_datatype(&ds.datatype)?;

    // Always include max_dims in compat mode.
    let effective_max = Some(ds.max_dims.clone().unwrap_or_else(|| ds.shape.clone()));
    let ds_body = encode_dataspace(&ds.shape, effective_max.as_deref());
    let fv_body = encode_fill_value_msg(&ds.fill_value, true);

    let data_size = ds.data.len() as u64;
    let layout_body = encode_contiguous_layout(dummy_data_addr, data_size);

    let attr_bodies: Vec<Vec<u8>> = ds
        .attributes
        .iter()
        .map(encode_attribute)
        .collect::<Result<Vec<_>>>()?;

    let mut messages: Vec<OhdrMsg> = vec![
        (MessageType::Dataspace.as_u8(), ds_body, 0),
        (MessageType::Datatype.as_u8(), dt_body, 0x01),
        (MessageType::FillValue.as_u8(), fv_body, 0x01),
        (MessageType::DataLayout.as_u8(), layout_body, 0),
        (
            MessageType::AttributeInfo.as_u8(),
            encode_attribute_info(),
            0x04,
        ),
    ];
    for body in attr_bodies {
        messages.push((MessageType::Attribute.as_u8(), body, 0));
    }
    Ok(messages)
}

fn clone_attr(attr: &crate::writer::types::AttrData) -> crate::writer::types::AttrData {
    crate::writer::types::AttrData {
        name: attr.name.clone(),
        datatype: attr.datatype.clone(),
        shape: attr.shape.clone(),
        value: attr.value.clone(),
    }
}
