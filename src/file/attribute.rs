use crate::btree2::BTree2Header;
use crate::btree2::{self};
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::Error;
use crate::error::Result;
use crate::file::hdf5_file::File;
use crate::file::helpers::read_offset_from_slice;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::{self};
use crate::io::ReadAt;
use crate::object_header::ObjectHeader;
use crate::object_header::messages::MessageType;

/// An attribute (name + value) on a group or dataset.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub datatype: Datatype,
    pub dataspace: Dataspace,
    pub raw_value: Vec<u8>,
}

/// Parse attribute messages from an object header.
pub(crate) fn parse_attributes<R: ReadAt + ?Sized>(
    header: &ObjectHeader,
    file: &File<R>,
) -> Result<Vec<Attribute>> {
    let mut attrs = Vec::new();

    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            if let Ok(attr) = parse_attribute_message(&msg.data, file) {
                attrs.push(attr);
            }
        }
    }

    if !attrs.is_empty() {
        return Ok(attrs);
    }

    // Look for AttributeInfo message → dense attribute storage via fractal heap + B-tree v2
    for msg in &header.messages {
        if msg.msg_type == MessageType::AttributeInfo {
            return parse_dense_attributes(&msg.data, file);
        }
    }

    Ok(attrs)
}

/// Parse attribute messages, preferring creation order when available.
pub(crate) fn parse_attributes_by_creation_order<R: ReadAt + ?Sized>(
    header: &ObjectHeader,
    file: &File<R>,
) -> Result<Vec<Attribute>> {
    let mut attrs = Vec::new();

    // Direct Attribute messages don't carry creation order — return as-is
    for msg in &header.messages {
        if msg.msg_type == MessageType::Attribute {
            if let Ok(attr) = parse_attribute_message(&msg.data, file) {
                attrs.push(attr);
            }
        }
    }

    if !attrs.is_empty() {
        return Ok(attrs);
    }

    for msg in &header.messages {
        if msg.msg_type == MessageType::AttributeInfo {
            return parse_dense_attributes_by_creation_order(&msg.data, file);
        }
    }

    Ok(attrs)
}

/// Parse an Attribute Info message and extract addresses.
///
/// Returns (fheap_addr, bt2_name_addr, bt2_corder_addr).
fn parse_attr_info_addrs<R: ReadAt + ?Sized>(
    data: &[u8],
    file: &File<R>,
) -> Result<(u64, u64, u64)> {
    let so = file.size_of_offsets();
    let o = so as usize;

    if data.len() < 2 {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute info message too short".into(),
        });
    }

    let _version = data[0];
    let flags = data[1];
    let mut pos = 2;

    // Optional max creation order
    if (flags & 0x01) != 0 {
        pos += 2;
    }

    if pos + o > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute info: truncated at fractal heap address".into(),
        });
    }
    let fheap_addr = read_offset_from_slice(data, pos, so);
    pos += o;

    if pos + o > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute info: truncated at B-tree address".into(),
        });
    }
    let bt2_name_addr = read_offset_from_slice(data, pos, so);
    pos += o;

    // Optional: B-tree v2 address (creation order index, if flags bit 1 set)
    let bt2_corder_addr = if (flags & 0x02) != 0 && pos + o <= data.len() {
        read_offset_from_slice(data, pos, so)
    } else {
        u64::MAX
    };

    Ok((fheap_addr, bt2_name_addr, bt2_corder_addr))
}

/// Read attributes from fractal heap using the given B-tree v2 index.
fn read_dense_attrs_from_btree<R: ReadAt + ?Sized>(
    file: &File<R>,
    fheap_addr: u64,
    bt2_addr: u64,
    by_creation_order: bool,
) -> Result<Vec<Attribute>> {
    let so = file.size_of_offsets();
    let sl = file.size_of_lengths();

    if fheap_addr == u64::MAX || bt2_addr == u64::MAX {
        return Ok(Vec::new());
    }

    let fheap = FractalHeapHeader::parse(&*file.reader, fheap_addr, so, sl)?;
    let bt2 = BTree2Header::parse(&*file.reader, bt2_addr, so, sl)?;

    let mut attrs = Vec::new();
    let heap_id_len = fheap.heap_id_length as usize;

    btree2::iterate_records(&*file.reader, &bt2, so, |record| {
        let heap_id = if by_creation_order {
            // Type 9 record: heap_id + flags(1) + creation_order(4)
            btree2::parse_attribute_creation_order_record(&record.data, heap_id_len)
                .map(|(_order, id)| id)
        } else {
            // Type 8 record: heap_id + flags(1) + creation_order(4) + hash(4)
            btree2::parse_attribute_name_record(&record.data, heap_id_len)
        };
        if let Some(heap_id) = heap_id {
            let attr_data =
                fractal_heap::read_managed_object(&*file.reader, &fheap, &heap_id, so, sl)?;
            if let Ok(attr) = parse_attribute_message(&attr_data, file) {
                attrs.push(attr);
            }
        }
        Ok(())
    })?;

    Ok(attrs)
}

/// Parse dense attributes from an Attribute Info message (name order).
fn parse_dense_attributes<R: ReadAt + ?Sized>(
    data: &[u8],
    file: &File<R>,
) -> Result<Vec<Attribute>> {
    let (fheap_addr, bt2_name_addr, _) = parse_attr_info_addrs(data, file)?;
    read_dense_attrs_from_btree(file, fheap_addr, bt2_name_addr, false)
}

/// Parse dense attributes from an Attribute Info message (creation order).
fn parse_dense_attributes_by_creation_order<R: ReadAt + ?Sized>(
    data: &[u8],
    file: &File<R>,
) -> Result<Vec<Attribute>> {
    let (fheap_addr, bt2_name_addr, bt2_corder_addr) = parse_attr_info_addrs(data, file)?;
    if bt2_corder_addr != u64::MAX {
        read_dense_attrs_from_btree(file, fheap_addr, bt2_corder_addr, true)
    } else {
        // Fall back to name order
        read_dense_attrs_from_btree(file, fheap_addr, bt2_name_addr, false)
    }
}

/// Parse an attribute message body.
///
/// Attribute message layout (version 3):
/// ```text
/// Byte 0:    Version (1, 2, or 3)
/// Byte 1:    Flags (bit 0: datatype shared, bit 1: dataspace shared)
/// Byte 2-3:  Name size (u16)
/// Byte 4-5:  Datatype size (u16)
/// Byte 6-7:  Dataspace size (u16)
/// Byte 8:    Character set (version 3 only: 0=ASCII, 1=UTF-8)
/// Name (null-terminated, NOT padded in version 3)
/// Datatype message
/// Dataspace message
/// Value
/// ```
fn parse_attribute_message<R: ReadAt + ?Sized>(data: &[u8], file: &File<R>) -> Result<Attribute> {
    if data.len() < 6 {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute message too short".into(),
        });
    }

    let version = data[0];
    let flags = data[1];
    let name_size = u16::from_le_bytes([data[2], data[3]]) as usize;
    let dt_size = u16::from_le_bytes([data[4], data[5]]) as usize;
    let ds_size = u16::from_le_bytes([data[6], data[7]]) as usize;

    let mut pos = match version {
        1 | 2 => 8,
        3 => 9, // extra charset byte
        _ => {
            return Err(Error::InvalidObjectHeader {
                msg: format!("unsupported attribute message version {}", version),
            });
        }
    };

    // Name
    if pos + name_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute name truncated".into(),
        });
    }
    let name = String::from_utf8_lossy(&data[pos..pos + name_size])
        .trim_end_matches('\0')
        .to_string();
    pos += name_size;

    // Version 1 pads name, datatype, dataspace to 8-byte boundaries
    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Datatype (may be shared — attribute flags bit 0)
    if pos + dt_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute datatype truncated".into(),
        });
    }
    let dt_data = &data[pos..pos + dt_size];
    let datatype = if (flags & 0x01) != 0 {
        let resolved = file.resolve_shared_message(dt_data, MessageType::Datatype)?;
        Datatype::parse(&resolved)?
    } else {
        Datatype::parse(dt_data)?
    };
    pos += dt_size;

    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Dataspace (may be shared — attribute flags bit 1)
    if pos + ds_size > data.len() {
        return Err(Error::InvalidObjectHeader {
            msg: "attribute dataspace truncated".into(),
        });
    }
    let ds_data = &data[pos..pos + ds_size];
    let dataspace = if (flags & 0x02) != 0 {
        let resolved = file.resolve_shared_message(ds_data, MessageType::Dataspace)?;
        Dataspace::parse(&resolved)?
    } else {
        Dataspace::parse(ds_data)?
    };
    pos += ds_size;

    if version == 1 {
        pos = (pos + 7) & !7;
    }

    // Value (remaining bytes)
    let raw_value = data[pos..].to_vec();

    Ok(Attribute {
        name,
        datatype,
        dataspace,
        raw_value,
    })
}
