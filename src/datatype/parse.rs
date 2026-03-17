use crate::datatype::members::CompoundMember;
use crate::datatype::members::EnumMember;
use crate::datatype::primitives::ByteOrder;
use crate::datatype::primitives::CharacterSet;
use crate::datatype::primitives::DatatypeClass;
use crate::datatype::primitives::ReferenceType;
use crate::datatype::primitives::StringPadding;
use crate::datatype::types::Datatype;
use crate::error::Error;
use crate::error::Result;

impl Datatype {
    /// Parse a datatype message from raw bytes.
    ///
    /// `data` should contain the full datatype message starting at the
    /// class+version word.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::InvalidDatatype {
                msg: "datatype message too short".into(),
            });
        }

        let class_and_version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let class_id = (class_and_version & 0x0F) as u8;
        let version = ((class_and_version >> 4) & 0x0F) as u8;
        let class_bits = class_and_version >> 8; // 24-bit class-specific bitfield
        let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        let props = &data[8..];
        let class = DatatypeClass::from_u8(class_id)?;

        match class {
            DatatypeClass::FixedPoint => Self::parse_fixed_point(size, class_bits, props),
            DatatypeClass::FloatingPoint => Self::parse_floating_point(size, class_bits, props),
            DatatypeClass::String => Self::parse_string(size, class_bits),
            DatatypeClass::Compound => {
                let nmembers = (class_bits & 0xFFFF) as usize;
                Self::parse_compound(size, version, nmembers, props)
            }
            DatatypeClass::Enum => Self::parse_enum(size, version, class_bits, props),
            DatatypeClass::Array => Self::parse_array(version, props),
            DatatypeClass::VarLen => Self::parse_varlen(class_bits, props),
            DatatypeClass::Opaque => Self::parse_opaque(size, class_bits, props),
            DatatypeClass::BitField => Self::parse_bitfield(size, class_bits, props),
            DatatypeClass::Reference => Self::parse_reference(class_bits),
            DatatypeClass::Time => Self::parse_time(size, props),
            DatatypeClass::Complex => Self::parse_complex(size, props),
        }
    }

    fn parse_fixed_point(size: u32, class_bits: u32, props: &[u8]) -> Result<Self> {
        if props.len() < 4 {
            return Err(Error::InvalidDatatype {
                msg: "fixed-point properties too short".into(),
            });
        }

        let byte_order = if (class_bits & 0x01) == 0 {
            ByteOrder::LittleEndian
        } else {
            ByteOrder::BigEndian
        };
        let signed = (class_bits & 0x08) != 0;

        let bit_offset = u16::from_le_bytes([props[0], props[1]]);
        let bit_precision = u16::from_le_bytes([props[2], props[3]]);

        Ok(Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
        })
    }

    fn parse_floating_point(size: u32, class_bits: u32, props: &[u8]) -> Result<Self> {
        if props.len() < 12 {
            return Err(Error::InvalidDatatype {
                msg: "floating-point properties too short".into(),
            });
        }

        let byte_order_lo = class_bits & 0x01;
        let byte_order_hi = (class_bits >> 6) & 0x01;
        let byte_order = match (byte_order_hi, byte_order_lo) {
            (0, 0) => ByteOrder::LittleEndian,
            (0, 1) => ByteOrder::BigEndian,
            (1, 1) => ByteOrder::Vax, // VAX requires both bits set (H5Odtype.c:189-201)
            _ => {
                return Err(Error::InvalidDatatype {
                    msg: "invalid floating-point byte order".into(),
                });
            }
        };

        let bit_offset = u16::from_le_bytes([props[0], props[1]]);
        let bit_precision = u16::from_le_bytes([props[2], props[3]]);
        let exponent_location = props[4];
        let exponent_size = props[5];
        let mantissa_location = props[6];
        let mantissa_size = props[7];
        let exponent_bias = u32::from_le_bytes([props[8], props[9], props[10], props[11]]);

        Ok(Datatype::FloatingPoint {
            size,
            byte_order,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        })
    }

    fn parse_string(size: u32, class_bits: u32) -> Result<Self> {
        let padding = match class_bits & 0x0F {
            0 => StringPadding::NullTerminate,
            1 => StringPadding::NullPad,
            2 => StringPadding::SpacePad,
            p => {
                return Err(Error::InvalidDatatype {
                    msg: format!("unknown string padding type {}", p),
                });
            }
        };
        let char_set = match (class_bits >> 4) & 0x0F {
            0 => CharacterSet::Ascii,
            1 => CharacterSet::Utf8,
            c => {
                return Err(Error::InvalidDatatype {
                    msg: format!("unknown string charset {}", c),
                });
            }
        };

        Ok(Datatype::String {
            size,
            padding,
            char_set,
        })
    }

    fn parse_compound(size: u32, version: u8, nmembers: usize, props: &[u8]) -> Result<Self> {
        // Compound type layout depends on version:
        //   v1: name (null-term, padded to 8), byte_offset(4), dimensionality(1),
        //       padding(3), dim_perm(4), padding(4), dim_sizes(4*4), datatype(8+)
        //   v2: name (null-term, padded to 8), byte_offset(4), datatype(8+)
        //   v3: name (null-term, NOT padded), byte_offset(1/2/4 based on type size),
        //       datatype(8+)
        let mut members = Vec::with_capacity(nmembers);
        let mut pos = 0;

        for _ in 0..nmembers {
            if pos >= props.len() {
                break;
            }

            // Read null-terminated name
            let name_start = pos;
            while pos < props.len() && props[pos] != 0 {
                pos += 1;
            }
            let name = String::from_utf8_lossy(&props[name_start..pos]).to_string();
            let name_len = pos - name_start;
            pos += 1; // skip null terminator

            // V1/v2: pad name (incl. null terminator) to 8-byte multiple from name start.
            // C formula: *pp += ((strlen + 8) / 8) * 8  (H5Odtype.c:427)
            if version <= 2 {
                pos = name_start + ((name_len + 8) / 8) * 8;
            }

            // Byte offset of this member within the compound type
            let byte_offset = match version {
                1 | 2 => {
                    if pos + 4 > props.len() {
                        return Err(Error::InvalidDatatype {
                            msg: "compound member offset truncated".into(),
                        });
                    }
                    let off = u32::from_le_bytes([
                        props[pos],
                        props[pos + 1],
                        props[pos + 2],
                        props[pos + 3],
                    ]);
                    pos += 4;
                    off
                }
                3..=5 => {
                    // Offset width: H5VM_limit_enc_size(size) = (log2(size)/8) + 1
                    let off_size = if size <= 0xFF {
                        1
                    } else if size <= 0xFFFF {
                        2
                    } else if size <= 0xFFFFFF {
                        3
                    } else {
                        4
                    };
                    if pos + off_size > props.len() {
                        return Err(Error::InvalidDatatype {
                            msg: "compound member offset truncated".into(),
                        });
                    }
                    let off = match off_size {
                        1 => props[pos] as u32,
                        2 => u16::from_le_bytes([props[pos], props[pos + 1]]) as u32,
                        3 => {
                            (props[pos] as u32)
                                | ((props[pos + 1] as u32) << 8)
                                | ((props[pos + 2] as u32) << 16)
                        }
                        _ => u32::from_le_bytes([
                            props[pos],
                            props[pos + 1],
                            props[pos + 2],
                            props[pos + 3],
                        ]),
                    };
                    pos += off_size;
                    off
                }
                _ => {
                    return Err(Error::InvalidDatatype {
                        msg: format!("unsupported compound datatype version {}", version),
                    });
                }
            };

            // V1 has extra fields: dimensionality(1) + padding(3) + dim_perm(4)
            //                       + padding(4) + dim_sizes(4*4 = 16)
            if version == 1 {
                pos += 1 + 3 + 4 + 4 + 16; // 28 bytes of v1-specific fields
            }

            // Parse the member's datatype (recursive)
            if pos + 8 > props.len() {
                return Err(Error::InvalidDatatype {
                    msg: "compound member datatype truncated".into(),
                });
            }
            let member_dt = Datatype::parse(&props[pos..])?;
            // The datatype message size: 8 (header) + properties
            let dt_total_size = Self::encoded_size(&props[pos..])?;
            pos += dt_total_size;

            members.push(CompoundMember {
                name,
                byte_offset,
                datatype: member_dt,
            });
        }

        Ok(Datatype::Compound { size, members })
    }

    /// Compute the encoded size of a datatype message starting at `data`.
    pub(crate) fn encoded_size(data: &[u8]) -> Result<usize> {
        if data.len() < 8 {
            return Err(Error::InvalidDatatype {
                msg: "datatype too short to determine size".into(),
            });
        }
        let class_and_version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let class_id = (class_and_version & 0x0F) as u8;
        let version = ((class_and_version >> 4) & 0x0F) as u8;
        let class_bits = class_and_version >> 8;
        let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        let props_size = match class_id {
            0 => 4,  // FixedPoint: bit_offset(2) + bit_precision(2)
            1 => 12, // FloatingPoint
            2 => 2,  // Time
            3 => 0,  // String (all info in class_bits)
            4 => 4,  // BitField
            5 => {
                // Opaque: tag (class_bits & 0xFF) padded to 8
                let tag_len = (class_bits & 0xFF) as usize;
                (tag_len + 7) & !7
            }
            6 => {
                // Compound: need to sum up all member sizes (recursive)
                let nmembers = (class_bits & 0xFFFF) as usize;
                let mut pos = 0;
                let props = &data[8..];
                for _ in 0..nmembers {
                    // name (with padding from name start for v1/v2)
                    let name_start = pos;
                    while pos < props.len() && props[pos] != 0 {
                        pos += 1;
                    }
                    let name_len = pos - name_start;
                    pos += 1;
                    if version < 3 {
                        pos = name_start + ((name_len + 8) / 8) * 8;
                    }
                    // byte offset
                    if version >= 3 {
                        let off_size = if size <= 0xFF {
                            1
                        } else if size <= 0xFFFF {
                            2
                        } else if size <= 0xFFFFFF {
                            3
                        } else {
                            4
                        };
                        pos += off_size;
                    } else {
                        pos += 4;
                    }
                    // v1 extra fields
                    if version == 1 {
                        pos += 28;
                    }
                    // recursive datatype
                    if pos + 8 <= props.len() {
                        let sub_size = Self::encoded_size(&props[pos..])?;
                        pos += sub_size;
                    }
                }
                pos
            }
            7 => 0, // Reference
            8 => {
                // Enum: base_type + names + values (H5Odtype.c:1715-1730)
                let nmembers = (class_bits & 0xFFFF) as usize;
                let props = &data[8..];
                // Base type
                let base_encoded = Self::encoded_size(props)?;
                let base_elem_size =
                    u32::from_le_bytes([props[4], props[5], props[6], props[7]]) as usize;
                let mut pos = base_encoded;
                // Member names
                for _ in 0..nmembers {
                    let name_start = pos;
                    while pos < props.len() && props[pos] != 0 {
                        pos += 1;
                    }
                    let name_len = pos - name_start;
                    pos += 1;
                    if version < 3 {
                        pos = name_start + ((name_len + 8) / 8) * 8;
                    }
                }
                // Member values (packed)
                pos += nmembers * base_elem_size;
                pos
            }
            9 => {
                // VarLen: just the base type (H5Odtype.c:1732-1734)
                let props = &data[8..];
                if props.len() >= 8 {
                    Self::encoded_size(props)?
                } else {
                    0
                }
            }
            10 => {
                // Array (H5Odtype.c:1736-1744)
                let props = &data[8..];
                if props.is_empty() {
                    return Err(Error::InvalidDatatype {
                        msg: "array encoded_size: no properties".into(),
                    });
                }
                let ndims = props[0] as usize;
                let mut pos = 1;
                if version < 3 {
                    pos += 3; // 3 reserved bytes
                }
                pos += 4 * ndims; // dimension sizes
                if version < 3 {
                    pos += 4 * ndims; // permutation indices
                }
                // Base type
                if pos + 8 <= props.len() {
                    pos += Self::encoded_size(&props[pos..])?;
                }
                pos
            }
            11
                // Complex: base floating-point type
                if data.len() > 8 => {
                    Self::encoded_size(&data[8..])?
                }
            _ => 0,
        };

        Ok(8 + props_size)
    }

    fn parse_enum(_size: u32, version: u8, class_bits: u32, props: &[u8]) -> Result<Self> {
        let nmembers = (class_bits & 0xFFFF) as usize;

        // First: the base datatype message (at least 8 bytes)
        if props.len() < 8 {
            return Err(Error::InvalidDatatype {
                msg: "enum base type truncated".into(),
            });
        }
        let base = Box::new(Datatype::parse(props)?);
        let base_size = Self::encoded_size(props)?;
        let mut pos = base_size;

        // Then: member names (null-terminated, padded to 8 in v1/v2, unpadded in v3)
        let mut members = Vec::with_capacity(nmembers);
        for _ in 0..nmembers {
            let name_start = pos;
            while pos < props.len() && props[pos] != 0 {
                pos += 1;
            }
            let name = String::from_utf8_lossy(&props[name_start..pos]).to_string();
            pos += 1; // skip null
            if version < 3 {
                pos = (pos + 7) & !7;
            }
            members.push(EnumMember {
                name,
                value: Vec::new(),
            });
        }

        // Then: member values (each is base_type.element_size() bytes)
        let val_size = base.element_size() as usize;
        for member in &mut members {
            if pos + val_size <= props.len() {
                member.value = props[pos..pos + val_size].to_vec();
                pos += val_size;
            }
        }

        Ok(Datatype::Enum { base, members })
    }

    fn parse_array(version: u8, props: &[u8]) -> Result<Self> {
        if props.is_empty() {
            return Err(Error::InvalidDatatype {
                msg: "array properties too short".into(),
            });
        }

        let ndims = props[0] as usize;
        let mut pos = 1;

        // Version 1 has 3 bytes of padding after ndims
        if version < 2 {
            pos += 3;
        }

        // Read dimension sizes (ndims * 4 bytes)
        if pos + ndims * 4 > props.len() {
            return Err(Error::InvalidDatatype {
                msg: "array dimension sizes truncated".into(),
            });
        }
        let mut dimensions = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let dim =
                u32::from_le_bytes([props[pos], props[pos + 1], props[pos + 2], props[pos + 3]]);
            dimensions.push(dim);
            pos += 4;
        }

        // Version 1 has ndims * 4 bytes of permutation indices (unused, must be zeros)
        if version < 2 {
            pos += ndims * 4;
        }

        // Parse the element type (recursive)
        if pos + 8 > props.len() {
            return Err(Error::InvalidDatatype {
                msg: "array element type truncated".into(),
            });
        }
        let element_type = Box::new(Datatype::parse(&props[pos..])?);

        Ok(Datatype::Array {
            element_type,
            dimensions,
        })
    }

    fn parse_varlen(class_bits: u32, props: &[u8]) -> Result<Self> {
        // class_bits: bits 0-3 = type (0=sequence, 1=string)
        //             bits 4-7 = padding type (for strings)
        //             bits 8-11 = character set (for strings)
        let vlen_type = class_bits & 0x0F;

        if vlen_type == 1 {
            // Variable-length string: element_type is the character type but
            // we represent it as a VarLen with the string's charset info embedded.
            let padding = match (class_bits >> 4) & 0x0F {
                0 => StringPadding::NullTerminate,
                1 => StringPadding::NullPad,
                2 => StringPadding::SpacePad,
                _ => StringPadding::NullTerminate,
            };
            let char_set = match (class_bits >> 8) & 0x0F {
                0 => CharacterSet::Ascii,
                1 => CharacterSet::Utf8,
                _ => CharacterSet::Ascii,
            };
            // The base type in props describes the character type but for vlen strings
            // we wrap it as String info
            let element_type = if props.len() >= 8 {
                Box::new(Datatype::parse(props)?)
            } else {
                // Fallback: assume u8 characters
                Box::new(Datatype::FixedPoint {
                    size: 1,
                    byte_order: ByteOrder::LittleEndian,
                    signed: false,
                    bit_offset: 0,
                    bit_precision: 8,
                })
            };
            Ok(Datatype::VarLen {
                element_type,
                is_string: true,
                padding: Some(padding),
                char_set: Some(char_set),
            })
        } else {
            // Variable-length sequence
            if props.len() < 8 {
                return Err(Error::InvalidDatatype {
                    msg: "vlen element type truncated".into(),
                });
            }
            let element_type = Box::new(Datatype::parse(props)?);
            Ok(Datatype::VarLen {
                element_type,
                is_string: false,
                padding: None,
                char_set: None,
            })
        }
    }

    fn parse_opaque(size: u32, class_bits: u32, props: &[u8]) -> Result<Self> {
        // class_bits contains the tag length (lower 8 bits)
        let tag_len = (class_bits & 0xFF) as usize;
        if props.len() < tag_len {
            return Err(Error::InvalidDatatype {
                msg: "opaque tag extends past properties".into(),
            });
        }
        let tag = String::from_utf8_lossy(&props[..tag_len])
            .trim_end_matches('\0')
            .to_string();
        Ok(Datatype::Opaque { size, tag })
    }

    fn parse_bitfield(size: u32, class_bits: u32, props: &[u8]) -> Result<Self> {
        if props.len() < 4 {
            return Err(Error::InvalidDatatype {
                msg: "bitfield properties too short".into(),
            });
        }
        let byte_order = if (class_bits & 0x01) == 0 {
            ByteOrder::LittleEndian
        } else {
            ByteOrder::BigEndian
        };
        let bit_offset = u16::from_le_bytes([props[0], props[1]]);
        let bit_precision = u16::from_le_bytes([props[2], props[3]]);
        Ok(Datatype::BitField {
            size,
            byte_order,
            bit_offset,
            bit_precision,
        })
    }

    fn parse_reference(class_bits: u32) -> Result<Self> {
        let ref_type = match class_bits & 0x0F {
            0 => ReferenceType::Object,
            1 => ReferenceType::DatasetRegion,
            r => {
                return Err(Error::InvalidDatatype {
                    msg: format!("unknown reference type {}", r),
                });
            }
        };
        Ok(Datatype::Reference { ref_type })
    }

    fn parse_complex(size: u32, props: &[u8]) -> Result<Self> {
        // Properties contain the base floating-point datatype message.
        if props.len() < 8 {
            return Err(Error::InvalidDatatype {
                msg: "complex base type truncated".into(),
            });
        }
        let base = Box::new(Datatype::parse(props)?);
        Ok(Datatype::Complex { size, base })
    }

    fn parse_time(size: u32, props: &[u8]) -> Result<Self> {
        if props.len() < 2 {
            return Err(Error::InvalidDatatype {
                msg: "time properties too short".into(),
            });
        }
        let bit_precision = u16::from_le_bytes([props[0], props[1]]);
        Ok(Datatype::Time {
            size,
            bit_precision,
        })
    }
}
