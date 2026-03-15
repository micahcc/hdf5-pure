use crate::datatype::primitives::ByteOrder;
use crate::datatype::primitives::CharacterSet;
use crate::datatype::primitives::StringPadding;
use crate::datatype::types::Datatype;

impl Datatype {
    /// Create a native little-endian signed 8-bit integer type.
    pub fn native_i8() -> Self {
        Self::FixedPoint {
            size: 1,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 8,
        }
    }

    /// Create a native little-endian signed 16-bit integer type.
    pub fn native_i16() -> Self {
        Self::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 16,
        }
    }

    /// Create a native little-endian signed 32-bit integer type.
    pub fn native_i32() -> Self {
        Self::FixedPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    /// Create a native little-endian signed 64-bit integer type.
    pub fn native_i64() -> Self {
        Self::FixedPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 64,
        }
    }

    /// Create a native little-endian unsigned 8-bit integer type.
    pub fn native_u8() -> Self {
        Self::FixedPoint {
            size: 1,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        }
    }

    /// Create a native little-endian unsigned 16-bit integer type.
    pub fn native_u16() -> Self {
        Self::FixedPoint {
            size: 2,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 16,
        }
    }

    /// Create a native little-endian unsigned 32-bit integer type.
    pub fn native_u32() -> Self {
        Self::FixedPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    /// Create a native little-endian unsigned 64-bit integer type.
    pub fn native_u64() -> Self {
        Self::FixedPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 64,
        }
    }

    /// Create a native little-endian 32-bit IEEE float type.
    pub fn native_f32() -> Self {
        Self::FloatingPoint {
            size: 4,
            byte_order: ByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        }
    }

    /// Create a native little-endian 64-bit IEEE float type.
    pub fn native_f64() -> Self {
        Self::FloatingPoint {
            size: 8,
            byte_order: ByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        }
    }

    /// Create a variable-length UTF-8 string type.
    pub fn vlen_utf8_string() -> Self {
        Self::VarLen {
            element_type: Box::new(Self::FixedPoint {
                size: 1,
                signed: false,
                byte_order: ByteOrder::LittleEndian,
                bit_offset: 0,
                bit_precision: 8,
            }),
            is_string: true,
            padding: Some(StringPadding::NullTerminate),
            char_set: Some(CharacterSet::Utf8),
        }
    }

    /// Create a variable-length sequence type of the given base element type.
    pub fn vlen_sequence(element_type: Datatype) -> Self {
        Self::VarLen {
            element_type: Box::new(element_type),
            is_string: false,
            padding: None,
            char_set: None,
        }
    }

    /// Create a fixed-length ASCII string type.
    pub fn fixed_string(len: u32) -> Self {
        Self::String {
            size: len,
            padding: StringPadding::NullTerminate,
            char_set: CharacterSet::Ascii,
        }
    }
}
