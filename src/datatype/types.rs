use crate::datatype::members::CompoundMember;
use crate::datatype::members::EnumMember;
use crate::datatype::primitives::ByteOrder;
use crate::datatype::primitives::CharacterSet;
use crate::datatype::primitives::ReferenceType;
use crate::datatype::primitives::StringPadding;

/// A decoded HDF5 datatype message.
///
/// ## On-disk layout (datatype message in object header)
///
/// ```text
/// Byte 0-3: class_and_version (4 bits class, 4 bits version, 24 bits class-specific bitfield)
/// Byte 4-7: size (4 bytes LE, total size of one element in bytes)
/// Byte 8+:  class-specific properties
/// ```
#[derive(Debug, Clone)]
pub enum Datatype {
    /// Fixed-point (integer) type.
    FixedPoint {
        size: u32,
        byte_order: ByteOrder,
        signed: bool,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// IEEE floating-point type.
    FloatingPoint {
        size: u32,
        byte_order: ByteOrder,
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    },
    /// Fixed-length string.
    String {
        size: u32,
        padding: StringPadding,
        char_set: CharacterSet,
    },
    /// Compound type (struct-like).
    Compound {
        size: u32,
        members: Vec<CompoundMember>,
    },
    /// Enumeration type.
    Enum {
        base: Box<Datatype>,
        members: Vec<EnumMember>,
    },
    /// Array type.
    Array {
        element_type: Box<Datatype>,
        dimensions: Vec<u32>,
    },
    /// Variable-length type.
    VarLen {
        element_type: Box<Datatype>,
        /// True if this is a vlen string (class_bits type=1), false for sequence.
        is_string: bool,
        /// String padding (only for vlen strings).
        padding: Option<StringPadding>,
        /// Character set (only for vlen strings).
        char_set: Option<CharacterSet>,
    },
    /// Opaque type.
    Opaque { size: u32, tag: String },
    /// Bitfield type.
    BitField {
        size: u32,
        byte_order: ByteOrder,
        bit_offset: u16,
        bit_precision: u16,
    },
    /// Reference type.
    Reference { ref_type: ReferenceType },
    /// Time type (rarely used).
    Time { size: u32, bit_precision: u16 },
    /// Complex number type (HDF5 2.0+).
    ///
    /// On-disk: two consecutive values of the base floating-point type
    /// (real part, then imaginary part). `size` = 2 * base element size.
    Complex { size: u32, base: Box<Datatype> },
}

impl Datatype {
    /// The size of one element of this type in bytes.
    pub fn element_size(&self) -> u32 {
        match self {
            Self::FixedPoint { size, .. } => *size,
            Self::FloatingPoint { size, .. } => *size,
            Self::String { size, .. } => *size,
            Self::Compound { size, .. } => *size,
            Self::Enum { base, .. } => base.element_size(),
            Self::Array {
                element_type,
                dimensions,
            } => {
                let count: u32 = dimensions.iter().product();
                element_type.element_size() * count
            }
            Self::VarLen { .. } => {
                // HDF5 vlen is stored as a (size, pointer) pair in memory,
                // but on disk it's in the global heap. The "size" field in the
                // datatype message is typically 4+offset_size.
                // We'll return the on-disk element size from the message.
                // This should be overridden by the actual message size field.
                16
            }
            Self::Opaque { size, .. } => *size,
            Self::BitField { size, .. } => *size,
            Self::Reference { ref_type } => match ref_type {
                ReferenceType::Object => 8,
                ReferenceType::DatasetRegion => 12,
            },
            Self::Time { size, .. } => *size,
            Self::Complex { size, .. } => *size,
        }
    }
}
