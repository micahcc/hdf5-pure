use crate::error::Error;
use crate::error::Result;

/// HDF5 datatype class IDs (from the on-disk encoding).
///
/// Reference: H5Tpublic.h `H5T_class_t`, and HDF5 File Format Spec section III.D.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DatatypeClass {
    FixedPoint = 0,
    FloatingPoint = 1,
    Time = 2,
    String = 3,
    BitField = 4,
    Opaque = 5,
    Compound = 6,
    Reference = 7,
    Enum = 8,
    VarLen = 9,
    Array = 10,
    Complex = 11,
}

impl DatatypeClass {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::FixedPoint),
            1 => Ok(Self::FloatingPoint),
            2 => Ok(Self::Time),
            3 => Ok(Self::String),
            4 => Ok(Self::BitField),
            5 => Ok(Self::Opaque),
            6 => Ok(Self::Compound),
            7 => Ok(Self::Reference),
            8 => Ok(Self::Enum),
            9 => Ok(Self::VarLen),
            10 => Ok(Self::Array),
            11 => Ok(Self::Complex),
            _ => Err(Error::UnsupportedDatatypeClass { class: v }),
        }
    }
}

/// Byte order for fixed-point and floating-point types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
    /// VAX mixed-endian (rare, floating-point only).
    Vax,
}

/// String padding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPadding {
    NullTerminate,
    NullPad,
    SpacePad,
}

/// String character set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacterSet {
    Ascii,
    Utf8,
}

/// Reference type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceType {
    Object,
    DatasetRegion,
}
