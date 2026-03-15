//! Round-trip tests: build HDF5 files in memory with the writer, then read them
//! back with the reader and verify the data matches what was written.

use hdf5_io::Datatype;
use hdf5_io::File;
use hdf5_io::writer::FileWriter;
use hdf5_io::writer::StorageLayout;

// ── simple_contiguous_v2 ──

#[test]
fn write_simple_contiguous() {
    let mut w = FileWriter::new();
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("data", Datatype::native_f64(), &[4], data);
    ds.add_attribute("units", Datatype::fixed_string(5), &[], b"m/s\0\0".to_vec());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    assert_eq!(root.members().unwrap(), vec!["data"]);

    let ds = root.dataset("data").unwrap();
    match ds.datatype().unwrap() {
        Datatype::FloatingPoint { size, .. } => assert_eq!(size, 8),
        other => panic!("expected FloatingPoint, got {:?}", other),
    }
    assert_eq!(ds.shape().unwrap(), vec![4]);

    let raw = ds.read_raw().unwrap();
    let values: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);

    let attrs = ds.attributes().unwrap();
    let units = attrs.iter().find(|a| a.name == "units").unwrap();
    let val_str = String::from_utf8_lossy(&units.raw_value);
    assert!(val_str.starts_with("m/s"));
}

// ── nested_groups_v2 ──

#[test]
fn write_nested_groups() {
    let mut w = FileWriter::new();
    let root = w.root_mut();
    let g1 = root.add_group("group1");

    // /group1/subgroup/temps (f32)
    let sub = g1.add_group("subgroup");
    let temps_data: Vec<u8> = [20.5f32, 21.0, 19.8]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    sub.add_dataset("temps", Datatype::native_f32(), &[3], temps_data);

    // /group1/ids (u8)
    g1.add_dataset("ids", Datatype::native_u8(), &[5], vec![10, 20, 30, 40, 50]);

    let bytes = w.to_bytes().unwrap();
    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    assert_eq!(root.members().unwrap(), vec!["group1"]);

    let g1 = root.group("group1").unwrap();
    let g1_members = g1.members().unwrap();
    assert!(g1_members.contains(&"ids".to_string()));
    assert!(g1_members.contains(&"subgroup".to_string()));

    let sub = g1.group("subgroup").unwrap();
    assert_eq!(sub.members().unwrap(), vec!["temps"]);

    // Read /group1/ids
    let ds = g1.dataset("ids").unwrap();
    assert_eq!(ds.read_raw().unwrap(), vec![10, 20, 30, 40, 50]);

    // Read /group1/subgroup/temps
    let ds = sub.dataset("temps").unwrap();
    let raw = ds.read_raw().unwrap();
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![20.5, 21.0, 19.8]);
}

// ── compact_v2 ──

#[test]
fn write_compact() {
    let mut w = FileWriter::new();
    let data: Vec<u8> = [100i16, 200, 300, 400]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let ds = w
        .root_mut()
        .add_dataset("small", Datatype::native_i16(), &[4], data);
    ds.set_layout(StorageLayout::Compact);
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("small").unwrap();

    match ds.datatype().unwrap() {
        Datatype::FixedPoint { size, signed, .. } => {
            assert_eq!(size, 2);
            assert!(signed);
        }
        other => panic!("expected FixedPoint, got {:?}", other),
    }
    assert_eq!(ds.shape().unwrap(), vec![4]);

    let raw = ds.read_raw().unwrap();
    let values: Vec<i16> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![100, 200, 300, 400]);
}

// ── compound ──

#[test]
fn write_compound() {
    let dt = Datatype::Compound {
        size: 12,
        members: vec![
            hdf5_io::datatype::CompoundMember {
                name: "id".to_string(),
                byte_offset: 0,
                datatype: Datatype::native_i32(),
            },
            hdf5_io::datatype::CompoundMember {
                name: "x".to_string(),
                byte_offset: 4,
                datatype: Datatype::native_f32(),
            },
            hdf5_io::datatype::CompoundMember {
                name: "y".to_string(),
                byte_offset: 8,
                datatype: Datatype::native_f32(),
            },
        ],
    };

    let mut data = Vec::new();
    let records: Vec<(i32, f32, f32)> = vec![(1, 1.0, 2.0), (2, 3.0, 4.0), (3, 5.0, 6.0)];
    for (id, x, y) in &records {
        data.extend_from_slice(&id.to_le_bytes());
        data.extend_from_slice(&x.to_le_bytes());
        data.extend_from_slice(&y.to_le_bytes());
    }

    let mut w = FileWriter::new();
    w.root_mut().add_dataset("points", dt, &[3], data);
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("points").unwrap();
    let dt = ds.datatype().unwrap();
    match &dt {
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
        other => panic!("expected Compound, got {:?}", other),
    }

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 36);
    for (i, record) in raw.chunks_exact(12).enumerate() {
        let id = i32::from_le_bytes(record[0..4].try_into().unwrap());
        let x = f32::from_le_bytes(record[4..8].try_into().unwrap());
        let y = f32::from_le_bytes(record[8..12].try_into().unwrap());
        assert_eq!(id, (i + 1) as i32);
        assert_eq!(x, (2 * i + 1) as f32);
        assert_eq!(y, (2 * i + 2) as f32);
    }
}

// ── enum ──

#[test]
fn write_enum() {
    let dt = Datatype::Enum {
        base: Box::new(Datatype::native_i8()),
        members: vec![
            hdf5_io::datatype::EnumMember {
                name: "RED".to_string(),
                value: vec![0],
            },
            hdf5_io::datatype::EnumMember {
                name: "GREEN".to_string(),
                value: vec![1],
            },
            hdf5_io::datatype::EnumMember {
                name: "BLUE".to_string(),
                value: vec![2],
            },
        ],
    };

    let data = vec![0u8, 1, 2, 1, 0];
    let mut w = FileWriter::new();
    w.root_mut().add_dataset("colors", dt, &[5], data);
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("colors").unwrap();

    match &ds.datatype().unwrap() {
        Datatype::Enum { base, members } => {
            match base.as_ref() {
                Datatype::FixedPoint { size, signed, .. } => {
                    assert_eq!(*size, 1);
                    assert!(signed);
                }
                other => panic!("expected FixedPoint base, got {:?}", other),
            }
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "RED");
            assert_eq!(members[0].value, vec![0]);
            assert_eq!(members[1].name, "GREEN");
            assert_eq!(members[1].value, vec![1]);
            assert_eq!(members[2].name, "BLUE");
            assert_eq!(members[2].value, vec![2]);
        }
        other => panic!("expected Enum, got {:?}", other),
    }

    assert_eq!(ds.read_raw().unwrap(), vec![0, 1, 2, 1, 0]);
}

// ── array ──

#[test]
fn write_array() {
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

    match &ds.datatype().unwrap() {
        Datatype::Array {
            element_type,
            dimensions,
        } => {
            assert_eq!(dimensions, &[3]);
            assert_eq!(element_type.element_size(), 4);
        }
        other => panic!("expected Array, got {:?}", other),
    }

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 48);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, (1..=12).collect::<Vec<i32>>());
}

// ── complex ──

#[test]
fn write_complex() {
    let dt = Datatype::Complex {
        size: 16,
        base: Box::new(Datatype::native_f64()),
    };

    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.0, -5.0];
    let data: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut w = FileWriter::new();
    w.root_mut()
        .add_dataset("complex_data", dt, &[4], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("complex_data").unwrap();

    match &ds.datatype().unwrap() {
        Datatype::Complex { size, base } => {
            assert_eq!(*size, 16);
            assert_eq!(base.element_size(), 8);
        }
        other => panic!("expected Complex, got {:?}", other),
    }

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 64);
    let parsed: Vec<(f64, f64)> = raw
        .chunks_exact(16)
        .map(|c| {
            let re = f64::from_le_bytes(c[0..8].try_into().unwrap());
            let im = f64::from_le_bytes(c[8..16].try_into().unwrap());
            (re, im)
        })
        .collect();
    assert_eq!(parsed[0], (1.0, 2.0));
    assert_eq!(parsed[1], (3.0, 4.0));
    assert_eq!(parsed[2], (-1.0, 0.0));
    assert_eq!(parsed[3], (0.0, -5.0));
}

// ── fill_value ──

#[test]
fn write_fill_value() {
    let data: Vec<u8> = [10i32, 20, 30, 40, -999, -999]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let mut w = FileWriter::new();
    let ds = w
        .root_mut()
        .add_dataset("filled", Datatype::native_i32(), &[6], data);
    ds.set_fill_value((-999i32).to_le_bytes().to_vec());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("filled").unwrap();

    let fv = ds.fill_value().unwrap();
    assert!(fv.defined);
    let val_bytes = fv.value.expect("expected fill value bytes");
    assert_eq!(val_bytes.len(), 4);
    let fill_val = i32::from_le_bytes(val_bytes.try_into().unwrap());
    assert_eq!(fill_val, -999);

    let raw = ds.read_raw().unwrap();
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![10, 20, 30, 40, -999, -999]);
}

// ── big_endian ──

#[test]
fn write_big_endian() {
    let dt = Datatype::FixedPoint {
        size: 4,
        byte_order: hdf5_io::datatype::ByteOrder::BigEndian,
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

    let raw = ds.read_raw().unwrap();
    let be_first = i32::from_be_bytes(raw[..4].try_into().unwrap());
    assert_eq!(be_first, 1);

    let native = ds.read_native().unwrap();
    let native_values: Vec<i32> = native
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(native_values, vec![1, 256, 65536, -1, 1000000, 0]);
}

// ── nil_messages (many attributes) ──

#[test]
fn write_nil_messages() {
    let mut w = FileWriter::new();
    let data: Vec<u8> = (1..=4i32).flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("data", Datatype::native_i32(), &[4], data);

    for i in 0..12 {
        let name = format!("attribute_{}", i);
        let val = ((i + 1) * 10 as i32).to_le_bytes().to_vec();
        ds.add_attribute(&name, Datatype::native_i32(), &[], val);
    }
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("data").unwrap();

    let raw = ds.read_raw().unwrap();
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1, 2, 3, 4]);

    let attrs = ds.attributes().unwrap();
    assert_eq!(attrs.len(), 12);
}

// ── compound_complex_members (compound with enum + array) ──

#[test]
fn write_compound_complex_members() {
    // Enum type: RED=0, GREEN=1, BLUE=2 (int32 base)
    let enum_type = Datatype::Enum {
        base: Box::new(Datatype::native_i32()),
        members: vec![
            hdf5_io::datatype::EnumMember {
                name: "RED".to_string(),
                value: 0i32.to_le_bytes().to_vec(),
            },
            hdf5_io::datatype::EnumMember {
                name: "GREEN".to_string(),
                value: 1i32.to_le_bytes().to_vec(),
            },
            hdf5_io::datatype::EnumMember {
                name: "BLUE".to_string(),
                value: 2i32.to_le_bytes().to_vec(),
            },
        ],
    };

    // Array type: int32[3]
    let array_type = Datatype::Array {
        element_type: Box::new(Datatype::native_i32()),
        dimensions: vec![3],
    };

    // Compound: { color: enum(4), coords: int32[3](12), id: int32(4) } = 20 bytes
    let dt = Datatype::Compound {
        size: 20,
        members: vec![
            hdf5_io::datatype::CompoundMember {
                name: "color".to_string(),
                byte_offset: 0,
                datatype: enum_type,
            },
            hdf5_io::datatype::CompoundMember {
                name: "coords".to_string(),
                byte_offset: 4,
                datatype: array_type,
            },
            hdf5_io::datatype::CompoundMember {
                name: "id".to_string(),
                byte_offset: 16,
                datatype: Datatype::native_i32(),
            },
        ],
    };

    // 3 records
    let mut data = vec![0u8; 60];
    // Record 0: color=RED(0), coords=[10,20,30], id=100
    data[0..4].copy_from_slice(&0i32.to_le_bytes());
    data[4..8].copy_from_slice(&10i32.to_le_bytes());
    data[8..12].copy_from_slice(&20i32.to_le_bytes());
    data[12..16].copy_from_slice(&30i32.to_le_bytes());
    data[16..20].copy_from_slice(&100i32.to_le_bytes());
    // Record 1: color=GREEN(1), coords=[40,50,60], id=200
    data[20..24].copy_from_slice(&1i32.to_le_bytes());
    data[24..28].copy_from_slice(&40i32.to_le_bytes());
    data[28..32].copy_from_slice(&50i32.to_le_bytes());
    data[32..36].copy_from_slice(&60i32.to_le_bytes());
    data[36..40].copy_from_slice(&200i32.to_le_bytes());
    // Record 2: color=BLUE(2), coords=[70,80,90], id=300
    data[40..44].copy_from_slice(&2i32.to_le_bytes());
    data[44..48].copy_from_slice(&70i32.to_le_bytes());
    data[48..52].copy_from_slice(&80i32.to_le_bytes());
    data[52..56].copy_from_slice(&90i32.to_le_bytes());
    data[56..60].copy_from_slice(&300i32.to_le_bytes());

    let mut w = FileWriter::new();
    w.root_mut().add_dataset("records", dt, &[3], data.clone());
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let ds = file.root_group().unwrap().dataset("records").unwrap();

    match &ds.datatype().unwrap() {
        Datatype::Compound { size, members } => {
            assert_eq!(*size, 20);
            assert_eq!(members.len(), 3);
            assert_eq!(members[0].name, "color");
            assert_eq!(members[1].name, "coords");
            assert_eq!(members[2].name, "id");
        }
        other => panic!("expected Compound, got {:?}", other),
    }

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 60);
    assert_eq!(raw, data);
}

// ── shuffle_deflate_v3 (single chunk, shuffle + deflate) ──

#[test]
fn write_shuffle_deflate() {
    use hdf5_io::writer::ChunkFilter;

    let mut w = FileWriter::new();
    let expected: Vec<f32> = (0..20).map(|i| i as f32 * 1.5).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("shuffled", Datatype::native_f32(), &[20], data);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![20],
        filters: vec![ChunkFilter::Shuffle, ChunkFilter::Deflate(6)],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("shuffled").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![20]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 80);
    let values: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── fletcher32 (single chunk, fletcher32 checksum) ──

#[test]
fn write_fletcher32() {
    use hdf5_io::writer::ChunkFilter;

    let mut w = FileWriter::new();
    let expected: Vec<i32> = (1..=10).map(|i| i * 100).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("checksummed", Datatype::native_i32(), &[10], data);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![10],
        filters: vec![ChunkFilter::Fletcher32],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("checksummed").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![10]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 40);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── implicit_chunks (unfiltered multi-chunk) ──

#[test]
fn write_implicit_chunks() {
    let mut w = FileWriter::new();
    let expected = vec![1.1f64, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("implicit", Datatype::native_f64(), &[8], data);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![4],
        filters: vec![],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("implicit").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![8]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 64);
    let values: Vec<f64> = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── edge_chunks (2D, non-aligned chunk boundaries) ──

#[test]
fn write_edge_chunks() {
    let mut w = FileWriter::new();
    let expected: Vec<i32> = (0..35).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("edge", Datatype::native_i32(), &[7, 5], data);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![4, 3],
        filters: vec![],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("edge").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![7, 5]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 140);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── chunked_deflate_v3 (multi-chunk with deflate, fixed array index) ──

#[test]
fn write_chunked_deflate() {
    use hdf5_io::writer::ChunkFilter;

    let mut w = FileWriter::new();
    let expected: Vec<i32> = (0..100).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("compressed", Datatype::native_i32(), &[10, 10], data);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![5, 5],
        filters: vec![ChunkFilter::Deflate(6)],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("compressed").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![10, 10]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 400);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── empty_chunked (chunked dataset with no data written - all zeros) ──

#[test]
fn write_empty_chunked() {
    let mut w = FileWriter::new();
    let data = vec![0u8; 40]; // 10 * 4 bytes, all zeros
    let ds = w
        .root_mut()
        .add_dataset("empty", Datatype::native_i32(), &[10], data);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![5],
        filters: vec![],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("empty").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![10]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 40);
    assert!(raw.iter().all(|&b| b == 0));
}

// ── extensible_array (EA index, small, all fit in index block) ──

#[test]
fn write_extensible_array() {
    let mut w = FileWriter::new();
    let expected: Vec<i32> = (1..=15).map(|i| i * 100).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("extarray", Datatype::native_i32(), &[15], data);
    ds.set_max_dims(&[u64::MAX]);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![5],
        filters: vec![],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("extarray").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![15]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 60);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── ea_large (EA index, 25 chunks, needs data blocks) ──

#[test]
fn write_ea_large() {
    let mut w = FileWriter::new();
    let expected: Vec<i32> = (0..100).map(|i| i * 10).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("large_ea", Datatype::native_i32(), &[100], data);
    ds.set_max_dims(&[u64::MAX]);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![4],
        filters: vec![],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("large_ea").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![100]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 400);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── btree_v2_chunks (2D, both dims unlimited, BT2 index) ──

#[test]
fn write_btree_v2_chunks() {
    let mut w = FileWriter::new();
    let expected: Vec<i32> = (0..24).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("bt2chunked", Datatype::native_i32(), &[6, 4], data);
    ds.set_max_dims(&[u64::MAX, u64::MAX]);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![3, 2],
        filters: vec![],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("bt2chunked").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![6, 4]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 96);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── btree_v2_deep (20x10, chunk 1x1, 200 chunks) ──

#[test]
fn write_btree_v2_deep() {
    let mut w = FileWriter::new();
    let expected: Vec<i32> = (0..200).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("deep", Datatype::native_i32(), &[20, 10], data);
    ds.set_max_dims(&[u64::MAX, u64::MAX]);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![1, 1],
        filters: vec![],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("deep").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![20, 10]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 800);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── btree_v2_filtered (2D, both dims unlimited, deflate, BT2 index) ──

#[test]
fn write_btree_v2_filtered() {
    use hdf5_io::writer::ChunkFilter;

    let mut w = FileWriter::new();
    let expected: Vec<i32> = (0..24).collect();
    let data: Vec<u8> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("filtered", Datatype::native_i32(), &[6, 4], data);
    ds.set_max_dims(&[u64::MAX, u64::MAX]);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![3, 2],
        filters: vec![ChunkFilter::Deflate(6)],
    });
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("filtered").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![6, 4]);

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 96);
    let values: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, expected);
}

// ── chunked_btree_v1 ──

#[test]
fn write_chunked_btree_v1() {
    use hdf5_io::writer::ChunkFilter;

    let mut w = FileWriter::new();
    let vals: Vec<i32> = (1..=12).map(|i| i * 10).collect();
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ds = w
        .root_mut()
        .add_dataset("chunked_v3", Datatype::native_i32(), &[12], data);
    ds.set_layout(StorageLayout::Chunked {
        chunk_dims: vec![4],
        filters: vec![ChunkFilter::Deflate(4)],
    });
    ds.set_layout_version(3);
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    assert_eq!(root.members().unwrap(), vec!["chunked_v3"]);

    let ds = root.dataset("chunked_v3").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![12]);
    match ds.datatype().unwrap() {
        Datatype::FixedPoint { size, .. } => assert_eq!(size, 4),
        other => panic!("expected FixedPoint, got {:?}", other),
    }

    let raw = ds.read_raw().unwrap();
    assert_eq!(raw.len(), 48);
    let read_vals: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected_vals: Vec<i32> = (1..=12).map(|i| i * 10).collect();
    assert_eq!(read_vals, expected_vals);
}

// ── vlen_strings ──

#[test]
fn write_vlen_strings() {
    let mut w = FileWriter::new();
    let strings = vec!["hello", "world", "HDF5", "variable-length"];
    let elements: Vec<Vec<u8>> = strings.iter().map(|s| s.as_bytes().to_vec()).collect();
    w.root_mut()
        .add_vlen_dataset("names", Datatype::vlen_utf8_string(), &[4], elements);
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("names").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4]);

    let read_strings = ds.read_vlen_strings().unwrap();
    assert_eq!(read_strings, vec!["hello", "world", "HDF5", "variable-length"]);
}

// ── vlen_sequence ──

#[test]
fn write_vlen_sequence() {
    let mut w = FileWriter::new();
    let seq0: Vec<u8> = vec![10i32, 20].iter().flat_map(|v| v.to_le_bytes()).collect();
    let seq1: Vec<u8> = vec![100i32, 200, 300, 400].iter().flat_map(|v| v.to_le_bytes()).collect();
    let seq2: Vec<u8> = vec![42i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let elements = vec![seq0, seq1, seq2];
    w.root_mut().add_vlen_dataset(
        "sequences",
        Datatype::vlen_sequence(Datatype::native_i32()),
        &[3],
        elements,
    );
    let bytes = w.to_bytes().unwrap();

    let file = File::from_bytes(bytes.into_boxed_slice()).unwrap();
    let root = file.root_group().unwrap();
    let ds = root.dataset("sequences").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![3]);

    let vlen_data = ds.read_vlen().unwrap();
    assert_eq!(vlen_data.len(), 3);

    let s0: Vec<i32> = vlen_data[0].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(s0, vec![10, 20]);
    let s1: Vec<i32> = vlen_data[1].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(s1, vec![100, 200, 300, 400]);
    let s2: Vec<i32> = vlen_data[2].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(s2, vec![42]);
}
