use crate::error::Error;
use crate::error::Result;
use crate::writer::types::ChunkFilter;

/// Apply filters in forward direction (compression) to chunk data.
pub(crate) fn apply_filters_forward(
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
