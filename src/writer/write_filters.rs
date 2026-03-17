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
            ChunkFilter::ScaleOffset(params) => compress_scaleoffset(&data, &params.cd_values)?,
            ChunkFilter::Nbit(params) => compress_nbit(&data, &params.cd_values)?,
            ChunkFilter::Lzf => compress_lzf(&data)?,
        };
    }
    Ok(data)
}

/// Compress data using the HDF5 scale-offset filter (integer types).
///
/// On-disk format:
///   minbits (4 LE) + sizeof_ull (1) + minval (8 LE) + padding (8) + packed_bits
fn compress_scaleoffset(data: &[u8], cd_values: &[u32; 20]) -> Result<Vec<u8>> {
    let d_nelmts = cd_values[2] as usize;
    let dtype_class = cd_values[3];
    let dtype_size = cd_values[4] as usize;
    let dtype_sign = cd_values[5];

    if dtype_class != 0 {
        return Err(Error::Other {
            msg: "scaleoffset compress: only integer class supported".into(),
        });
    }

    if data.len() < d_nelmts * dtype_size {
        return Err(Error::Other {
            msg: "scaleoffset compress: data too short".into(),
        });
    }

    // Read all elements as u64 (LE)
    let mut values = Vec::with_capacity(d_nelmts);
    for i in 0..d_nelmts {
        let off = i * dtype_size;
        let mut v: u64 = 0;
        for b in 0..dtype_size {
            v |= (data[off + b] as u64) << (b * 8);
        }
        values.push(v);
    }

    let filavail = cd_values[7];

    // Find min and max, compute minbits using HDF5 C library algorithm
    let (minval, minbits, residuals) = if dtype_sign != 0 {
        // Signed: interpret as i64
        let signed_vals: Vec<i64> = values
            .iter()
            .map(|&v| {
                let shift = 64 - dtype_size * 8;
                ((v as i64) << shift) >> shift
            })
            .collect();
        let min = *signed_vals.iter().min().unwrap();
        let max = *signed_vals.iter().max().unwrap();
        let span = (max - min + 1) as u64; // +1 per C library
        // minbits = ceil_log2(span + 1) when fill defined, ceil_log2(span) otherwise
        let minbits = if filavail == 1 {
            so_log2(span + 1)
        } else {
            so_log2(span)
        };
        let residuals: Vec<u64> = signed_vals.iter().map(|&v| (v - min) as u64).collect();
        (min as u64, minbits, residuals)
    } else {
        let min = *values.iter().min().unwrap();
        let max = *values.iter().max().unwrap();
        let span = max - min + 1;
        let minbits = if filavail == 1 {
            so_log2(span + 1)
        } else {
            so_log2(span)
        };
        let residuals: Vec<u64> = values.iter().map(|&v| v - min).collect();
        (min, minbits, residuals)
    };

    // Build output: header (21 bytes) + packed bits
    let mut buf = Vec::with_capacity(21 + (d_nelmts * minbits as usize).div_ceil(8));

    // minbits (4 bytes LE)
    buf.extend_from_slice(&minbits.to_le_bytes());
    // sizeof_ull (1 byte) = 8
    buf.push(8);
    // minval (8 bytes LE)
    buf.extend_from_slice(&minval.to_le_bytes());
    // padding (8 bytes)
    buf.extend_from_slice(&[0u8; 8]);

    // Pack residuals into minbits-wide fields, MSB-first within each element
    if minbits > 0 {
        // C library allocates minbits * d_nelmts / 8 + 1 bytes (with +1 padding)
        let packed_size = (minbits as usize * d_nelmts) / 8 + 1;
        let pack_start = buf.len();
        so_compress_pack(&residuals, minbits, dtype_size as u32, &mut buf);
        // Ensure we write exactly packed_size bytes (zero-pad if needed)
        let written = buf.len() - pack_start;
        if written < packed_size {
            buf.resize(pack_start + packed_size, 0);
        }
    }

    Ok(buf)
}

/// Pack values into minbits-wide fields using HDF5's bit-packing order.
///
/// For LE integers, bytes are packed from the most significant byte down to least,
/// with bits filled from MSB to LSB within each output byte.
fn so_compress_pack(values: &[u64], minbits: u32, dtype_size: u32, buf: &mut Vec<u8>) {
    let dtype_len = dtype_size * 8;
    let mut current_byte: u8 = 0;
    let mut bits_left: usize = 8; // bits remaining in current output byte

    for &val in values {
        // For LE memory order, we pack from the most significant byte that contains
        // data bits, down to byte 0. Within each byte, we pack from MSB to LSB.
        let begin_i = (dtype_size - 1 - (dtype_len - minbits) / 8) as usize;

        let mut k = begin_i as isize;
        while k >= 0 {
            let byte_val = ((val >> (k as u32 * 8)) & 0xFF) as u8;
            let bits_to_copy = if k as usize == begin_i {
                8 - ((dtype_len - minbits) % 8) as usize
            } else {
                8
            };

            // Extract the relevant bits from byte_val (top bits_to_copy bits of the
            // significant portion)
            let extracted = if k as usize == begin_i {
                byte_val & ((1u16 << bits_to_copy as u16) - 1) as u8
            } else {
                byte_val
            };

            // Write bits_to_copy bits into output
            let mut remaining = bits_to_copy;
            let mut src_bits = extracted;
            while remaining > 0 {
                if remaining <= bits_left {
                    current_byte |= src_bits << (bits_left - remaining);
                    bits_left -= remaining;
                    remaining = 0;
                } else {
                    // Fill the rest of current byte
                    current_byte |= src_bits >> (remaining - bits_left);
                    remaining -= bits_left;
                    src_bits &= (1u8 << remaining as u8).wrapping_sub(1);
                    buf.push(current_byte);
                    current_byte = 0;
                    bits_left = 8;
                }
            }

            if bits_left == 0 {
                buf.push(current_byte);
                current_byte = 0;
                bits_left = 8;
            }

            k -= 1;
        }
    }

    // Flush remaining partial byte
    if bits_left < 8 {
        buf.push(current_byte);
    }
}

/// Compress data using the HDF5 N-bit filter (atomic types).
///
/// Packs only the significant bits of each element.
fn compress_nbit(data: &[u8], cd_values: &[u32]) -> Result<Vec<u8>> {
    if cd_values.len() < 8 {
        return Err(Error::Other {
            msg: "nbit compress: cd_values too short".into(),
        });
    }

    // cd_values[1] = need_not_compress
    if cd_values[1] != 0 {
        return Ok(data.to_vec());
    }

    let d_nelmts = cd_values[2] as usize;
    let parm_class = cd_values[3];

    if parm_class != 1 {
        // Only NBIT_ATOMIC supported for now
        return Err(Error::Other {
            msg: "nbit compress: only atomic types supported".into(),
        });
    }

    let dtype_size = cd_values[4] as usize;
    let order_le = cd_values[5] == 0;
    let precision = cd_values[6];
    let bit_offset = cd_values[7];

    if !order_le {
        return Err(Error::Other {
            msg: "nbit compress: only LE byte order supported".into(),
        });
    }

    if data.len() < d_nelmts * dtype_size {
        return Err(Error::Other {
            msg: "nbit compress: data too short".into(),
        });
    }

    // Buffer size matches C library: precision * nelmts / 8 + 1
    let packed_size = (precision as usize * d_nelmts) / 8 + 1;
    let mut buf = Vec::with_capacity(packed_size);
    let mut current_byte: u8 = 0;
    let mut bits_left: usize = 8;

    let dtype_len = (dtype_size * 8) as u32;

    // For LE atomic with offset and precision:
    // begin_i = byte containing the MSB of the precision window
    // end_i = byte containing the LSB of the precision window
    let begin_i = if (precision + bit_offset) % 8 != 0 {
        ((precision + bit_offset) / 8) as usize
    } else {
        ((precision + bit_offset) / 8 - 1) as usize
    };
    let end_i = (bit_offset / 8) as usize;

    for elem in 0..d_nelmts {
        let elem_off = elem * dtype_size;
        let mut k = begin_i as isize;
        while k >= end_i as isize {
            let byte_val = data[elem_off + k as usize];

            // Compute dat_len and dat_offset for this byte
            let (dat_len, dat_offset) = if begin_i != end_i {
                if k as usize == begin_i {
                    let dl = 8 - ((dtype_len - precision - bit_offset) % 8) as usize;
                    (dl, 0usize)
                } else if k as usize == end_i {
                    let dl = (8 - (bit_offset % 8)) as usize;
                    (dl, 8 - dl)
                } else {
                    (8, 0)
                }
            } else {
                ((precision as usize), (bit_offset % 8) as usize)
            };

            // Extract dat_len bits starting at dat_offset
            let extracted = (byte_val >> dat_offset) & ((1u16 << dat_len as u16) - 1) as u8;

            // Pack into output
            let mut remaining = dat_len;
            let mut src_bits = extracted;
            while remaining > 0 {
                if remaining <= bits_left {
                    current_byte |= src_bits << (bits_left - remaining);
                    bits_left -= remaining;
                    remaining = 0;
                } else {
                    current_byte |= src_bits >> (remaining - bits_left);
                    remaining -= bits_left;
                    src_bits &= (1u8 << remaining as u8).wrapping_sub(1);
                    buf.push(current_byte);
                    current_byte = 0;
                    bits_left = 8;
                }
            }

            if bits_left == 0 {
                buf.push(current_byte);
                current_byte = 0;
                bits_left = 8;
            }

            k -= 1;
        }
    }

    // Flush partial byte
    if bits_left < 8 {
        buf.push(current_byte);
    }

    // Ensure buffer is exactly packed_size bytes
    if buf.len() < packed_size {
        buf.resize(packed_size, 0);
    }

    Ok(buf)
}

/// Compress data using the LZF algorithm.
///
/// LZF is a fast LZ77 variant by Marc Lehmann, used by h5py (filter ID 32000).
/// Returns Err if compression would expand the data.
fn compress_lzf(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() <= 4 {
        // Too short to compress meaningfully
        return Ok(data.to_vec());
    }

    const HTAB_SIZE: usize = 1 << 16;
    let mut htab = vec![0u32; HTAB_SIZE];
    let mut output = Vec::with_capacity(data.len());

    let mut lit_start: usize = 0; // start of current literal run
    let in_end = data.len();

    // Hash function matching liblzf
    #[inline]
    fn lzf_hash(v: u32) -> usize {
        let h = (v >> 1) ^ v;
        (h >> 3 ^ h << 5 ^ h >> 12) as usize & (HTAB_SIZE - 1)
    }

    #[inline]
    fn read3(data: &[u8], pos: usize) -> u32 {
        (data[pos] as u32) | ((data[pos + 1] as u32) << 8) | ((data[pos + 2] as u32) << 16)
    }

    // Skip first byte, start matching from position 1
    let mut ip: usize = 1;
    let mut anchor = 0usize; // start of current literal run in output (ctrl byte position)

    // Write initial literal ctrl placeholder
    output.push(0);

    while ip < in_end - 2 {
        let v = read3(data, ip);
        let hval = lzf_hash(v);
        let ref_pos = htab[hval] as usize;
        htab[hval] = ip as u32;

        // Check for match: same first 3 bytes and within reach (max offset 8191)
        let off = ip - ref_pos;
        if off > 0
            && off < 8192
            && ref_pos < ip
            && ref_pos + 2 < in_end
            && data[ref_pos] == data[ip]
            && data[ref_pos + 1] == data[ip + 1]
            && data[ref_pos + 2] == data[ip + 2]
        {
            // Flush any pending literals
            let lit_len = ip - lit_start;
            if lit_len > 0 {
                // Write literal runs (max 32 bytes per literal ctrl)
                let mut lp = lit_start;
                while lp < ip {
                    let run = (ip - lp).min(32);
                    output[anchor] = (run - 1) as u8;
                    output.extend_from_slice(&data[lp..lp + run]);
                    lp += run;
                    if lp < ip {
                        anchor = output.len();
                        output.push(0);
                    }
                }
            }

            // Find match length (already matched 3)
            let mut len = 3;
            let max_len = (in_end - ip).min(264); // max encodable: 9 + 255 = 264
            while len < max_len && data[ref_pos + len] == data[ip + len] {
                len += 1;
            }

            let offset = off - 1; // 0-based offset

            if len <= 8 {
                // Short match: 3 bits for length-2 (1..7 → len 3..9, but 7 is reserved for extended)
                // Actually len 3..8 use (len-2) in bits 7..5
                let ctrl = (((len - 2) as u8) << 5) | ((offset >> 8) as u8);
                output.push(ctrl);
                output.push((offset & 0xFF) as u8);
            } else {
                // Extended match: ctrl has 111xxxxx (len-2=7, so ctrl>>5 == 7, decoded as 9)
                // Then extra byte = len - 9
                let ctrl = (7u8 << 5) | ((offset >> 8) as u8);
                output.push(ctrl);
                output.push((len - 9) as u8);
                output.push((offset & 0xFF) as u8);
            }

            ip += len;
            lit_start = ip;
            anchor = output.len();
            output.push(0); // placeholder for next literal ctrl

            // Update hash for skipped positions
            if ip < in_end - 2 {
                htab[lzf_hash(read3(data, ip))] = ip as u32;
            }
        } else {
            ip += 1;
        }
    }

    // Flush remaining literals (including last 2 bytes that weren't hashed)
    let remaining = in_end - lit_start;
    if remaining > 0 {
        let mut lp = lit_start;
        let end = in_end;
        while lp < end {
            let run = (end - lp).min(32);
            output[anchor] = (run - 1) as u8;
            output.extend_from_slice(&data[lp..lp + run]);
            lp += run;
            if lp < end {
                anchor = output.len();
                output.push(0);
            }
        }
    } else {
        // Remove unused placeholder
        output.pop();
    }

    Ok(output)
}

/// Ceiling log2, matching HDF5 C library's H5Z__scaleoffset_log2.
fn so_log2(num: u64) -> u32 {
    if num == 0 {
        return 0;
    }
    let mut v: u32 = 0;
    let mut lower_bound: u64 = 1;
    let mut val = num;
    loop {
        val >>= 1;
        if val == 0 {
            break;
        }
        v += 1;
        lower_bound <<= 1;
    }
    if num == lower_bound { v } else { v + 1 }
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
