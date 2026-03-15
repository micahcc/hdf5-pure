use crate::error::Error;
use crate::error::Result;

const SO_CLS_INTEGER: u32 = 0;
const SO_CLS_FLOAT: u32 = 1;
const SO_PARM_SCALETYPE: usize = 0;
const SO_PARM_SCALEFACTOR: usize = 1;
const SO_PARM_NELMTS: usize = 2;
const SO_PARM_CLASS: usize = 3;
const SO_PARM_SIZE: usize = 4;
const SO_PARM_SIGN: usize = 5;
const SO_PARM_ORDER: usize = 6;
const SO_PARM_FILAVAIL: usize = 7;

const SO_FLOAT_DSCALE: u32 = 0;
const SO_FILL_DEFINED: u32 = 1;

fn so_decompress_one_byte(
    data: &mut [u8],
    data_offset: usize,
    k: usize,
    begin_i: usize,
    buffer: &[u8],
    j: &mut usize,
    bits_to_fill: &mut usize,
    minbits: u32,
    dtype_len: u32,
) {
    if *j >= buffer.len() {
        return;
    }

    let val = buffer[*j];
    let bits_to_copy = if k == begin_i {
        8 - ((dtype_len - minbits) % 8) as usize
    } else {
        8
    };

    if *bits_to_fill > bits_to_copy {
        data[data_offset + k] =
            ((val >> (*bits_to_fill - bits_to_copy)) as u32 & ((1u32 << bits_to_copy) - 1)) as u8;
        *bits_to_fill -= bits_to_copy;
    } else {
        data[data_offset + k] = (((val as u32) & ((1u32 << *bits_to_fill) - 1))
            << (bits_to_copy - *bits_to_fill)) as u8;
        let remaining = bits_to_copy - *bits_to_fill;
        *j += 1;
        *bits_to_fill = 8;
        if remaining == 0 {
            return;
        }
        if *j >= buffer.len() {
            return;
        }
        let val2 = buffer[*j];
        data[data_offset + k] |=
            ((val2 >> (*bits_to_fill - remaining)) as u32 & ((1u32 << remaining) - 1)) as u8;
        *bits_to_fill -= remaining;
    }
}

fn so_decompress_one_atomic(
    data: &mut [u8],
    data_offset: usize,
    buffer: &[u8],
    j: &mut usize,
    bits_to_fill: &mut usize,
    size: u32,
    minbits: u32,
    mem_order_le: bool,
) {
    let dtype_len = size * 8;

    if mem_order_le {
        let begin_i = (size as usize) - 1 - ((dtype_len - minbits) / 8) as usize;
        let mut k = begin_i as isize;
        while k >= 0 {
            so_decompress_one_byte(
                data,
                data_offset,
                k as usize,
                begin_i,
                buffer,
                j,
                bits_to_fill,
                minbits,
                dtype_len,
            );
            k -= 1;
        }
    } else {
        let begin_i = ((dtype_len - minbits) / 8) as usize;
        for k in begin_i..size as usize {
            so_decompress_one_byte(
                data,
                data_offset,
                k,
                begin_i,
                buffer,
                j,
                bits_to_fill,
                minbits,
                dtype_len,
            );
        }
    }
}

pub(crate) fn decompress_scaleoffset(data: &[u8], cd_values: &[u32]) -> Result<Vec<u8>> {
    if cd_values.len() < 8 {
        return Err(Error::DecompressionError {
            msg: "scaleoffset: cd_values too short".into(),
        });
    }

    let scale_type = cd_values[SO_PARM_SCALETYPE];
    let scale_factor = cd_values[SO_PARM_SCALEFACTOR] as i32;
    let d_nelmts = cd_values[SO_PARM_NELMTS] as usize;
    let dtype_class = cd_values[SO_PARM_CLASS];
    let dtype_size = cd_values[SO_PARM_SIZE] as usize;
    let dtype_sign = cd_values[SO_PARM_SIGN];
    let dtype_order = cd_values[SO_PARM_ORDER];
    let filavail = cd_values[SO_PARM_FILAVAIL];

    if data.len() < 5 {
        return Err(Error::DecompressionError {
            msg: "scaleoffset: buffer too short".into(),
        });
    }

    // Read minbits (4 bytes LE)
    let minbits = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);

    // Read minval_size and minval
    let stored_minval_size = data[4] as usize;
    let minval_size = stored_minval_size.min(8);
    let mut minval: u64 = 0;
    if data.len() < 5 + minval_size {
        return Err(Error::DecompressionError {
            msg: "scaleoffset: buffer too short for minval".into(),
        });
    }
    for i in 0..minval_size {
        minval |= (data[5 + i] as u64) << (i * 8);
    }

    let buf_offset: usize = 21;
    let size_out = d_nelmts * dtype_size;

    // Special case: minbits == full precision — raw copy
    if minbits == (dtype_size as u32) * 8 {
        if data.len() < buf_offset + size_out {
            return Err(Error::DecompressionError {
                msg: "scaleoffset: buffer too short for full copy".into(),
            });
        }
        let mut output = data[buf_offset..buf_offset + size_out].to_vec();

        // Convert byte order if needed (data is stored in dataset order, we want native LE)
        if dtype_order != 0 {
            // dtype_order != LE means data is BE, need to swap to native LE
            for chunk in output.chunks_exact_mut(dtype_size) {
                chunk.reverse();
            }
        }
        return Ok(output);
    }

    let mut output = vec![0u8; size_out];

    // Decompress packed data
    if minbits != 0 {
        let packed = if buf_offset < data.len() {
            &data[buf_offset..]
        } else {
            &[] as &[u8]
        };
        let mut j: usize = 0;
        let mut bits_to_fill: usize = 8;

        // Scale-offset always uses LE memory order for the bit unpacking
        // (it stores data in native order during compression)
        let mem_order_le = true;

        for i in 0..d_nelmts {
            so_decompress_one_atomic(
                &mut output,
                i * dtype_size,
                packed,
                &mut j,
                &mut bits_to_fill,
                dtype_size as u32,
                minbits,
                mem_order_le,
            );
        }
    }

    // Post-decompress: add minval back
    if dtype_class == SO_CLS_INTEGER {
        so_postdecompress_int(
            &mut output,
            d_nelmts,
            dtype_size,
            dtype_sign,
            filavail,
            cd_values,
            minbits,
            minval,
        );
    } else if dtype_class == SO_CLS_FLOAT && scale_type == SO_FLOAT_DSCALE {
        so_postdecompress_float(
            &mut output,
            d_nelmts,
            dtype_size,
            filavail,
            cd_values,
            minbits,
            minval,
            scale_factor as f64,
        );
    }

    // Convert byte order if dataset is BE and we're on LE
    if dtype_order != 0 {
        for chunk in output.chunks_exact_mut(dtype_size) {
            chunk.reverse();
        }
    }

    Ok(output)
}

fn so_postdecompress_int(
    data: &mut [u8],
    d_nelmts: usize,
    dtype_size: usize,
    dtype_sign: u32,
    filavail: u32,
    cd_values: &[u32],
    minbits: u32,
    minval: u64,
) {
    let is_signed = dtype_sign != 0;

    for i in 0..d_nelmts {
        let off = i * dtype_size;

        // Read element as u64 LE
        let mut elem: u64 = 0;
        for b in 0..dtype_size {
            elem |= (data[off + b] as u64) << (b * 8);
        }

        // Check for fill value sentinel
        if filavail == SO_FILL_DEFINED && minbits < 64 {
            let sentinel = (1u64 << minbits) - 1;
            if elem == sentinel {
                // Reconstruct fill value from cd_values[8..]
                let mut filval: u64 = 0;
                let filval_start = SO_PARM_FILAVAIL + 1; // cd_values[8]
                for b in 0..dtype_size.min(cd_values.len().saturating_sub(filval_start) * 4) {
                    let cd_idx = filval_start + b / 4;
                    if cd_idx >= cd_values.len() {
                        break;
                    }
                    let byte_in_cd = b % 4;
                    filval |= (((cd_values[cd_idx] >> (byte_in_cd * 8)) & 0xFF) as u64) << (b * 8);
                }
                for b in 0..dtype_size {
                    data[off + b] = (filval >> (b * 8)) as u8;
                }
                continue;
            }
        }

        // Add minval
        let result = if is_signed {
            let sminval = minval as i64;
            let selem = elem as i64;
            (selem.wrapping_add(sminval)) as u64
        } else {
            elem.wrapping_add(minval)
        };

        for b in 0..dtype_size {
            data[off + b] = (result >> (b * 8)) as u8;
        }
    }
}

fn so_postdecompress_float(
    data: &mut [u8],
    d_nelmts: usize,
    dtype_size: usize,
    filavail: u32,
    cd_values: &[u32],
    minbits: u32,
    minval: u64,
    d_val: f64,
) {
    // minval is the bit pattern of the float minimum, stored as u64 LE
    // For float: reinterpret as f32; for double: reinterpret as f64
    let min_float: f64 = if dtype_size == 4 {
        f32::from_le_bytes((minval as u32).to_le_bytes()) as f64
    } else {
        f64::from_le_bytes(minval.to_le_bytes())
    };

    let pow10 = 10.0f64.powf(d_val);

    for i in 0..d_nelmts {
        let off = i * dtype_size;

        // Read the integer value (unpacked differences are stored as integers)
        let mut ival: u64 = 0;
        for b in 0..dtype_size {
            ival |= (data[off + b] as u64) << (b * 8);
        }

        // Check fill value sentinel
        if filavail == SO_FILL_DEFINED && minbits < 64 {
            let sentinel = (1u64 << minbits) - 1;
            if ival == sentinel {
                let mut filval: u64 = 0;
                let filval_start = SO_PARM_FILAVAIL + 1;
                for b in 0..dtype_size.min(cd_values.len().saturating_sub(filval_start) * 4) {
                    let cd_idx = filval_start + b / 4;
                    if cd_idx >= cd_values.len() {
                        break;
                    }
                    let byte_in_cd = b % 4;
                    filval |= (((cd_values[cd_idx] >> (byte_in_cd * 8)) & 0xFF) as u64) << (b * 8);
                }
                for b in 0..dtype_size {
                    data[off + b] = (filval >> (b * 8)) as u8;
                }
                continue;
            }
        }

        // D-scale postprocess: float_val = (integer_val as float) / 10^D + min
        let result: f64 = if dtype_size == 4 {
            let int_as_signed = ival as i32;
            (int_as_signed as f64) / pow10 + min_float
        } else {
            let int_as_signed = ival as i64;
            (int_as_signed as f64) / pow10 + min_float
        };

        // Write back
        if dtype_size == 4 {
            let bytes = (result as f32).to_le_bytes();
            data[off..off + 4].copy_from_slice(&bytes);
        } else {
            let bytes = result.to_le_bytes();
            data[off..off + 8].copy_from_slice(&bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaleoffset_int_simple() {
        // Manually construct a scaleoffset buffer:
        // minbits=3 (values 0..7 fit in 3 bits), minval=1000 as u64
        // d_nelmts=4, dtype_size=4, dtype_class=INTEGER(0), sign=1(signed), order=0(LE)
        // cd_values: [scaletype=2, scalefactor=0, nelmts=4, class=0, size=4, sign=1, order=0, filavail=0]
        let cd = vec![2u32, 0, 4, 0, 4, 1, 0, 0];

        // Buffer: minbits(4) + sizeof_ull(1) + minval(8) + padding(8) + packed data
        let mut buf = vec![0u8; 21];
        // minbits = 3
        buf[0] = 3;
        buf[1] = 0;
        buf[2] = 0;
        buf[3] = 0;
        // sizeof_ull = 8
        buf[4] = 8;
        // minval = 1000 = 0x3E8
        buf[5] = 0xE8;
        buf[6] = 0x03;
        buf[7] = 0;
        buf[8] = 0;
        buf[9] = 0;
        buf[10] = 0;
        buf[11] = 0;
        buf[12] = 0;
        // Packed data: values 0,1,2,3 in 3-bit fields, MSB-first
        // 000 001 010 011 = 0b00000101_0011xxxx = 0x05 0x30
        buf.push(0b00000101);
        buf.push(0b00110000);

        let result = decompress_scaleoffset(&buf, &cd).unwrap();
        assert_eq!(result.len(), 16); // 4 * 4 bytes
        let values: Vec<i32> = result
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(values, vec![1000, 1001, 1002, 1003]);
    }
}
