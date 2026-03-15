use crate::error::Error;
use crate::error::Result;

const NBIT_ATOMIC: u32 = 1;
const NBIT_ARRAY: u32 = 2;
const NBIT_COMPOUND: u32 = 3;
const NBIT_NOOPTYPE: u32 = 4;
const NBIT_ORDER_LE: u32 = 0;

struct NbitAtomicParms {
    size: u32,
    order: u32,
    precision: u32,
    offset: u32,
}

struct NbitState<'a> {
    buffer: &'a [u8],
    j: usize,
    buf_len: usize, // bits remaining in current byte
}

impl<'a> NbitState<'a> {
    fn new(buffer: &'a [u8]) -> Self {
        NbitState {
            buffer,
            j: 0,
            buf_len: 8,
        }
    }

    fn next_byte(&mut self) {
        self.j += 1;
        self.buf_len = 8;
    }

    fn cur(&self) -> u8 {
        if self.j < self.buffer.len() {
            self.buffer[self.j]
        } else {
            0
        }
    }
}

fn nbit_decompress_one_byte(
    data: &mut [u8],
    data_offset: usize,
    k: usize,
    begin_i: usize,
    end_i: usize,
    state: &mut NbitState,
    p: &NbitAtomicParms,
    datatype_len: u32,
) {
    let (dat_len, dat_offset);

    if begin_i != end_i {
        if k == begin_i {
            dat_len = 8 - ((datatype_len - p.precision - p.offset) % 8) as usize;
            dat_offset = 0;
        } else if k == end_i {
            dat_len = (8 - (p.offset % 8)) as usize;
            dat_offset = 8 - dat_len;
        } else {
            dat_len = 8;
            dat_offset = 0;
        }
    } else {
        dat_offset = (p.offset % 8) as usize;
        dat_len = p.precision as usize;
    }

    let val = state.cur();

    if state.buf_len > dat_len {
        data[data_offset + k] = (((val >> (state.buf_len - dat_len)) as u32
            & ((1u32 << dat_len) - 1))
            << dat_offset) as u8;
        state.buf_len -= dat_len;
    } else {
        data[data_offset + k] = ((((val as u32) & ((1u32 << state.buf_len) - 1))
            << (dat_len - state.buf_len))
            << dat_offset) as u8;
        let remaining = dat_len - state.buf_len;
        state.next_byte();
        if remaining == 0 {
            return;
        }
        let val2 = state.cur();
        data[data_offset + k] |= ((((val2 >> (state.buf_len - remaining)) as u32)
            & ((1u32 << remaining) - 1))
            << dat_offset) as u8;
        state.buf_len -= remaining;
    }
}

fn nbit_decompress_one_atomic(
    data: &mut [u8],
    data_offset: usize,
    state: &mut NbitState,
    p: &NbitAtomicParms,
) {
    let datatype_len = p.size * 8;

    if p.order == NBIT_ORDER_LE {
        let begin_i = if (p.precision + p.offset) % 8 != 0 {
            ((p.precision + p.offset) / 8) as usize
        } else {
            ((p.precision + p.offset) / 8 - 1) as usize
        };
        let end_i = (p.offset / 8) as usize;

        let mut k = begin_i as isize;
        while k >= end_i as isize {
            nbit_decompress_one_byte(
                data,
                data_offset,
                k as usize,
                begin_i,
                end_i,
                state,
                p,
                datatype_len,
            );
            k -= 1;
        }
    } else {
        let begin_i = ((datatype_len - p.precision - p.offset) / 8) as usize;
        let end_i = if p.offset % 8 != 0 {
            ((datatype_len - p.offset) / 8) as usize
        } else {
            ((datatype_len - p.offset) / 8 - 1) as usize
        };

        for k in begin_i..=end_i {
            nbit_decompress_one_byte(data, data_offset, k, begin_i, end_i, state, p, datatype_len);
        }
    }
}

fn nbit_decompress_one_nooptype(
    data: &mut [u8],
    data_offset: usize,
    state: &mut NbitState,
    size: usize,
) {
    for i in 0..size {
        let val = state.cur();
        data[data_offset + i] =
            (((val as u32) & ((1u32 << state.buf_len) - 1)) << (8 - state.buf_len)) as u8;
        let remaining = 8 - state.buf_len;
        state.next_byte();
        if remaining == 0 {
            continue;
        }
        let val2 = state.cur();
        data[data_offset + i] |=
            ((val2 >> (state.buf_len - remaining)) as u32 & ((1u32 << remaining) - 1)) as u8;
        state.buf_len -= remaining;
    }
}

fn nbit_decompress_one_array(
    data: &mut [u8],
    data_offset: usize,
    state: &mut NbitState,
    parms: &[u32],
    parms_index: &mut usize,
) -> Result<()> {
    let total_size = parms[*parms_index] as usize;
    *parms_index += 1;
    let base_class = parms[*parms_index];
    *parms_index += 1;

    match base_class {
        NBIT_ATOMIC => {
            let p = NbitAtomicParms {
                size: parms[*parms_index],
                order: parms[*parms_index + 1],
                precision: parms[*parms_index + 2],
                offset: parms[*parms_index + 3],
            };
            *parms_index += 4;
            let n = total_size / p.size as usize;
            for i in 0..n {
                nbit_decompress_one_atomic(data, data_offset + i * p.size as usize, state, &p);
            }
        }
        NBIT_ARRAY => {
            let base_size = parms[*parms_index] as usize;
            let n = total_size / base_size;
            let begin_index = *parms_index;
            for i in 0..n {
                nbit_decompress_one_array(
                    data,
                    data_offset + i * base_size,
                    state,
                    parms,
                    parms_index,
                )?;
                *parms_index = begin_index;
            }
        }
        NBIT_COMPOUND => {
            let base_size = parms[*parms_index] as usize;
            let n = total_size / base_size;
            let begin_index = *parms_index;
            for i in 0..n {
                nbit_decompress_one_compound(
                    data,
                    data_offset + i * base_size,
                    state,
                    parms,
                    parms_index,
                )?;
                *parms_index = begin_index;
            }
        }
        NBIT_NOOPTYPE => {
            *parms_index += 1; // skip size
            nbit_decompress_one_nooptype(data, data_offset, state, total_size);
        }
        _ => {
            return Err(Error::DecompressionError {
                msg: format!("nbit: unknown class {}", base_class),
            });
        }
    }
    Ok(())
}

fn nbit_decompress_one_compound(
    data: &mut [u8],
    data_offset: usize,
    state: &mut NbitState,
    parms: &[u32],
    parms_index: &mut usize,
) -> Result<()> {
    let _size = parms[*parms_index] as usize;
    *parms_index += 1;
    let nmembers = parms[*parms_index] as usize;
    *parms_index += 1;

    for _ in 0..nmembers {
        let member_offset = parms[*parms_index] as usize;
        *parms_index += 1;
        let member_class = parms[*parms_index];
        *parms_index += 1;

        match member_class {
            NBIT_ATOMIC => {
                let p = NbitAtomicParms {
                    size: parms[*parms_index],
                    order: parms[*parms_index + 1],
                    precision: parms[*parms_index + 2],
                    offset: parms[*parms_index + 3],
                };
                *parms_index += 4;
                nbit_decompress_one_atomic(data, data_offset + member_offset, state, &p);
            }
            NBIT_ARRAY => {
                nbit_decompress_one_array(
                    data,
                    data_offset + member_offset,
                    state,
                    parms,
                    parms_index,
                )?;
            }
            NBIT_COMPOUND => {
                nbit_decompress_one_compound(
                    data,
                    data_offset + member_offset,
                    state,
                    parms,
                    parms_index,
                )?;
            }
            NBIT_NOOPTYPE => {
                let member_size = parms[*parms_index] as usize;
                *parms_index += 1;
                nbit_decompress_one_nooptype(data, data_offset + member_offset, state, member_size);
            }
            _ => {
                return Err(Error::DecompressionError {
                    msg: format!("nbit: unknown member class {}", member_class),
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn decompress_nbit(data: &[u8], cd_values: &[u32]) -> Result<Vec<u8>> {
    if cd_values.len() < 5 {
        return Err(Error::DecompressionError {
            msg: "nbit: cd_values too short".into(),
        });
    }

    // cd_values[1]: need_not_compress flag
    if cd_values[1] != 0 {
        return Ok(data.to_vec());
    }

    let d_nelmts = cd_values[2] as usize;
    let elem_size = cd_values[4] as usize;
    let size_out = d_nelmts * elem_size;

    let mut output = vec![0u8; size_out];
    let mut state = NbitState::new(data);

    // parms start at cd_values[3]
    let parms = &cd_values[3..];

    match parms[0] {
        NBIT_ATOMIC => {
            let p = NbitAtomicParms {
                size: parms[1],
                order: parms[2],
                precision: parms[3],
                offset: parms[4],
            };
            if p.precision > p.size * 8 || (p.precision + p.offset) > p.size * 8 {
                return Err(Error::DecompressionError {
                    msg: "nbit: invalid precision/offset".into(),
                });
            }
            for i in 0..d_nelmts {
                nbit_decompress_one_atomic(&mut output, i * p.size as usize, &mut state, &p);
            }
        }
        NBIT_ARRAY => {
            let size = parms[1] as usize;
            let mut parms_index: usize = 1; // relative to parms (which is cd_values[3..])
            for i in 0..d_nelmts {
                nbit_decompress_one_array(
                    &mut output,
                    i * size,
                    &mut state,
                    parms,
                    &mut parms_index,
                )?;
                parms_index = 1;
            }
        }
        NBIT_COMPOUND => {
            let size = parms[1] as usize;
            let mut parms_index: usize = 1;
            for i in 0..d_nelmts {
                nbit_decompress_one_compound(
                    &mut output,
                    i * size,
                    &mut state,
                    parms,
                    &mut parms_index,
                )?;
                parms_index = 1;
            }
        }
        _ => {
            return Err(Error::DecompressionError {
                msg: format!("nbit: unsupported top-level class {}", parms[0]),
            });
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nbit_need_not_compress() {
        // cd_values[1] = 1 means no compression needed
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let cd = vec![8, 1, 2, 1, 2, 0, 16, 0]; // need_not_compress=1
        let result = decompress_nbit(&data, &cd).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn nbit_atomic_u8_4bit() {
        // 4 uint8 values with 4-bit precision, offset 0, LE
        // Values: 0x0A, 0x0B, 0x0C, 0x0D (only low 4 bits significant)
        // Packed: 0xAB, 0xCD (4 bits each, MSB-first in buffer)
        let packed = vec![0xAB, 0xCD];
        let cd = vec![8, 0, 4, 1, 1, 0, 4, 0]; // ATOMIC, size=1, order=LE, prec=4, off=0
        let result = decompress_nbit(&packed, &cd).unwrap();
        assert_eq!(result, vec![0x0A, 0x0B, 0x0C, 0x0D]);
    }
}
