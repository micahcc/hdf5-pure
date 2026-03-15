use crate::error::Error;
use crate::error::Result;

/// Decompress LZF compressed data.
///
/// LZF is a simple LZ77 variant by Marc Lehmann, used by h5py (filter ID 32000).
/// Format: stream of literal runs and back-references.
///   - If byte >> 5 == 0: literal run, length = byte + 1, followed by that many bytes.
///   - Otherwise: back-reference. length = (byte >> 5) + 2.
///     If length == 9 (all 3 high bits set), read one more byte and add to length + 2.
///     Offset = ((byte & 0x1f) << 8) | next_byte, plus 1.
pub(crate) fn decompress_lzf(data: &[u8], out_size_hint: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(if out_size_hint > 0 {
        out_size_hint
    } else {
        data.len() * 2
    });
    let mut ip = 0; // input position

    while ip < data.len() {
        let ctrl = data[ip] as usize;
        ip += 1;

        if ctrl < (1 << 5) {
            // Literal run: ctrl + 1 bytes
            let len = ctrl + 1;
            if ip + len > data.len() {
                return Err(Error::DecompressionError {
                    msg: format!("lzf: literal overflows input at ip={}, len={}", ip, len),
                });
            }
            output.extend_from_slice(&data[ip..ip + len]);
            ip += len;
        } else {
            // Back-reference
            let mut len = (ctrl >> 5) + 2;
            if len == 9 {
                // Extended length
                if ip >= data.len() {
                    return Err(Error::DecompressionError {
                        msg: "lzf: truncated extended length".into(),
                    });
                }
                len += data[ip] as usize;
                ip += 1;
            }

            if ip >= data.len() {
                return Err(Error::DecompressionError {
                    msg: "lzf: truncated back-reference offset".into(),
                });
            }
            let offset = ((ctrl & 0x1f) << 8) | (data[ip] as usize);
            ip += 1;

            // offset is 0-based from current output position - 1
            let ref_pos = output.len().wrapping_sub(offset + 1);
            if ref_pos >= output.len() {
                return Err(Error::DecompressionError {
                    msg: format!(
                        "lzf: back-reference out of bounds: ref_pos={}, output_len={}",
                        ref_pos,
                        output.len()
                    ),
                });
            }

            // Copy byte-by-byte since source and dest may overlap
            for i in 0..len {
                let b = output[ref_pos + i];
                output.push(b);
            }
        }
    }

    Ok(output)
}
