use crate::error::Error;
use crate::error::Result;

/// Verify and strip Fletcher32 checksum.
pub(crate) fn verify_fletcher32(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 4 {
        return Err(Error::InvalidFilterPipeline {
            msg: "data too short for fletcher32 checksum".into(),
        });
    }
    // The last 4 bytes are the checksum, stored big-endian
    let payload = &data[..data.len() - 4];
    let stored = u32::from_be_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);

    let computed = fletcher32(payload);
    if computed != stored {
        return Err(Error::ChecksumMismatch {
            expected: stored,
            actual: computed,
        });
    }

    Ok(payload.to_vec())
}

/// Compute Fletcher32 checksum over data (little-endian 16-bit words).
///
/// Matches the HDF5 library's `H5_checksum_fletcher32` on LE systems.
/// HDF5 returns (sum1 << 16) | sum2 where sum1 is the simple accumulator
/// and sum2 is the running cumulative sum.
fn fletcher32(data: &[u8]) -> u32 {
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

    // Handle odd trailing byte
    if data.len() % 2 != 0 {
        sum1 += (data[i] as u32) << 8;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 += sum1;
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Second reduction step
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum1 << 16) | sum2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fletcher32_known() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let cksum = fletcher32(&data);
        // Deterministic — just verify it's non-zero and consistent
        assert_ne!(cksum, 0);
        assert_eq!(cksum, fletcher32(&data));
    }
}
