use crate::error::Error;
use crate::error::Result;

/// Decompress DEFLATE (zlib) compressed data.
pub(crate) fn decompress_deflate(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read;

    use flate2::read::ZlibDecoder;

    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::new();
    decoder
        .read_to_end(&mut output)
        .map_err(|e| Error::DecompressionError {
            msg: format!("deflate: {}", e),
        })?;
    Ok(output)
}
