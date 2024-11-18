use crate::{bail, Error, Result};
use core::str;
use std::io::{BufRead, BufReader, Read};

#[derive(Debug, Default)]
pub struct Ppm {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

// TODO: Error type
// TODO: Impl would break if image data starts with ASCII whitespace bytes

const ASCII_WHITESPACE: [u8; 4] = [b' ', b'\n', b'\r', b'\t'];

fn is_ascii_whitespace(byte: &u8) -> bool {
    ASCII_WHITESPACE.contains(byte)
}

fn token_range(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut iter = bytes.iter();
    let start = iter.position(|b| !is_ascii_whitespace(b))?;
    let end = start + 1 + iter.position(is_ascii_whitespace)?;
    Some((start, end))
}

impl Ppm {
    pub const FORMAT: &str = "P6";
    pub const COLOR_CHANNEL_COUNT: usize = 3;

    pub fn new() -> Self {
        Self::default()
    }
    pub fn from_reader(reader: impl Read) -> Result<Self> {
        Self::from_buffered_reader(BufReader::new(reader))
    }
    pub fn from_buffered_reader(mut reader: impl BufRead) -> Result<Self> {
        let mut result = Self::new();

        let buffer = reader.fill_buf()?;

        let (token_start, token_end) =
            token_range(buffer).ok_or(Error::Generic("missing format token"))?;
        let format = str::from_utf8(&buffer[token_start..token_end])?;
        if format != Self::FORMAT {
            println!("{format}");
            bail!(Error::Generic("unsupported ppm format"));
        }

        reader.consume(token_end);
        let buffer = reader.fill_buf()?;

        let (token_start, token_end) =
            token_range(buffer).ok_or(Error::Generic("missing width"))?;
        result.width = str::from_utf8(&buffer[token_start..token_end])?.parse()?;

        reader.consume(token_end);
        let buffer = reader.fill_buf()?;

        let (token_start, token_end) =
            token_range(buffer).ok_or(Error::Generic("missing height"))?;
        result.height = str::from_utf8(&buffer[token_start..token_end])?.parse()?;

        reader.consume(token_end);
        let buffer = reader.fill_buf()?;

        let (token_start, token_end) =
            token_range(buffer).ok_or(Error::Generic("missing max value"))?;
        let max_value: usize = str::from_utf8(&buffer[token_start..token_end])?.parse()?;

        // TODO: Error not assert
        debug_assert_eq!(max_value, 255);
        reader.consume(token_end);
        let buffer = reader.fill_buf()?;

        let trim_end_count = buffer
            .iter()
            .position(|b| !is_ascii_whitespace(b))
            .unwrap_or_default();
        reader.consume(trim_end_count);

        let _ = reader.read_to_end(&mut result.data)?;

        if result.data.len() != result.width * result.height * Self::COLOR_CHANNEL_COUNT {
            bail!(Error::Generic("invalid size for data segment"));
        }

        Ok(result)
    }
}
