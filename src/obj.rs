use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::io::{BufRead, BufReader, Read};

use crate::bail;
use crate::math::{Vector2, Vector3};
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    GeometricVertex,
    TextureVertex,
    // VertexNormal,
    // ParameterSpaceVertex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownKeywordError(pub String);

impl UnknownKeywordError {
    pub fn new(keyword: impl Into<String>) -> Self {
        Self(keyword.into())
    }
}

impl fmt::Display for UnknownKeywordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown keyword '{}'", self.0)
    }
}

impl StdError for UnknownKeywordError {}

impl TryFrom<&str> for Keyword {
    type Error = UnknownKeywordError;

    fn try_from(token: &str) -> Result<Self, Self::Error> {
        use Keyword::*;

        let keyword = match token {
            "v" => GeometricVertex,
            "vt" => TextureVertex,
            unknown => return Err(UnknownKeywordError::new(unknown)),
        };

        Ok(keyword)
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Obj {
    pub geometric_vertices: Vec<Vector3<f32>>,
    pub texture_vertices: Vec<Vector2<f32>>,
    pub unknown_keywords: HashMap<String, usize>,
}

impl Obj {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn from_reader(reader: impl Read) -> Result<Self> {
        Self::from_buffered_reader(BufReader::new(reader))
    }
    pub fn from_buffered_reader(reader: impl BufRead) -> Result<Self> {
        let mut result = Self::new();

        for line in reader.lines() {
            let line = line?;
            let mut tokens = line.split_whitespace();

            let token = match tokens.next() {
                Some(token) => token,
                None => continue,
            };

            let keyword = match Keyword::try_from(token) {
                Err(UnknownKeywordError(unknown)) => {
                    *result.unknown_keywords.entry(unknown).or_default() += 1;
                    continue;
                }
                Ok(keyword) => keyword,
            };

            match keyword {
                Keyword::GeometricVertex => {
                    result.geometric_vertices.push(parse_vector3(tokens)?);
                }
                Keyword::TextureVertex => {
                    result.texture_vertices.push(parse_vector2(tokens)?);
                }
            }
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenCountError {
    expected: usize,
    got: usize,
}

impl TokenCountError {
    pub fn new(expected: usize, got: usize) -> Self {
        Self { expected, got }
    }
}

impl fmt::Display for TokenCountError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unexpected number of tokens (expected {}, got {})",
            self.expected, self.got
        )
    }
}

impl StdError for TokenCountError {}

pub fn parse_vector3<'a>(mut tokens: impl Iterator<Item = &'a str>) -> Result<Vector3<f32>> {
    let mut result = [0f32; 3];
    for (idx, item) in result.iter_mut().enumerate() {
        *item = tokens.next().ok_or(TokenCountError::new(3, idx))?.parse()?;
    }

    if tokens.next().is_some() {
        // TODO: Special case for 4 point vertices?
        bail!(TokenCountError::new(3, 4));
    }

    Ok(Vector3::from_array(result))
}

pub fn parse_vector2<'a>(mut tokens: impl Iterator<Item = &'a str>) -> Result<Vector2<f32>> {
    let mut result = [0f32; 2];
    for (idx, item) in result.iter_mut().enumerate() {
        *item = tokens.next().ok_or(TokenCountError::new(2, idx))?.parse()?;
    }

    if tokens.next().is_some() {
        // TODO: Special case for 3 point vertices?
        bail!(TokenCountError::new(2, 4));
    }

    Ok(Vector2::from_array(result))
}
