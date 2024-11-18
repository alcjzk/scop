use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::io::{BufRead, BufReader, Read};

use crate::math::{Vector2, Vector3};
use crate::{bail, Error, Result};
pub type VertexIndex = isize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    GeometricVertex,
    TextureVertex,
    // VertexNormal,
    // ParameterSpaceVertex,
    Face,
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
            "f" => Face,
            unknown => return Err(UnknownKeywordError::new(unknown)),
        };

        Ok(keyword)
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Obj {
    pub geometric_vertices: Vec<Vector3<f32>>,
    pub texture_vertices: Vec<Vector2<f32>>,
    pub geometric_indices: Vec<VertexIndex>,
    pub texture_indices: Vec<VertexIndex>,
    pub face_vertex_counts: Vec<usize>,
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
                Keyword::Face => {
                    let mut count = 0;

                    for vertices in FaceVertices::new(tokens) {
                        let vertices = vertices?;
                        result.geometric_indices.push(vertices.geometric_index);

                        if let Some(texture_index) = vertices.texture_index {
                            result.texture_indices.push(texture_index);
                        }

                        // TODO: Enforce consistent format?

                        count += 1;
                    }

                    result.face_vertex_counts.push(count);
                }
            }
        }
        Ok(result)
    }
    pub fn is_single_index(&self) -> bool {
        if self.texture_indices.is_empty() {
            return true;
        }
        self.geometric_indices == self.texture_indices
    }
    pub fn is_triangle_only(&self) -> bool {
        self.face_vertex_counts.iter().copied().all(|c| c == 3)
    }
    pub fn triangulate(self) -> Self {
        // TODO: Enforce vertex counts >= 3?
        let triangle_count = self
            .face_vertex_counts
            .iter()
            .copied()
            .fold(0, |accumulator, vertex_count| {
                accumulator + (3 * (vertex_count - 2))
            });

        let mut geometric_indices = Vec::with_capacity(triangle_count);
        let mut texture_indices = match self.texture_indices.is_empty() {
            true => vec![],
            false => Vec::with_capacity(triangle_count),
        };

        let mut offset = 0;
        for mut vertex_count in self.face_vertex_counts.iter().copied() {
            let next_offset = offset + vertex_count;

            while vertex_count > 3 {
                geometric_indices.push(self.geometric_indices[offset]);
                geometric_indices.push(self.geometric_indices[offset + vertex_count - 1]);
                geometric_indices.push(self.geometric_indices[offset + vertex_count - 2]);

                if !self.texture_indices.is_empty() {
                    texture_indices.push(self.texture_indices[offset]);
                    texture_indices.push(self.texture_indices[offset + vertex_count - 1]);
                    texture_indices.push(self.texture_indices[offset + vertex_count - 2]);
                }

                vertex_count -= 1;
            }

            geometric_indices.extend(&self.geometric_indices[offset..offset + 3]);
            if !self.texture_indices.is_empty() {
                texture_indices.extend(&self.texture_indices[offset..offset + 3]);
            }
            offset = next_offset;
        }

        debug_assert_eq!(geometric_indices.capacity(), triangle_count);

        Self {
            geometric_indices,
            texture_indices,
            face_vertex_counts: vec![],
            ..self
        }
    }
    pub fn make_single_index(mut self) -> Self {
        if self.texture_vertices.is_empty() {
            return self;
        }

        // TODO: Guarantee with format
        debug_assert_eq!(self.geometric_vertices.len(), self.texture_vertices.len());

        for (geometric_index, texture_index) in self
            .geometric_indices
            .iter_mut()
            .zip(self.texture_indices.iter_mut())
        {
            if *geometric_index != *texture_index {
                self.geometric_vertices
                    .push(self.geometric_vertices[*geometric_index as usize]);
                *geometric_index = (self.geometric_vertices.len() - 1) as _;
                self.texture_vertices
                    .push(self.texture_vertices[*texture_index as usize]);
                *texture_index = (self.texture_vertices.len() - 1) as _;
                debug_assert_eq!(*geometric_index, *texture_index);
            }
        }

        self
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

/// Generic set of indices for a single face vertex.
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct FaceVertexIndices {
    pub geometric_index: VertexIndex,
    pub texture_index: Option<VertexIndex>,
    pub normal_index: Option<VertexIndex>,
}

impl FaceVertexIndices {
    pub fn from_tokens<'a>(mut tokens: impl Iterator<Item = &'a str>) -> Result<Self> {
        let geometric_index = tokens
            .next()
            .ok_or(Error::Generic("face does not define a geometric index"))?
            .parse::<VertexIndex>()?
            - 1;

        let texture_index = tokens
            .next()
            .map(|t| -> Result<_> { Ok(t.parse::<VertexIndex>()? - 1) })
            .transpose()?;

        let normal_index = tokens
            .next()
            .map(|t| -> Result<_> { Ok(t.parse::<VertexIndex>()? - 1) })
            .transpose()?;

        Ok(Self {
            geometric_index,
            texture_index,
            normal_index,
        })
    }
}

/// Iterator over vertices of a face.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Debug, Clone, Copy)]
struct FaceVertices<I> {
    tokens: I,
}

impl<T> FaceVertices<T> {
    fn new<'a>(tokens: T) -> Self
    where
        T: Iterator<Item = &'a str>,
    {
        Self { tokens }
    }
}

impl<'a, I> Iterator for FaceVertices<I>
where
    I: Iterator<Item = &'a str>,
{
    type Item = Result<FaceVertexIndices>;

    fn next(&mut self) -> Option<Self::Item> {
        let index_tokens = self.tokens.next()?.split('/');

        Some(FaceVertexIndices::from_tokens(index_tokens))
    }
}
