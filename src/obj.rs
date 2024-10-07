use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};

use crate::Result;

pub type Keyword = String;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Obj {
    pub unknown_keywords: HashMap<Keyword, usize>,
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

            let keyword = match tokens.next() {
                Some(keyword) => keyword,
                None => continue,
            };

            match keyword {
                unknown => *result.unknown_keywords.entry(unknown.into()).or_default() += 1,
            }
        }
        Ok(result)
    }
}
