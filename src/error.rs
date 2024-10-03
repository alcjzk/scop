use std::error::Error as StdError;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    Generic(&'static str),
    Other(Box<dyn StdError + Send + Sync + 'static>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Generic(message) => write!(f, "{message}"),
            Error::Other(error) => write!(f, "{error}"),
        }
    }
}

impl<E> From<E> for Error
where
    E: StdError + Send + Sync + 'static,
{
    fn from(value: E) -> Self {
        Self::Other(Box::new(value))
    }
}
