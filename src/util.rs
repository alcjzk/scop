use crate::Result;

use std::collections::HashSet;

pub trait IteratorExt: Iterator {
    fn collect_vec(self) -> Vec<Self::Item>
    where
        Self: Sized,
    {
        self.collect()
    }
    fn collect_set(self) -> HashSet<Self::Item>
    where
        Self: Sized,
        Self::Item: std::hash::Hash + Eq,
    {
        self.collect()
    }
    fn try_collect_vec<T, E>(self) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
        Result<Vec<T>>: FromIterator<Result<T, E>>,
    {
        self.collect()
    }
}

impl<T: Iterator> IteratorExt for T {}

macro_rules! bail {
    ($e:expr) => {
        return Err($e.into());
    };
}
pub(crate) use bail;
