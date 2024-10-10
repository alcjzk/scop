use std::iter::Sum;
use std::ops::{Add, Deref, DerefMut, Div, Mul, Sub};

pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vector4<T> = Vector<T, 4>;

pub trait Float: Copy {
    fn sqrt(self) -> Self;
}

impl Float for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

pub trait Dot {
    type Output;

    fn dot(self, rhs: Self) -> Self::Output;
}

pub trait Cross {
    type Output;
    /// Returns the cross product or `self` and `rhs`.
    fn cross(self, rhs: Self) -> Self::Output;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vector<T, const N: usize>([T; N]);

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Self([T::default(); N])
    }
}

impl<T, const N: usize> Deref for Vector<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Vector<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const N: usize> Div<T> for Vector<T, N>
where
    T: Div<Output = T> + Copy,
{
    type Output = Self;

    fn div(mut self, rhs: T) -> Self::Output {
        for value in self.iter_mut() {
            *value = *value / rhs;
        }
        self
    }
}

impl<T, const N: usize> Mul<T> for Vector<T, N>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        for value in self.iter_mut() {
            *value = *value * rhs;
        }
        self
    }
}

impl<T, const N: usize> Sub for Vector<T, N>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for (a, b) in self.iter_mut().zip(rhs.into_iter()) {
            *a = *a - b;
        }
        self
    }
}

impl<T, const N: usize> Add for Vector<T, N>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        for (a, b) in self.iter_mut().zip(rhs.into_iter()) {
            *a = *a + b;
        }
        self
    }
}

impl<T, const N: usize> Vector<T, N> {
    /// Returns the dot product of `self` and `rhs`.
    pub fn dot(self, rhs: Self) -> T
    where
        T: Copy + Mul<Output = T> + Sum,
    {
        self.iter()
            .copied()
            .zip(rhs.iter().copied())
            .map(|(a, b)| a * b)
            .sum()
    }
    /// Returns the vectors magnitude.
    pub fn magnitude(self) -> T
    where
        T: Copy + Mul<Output = T> + Sum + Float,
    {
        self.dot(self).sqrt()
    }
    /// Returns the vector normalized.
    pub fn normalize(self) -> Self
    where
        T: Copy + Mul<Output = T> + Sum + Float,
        Self: Div<T, Output = Self>,
    {
        self / self.magnitude()
    }
    /// Constructs the `Vector<T, N>` from an array.
    pub const fn from_array(array: [T; N]) -> Self {
        Self(array)
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(value: [T; N]) -> Self {
        Self::from_array(value)
    }
}

impl<T> Cross for Vector2<T>
where
    T: Copy + Sub<Output = T> + Mul<Output = T>,
{
    type Output = T;

    fn cross(self, rhs: Self) -> Self::Output {
        self[0] * rhs[1] - self[1] * rhs[0]
    }
}

impl<T> Cross for Vector<T, 3>
where
    T: Copy + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn cross(self, rhs: Self) -> Self::Output {
        Self([
            self[1] * rhs[2] - rhs[1] * self[2],
            self[2] * rhs[0] - rhs[2] * self[0],
            self[0] * rhs[1] - rhs[0] * self[1],
        ])
    }
}
