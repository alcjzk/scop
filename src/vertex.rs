use crate::{Vector2, Vector3};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Vertex<T: Copy> {
    pub position: Vector3<T>,
    pub color: ColorRGB<T>,
    pub texture_position: Vector2<T>,
}

#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ColorRGB<T: Copy>(Vector3<T>);

impl ColorRGB<f32> {
    pub const RED: Self = Self(Vector3::from_array([1.0, 0.0, 0.0]));
    pub const GREEN: Self = Self(Vector3::from_array([0.0, 1.0, 0.0]));
    pub const BLUE: Self = Self(Vector3::from_array([0.0, 0.0, 1.0]));
}
