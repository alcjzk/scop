use crate::{Vector2, Vector3};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Vertex<T: Copy> {
    pub position: Vector3<T>,
    pub texture_position: Vector2<T>,
}
