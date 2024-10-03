use crate::Matrix;

#[repr(C)]
pub struct UniformBufferObject {
    pub model: Matrix<f32, 4, 4>,
    pub view: Matrix<f32, 4, 4>,
    pub projection: Matrix<f32, 4, 4>,
}
