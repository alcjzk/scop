use crate::Matrix;

#[repr(C, align(64))]
#[derive(Debug, Default)]
pub struct UniformBufferObject {
    pub model: Matrix<f32, 4, 4>,
    pub view: Matrix<f32, 4, 4>,
    pub projection: Matrix<f32, 4, 4>,
    pub texture_opacity: f32,
}
