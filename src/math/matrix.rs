use std::ops::{Add, Deref, DerefMut, Mul};

use super::{Vector, Vector3};

use super::Cross as _;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Matrix<T, const W: usize, const H: usize>([Vector<T, W>; H]);

impl<T, const W: usize, const H: usize> Default for Matrix<T, W, H>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Self([Vector::default(); H])
    }
}

impl<T, const W: usize, const H: usize> Deref for Matrix<T, W, H> {
    type Target = [Vector<T, W>; H];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const W: usize, const H: usize> DerefMut for Matrix<T, W, H> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Matrix<f32, 4, 4> {
    pub fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0, 0.0].into(),
            [0.0, 0.0, 1.0, 0.0].into(),
            [0.0, 0.0, 0.0, 1.0].into(),
        ])
    }
    pub fn translate<O>(offset: O) -> Self
    where
        O: Into<Vector3<f32>>,
    {
        let offset = offset.into();
        let mut matrix = Self::identity();

        matrix[3][0] = offset[0];
        matrix[3][1] = offset[1];
        matrix[3][2] = offset[2];

        matrix
    }
    pub fn scale(scalar: f32) -> Self {
        let mut matrix = Self::identity();

        matrix[0][0] = scalar;
        matrix[1][1] = scalar;
        matrix[2][2] = scalar;

        matrix
    }
    /// Vulkan compatible perspective projection, corresponds to GLM RH ZO.
    pub fn perspective(fov_y: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        let tan_half_fov_y = (fov_y / 2.0).tan();
        let mut matrix = Self::default();

        matrix[0][0] = 1.0 / (aspect_ratio * tan_half_fov_y);
        matrix[1][1] = 1.0 / tan_half_fov_y;
        matrix[2][2] = far / (near - far);
        matrix[2][3] = -1.0;
        matrix[3][2] = -(far * near) / (far - near);

        matrix
    }
    pub fn look_at<E, C, U>(eyes: E, center: C, up: U) -> Self
    where
        E: Into<Vector<f32, 3>>,
        C: Into<Vector<f32, 3>>,
        U: Into<Vector<f32, 3>>,
    {
        let eyes = eyes.into();
        let center = center.into();
        let up = up.into();

        let f = (center - eyes).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        let mut matrix = Self::identity();
        matrix[0][0] = s[0];
        matrix[1][0] = s[1];
        matrix[2][0] = s[2];

        matrix[0][1] = u[0];
        matrix[1][1] = u[1];
        matrix[2][1] = u[2];

        matrix[0][2] = -f[0];
        matrix[1][2] = -f[1];
        matrix[2][2] = -f[2];

        matrix[3][0] = -(s.dot(eyes));
        matrix[3][1] = -(u.dot(eyes));
        matrix[3][2] = f.dot(eyes);

        matrix
    }
    pub fn rotate<A>(&self, radians: f32, axis: A) -> Self
    where
        A: Into<Vector<f32, 3>>,
    {
        let axis = axis.into();
        let a = radians;
        let c = a.cos();
        let s = a.sin();

        let axis = axis.normalize();
        let temp = axis * (1.0 - c);

        let mut rotate = Self::default();

        rotate[0][0] = c + temp[0] * axis[0];
        rotate[0][1] = temp[0] * axis[1] + s * axis[2];
        rotate[0][2] = temp[0] * axis[2] - s * axis[1];

        rotate[1][0] = temp[1] * axis[0] - s * axis[2];
        rotate[1][1] = c + temp[1] * axis[1];
        rotate[1][2] = temp[1] * axis[2] + s * axis[0];

        rotate[2][0] = temp[2] * axis[0] + s * axis[1];
        rotate[2][1] = temp[2] * axis[1] - s * axis[0];
        rotate[2][2] = c + temp[2] * axis[2];

        let mut result = Self::default();
        result[0] = self[0] * rotate[0][0] + self[1] * rotate[0][1] + self[2] * rotate[0][2];
        result[1] = self[0] * rotate[1][0] + self[1] * rotate[1][1] + self[2] * rotate[1][2];
        result[2] = self[0] * rotate[2][0] + self[1] * rotate[2][1] + self[2] * rotate[2][2];
        result[3] = self[3];

        result
    }
}

impl<T> Mul<&Matrix<T, 4, 4>> for &Matrix<T, 4, 4>
where
    T: Mul<Output = T> + Add<Output = T> + Default + Copy,
{
    type Output = Matrix<T, 4, 4>;

    fn mul(self, rhs: &Matrix<T, 4, 4>) -> Self::Output {
        let mut out = Matrix::default();
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = self[i][0] * rhs[0][j]
                    + self[i][1] * rhs[1][j]
                    + self[i][2] * rhs[2][j]
                    + self[i][3] * rhs[3][j];
            }
        }
        out
    }
}

impl<T> Mul<Matrix<T, 4, 4>> for Matrix<T, 4, 4>
where
    T: Mul<Output = T> + Add<Output = T> + Default + Copy,
{
    type Output = Matrix<T, 4, 4>;

    fn mul(self, rhs: Matrix<T, 4, 4>) -> Self::Output {
        let mut out = Matrix::default();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    out[j][i] = out[j][i] + self[k][i] * rhs[j][k];
                }
            }
        }
        out
    }
}

//impl<T> Mul<Matrix<T, 4, 4>> for Matrix<T, 4, 4>
//where
//    T: Mul<Output = T> + Add<Output = T> + Default + Copy,
//{
//    type Output = Matrix<T, 4, 4>;
//
//    fn mul(self, rhs: Matrix<T, 4, 4>) -> Self::Output {
//        let mut out = Matrix::default();
//        for i in 0..4 {
//            for j in 0..4 {
//                out[i][j] = self[i][0] * rhs[0][j]
//                    + self[i][1] * rhs[1][j]
//                    + self[i][2] * rhs[2][j]
//                    + self[i][3] * rhs[3][j];
//            }
//        }
//        out
//    }
//}
