pub mod matrix;
pub mod vector;

pub use matrix::*;
pub use vector::*;

pub fn lerp(from: f32, to: f32, amount: f32) -> f32 {
    from * (1.0 - amount) + to * amount
}
