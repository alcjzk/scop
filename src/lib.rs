#![feature(try_find)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

mod error;
mod input_manager;
mod math;
mod obj;
mod ppm;
mod ubo;
mod util;
mod vertex;
mod vulkan;

pub use error::*;
pub use input_manager::*;
pub use math::*;
pub use obj::*;
pub use ppm::*;
pub use ubo::*;
pub use util::*;
pub use vertex::*;
pub use vulkan::*;

pub type Result<T, E = Error> = std::result::Result<T, E>;
