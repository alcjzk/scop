use scop::{Error, Obj, Result};

use std::env;
use std::fs::OpenOptions;

fn main() -> Result<()> {
    let path = env::args()
        .nth(1)
        .ok_or(Error::Generic("obj file path not specified"))?;

    let file = OpenOptions::new().read(true).open(path)?;
    let obj = Obj::from_reader(&file)?;

    println!("Geometric vertices: {}", obj.geometric_vertices.len());
    println!("Texture vertices: {}", obj.texture_vertices.len());
    println!("Geometric indices: {}", obj.geometric_indices.len());
    println!("Texture indices: {}", obj.texture_indices.len());
    println!("Face vertex counts: {}", obj.face_vertex_counts.len());
    println!("--");
    println!("Is single index: {}", obj.is_single_index());
    println!("Is triangle only: {}", obj.is_triangle_only());
    println!("--");
    println!("Unknown keywords:");
    for (keyword, count) in obj.unknown_keywords.iter() {
        println!("- '{keyword}': {count}");
    }

    Ok(())
}
