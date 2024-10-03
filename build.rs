use std::fs;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=shaders");

    let out_dir = std::env::var("OUT_DIR").unwrap();

    for path in fs::read_dir("shaders").unwrap() {
        let path = path.unwrap().path();

        if let Some(file_name) = path.file_name() {
            let file_name = file_name.to_str().unwrap();
            if file_name.ends_with(".vert") || file_name.ends_with(".frag") {
                Command::new("glslc")
                    .args([
                        path.to_str().unwrap(),
                        "-o",
                        &format!("{}/{}.spv", out_dir, file_name),
                    ])
                    .spawn()
                    .unwrap()
                    .wait()
                    .unwrap();
            }
        }
    }
}
