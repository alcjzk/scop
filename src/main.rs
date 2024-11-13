use scop::{ColorRGB, Error, Obj, Renderer, Result, Vertex};

use ash::vk;
use std::env;
use std::fs::OpenOptions;
use std::path::Path;

use glfw::{fail_on_errors, Action, ClientApiHint, Key, WindowHint, WindowMode};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let obj_path = args
        .next()
        .unwrap_or_else(|| env::var("OBJ_PATH").expect("no object file specified"));
    let texture_path = args
        .next()
        .unwrap_or_else(|| env::var("TEXTURE_PATH").expect("no texture file specified"));

    let mut glfw = glfw::init(fail_on_errors!())?;

    glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));

    if !glfw.vulkan_supported() {
        return Err(Error::Generic("vulkan not supported"));
    }

    let (mut window, events) = glfw
        .create_window(800, 800, "scop", WindowMode::Windowed)
        .ok_or(Error::Generic("failed to create glfw window"))?;

    window.set_key_polling(true);

    let (vertices, indices) = load_model(&obj_path)?;

    let mut renderer = Renderer::new(&glfw, &window, vertices, indices, texture_path)?;

    while !window.should_close() {
        glfw.poll_events();

        let mut should_resize = unsafe { renderer.draw_frame()? };

        for (_, event) in glfw::flush_messages(&events) {
            println!("{event:#?}");

            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true);
                }
                glfw::WindowEvent::FramebufferSize(_, _) => {
                    should_resize = true;
                }
                _ => {}
            }
        }

        if should_resize {
            unsafe {
                let (width, height) = window.get_framebuffer_size();
                renderer.resize(vk::Extent2D {
                    width: width as _,
                    height: height as _,
                })?;
            }
        }
    }

    Ok(())
}

//pub fn load_model<P: AsRef<Path> + std::fmt::Debug>(
//    path: P,
//) -> Result<(Vec<Vertex<f32>>, Vec<u16>)> {
//    let (models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;
//
//    let mut vertices = vec![];
//    let mut indices = vec![];
//
//    for model in models {
//        for index in model.mesh.indices {
//            let vertex = Vertex {
//                position: Vector::from_array([
//                    model.mesh.positions[(3 * index) as usize],
//                    model.mesh.positions[(3 * index + 1) as usize],
//                    model.mesh.positions[(3 * index + 2) as usize],
//                ]),
//                texture_position: Vector::from_array([
//                    model.mesh.texcoords[(2 * index) as usize],
//                    1.0 - model.mesh.texcoords[(2 * index + 1) as usize],
//                ]),
//                color: ColorRGB::RED,
//            };
//
//            vertices.push(vertex);
//            indices.push(indices.len() as _);
//        }
//    }
//
//    Ok((vertices, indices))
//}

pub fn load_model<P: AsRef<Path> + std::fmt::Debug>(
    path: P,
) -> Result<(Vec<Vertex<f32>>, Vec<u16>)> {
    let file = OpenOptions::new().read(true).open(path)?;
    let obj = Obj::from_reader(&file)?.make_single_index();

    let mut vertices = vec![];
    let mut indices = vec![];

    for index in obj.geometric_indices {
        let mut vertex = Vertex {
            position: obj.geometric_vertices[index as usize],
            texture_position: obj.texture_vertices[index as usize],
            color: ColorRGB::RED,
        };

        vertex.texture_position[1] = 1.0 - vertex.texture_position[1];

        vertices.push(vertex);
        indices.push(indices.len() as _);
    }

    Ok((vertices, indices))
}
