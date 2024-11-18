use glfw::{Action, Key};

#[derive(Debug, Default)]
pub struct InputManager {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub forward: bool,
    pub backward: bool,
    pub toggle_texture: bool,
}

impl InputManager {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn handle_key_event(&mut self, key: Key, action: Action) {
        let value = match key {
            Key::W => Some(&mut self.up),
            Key::S => Some(&mut self.down),
            Key::A => Some(&mut self.left),
            Key::D => Some(&mut self.right),
            Key::E => Some(&mut self.forward),
            Key::Q => Some(&mut self.backward),
            Key::T if action == Action::Press => {
                self.toggle_texture = true;
                return;
            }
            _ => None,
        };

        if let Some(value) = value {
            match action {
                Action::Press => *value = true,
                Action::Release => *value = false,
                _ => (),
            }
        }
    }
    pub fn toggle_texture(&mut self) -> bool {
        std::mem::take(&mut self.toggle_texture)
    }
}
