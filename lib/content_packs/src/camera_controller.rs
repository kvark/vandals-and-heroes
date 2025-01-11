use vandals_and_heroes::Camera;

pub struct CameraController {
    camera: Camera,
    in_camera_drag: bool,
    last_mouse_pos: [i32; 2],
}

impl CameraController {
    pub fn new(camera: Camera) -> CameraController {
        Self {
            camera,
            in_camera_drag: false,
            last_mouse_pos: [0, 0],
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn on_event(&mut self, event: &winit::event::WindowEvent) {
        match *event {
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let delta = 0.1;
                self.camera.on_key(key_code, delta);
            }
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                self.camera.on_wheel(delta);
            }
            winit::event::WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                self.in_camera_drag = true;
            }
            winit::event::WindowEvent::MouseInput {
                state: winit::event::ElementState::Released,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                self.in_camera_drag = false;
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                if self.in_camera_drag {
                    self.camera.on_drag(
                        self.last_mouse_pos[0] as f32 - position.x as f32,
                        self.last_mouse_pos[1] as f32 - position.y as f32,
                    );
                }
                self.last_mouse_pos = [position.x as i32, position.y as i32];
            }
            _ => {}
        }
    }
}
