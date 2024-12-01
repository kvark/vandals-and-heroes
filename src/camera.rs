use std::ops::Range;

const MAX_FLY_SPEED: f32 = 1000000.0;

pub struct Camera {
    pub pos: nalgebra::Vector3<f32>,
    pub rot: nalgebra::UnitQuaternion<f32>,
    pub clip: Range<f32>,
    pub fov_y: f32,
    pub fly_speed: f32,
    pub rotate_speed: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pos: nalgebra::Vector3::default(),
            rot: nalgebra::UnitQuaternion::identity(),
            clip: 0.1..100.0,
            fov_y: 1.0,
            fly_speed: 10.0,
            rotate_speed: 500.0,
        }
    }
}

impl Camera {
    pub fn move_by(&mut self, offset: nalgebra::Vector3<f32>) {
        self.pos += self.rot * offset;
    }

    pub fn rotate_z_by(&mut self, angle: f32) {
        let rotation =
            nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), angle);
        self.rot = self.rot * rotation;
    }

    pub fn on_key(&mut self, code: winit::keyboard::KeyCode, delta: f32) -> bool {
        use winit::keyboard::KeyCode as Kc;

        let move_offset = self.fly_speed * delta;
        let rotate_offset_z = self.rotate_speed * delta;
        match code {
            Kc::KeyW => {
                self.move_by(nalgebra::Vector3::new(0.0, 0.0, move_offset));
            }
            Kc::KeyS => {
                self.move_by(nalgebra::Vector3::new(0.0, 0.0, -move_offset));
            }
            Kc::KeyA => {
                self.move_by(nalgebra::Vector3::new(-move_offset, 0.0, 0.0));
            }
            Kc::KeyD => {
                self.move_by(nalgebra::Vector3::new(move_offset, 0.0, 0.0));
            }
            Kc::KeyZ => {
                self.move_by(nalgebra::Vector3::new(0.0, -move_offset, 0.0));
            }
            Kc::KeyX => {
                self.move_by(nalgebra::Vector3::new(0.0, move_offset, 0.0));
            }
            Kc::KeyQ => {
                self.rotate_z_by(rotate_offset_z);
            }
            Kc::KeyE => {
                self.rotate_z_by(-rotate_offset_z);
            }
            _ => return false,
        }

        true
    }

    pub fn on_wheel(&mut self, delta: winit::event::MouseScrollDelta) {
        let shift = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, lines) => lines,
            winit::event::MouseScrollDelta::PixelDelta(position) => position.y as f32,
        };
        self.fly_speed = (self.fly_speed * shift.exp()).clamp(1.0, MAX_FLY_SPEED);
    }
}
