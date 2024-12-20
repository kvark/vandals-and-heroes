use blade_graphics as gpu;
use vandals_and_heroes::{
    camera::Camera,
    config::{Config, Car, Map},
    model::{ModelInstance},
    render::{Render}
};

use std::{f32, fs, path, thread, time};

pub struct Game {
    // engine stuff
    #[allow(dead_code)] //TODO
    choir: choir::Choir,
    render: Render,
    // windowing
    pub window: winit::window::Window,
    window_size: winit::dpi::PhysicalSize<u32>,
    // navigation
    camera: Camera,
    in_camera_drag: bool,
    last_mouse_pos: [i32; 2],
    // game
    car_body: ModelInstance,
}

struct QuitEvent;

impl Game {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        log::info!("Initializing");

        let config: Config = ron::de::from_bytes(
            &fs::read("data/config.ron").expect("Unable to open the main config"),
        )
        .expect("Unable to parse the main config");

        let choir = choir::Choir::default();
        let gpu_context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                ..Default::default()
            })
        }
            .expect("Unable to initialize GPU");

        log::info!("Creating the window");
        let window_attributes = winit::window::Window::default_attributes()
            .with_title("Vandals and Heroes")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 800));
        #[allow(deprecated)] //TODO
        let window = event_loop.create_window(window_attributes).unwrap();
        let window_size = window.inner_size();
        let extent = gpu::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };

        let gpu_surface = gpu_context.create_surface(&window).unwrap();
        let mut render = Render::new(gpu_context, gpu_surface, extent);
        render.set_ray_params(&config.ray);

        let mut loader = render.start_loading();

        let mut car_body = {
            log::info!("Loading car: {}", config.car);
            let car_path = path::PathBuf::from("data/cars").join(config.car);
            let _car_config: Car = ron::de::from_bytes(
                &fs::read(car_path.join("car.ron")).expect("Unable to open the car config"),
            )
                .expect("Unable to parse the car config");
            let model = loader.load_gltf(&car_path.join("body.glb"));
            ModelInstance {
                model,
                pos: Default::default(),
                rot: nalgebra::UnitQuaternion::from_axis_angle(
                    &nalgebra::Vector3::y_axis(),
                    0.5 * f32::consts::PI,
                ),
            }
        };

        let mut camera = Camera::default();
        {
            log::info!("Loading map: {}", config.map);
            let map_path = path::PathBuf::from("data/maps").join(config.map);
            let map_config: Map = ron::de::from_bytes(
                &fs::read(map_path.join("map.ron")).expect("Unable to open the map config"),
            )
                .expect("Unable to parse the map config");

            let (map_texture, map_extent) = loader.load_png(&map_path.join("map.png"));

            let circumference = 2.0 * f32::consts::PI * map_config.radius.start;
            let length = circumference * (map_extent.height as f32) / (map_extent.width as f32);
            log::info!("Derived map length to be {}", length);
            camera.pos = nalgebra::Vector3::new(0.0, map_config.radius.end, 0.1 * length);
            camera.rot = nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::x_axis(),
                0.3 * f32::consts::PI,
            );
            camera.clip.end = length;

            car_body.pos = nalgebra::Vector3::new(
                0.0,
                0.35 * map_config.radius.start + 0.65 * map_config.radius.end,
                0.1 * length,
            );

            let submission = loader.finish();
            render.accept_submission(submission);
            render.set_map(map_texture, map_config.radius, length);
        }

        Self {
            choir,
            render,
            window,
            window_size,
            camera,
            in_camera_drag: false,
            last_mouse_pos: [0; 2],
            car_body,
        }
    }

    fn redraw(&mut self) -> time::Duration {
        let models = [&self.car_body];
        self.render.draw(&self.camera, &models);
        time::Duration::from_millis(16)
    }

    pub fn on_event(
        &mut self,
        event: &winit::event::WindowEvent,
    ) -> Result<winit::event_loop::ControlFlow, QuitEvent> {
        match *event {
            winit::event::WindowEvent::Resized(size) => {
                if size != self.window_size {
                    log::info!("Resizing to {:?}", size);
                    self.window_size = size;
                    self.render.resize(gpu::Extent {
                        width: size.width,
                        height: size.height,
                        depth: 1,
                    });
                }
            }
            winit::event::WindowEvent::KeyboardInput {
                event:
                winit::event::KeyEvent {
                    physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                    state: winit::event::ElementState::Pressed,
                    ..
                },
                ..
            } => match key_code {
                winit::keyboard::KeyCode::Escape => {
                    return Err(QuitEvent);
                }
                _ => {
                    let delta = 0.1;
                    self.camera.on_key(key_code, delta);
                }
            },
            winit::event::WindowEvent::KeyboardInput {
                event:
                winit::event::KeyEvent {
                    physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                    state: winit::event::ElementState::Released,
                    ..
                },
                ..
            } => match key_code {
                _ => {}
            },
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
            winit::event::WindowEvent::CloseRequested => {
                return Err(QuitEvent);
            }
            winit::event::WindowEvent::RedrawRequested => {
                let wait = self.redraw();

                return Ok(
                    if let Some(repaint_after_instant) = std::time::Instant::now().checked_add(wait)
                    {
                        winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
                    } else {
                        winit::event_loop::ControlFlow::Wait
                    },
                );
            }
            _ => {}
        }

        Ok(winit::event_loop::ControlFlow::Poll)
    }
}

impl Drop for Game {
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }
        log::info!("Deinitializing");
        self.render.wait_for_gpu();
        self.car_body.model.free(self.render.context());
        self.render.deinit();
    }
}

fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut game = Game::new(&event_loop);

    #[allow(deprecated)] //TODO
    event_loop
        .run(|event, target| match event {
            winit::event::Event::AboutToWait => {
                game.window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match game.on_event(&event) {
                Ok(control_flow) => {
                    target.set_control_flow(control_flow);
                }
                Err(QuitEvent) => {
                    target.exit();
                }
            },
            _ => {}
        })
        .unwrap();
}
