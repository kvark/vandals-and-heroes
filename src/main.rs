#![allow(irrefutable_let_patterns)]

mod camera;
mod config;
mod render;

use blade_graphics as gpu;
use camera::Camera;
use std::{f32, fs, thread, time};

struct Game {
    // engine stuff
    #[allow(dead_code)] //TODO
    choir: choir::Choir,
    render: render::Render,
    // windowing
    window: winit::window::Window,
    window_size: winit::dpi::PhysicalSize<u32>,
    // game data
    camera: Camera,
}

struct QuitEvent;

impl Game {
    fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        log::info!("Initializing");

        let config: config::Main = ron::de::from_bytes(
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
        let mut render = render::Render::new(gpu_context, gpu_surface, extent);

        let mut camera = Camera::default();

        {
            log::info!("Loading map: {}", config.map);
            let png_path = format!("data/maps/{}/map.png", config.map);
            let decoder = png::Decoder::new(fs::File::open(png_path).unwrap());
            let reader = decoder.read_info().unwrap();
            let map_extent = render.load_map(reader);

            let circumference = 2.0 * f32::consts::PI * config.map_radius.start;
            let length = circumference * (map_extent.height as f32) / (map_extent.width as f32);
            log::info!("Derived map length to be {}", length);
            camera.pos = nalgebra::Vector3::new(config.map_radius.end, 0.0, 0.1 * length);
            camera.rot = nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::y_axis(),
                -0.3 * f32::consts::PI,
            );

            render.set_map_view(config.map_radius, length);
        }

        Self {
            choir,
            render,
            window,
            window_size,
            camera,
        }
    }

    fn redraw(&mut self) -> time::Duration {
        self.render.draw(&self.camera);
        time::Duration::from_millis(16)
    }

    fn on_event(
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
