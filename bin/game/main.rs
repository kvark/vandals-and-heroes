use blade_graphics as gpu;
use vandals_and_heroes::{config, Camera, Object, Physics, Render, TerrainBody};

use std::{f32, fs, path, sync::Arc, thread, time};

struct Axle {
    wheels: Vec<Object>,
    radius: f32,
}

struct Car {
    body: Object,
    axles: Vec<Axle>,
}

pub struct Game {
    // engine stuff
    #[allow(dead_code)] //TODO
    choir: choir::Choir,
    render: Render,
    physics: Physics,
    // windowing
    window: winit::window::Window,
    window_size: winit::dpi::PhysicalSize<u32>,
    // navigation
    camera: Camera,
    in_camera_drag: bool,
    last_mouse_pos: [i32; 2],
    // game
    terrain_body: TerrainBody,
    car: Car,
}

pub struct QuitEvent;

impl Game {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        log::info!("Initializing");

        let config: config::Config = ron::de::from_bytes(
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

        let (map_texture, terrain_body) = {
            log::info!("Loading map: {}", config.map);
            let map_path = path::PathBuf::from("data/maps").join(config.map);
            let mut map_config: config::Map = ron::de::from_bytes(
                &fs::read(map_path.join("map.ron")).expect("Unable to open the map config"),
            )
            .expect("Unable to parse the map config");

            loader.load_terrain(&map_path.join("map.png"), &mut map_config)
        };

        let camera = Camera {
            pos: nalgebra::Vector3::new(
                0.0,
                terrain_body.config.radius.end,
                0.1 * terrain_body.config.length,
            ),
            rot: nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::x_axis(),
                0.3 * f32::consts::PI,
            ),
            clip: 1.0..terrain_body.config.length,
            ..Default::default()
        };

        let mut physics = Physics::default();
        let car = {
            log::info!("Loading car: {}", config.car);
            let mc = &terrain_body.config;
            let car_path = path::PathBuf::from("data/cars").join(config.car);
            let car_config: config::Car = ron::de::from_bytes(
                &fs::read(car_path.join("car.ron")).expect("Unable to open the car config"),
            )
            .expect("Unable to parse the car config");

            let car_body = Arc::new(loader.load_gltf(&car_path, &car_config.body));
            let body_isometry = nalgebra::Isometry3 {
                translation: nalgebra::Vector3::new(
                    0.0,
                    0.35 * mc.radius.start + 0.65 * mc.radius.end,
                    0.1 * mc.length,
                )
                .into(),
                rotation: nalgebra::UnitQuaternion::from_axis_angle(
                    &nalgebra::Vector3::y_axis(),
                    0.5 * f32::consts::PI,
                ),
            };

            let mut axles = Vec::new();
            for axle in car_config.axles.iter() {
                let car_wheel = Arc::new(loader.load_gltf(&car_path, &axle.wheel));
                let mut wheels = Vec::new();
                let wheel_radius = car_wheel
                    .collider
                    .shape()
                    .compute_local_bounding_sphere()
                    .radius;
                for &axle_x in axle.xs.iter() {
                    let local_isometry = nalgebra::Isometry3 {
                        translation: nalgebra::Vector3::new(axle_x, axle.y, axle.z).into(),
                        rotation: nalgebra::UnitQuaternion::identity(),
                    };
                    let object = physics
                        .create_object(Arc::clone(&car_wheel), body_isometry * local_isometry);
                    /*engine.add_joint(
                        vehicle.body_handle,
                        wheel_handle,
                        blade::JointDesc {
                            linear: blade::FreedomAxis::ALL_FREE,
                            angular: blade::FreedomAxis::ALL_FREE,
                            ..Default::default()
                        },
                    );*/
                    //let _ = physics.create_joint(&car_body.rigid_body, &object.rigid_body);
                    wheels.push(object);
                }
                axles.push(Axle {
                    wheels,
                    radius: wheel_radius,
                });
            }

            Car {
                body: physics.create_object(car_body, body_isometry),
                axles,
            }
        };

        let submission = loader.finish();
        render.accept_submission(submission);

        render.set_map(map_texture, &terrain_body.config);
        //let terrain_body = physics.create_terrain(map_collider);

        Self {
            choir,
            render,
            physics,
            window,
            window_size,
            camera,
            in_camera_drag: false,
            last_mouse_pos: [0; 2],
            terrain_body,
            car,
        }
    }

    fn update_physics(&mut self) {
        let mut objects = vec![&mut self.car.body];
        for axle in self.car.axles.iter_mut() {
            for wheel in axle.wheels.iter_mut() {
                let center = wheel.transform.translation.vector;
                let up_dir = nalgebra::Vector3::new(center.x, center.y, 0.0).normalize();
                let ground_point = nalgebra::Point3::from(center - axle.radius * up_dir);
                let impulse = self.terrain_body.compute_resistance(ground_point);
                if impulse.magnitude_squared() > 0.0 {
                    self.physics.apply_impulse(wheel.rigid_body, impulse);
                } else {
                    objects.push(wheel);
                }
            }
        }
        for object in objects.iter_mut() {
            self.physics
                .update_gravity(object.rigid_body, &self.terrain_body);
        }

        self.physics.step();

        for object in objects.iter_mut() {
            object.transform = self.physics.get_transform(object.rigid_body);
        }
    }

    fn redraw(&mut self) -> time::Duration {
        //TODO: detach from rendering
        self.update_physics();

        let mut objects = vec![&self.car.body];
        for axle in self.car.axles.iter() {
            for wheel in axle.wheels.iter() {
                objects.push(wheel);
            }
        }
        self.render.draw(&self.camera, &objects);

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
        self.car.body.model.free(self.render.context());
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
