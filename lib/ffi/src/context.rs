use blade_graphics as gpu;
use blade_graphics::Extent;
use std::{fs, path};
use vandals_and_heroes::{camera::Camera, config, render::Render};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub(crate) struct Context {
    camera: Camera,
    render: Render,
}

impl Context {
    pub(crate) fn new<I: HasWindowHandle + HasDisplayHandle>(
        extent: gpu::Extent,
        handle: &I,
    ) -> Option<Self> {
        let gpu_context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                ..Default::default()
            })
        }
        .expect("Unable to initialize GPU");

        let gpu_surface = match gpu_context.create_surface(&handle) {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to create surface: {:?}", e);
                return None;
            }
        };

        let mut render = Render::new(gpu_context, gpu_surface, extent);

        let config: config::Config = ron::de::from_bytes(
            &fs::read("data/config.ron").expect("Unable to open the main config"),
        )
        .expect("Unable to parse the main config");
        render.set_ray_params(&config.ray);

        let camera = Camera::default();
        Some(Self { camera, render })
    }

    pub(crate) fn deinit(&mut self) {
        log::info!("Deinitializing");
        self.render.wait_for_gpu();
        self.render.deinit();
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.render.resize(gpu::Extent {
            width,
            height,
            depth: 1,
        });
    }

    pub(crate) fn render(&mut self) {
        self.render.draw(&self.camera, &[])
    }

    pub(crate) fn set_map(&mut self, map: config::Map, width: u32, height: u32, bytes: &[u8]) {
        let mut loader = self.render.start_loading();

        log::info!("Loading map: {:?}", map.radius);

        let map_texture = loader.load_terrain(
            Extent {
                width,
                height,
                depth: 1,
            },
            bytes,
        );

        let circumference = 2.0 * std::f32::consts::PI * map.radius.start;
        let length = circumference * (height as f32) / (width as f32);
        log::info!("Derived map length to be {}", length);

        self.camera.pos = nalgebra::Vector3::new(0.0, map.radius.end, 0.1 * length);
        self.camera.rot = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::x_axis(),
            0.3 * std::f32::consts::PI,
        );
        self.camera.clip.end = length;

        let submission = loader.finish();
        self.render.accept_submission(submission);
        self.render.set_map(map_texture, map.radius, length);
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.deinit();
    }
}
