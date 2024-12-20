use std::{fs, path};
use blade_graphics as gpu;

use vandals_and_heroes::{
    camera::Camera,
    render::Render,
    config
};

use raw_window_handle::{HasWindowHandle, HasDisplayHandle};


pub(crate) struct Context {
    camera: Camera,
    render: Render
}

impl Context {
    pub(crate) fn new<
        I: HasWindowHandle + HasDisplayHandle,
    >(extent: gpu::Extent,  handle: &I) -> Option<Self> {
        let gpu_context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                ..Default::default()
            })
        }.expect("Unable to initialize GPU");

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
        ).expect("Unable to parse the main config");
        render.set_ray_params(&config.ray);

        let mut loader = render.start_loading();
        let mut camera = Camera::default();
        {
            log::info!("Loading map: {}", config.map);
            let map_path = path::PathBuf::from("data/maps").join(config.map);
            let map_config: config::Map = ron::de::from_bytes(
                &fs::read(map_path.join("map.ron")).expect("Unable to open the map config"),
            )
                .expect("Unable to parse the map config");

            let (map_texture, map_extent) = loader.load_png(&map_path.join("map.png"));

            let circumference = 2.0 * std::f32::consts::PI * map_config.radius.start;
            let length = circumference * (map_extent.height as f32) / (map_extent.width as f32);
            log::info!("Derived map length to be {}", length);
            camera.pos = nalgebra::Vector3::new(0.0, map_config.radius.end, 0.1 * length);
            camera.rot = nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::x_axis(),
                0.3 * std::f32::consts::PI,
            );
            camera.clip.end = length;

            let submission = loader.finish();
            render.accept_submission(submission);
            render.set_map(map_texture, map_config.radius, length);
        }

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
}

impl Drop for Context {
    fn drop(&mut self) {
        self.deinit();
    }
}