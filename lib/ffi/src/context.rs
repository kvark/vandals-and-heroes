use blade_graphics as gpu;
use std::fs;
use vandals_and_heroes::{config, Camera, Render, Terrain};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct Context {
    camera: Camera,
    render: Render,
    terrain: Option<Terrain>,
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
        Some(Self {
            camera,
            render,
            terrain: None,
        })
    }

    pub(crate) fn deinit(&mut self) {
        log::info!("Deinitializing");
        self.render.wait_for_gpu();
        if let Some(terrain) = &self.terrain {
            terrain.texture.deinit(self.render.context());
        }
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
        if let Some(terrain) = &self.terrain {
            self.render.draw(&self.camera, terrain, &Vec::new())
        }
    }

    pub(crate) fn create_map(
        &mut self,
        mut config: config::Map,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) {
        let mut loader = self.render.start_loading();

        log::info!("Loading map: {:?}", config.radius);

        let map_texture = loader.load_terrain(
            gpu::Extent {
                width,
                height,
                depth: 1,
            },
            bytes,
        );

        if config.length == 0.0 {
            let circumference = 2.0 * std::f32::consts::PI * config.radius.start;
            config.length = circumference * (height as f32) / (width as f32);
            log::info!("Derived map length to be {}", config.length);
        }

        self.camera.pos = nalgebra::Vector3::new(0.0, config.radius.end, 0.1 * config.length);
        self.camera.rot = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::x_axis(),
            0.3 * std::f32::consts::PI,
        );
        self.camera.clip.end = config.length;

        let submission = loader.finish();
        self.render.accept_submission(submission);
        self.terrain = Some(Terrain {
            texture: map_texture,
            config,
        });
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.deinit();
    }
}
