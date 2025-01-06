use std::collections::{HashMap, HashSet};
use std::{fs, time};
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;
use blade_graphics as gpu;
use nalgebra::{Matrix4, UnitQuaternion, Vector3};
use winit::event_loop::EventLoop;
use vandals_and_heroes::{config, Render, Camera, Loader, Terrain, Physics};
use crate::definitions::{ObjectDesc, HeightMapDesc};
use crate::templates::{ObjectTemplate};
use crate::content_pack::ContentPack;
use crate::instances::{Object, TerrainObject};
use crate::camera_controller::CameraController;

pub struct Game {
    render: Render,
    physics: Physics,
    pub window: winit::window::Window,
    window_size: winit::dpi::PhysicalSize<u32>,
    camera_controller: CameraController,
    content_pack: ContentPack,
    templates: HashMap<String, ObjectTemplate>,
    terrain: Option<TerrainObject>,
    instances: Vec<Object>,
}


pub struct QuitEvent;

impl Game {
    pub fn new(event_loop: &EventLoop<()>, mods_directory: &Path) -> Game {
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

        let gpu_context = unsafe {
            blade_graphics::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                ..Default::default()
            })
        }.expect("Unable to initialize GPU");

        let gpu_surface = gpu_context.create_surface(&window).unwrap();

        let mut render = Render::new(gpu_context, gpu_surface, extent);

        let config: config::Config = ron::de::from_bytes(
            &fs::read("data/config.ron").expect("Unable to open the main config"),
        ).expect("Unable to parse the main config");
        render.set_ray_params(&config.ray);

        Self {
            camera_controller: CameraController::new(Camera::default()),
            render,
            physics: Physics::default(),
            content_pack: ContentPack::new(mods_directory),
            window,
            window_size,
            terrain: None,
            templates: HashMap::new(),
            instances: Vec::new(),
        }
    }
    
    pub fn load_level(&mut self, level_id: &str) {
        let level = self.content_pack.get_level_by_id(level_id).unwrap();
    
        let entity_ids: HashSet<String> = level.objects.iter()
            .map(|o| o.entity_id.clone())
            .collect();
    

        let mut loader = self.render.start_loading();
        self.templates = entity_ids.iter()
            .filter_map(|id| 
                if let Some(object) = self.content_pack.get_entity_by_id(id) { 
                    Some((id.clone(), object.load(&self.content_pack, &mut loader)))
                } else {
                    log::warn!("Cannot find entity with id: {}", id);
                    None
                })
            .collect();

        {
            let terrain = Self::load_heightmap(&self.content_pack, &mut loader, &level.height_map);
            let camera = self.camera_controller.camera_mut();

            camera.pos = Vector3::new(0.0, 20.0, 0.1 * terrain.config.length);
            camera.rot = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.3 * PI);
            camera.clip.end = terrain.config.length;

            let body = self.physics.create_terrain(&terrain.config);
            self.terrain = Some(TerrainObject { terrain, body });
        }
        
        let submission = loader.finish();
        self.render.accept_submission(submission);

        self.instances = level.objects.iter()
            .map(|level_object| {
                let template = self.templates.get(&level_object.entity_id).unwrap();
                template.instantiate(&self.content_pack, &mut self.physics, level_object.transform.clone().into())
            })
            .collect();
    }

    pub fn on_event(
        &mut self,
        event: &winit::event::WindowEvent,
    ) -> Result<Option<winit::event_loop::ControlFlow>, QuitEvent> {
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
            },
            winit::event::WindowEvent::CloseRequested => {
                return Err(QuitEvent);
            },
            winit::event::WindowEvent::RedrawRequested => {
                self.redraw();

                let wait = time::Duration::from_millis(16);

                return Ok(
                    if let Some(repaint_after_instant) = std::time::Instant::now().checked_add(wait)
                    {
                        Some(winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant))
                    } else {
                        Some(winit::event_loop::ControlFlow::Wait)
                    },
                );
            }
            _ => self.camera_controller.on_event(event)
        }
        Ok(None)
    }

    fn load_heightmap(content: &ContentPack, loader: &mut Loader, def: &HeightMapDesc) -> Terrain {
        let (texture, extent) = loader.load_png(&content.get_resource_path(&def.image_path));
        let circumference = 2.0 * PI * def.radius.start;
        let length = circumference * (extent.height as f32) / (extent.width as f32);
        Terrain {
            texture,
            config: config::Map {
                radius: def.radius.clone(),
                length,
                density: def.density,
            }
        }
    }

    fn redraw(&mut self) {
        for instance in &self.instances {
            if let Some(body) = instance.body.as_ref() {
                self.physics.update_gravity(
                    body.rigid_body_handle,
                    &self.terrain.as_ref().unwrap().body
                );
            }
        }
        self.physics.step();

        for instance in &mut self.instances {
            if let Some(body) = &instance.body {
                instance.transform = self.physics.get_transform(body.rigid_body_handle);
            }
            if let Some(model_instance) = &mut instance.model_instance {
                model_instance.transform = instance.transform;
            }
        }

        let terrain = &self.terrain.as_ref().unwrap();

        let model_instances = self.instances.iter()
            .filter_map(|instance| instance.model_instance.as_ref())
            .collect();

        self.render.draw(self.camera_controller.camera(), &terrain.terrain, &model_instances);
    }
}

impl ObjectDesc {
    fn load(&self, content: &ContentPack, loader: &mut Loader) -> ObjectTemplate {
        let identity = Matrix4::identity();
        let model_desc = self.scene_path.as_ref()
            .map(|path| Loader::read_gltf(&content.get_resource_path(path), identity));
        
        let model = model_desc.as_ref()
            .map(|model_desc| loader.load_model(model_desc))
            .map(Arc::new);
        
        ObjectTemplate {
            desc: self.clone(),
            model,
        }
    }
}


impl Drop for Game {
    fn drop(&mut self) {
        self.render.wait_for_gpu();
        self.instances.clear();
        for entity in self.templates.values_mut() {
            entity.deinit(self.render.context());
        }
        if let Some(terrain) = self.terrain.as_mut() {
            terrain.terrain.texture.deinit(self.render.context());
        }
        self.render.deinit();
    }
}