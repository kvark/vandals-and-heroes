use blade_graphics as gpu;
use std::ptr;

const DEPTH_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Depth32Float;

#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: u32,
    pub tex_coords: [f32; 2],
    pub _pad: [u32; 2],
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CameraParams {
    pos: [f32; 3],
    pad: u32,
    rot: [f32; 4],
    half_plane: [f32; 2],
    clip: [f32; 2],
}

#[derive(blade_macros::ShaderData)]
struct GlobalData {
    g_camera: CameraParams,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct RayParams {
    march_count: u32,
    march_closest_power: f32,
    bisect_count: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TerrainParams {
    radius_start: f32,
    radius_end: f32,
    length: f32,
}

#[derive(blade_macros::ShaderData)]
struct TerrainData {
    g_ray_params: RayParams,
    g_terrain_params: TerrainParams,
    g_terrain: gpu::TextureView,
    g_terrain_sampler: gpu::Sampler,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct ModelParams {
    transform: [[f32; 4]; 3],
    base_color_factor: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct ModelData {
    g_vertices: gpu::BufferPiece,
    g_params: ModelParams,
    g_base_color: gpu::TextureView,
    g_normal: gpu::TextureView,
    g_sampler: gpu::Sampler,
}

#[derive(Default)]
struct DummyResources {
    white_texture: super::Texture,
    black_opaque_texture: super::Texture,
}

impl DummyResources {
    fn new(context: &gpu::Context, encoder: &mut gpu::CommandEncoder) -> (Self, gpu::Buffer) {
        let mut this = Self::default();
        // create resources
        this.white_texture.init_2d(
            context,
            "dummy/white",
            gpu::TextureFormat::Rgba8Unorm,
            gpu::Extent::default(),
            gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        );
        encoder.init_texture(this.white_texture.raw);
        this.black_opaque_texture.init_2d(
            context,
            "dummy/black-opaque",
            gpu::TextureFormat::Rgba8Unorm,
            gpu::Extent::default(),
            gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        );
        encoder.init_texture(this.black_opaque_texture.raw);
        // initialize contents
        let data = [0xFFFFFFFFu32, 0xFF000000];
        let size = data.len() * std::mem::size_of::<u32>();
        let stage = context.create_buffer(gpu::BufferDesc {
            name: "dummy/stage",
            size: size as u64,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr() as *const u8, stage.data(), size);
        }
        let mut transfer = encoder.transfer("dummy init");
        transfer.copy_buffer_to_texture(
            stage.at(0),
            4,
            this.white_texture.raw.into(),
            gpu::Extent::default(),
        );
        transfer.copy_buffer_to_texture(
            stage.at(4),
            4,
            this.black_opaque_texture.raw.into(),
            gpu::Extent::default(),
        );
        // done
        (this, stage)
    }

    fn deinit(&mut self, context: &gpu::Context) {
        self.white_texture.deinit(context);
        self.black_opaque_texture.deinit(context);
    }
}

pub struct Render {
    aspect_ratio: f32,
    ray_params: RayParams,
    terrain_params: TerrainParams,
    depth_texture: super::Texture,
    terrain_texture: super::Texture,
    terrain_sampler: gpu::Sampler,
    terrain_draw_pipeline: gpu::RenderPipeline,
    model_draw_pipeline: gpu::RenderPipeline,
    model_sampler: gpu::Sampler,
    dummy: DummyResources,
    command_encoder: gpu::CommandEncoder,
    last_submission: Option<super::Submission>,
    gpu_surface: gpu::Surface,
    gpu_context: gpu::Context,
}

impl Render {
    fn make_surface_config(size: gpu::Extent) -> gpu::SurfaceConfig {
        gpu::SurfaceConfig {
            size,
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Recent,
            ..Default::default()
        }
    }

    pub fn new(
        gpu_context: gpu::Context,
        mut gpu_surface: gpu::Surface,
        extent: gpu::Extent,
    ) -> Self {
        let mut command_encoder = gpu_context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });
        command_encoder.start();
        let (dummy, dummy_stage) = DummyResources::new(&gpu_context, &mut command_encoder);
        let last_submission = Some(super::Submission {
            sync_point: gpu_context.submit(&mut command_encoder),
            temp_buffers: vec![dummy_stage],
        });

        gpu_context.reconfigure_surface(&mut gpu_surface, Self::make_surface_config(extent));
        let surface_info = gpu_surface.info();

        let terrain_shader = {
            let source = std::fs::read_to_string("shaders/terrain-draw.wgsl").unwrap();
            gpu_context.create_shader(gpu::ShaderDesc { source: &source })
        };
        let model_shader = {
            let source = std::fs::read_to_string("shaders/model-draw.wgsl").unwrap();
            gpu_context.create_shader(gpu::ShaderDesc { source: &source })
        };
        let global_layout = <GlobalData as gpu::ShaderData>::layout();
        let terrain_layout = <TerrainData as gpu::ShaderData>::layout();
        let model_layout = <ModelData as gpu::ShaderData>::layout();
        model_shader.check_struct_size::<Vertex>();

        let mut depth_texture = super::Texture::default();
        depth_texture.init_2d(
            &gpu_context,
            "depth",
            DEPTH_FORMAT,
            extent,
            gpu::TextureUsage::TARGET,
        );

        Self {
            aspect_ratio: extent.width as f32 / extent.height as f32,
            ray_params: RayParams::default(),
            terrain_params: TerrainParams::default(),
            depth_texture,
            terrain_texture: super::Texture::default(),
            terrain_sampler: gpu_context.create_sampler(gpu::SamplerDesc {
                name: "terrain",
                address_modes: [
                    gpu::AddressMode::Repeat,
                    gpu::AddressMode::ClampToEdge,
                    gpu::AddressMode::ClampToEdge,
                ],
                mag_filter: gpu::FilterMode::Linear,
                min_filter: gpu::FilterMode::Linear,
                ..Default::default()
            }),
            terrain_draw_pipeline: gpu_context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "terrain-draw",
                data_layouts: &[&global_layout, &terrain_layout],
                vertex: terrain_shader.at("vs_terrain_draw"),
                vertex_fetches: &[],
                primitive: gpu::PrimitiveState::default(),
                depth_stencil: Some(gpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: gpu::CompareFunction::Always,
                    stencil: gpu::StencilState::default(),
                    bias: gpu::DepthBiasState::default(),
                }),
                fragment: Some(terrain_shader.at("fs_terrain_ray_march")),
                color_targets: &[surface_info.format.into()],
                multisample_state: Default::default(),
            }),
            model_draw_pipeline: gpu_context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "model-draw",
                data_layouts: &[&global_layout, &model_layout],
                vertex: model_shader.at("vs_model"),
                vertex_fetches: &[],
                primitive: gpu::PrimitiveState::default(),
                depth_stencil: Some(gpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: gpu::CompareFunction::Less,
                    stencil: gpu::StencilState::default(),
                    bias: gpu::DepthBiasState::default(),
                }),
                fragment: Some(model_shader.at("fs_model")),
                color_targets: &[surface_info.format.into()],
                multisample_state: Default::default(),
            }),
            model_sampler: gpu_context.create_sampler(gpu::SamplerDesc {
                name: "model",
                address_modes: [gpu::AddressMode::ClampToEdge; 3],
                mag_filter: gpu::FilterMode::Linear,
                min_filter: gpu::FilterMode::Linear,
                ..Default::default()
            }),
            dummy,
            command_encoder,
            last_submission,
            gpu_surface,
            gpu_context,
        }
    }

    pub fn wait_for_gpu(&mut self) {
        if let Some(sub) = self.last_submission.take() {
            self.gpu_context.wait_for(&sub.sync_point, !0);
            for buffer in sub.temp_buffers {
                self.gpu_context.destroy_buffer(buffer);
            }
        }
    }

    pub fn deinit(&mut self) {
        self.depth_texture.deinit(&self.gpu_context);
        self.terrain_texture.deinit(&self.gpu_context);
        self.gpu_context.destroy_sampler(self.terrain_sampler);
        self.gpu_context.destroy_sampler(self.model_sampler);
        self.dummy.deinit(&self.gpu_context);

        self.gpu_context
            .destroy_render_pipeline(&mut self.model_draw_pipeline);
        self.gpu_context
            .destroy_render_pipeline(&mut self.terrain_draw_pipeline);
        self.gpu_context
            .destroy_command_encoder(&mut self.command_encoder);
        self.gpu_context.destroy_surface(&mut self.gpu_surface);
    }

    pub fn context(&self) -> &gpu::Context {
        &self.gpu_context
    }

    pub fn resize(&mut self, extent: gpu::Extent) {
        self.wait_for_gpu();
        self.gpu_context
            .reconfigure_surface(&mut self.gpu_surface, Self::make_surface_config(extent));

        self.aspect_ratio = extent.width as f32 / extent.height as f32;
        self.depth_texture.init_2d(
            &self.gpu_context,
            "depth",
            DEPTH_FORMAT,
            extent,
            gpu::TextureUsage::TARGET,
        );
    }

    pub fn start_loading(&mut self) -> super::Loader {
        super::Loader::new(&self.gpu_context, &mut self.command_encoder)
    }

    pub fn accept_submission(&mut self, submission: super::Submission) {
        self.wait_for_gpu();
        self.last_submission = Some(submission);
    }

    pub fn set_map(&mut self, texture: super::Texture, config: &super::config::Map) {
        self.terrain_texture.deinit(&self.gpu_context);
        self.terrain_texture = texture;
        self.terrain_params = TerrainParams {
            radius_start: config.radius.start,
            radius_end: config.radius.end,
            length: config.length,
        };
    }

    pub fn set_ray_params(&mut self, rc: &super::config::Ray) {
        self.ray_params = RayParams {
            march_count: rc.march_count,
            march_closest_power: rc.march_closest_power,
            bisect_count: rc.bisect_count,
        };
    }

    pub fn draw(&mut self, camera: &super::Camera, objects: &[&super::Object]) {
        let half_y = (0.5 * camera.fov_y).tan();
        let camera_params = CameraParams {
            pos: camera.pos.into(),
            pad: 0,
            rot: (*camera.rot.as_vector()).into(),
            half_plane: [self.aspect_ratio * half_y, half_y],
            clip: [camera.clip.start, camera.clip.end],
        };

        let frame = self.gpu_surface.acquire_frame();
        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());
        self.command_encoder.init_texture(self.depth_texture.raw);

        if let mut pass = self.command_encoder.render(
            "draw",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: frame.texture_view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: Some(gpu::RenderTarget {
                    view: self.depth_texture.view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }),
            },
        ) {
            if let mut pen = pass.with(&self.terrain_draw_pipeline) {
                pen.bind(
                    0,
                    &GlobalData {
                        g_camera: camera_params,
                    },
                );
                pen.bind(
                    1,
                    &TerrainData {
                        g_ray_params: self.ray_params,
                        g_terrain_params: self.terrain_params,
                        g_terrain: self.terrain_texture.view,
                        g_terrain_sampler: self.terrain_sampler,
                    },
                );
                pen.draw(0, 3, 0, 1);
            }
            if let mut pen = pass.with(&self.model_draw_pipeline) {
                pen.bind(
                    0,
                    &GlobalData {
                        g_camera: camera_params,
                    },
                );
                for object in objects {
                    let base_transform = object.transform.to_matrix();
                    for geometry in object.model.geometries.iter() {
                        let material = &object.model.materials[geometry.material_index];
                        pen.bind(
                            1,
                            &ModelData {
                                g_vertices: geometry.buffer.into(),
                                g_params: ModelParams {
                                    transform: geometry.rendering_transform(&base_transform),
                                    base_color_factor: material.base_color_factor,
                                },
                                g_base_color: match material.base_color_texture {
                                    Some(ref t) => t.view,
                                    None => self.dummy.white_texture.view,
                                },
                                g_normal: match material.normal_texture {
                                    Some(ref t) => t.view,
                                    None => self.dummy.black_opaque_texture.view,
                                },
                                g_sampler: self.model_sampler,
                            },
                        );
                        match geometry.index_type {
                            Some(ty) => {
                                let index_buf = geometry.buffer.at(geometry.index_offset);
                                pen.draw_indexed(
                                    index_buf,
                                    ty,
                                    3 * geometry.triangle_count,
                                    0,
                                    0,
                                    1,
                                );
                            }
                            None => {
                                let vr = &geometry.vertex_range;
                                pen.draw(vr.start, vr.end - vr.start, 0, 1);
                            }
                        }
                    }
                }
            }
        }

        self.command_encoder.present(frame);
        let sync_point = self.gpu_context.submit(&mut self.command_encoder);
        self.accept_submission(super::Submission {
            sync_point,
            temp_buffers: Vec::new(),
        });
    }
}
