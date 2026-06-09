use crate::Terrain;
use blade_graphics as gpu;
use std::ptr;

const DEPTH_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Depth32Float;
// R16Float instead of Depth32Float so the bilinear-sampler is guaranteed to
// filter on every backend (depth formats need the `float32-filterable`
// extension to filter, and some drivers silently fall back to nearest
// — which is what shows up as "layered" stair-stepped shadows). We use
// hardware MIN blend instead of a depth Less-test, so smaller cyl_depth
// values (closer to "sun at radial infinity") still win.
const SHADOW_FORMAT: gpu::TextureFormat = gpu::TextureFormat::R16Float;
// Default until the heightmap dimensions are known; `set_shadow_extent` will
// resize the texture to match the loaded map's resolution.
const DEFAULT_SHADOW_EXTENT: gpu::Extent = gpu::Extent {
    width: 1024,
    height: 1024,
    depth: 1,
};

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

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CylParams {
    radius_start: f32,
    radius_end: f32,
    length: f32,
    /// Radial "sun-at-infinity" plane for the shadow map; r in [radius_start,
    /// shadow_radius_top] maps to depth in [1, 0]. Chosen wider than radius_end
    /// so vehicles sitting above the heightmap peaks fit in the depth range.
    shadow_radius_top: f32,
}

#[derive(blade_macros::ShaderData)]
struct MainGlobalData {
    g_camera: CameraParams,
    g_cyl: CylParams,
    g_shadow: gpu::TextureView,
    g_shadow_sampler: gpu::Sampler,
    g_environment: gpu::TextureView,
    g_env_sampler: gpu::Sampler,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct RayParams {
    march_count: u32,
    march_closest_power: f32,
    bisect_count: u32,
}

#[derive(blade_macros::ShaderData)]
struct TerrainData {
    g_ray_params: RayParams,
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

// Shadow pass bind groups (note: g_shadow is the render target during these passes,
// so it MUST NOT appear as a resource here).

#[derive(blade_macros::ShaderData)]
struct ShadowGlobalData {
    g_cyl: CylParams,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct ShadowModelParams {
    transform: [[f32; 4]; 3],
}

#[derive(blade_macros::ShaderData)]
struct ShadowModelData {
    g_vertices: gpu::BufferPiece,
    g_params: ShadowModelParams,
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
    depth_texture: super::Texture,
    shadow_texture: super::Texture,
    terrain_sampler: gpu::Sampler,
    env_sampler: gpu::Sampler,
    shadow_sampler: gpu::Sampler,
    terrain_draw_pipeline: gpu::RenderPipeline,
    model_draw_pipeline: gpu::RenderPipeline,
    shadow_model_pipeline: gpu::RenderPipeline,
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

        // Prepend shaders/common.wgsl into every shader so shared constants,
        // helpers, and the g_cyl binding live in one place.
        let common_src = std::fs::read_to_string("shaders/common.wgsl").unwrap();
        let load_shader = |path: &str| -> gpu::Shader {
            let body = std::fs::read_to_string(path).unwrap();
            let source = format!("{common_src}\n{body}");
            gpu_context.create_shader(gpu::ShaderDesc {
                source: &source,
                naga_module: None,
            })
        };
        let terrain_shader = load_shader("shaders/terrain-draw.wgsl");
        let model_shader = load_shader("shaders/model-draw.wgsl");
        let shadow_shader = load_shader("shaders/shadow.wgsl");
        let main_global_layout = <MainGlobalData as gpu::ShaderData>::layout();
        let terrain_layout = <TerrainData as gpu::ShaderData>::layout();
        let model_layout = <ModelData as gpu::ShaderData>::layout();
        let shadow_global_layout = <ShadowGlobalData as gpu::ShaderData>::layout();
        let shadow_model_layout = <ShadowModelData as gpu::ShaderData>::layout();
        model_shader.check_struct_size::<Vertex>();

        let mut depth_texture = super::Texture::default();
        depth_texture.init_2d(
            &gpu_context,
            "depth",
            DEPTH_FORMAT,
            extent,
            gpu::TextureUsage::TARGET,
        );

        let mut shadow_texture = super::Texture::default();
        shadow_texture.init_2d(
            &gpu_context,
            "shadow",
            SHADOW_FORMAT,
            DEFAULT_SHADOW_EXTENT,
            gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        );

        Self {
            aspect_ratio: extent.width as f32 / extent.height as f32,
            ray_params: RayParams::default(),
            depth_texture,
            shadow_texture,
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
            env_sampler: gpu_context.create_sampler(gpu::SamplerDesc {
                name: "environment",
                address_modes: [
                    gpu::AddressMode::Repeat,
                    gpu::AddressMode::ClampToEdge,
                    gpu::AddressMode::ClampToEdge,
                ],
                mag_filter: gpu::FilterMode::Linear,
                min_filter: gpu::FilterMode::Linear,
                ..Default::default()
            }),
            shadow_sampler: gpu_context.create_sampler(gpu::SamplerDesc {
                name: "shadow",
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
                data_layouts: &[&main_global_layout, &terrain_layout],
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
                data_layouts: &[&main_global_layout, &model_layout],
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
            shadow_model_pipeline: gpu_context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "shadow-model",
                data_layouts: &[&shadow_global_layout, &shadow_model_layout],
                vertex: shadow_shader.at("vs_shadow_model"),
                vertex_fetches: &[],
                primitive: gpu::PrimitiveState::default(),
                depth_stencil: None,
                fragment: Some(shadow_shader.at("fs_shadow_model")),
                color_targets: &[gpu::ColorTargetState {
                    format: SHADOW_FORMAT,
                    blend: Some(gpu::BlendState {
                        color: gpu::BlendComponent {
                            src_factor: gpu::BlendFactor::One,
                            dst_factor: gpu::BlendFactor::One,
                            operation: gpu::BlendOperation::Min,
                        },
                        alpha: gpu::BlendComponent::REPLACE,
                    }),
                    write_mask: gpu::ColorWrites::RED,
                }],
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
            let _ = self.gpu_context.wait_for(&sub.sync_point, !0);
            for buffer in sub.temp_buffers {
                self.gpu_context.destroy_buffer(buffer);
            }
        }
    }

    pub fn deinit(&mut self) {
        self.depth_texture.deinit(&self.gpu_context);
        self.shadow_texture.deinit(&self.gpu_context);
        self.gpu_context.destroy_sampler(self.terrain_sampler);
        self.gpu_context.destroy_sampler(self.env_sampler);
        self.gpu_context.destroy_sampler(self.shadow_sampler);
        self.gpu_context.destroy_sampler(self.model_sampler);
        self.dummy.deinit(&self.gpu_context);

        self.gpu_context
            .destroy_render_pipeline(&mut self.model_draw_pipeline);
        self.gpu_context
            .destroy_render_pipeline(&mut self.terrain_draw_pipeline);
        self.gpu_context
            .destroy_render_pipeline(&mut self.shadow_model_pipeline);
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

    pub fn start_loading(&mut self) -> super::Loader<'_> {
        super::Loader::new(&self.gpu_context, &mut self.command_encoder)
    }

    pub fn accept_submission(&mut self, submission: super::Submission) {
        self.wait_for_gpu();
        self.last_submission = Some(submission);
    }

    /// Resize the cylindrical shadow texture to match the loaded heightmap.
    /// Should be called once after the terrain PNG is loaded so a shadow texel
    /// corresponds 1:1 to a heightmap texel.
    pub fn set_shadow_extent(&mut self, extent: gpu::Extent) {
        self.wait_for_gpu();
        self.shadow_texture.init_2d(
            &self.gpu_context,
            "shadow",
            SHADOW_FORMAT,
            extent,
            gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        );
        log::info!(
            "Shadow texture sized to {}x{} (R16Float, {} MiB)",
            extent.width,
            extent.height,
            (extent.width as u64 * extent.height as u64 * 2) >> 20,
        );
    }

    pub fn set_ray_params(&mut self, rc: &super::RayConfig) {
        self.ray_params = RayParams {
            march_count: rc.march_count,
            march_closest_power: rc.march_closest_power,
            bisect_count: rc.bisect_count,
        };
    }

    pub fn draw(
        &mut self,
        camera: &super::Camera,
        terrain: &Terrain,
        models: &Vec<&super::ModelInstance>,
    ) {
        let half_y = (0.5 * camera.fov_y).tan();
        let camera_params = CameraParams {
            pos: camera.pos.into(),
            pad: 0,
            rot: (*camera.rot.as_vector()).into(),
            half_plane: [self.aspect_ratio * half_y, half_y],
            clip: [camera.clip.start, camera.clip.end],
        };
        let cyl_params = CylParams {
            radius_start: terrain.config.radius.start,
            radius_end: terrain.config.radius.end,
            length: terrain.config.length,
            shadow_radius_top: 2.0 * terrain.config.radius.end - terrain.config.radius.start,
        };
        // Fall back to the white dummy texture so the env-modulated lighting still
        // shows the albedo when no environment map is configured.
        let env_view = terrain
            .env_texture
            .as_ref()
            .map(|t| t.view)
            .unwrap_or(self.dummy.white_texture.view);

        let frame = self.gpu_surface.acquire_frame();
        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());
        self.command_encoder.init_texture(self.depth_texture.raw);
        self.command_encoder.init_texture(self.shadow_texture.raw);

        // ===== Shadow pass: rebuild the cylindrical shadow map every frame =====
        if let mut pass = self.command_encoder.render(
            "shadow",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: self.shadow_texture.view,
                    // Clear to white (= 1.0 in R16Float) = "no occluder, full sky".
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        ) {
            // Terrain is its own topmost surface, so we don't bake it; any
            // dynamic mesh below writes a smaller depth (Min-blend) and shows
            // up as a cast shadow at shading time.
            if let mut pen = pass.with(&self.shadow_model_pipeline) {
                pen.bind(0, &ShadowGlobalData { g_cyl: cyl_params });
                for model_instance in models {
                    let base_transform = model_instance.transform.to_matrix();
                    for (gi, geometry) in model_instance.model.geometries.iter().enumerate() {
                        if let Some(filter) = model_instance.geometry_filter.as_ref() {
                            if !filter.contains(&gi) {
                                continue;
                            }
                        }
                        pen.bind(
                            1,
                            &ShadowModelData {
                                g_vertices: geometry.buffer.into(),
                                g_params: ShadowModelParams {
                                    transform: geometry.rendering_transform(&base_transform),
                                },
                            },
                        );
                        // Two instances. The first renders the model at its
                        // unwrapped θ; the second is shifted by ±2π so any
                        // half that would otherwise clip off the side of
                        // the shadow map — because the model straddles
                        // θ = ±π — appears on the opposite edge instead.
                        // See vs_shadow_model for the full reasoning.
                        match geometry.index_type {
                            Some(ty) => {
                                let index_buf = geometry.buffer.at(geometry.index_offset);
                                pen.draw_indexed(
                                    index_buf,
                                    ty,
                                    3 * geometry.triangle_count,
                                    0,
                                    0,
                                    2,
                                );
                            }
                            None => {
                                let vr = &geometry.vertex_range;
                                pen.draw(vr.start, vr.end - vr.start, 0, 2);
                            }
                        }
                    }
                }
            }
        }

        // ===== Main pass =====
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
            let main_global = MainGlobalData {
                g_camera: camera_params,
                g_cyl: cyl_params,
                g_shadow: self.shadow_texture.view,
                g_shadow_sampler: self.shadow_sampler,
                g_environment: env_view,
                g_env_sampler: self.env_sampler,
            };

            if let mut pen = pass.with(&self.terrain_draw_pipeline) {
                pen.bind(0, &main_global);
                pen.bind(
                    1,
                    &TerrainData {
                        g_ray_params: self.ray_params,
                        g_terrain: terrain.texture.view,
                        g_terrain_sampler: self.terrain_sampler,
                    },
                );
                pen.draw(0, 3, 0, 1);
            }
            if let mut pen = pass.with(&self.model_draw_pipeline) {
                pen.bind(0, &main_global);
                for model_instance in models {
                    let base_transform = model_instance.transform.to_matrix();
                    for (gi, geometry) in model_instance.model.geometries.iter().enumerate() {
                        if let Some(filter) = model_instance.geometry_filter.as_ref() {
                            if !filter.contains(&gi) {
                                continue;
                            }
                        }
                        let material = &model_instance.model.materials[geometry.material_index];
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
