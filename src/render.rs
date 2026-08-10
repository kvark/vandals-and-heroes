use crate::Terrain;
use crate::config::WorldShape;
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

/// Distance at which a terrain chunk drops to the next LOD. Chunks are
/// ~5 m across on Fostral-scale maps, so the finest mesh covers everything
/// the player can inspect while the horizon renders at a fraction of the
/// triangles.
const LOD_DISTANCE: f32 = 48.0;

#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: u32,
    pub tex_coords: [f32; 2],
    pub _pad: [u32; 2],
}

impl gpu::Vertex for Vertex {
    fn layout() -> gpu::VertexLayout {
        gpu::VertexLayout {
            attributes: vec![
                (
                    "position",
                    gpu::VertexAttribute {
                        offset: 0,
                        format: gpu::VertexFormat::F32Vec3,
                    },
                ),
                (
                    "normal",
                    gpu::VertexAttribute {
                        offset: 12,
                        format: gpu::VertexFormat::U32,
                    },
                ),
                (
                    "tex_coords",
                    gpu::VertexAttribute {
                        offset: 16,
                        format: gpu::VertexFormat::F32Vec2,
                    },
                ),
            ],
            stride: std::mem::size_of::<Vertex>() as u32,
        }
    }
}

/// Terrain chunk vertex: a bare world position. Everything else — colour,
/// normal, AO — is derived per-fragment from the terrain texture.
#[repr(C)]
pub struct TerrainVertex {
    pub position: [f32; 3],
}

impl gpu::Vertex for TerrainVertex {
    fn layout() -> gpu::VertexLayout {
        gpu::VertexLayout {
            attributes: vec![(
                "position",
                gpu::VertexAttribute {
                    offset: 0,
                    format: gpu::VertexFormat::F32Vec3,
                },
            )],
            stride: std::mem::size_of::<TerrainVertex>() as u32,
        }
    }
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
    /// 0 = cylinder, 1 = sphere, 2 = torus — same discriminants as the
    /// SHAPE_* constants in shaders/common.wgsl.
    world_shape: u32,
    /// Torus centreline radius (`length / 2π`); unused for other shapes.
    major_radius: f32,
    /// Output gamma exponent — see shaders/common.wgsl. 1.0 on sRGB
    /// surfaces, 1/2.2 on linear (WebGL2) ones.
    gamma: f32,
    /// Cast-shadow sampling toggle — see shaders/common.wgsl. Off on the
    /// web, where blade's GLES shadow-target writes are unreliable.
    shadows_enabled: u32,
}

impl CylParams {
    fn new(config: &crate::MapConfig, gamma: f32) -> Self {
        Self {
            radius_start: config.radius.start,
            radius_end: config.radius.end,
            length: config.length,
            shadow_radius_top: 2.0 * config.radius.end - config.radius.start,
            world_shape: match config.shape {
                WorldShape::Cylinder => 0,
                WorldShape::Sphere => 1,
                WorldShape::Torus => 2,
            },
            major_radius: config.length / std::f32::consts::TAU,
            gamma,
            shadows_enabled: !cfg!(target_arch = "wasm32") as u32,
        }
    }
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

#[derive(blade_macros::ShaderData)]
struct TerrainMeshData {
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
        encoder.init_texture(this.white_texture.raw());
        this.black_opaque_texture.init_2d(
            context,
            "dummy/black-opaque",
            gpu::TextureFormat::Rgba8Unorm,
            gpu::Extent::default(),
            gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        );
        encoder.init_texture(this.black_opaque_texture.raw());
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
        // Push the CPU-side contents to the GL buffer object — a no-op on
        // Vulkan/Metal, required on the WebGL2 backend where mapped memory
        // is a CPU mirror.
        context.sync_buffer(stage);
        let mut transfer = encoder.transfer("dummy init");
        transfer.copy_buffer_to_texture(
            stage.at(0),
            4,
            this.white_texture.raw().into(),
            gpu::Extent::default(),
        );
        transfer.copy_buffer_to_texture(
            stage.at(4),
            4,
            this.black_opaque_texture.raw().into(),
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

/// One frustum-surviving chunk, with the LOD chosen for its distance.
struct ChunkDraw {
    chunk_index: usize,
    lod: usize,
    distance: f32,
}

/// Pick the visible chunks, their LODs, and a near-to-far order.
///
/// Culling happens in camera space: the 8 corners of a chunk's world AABB
/// are tested against the 6 frustum half-spaces of the game's custom
/// projection (X-right, Y-down, Z-forward; |x| < half_plane.x · z).
fn cull_chunks(
    camera: &super::Camera,
    aspect_half_plane: [f32; 2],
    terrain: &Terrain,
) -> Vec<ChunkDraw> {
    profiling::scope!("Render::cull_chunks");
    let inv_rot = camera.rot.inverse();
    let (hx, hy) = (aspect_half_plane[0], aspect_half_plane[1]);
    let mut draws = Vec::with_capacity(terrain.chunks.len());
    for (chunk_index, chunk) in terrain.chunks.iter().enumerate() {
        let mut all_out = [true; 6];
        for corner in 0..8 {
            let world = nalgebra::Vector3::new(
                if corner & 1 == 0 {
                    chunk.min[0]
                } else {
                    chunk.max[0]
                },
                if corner & 2 == 0 {
                    chunk.min[1]
                } else {
                    chunk.max[1]
                },
                if corner & 4 == 0 {
                    chunk.min[2]
                } else {
                    chunk.max[2]
                },
            );
            let p = inv_rot * (world - camera.pos);
            let tests = [
                p.z < camera.clip.start,
                p.z > camera.clip.end,
                p.x > hx * p.z,
                -p.x > hx * p.z,
                p.y > hy * p.z,
                -p.y > hy * p.z,
            ];
            for (out, test) in all_out.iter_mut().zip(tests) {
                *out &= test;
            }
        }
        if all_out.iter().any(|&out| out) {
            continue;
        }
        let center = chunk.center;
        let distance = (nalgebra::Vector3::new(center[0], center[1], center[2]) - camera.pos)
            .norm();
        let lod = ((distance / LOD_DISTANCE).max(1.0).log2().floor() as usize)
            .min(chunk.lods.len() - 1);
        draws.push(ChunkDraw {
            chunk_index,
            lod,
            distance,
        });
    }
    // Near to far: the terrain fragment shader is not cheap (it re-derives
    // the surface gradient and AO per pixel), so letting the depth test
    // reject occluded chunks before shading them is worth the sort.
    draws.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
    draws
}

pub struct Render {
    aspect_ratio: f32,
    /// Cached colour format of the surface, used both for the on-screen
    /// frame and for off-screen snapshot targets so they share pipelines.
    surface_format: gpu::TextureFormat,
    /// See `CameraParams::gamma`.
    gamma: f32,
    depth_texture: super::Texture,
    shadow_texture: super::Texture,
    terrain_sampler: gpu::Sampler,
    env_sampler: gpu::Sampler,
    shadow_sampler: gpu::Sampler,
    sky_pipeline: gpu::RenderPipeline,
    terrain_mesh_pipeline: gpu::RenderPipeline,
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
        // helpers, and the g_cyl binding live in one place. Natively the
        // sources come from disk (live shader iteration); on the web they
        // are embedded — there is no filesystem under WebGL2.
        #[cfg(not(target_arch = "wasm32"))]
        let read_source = |name: &str| -> String {
            std::fs::read_to_string(format!("shaders/{name}.wgsl")).unwrap()
        };
        #[cfg(target_arch = "wasm32")]
        let read_source = |name: &str| -> String {
            match name {
                "common" => include_str!("../shaders/common.wgsl"),
                "terrain-mesh" => include_str!("../shaders/terrain-mesh.wgsl"),
                "model-draw" => include_str!("../shaders/model-draw.wgsl"),
                "shadow" => include_str!("../shaders/shadow.wgsl"),
                other => panic!("unknown shader {other}"),
            }
            .to_string()
        };
        let common_src = read_source("common");
        let load_shader = |name: &str| -> gpu::Shader {
            let body = read_source(name);
            let source = format!("{common_src}\n{body}");
            gpu_context.create_shader(gpu::ShaderDesc {
                source: &source,
                naga_module: None,
            })
        };
        let terrain_shader = load_shader("terrain-mesh");
        let model_shader = load_shader("model-draw");
        let shadow_shader = load_shader("shadow");
        let main_global_layout = <MainGlobalData as gpu::ShaderData>::layout();
        let terrain_layout = <TerrainMeshData as gpu::ShaderData>::layout();
        let model_layout = <ModelData as gpu::ShaderData>::layout();
        let shadow_global_layout = <ShadowGlobalData as gpu::ShaderData>::layout();
        let shadow_model_layout = <ShadowModelData as gpu::ShaderData>::layout();
        let model_vertex_layout = <Vertex as gpu::Vertex>::layout();
        let terrain_vertex_layout = <TerrainVertex as gpu::Vertex>::layout();

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
            gamma: match surface_info.format {
                gpu::TextureFormat::Rgba8UnormSrgb | gpu::TextureFormat::Bgra8UnormSrgb => 1.0,
                _ => 1.0 / 2.2,
            },
            surface_format: surface_info.format,
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
            sky_pipeline: gpu_context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "sky",
                data_layouts: &[&main_global_layout],
                vertex: terrain_shader.at("vs_sky"),
                vertex_fetches: &[],
                primitive: gpu::PrimitiveState::default(),
                depth_stencil: Some(gpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: gpu::CompareFunction::Always,
                    stencil: gpu::StencilState::default(),
                    bias: gpu::DepthBiasState::default(),
                }),
                fragment: Some(terrain_shader.at("fs_sky")),
                color_targets: &[surface_info.format.into()],
                multisample_state: Default::default(),
            }),
            terrain_mesh_pipeline: gpu_context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "terrain-mesh",
                data_layouts: &[&main_global_layout, &terrain_layout],
                vertex: terrain_shader.at("vs_terrain_mesh"),
                vertex_fetches: &[gpu::VertexFetchState {
                    layout: &terrain_vertex_layout,
                    instanced: false,
                }],
                primitive: gpu::PrimitiveState::default(),
                depth_stencil: Some(gpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: gpu::CompareFunction::Less,
                    stencil: gpu::StencilState::default(),
                    bias: gpu::DepthBiasState::default(),
                }),
                fragment: Some(terrain_shader.at("fs_terrain_mesh")),
                color_targets: &[surface_info.format.into()],
                multisample_state: Default::default(),
            }),
            model_draw_pipeline: gpu_context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "model-draw",
                data_layouts: &[&main_global_layout, &model_layout],
                vertex: model_shader.at("vs_model"),
                vertex_fetches: &[gpu::VertexFetchState {
                    layout: &model_vertex_layout,
                    instanced: false,
                }],
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
                vertex_fetches: &[gpu::VertexFetchState {
                    layout: &model_vertex_layout,
                    instanced: false,
                }],
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
            .destroy_render_pipeline(&mut self.sky_pipeline);
        self.gpu_context
            .destroy_render_pipeline(&mut self.terrain_mesh_pipeline);
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

    /// Record the shadow pass and the main colour pass into the shared
    /// command encoder. Used both by the on-screen `draw` and the off-screen
    /// `render_to_buffer` so the two can never disagree on what a frame is.
    fn encode_frame(
        &mut self,
        target_view: gpu::TextureView,
        camera: &super::Camera,
        half_plane: [f32; 2],
        terrain: &Terrain,
        models: &Vec<&super::ModelInstance>,
    ) {
        let camera_params = CameraParams {
            pos: camera.pos.into(),
            pad: 0,
            rot: (*camera.rot.as_vector()).into(),
            half_plane,
            clip: [camera.clip.start, camera.clip.end],
        };
        let cyl_params = CylParams::new(&terrain.config, self.gamma);
        // Fall back to the white dummy texture so the env-modulated lighting still
        // shows the albedo when no environment map is configured.
        let env_view = terrain
            .env_texture
            .as_ref()
            .map(|t| t.view())
            .unwrap_or_else(|| self.dummy.white_texture.view());
        let chunk_draws = cull_chunks(camera, half_plane, terrain);

        self.command_encoder.init_texture(self.depth_texture.raw());
        self.command_encoder.init_texture(self.shadow_texture.raw());

        // ===== Shadow pass: rebuild the shadow map every frame =====
        if let mut pass = self.command_encoder.render(
            "shadow",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: self.shadow_texture.view(),
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
                                g_params: ShadowModelParams {
                                    transform: geometry.rendering_transform(&base_transform),
                                },
                            },
                        );
                        pen.bind_vertex(0, geometry.buffer.at(0));
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
                    view: target_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: Some(gpu::RenderTarget {
                    view: self.depth_texture.view(),
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::White),
                    finish_op: gpu::FinishOp::Store,
                }),
            },
        ) {
            let main_global = MainGlobalData {
                g_camera: camera_params,
                g_cyl: cyl_params,
                g_shadow: self.shadow_texture.view(),
                g_shadow_sampler: self.shadow_sampler,
                g_environment: env_view,
                g_env_sampler: self.env_sampler,
            };

            if let mut pen = pass.with(&self.sky_pipeline) {
                pen.bind(0, &main_global);
                pen.draw(0, 3, 0, 1);
            }
            if let mut pen = pass.with(&self.terrain_mesh_pipeline) {
                pen.bind(0, &main_global);
                pen.bind(
                    1,
                    &TerrainMeshData {
                        g_terrain: terrain.texture.view(),
                        g_terrain_sampler: self.terrain_sampler,
                    },
                );
                for draw in &chunk_draws {
                    let chunk = &terrain.chunks[draw.chunk_index];
                    let (first, count) = chunk.lods[draw.lod];
                    if count == 0 {
                        continue;
                    }
                    pen.bind_vertex(0, chunk.buffer.at(0));
                    match chunk.index_offset {
                        Some(index_offset) => pen.draw_indexed(
                            chunk.buffer.at(index_offset + first as u64 * 4),
                            gpu::IndexType::U32,
                            count,
                            0,
                            0,
                            1,
                        ),
                        // Web: pre-expanded triangle list, same element ranges.
                        None => pen.draw(first, count, 0, 1),
                    }
                }
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
                                g_params: ModelParams {
                                    transform: geometry.rendering_transform(&base_transform),
                                    base_color_factor: material.base_color_factor,
                                },
                                g_base_color: match material.base_color_texture {
                                    Some(ref t) => t.view(),
                                    None => self.dummy.white_texture.view(),
                                },
                                g_normal: match material.normal_texture {
                                    Some(ref t) => t.view(),
                                    None => self.dummy.black_opaque_texture.view(),
                                },
                                g_sampler: self.model_sampler,
                            },
                        );
                        pen.bind_vertex(0, geometry.buffer.at(0));
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
    }

    pub fn draw(
        &mut self,
        camera: &super::Camera,
        terrain: &Terrain,
        models: &Vec<&super::ModelInstance>,
    ) {
        let half_y = (0.5 * camera.fov_y).tan();
        let half_plane = [self.aspect_ratio * half_y, half_y];

        let frame = self.gpu_surface.acquire_frame();
        self.command_encoder.start();
        self.command_encoder.init_texture(frame.texture());
        self.encode_frame(frame.texture_view(), camera, half_plane, terrain, models);
        self.command_encoder.present(frame);
        let sync_point = self.gpu_context.submit(&mut self.command_encoder);
        self.accept_submission(super::Submission {
            sync_point,
            temp_buffers: Vec::new(),
        });
    }

    /// Render one frame off-screen and read the pixels back to the CPU.
    /// `extent` must match the depth+shadow textures the renderer was
    /// configured with (call `resize` first if necessary). Returns BGRA bytes
    /// in row-major order, `extent.height` rows of `extent.width * 4` bytes.
    pub fn render_to_buffer(
        &mut self,
        camera: &super::Camera,
        terrain: &Terrain,
        models: &Vec<&super::ModelInstance>,
        extent: gpu::Extent,
    ) -> Vec<u8> {
        // Make sure any in-flight work that referenced the encoder's resources
        // has finished before we reuse it for the off-screen pass.
        self.wait_for_gpu();

        let half_y = (0.5 * camera.fov_y).tan();
        let aspect = extent.width as f32 / extent.height as f32;
        let half_plane = [aspect * half_y, half_y];

        // Off-screen colour target — same format as the surface so the
        // existing pipelines accept it.
        let target = self.gpu_context.create_texture(gpu::TextureDesc {
            name: "snapshot/color",
            format: self.surface_format,
            size: extent,
            sample_count: 1,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
            external: None,
        });
        let target_view = self.gpu_context.create_texture_view(
            target,
            gpu::TextureViewDesc {
                name: "snapshot/color",
                format: self.surface_format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let row_bytes = extent.width * 4;
        let readback_size = (row_bytes as u64) * (extent.height as u64);
        let readback = self.gpu_context.create_buffer(gpu::BufferDesc {
            name: "snapshot/readback",
            size: readback_size,
            memory: gpu::Memory::Shared,
        });

        self.command_encoder.start();
        self.command_encoder.init_texture(target);
        self.encode_frame(target_view, camera, half_plane, terrain, models);

        // Pull the rendered colour into the readback buffer.
        if let mut transfer = self.command_encoder.transfer("snapshot/copy") {
            transfer.copy_texture_to_buffer(target.into(), readback.into(), row_bytes, extent);
        }

        let sync_point = self.gpu_context.submit(&mut self.command_encoder);
        let _ = self.gpu_context.wait_for(&sync_point, !0);

        let bytes = unsafe {
            std::slice::from_raw_parts(readback.data(), readback_size as usize).to_vec()
        };

        self.gpu_context.destroy_texture_view(target_view);
        self.gpu_context.destroy_texture(target);
        self.gpu_context.destroy_buffer(readback);

        bytes
    }
}
