use blade_graphics as gpu;
use std::{fs, mem, ops::Range, slice};

const DEPTH_FORMAT: gpu::TextureFormat = gpu::TextureFormat::Depth32Float;

#[derive(Default)]
struct Texture {
    raw: gpu::Texture,
    view: gpu::TextureView,
}

impl Texture {
    fn new_2d(
        context: &gpu::Context,
        name: &str,
        format: gpu::TextureFormat,
        size: gpu::Extent,
        usage: gpu::TextureUsage,
    ) -> Self {
        let raw = context.create_texture(gpu::TextureDesc {
            name,
            format,
            size,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage,
        });
        let view = context.create_texture_view(
            raw,
            gpu::TextureViewDesc {
                name,
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        Self { raw, view }
    }

    fn deinit(&mut self, context: &gpu::Context) {
        if self.raw != gpu::Texture::default() {
            context.destroy_texture_view(mem::take(&mut self.view));
            context.destroy_texture(mem::take(&mut self.raw));
        }
    }
}

pub struct MapView {
    pub radius: Range<f32>,
    pub length: f32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CameraParams {
    pos: [f32; 4],
    rot: [f32; 4],
    fov: [f32; 2],
    pad: [f32; 2],
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct DrawParams {
    radius_start: f32,
    radius_end: f32,
    length: f32,
    pad: f32,
}

#[derive(blade_macros::ShaderData)]
struct DrawData {
    g_camera: CameraParams,
    g_params: DrawParams,
    g_map: gpu::TextureView,
    g_sampler: gpu::Sampler,
}

struct Submission {
    sync_point: gpu::SyncPoint,
    temp_buffers: Vec<gpu::Buffer>,
}

pub struct Render {
    draw_params: DrawParams,
    depth_texture: Texture,
    map_texture: Texture,
    map_sampler: gpu::Sampler,
    map_draw_pipeline: gpu::RenderPipeline,
    command_encoder: gpu::CommandEncoder,
    last_submission: Option<Submission>,
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
        let command_encoder = gpu_context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "main",
            buffer_count: 2,
        });

        gpu_context.reconfigure_surface(&mut gpu_surface, Self::make_surface_config(extent));
        let surface_info = gpu_surface.info();

        let source = std::fs::read_to_string("shaders/map-draw.wgsl").unwrap();
        let shader = gpu_context.create_shader(gpu::ShaderDesc { source: &source });
        let global_layout = <DrawData as gpu::ShaderData>::layout();
        Self {
            draw_params: DrawParams::default(),
            depth_texture: Texture::default(),
            map_texture: Texture::default(),
            map_sampler: gpu_context.create_sampler(gpu::SamplerDesc {
                name: "map",
                address_modes: [
                    gpu::AddressMode::Repeat,
                    gpu::AddressMode::ClampToEdge,
                    gpu::AddressMode::ClampToEdge,
                ],
                mag_filter: gpu::FilterMode::Linear,
                min_filter: gpu::FilterMode::Linear,
                ..Default::default()
            }),
            map_draw_pipeline: gpu_context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "map-draw",
                data_layouts: &[&global_layout],
                vertex: shader.at("vs_draw"),
                vertex_fetches: &[],
                primitive: gpu::PrimitiveState::default(),
                depth_stencil: Some(gpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: gpu::CompareFunction::Always,
                    stencil: gpu::StencilState::default(),
                    bias: gpu::DepthBiasState::default(),
                }),
                fragment: shader.at("fs_draw"),
                color_targets: &[surface_info.format.into()],
            }),
            command_encoder,
            last_submission: None,
            gpu_surface,
            gpu_context,
        }
    }

    fn wait_for_gpu(&mut self) {
        if let Some(sub) = self.last_submission.take() {
            self.gpu_context.wait_for(&sub.sync_point, !0);
            for buffer in sub.temp_buffers {
                self.gpu_context.destroy_buffer(buffer);
            }
        }
    }

    pub fn deinit(&mut self) {
        self.wait_for_gpu();

        self.depth_texture.deinit(&self.gpu_context);
        self.map_texture.deinit(&self.gpu_context);
        self.gpu_context.destroy_sampler(self.map_sampler);

        self.gpu_context
            .destroy_render_pipeline(&mut self.map_draw_pipeline);
        self.gpu_context
            .destroy_command_encoder(&mut self.command_encoder);
        self.gpu_context.destroy_surface(&mut self.gpu_surface);
    }

    pub fn resize(&mut self, extent: gpu::Extent) {
        self.wait_for_gpu();
        self.gpu_context
            .reconfigure_surface(&mut self.gpu_surface, Self::make_surface_config(extent));

        self.depth_texture.deinit(&self.gpu_context);
        self.depth_texture = Texture::new_2d(
            &self.gpu_context,
            "depth",
            DEPTH_FORMAT,
            extent,
            gpu::TextureUsage::TARGET,
        );
    }

    pub fn load_map(&mut self, mut reader: png::Reader<fs::File>, view: MapView) {
        self.draw_params = DrawParams {
            radius_start: view.radius.start,
            radius_end: view.radius.end,
            length: view.length,
            pad: 0.0,
        };
        self.map_texture.deinit(&self.gpu_context);

        let stage_buffer = self.gpu_context.create_buffer(gpu::BufferDesc {
            name: "map stage",
            size: reader.output_buffer_size() as u64,
            memory: gpu::Memory::Upload,
        });
        let info = reader
            .next_frame(unsafe {
                slice::from_raw_parts_mut(stage_buffer.data(), reader.output_buffer_size())
            })
            .unwrap();

        let extent = gpu::Extent {
            width: info.width,
            height: info.height,
            depth: 1,
        };
        self.map_texture = Texture::new_2d(
            &self.gpu_context,
            "map",
            gpu::TextureFormat::Rgba8Unorm,
            extent,
            gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        );

        self.command_encoder.start();
        self.command_encoder.init_texture(self.map_texture.raw);
        if let mut pass = self.command_encoder.transfer("map init") {
            pass.copy_buffer_to_texture(
                stage_buffer.into(),
                info.width * 4,
                self.map_texture.raw.into(),
                extent,
            );
        }

        let sync_point = self.gpu_context.submit(&mut self.command_encoder);
        self.wait_for_gpu();
        self.last_submission = Some(Submission {
            sync_point,
            temp_buffers: vec![stage_buffer],
        });
    }

    pub fn draw(&mut self) {
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
            let mut pen = pass.with(&self.map_draw_pipeline);
            pen.bind(
                0,
                &DrawData {
                    g_camera: CameraParams {
                        pos: [0.0; 4],
                        rot: [0.0, 0.0, 0.0, 1.0],
                        fov: [0.3, 0.3],
                        pad: [0.0; 2],
                    },
                    g_params: self.draw_params,
                    g_map: self.map_texture.view,
                    g_sampler: self.map_sampler,
                },
            );
            pen.draw(0, 3, 0, 1);
        }

        self.command_encoder.present(frame);
        let sync_point = self.gpu_context.submit(&mut self.command_encoder);
        self.wait_for_gpu();
        self.last_submission = Some(Submission {
            sync_point,
            temp_buffers: Vec::new(),
        });
    }
}
