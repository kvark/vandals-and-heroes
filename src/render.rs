use blade_graphics as gpu;

pub struct Render {
    depth_texture: gpu::Texture,
    depth_view: gpu::TextureView,
    map_draw_pipeline: gpu::RenderPipeline,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CameraParams {
    pos: [f32; 4],
    rot: [f32; 4],
    fov: [f32; 2],
    pad: [f32; 2],
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct DrawParams {
    screen_size: [f32; 2],
}

#[derive(blade_macros::ShaderData)]
struct DrawData {
    g_camera: CameraParams,
    //g_params: DrawParams,
}

impl Render {
    pub fn new(
        context: &gpu::Context,
        extent: gpu::Extent,
        surface_info: gpu::SurfaceInfo,
    ) -> Self {
        let depth_format = gpu::TextureFormat::Depth32Float;
        let depth_texture = context.create_texture(gpu::TextureDesc {
            name: "depth",
            format: depth_format,
            size: extent,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
        });
        let depth_view = context.create_texture_view(
            depth_texture,
            gpu::TextureViewDesc {
                name: "depth",
                format: depth_format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        let source = std::fs::read_to_string("shaders/map-draw.wgsl").unwrap();
        let shader = context.create_shader(gpu::ShaderDesc { source: &source });
        let global_layout = <DrawData as gpu::ShaderData>::layout();
        Self {
            depth_texture,
            depth_view,
            map_draw_pipeline: context.create_render_pipeline(gpu::RenderPipelineDesc {
                name: "map-draw",
                data_layouts: &[&global_layout],
                vertex: shader.at("vs_draw"),
                vertex_fetches: &[],
                primitive: gpu::PrimitiveState::default(),
                depth_stencil: Some(gpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: true,
                    depth_compare: gpu::CompareFunction::Always,
                    stencil: gpu::StencilState::default(),
                    bias: gpu::DepthBiasState::default(),
                }),
                fragment: shader.at("fs_draw"),
                color_targets: &[surface_info.format.into()],
            }),
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_texture_view(self.depth_view);
        context.destroy_texture(self.depth_texture);
        context.destroy_render_pipeline(&mut self.map_draw_pipeline);
    }

    pub fn draw(&self, encoder: &mut gpu::CommandEncoder, main_view: gpu::TextureView) {
        if let mut pass = encoder.render(
            "draw",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: main_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: Some(gpu::RenderTarget {
                    view: self.depth_view,
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
                },
            );
            pen.draw(0, 3, 0, 1);
        }
    }
}
