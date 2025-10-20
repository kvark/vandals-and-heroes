use blade_graphics as gpu;

#[derive(Default)]
pub struct Texture {
    pub raw: gpu::Texture,
    pub view: gpu::TextureView,
}

impl Texture {
    pub fn init_2d(
        &mut self,
        context: &gpu::Context,
        name: &str,
        format: gpu::TextureFormat,
        size: gpu::Extent,
        usage: gpu::TextureUsage,
    ) {
        self.deinit(context);
        self.raw = context.create_texture(gpu::TextureDesc {
            name,
            format,
            size,
            sample_count: 1,
            array_layer_count: 1,
            mip_level_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage,
            external: None,
        });
        self.view = context.create_texture_view(
            self.raw,
            gpu::TextureViewDesc {
                name,
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
    }

    pub fn deinit(&self, context: &gpu::Context) {
        if self.raw != Default::default() {
            context.destroy_texture_view(self.view);
            context.destroy_texture(self.raw);
        }
    }
}
