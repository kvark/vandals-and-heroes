use blade_graphics as gpu;

/// An owned texture + view pair.
///
/// The handles live in an `Option` rather than being compared against a
/// `Default` sentinel: blade's GLES/WebGL2 backend handle types don't
/// implement `Default`, and an explicit empty state is clearer anyway.
#[derive(Default)]
pub struct Texture {
    inner: Option<(gpu::Texture, gpu::TextureView)>,
}

impl Texture {
    pub fn raw(&self) -> gpu::Texture {
        self.inner.expect("texture is not initialized").0
    }

    pub fn view(&self) -> gpu::TextureView {
        self.inner.expect("texture is not initialized").1
    }

    pub fn init_2d(
        &mut self,
        context: &gpu::Context,
        name: &str,
        format: gpu::TextureFormat,
        size: gpu::Extent,
        usage: gpu::TextureUsage,
    ) {
        self.deinit(context);
        let raw = context.create_texture(gpu::TextureDesc {
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
        let view = context.create_texture_view(
            raw,
            gpu::TextureViewDesc {
                name,
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        self.inner = Some((raw, view));
    }

    pub fn deinit(&self, context: &gpu::Context) {
        if let Some((raw, view)) = self.inner {
            context.destroy_texture_view(view);
            context.destroy_texture(raw);
        }
    }
}
