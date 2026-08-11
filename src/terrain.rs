use crate::config;
use crate::texture::Texture;
use blade_graphics as gpu;

/// GPU half of one TIN chunk: a vertex buffer plus the u32 index data of
/// every LOD, and the metadata the renderer needs for frustum culling and
/// LOD selection. Indices live in their own buffer because WebGL2 assigns
/// a buffer to the element-array or data class on first bind.
pub struct TerrainChunk {
    pub vertex_buffer: gpu::Buffer,
    pub index_buffer: gpu::Buffer,
    /// `(first index, index count)` per LOD, finest first.
    pub lods: Vec<(u32, u32)>,
    pub center: [f32; 3],
    /// World AABB, for frustum culling.
    pub min: [f32; 3],
    pub max: [f32; 3],
}

pub struct Terrain {
    pub texture: Texture,
    pub env_texture: Option<Texture>,
    pub config: config::Map,
    /// The triangulated terrain, chunked for culling/LOD. Built by
    /// `tin::build` and uploaded by `Loader::load_terrain_mesh`.
    pub chunks: Vec<TerrainChunk>,
}

impl Terrain {
    pub fn free(&self, context: &gpu::Context) {
        self.texture.deinit(context);
        if let Some(env) = self.env_texture.as_ref() {
            env.deinit(context);
        }
        for chunk in &self.chunks {
            context.destroy_buffer(chunk.vertex_buffer);
            context.destroy_buffer(chunk.index_buffer);
        }
    }
}
