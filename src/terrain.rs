use crate::config;
use crate::texture::Texture;
use blade_graphics as gpu;

/// GPU half of one TIN chunk: a single buffer holding the vertex data
/// followed by the index data of every LOD, plus the metadata the renderer
/// needs for frustum culling and LOD selection.
pub struct TerrainChunk {
    pub buffer: gpu::Buffer,
    /// Byte offset of the (u32) index data inside `buffer`. `None` on the
    /// web, where WebGL2's element-buffer type-locking forces non-indexed
    /// draws (the buffer then holds pre-expanded triangle vertices and the
    /// LOD ranges count vertices instead of indices).
    pub index_offset: Option<u64>,
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
            context.destroy_buffer(chunk.buffer);
        }
    }
}
