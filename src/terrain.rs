use crate::config;
use crate::texture::Texture;
use crate::voxels::Voxels;

pub struct Terrain {
    pub texture: Texture,
    pub env_texture: Option<Texture>,
    pub config: config::Map,
    /// Voxel acceleration structure for the HiZ raycaster. Populated by
    /// `VoxelBaker::bake` from the heightmap during load.
    pub voxels: Voxels,
}
