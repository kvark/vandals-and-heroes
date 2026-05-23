use crate::config;
use crate::texture::Texture;

pub struct Terrain {
    pub texture: Texture,
    pub env_texture: Option<Texture>,
    pub config: config::Map,
}
