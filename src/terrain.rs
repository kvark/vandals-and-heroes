use crate::config;
use crate::texture::Texture;

pub struct Terrain {
    pub texture: Texture,
    pub config: config::Map,
}
