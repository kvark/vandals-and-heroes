use blade_graphics as gpu;
use std::path::Path;

#[derive(Default)]
pub struct Model {
    vertex_count: usize,
    triangle_count: usize,
    vertex_buf: gpu::Buffer,
    index_buf: gpu::Buffer,
    pos: nalgebra::Vector3<f32>,
    rot: nalgebra::UnitQuaternion<f32>,
}

impl Model {
    pub fn new(path: &Path) -> Self {
        let _gltf = gltf::Gltf::open(path);
        Model::default()
    }
}
