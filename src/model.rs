use blade_graphics as gpu;
use std::ops::Range;

#[derive(Default)]
pub struct Geometry {
    pub name: String,
    pub vertex_range: Range<u32>,
    pub index_offset: u64,
    pub index_type: Option<gpu::IndexType>,
    pub triangle_count: u32,
    pub transform: nalgebra::Matrix4<f32>,
    pub material_index: usize,
    pub buffer: gpu::Buffer,
}

impl Geometry {
    pub(super) fn rendering_transform(&self, base: &nalgebra::Matrix4<f32>) -> [[f32; 4]; 3] {
        *(base * self.transform).remove_row(3).transpose().as_ref()
    }
}

#[derive(Default)]
pub struct Material {
    pub base_color_texture: Option<super::Texture>,
    pub base_color_factor: [f32; 4],
    pub normal_texture: Option<super::Texture>,
    pub normal_scale: f32,
    pub transparent: bool,
}

pub struct Model {
    pub materials: Vec<Material>,
    pub geometries: Vec<Geometry>,
    pub collider: rapier3d::geometry::Collider,
}

impl Model {
    pub fn free(&self, context: &gpu::Context) {
        for geometry in self.geometries.iter() {
            context.destroy_buffer(geometry.buffer);
        }
        for material in self.materials.iter() {
            if let Some(ref texture) = material.base_color_texture {
                texture.deinit(context);
            }
            if let Some(ref texture) = material.normal_texture {
                texture.deinit(context);
            }
        }
    }
}
