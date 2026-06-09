use blade_graphics as gpu;
use nalgebra::{Point2, Point3, Vector3};
use std::ops::Range;
use std::sync::Arc;

#[derive(Default)]
pub struct VertexDesc {
    pub pos: Point3<f32>,
    pub tex_coords: Point2<f32>,
    pub normal: Vector3<f32>,
}

pub struct GeometryDesc {
    pub name: String,
    pub vertices: Vec<VertexDesc>,
    pub indices: Vec<[u32; 3]>,
    pub index_type: Option<gpu::IndexType>,
    pub transform: nalgebra::Matrix4<f32>,
    pub material_index: usize,
}

#[derive(Default)]
pub struct MaterialDesc {
    pub name: Option<String>,
    // TODO: base_color_texture
    pub base_color_factor: [f32; 4],
    // TODO: normal_texture
    pub normal_scale: f32,
    pub transparent: bool,
}

pub struct ModelDesc {
    pub materials: Vec<MaterialDesc>,
    pub geometries: Vec<GeometryDesc>,
}

impl ModelDesc {
    pub fn positions(&self) -> Vec<Point3<f32>> {
        self.positions_filtered(|_| true)
    }

    pub fn indices(&self) -> Vec<[u32; 3]> {
        self.indices_filtered(|_| true)
    }

    /// Returns vertex positions only for geometries whose material passes `keep`.
    pub fn positions_filtered<F>(&self, keep: F) -> Vec<Point3<f32>>
    where
        F: Fn(&MaterialDesc) -> bool,
    {
        self.geometries
            .iter()
            .filter(|g| keep(&self.materials[g.material_index]))
            .flat_map(|g| {
                g.vertices
                    .iter()
                    .map(|v| g.transform * v.pos.to_homogeneous())
                    .map(|pos| pos.xyz().into())
            })
            .collect()
    }

    /// Returns triangle indices, renumbered to match `positions_filtered(keep)`.
    pub fn indices_filtered<F>(&self, keep: F) -> Vec<[u32; 3]>
    where
        F: Fn(&MaterialDesc) -> bool,
    {
        let mut last_index = 0;
        let mut indices = Vec::new();
        for geometry in &self.geometries {
            if !keep(&self.materials[geometry.material_index]) {
                continue;
            }
            let vertices_count = geometry.vertices.len();
            for tri in &geometry.indices {
                indices.push([
                    tri[0] + last_index,
                    tri[1] + last_index,
                    tri[2] + last_index,
                ]);
            }
            last_index += vertices_count as u32;
        }
        indices
    }
}

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

pub struct ModelInstance {
    pub model: Arc<Model>,
    pub transform: nalgebra::Isometry3<f32>,
    /// Indices of geometries to render. `None` = render all. Used so the
    /// chassis instance can skip wheel geometries and per-wheel instances
    /// can render just the wheel mesh at the corresponding rigid-body pose.
    pub geometry_filter: Option<Vec<usize>>,
}
