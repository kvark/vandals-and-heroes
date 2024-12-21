use blade_graphics as gpu;

use base64::engine::{general_purpose::URL_SAFE as ENCODING_ENGINE, Engine as _};
use std::{fs, mem, path::Path, slice};
use blade_graphics::Extent;
use crate::texture::Texture;

pub struct Loader<'a> {
    context: &'a gpu::Context,
    encoder: &'a mut gpu::CommandEncoder,
    temp_buffers: Vec<gpu::Buffer>,
}

fn pack4x8snorm(v: [f32; 4]) -> u32 {
    v.iter().rev().fold(0u32, |u, f| {
        (u << 8) | (f.clamp(-1.0, 1.0) * 127.0 + 0.5) as i8 as u8 as u32
    })
}

fn encode_normal(v: [f32; 3]) -> u32 {
    pack4x8snorm([v[0], v[1], v[2], 0.0])
}

impl<'a> Loader<'a> {
    pub fn new(context: &'a gpu::Context, encoder: &'a mut gpu::CommandEncoder) -> Self {
        encoder.start();
        Self {
            context,
            encoder,
            temp_buffers: Vec::new(),
        }
    }

    pub fn finish(self) -> super::Submission {
        super::Submission {
            sync_point: self.context.submit(self.encoder),
            temp_buffers: self.temp_buffers,
        }
    }

    fn populate_gltf(
        &mut self,
        model: &mut super::Model,
        g_node: gltf::Node,
        parent_transform: nalgebra::Matrix4<f32>,
        data_buffers: &[Vec<u8>],
    ) {
        let local_transform = nalgebra::Matrix4::from(g_node.transform().matrix());
        let transform = parent_transform * local_transform;

        if let Some(g_mesh) = g_node.mesh() {
            let mut transfer = self.encoder.transfer("load mesh");
            let name = g_node.name().unwrap_or("");

            for (prim_index, g_primitive) in g_mesh.primitives().enumerate() {
                if g_primitive.mode() != gltf::mesh::Mode::Triangles {
                    log::warn!(
                        "Skipping primitive '{}'[{}] for having mesh mode {:?}",
                        name,
                        prim_index,
                        g_primitive.mode()
                    );
                    continue;
                }

                let reader = g_primitive.reader(|buffer| Some(&data_buffers[buffer.index()]));
                let vertex_count = g_primitive.get(&gltf::Semantic::Positions).unwrap().count();

                let index_reader = reader.read_indices().map(|read| read.into_u32());
                let index_count = index_reader.as_ref().map_or(0, |iter| iter.len());
                let index_offset = vertex_count * mem::size_of::<super::Vertex>();

                let total_size = index_offset + index_count * mem::size_of::<u32>();
                let buffer = self.context.create_buffer(gpu::BufferDesc {
                    name: &name,
                    size: total_size as u64,
                    memory: gpu::Memory::Device,
                });
                let stage_buffer = self.context.create_buffer(gpu::BufferDesc {
                    name: &name,
                    size: total_size as u64,
                    memory: gpu::Memory::Upload,
                });

                // Read the indices into memory
                profiling::scope!("Read data");
                if let Some(reader) = index_reader {
                    let indices = unsafe {
                        slice::from_raw_parts_mut(
                            stage_buffer.data().add(index_offset) as *mut u32,
                            index_count,
                        )
                    };
                    for (id, is) in indices.iter_mut().zip(reader) {
                        *id = is;
                    }
                }

                // Read the vertices into memory
                let vertices = unsafe {
                    slice::from_raw_parts_mut(
                        stage_buffer.data() as *mut super::Vertex,
                        vertex_count,
                    )
                };
                for (v, pos) in vertices.iter_mut().zip(reader.read_positions().unwrap()) {
                    for component in pos {
                        assert!(component.is_finite());
                    }
                    v.position = pos;
                }
                if let Some(iter) = reader.read_tex_coords(0) {
                    for (v, tc) in vertices.iter_mut().zip(iter.into_f32()) {
                        v.tex_coords = tc;
                    }
                } else {
                    log::warn!("No tex coords in {name}");
                }
                if let Some(iter) = reader.read_normals() {
                    assert_eq!(
                        vertices.len(),
                        iter.len(),
                        "geometry {name} doesn't have enough normals"
                    );
                    for (v, normal) in vertices.iter_mut().zip(iter) {
                        v.normal = encode_normal(normal);
                        assert_ne!(v.normal, 0);
                    }
                } else {
                    log::warn!("No normals in {name}");
                }

                transfer.copy_buffer_to_buffer(
                    stage_buffer.into(),
                    buffer.into(),
                    total_size as u64,
                );
                self.temp_buffers.push(stage_buffer);

                model.geometries.push(super::Geometry {
                    name: name.to_string(),
                    vertex_range: 0..vertex_count as u32,
                    index_offset: index_offset as u64,
                    index_type: if index_count > 0 {
                        Some(gpu::IndexType::U32)
                    } else {
                        None
                    },
                    triangle_count: (if index_count > 0 {
                        index_count
                    } else {
                        vertex_count
                    }) as u32
                        / 3,
                    transform,
                    material_index: match g_primitive.material().index() {
                        Some(index) => index + 1,
                        None => 0,
                    },
                    buffer,
                });
            }
        }

        for child in g_node.children() {
            self.populate_gltf(model, child, transform, data_buffers);
        }
    }

    pub fn load_gltf(&mut self, path: &Path) -> super::Model {
        let mut model = super::Model::default();
        let gltf::Gltf { document, mut blob } = gltf::Gltf::open(path).unwrap();

        // extract buffers
        let mut data_buffers = Vec::new();
        for buffer in document.buffers() {
            let mut data = match buffer.source() {
                gltf::buffer::Source::Uri(uri) => {
                    if let Some(rest) = uri.strip_prefix("data:") {
                        let (_before, after) = rest.split_once(";base64,").unwrap();
                        ENCODING_ENGINE.decode(after).unwrap()
                    } else if let Some(rest) = uri.strip_prefix("file://") {
                        fs::read(path.join(rest)).unwrap()
                    } else if let Some(rest) = uri.strip_prefix("file:") {
                        fs::read(path.join(rest)).unwrap()
                    } else {
                        fs::read(path.join(uri)).unwrap()
                    }
                }
                gltf::buffer::Source::Bin => blob.take().unwrap(),
            };
            assert!(data.len() >= buffer.length());
            while data.len() % 4 != 0 {
                data.push(0);
            }
            data_buffers.push(data);
        }

        // load materials
        model.materials.push(super::Material::default()); // default goes first
        for g_material in document.materials() {
            let pbr = g_material.pbr_metallic_roughness();
            model.materials.push(super::Material {
                base_color_texture: pbr.base_color_texture().map(|_info| todo!()),
                base_color_factor: pbr.base_color_factor(),
                normal_texture: g_material.normal_texture().map(|_info| todo!()),
                normal_scale: g_material.normal_texture().map_or(0.0, |info| info.scale()),
                transparent: g_material.alpha_mode() != gltf::material::AlphaMode::Opaque,
            });
        }

        // load nodes
        for g_scene in document.scenes() {
            for g_node in g_scene.nodes() {
                self.populate_gltf(
                    &mut model,
                    g_node,
                    nalgebra::Matrix4::identity(),
                    &data_buffers,
                );
            }
        }
        model
    }

    pub fn load_terrain(&mut self, extent: Extent, buf: &[u8]) -> Texture {

        let stage_buffer = self.context.create_buffer(gpu::BufferDesc {
            name: "stage png",
            size: buf.len() as u64,
            memory: gpu::Memory::Upload,
        });

        unsafe {
            let parts_mut = slice::from_raw_parts_mut(stage_buffer.data(), buf.len());
            std::ptr::copy(buf.as_ptr(), parts_mut.as_mut_ptr(), buf.len());
        }

        let mut texture = Texture::default();
        texture.init_2d(
            &self.context,
            "terrain",
            gpu::TextureFormat::Rgba8UnormSrgb,
            extent,
            gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        );

        self.encoder.init_texture(texture.raw);
        if let mut pass = self.encoder.transfer("terraian init") {
            pass.copy_buffer_to_texture(
                stage_buffer.into(),
                extent.width * 4,
                texture.raw.into(),
                extent,
            );
        }

        self.temp_buffers.push(stage_buffer);
        texture
    }

    pub fn load_png(&mut self, path: &Path) -> (Texture, Extent) {
        let decoder = png::Decoder::new(fs::File::open(path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut vec = vec![0u8; reader.output_buffer_size()];
        let info = reader
            .next_frame(vec.as_mut_slice())
            .unwrap();

        let extent = Extent {
            width: info.width,
            height: info.height,
            depth: 1,
        };
        let texture = self.load_terrain(extent, vec.as_slice());
        (texture, extent)
    }
}
