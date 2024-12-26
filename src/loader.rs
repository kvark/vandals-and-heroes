use blade_graphics as gpu;

use crate::texture::Texture;
use base64::engine::{general_purpose::URL_SAFE as ENCODING_ENGINE, Engine as _};
use std::{f32, fs, mem, path::Path, slice};

pub struct Loader<'a> {
    context: &'a gpu::Context,
    encoder: &'a mut gpu::CommandEncoder,
    temp_buffers: Vec<gpu::Buffer>,
    flat_vertices: Vec<nalgebra::Point3<f32>>,
    flat_triangles: Vec<[u32; 3]>,
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
            flat_vertices: Vec::new(),
            flat_triangles: Vec::new(),
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
        geometries: &mut Vec<super::Geometry>,
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
                let base_vertex = self.flat_vertices.len() as u32;
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
                    for tri in indices.chunks(3) {
                        self.flat_triangles.push([
                            base_vertex + tri[0],
                            base_vertex + tri[1],
                            base_vertex + tri[2],
                        ]);
                    }
                } else {
                    for tri_index in 0..vertex_count as u32 / 3 {
                        let base = base_vertex + tri_index * 3;
                        self.flat_triangles.push([base + 0, base + 1, base + 2]);
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
                    let transformed = transform * nalgebra::Point3::from(pos).to_homogeneous();
                    self.flat_vertices.push(transformed.xyz().into());
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

                geometries.push(super::Geometry {
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
            self.populate_gltf(geometries, child, transform, data_buffers);
        }
    }

    pub fn load_gltf(&mut self, base_path: &Path, config: &super::config::Model) -> super::Model {
        let path = base_path.join(&config.model);
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
                        fs::read(base_path.join(rest)).unwrap()
                    } else if let Some(rest) = uri.strip_prefix("file:") {
                        fs::read(base_path.join(rest)).unwrap()
                    } else {
                        fs::read(base_path.join(uri)).unwrap()
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
        let mut materials = vec![super::Material::default()]; // default goes first
        for g_material in document.materials() {
            let pbr = g_material.pbr_metallic_roughness();
            materials.push(super::Material {
                base_color_texture: pbr.base_color_texture().map(|_info| todo!()),
                base_color_factor: pbr.base_color_factor(),
                normal_texture: g_material.normal_texture().map(|_info| todo!()),
                normal_scale: g_material.normal_texture().map_or(0.0, |info| info.scale()),
                transparent: g_material.alpha_mode() != gltf::material::AlphaMode::Opaque,
            });
        }

        // load nodes
        let mut geometries = Vec::new();
        for g_scene in document.scenes() {
            let base_transform = nalgebra::Similarity3::from_scaling(config.scale);
            for g_node in g_scene.nodes() {
                self.populate_gltf(
                    &mut geometries,
                    g_node,
                    base_transform.to_homogeneous(),
                    &data_buffers,
                );
            }
        }

        // create the collider
        let builder = match config.shape {
            super::config::Shape::Mesh => rapier3d::geometry::ColliderBuilder::trimesh(
                mem::take(&mut self.flat_vertices),
                mem::take(&mut self.flat_triangles),
            ),
            super::config::Shape::Cylinder { depth, radius } => {
                rapier3d::geometry::ColliderBuilder::cylinder(0.5 * depth, radius)
            }
        };
        let collider = builder
            .density(config.density)
            .friction(config.friction)
            .build();

        super::Model {
            materials,
            geometries,
            collider,
        }
    }

    pub fn load_terrain_texture(&mut self, extent: gpu::Extent, buf: &[u8]) -> Texture {
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
        if let mut pass = self.encoder.transfer("terrain init") {
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

    fn load_terrain_collider(
        &self,
        extent: gpu::Extent,
        buf: &[u8],
        config: &super::config::Map,
    ) -> rapier3d::geometry::Collider {
        let mut vertices = Vec::new();
        let mut points = Vec::<[f64; 3]>::new();
        for y in 0..=extent.height {
            for x in 0..=extent.width {
                // handle wraparound for both axis
                let height =
                    buf[((y % extent.height) * extent.width + x % extent.width) as usize * 4 + 3]; //TODO: height scale
                points.push([x as f64, y as f64, height as f64]);
                let r = config.radius.start
                    + (height as f32) / 255.0 * (config.radius.end - config.radius.start);
                let alpha = x as f32 / extent.width as f32 * 2.0 * f32::consts::PI;
                let d = (y as f32 / extent.height as f32 - 0.5) * config.length;
                vertices.push(nalgebra::Point3::new(r * alpha.cos(), r * alpha.sin(), d));
            }
        }

        //TODO: https://github.com/hugoledoux/startin/issues/24
        /*let use_triangulation = false;
        let triangles = if use_triangulation {
            println!("Triangulating...");
            let mut dt = startin::Triangulation::new();
            dt.insert(&points, startin::InsertionStrategy::AsIs);
            println!("Done");
            dt.all_finite_triangles()
                .into_iter()
                .map(|t| [t.v[0] as u32, t.v[1] as u32, t.v[2] as u32])
                .collect::<Vec<_>>()
        } else*/
        let mut triangles = Vec::with_capacity(vertices.len() * 2);
        for y in 0..extent.height {
            for x in 0..extent.width {
                let a = y * (extent.width + 1) + x;
                let b = (y + 1) * (extent.width + 1) + x;
                triangles.push([a, a + 1, b]);
                triangles.push([a + 1, b + 1, b]);
            }
        }
        println!("Creating collider...");
        let builder = rapier3d::geometry::ColliderBuilder::trimesh(vertices, triangles);
        builder.density(config.density).build()
    }

    pub fn load_terrain(
        &mut self,
        path: &Path,
        map_config: &mut super::config::Map,
    ) -> (Texture, rapier3d::geometry::Collider, gpu::Extent) {
        let decoder = png::Decoder::new(fs::File::open(path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut vec = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(vec.as_mut_slice()).unwrap();

        let extent = gpu::Extent {
            width: info.width,
            height: info.height,
            depth: 1,
        };
        if map_config.length == 0.0 {
            let circumference = 2.0 * f32::consts::PI * map_config.radius.start;
            map_config.length = circumference * (extent.height as f32) / (extent.width as f32);
            log::info!("Derived map length to be {}", map_config.length);
        }

        let texture = self.load_terrain_texture(extent, vec.as_slice());
        let collider = self.load_terrain_collider(extent, vec.as_slice(), map_config);
        println!("Terrain is loaded");
        (texture, collider, extent)
    }
}
