use blade_graphics as gpu;

use crate::model::VertexDesc;
use crate::texture::Texture;
use crate::{Geometry, Material, MaterialDesc, Model, ModelDesc};
use base64::engine::{Engine as _, general_purpose::URL_SAFE as ENCODING_ENGINE};
use blade_graphics::Extent;
use std::{fs, mem, path::Path, ptr, slice};

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

    pub fn context(&self) -> &gpu::Context {
        self.context
    }

    fn populate_gltf(
        geometries: &mut Vec<super::GeometryDesc>,
        g_node: gltf::Node,
        parent_transform: nalgebra::Matrix4<f32>,
        data_buffers: &[Vec<u8>],
    ) {
        let local_transform = nalgebra::Matrix4::from(g_node.transform().matrix());
        let transform = parent_transform * local_transform;

        if let Some(g_mesh) = g_node.mesh() {
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

                let index_reader = reader
                    .read_indices()
                    .map(gltf::mesh::util::ReadIndices::into_u32);
                let index_count = index_reader
                    .as_ref()
                    .map_or(0, std::iter::ExactSizeIterator::len);

                profiling::scope!("Read data");
                let indices: Vec<u32> = if let Some(reader) = index_reader {
                    reader.collect()
                } else {
                    (0..vertex_count as u32).collect()
                };

                let mut vertices = Vec::with_capacity(vertex_count);
                for pos in reader.read_positions().unwrap() {
                    for component in pos {
                        assert!(component.is_finite());
                    }
                    vertices.push(VertexDesc {
                        pos: pos.into(),
                        ..VertexDesc::default()
                    });
                }

                if let Some(iter) = reader.read_tex_coords(0) {
                    for (v, uv) in vertices.iter_mut().zip(iter.into_f32()) {
                        v.tex_coords = nalgebra::Point2::from(uv);
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
                        v.normal = nalgebra::Vector3::from(normal);
                    }
                } else {
                    log::warn!("No normals in {name}");
                }

                geometries.push(super::GeometryDesc {
                    name: name.to_string(),
                    indices: indices
                        .chunks(3)
                        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                        .collect(),
                    vertices,
                    index_type: if index_count > 0 {
                        Some(gpu::IndexType::U32)
                    } else {
                        None
                    },
                    transform,
                    material_index: match g_primitive.material().index() {
                        Some(index) => index + 1,
                        None => 0,
                    },
                });
            }
        }

        for child in g_node.children() {
            Self::populate_gltf(geometries, child, transform, data_buffers);
        }
    }

    pub fn read_gltf(path: &Path, base_transform: nalgebra::Matrix4<f32>) -> super::ModelDesc {
        Self::read_gltf_data(&fs::read(path).unwrap(), path, base_transform)
    }

    /// Parse a GLB/glTF from bytes. `path` is only used to resolve `file:`
    /// buffer URIs (unused by self-contained GLBs, which is what the web
    /// build embeds).
    pub fn read_gltf_data(
        data: &[u8],
        path: &Path,
        base_transform: nalgebra::Matrix4<f32>,
    ) -> super::ModelDesc {
        let gltf::Gltf { document, mut blob } = gltf::Gltf::from_slice(data).unwrap();

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
        let mut materials = vec![MaterialDesc::default()]; // default goes first
        for g_material in document.materials() {
            let pbr = g_material.pbr_metallic_roughness();
            materials.push(MaterialDesc {
                name: g_material.name().map(str::to_owned),
                base_color_factor: pbr.base_color_factor(),
                normal_scale: g_material.normal_texture().map_or(0.0, |info| info.scale()),
                transparent: g_material.alpha_mode() != gltf::material::AlphaMode::Opaque,
            });
        }

        // load nodes
        let mut geometries = Vec::new();
        for g_scene in document.scenes() {
            for g_node in g_scene.nodes() {
                Self::populate_gltf(&mut geometries, g_node, base_transform, &data_buffers);
            }
        }

        super::ModelDesc {
            materials,
            geometries,
        }
    }

    pub fn load_model(&mut self, model: &ModelDesc) -> Model {
        let geometries = model
            .geometries
            .iter()
            .map(|geometry| {
                let mut transfer = self.encoder.transfer("load mesh");
                let vertex_count = geometry.vertices.len();
                let vertex_size = vertex_count * mem::size_of::<super::Vertex>();
                let index_count = if geometry.index_type.is_some() {
                    geometry.indices.len() * 3
                } else {
                    0
                };
                let index_size = index_count * mem::size_of::<u32>();

                // Vertices and indices go into separate device buffers: WebGL2
                // assigns a buffer to the element-array or data class on its
                // first bind, so the two kinds can never share one. The sync
                // right after creation is what performs that classification
                // (and allocates the GL storage) on the web — it is a no-op
                // on the other backends.
                let vertex_buffer = self.context.create_buffer(gpu::BufferDesc {
                    name: &geometry.name,
                    size: vertex_size as u64,
                    memory: gpu::Memory::Device,
                });
                self.context
                    .sync_buffer(vertex_buffer, gpu::BufferTarget::Data);
                let index_buffer = geometry.index_type.map(|ty| {
                    let buffer = self.context.create_buffer(gpu::BufferDesc {
                        name: &format!("{}/index", geometry.name),
                        size: index_size as u64,
                        memory: gpu::Memory::Device,
                    });
                    self.context.sync_buffer(buffer, gpu::BufferTarget::Index);
                    (buffer, ty)
                });

                // One staging blob for both: copies bind COPY_READ/COPY_WRITE,
                // which are exempt from WebGL2's class assignment.
                let stage_buffer = self.context.create_buffer(gpu::BufferDesc {
                    name: &geometry.name,
                    size: (vertex_size + index_size) as u64,
                    memory: gpu::Memory::Upload,
                });
                if index_count > 0 {
                    let indices = unsafe {
                        slice::from_raw_parts_mut(
                            stage_buffer.data().add(vertex_size) as *mut u32,
                            index_count,
                        )
                    };
                    for (id, is) in indices
                        .iter_mut()
                        .zip(geometry.indices.iter().flat_map(|&i| i))
                    {
                        *id = is;
                    }
                }

                let vertices = unsafe {
                    slice::from_raw_parts_mut(
                        stage_buffer.data() as *mut super::Vertex,
                        geometry.vertices.len(),
                    )
                };
                for (vertex, desc) in vertices.iter_mut().zip(geometry.vertices.iter()) {
                    vertex.position = desc.pos.into();
                    vertex.normal = encode_normal(desc.normal.into());
                    assert_ne!(vertex.normal, 0);
                    vertex.tex_coords = desc.tex_coords.into();
                }
                self.context
                    .sync_buffer(stage_buffer, gpu::BufferTarget::Data);
                transfer.copy_buffer_to_buffer(
                    stage_buffer.into(),
                    vertex_buffer.into(),
                    vertex_size as u64,
                );
                if let Some((index_buffer, _)) = index_buffer {
                    transfer.copy_buffer_to_buffer(
                        stage_buffer.at(vertex_size as u64),
                        index_buffer.into(),
                        index_size as u64,
                    );
                }
                self.temp_buffers.push(stage_buffer);
                Geometry {
                    name: geometry.name.clone(),
                    vertex_range: 0..vertex_count as u32,
                    triangle_count: (if index_count > 0 {
                        index_count
                    } else {
                        vertex_count
                    }) as u32
                        / 3,
                    transform: geometry.transform,
                    material_index: geometry.material_index,
                    vertex_buffer,
                    index_buffer,
                }
            })
            .collect();

        let materials = model
            .materials
            .iter()
            .map(|material| Material {
                base_color_texture: None,
                base_color_factor: material.base_color_factor,
                normal_texture: None,
                normal_scale: material.normal_scale,
                transparent: material.transparent,
            })
            .collect();
        Model {
            materials,
            geometries,
        }
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
        self.context.sync_buffer(stage_buffer, gpu::BufferTarget::Data);

        let mut texture = Texture::default();
        texture.init_2d(
            self.context,
            "terrain",
            gpu::TextureFormat::Rgba8UnormSrgb,
            extent,
            gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        );

        self.encoder.init_texture(texture.raw());
        if let mut pass = self.encoder.transfer("terraian init") {
            pass.copy_buffer_to_texture(
                stage_buffer.into(),
                extent.width * 4,
                texture.raw().into(),
                extent,
            );
        }

        self.temp_buffers.push(stage_buffer);
        texture
    }

    pub fn load_environment(&mut self, path: &Path) -> Texture {
        self.load_environment_data(&fs::read(path).unwrap())
    }

    pub fn load_environment_data(&mut self, data: &[u8]) -> Texture {
        let decoder = png::Decoder::new(std::io::Cursor::new(data));
        let mut reader = decoder.read_info().unwrap();
        let mut decoded = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(decoded.as_mut_slice()).unwrap();

        let extent = Extent {
            width: info.width,
            height: info.height,
            depth: 1,
        };
        let pixel_count = (extent.width as usize) * (extent.height as usize);
        // GPU formats require 4 channels — expand RGB → RGBA with opaque alpha.
        let rgba: Vec<u8> = match info.color_type {
            png::ColorType::Rgb => (0..pixel_count)
                .flat_map(|i| [decoded[i * 3], decoded[i * 3 + 1], decoded[i * 3 + 2], 0xFF])
                .collect(),
            png::ColorType::Rgba => decoded,
            other => panic!("Unsupported environment PNG color type: {:?}", other),
        };

        let stage_buffer = self.context.create_buffer(gpu::BufferDesc {
            name: "stage environment",
            size: rgba.len() as u64,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            ptr::copy_nonoverlapping(rgba.as_ptr(), stage_buffer.data(), rgba.len());
        }
        self.context.sync_buffer(stage_buffer, gpu::BufferTarget::Data);

        let mut texture = Texture::default();
        texture.init_2d(
            self.context,
            "environment",
            gpu::TextureFormat::Rgba8UnormSrgb,
            extent,
            gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        );

        self.encoder.init_texture(texture.raw());
        if let mut pass = self.encoder.transfer("environment init") {
            pass.copy_buffer_to_texture(
                stage_buffer.into(),
                extent.width * 4,
                texture.raw().into(),
                extent,
            );
        }

        self.temp_buffers.push(stage_buffer);
        texture
    }

    /// Upload the TIN chunks into per-chunk vertex and index buffers, ready
    /// for the terrain-mesh pipeline.
    pub fn load_terrain_mesh(&mut self, mesh: &crate::tin::TerrainMesh) -> Vec<super::TerrainChunk> {
        profiling::scope!("Loader::load_terrain_mesh");
        let mut total_bytes = 0u64;
        let chunks = mesh
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let vertex_bytes: &[u8] = bytemuck::cast_slice(&chunk.vertices);
                let index_bytes: &[u8] = bytemuck::cast_slice(&chunk.indices);
                let total_size = (vertex_bytes.len() + index_bytes.len()) as u64;
                total_bytes += total_size;
                let name = format!("terrain chunk {i}");
                // Separate buffers per class; see load_model for the WebGL2
                // reasoning behind the split and the post-creation syncs.
                let vertex_buffer = self.context.create_buffer(gpu::BufferDesc {
                    name: &name,
                    size: vertex_bytes.len() as u64,
                    memory: gpu::Memory::Device,
                });
                self.context
                    .sync_buffer(vertex_buffer, gpu::BufferTarget::Data);
                let index_buffer = self.context.create_buffer(gpu::BufferDesc {
                    name: &format!("{name}/index"),
                    size: index_bytes.len() as u64,
                    memory: gpu::Memory::Device,
                });
                self.context
                    .sync_buffer(index_buffer, gpu::BufferTarget::Index);
                let stage_buffer = self.context.create_buffer(gpu::BufferDesc {
                    name: &name,
                    size: total_size,
                    memory: gpu::Memory::Upload,
                });
                unsafe {
                    ptr::copy_nonoverlapping(
                        vertex_bytes.as_ptr(),
                        stage_buffer.data(),
                        vertex_bytes.len(),
                    );
                    ptr::copy_nonoverlapping(
                        index_bytes.as_ptr(),
                        stage_buffer.data().add(vertex_bytes.len()),
                        index_bytes.len(),
                    );
                }
                self.context
                    .sync_buffer(stage_buffer, gpu::BufferTarget::Data);
                let mut transfer = self.encoder.transfer("load terrain chunk");
                transfer.copy_buffer_to_buffer(
                    stage_buffer.into(),
                    vertex_buffer.into(),
                    vertex_bytes.len() as u64,
                );
                transfer.copy_buffer_to_buffer(
                    stage_buffer.at(vertex_bytes.len() as u64),
                    index_buffer.into(),
                    index_bytes.len() as u64,
                );
                self.temp_buffers.push(stage_buffer);
                super::TerrainChunk {
                    vertex_buffer,
                    index_buffer,
                    lods: chunk.lods.clone(),
                    center: chunk.center(),
                    min: chunk.min,
                    max: chunk.max,
                }
            })
            .collect();
        log::info!(
            "Terrain mesh uploaded: {} chunks, {} MiB",
            mesh.chunks.len(),
            total_bytes >> 20,
        );
        chunks
    }

    pub fn load_png(&mut self, path: &Path) -> (Texture, Extent, Vec<u8>) {
        self.load_png_data(&fs::read(path).unwrap(), 1)
    }

    /// Decode an RGBA map PNG from bytes, optionally box-downsampling it by
    /// an integer `downsample` factor first. The web build uses a factor of
    /// 4: Fostral's 3 cm texels are far denser than the gameplay needs, and
    /// shrinking them keeps the single-threaded TIN build, the GPU buffers,
    /// and the shadow map inside browser budgets.
    pub fn load_png_data(&mut self, data: &[u8], downsample: u32) -> (Texture, Extent, Vec<u8>) {
        let decoder = png::Decoder::new(std::io::Cursor::new(data));
        let mut reader = decoder.read_info().unwrap();
        let mut vec = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(vec.as_mut_slice()).unwrap();

        let mut extent = Extent {
            width: info.width,
            height: info.height,
            depth: 1,
        };
        if downsample > 1 {
            let (small, w, h) = downsample_rgba(&vec, extent.width, extent.height, downsample);
            log::info!(
                "Downsampled map {}x{} -> {}x{} ({}x)",
                extent.width,
                extent.height,
                w,
                h,
                downsample
            );
            vec = small;
            extent.width = w;
            extent.height = h;
        }
        // Pull the alpha channel out for CPU-side use (heightmap collision).
        // Map is laid out as RGBA8 — see shaders/terrain-mesh.wgsl: ground_radius is mixed by texel.a.
        let pixel_count = (extent.width as usize) * (extent.height as usize);
        let alpha: Vec<u8> = (0..pixel_count).map(|i| vec[i * 4 + 3]).collect();
        let texture = self.load_terrain(extent, vec.as_slice());
        (texture, extent, alpha)
    }
}

/// Box-filter an RGBA8 image by an integer factor (all four channels — the
/// alpha carries the height).
fn downsample_rgba(src: &[u8], width: u32, height: u32, factor: u32) -> (Vec<u8>, u32, u32) {
    let (w, h) = ((width / factor).max(1), (height / factor).max(1));
    let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
    let inv = 1.0 / (factor * factor) as f32;
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            for sy in 0..factor {
                for sx in 0..factor {
                    let si = (((y * factor + sy) * width + x * factor + sx) as usize) * 4;
                    for c in 0..4 {
                        acc[c] += src[si + c] as f32;
                    }
                }
            }
            let di = ((y * w + x) as usize) * 4;
            for c in 0..4 {
                out[di + c] = (acc[c] * inv + 0.5) as u8;
            }
        }
    }
    (out, w, h)
}
