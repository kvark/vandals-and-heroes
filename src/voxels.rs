use blade_graphics as gpu;

use crate::texture::Texture;

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BakeParams {
    radius_start: f32,
    radius_end: f32,
    _pad: [f32; 2],
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MipParams {
    src_lod: u32,
    _pad: [u32; 3],
}

#[derive(blade_macros::ShaderData)]
struct BakeInitData {
    g_voxels: gpu::BufferPiece,
    g_bake: BakeParams,
    g_heightmap: gpu::TextureView,
    g_heightmap_sampler: gpu::Sampler,
}

#[derive(blade_macros::ShaderData)]
struct BakeMipData {
    g_voxels: gpu::BufferPiece,
    g_mip: MipParams,
}

/// Hard cap on the number of LODs the storage buffer can hold (matches
/// `VOXEL_MAX_LODS` in shaders/voxel.wgsl). Bigger LOD chains are an outright
/// waste — even our biggest map fits in well under 16 levels.
pub const MAX_LODS: usize = 16;

/// Bytes occupied by the metadata prefix of the storage buffer:
/// `lod_count: vec4u` (16 B) + `lods: array<VoxelLod, 16>` (16 × 16 B).
/// The voxel `occupancy` array starts at this offset.
const METADATA_BYTES: u64 = 16 + (16 * 16);

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct VoxelLodDesc {
    /// Cells along (θ, z, r) at this LOD. Stored as i32 to match WGSL `vec3i`
    /// (the bake/draw shaders use signed coords so wraparound on θ is cheap).
    pub dim: [i32; 3],
    /// Byte offset / 4 (u32-word index) of this LOD's occupancy data within
    /// the storage buffer's `occupancy` array.
    pub offset: u32,
}

/// Voxel occupancy grid in cylindrical (θ, z, r) coordinates — the
/// acceleration structure for the curved-space HiZ ray traversal.
///
/// Layout in the storage buffer mirrors the WGSL `VoxelData` struct:
///   - `lod_count` (vec4u, 16 B)
///   - `lods[16]` (16 × 16 B = 256 B)
///   - `occupancy: array<u32>` (1 bit per voxel, bit-packed)
/// Mip chain is built any-child-occupied; chain stops when all axes hit 1.
pub struct Voxels {
    pub buffer: gpu::Buffer,
    pub buffer_size: u64,
    /// LOD 0 dimensions (cells along θ, z, r).
    pub base_dim: [u32; 3],
    /// LODs in coarse-to-fine? No — index 0 is the finest LOD; the chain
    /// builds toward coarser parents at higher indices, matching vange-rs.
    pub lods: Vec<VoxelLodDesc>,
}

impl Voxels {
    /// Build the LOD chain by halving each axis (floor div, min 1) until all
    /// axes settle at 1. Floor div is the correct conservative choice for a
    /// max-reduction mip: a child whose parent index = floor(child / 2) lands
    /// in the right parent cell. The mip kernel skips out-of-range children.
    fn compute_lods(base_dim: [u32; 3]) -> Vec<VoxelLodDesc> {
        let mut lods = Vec::new();
        let mut dim = base_dim;
        let mut offset_words: u32 = 0;
        for _ in 0..MAX_LODS {
            lods.push(VoxelLodDesc {
                dim: [dim[0] as i32, dim[1] as i32, dim[2] as i32],
                offset: offset_words,
            });
            let voxels: u64 = (dim[0] as u64) * (dim[1] as u64) * (dim[2] as u64);
            let words: u32 = voxels.div_ceil(32) as u32;
            offset_words = offset_words
                .checked_add(words)
                .expect("voxel-grid words overflow u32");
            if dim[0] == 1 && dim[1] == 1 && dim[2] == 1 {
                break;
            }
            dim = [dim[0].max(2) / 2, dim[1].max(2) / 2, dim[2].max(2) / 2];
        }
        lods
    }

    pub fn new(context: &gpu::Context, base_dim: [u32; 3]) -> Self {
        let lods = Self::compute_lods(base_dim);
        let occupancy_words: u64 = lods
            .iter()
            .map(|l| {
                let v = (l.dim[0] as u64) * (l.dim[1] as u64) * (l.dim[2] as u64);
                v.div_ceil(32)
            })
            .sum();
        let buffer_size = METADATA_BYTES + occupancy_words * 4;
        let buffer = context.create_buffer(gpu::BufferDesc {
            name: "voxel-grid",
            size: buffer_size,
            memory: gpu::Memory::Device,
        });
        log::info!(
            "Voxel grid: base {}×{}×{}, {} LODs, {} MiB",
            base_dim[0],
            base_dim[1],
            base_dim[2],
            lods.len(),
            buffer_size >> 20,
        );
        Self {
            buffer,
            buffer_size,
            base_dim,
            lods,
        }
    }

    /// Upload the `lod_count + lods[16]` metadata prefix. Returns the
    /// allocated upload buffer so the caller can park it in
    /// `Submission::temp_buffers` until the copy completes.
    pub fn upload_metadata(
        &self,
        context: &gpu::Context,
        transfer: &mut gpu::TransferCommandEncoder<'_>,
    ) -> gpu::Buffer {
        let mut blob = vec![0u8; METADATA_BYTES as usize];
        let lod_count_bytes: [u8; 16] = bytemuck::cast([self.lods.len() as u32, 0, 0, 0]);
        blob[0..16].copy_from_slice(&lod_count_bytes);
        // Pad to 16 LODs — unused slots stay zeroed (dim = (0,0,0), offset = 0).
        let mut padded: [VoxelLodDesc; MAX_LODS] = [VoxelLodDesc::default(); MAX_LODS];
        for (dst, src) in padded.iter_mut().zip(self.lods.iter()) {
            *dst = *src;
        }
        let lods_bytes: &[u8] = bytemuck::cast_slice(&padded);
        blob[16..16 + lods_bytes.len()].copy_from_slice(lods_bytes);

        let stage = context.create_buffer(gpu::BufferDesc {
            name: "voxel-grid/meta-upload",
            size: METADATA_BYTES,
            memory: gpu::Memory::Upload,
        });
        unsafe {
            std::ptr::copy_nonoverlapping(blob.as_ptr(), stage.data(), blob.len());
        }
        transfer.copy_buffer_to_buffer(stage.into(), self.buffer.into(), METADATA_BYTES);
        stage
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        if self.buffer != gpu::Buffer::default() {
            context.destroy_buffer(self.buffer);
            self.buffer = gpu::Buffer::default();
        }
    }
}

/// Owns the compute pipelines that populate the voxel grid from a heightmap.
/// Runs `cs_init` (LOD 0) then a chain of `cs_mip` dispatches up the LOD
/// pyramid. The baker is reusable across maps as long as the WGSL shaders
/// don't change — the bake parameters are bound per-dispatch.
pub struct VoxelBaker {
    init_pipeline: gpu::ComputePipeline,
    mip_pipeline: gpu::ComputePipeline,
    heightmap_sampler: gpu::Sampler,
}

impl VoxelBaker {
    pub fn new(context: &gpu::Context) -> Self {
        // Prepend shaders/voxel.wgsl into voxel-bake.wgsl, same pattern as
        // common.wgsl + terrain-draw.wgsl. Keeps the shared declarations
        // (VoxelLod, VOXEL_MAX_LODS, voxel_bit_addr) authoritative in
        // one file.
        let voxel_inc =
            std::fs::read_to_string("shaders/voxel.wgsl").expect("read shaders/voxel.wgsl");
        let bake_body = std::fs::read_to_string("shaders/voxel-bake.wgsl")
            .expect("read shaders/voxel-bake.wgsl");
        let source = format!("{voxel_inc}\n{bake_body}");
        let shader = context.create_shader(gpu::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let init_layout = <BakeInitData as gpu::ShaderData>::layout();
        let mip_layout = <BakeMipData as gpu::ShaderData>::layout();
        let init_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "voxel-bake/init",
            data_layouts: &[&init_layout],
            compute: shader.at("cs_init"),
        });
        let mip_pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "voxel-bake/mip",
            data_layouts: &[&mip_layout],
            compute: shader.at("cs_mip"),
        });
        let heightmap_sampler = context.create_sampler(gpu::SamplerDesc {
            name: "voxel-bake/heightmap",
            // Repeat on θ (heightmap wraps), clamp on z. Nearest filter
            // matches the bake's cell-centre sampling — no interpolation
            // between texels.
            address_modes: [
                gpu::AddressMode::Repeat,
                gpu::AddressMode::ClampToEdge,
                gpu::AddressMode::ClampToEdge,
            ],
            mag_filter: gpu::FilterMode::Nearest,
            min_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self {
            init_pipeline,
            mip_pipeline,
            heightmap_sampler,
        }
    }

    /// Run the full bake chain. Caller owns synchronisation: the encoder
    /// must already be started, and the metadata prefix must already be
    /// uploaded into `voxels.buffer` (call `Voxels::upload_metadata` in a
    /// transfer pass before invoking this).
    pub fn bake(
        &self,
        encoder: &mut gpu::CommandEncoder,
        voxels: &Voxels,
        heightmap: &Texture,
        radius_start: f32,
        radius_end: f32,
    ) {
        // Per-word parallelisation: each thread owns one destination u32
        // word of the occupancy bit-grid. WGSL workgroup_size is (64, 1, 1)
        // so each workgroup covers 64 words. Dispatch x = ceil(words / 64).
        const WG_SIZE: u32 = 64;
        let words_in_lod = |dim: [i32; 3]| -> u32 {
            let voxels = (dim[0] as u64) * (dim[1] as u64) * (dim[2] as u64);
            voxels.div_ceil(32) as u32
        };
        let mut pass = encoder.compute("voxel-bake");
        // ---- LOD 0 init ----
        {
            let words = words_in_lod(voxels.lods[0].dim);
            let mut pen = pass.with(&self.init_pipeline);
            pen.bind(
                0,
                &BakeInitData {
                    g_voxels: voxels.buffer.into(),
                    g_bake: BakeParams {
                        radius_start,
                        radius_end,
                        _pad: [0.0; 2],
                    },
                    g_heightmap: heightmap.view,
                    g_heightmap_sampler: self.heightmap_sampler,
                },
            );
            pen.dispatch([words.div_ceil(WG_SIZE), 1, 1]);
        }
        pass.barrier();
        // ---- Mip chain ----
        for src_lod in 0..(voxels.lods.len() - 1) {
            let words = words_in_lod(voxels.lods[src_lod + 1].dim);
            {
                let mut pen = pass.with(&self.mip_pipeline);
                pen.bind(
                    0,
                    &BakeMipData {
                        g_voxels: voxels.buffer.into(),
                        g_mip: MipParams {
                            src_lod: src_lod as u32,
                            _pad: [0; 3],
                        },
                    },
                );
                pen.dispatch([words.div_ceil(WG_SIZE), 1, 1]);
            }
            pass.barrier();
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        context.destroy_compute_pipeline(&mut self.init_pipeline);
        context.destroy_compute_pipeline(&mut self.mip_pipeline);
        context.destroy_sampler(self.heightmap_sampler);
    }
}

/// Pick the LOD 0 dimensions for the voxel grid given a heightmap. We
/// constrain ourselves to powers of two on each axis so the mip chain
/// stays clean, and we hard-cap each axis at the heightmap's own
/// resolution along that axis (no point going finer than source data).
/// Radial bin count is independent: chosen as a multiple of 32 so a
/// column's bits fit nicely into whole u32 words during bake.
pub fn pick_voxel_dim(map_width: u32, map_height: u32, target_radial: u32) -> [u32; 3] {
    fn nearest_pow2_at_most(v: u32) -> u32 {
        if v <= 1 {
            return 1;
        }
        1u32 << (31 - v.leading_zeros())
    }
    // Halve heightmap u/v so each voxel covers ~2 texels (cheap conservative
    // bound; the bisection at LOD-0 refinement still hits the full-res
    // heightmap). Halving lets us fit Fostral's 16384-tall map inside a
    // sensible storage budget at 128 radial bins. If we ever want full
    // heightmap-texel fidelity we can drop this halving and live with the
    // larger buffer.
    let dim_theta = nearest_pow2_at_most(map_width / 2).max(32);
    let dim_z = nearest_pow2_at_most(map_height / 2).max(32);
    let dim_r = target_radial.max(32);
    [dim_theta, dim_z, dim_r]
}
