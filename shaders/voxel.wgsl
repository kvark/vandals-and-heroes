// Shared voxel-grid declarations. Prepended into voxel-bake.wgsl and into
// the voxel HiZ DDA path in terrain-draw.wgsl (alongside common.wgsl).
//
// Grid lives in CURVED coordinates: (θ-cell, z-cell, r-cell) for the cylinder
// world, (θ-cell, sinφ-cell, r-cell) for the sphere world. The ray is straight
// in 3D Cartesian; the DDA computes analytical cell-boundary crossings against
// the curved cell faces. See VOXEL_TRACING.md for the design rationale.
//
// Occupancy is bit-packed into the storage buffer's `occupancy` array — one
// bit per voxel, 32 voxels per u32 word, linear order with **r varying
// fastest** (θ slowest):
//   bit_index = (i_theta * dim_z + i_z) * dim_r + i_r
//   word      = occupancy[ bit_index / 32 ]
//   mask      = 1u << (bit_index % 32)
// Why r-fastest: a single bake thread owns one (i_θ, i_z) column and writes
// all dim_r/32 contiguous words for that column without any other thread
// touching the same words → no atomics. Bonus, DDA "descend into finer LOD"
// reads radial neighbours, which now share a cache line.

const VOXEL_MAX_LODS: u32 = 16u;

struct VoxelLod {
    dim: vec3i,    // cells along (θ, z, r) at this LOD
    offset: u32,   // u32-word offset into `occupancy` for this LOD's base
}

// linearize: voxel coords → bit address inside the occupancy array.
// `lod` provides both this LOD's dim and its base offset, so callers don't
// need to know LOD layout details.
struct VoxelBitAddr {
    word: u32,
    mask: u32,
}
fn voxel_bit_addr(coords: vec3i, lod: VoxelLod) -> VoxelBitAddr {
    // wrap θ (periodic), clamp z and r (bounded — the DDA must already have
    // exited the grid before this happens).
    let dim = lod.dim;
    var c = coords;
    c.x = c.x % dim.x;
    c.x = select(c.x, c.x + dim.x, c.x < 0);
    // r varies fastest, θ slowest: bit = (θ * dim_z + z) * dim_r + r.
    let bit = u32(c.z) + u32(dim.z) * (u32(c.y) + u32(dim.y) * u32(c.x));
    var addr: VoxelBitAddr;
    addr.word = lod.offset + bit / 32u;
    addr.mask = 1u << (bit & 31u);
    return addr;
}
