// Voxel-grid bake passes. `cs_init` populates LOD 0 from the heightmap;
// `cs_mip` max-reduces LOD N → LOD N+1 (any-child-occupied ⇒ parent
// occupied). The chain stops in src/voxels.rs once any axis hits 1 voxel.
//
// Single-layer surface: a voxel at radial bin k is occupied iff the column's
// ground radius is ≥ the voxel's lower-r face. Multi-layer extension (tunnel
// floors / overhangs) plugs in by adding extra intervals to the `occupied`
// test below.

// Plain (non-atomic) u32 storage: with the r-fastest bit ordering, every
// bake thread owns disjoint u32 words for the column it processes, so we
// avoid both the validation hassle of vulkanMemoryModelDeviceScope and the
// per-write atomic cost.
struct VoxelData {
    lod_count: vec4u,
    lods: array<VoxelLod, VOXEL_MAX_LODS>,
    occupancy: array<u32>,
}
var<storage, read_write> g_voxels: VoxelData;

struct BakeParams {
    radius_start: f32,
    radius_end: f32,
    _pad0: f32,
    _pad1: f32,
}
var<uniform> g_bake: BakeParams;

var g_heightmap: texture_2d<f32>;
var g_heightmap_sampler: sampler;

struct MipParams {
    src_lod: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
var<uniform> g_mip: MipParams;

// MAX over all heightmap texels in the voxel column's (θ, z) footprint.
//
// The bake's occupancy criterion is "is r_low below the column's ground?",
// so it must use the WORST-CASE (highest) ground inside the footprint —
// any heightmap texel reaching above r_low needs the cell marked occupied,
// even if the other texels are short. The previous bilinear-at-centre
// sample returned the 4-texel AVERAGE, which silently dropped occupancy
// for cells whose only-tall texel was outvoted by short neighbours (the
// signature symptom: rays going UP at tall structures missed the surface
// entirely and the sky env bled through).
//
// For our typical (heightmap × voxel) ratios this is a 2×2 or 4×4 max,
// scaled by dim.x / heightmap_width along each axis.
fn sample_height_at_cell(i_theta: i32, i_z: i32, dim: vec3i) -> f32 {
    let dims = vec2i(textureDimensions(g_heightmap, 0));
    let texels_per_cell_x = max(dims.x / dim.x, 1);
    let texels_per_cell_y = max(dims.y / dim.y, 1);
    let base_x = i_theta * texels_per_cell_x;
    let base_y = i_z * texels_per_cell_y;
    var max_alpha: f32 = 0.0;
    for (var dx: i32 = 0; dx < texels_per_cell_x; dx = dx + 1) {
        for (var dy: i32 = 0; dy < texels_per_cell_y; dy = dy + 1) {
            let uv = (vec2f(f32(base_x + dx), f32(base_y + dy)) + 0.5)
                / vec2f(dims);
            let alpha = textureSampleLevel(g_heightmap, g_heightmap_sampler, uv, 0.0).a;
            max_alpha = max(max_alpha, alpha);
        }
    }
    return max_alpha;
}

// Both kernels parallelise per destination u32 word (not per voxel). With
// r-fastest packing this gives one thread complete ownership of each word —
// no races, no atomics, validation-layer friendly. The thread iterates 32
// dst bits inside the word, decoding (i_θ, i_z, i_r) per bit.

@compute @workgroup_size(64, 1, 1)
fn cs_init(@builtin(global_invocation_id) gid: vec3u) {
    let lod = g_voxels.lods[0];
    let dim = lod.dim;
    let total_voxels = u32(dim.x) * u32(dim.y) * u32(dim.z);
    let total_words = (total_voxels + 31u) / 32u;
    let word_idx = gid.x;
    if (word_idx >= total_words) {
        return;
    }

    let dr_per_cell = (g_bake.radius_end - g_bake.radius_start) / f32(dim.z);
    let dr = u32(dim.z);
    let dy = u32(dim.y);
    let bit_base = word_idx * 32u;

    // Fast path: when all 32 bits in a word land in the same (i_θ, i_z)
    // column, we sample the heightmap once and only vary r. With dim.z a
    // multiple of 32 (our default) and word-aligned addressing, this is
    // always true; the slow path catches edge cases for safety.
    let zt_first = bit_base / dr;
    let zt_last = (bit_base + 31u) / dr;
    let same_column = zt_first == zt_last;

    var word: u32 = 0u;
    if (same_column) {
        let i_z = i32(zt_first % dy);
        let i_theta = i32(zt_first / dy);
        let i_r_first = bit_base % dr;
        let alpha = sample_height_at_cell(i_theta, i_z, dim);
        let ground_r = mix(g_bake.radius_start, g_bake.radius_end, alpha);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let i_r = i_r_first + k;
            if (i_r >= dr) { break; }
            let r_low = g_bake.radius_start + f32(i_r) * dr_per_cell;
            if (r_low < ground_r) {
                word = word | (1u << k);
            }
        }
    } else {
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let bit_lin = bit_base + k;
            if (bit_lin >= total_voxels) { break; }
            let i_r = bit_lin % dr;
            let zt = bit_lin / dr;
            let i_z = i32(zt % dy);
            let i_theta = i32(zt / dy);
            let alpha = sample_height_at_cell(i_theta, i_z, dim);
            let ground_r = mix(g_bake.radius_start, g_bake.radius_end, alpha);
            let r_low = g_bake.radius_start + f32(i_r) * dr_per_cell;
            if (r_low < ground_r) {
                word = word | (1u << k);
            }
        }
    }
    g_voxels.occupancy[lod.offset + word_idx] = word;
}

@compute @workgroup_size(64, 1, 1)
fn cs_mip(@builtin(global_invocation_id) gid: vec3u) {
    let src_lod = g_voxels.lods[g_mip.src_lod];
    let dst_lod = g_voxels.lods[g_mip.src_lod + 1u];
    let dst_dim = dst_lod.dim;
    let dst_total_voxels = u32(dst_dim.x) * u32(dst_dim.y) * u32(dst_dim.z);
    let dst_total_words = (dst_total_voxels + 31u) / 32u;
    let word_idx = gid.x;
    if (word_idx >= dst_total_words) {
        return;
    }

    let dr_dst = u32(dst_dim.z);
    let dy_dst = u32(dst_dim.y);
    let bit_base = word_idx * 32u;

    var word: u32 = 0u;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        let bit_lin = bit_base + k;
        if (bit_lin >= dst_total_voxels) { break; }
        let i_r_dst = i32(bit_lin % dr_dst);
        let zt = bit_lin / dr_dst;
        let i_z_dst = i32(zt % dy_dst);
        let i_theta_dst = i32(zt / dy_dst);

        var occ: bool = false;
        for (var c: u32 = 0u; c < 8u; c = c + 1u) {
            let off = vec3i(i32(c & 1u), i32((c >> 1u) & 1u), i32((c >> 2u) & 1u));
            let child = vec3i(i_theta_dst, i_z_dst, i_r_dst) * 2 + off;
            if (any(child >= src_lod.dim)) { continue; }
            let addr = voxel_bit_addr(child, src_lod);
            if ((g_voxels.occupancy[addr.word] & addr.mask) != 0u) {
                occ = true;
                break;
            }
        }
        if (occ) {
            word = word | (1u << k);
        }
    }
    g_voxels.occupancy[dst_lod.offset + word_idx] = word;
}
