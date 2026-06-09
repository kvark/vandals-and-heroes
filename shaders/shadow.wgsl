// Cylindrical "shadow map" pass.
//
// Rasterizes dynamic occluders into a depth texture parameterised by (theta, z)
// on the cylinder. The unwrap is done explicitly in the vertex shader (no
// perspective divide, just orthographic in cylindrical coords). cyl_depth()
// (in common.wgsl) maps r → depth so larger r ("closer to sun at radial
// infinity") gets smaller depth. Standard depth-Less wins.
//
// The terrain is not baked in: the heightmap defines a single r per (theta, z),
// so by construction the terrain is its own topmost surface and only dynamic
// meshes can cast shadows on it. The shadow texture is cleared to 1.0 (= "no
// occluder above"), then dynamic models write smaller depth where they sit
// above the ground.

struct ModelParams {
    transform: mat3x4f,
}
var<uniform> g_params: ModelParams;

struct Vertex {
    position: vec3f,
    normal: u32,
    tex_coords: vec2f,
    pad: vec2f,
}
var<storage, read> g_vertices: array<Vertex>;

struct ShadowModelOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) depth: f32,
}

@vertex
fn vs_shadow_model(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> ShadowModelOut {
    let v = g_vertices[vi];
    let p_world = (transpose(g_params.transform) * vec4f(v.position, 1.0)).xyz;
    let r = length(p_world.xy);

    // Two-part seam handling:
    //
    // 1) Per-vertex θ unwrap, anchored at the model origin (the .w
    //    components of its rendering transform). Without this, a triangle
    //    straddling θ = ±π would interpolate clip_x from +1 to -1 *the long
    //    way*, smearing a horizontal stripe across the shadow map. After
    //    unwrap, every triangle's vertices sit within ±π of the anchor and
    //    triangles are coherent.
    //
    // 2) Two instance copies, the second shifted by ±2π. Unwrap alone is
    //    not enough: if the model anchor sits near ±π, the unwrapped
    //    triangles still clip off the side of the shadow map. The second
    //    instance shifts clip_x by ±2 so the half that would otherwise be
    //    off-screen renders on the opposite edge. For models far from the
    //    seam the second instance is fully off-screen and rasterized away
    //    for free; for models near the seam it stitches the shadow across.
    let model_origin = vec3f(
        g_params.transform[0].w,
        g_params.transform[1].w,
        g_params.transform[2].w,
    );
    let theta_anchor = atan2(model_origin.y, model_origin.x);
    let theta_raw = atan2(p_world.y, p_world.x);
    var theta = theta_raw;
    let diff = theta_raw - theta_anchor;
    if (diff > PI) {
        theta = theta_raw - TAU;
    } else if (diff < -PI) {
        theta = theta_raw + TAU;
    }

    // Shift the second instance to the opposite side of the seam from the
    // anchor. If the anchor is at positive θ, the seam is to the right at
    // +π and the wrapped copy should appear on the left edge — shift by
    // -2π. Symmetric for negative anchors.
    var clip_x = theta / PI;
    if (ii == 1u) {
        if (theta_anchor >= 0.0) {
            clip_x = clip_x - 2.0;
        } else {
            clip_x = clip_x + 2.0;
        }
    }
    let clip_y = -p_world.z * 2.0 / g_cyl.length;
    var vo: ShadowModelOut;
    vo.clip_pos = vec4f(clip_x, clip_y, 0.5, 1.0);
    vo.depth = cyl_depth(r);
    return vo;
}

@fragment
fn fs_shadow_model(in: ShadowModelOut) -> @location(0) f32 {
    return in.depth;
}
