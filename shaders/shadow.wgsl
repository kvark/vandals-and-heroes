// "Sun-at-radial-infinity" shadow map pass.
//
// Rasterizes dynamic occluders into a depth texture parameterised by the
// same (u, v) space as the height map — (θ, z) on the cylinder, Lambert
// (θ, sin φ) on the sphere, (tube angle, centreline arc) on the torus. The
// unwrap is done explicitly in the vertex shader (no perspective divide,
// just orthographic in radial coords). cyl_depth() (in common.wgsl) maps
// r → depth so larger r ("closer to the sun") gets smaller depth, and the
// R16Float target uses Min blending so the closest occluder wins.
//
// The terrain is not baked in: the heightmap defines a single r per (u, v),
// so by construction the terrain is its own topmost surface and only dynamic
// meshes can cast shadows on it. The shadow texture is cleared to 1.0 (= "no
// occluder above"), then dynamic models write smaller depth where they sit
// above the ground.

struct ModelParams {
    transform: mat3x4f,
}
var<uniform> g_params: ModelParams;

// Fetched from the same vertex buffer as model-draw; only the position
// attribute is consumed here.
struct Vertex {
    position: vec3f,
}

struct ShadowModelOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) depth: f32,
}

@vertex
fn vs_shadow_model(
    v: Vertex,
    @builtin(instance_index) ii: u32,
) -> ShadowModelOut {
    let p_world = (transpose(g_params.transform) * vec4f(v.position, 1.0)).xyz;
    let rc = cartesian_to_radial(p_world);

    // Two-part seam handling:
    //
    // 1) Per-vertex α unwrap, anchored at the model origin (the .w
    //    components of its rendering transform). Without this, a triangle
    //    straddling α = ±π would interpolate clip_x from +1 to -1 *the long
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
    let anchor_rc = cartesian_to_radial(model_origin);
    let theta_anchor = anchor_rc.alpha;
    let theta_raw = rc.alpha;
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
    // clip_y is the vertical clip-space coord, matching shadow_uv's v axis:
    // cylinder/torus map depth/length to [-1, 1]; the sphere maps -sin φ
    // (negated so increasing latitude keeps the "north is up" convention).
    var clip_y: f32;
    if (g_cyl.world_shape == SHAPE_SPHERE) {
        clip_y = -rc.depth;
    } else {
        clip_y = -rc.depth * 2.0 / g_cyl.length;
    }
    var vo: ShadowModelOut;
    vo.clip_pos = vec4f(clip_x, clip_y, 0.5, 1.0);
    vo.depth = cyl_depth(rc.radius);
    return vo;
}

@fragment
fn fs_shadow_model(in: ShadowModelOut) -> @location(0) f32 {
    return in.depth;
}
