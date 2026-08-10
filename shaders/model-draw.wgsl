// Shared constants, qrot/qinv, CylParams + g_cyl + the radial-coordinate
// helpers live in common.wgsl and are prepended at shader-load time.

struct CameraParams {
    pos: vec3f,
    rot: vec4f,
    half_plane: vec2f,
    clip_near: f32,
    clip_far: f32,
}
var<uniform> g_camera: CameraParams;

var g_shadow: texture_2d<f32>;
var g_shadow_sampler: sampler;

var g_environment: texture_2d<f32>;
var g_env_sampler: sampler;

struct ModelParams {
    transform: mat3x4f,
    base_color_factor: vec4f,
}
var<uniform> g_params: ModelParams;

var g_base_color: texture_2d<f32>;
var g_normal: texture_2d<f32>;
var g_sampler: sampler;

fn sky_visibility(p_world: vec3f) -> f32 {
    if (g_cyl.shadows_enabled == 0u) {
        return 1.0;
    }
    let rc = cartesian_to_radial(p_world);
    let uv = shadow_uv(rc);
    let d_frag = cyl_depth(rc.radius);
    let texel = 1.0 / vec2f(textureDimensions(g_shadow, 0));
    let off = texel * SHADOW_SAMPLE_SPREAD;
    var sum = 0.0;
    var count = 0.0;
    for (var dy = -SHADOW_PCF_RADIUS; dy <= SHADOW_PCF_RADIUS; dy = dy + 1) {
        for (var dx = -SHADOW_PCF_RADIUS; dx <= SHADOW_PCF_RADIUS; dx = dx + 1) {
            let p = uv + vec2f(f32(dx), f32(dy)) * off;
            let d_shadow = textureSampleLevel(g_shadow, g_shadow_sampler, p, 0.0).r;
            sum = sum + smoothstep(d_frag - SHADOW_SOFTNESS, d_frag + SHADOW_BIAS, d_shadow);
            count = count + 1.0;
        }
    }
    return sum / count;
}

// Fetched from a plain vertex buffer (see `Vertex` on the Rust side) so the
// pipeline runs on WebGL2-class devices with no storage buffers.
struct Vertex {
    position: vec3f,
    normal: u32,
    tex_coords: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) tex_coords: vec2f,
    @location(1) world_pos: vec3f,
    @location(2) world_normal: vec3f,
    // Forwarded from g_params so the fragment stage does not reference the
    // uniform block at all — naga's GLSL-ES output cannot link a block that
    // both stages of a pipeline use.
    @location(3) base_color_factor: vec4f,
}

@vertex
fn vs_model(v: Vertex) -> VertexOutput {
    let p_world = (transpose(g_params.transform) * vec4f(v.position, 1.0)).xyz;
    let p_camera = qrot(qinv(g_camera.rot), p_world - g_camera.pos);
    var vo: VertexOutput;
    let depth = (p_camera.z - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
    vo.clip_pos = vec4f(p_camera.xy / g_camera.half_plane, depth * p_camera.z, p_camera.z);
    vo.tex_coords = v.tex_coords;
    vo.world_pos = p_world;
    vo.base_color_factor = g_params.base_color_factor;
    let local_normal = normalize(unpack4x8snorm(v.normal).xyz);
    // The transform's upper 3x3 (after transpose) is the rotation+scale. For a
    // rigid (or near-rigid) transform, applying it to the normal is fine; for
    // scaled transforms we would want the inverse-transpose.
    let m = transpose(g_params.transform);
    let n_world = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz) * local_normal;
    vo.world_normal = normalize(n_world);
    return vo;
}

// Ambient floor for surfaces facing away from the radial "sun". Without it,
// undersides go pure black; with it, they keep the albedo at a fraction of full
// brightness — closer to the matte-rust look.
const MODEL_AMBIENT: f32 = 0.3;

@fragment
fn fs_model(vi: VertexOutput) -> @location(0) vec4f {
    let base_color = textureSample(g_base_color, g_sampler, vi.tex_coords);
    let albedo = vi.base_color_factor * base_color;
    // Non-reflective shading: treat the "sun" as the radial-outward direction
    // (matches the inward gravity convention). Lambert against that direction
    // gives the silhouette some shape without sampling any env-map colour.
    let radial_out = world_up(vi.world_pos);
    let n_dot_r = max(0.0, dot(vi.world_normal, radial_out));
    let light = mix(MODEL_AMBIENT, 1.0, n_dot_r);
    let vis = sky_visibility(vi.world_pos);
    return vec4f(tone(albedo.rgb * light * vis), albedo.a);
}
