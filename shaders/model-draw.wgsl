// Shared constants, qrot/qinv, CylParams + g_cyl + cyl_depth live in common.wgsl
// and are prepended at shader-load time.

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

struct Vertex {
    position: vec3f,
    normal: u32,
    tex_coords: vec2f,
    pad: vec2f,
}
var<storage, read> g_vertices: array<Vertex>;

var g_base_color: texture_2d<f32>;
var g_normal: texture_2d<f32>;
var g_sampler: sampler;

fn sample_environment(dir: vec3f) -> vec3f {
    let d = normalize(dir);
    let u = atan2(d.y, d.x) / TAU + 0.5;
    let v = acos(clamp(d.z, -1.0, 1.0)) / PI;
    return textureSampleLevel(g_environment, g_env_sampler, vec2f(u, v), 0.0).rgb;
}

fn sky_visibility(p_world: vec3f) -> f32 {
    let theta = atan2(p_world.y, p_world.x);
    let r = length(p_world.xy);
    let uv = vec2f(theta / TAU + 0.5, p_world.z / g_cyl.length + 0.5);
    let d_frag = cyl_depth(r);
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

struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) tex_coords: vec2f,
    @location(1) world_pos: vec3f,
    @location(2) world_normal: vec3f,
}

@vertex
fn vs_model(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let v = g_vertices[vi];
    let p_world = (transpose(g_params.transform) * vec4f(v.position, 1.0)).xyz;
    let p_camera = qrot(qinv(g_camera.rot), p_world - g_camera.pos);
    var vo: VertexOutput;
    let depth = (p_camera.z - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
    vo.clip_pos = vec4f(p_camera.xy / g_camera.half_plane, depth * p_camera.z, p_camera.z);
    vo.tex_coords = v.tex_coords;
    vo.world_pos = p_world;
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
    let albedo = g_params.base_color_factor * base_color;
    // Non-reflective shading: treat the "sun" as a radial-outward direction
    // (matches the inward gravity convention). Lambert against that direction
    // gives the silhouette some shape without sampling any env-map colour.
    let radial_xy = vi.world_pos.xy;
    let r = max(length(radial_xy), 1e-6);
    let radial_out = vec3f(radial_xy / r, 0.0);
    let n_dot_r = max(0.0, dot(vi.world_normal, radial_out));
    let light = mix(MODEL_AMBIENT, 1.0, n_dot_r);
    let vis = sky_visibility(vi.world_pos);
    return vec4f(albedo.rgb * light * vis, albedo.a);
}
