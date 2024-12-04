//TODO: share the header

fn qrot(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}
fn qinv(q: vec4f) -> vec4f {
    return vec4f(-q.xyz,q.w);
}

struct CameraParams {
    pos: vec3f,
    rot: vec4f,
    half_plane: vec2f,
    clip_near: f32,
    clip_far: f32,
}
var<uniform> g_camera: CameraParams;

struct ModelParams {
    transform: mat3x4f,
    base_color_factor: vec4f,
}
var<uniform> g_params: ModelParams;

struct Vertex {
    position: vec3f,
    normal: vec3f,
    tex_coords: vec2f,
}
var<storage, read> g_vertices: array<Vertex>;

var g_base_color: texture_2d<f32>;
var g_normal: texture_2d<f32>;
var g_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) tex_coords: vec2f,
}

@vertex
fn vs_model(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let v = g_vertices[vi];
    let p_world = transpose(g_params.transform) * vec4f(v.position, 1.0);
    let p_camera = qrot(qinv(g_camera.rot), p_world - g_camera.pos);
    var vo: VertexOutput;
    let depth = (p_camera.z - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
    vo.clip_pos = vec4f(g_camera.half_plane * p_camera.xy, depth * p_camera.z, p_camera.z);
    vo.tex_coords = v.tex_coords;
    return vo;
}

@fragment
fn fs_model(vi: VertexOutput) -> @location(0) vec4f {
    let base_color = textureSample(g_base_color, g_sampler, vi.tex_coords);
    return g_params.base_color_factor * base_color;
}
