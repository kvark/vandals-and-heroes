fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}

struct CameraParams {
    pos: vec3<f32>,
    rot: vec4<f32>,
    fov: vec2<f32>,
}

struct DrawParams {
    screen_size: vec2<f32>,
}
var<uniform> g_camera: CameraParams;
var<uniform> g_params: DrawParams;
//var input: texture_2d<f32>;
//var samp: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ray_dir: vec3<f32>,
}

@vertex
fn vs_draw(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;
    let tc = vec2<f32>(vec2<u32>(vi & 1u, (vi & 2u) >> 1u));
    let pos = 4.0 * tc - 1.0;
    vo.clip_pos = vec4<f32>(pos * vec2<f32>(1.0, -1.0), 0.0, 1.0);
    let local_dir = vec3<f32>(sin(pos * g_camera.fov), 1.0);
    vo.ray_dir = normalize(qrot(g_camera.rot, local_dir));
    return vo;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_draw(in: VertexOutput) -> FragmentOutput {
    return FragmentOutput(vec4<f32>(in.ray_dir, 1.0), 1.0);
}
