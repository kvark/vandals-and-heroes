fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}
fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz,q.w);
}

struct CameraParams {
    pos: vec3<f32>,
    rot: vec4<f32>,
    fov: vec2<f32>,
    clip_near: f32,
    clip_far: f32,
}

struct DrawParams {
    radius_start: f32,
    radius_end: f32,
    length: f32,
}
var<uniform> g_camera: CameraParams;
var<uniform> g_params: DrawParams;
var g_map: texture_2d<f32>;
var g_sampler: sampler;

struct RadialCoordinates {
    alpha: f32,
    radius: f32,
    depth: f32,
}
fn cartesian_to_radial(p: vec3<f32>) -> RadialCoordinates {
    var rc: RadialCoordinates;
    rc.alpha = atan2(p.x, p.y);
    rc.radius = length(p.xy);
    rc.depth = p.z;
    return rc;
}
fn sample_map(rc: RadialCoordinates) -> vec4<f32> {
    let tc = vec2<f32>(rc.alpha, rc.depth);
    return textureSampleLevel(g_map, g_sampler, tc, 0.0);
}

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
    let max_distance = 100.0;
    let num_steps = 20;
    let step = in.ray_dir * max_distance / f32(num_steps);
    var position = g_camera.pos.xyz;
    for (var i = 0; i < num_steps; i += 1) {
        position += step;
        let rc = cartesian_to_radial(position);
        let texel = sample_map(rc);
        let ground_radius = mix(g_params.radius_start, g_params.radius_end, texel.a);
        if (rc.radius <= ground_radius) {
            // hit!
            let pos_local = qrot(qinv(g_camera.rot), position - g_camera.pos.xyz);
            let normalized_depth = (pos_local.z - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
            return FragmentOutput(texel, normalized_depth);
        }
    }

    return FragmentOutput(vec4<f32>(0.1, 0.2, 0.3, 1.0), 1.0);
}
