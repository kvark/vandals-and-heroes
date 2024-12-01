const PI: f32 = 3.1415926;

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0*cross(q.xyz, cross(q.xyz,v) + q.w*v);
}
fn qinv(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz,q.w);
}

struct CameraParams {
    pos: vec3<f32>,
    rot: vec4<f32>,
    half_plane: vec2<f32>,
    clip_near: f32,
    clip_far: f32,
}
var<uniform> g_camera: CameraParams;

fn project_depth(pos: vec3<f32>) -> f32 {
    let pos_local = qrot(qinv(g_camera.rot), pos - g_camera.pos.xyz);
    return (pos_local.z - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
}

struct DrawParams {
    radius_start: f32,
    radius_end: f32,
    length: f32,
}
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
    rc.alpha = atan2(p.y, p.x);
    rc.radius = length(p.xy);
    rc.depth = p.z;
    return rc;
}

fn sample_map(rc: RadialCoordinates) -> vec4<f32> {
    let tc = vec2<f32>(rc.alpha / (2.0 * PI), rc.depth / g_params.length);
    return textureSampleLevel(g_map, g_sampler, tc, 0.0);
}

fn compute_ray_distance(dir: vec3<f32>) -> vec2<f32> {
    var result = vec2<f32>(g_camera.clip_near, g_camera.clip_far);
    if (abs(dir.z) > 0.1) {
        // intersect with bottom or top
        let limit = (select(0.0, g_params.length, dir.z > 0.0) - g_camera.pos.z) / dir.z;
        result.y = clamp(result.y, result.x, limit);
    }
    return result;
    // Find the point closest to the cylinder
    /*
    let t_perp = -dot(g_camera.pos.xy, dir.xy);
    let radius = length(g_camera.pos.xy + t_perp * dir.xy);
    let p = g_camera.pos + t * dir;
    dot(p.xy, p.xy) = some_radius^2
    dot(g_camera.pos.xy + t * dir.xy, g_camera.pos.xy + t * dir.xy) == some_radius ^ 2
    (g_camera.pos.x + t*dir.x) & ^ 2 + (g_camera.pos.y + t*dir.y) == radius^2
    g_camera.pos.x^2 + g_camera.pos.y^2 + t^2*(dir.x^2 + dir.y^2) + 2*t*(dir.x + dir.y) == radius^2
    t^2 * length(dir.xy)^2 + t * 2 * (dir.x + dir.y) + length(g_camera.pos.xy)^2 - radius^2 == 0
    */
    //t = -2 * (dir.x + dir.y) +-
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ray_dir: vec3<f32>,
}

@vertex
fn vs_draw(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;
    let ic = vec2<u32>(vi & 1u, (vi & 2u) >> 1u);
    //Note: camera coordinate system is X-right, Y-down, Z-forward
    let pos = 2.0 * vec2<f32>(vec2<u32>(ic.x, 1u - ic.y)) - 1.0;
    vo.clip_pos = vec4<f32>(pos, 0.0, 1.0);
    let local_dir = vec3<f32>(pos * g_camera.half_plane, 1.0);
    vo.ray_dir = qrot(g_camera.rot, local_dir);
    return vo;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_draw(in: VertexOutput) -> FragmentOutput {
    let distances = compute_ray_distance(in.ray_dir);
    let num_steps = 20;
    let step = in.ray_dir * (distances.y - distances.x) / f32(num_steps);
    var position = g_camera.pos.xyz + in.ray_dir * distances.x;
    for (var i = 0; i < num_steps; i += 1) {
        position += step;
        let rc = cartesian_to_radial(position);
        let texel = sample_map(rc);
        let ground_radius = mix(g_params.radius_start, g_params.radius_end, texel.a);
        if (rc.radius <= ground_radius) {
            // hit!
            let normalized_depth = project_depth(position);
            return FragmentOutput(texel, normalized_depth);
        }
    }

    // miss!
    return FragmentOutput(vec4<f32>(0.1, 0.2, 0.3, 1.0), 1.0);
}
