const PI: f32 = 3.1415926;

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
fn cartesian_to_radial(p: vec3f) -> RadialCoordinates {
    var rc: RadialCoordinates;
    rc.alpha = atan2(p.y, p.x);
    rc.radius = length(p.xy);
    rc.depth = p.z;
    return rc;
}

fn sample_map(rc: RadialCoordinates) -> vec4f {
    let tc = vec2f(rc.alpha / (2.0 * PI), rc.depth / g_params.length);
    return textureSampleLevel(g_map, g_sampler, tc, 0.0);
}

fn intersect_ray_with_map_radius(dir: vec2f, radius: f32) -> vec2f {
    // let cp = g_camera.pos.xy;
    // dot(cp + t * dir, cp + t * dir) == radius ^ 2
    // cp.x^2 + cp.y^2 + t^2*(dir.x^2 + dir.y^2) + 2*t*(dir.x*cp.x + dir.y*cp.y) == radius^2
    // t^2 * length(dir.xy)^2 + t * 2 * dot(dir, cp) + length(cp)^2 - radius^2 == 0
    let a = dot(dir, dir);
    let b = 2.0 * dot(dir, g_camera.pos.xy);
    let c = dot(g_camera.pos.xy, g_camera.pos.xy) - radius * radius;
    let d = b * b - 4 * a * c;
    if (d < 0.0) {
        return vec2f(0.0);
    }
    let signs = select(vec2f(1.0, -1.0), vec2f(-1.0, 1.0), a > 0.0);
    return (signs * sqrt(d) - b) / (2.0 * a);
}

fn compute_ray_distance(dir: vec3f) -> vec2f {
    var result = vec2f(g_camera.clip_near, g_camera.clip_far);
    // intersect with bottom or top
    let limit = (select(0.0, g_params.length, dir.z > 0.0) - g_camera.pos.z) / dir.z;
    result.y = min(result.y, limit);
    if (result.x >= result.y) {
        // outside of the cylinder length
        return vec2f(0.0);
    }
    let t_end = intersect_ray_with_map_radius(dir.xy, g_params.radius_end);
    result.x = max(result.x, t_end.x);
    result.y = min(result.y, t_end.y);
    if (result.x >= result.y) {
        // ray isn't intersecting with the outer cylinder, it's a guaranteed miss
        return vec2f(0.0);
    }
    let t_start = intersect_ray_with_map_radius(dir.xy, g_params.radius_start);
    if (t_start.y > t_start.x) {
        // stop the ray when it hits the surface
        result.y = min(result.y, t_start.x);
    }
    return result;
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) ray_dir: vec3f,
}

@vertex
fn vs_draw(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var vo: VertexOutput;
    let ic = vec2<u32>(vi & 1u, (vi & 2u) >> 1u);
    //Note: camera coordinate system is X-right, Y-down, Z-forward
    let pos = (4.0 * vec2f(ic) - 1.0) * vec2f(1.0, -1.0);
    vo.clip_pos = vec4f(pos, 0.0, 1.0);
    let local_dir = vec3f(pos * g_camera.half_plane, 1.0);
    vo.ray_dir = qrot(g_camera.rot, local_dir);
    return vo;
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_draw(in: VertexOutput) -> FragmentOutput {
    let distances = compute_ray_distance(in.ray_dir);
    if (distances.x < distances.y) {
        let num_steps = 20;
        for (var i = 0; i < num_steps; i += 1) {
            let distance_ratio = pow(f32(i + 1) / f32(num_steps), 2.0);
            let distance = mix(distances.x, distances.y, distance_ratio);
            var position = g_camera.pos.xyz + distance * in.ray_dir;
            let rc = cartesian_to_radial(position);
            let texel = sample_map(rc);
            let ground_radius = mix(g_params.radius_start, g_params.radius_end, texel.a);
            if (rc.radius <= ground_radius) {
                // hit!
                let normalized_depth = (distance - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
                return FragmentOutput(texel, normalized_depth);
            }
        }
    }

    // miss!
    return FragmentOutput(vec4f(0.1, 0.2, 0.3, 1.0), 1.0);
}
