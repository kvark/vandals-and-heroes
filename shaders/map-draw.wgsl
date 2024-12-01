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

fn intersect_ray_with_map_radius(dir: vec2<f32>, radius: f32) -> vec2<f32> {
    //dot(g_camera.pos.xy + t * dir.xy, g_camera.pos.xy + t * dir.xy) == some_radius ^ 2
    //(g_camera.pos.x + t*dir.x) & ^ 2 + (g_camera.pos.y + t*dir.y) == radius^2
    //g_camera.pos.x^2 + g_camera.pos.y^2 + t^2*(dir.x^2 + dir.y^2) + 2*t*(dir.x + dir.y) == radius^2
    //t^2 * length(dir.xy)^2 + t * 2 * (dir.x + dir.y) + length(g_camera.pos.xy)^2 - radius^2 == 0
    let a = dot(dir, dir);
    let b = 2.0 * (dir.x + dir.y);
    let c = dot(g_camera.pos.xy, g_camera.pos.xy) - radius * radius;
    let d = b * b - 4 * a * c;
    if (d < 0.0) {
        return vec2<f32>(0.0);
    }
    let mul = select(vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), a > 0.0);
    return (mul * sqrt(d) - b) / (2.0 * a);
}

fn compute_ray_distance(dir: vec3<f32>) -> vec2<f32> {
    var result = vec2<f32>(g_camera.clip_near, g_camera.clip_far);
    if (true) {
        return result;
    }
    if (abs(dir.z) > 0.1) {
        // intersect with bottom or top
        let limit = (select(0.0, g_params.length, dir.z > 0.0) - g_camera.pos.z) / dir.z;
        result.y = min(result.y, limit);
        if (result.x >= result.y) {
            // outside of the cylinder length
            return vec2<f32>(0.0);
        }
    }
    let t_end = intersect_ray_with_map_radius(dir.xy, g_params.radius_end);
    result.x = max(result.x, t_end.x);
    result.y = min(result.y, t_end.y);
    if (result.x >= result.y) {
        // ray isn't intersecting with the outer cylinder, it's a guaranteed miss
        return vec2<f32>(0.0);
    }
    let t_start = intersect_ray_with_map_radius(dir.xy, g_params.radius_start);
    if (t_start.y > t_start.x) {
        // stop the ray when it hits the surface
        result.y = min(result.y, t_start.x);
    }
    return result;
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
    let pos = (4.0 * vec2<f32>(ic) - 1.0) * vec2<f32>(1.0, -1.0);
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
    if (distances.x < distances.y) {
        let num_steps = 20;
        var distance = distances.x;
        let distance_step = (distances.y - distances.x) / f32(num_steps);
        for (var i = 0; i < num_steps; i += 1) {
            distance += distance_step;
            var position = g_camera.pos.xyz + distance * in.ray_dir;
            let rc = cartesian_to_radial(position);
            let texel = sample_map(rc);
            let ground_radius = mix(g_params.radius_start, g_params.radius_end, texel.a);
            if (rc.radius <= ground_radius) {
                // hit!
                let normalized_depth = (distance - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
                let texel = vec4<f32>(position.z / g_params.length, rc.alpha / PI, 0.0, 1.0);
                return FragmentOutput(texel, normalized_depth);
            }
        }
    }

    // miss!
    return FragmentOutput(vec4<f32>(0.1, 0.2, 0.3, 1.0), 1.0);
}
