// Terrain mesh rasterizer.
//
// The CPU fits a triangulated irregular network to the height map (see
// `src/tin.rs`), so the vertex stage is just the camera transform.
// Everything about the *shading* still comes from the terrain texture — the
// colour, the surface gradient for the normal, the horizon AO. That matters:
// the triangles are deliberately coarse, but colour boundaries stay at full
// texel resolution, so this mode keeps the ray-traced-era colouring instead
// of flat-shading each triangle.
//
// Shared constants, qrot/qinv, CylParams + g_cyl, cartesian_to_radial and
// friends live in common.wgsl and are prepended at shader-load time.

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

var g_terrain: texture_2d<f32>;
var g_terrain_sampler: sampler;

fn sample_environment(dir: vec3f) -> vec3f {
    let d = normalize(dir);
    // World Z is the cylinder axis ("up" for the env panorama). Equirectangular UV:
    // u wraps around the horizontal angle, v goes from top (z=+1) to bottom (z=-1).
    let u = atan2(d.y, d.x) / TAU + 0.5;
    let v = acos(clamp(d.z, -1.0, 1.0)) / PI;
    return textureSampleLevel(g_environment, g_env_sampler, vec2f(u, v), 0.0).rgb;
}

fn sample_map(rc: RadialCoordinates) -> vec4f {
    return textureSampleLevel(g_terrain, g_terrain_sampler, terrain_uv(rc), 0.0);
}

// Bilinear height gradient → world-space outward surface normal, smooth
// across triangle boundaries. On the cylinder the surface is
// p(θ, z) = (r·cos θ, r·sin θ, z); on the sphere
// p(θ, φ) = r · (cos φ·cos θ, cos φ·sin θ, sin φ); on the torus
// p(θ, φ) = C(φ) + r·(cos θ·e_r(φ) + sin θ·Z). The cross product of the two
// tangents is the outward normal in every case.
fn terrain_normal(rc: RadialCoordinates) -> vec3f {
    let tc = terrain_uv(rc);
    let dims = vec2f(textureDimensions(g_terrain, 0));
    let texel = 1.0 / dims;
    let h_l = textureSampleLevel(g_terrain, g_terrain_sampler, tc - vec2f(texel.x, 0.0), 0.0).a;
    let h_r = textureSampleLevel(g_terrain, g_terrain_sampler, tc + vec2f(texel.x, 0.0), 0.0).a;
    let h_b = textureSampleLevel(g_terrain, g_terrain_sampler, tc - vec2f(0.0, texel.y), 0.0).a;
    let h_t = textureSampleLevel(g_terrain, g_terrain_sampler, tc + vec2f(0.0, texel.y), 0.0).a;
    let dr_range = g_cyl.radius_end - g_cyl.radius_start;
    let dh_du = (h_r - h_l) * 0.5 / texel.x; // d(alpha) / d(u)
    let dh_dv = (h_t - h_b) * 0.5 / texel.y; // d(alpha) / d(v)
    let cos_t = cos(rc.alpha);
    let sin_t = sin(rc.alpha);
    let r = rc.radius;
    if (g_cyl.world_shape == SHAPE_SPHERE) {
        // u = θ/TAU, v = (sin φ + 1)/2, so dθ/du = TAU and d(sin φ)/dv = 2.
        // Reparameterise the surface by (θ, sin φ) = (θ, s):
        //   p(θ, s) = r(θ, s) · (sqrt(1 - s²) cos θ, sqrt(1 - s²) sin θ, s)
        let dr_dtheta = dh_du * dr_range / TAU;
        let dr_ds = dh_dv * dr_range * 0.5; // 2 dv = ds
        let s = rc.depth; // sin φ
        let c = sqrt(max(1.0 - s * s, 0.0)); // cos φ
        let radial = vec3f(c * cos_t, c * sin_t, s);
        // ∂p/∂θ = dr_dθ · radial + r · (-c sin θ, c cos θ, 0)
        let dp_dtheta = dr_dtheta * radial + vec3f(-r * c * sin_t, r * c * cos_t, 0.0);
        // ∂p/∂s  = dr_ds · radial + r · (-s/c · cos θ, -s/c · sin θ, 1)
        //   (derivative of (c cos θ, c sin θ, s) w.r.t. s, with dc/ds = -s/c)
        let dp_ds = dr_ds * radial + vec3f(-r * s / max(c, 1e-3) * cos_t, -r * s / max(c, 1e-3) * sin_t, r);
        return normalize(cross(dp_dtheta, dp_ds));
    }
    if (g_cyl.world_shape == SHAPE_TORUS) {
        // depth is the arc length: φ = depth / R. dφ/dv = TAU.
        let phi = rc.depth / g_cyl.length * TAU;
        let e_r = vec3f(cos(phi), sin(phi), 0.0);
        let e_phi = vec3f(-sin(phi), cos(phi), 0.0);
        let outward = cos_t * e_r + vec3f(0.0, 0.0, sin_t);
        let dr_dtheta = dh_du * dr_range / TAU;
        let dr_dphi = dh_dv * dr_range / TAU;
        let dp_dtheta = dr_dtheta * outward
            + r * (-sin_t * e_r + vec3f(0.0, 0.0, cos_t));
        let ring = g_cyl.major_radius + r * cos_t;
        let dp_dphi = dr_dphi * outward + ring * e_phi;
        // (θ, φ) order gives the inward normal on a torus; swap the cross.
        return normalize(cross(dp_dphi, dp_dtheta));
    }
    let dz_per_uv = g_cyl.length;
    let dr_du = dh_du * dr_range / TAU;
    let dr_dv = dh_dv * dr_range / dz_per_uv;
    let dp_dtheta = vec3f(dr_du * cos_t - r * sin_t, dr_du * sin_t + r * cos_t, 0.0);
    let dp_dz     = vec3f(dr_dv * cos_t,             dr_dv * sin_t,             1.0);
    return normalize(cross(dp_dtheta, dp_dz));
}

fn sky_visibility(rc: RadialCoordinates) -> f32 {
    let d_frag = cyl_depth(rc.radius);
    let uv = shadow_uv(rc);
    let texel = 1.0 / vec2f(textureDimensions(g_shadow, 0));
    let off = texel * SHADOW_SAMPLE_SPREAD;
    var sum = 0.0;
    var count = 0.0;
    // Smoothstep PCF over a (2·R+1)² grid. Bigger R + spread = softer shadow
    // with neighbouring vehicle parts merging into one blob.
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

// Horizon-based terrain AO in world-space tangent coordinates.
//
// For each of N tangent directions around the fragment, march along that
// direction and find the maximum elevation angle the actual terrain reaches
// above the fragment's tangent plane. The mean of sin(horizon_angle) over
// directions approximates the fraction of the upper hemisphere blocked.
//
// Each sample is dropped onto the local surface of revolution (via
// cartesian_to_radial + the height map), so the world's own curvature does
// not masquerade as elevation — only height-map deviation produces it.
fn terrain_ao(frag_pos: vec3f, rc: RadialCoordinates) -> f32 {
    let normal = terrain_normal(rc);

    // Tangent basis. The helper picks a vector well off the normal so the
    // cross product is stable everywhere on the surface.
    let helper = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(normal.z) > 0.9);
    let tangent = normalize(cross(normal, helper));
    let bitangent = cross(normal, tangent);

    let dr_range = g_cyl.radius_end - g_cyl.radius_start;
    let dist_steps = array<f32, 4>(0.5, 1.5, 3.5, 7.0);
    let dirs: i32 = 8;
    var sum_sin_horizon = 0.0;
    for (var di = 0; di < dirs; di = di + 1) {
        let angle = f32(di) * (TAU / f32(dirs));
        let dir = tangent * cos(angle) + bitangent * sin(angle);
        var max_tan = 0.0;
        for (var si = 0; si < 4; si = si + 1) {
            let d = dist_steps[si];
            let sample_pos = frag_pos + dir * d;
            let sample_rc = cartesian_to_radial(sample_pos);
            let h_alpha = sample_map(sample_rc).a;
            let ground_r = g_cyl.radius_start + h_alpha * dr_range;
            let outward = (sample_pos - sample_rc.centre) / max(sample_rc.radius, 1e-6);
            let ground_pos = sample_rc.centre + ground_r * outward;
            let elev = dot(ground_pos - frag_pos, normal);
            max_tan = max(max_tan, elev / d);
        }
        // tan → sin so the contribution caps near 1 for steep horizons.
        let sin_h = max_tan / sqrt(1.0 + max_tan * max_tan);
        sum_sin_horizon = sum_sin_horizon + sin_h;
    }
    let avg_sin = sum_sin_horizon / f32(dirs);
    // Amplify and floor: 1.5× makes typical 5-10° horizons noticeable; the
    // 0.3 floor keeps the deepest basins from going pure black.
    return clamp(1.0 - avg_sin * 1.5, 0.3, 1.0);
}

fn shade_terrain(frag_pos: vec3f, rc: RadialCoordinates, albedo: vec3f) -> vec3f {
    let normal = terrain_normal(rc);
    let env = sample_environment(normal);
    let light = mix(vec3f(1.0), env, ENV_TINT);
    let vis = sky_visibility(rc);
    let ao = terrain_ao(frag_pos, rc);
    return albedo * light * vis * ao;
}

// ===== Sky background =====
// A fullscreen triangle sampling the environment panorama along the view
// ray. Drawn before the terrain with depth writes off, so anything the
// terrain does not cover keeps the sky.

struct SkyOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) ray_dir: vec3f,
}

@vertex
fn vs_sky(@builtin(vertex_index) vi: u32) -> SkyOutput {
    var so: SkyOutput;
    let ic = vec2<u32>(vi & 1u, (vi & 2u) >> 1u);
    //Note: camera coordinate system is X-right, Y-down, Z-forward
    let pos = (4.0 * vec2f(ic) - 1.0) * vec2f(1.0, -1.0);
    so.clip_pos = vec4f(pos, 0.0, 1.0);
    let local_dir = vec3f(pos * g_camera.half_plane, 1.0);
    so.ray_dir = qrot(g_camera.rot, local_dir);
    return so;
}

@fragment
fn fs_sky(in: SkyOutput) -> @location(0) vec4f {
    return vec4f(tone(sample_environment(in.ray_dir)), 1.0);
}

// ===== Terrain mesh =====

// Fetched from a plain vertex buffer (see `TerrainVertex` on the Rust side)
// so the pipeline runs on WebGL2-class devices with no storage buffers.
struct TerrainVertex {
    position: vec3f,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
}

@vertex
fn vs_terrain_mesh(v: TerrainVertex) -> VertexOutput {
    let p_camera = qrot(qinv(g_camera.rot), v.position - g_camera.pos);
    var vo: VertexOutput;
    let depth = (p_camera.z - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
    vo.clip_pos = vec4f(p_camera.xy / g_camera.half_plane, depth * p_camera.z, p_camera.z);
    vo.world_pos = v.position;
    return vo;
}

@fragment
fn fs_terrain_mesh(in: VertexOutput) -> @location(0) vec4f {
    let rc = cartesian_to_radial(in.world_pos);
    let albedo = sample_map(rc).xyz;
    return vec4f(tone(shade_terrain(in.world_pos, rc, albedo)), 1.0);
}
