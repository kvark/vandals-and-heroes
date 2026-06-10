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

struct RayParams {
    march_count: u32,
    march_closest_power: f32,
    bisect_count: u32,
}
var<uniform> g_ray_params: RayParams;

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

// World point → heightmap coordinates. Two cases:
//
// * **Cylinder** (the default): `radius` = distance from the Z axis,
//   `centre` = projection of the point onto the Z axis, `uv` =
//   (θ/2π, z/length + 0.5).
//
// * **Sphere** (`g_cyl.is_sphere != 0`): `radius` = distance from the origin,
//   `centre` = origin, `uv` = Lambert equal-area cylindrical projection:
//   u = θ/2π, v = (sin φ + 1)/2 — each texel covers the same surface area
//   regardless of latitude (poles compress only in shape, not in area).
//
// `outward = (pos - centre) / radius` gives the local "up" direction in either
// world; that's the direction the terrain elevation grows along.
struct RadialCoordinates {
    alpha: f32,    // longitude θ (radians)
    radius: f32,   // distance from local centre
    depth: f32,    // axial coord (cylinder z) or sin(latitude) on sphere
    centre: vec3f, // local "axis" point — projection of pos onto Z for cyl,
                   //   origin for sphere
}
fn cartesian_to_radial(p: vec3f) -> RadialCoordinates {
    var rc: RadialCoordinates;
    if (g_cyl.is_sphere != 0u) {
        let r = max(length(p), 1e-6);
        rc.alpha = atan2(p.y, p.x);
        rc.radius = r;
        rc.depth = clamp(p.z / r, -1.0, 1.0); // sin φ
        rc.centre = vec3f(0.0);
    } else {
        rc.alpha = atan2(p.y, p.x);
        rc.radius = length(p.xy);
        rc.depth = p.z;
        rc.centre = vec3f(0.0, 0.0, p.z);
    }
    return rc;
}

fn terrain_uv(rc: RadialCoordinates) -> vec2f {
    if (g_cyl.is_sphere != 0u) {
        // Lambert equal-area cylindrical: u = θ/2π, v = (sin φ + 1) / 2.
        return vec2f(rc.alpha / TAU, (rc.depth + 1.0) * 0.5);
    }
    return vec2f(rc.alpha / TAU, rc.depth / g_cyl.length + 0.5);
}

fn sample_map(rc: RadialCoordinates) -> vec4f {
    return textureSampleLevel(g_terrain, g_terrain_sampler, terrain_uv(rc), 0.0);
}

// Bilinear gradient → world-space outward surface normal. On the cylinder the
// surface is p(θ, z) = (r·cos θ, r·sin θ, z); on the sphere it's
// p(θ, φ) = r · (cos φ·cos θ, cos φ·sin θ, sin φ) — the cross product of the
// two tangents is the outward normal in either case.
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
    if (g_cyl.is_sphere != 0u) {
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
    let dz_per_uv = g_cyl.length;
    let dr_du = dh_du * dr_range / TAU;
    let dr_dv = dh_dv * dr_range / dz_per_uv;
    let dp_dtheta = vec3f(dr_du * cos_t - r * sin_t, dr_du * sin_t + r * cos_t, 0.0);
    let dp_dz     = vec3f(dr_dv * cos_t,             dr_dv * sin_t,             1.0);
    return normalize(cross(dp_dtheta, dp_dz));
}

fn shadow_uv(rc: RadialCoordinates) -> vec2f {
    // Shadow-map convention: u = theta/(2π) + 0.5 (so the model rasterizer's
    // clip_x = theta/π maps to the same u after the viewport transform). The
    // v axis matches the heightmap's v: cylinder z/length + 0.5, or sphere
    // Lambert (sin φ + 1)/2 (rc.depth already holds sin φ in sphere mode).
    if (g_cyl.is_sphere != 0u) {
        return vec2f(rc.alpha / TAU + 0.5, (rc.depth + 1.0) * 0.5);
    }
    return vec2f(rc.alpha / TAU + 0.5, rc.depth / g_cyl.length + 0.5);
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

// Horizon-based terrain AO, computed in proper world-space tangent coordinates
// and matched to the cylinder geometry.
//
// For each of N tangent directions around the fragment, we march along that
// direction and find the maximum elevation angle the actual terrain reaches
// above the fragment's tangent plane. The mean of sin(horizon_angle) over
// directions approximates the fraction of the upper hemisphere blocked, which
// we use as an AO darkening factor.
//
// Critical detail for cylinder world: a flat sample-distance d on the
// fragment's flat tangent plane lands at a world point that, on a smooth
// cylinder of radius r, drifts BELOW the tangent plane by d²/(2r) due to the
// surface curving away. We compute elevation against the actual world ground
// position (frag_pos to ground_pos along the normal), which naturally cancels
// that curvature for a smooth cylinder — only the heightmap deviation produces
// elevation. Sampling distances must stay below sqrt(2·r·Δr_max) to detect
// occluders at the heightmap's full alpha range (~12 m for Fostral).
//
// Normal-sensitive: a vertical wall projects horizon contributions onto a
// vertical tangent plane, so the half of sample directions facing into the
// rock contribute and the half facing into open air don't. A peak finds every
// neighbour below its tangent plane → no occlusion. A basin floor sees its
// rim walls in every direction → high occlusion.
fn terrain_ao(rc: RadialCoordinates) -> f32 {
    let normal = terrain_normal(rc);
    let cos_t = cos(rc.alpha);
    let sin_t = sin(rc.alpha);
    var frag_pos: vec3f;
    if (g_cyl.is_sphere != 0u) {
        let c = sqrt(max(1.0 - rc.depth * rc.depth, 0.0));
        frag_pos = rc.radius * vec3f(c * cos_t, c * sin_t, rc.depth);
    } else {
        frag_pos = vec3f(rc.radius * cos_t, rc.radius * sin_t, rc.depth);
    }

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
        // March outward in this tangent direction, tracking the highest
        // elevation angle (tan, then sin) any sampled occluder reaches. Each
        // sample is dropped onto the local surface of revolution (cylinder or
        // sphere) so the curvature itself doesn't masquerade as elevation.
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

fn intersect_ray_with_map_radius(dir: vec2f, radius: f32) -> vec2f {
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

// Ray vs. concentric sphere of `radius` centred at the origin. Returns
// (t_near, t_far) with t_near ≤ t_far. (0, 0) when the ray misses.
fn intersect_ray_with_sphere(dir: vec3f, radius: f32) -> vec2f {
    let a = dot(dir, dir);
    let b = 2.0 * dot(dir, g_camera.pos.xyz);
    let c = dot(g_camera.pos.xyz, g_camera.pos.xyz) - radius * radius;
    let d = b * b - 4.0 * a * c;
    if (d < 0.0) {
        return vec2f(0.0);
    }
    let signs = select(vec2f(1.0, -1.0), vec2f(-1.0, 1.0), a > 0.0);
    return (signs * sqrt(d) - b) / (2.0 * a);
}

fn compute_ray_distance(dir: vec3f) -> vec2f {
    var result = vec2f(g_camera.clip_near, g_camera.clip_far);
    if (g_cyl.is_sphere != 0u) {
        let t_end = intersect_ray_with_sphere(dir, g_cyl.radius_end);
        result.x = max(result.x, t_end.x);
        result.y = min(result.y, t_end.y);
        if (result.x >= result.y) {
            return vec2f(0.0);
        }
        let t_start = intersect_ray_with_sphere(dir, g_cyl.radius_start);
        if (t_start.y > t_start.x) {
            result.y = min(result.y, t_start.x);
        }
        return result;
    }
    let limit = (g_cyl.length * select(-0.5, 0.5, dir.z > 0.0) - g_camera.pos.z) / dir.z;
    result.y = min(result.y, limit);
    if (result.x >= result.y) {
        return vec2f(0.0);
    }
    let t_end = intersect_ray_with_map_radius(dir.xy, g_cyl.radius_end);
    result.x = max(result.x, t_end.x);
    result.y = min(result.y, t_end.y);
    if (result.x >= result.y) {
        return vec2f(0.0);
    }
    let t_start = intersect_ray_with_map_radius(dir.xy, g_cyl.radius_start);
    if (t_start.y > t_start.x) {
        result.y = min(result.y, t_start.x);
    }
    return result;
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4f,
    @location(0) ray_dir: vec3f,
}

@vertex
fn vs_terrain_draw(@builtin(vertex_index) vi: u32) -> VertexOutput {
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

fn shade_terrain(rc: RadialCoordinates, albedo: vec3f) -> vec3f {
    let normal = terrain_normal(rc);
    let env = sample_environment(normal);
    let light = mix(vec3f(1.0), env, ENV_TINT);
    let vis = sky_visibility(rc);
    let ao = terrain_ao(rc);
    return albedo * light * vis * ao;
}

fn ray_bisect(direction: vec3f, start: f32, end: f32) -> FragmentOutput {
    var a = start;
    var b = end;
    var final_rc: RadialCoordinates;
    var final_albedo = vec3f(0.0);
    var hit = false;
    for (var i = 0u; i < g_ray_params.bisect_count; i += 1u) {
        let c = 0.5 * (a + b);
        var position = g_camera.pos.xyz + c * direction;
        let rc = cartesian_to_radial(position);
        let texel = sample_map(rc);
        let ground_radius = mix(g_cyl.radius_start, g_cyl.radius_end, texel.a);
        if (rc.radius <= ground_radius) {
            final_rc = rc;
            final_albedo = texel.rgb;
            hit = true;
            b = c;
        } else {
            a = c;
        }
    }

    let normalized_depth = (0.5 * (a+b) - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
    var color = vec4f(0.0);
    if (hit) {
        // Snap the radial coord onto the heightmap surface so the shadow-map depth comparison lines up.
        let surface_alpha = textureSampleLevel(g_terrain, g_terrain_sampler, terrain_uv(final_rc), 0.0).a;
        final_rc.radius = mix(g_cyl.radius_start, g_cyl.radius_end, surface_alpha);
        color = vec4f(shade_terrain(final_rc, final_albedo), 1.0);
    }
    return FragmentOutput(color, normalized_depth);
}

@fragment
fn fs_terrain_ray_march(in: VertexOutput) -> FragmentOutput {
    let distances = compute_ray_distance(in.ray_dir);
    var prev_distance = distances.x;
    if (distances.x < distances.y) {
        for (var i = 0u; i < g_ray_params.march_count; i += 1u) {
            let distance_ratio = pow(f32(i + 1u) / f32(g_ray_params.march_count), g_ray_params.march_closest_power);
            let distance = mix(distances.x, distances.y, distance_ratio);
            var position = g_camera.pos.xyz + distance * in.ray_dir;
            let rc = cartesian_to_radial(position);
            let texel = sample_map(rc);
            let ground_radius = mix(g_cyl.radius_start, g_cyl.radius_end, texel.a);
            if (rc.radius <= ground_radius) {
                return ray_bisect(in.ray_dir, prev_distance, distance);
            }
            prev_distance = distance;
        }
        // March completed without finding terrain. The end of the march was
        // already clamped to whichever of (clip_far, ±L/2, outer-cylinder
        // exit, inner-cylinder entry) came first, so the end position is
        // somewhere inside the heightmap shell. Render terrain at the end
        // unconditionally — without this, very deep valleys (α → 0) or any
        // ray the discrete march sample density miss let the sky bleed
        // through, which the player sees as transparent flickering ground.
        let end_pos = g_camera.pos.xyz + distances.y * in.ray_dir;
        let end_rc = cartesian_to_radial(end_pos);
        // The camera lives inside the heightmap shell with gravity pulling
        // it toward the Z axis. A ray either exits *outward* through the
        // outer cylinder (radius_end, the "sky" boundary) or *inward*
        // through the inner cylinder (radius_start, into the empty hollow
        // below the deepest valley). Outward exits are sky — let them fall
        // through to the env map. Inward exits are deep-valley rays the
        // discrete march sample density missed: those should show terrain
        // at the valley floor instead of bleeding the sky through.
        if (end_rc.radius <= g_cyl.radius_start + 0.05) {
            let texel = sample_map(end_rc);
            var floor_rc = end_rc;
            floor_rc.radius = g_cyl.radius_start;
            let normalized_depth =
                (distances.y - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
            let color = vec4f(shade_terrain(floor_rc, texel.rgb), 1.0);
            return FragmentOutput(color, normalized_depth);
        }
    }

    // miss → sky
    return FragmentOutput(vec4f(sample_environment(in.ray_dir), 1.0), 1.0);
}
