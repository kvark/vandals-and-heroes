// Shared constants, qrot/qinv, CylParams + g_cyl + cyl_depth live in common.wgsl
// and are prepended at shader-load time. The voxel-grid layout + helper
// `voxel_bit_addr` come from shaders/voxel.wgsl, prepended for the voxel
// raycast pipeline only.

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

// Voxel occupancy buffer + LOD metadata. Bound only by the voxel-raycast
// pipeline; the legacy ray-march pipeline leaves this binding unused.
struct VoxelData {
    lod_count: vec4u,
    lods: array<VoxelLod, VOXEL_MAX_LODS>,
    occupancy: array<u32>,
}
var<storage, read> g_voxels: VoxelData;

fn check_occupancy(coords: vec3i, lod: VoxelLod) -> bool {
    // θ wraps in voxel_bit_addr. Out-of-range z/r are reported as empty so the
    // outer loop can step on through without a false hit.
    if (coords.y < 0 || coords.y >= lod.dim.y) { return false; }
    if (coords.z < 0 || coords.z >= lod.dim.z) { return false; }
    let addr = voxel_bit_addr(coords, lod);
    return (g_voxels.occupancy[addr.word] & addr.mask) != 0u;
}

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
    // Clip to the inner cylinder ONLY when the ray actually crosses it
    // forward in time. For a camera just inside the shell looking outward,
    // both t_start roots are negative (the inner cylinder is entirely
    // behind us), and the old check `t_start.y > t_start.x` alone fired
    // anyway and pinned result.y to a large negative t — which made the
    // caller think there was no ray at all, and the whole bottom-up case
    // came back as solid env-sky. Requiring t_start.x > 0 here means we
    // only clip when the ray genuinely re-enters the inner-cylinder
    // hollow (camera outside r_start, looking toward and past the axis).
    if (t_start.x > 0.0 && t_start.y > t_start.x) {
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

// ==========================================================================
// HiZ voxel raycast in cylindrical coordinates.
//
// The acceleration structure is in CURVED space — voxels are (θ-cell, z-cell,
// r-cell) wedges, baked from the heightmap by shaders/voxel-bake.wgsl. The
// ray walks straight in 3D Cartesian; each step computes the analytical
// time to exit the current wedge across one of three face types:
//   * θ half-plane through the Z axis: x sin θ_b − y cos θ_b = 0,
//     linear in t with denom = D.x sin θ_b − D.y cos θ_b.
//   * z plane at constant z: linear in t.
//   * r cylinder |P.xy| = r_b: quadratic in t.
// On exit the new world position is reprojected to (i_θ, i_z, i_r) — robust
// against floating-point boundary cases at the cost of a few trig ops per
// step.
//
// Single-layer terrain leaf cell ⇒ fall through to ray_bisect against the
// heightmap, same as the legacy ray-march fallback.
// ==========================================================================

fn voxel_cell_size_theta(lod: VoxelLod) -> f32 {
    return TAU / f32(lod.dim.x);
}
fn voxel_cell_size_z(lod: VoxelLod) -> f32 {
    return g_cyl.length / f32(lod.dim.y);
}
fn voxel_cell_size_r(lod: VoxelLod) -> f32 {
    return (g_cyl.radius_end - g_cyl.radius_start) / f32(lod.dim.z);
}

// World position → integer (i_θ, i_z, i_r) cell index. θ is in [-π, π] from
// atan2 and we shift to [0, 2π) before discretising; z and r get the same
// "subtract origin, divide cell size" treatment. Returns out-of-range
// indices when pos is outside the heightmap shell — callers check.
fn voxel_world_to_cell(pos: vec3f, lod: VoxelLod) -> vec3i {
    let theta = atan2(pos.y, pos.x);  // [-π, π]
    let theta_wrapped = theta + select(0.0, TAU, theta < 0.0);
    let r = length(pos.xy);
    let dt = voxel_cell_size_theta(lod);
    let dz = voxel_cell_size_z(lod);
    let dr = voxel_cell_size_r(lod);
    let i_theta = i32(floor(theta_wrapped / dt));
    // The camera in cylinder mode starts at r == r_end exactly, which would
    // land i_r = dim.z (just past the last valid bin) without a clamp; the
    // out-of-bounds short-circuit then breaks the DDA before it ever steps.
    // Clamp at both ends — the main loop's `exit.t > t_end` (computed from
    // `compute_ray_distance`'s analytic shell intersection) handles genuine
    // exits through r_end, r_start, ±length/2.
    let i_z_raw = i32(floor((pos.z + 0.5 * g_cyl.length) / dz));
    let i_r_raw = i32(floor((r - g_cyl.radius_start) / dr));
    let i_z = clamp(i_z_raw, 0, lod.dim.y - 1);
    let i_r = clamp(i_r_raw, 0, lod.dim.z - 1);
    return vec3i(i_theta, i_z, i_r);
}

struct CellExit {
    t: f32,
    valid: bool,
}

// Smallest t > t_min at which the ray exits the current voxel cell. The
// cell's faces are: two θ half-planes, two z planes, two r cylinders. We
// solve each independently and take the minimum.
fn voxel_cell_exit_t(
    base: vec3f, dir: vec3f, t_min: f32,
    coords: vec3i, lod: VoxelLod,
) -> CellExit {
    // For numerical safety we step slightly past the boundary so the next
    // world_to_cell call lands cleanly in the neighbour cell.
    let EPS_T: f32 = 1e-5;

    // ---- θ faces ----
    let dt_theta = voxel_cell_size_theta(lod);
    let theta_low = f32(coords.x) * dt_theta;
    let theta_high = theta_low + dt_theta;
    // Each θ-boundary is a half-plane through the z-axis: x sin θ_b − y cos θ_b = 0
    // restricted to (x cos θ_b + y sin θ_b) > 0. The full-plane equation also
    // crosses at θ_b + π — that's an ENTIRELY different cell, not our exit,
    // so we discard solutions where the radial test fails.
    var t_theta: f32 = 1e30;
    for (var sgn = 0u; sgn < 2u; sgn = sgn + 1u) {
        let theta_b = select(theta_low, theta_high, sgn == 1u);
        let s = sin(theta_b);
        let c = cos(theta_b);
        let denom = dir.x * s - dir.y * c;
        if (abs(denom) < 1e-9) { continue; }
        let t = (base.y * c - base.x * s) / denom;
        if (t <= t_min + EPS_T || t >= t_theta) { continue; }
        // Verify crossing is on the right side of the z-axis (θ = θ_b, not θ_b + π).
        let px = base.x + t * dir.x;
        let py = base.y + t * dir.y;
        if (px * c + py * s <= 0.0) { continue; }
        t_theta = t;
    }

    // ---- z faces ----
    var t_z: f32 = 1e30;
    if (abs(dir.z) > 1e-9) {
        let dt_z = voxel_cell_size_z(lod);
        let z_low = f32(coords.y) * dt_z - 0.5 * g_cyl.length;
        let z_high = z_low + dt_z;
        let z_target = select(z_low, z_high, dir.z > 0.0);
        let t = (z_target - base.z) / dir.z;
        if (t > t_min + EPS_T) { t_z = t; }
    }

    // ---- r faces (quadratic) ----
    var t_r: f32 = 1e30;
    let a = dir.x * dir.x + dir.y * dir.y;
    if (a > 1e-12) {
        let dt_r = voxel_cell_size_r(lod);
        let r_low = g_cyl.radius_start + f32(coords.z) * dt_r;
        let r_high = r_low + dt_r;
        let b = 2.0 * (base.x * dir.x + base.y * dir.y);
        for (var sel = 0u; sel < 2u; sel = sel + 1u) {
            let r_b = select(r_low, r_high, sel == 1u);
            let c_term = base.x * base.x + base.y * base.y - r_b * r_b;
            let disc = b * b - 4.0 * a * c_term;
            if (disc < 0.0) { continue; }
            let s_disc = sqrt(disc);
            let t1 = (-b - s_disc) / (2.0 * a);
            let t2 = (-b + s_disc) / (2.0 * a);
            if (t1 > t_min + EPS_T && t1 < t_r) { t_r = t1; }
            if (t2 > t_min + EPS_T && t2 < t_r) { t_r = t2; }
        }
    }

    let t_exit = min(t_theta, min(t_z, t_r));
    var result: CellExit;
    result.t = t_exit;
    result.valid = t_exit < 1e29;
    return result;
}

@fragment
fn fs_terrain_voxel_cast(in: VertexOutput) -> FragmentOutput {
    let distances = compute_ray_distance(in.ray_dir);
    if (distances.x >= distances.y) {
        return FragmentOutput(vec4f(sample_environment(in.ray_dir), 1.0), 1.0);
    }
    // DDA descent + per-LOD-0-cell K=8 substep walk for transition detection,
    // PLUS cross-cell above_state tracking so inter-cell transitions are
    // caught too. The bisect bracket is always [substep_above, substep_below]
    // (substep positions are smooth fractions of cell boundaries, so adjacent
    // rays' brackets vary smoothly with ray angle) — never [cell_face,
    // cell_face] which would snap discontinuously across cell boundaries
    // and produce a visible cell-mosaic in the shading.
    //
    // The previous "drop the prev-above gate" variant missed grazing rays
    // where no single substep landed inside the surface, letting the DDA
    // exit through the shell and rendering env sky in the middle of solid
    // terrain. Tracking above_state with the cell entry probe catches
    // those: when the entry is above and exit is below, the surface IS in
    // [t_current, exit.t] and the K=8 walk narrows it down.
    let MAX_STEPS: u32 = 2048u;
    var t_current = distances.x;
    var pos = g_camera.pos.xyz + t_current * in.ray_dir;
    var lod_idx = g_voxels.lod_count.x - 1u;
    var coords = voxel_world_to_cell(pos, g_voxels.lods[lod_idx]);
    var above_state: bool;
    {
        let rc = cartesian_to_radial(pos);
        let ground_r = mix(g_cyl.radius_start, g_cyl.radius_end, sample_map(rc).a);
        above_state = rc.radius > ground_r;
    }
    var steps: u32 = 0u;
    loop {
        if (steps >= MAX_STEPS) { break; }
        steps = steps + 1u;
        let lod = g_voxels.lods[lod_idx];
        if (coords.y < 0 || coords.y >= lod.dim.y) { break; }
        if (coords.z < 0 || coords.z >= lod.dim.z) { break; }
        let occ = check_occupancy(coords, lod);
        if (occ && lod_idx > 0u) {
            lod_idx = lod_idx - 1u;
            coords = voxel_world_to_cell(pos, g_voxels.lods[lod_idx]);
            continue;
        }
        let exit = voxel_cell_exit_t(g_camera.pos.xyz, in.ray_dir, t_current, coords, lod);
        if (!exit.valid || exit.t > distances.y) { break; }
        // For empty cells (mip says no terrain in the wedge), ray is above
        // ground throughout — above_state stays true, no work needed. We
        // only do the K-walk in occupied leaf cells.
        if (occ && lod_idx == 0u) {
            const K: u32 = 8u;
            var prev_t = t_current;
            var prev_above = above_state;
            var found = false;
            var hit_a = t_current;
            var hit_b = exit.t;
            var exit_above = above_state;
            for (var i: u32 = 1u; i <= K; i = i + 1u) {
                let t_i = mix(t_current, exit.t, f32(i) / f32(K));
                let p = g_camera.pos.xyz + t_i * in.ray_dir;
                let rc = cartesian_to_radial(p);
                let ground_r = mix(g_cyl.radius_start, g_cyl.radius_end, sample_map(rc).a);
                let above = rc.radius > ground_r;
                // Catch BOTH transition directions. Above→below is the
                // common case (rays from above hitting the front face);
                // below→above is the "bottom-up" case (camera below
                // local ground, looking up at a peak; or a ray that
                // passed through a tall feature and is exiting the back
                // face).
                //
                // The bisect below convergeS only when the bracket is
                // [a = above, b = below] — `rc.radius <= ground_r ⇒ b = c`
                // pulls b toward a, so a must hold the above side. We
                // normalise the bracket here so the same bisect handles
                // both directions.
                if (prev_above != above) {
                    if (prev_above) {
                        hit_a = prev_t;
                        hit_b = t_i;
                    } else {
                        hit_a = t_i;
                        hit_b = prev_t;
                    }
                    found = true;
                    exit_above = above;
                    break;
                }
                prev_above = above;
                prev_t = t_i;
                exit_above = above;
            }
            if (found) {
                var a = hit_a;
                var b = hit_b;
                var final_rc: RadialCoordinates;
                var final_albedo = vec3f(0.0);
                for (var i = 0u; i < g_ray_params.bisect_count; i = i + 1u) {
                    let c = 0.5 * (a + b);
                    let p = g_camera.pos.xyz + c * in.ray_dir;
                    let rc = cartesian_to_radial(p);
                    let texel = sample_map(rc);
                    let ground_r = mix(g_cyl.radius_start, g_cyl.radius_end, texel.a);
                    if (rc.radius <= ground_r) {
                        final_rc = rc;
                        final_albedo = texel.rgb;
                        b = c;
                    } else {
                        a = c;
                    }
                }
                let surface_alpha = textureSampleLevel(g_terrain, g_terrain_sampler, terrain_uv(final_rc), 0.0).a;
                final_rc.radius = mix(g_cyl.radius_start, g_cyl.radius_end, surface_alpha);
                let depth = (0.5 * (a + b) - g_camera.clip_near) / (g_camera.clip_far - g_camera.clip_near);
                return FragmentOutput(vec4f(shade_terrain(final_rc, final_albedo), 1.0), depth);
            }
            // No transition. Final above_state is the last substep's above.
            above_state = exit_above;
        }
        let new_t = exit.t + 1e-5;
        pos = g_camera.pos.xyz + new_t * in.ray_dir;
        t_current = new_t;
        coords = voxel_world_to_cell(pos, g_voxels.lods[lod_idx]);
        // Multi-level unzoom: promote to the COARSEST LOD whose cell at
        // this pos is still empty. Single-step unzoom (one LOD per
        // iteration) is far too slow for nearly-horizontal rays through the
        // 8192-cell-tall heightmap shell — it ran the DDA out of its step
        // budget mid-traversal and left sky-coloured voids on far hilltops.
        loop {
            if (lod_idx + 1u >= g_voxels.lod_count.x) { break; }
            let parent = g_voxels.lods[lod_idx + 1u];
            let parent_coords = voxel_world_to_cell(pos, parent);
            if (check_occupancy(parent_coords, parent)) { break; }
            lod_idx = lod_idx + 1u;
            coords = parent_coords;
        }
    }
    return FragmentOutput(vec4f(sample_environment(in.ray_dir), 1.0), 1.0);
}

// HiZ DDA diagnostic fragment shader. Same loop as fs_terrain_voxel_cast,
// but emits step counts / LOD / hit status as colour channels instead of
// shaded terrain. Useful for catching DDA regressions in the future:
//   R = steps spent / 64
//   G = LOD index at break / lod_count
//   B = exit status: 0=miss, 0.5=exited grid sideways, 1=leaf-occupied hit
@fragment
fn fs_terrain_voxel_debug(in: VertexOutput) -> FragmentOutput {
    let distances = compute_ray_distance(in.ray_dir);
    if (distances.x >= distances.y) {
        return FragmentOutput(vec4f(0.0, 0.0, 0.0, 1.0), 1.0);
    }

    let MAX_STEPS: u32 = 256u;
    var t_current = distances.x;
    var pos = g_camera.pos.xyz + t_current * in.ray_dir;
    var lod_idx = g_voxels.lod_count.x - 1u;
    var coords = voxel_world_to_cell(pos, g_voxels.lods[lod_idx]);
    var steps: u32 = 0u;
    var hit_status: f32 = 0.0;
    var final_lod: u32 = lod_idx;

    loop {
        if (steps >= MAX_STEPS) { break; }
        steps = steps + 1u;
        let lod = g_voxels.lods[lod_idx];
        if (coords.y < 0 || coords.y >= lod.dim.y || coords.z < 0 || coords.z >= lod.dim.z) {
            hit_status = 0.5;
            break;
        }
        let occ = check_occupancy(coords, lod);
        if (occ && lod_idx > 0u) {
            lod_idx = lod_idx - 1u;
            coords = voxel_world_to_cell(pos, g_voxels.lods[lod_idx]);
            continue;
        }
        let exit = voxel_cell_exit_t(g_camera.pos.xyz, in.ray_dir, t_current, coords, lod);
        if (!exit.valid || exit.t > distances.y) { break; }
        if (occ && lod_idx == 0u) {
            hit_status = 1.0;
            final_lod = lod_idx;
            break;
        }
        let new_t = exit.t + 1e-5;
        pos = g_camera.pos.xyz + new_t * in.ray_dir;
        t_current = new_t;
        coords = voxel_world_to_cell(pos, g_voxels.lods[lod_idx]);
        if (lod_idx + 1u < g_voxels.lod_count.x) {
            let parent = g_voxels.lods[lod_idx + 1u];
            let parent_coords = voxel_world_to_cell(pos, parent);
            if (!check_occupancy(parent_coords, parent)) {
                lod_idx = lod_idx + 1u;
                coords = parent_coords;
            }
        }
        final_lod = lod_idx;
    }

    let r_chan = f32(steps) / 64.0;
    let g_chan = f32(final_lod) / f32(g_voxels.lod_count.x);
    return FragmentOutput(vec4f(r_chan, g_chan, hit_status, 1.0), 1.0);
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
