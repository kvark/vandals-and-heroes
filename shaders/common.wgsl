// Shared declarations prepended into every WGSL shader by Render::load_shader.
// Keep this to declarations that every consumer is happy to inherit: constants,
// pure helpers, and bindings that *every* pipeline includes (g_cyl is the only
// such binding right now — shadow + main pipelines all bind it). Anything used
// by only some pipelines (g_camera, g_environment, g_shadow, …) stays local to
// the shader files that need it, otherwise pipelines that don't bind it will
// fail to validate.

const PI: f32 = 3.1415926;
const TAU: f32 = 6.2831853;
// Sized several R16Float quantisation steps above zero (the step near 1.0
// is ~5e-4, half of the old 0.001 bias) so half-float rounding in the
// shadow target can not read as self-shadowing on any backend.
const SHADOW_BIAS: f32 = 0.004;
// 0.0 = pure white ambient (env map ignored); 1.0 = pure env map. In between
// mixes the two: 0.5 takes half the directional colour from the env map and
// half from neutral white, which avoids the whole scene turning a single tint.
const ENV_TINT: f32 = 0.5;
// Soft-shadow / cheap GI parameters. PCF samples a (2·R+1)² grid of taps at
// `SHADOW_SAMPLE_SPREAD` texels of spacing, and per-tap visibility uses
// `smoothstep` over a depth window of `SHADOW_SOFTNESS`. Result in [0, 1].
const SHADOW_SAMPLE_SPREAD: f32 = 6.0;
const SHADOW_SOFTNESS: f32 = 0.02;
const SHADOW_PCF_RADIUS: i32 = 2;

// World topology — matches `config::WorldShape` discriminants.
const SHAPE_CYLINDER: u32 = 0u;
const SHAPE_SPHERE: u32 = 1u;
const SHAPE_TORUS: u32 = 2u;

fn qrot(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn qinv(q: vec4f) -> vec4f {
    return vec4f(-q.xyz, q.w);
}

struct CylParams {
    radius_start: f32,
    radius_end: f32,
    length: f32,
    // Radial "sun-at-infinity" plane for the shadow map; r in
    // [radius_start, shadow_radius_top] maps to depth in [1, 0]. Chosen wider
    // than radius_end so vehicles sitting above the heightmap peaks fit inside
    // the depth range without clamping.
    shadow_radius_top: f32,
    // SHAPE_CYLINDER / SHAPE_SPHERE / SHAPE_TORUS.
    world_shape: u32,
    // Torus centreline radius (`length / 2π`); unused for other shapes.
    major_radius: f32,
    // Output gamma exponent: 1.0 on sRGB surfaces, 1/2.2 on linear ones
    // (WebGL2 canvases), where fragment shaders must encode manually.
    // Lives here rather than in CameraParams because naga's GLSL-ES output
    // cannot link a uniform block referenced by BOTH stages of a pipeline —
    // g_cyl is fragment-only in every draw pipeline and vertex-only in the
    // shadow pipeline, so it is safe in each.
    gamma: f32,
    // 0 disables the cast-shadow term: the WebGL2 backend's R16F + MIN-blend
    // shadow writes are unreliable (blade gles gap), so the web build skips
    // vehicle shadows rather than randomly black out the car.
    shadows_enabled: u32,
}
var<uniform> g_cyl: CylParams;

fn tone(color: vec3f) -> vec3f {
    return pow(max(color, vec3f(0.0)), vec3f(g_cyl.gamma));
}

fn cyl_depth(r: f32) -> f32 {
    return clamp(
        (g_cyl.shadow_radius_top - r) / (g_cyl.shadow_radius_top - g_cyl.radius_start),
        0.0, 1.0,
    );
}

// World point → height-map coordinates. Three cases:
//
// * **Cylinder**: `radius` = distance from the Z axis, `centre` = projection
//   of the point onto the Z axis, `depth` = z.
// * **Sphere**: `radius` = distance from the origin, `centre` = origin,
//   `depth` = sin(latitude) — Lambert equal-area cylindrical projection.
// * **Torus**: `centre` = nearest point of the centreline circle (radius
//   `major_radius` in the XY plane), `radius` = distance from it, `alpha` =
//   the tube angle around the centreline, `depth` = the *arc length* along
//   the centreline (φ · length / 2π) so the cylinder's `depth / length + 0.5`
//   texture mapping applies verbatim.
//
// `(pos - centre) / radius` gives the local "up" direction in every world;
// that's the direction the terrain elevation grows along.
struct RadialCoordinates {
    alpha: f32,    // angle around the local axis (radians)
    radius: f32,   // distance from local centre
    depth: f32,    // axial coordinate (see above)
    centre: vec3f, // local "axis" point
}

fn cartesian_to_radial(p: vec3f) -> RadialCoordinates {
    var rc: RadialCoordinates;
    if (g_cyl.world_shape == SHAPE_SPHERE) {
        let r = max(length(p), 1e-6);
        rc.alpha = atan2(p.y, p.x);
        rc.radius = r;
        rc.depth = clamp(p.z / r, -1.0, 1.0); // sin φ
        rc.centre = vec3f(0.0);
    } else if (g_cyl.world_shape == SHAPE_TORUS) {
        let rxy = max(length(p.xy), 1e-6);
        let phi = atan2(p.y, p.x);
        rc.centre = vec3f(p.xy * (g_cyl.major_radius / rxy), 0.0);
        let q = p - rc.centre;
        rc.radius = max(length(q), 1e-6);
        rc.alpha = atan2(p.z, rxy - g_cyl.major_radius);
        rc.depth = phi / TAU * g_cyl.length;
    } else {
        rc.alpha = atan2(p.y, p.x);
        rc.radius = length(p.xy);
        rc.depth = p.z;
        rc.centre = vec3f(0.0, 0.0, p.z);
    }
    return rc;
}

fn terrain_uv(rc: RadialCoordinates) -> vec2f {
    if (g_cyl.world_shape == SHAPE_SPHERE) {
        // Lambert equal-area cylindrical: u = θ/2π, v = (sin φ + 1) / 2.
        return vec2f(rc.alpha / TAU, (rc.depth + 1.0) * 0.5);
    }
    // Cylinder and torus share the formula: the torus `depth` is already the
    // arc length along the centreline.
    return vec2f(rc.alpha / TAU, rc.depth / g_cyl.length + 0.5);
}

fn shadow_uv(rc: RadialCoordinates) -> vec2f {
    // Shadow-map convention: u = alpha/(2π) + 0.5 (so the shadow rasterizer's
    // clip_x = alpha/π maps to the same u after the viewport transform). The
    // v axis matches the heightmap's v.
    if (g_cyl.world_shape == SHAPE_SPHERE) {
        return vec2f(rc.alpha / TAU + 0.5, (rc.depth + 1.0) * 0.5);
    }
    return vec2f(rc.alpha / TAU + 0.5, rc.depth / g_cyl.length + 0.5);
}

// Unit direction pointing radially away from the world's gravity anchor —
// "up" as the player experiences it.
fn world_up(p: vec3f) -> vec3f {
    let rc = cartesian_to_radial(p);
    return (p - rc.centre) / max(rc.radius, 1e-6);
}
