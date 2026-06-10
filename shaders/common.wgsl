// Shared declarations prepended into every WGSL shader by Render::load_shader.
// Keep this to declarations that every consumer is happy to inherit: constants,
// pure helpers, and bindings that *every* pipeline includes (g_cyl is the only
// such binding right now — shadow + main pipelines all bind it). Anything used
// by only some pipelines (g_camera, g_environment, g_shadow, …) stays local to
// the shader files that need it, otherwise pipelines that don't bind it will
// fail to validate.

const PI: f32 = 3.1415926;
const TAU: f32 = 6.2831853;
const SHADOW_BIAS: f32 = 0.001;
// 0.0 = pure white ambient (env map ignored); 1.0 = pure env map. In between
// mixes the two: 0.5 takes half the directional colour from the env map and
// half from neutral white, which avoids the whole scene turning a single tint.
const ENV_TINT: f32 = 0.5;
// Soft-shadow / cheap GI parameters. PCF samples a (2·R+1)² grid of taps at
// `SHADOW_SAMPLE_SPREAD` texels of spacing, and per-tap visibility uses
// `smoothstep` over a depth window of `SHADOW_SOFTNESS`. Result in [0, 1].
//
// Sized so the kernel covers the vehicle's u (circumferential) footprint
// densely — Fostral's shadow at 2048×16384 puts ~4.6 cm in a u-texel, so a
// 1 m-wide chassis spans ~22 texels. A 5×5 kernel at spread=6 covers 24 texels
// per dimension, so most samples land on the vehicle while a few drift off the
// edge to give a penumbra.
const SHADOW_SAMPLE_SPREAD: f32 = 6.0;
const SHADOW_SOFTNESS: f32 = 0.02;
const SHADOW_PCF_RADIUS: i32 = 2;

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
    // 0 = cylindrical world; 1 = spherical world. When the sphere mode is
    // active the heightmap wraps via Lambert equal-area cylindrical projection
    // and `length` is ignored (radius_end is the outer sky boundary).
    is_sphere: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
var<uniform> g_cyl: CylParams;

fn cyl_depth(r: f32) -> f32 {
    return clamp(
        (g_cyl.shadow_radius_top - r) / (g_cyl.shadow_radius_top - g_cyl.radius_start),
        0.0, 1.0,
    );
}
