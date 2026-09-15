use blade_graphics as gpu;
use vandals_and_heroes::{
    Camera, GeometryDesc, Loader, LocalLight, LocalLightKind, MaterialDesc, ModelDesc,
    ModelInstance, Physics, PhysicsBodyHandle, Recorder, Render, Terrain, TerrainBody,
    VertexDesc, config, config::WorldShape, tin,
};

use nalgebra::Matrix4;
use std::{f32, path, sync::Arc, thread};
// std::time::Instant panics on wasm32; web-time re-exports std on native.
use web_time as time;

mod assets;
mod snow;

pub struct Wheel {
    pub rigid_body: rapier3d::dynamics::RigidBodyHandle,
    /// Joint owning AngZ (drive) + LinY (suspension). For rear wheels this
    /// connects chassis ↔ wheel directly; for front wheels it connects the
    /// steering knuckle ↔ wheel.
    pub joint: rapier3d::dynamics::ImpulseJointHandle,
    /// `Some` for front wheels: the chassis ↔ knuckle joint owning AngY
    /// (steering). The hierarchy isolates the wheel's spin axis from the
    /// steering rotation so a single AngZ motor can't slew the wheel about
    /// chassis Z while AngY changes.
    pub steering_joint: Option<rapier3d::dynamics::ImpulseJointHandle>,
    /// Steering knuckle body for front wheels; teleported with the chassis on
    /// out-of-bounds respawn so the joint chain stays consistent.
    pub knuckle: Option<rapier3d::dynamics::RigidBodyHandle>,
    /// Chassis-local wheel anchor (from car.ron). Used to place wheels/knuckles
    /// relative to the chassis after a soft respawn.
    pub anchor_local: rapier3d::math::Vec3,
    /// True for the front-axle wheels (those in the chassis -X half, since the
    /// car's forward direction is -X). Steering applies to these wheels only;
    /// rear wheels just drive.
    pub is_steering: bool,
}

pub struct Object {
    /// Chassis: renders every non-wheel geometry at the chassis body's pose.
    pub chassis_instance: ModelInstance,
    /// One renderable per physics wheel, all reusing the same wheel mesh from
    /// the GLB (so vehicles without front-wheel meshes — like OxidizeMonk —
    /// still show every steered wheel).
    /// Render instances for the wheels that need a procedural mesh — typically
    /// the front (steered) wheels for OxidizeMonk, whose GLB already bakes
    /// rear wheels into the chassis mesh. `None` slots mean "skip rendering
    /// here, the GLB will draw it". Index matches `wheels`.
    pub wheel_instances: Vec<Option<ModelInstance>>,
    /// Chassis-local position the wheel template mesh was authored at. We
    /// subtract this when computing each wheel-instance transform so the mesh
    /// — which already contains the GLB anchor in its transform — ends up
    /// centred on the wheel rigid body.
    pub wheel_template_anchor: nalgebra::Vector3<f32>,
    pub rigid_body: rapier3d::dynamics::RigidBodyHandle,
    pub wheels: Vec<Wheel>,
    pub motor_max_velocity: f32,
    /// Chassis-local Y coordinate of the bottom of the AABB. Jump impulses are
    /// applied at this offset so the push-off torque points up through the
    /// vehicle, like real wheels pushing the body upward.
    pub chassis_bottom_y: f32,
    /// Chassis-local Y coordinate of the *top* of the AABB. When the chassis
    /// is upside-down, jump impulses apply here instead of `chassis_bottom_y`
    /// so the push always launches *away* from the surface the cabin is
    /// resting on.
    pub chassis_top_y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Driving,
    Paused,
}

/// Session-local wasteland contact beat: quiet → chase threat → cleared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContactPhase {
    Quiet,
    Threat,
    Cleared,
}

/// Heroes scrap-run mission after contact clears: find the depot beacon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScrapRunPhase {
    Idle,
    Active,
    Complete,
}

#[derive(Default)]
struct DriveInput {
    forward: bool,
    backward: bool,
    steer_left: bool,
    steer_right: bool,
    turbo: bool,
}

/// Player mechos radio callsign — car-as-character identity beat (no UI yet).
const PLAYER_CALLSIGN: &str = "Ash-Runner";

/// Canned wasteland-radio chatter. On startup we log a couple (rotated) via
/// `RUST_LOG=info` — tasteful heroes/vandals flavor, no UI framework.
const RADIO_CHATTER: &[&str] = &[
    "Ash-Runner online — heroes still hold the wasteland road.",
    "Net: vandals on the ridge. Keep rolling, Ash-Runner.",
    "Ash-Runner, dust clear. Heroes vs vandals — scrap run is yours.",
    "Wasteland whisper: Ash-Runner copies. Vandals quiet… for now.",
];

/// Radio lines fired when the wasteland contact beat trips.
const RADIO_CONTACT: &[&str] = &[
    "CONTACT — vandals spotted on your trail, Ash-Runner!",
    "Net spike: hostile mechos closing. Heroes, hold the scrap road!",
];
/// Mid-threat scrap-pressure chatter (rotated).
const RADIO_THREAT_TICK: &[&str] = &[
    "Scrap tick — drive train under fire. Ease off or burn through.",
    "Vandals still on you. Turbo eats armor; keep moving.",
    "Ash-Runner, dust plume behind — they are not giving up.",
];
/// Clear / escape lines when the chase pressure ends.
const RADIO_CLEAR: &[&str] = &[
    "Vandals broke off in the dust. Ash-Runner, road is yours again.",
    "Contact clear. Heroes still hold this stretch of wasteland.",
];

/// Wall-clock Driving time before an automatic "vandals spotted" contact.
const CONTACT_AUTO_SECS: f32 = 12.0;
/// How long chase/threat pressure lasts once contact trips.
const THREAT_DURATION_SECS: f32 = 16.0;
/// Drive-motor derate while under threat (speed-bump / scrap pressure).
const THREAT_DRIVE_FACTOR: f32 = 0.55;
/// Distance from spawn along chassis forward to the hostile mechos marker.
const CONTACT_MARKER_AHEAD: f32 = 55.0;
/// Entering this radius of the marker also trips contact (demo path).
const CONTACT_MARKER_RADIUS: f32 = 18.0;
/// How often scrap-pressure radio ticks fire during threat.
const THREAT_SCRAP_TICK_SECS: f32 = 4.0;
/// Pulsing red hostile marker / alarm light while threatened.
const THREAT_LIGHT_COLOR: [f32; 3] = [1.0, 0.22, 0.08];
const THREAT_LIGHT_RANGE: f32 = 22.0;

/// Radio lines when the scrap-depot beacon mission is assigned.
const RADIO_SCRAP_ASSIGN: &[&str] = &[
    "Ash-Runner — scrap run is live. Heroes stash beacon ahead; haul that scrap home.",
    "Net: cyan depot ping on your HUD. Grab the heroes stash before vandals sniff it.",
];
/// Radio lines when the player reaches the scrap depot.
const RADIO_SCRAP_DONE: &[&str] = &[
    "Scrap delivered. Heroes stash secured — Ash-Runner, motor sings cleaner now.",
    "Depot sealed. Good haul, Ash-Runner. Road belongs to the heroes tonight.",
];

/// Distance ahead of the car (at assign time) to the scrap depot beacon.
const SCRAP_BEACON_AHEAD: f32 = 42.0;
/// Lateral offset so the cyan depot reads apart from the red hostile marker.
const SCRAP_BEACON_LATERAL: f32 = 18.0;
/// Entering this radius completes the scrap run.
const SCRAP_BEACON_RADIUS: f32 = 14.0;
/// Cyan/amber heroes-stash beacon color (distinct from red threat).
const SCRAP_BEACON_COLOR: [f32; 3] = [0.25, 0.95, 0.85];
const SCRAP_BEACON_AMBER: [f32; 3] = [1.0, 0.72, 0.2];
const SCRAP_BEACON_RANGE: f32 = 20.0;
/// Brief motor boost after scrap delivery (reward beat).
const SCRAP_BOOST_FACTOR: f32 = 1.35;
const SCRAP_BOOST_SECS: f32 = 5.0;

/// Radio lines when Ash-Runner rams the chasing vandal proxy.
const RADIO_RAM_HIT: &[&str] = &[
    "Scrap hit! You rammed the vandal — keep punching through!",
    "Ash-Runner body-checks hostile mechos. Nice scrap!",
];
/// Radio when a solid ram early-clears the threat.
const RADIO_RAM_CLEAR: &[&str] = &[
    "Solid ram — vandals scatter! Contact broken by force.",
];
/// Radio when the chasing vandal bumps the player.
const RADIO_BUMPED: &[&str] = &[
    "Bump! Hostile scrap on your bumper — shake it off!",
];

/// Spawn the chase proxy this far behind the chassis when Threat begins.
const VANDAL_CHASE_SPAWN_BEHIND: f32 = 28.0;
/// Pursue speed (m/s) of the kinematic vandal chase marker during Threat.
const VANDAL_CHASE_SPEED: f32 = 14.0;
/// Overlap radius for player→vandal ram and vandal→player bump.
const VANDAL_RAM_RADIUS: f32 = 6.0;
/// Minimum closing speed (player toward vandal, m/s) to count as a scrap-hit ram.
const VANDAL_RAM_CLOSING_MIN: f32 = 8.0;
/// Closing speed that early-clears Threat (solid ram). Documented combat rule.
const VANDAL_RAM_EARLY_CLEAR: f32 = 18.0;
/// Seconds chipped off remaining threat time on a normal (non-clear) ram.
const VANDAL_RAM_TIMER_CHIP_SECS: f32 = 3.0;
/// How long the vandal chase proxy is stunned/slowed after being rammed.
const VANDAL_STUN_SECS: f32 = 2.5;
/// Chase speed multiplier while stunned.
const VANDAL_STUN_SPEED_FACTOR: f32 = 0.15;
/// Cooldown between ram / bump events (avoids radio spam).
const VANDAL_HIT_COOLDOWN_SECS: f32 = 2.5;
/// Brief title-flash duration after a vandal bump (camera-shake proxy).
const VANDAL_BUMP_TITLE_FLASH_SECS: f32 = 0.9;

/// Ash-Runner hull integrity (car-as-character scrap pressure). Full plate.
const HULL_MAX: f32 = 100.0;
/// At or below this: radio warning + extra motor derate; title shows CRITICAL.
const HULL_CRITICAL: f32 = 30.0;
/// Hull chipped when the chasing vandal bumps the player (soft bumps hurt).
const HULL_BUMP_CHIP: f32 = 12.0;
/// Hull chipped on a normal player→vandal ram (aggressor; mostly shrugs it off).
const HULL_RAM_CHIP: f32 = 2.0;
/// Solid early-clear rams chip nothing — you're punching through.
const HULL_SOLID_RAM_CHIP: f32 = 0.0;
/// Extra drive derate while hull is critical (stacks with threat derate).
const HULL_CRITICAL_DRIVE_FACTOR: f32 = 0.7;
/// Drive factor during brief post-breach limp before soft respawn.
const HULL_BREACH_DRIVE_FACTOR: f32 = 0.2;
/// Seconds of limp after hull ≤ 0 before soft respawn + hull restore.
const HULL_BREACH_LIMP_SECS: f32 = 2.0;
/// Scrap depot delivery restores this many hull points (capped at HULL_MAX).
const HULL_SCRAP_REPAIR: f32 = 100.0;
/// Subtle passive regen (pts/s) while Quiet/Cleared, not turbo, not scrap-boosting.
const HULL_REGEN_PER_SEC: f32 = 1.25;

/// Radio when hull first drops to critical.
const RADIO_HULL_CRITICAL: &[&str] = &[
    "Ash-Runner hull critical — scrap plating failing. Ease the pressure!",
    "Net: mechos integrity low. Heroes stash welds can patch you, Ash-Runner.",
];
/// Radio on soft fail (hull breached).
const RADIO_HULL_BREACH: &[&str] = &[
    "Ash-Runner hull breached — limping to last good scrap.",
];
/// Radio when depot delivery patches the hull.
const RADIO_HULL_REPAIR: &[&str] = &[
    "Depot weld complete — Ash-Runner hull patched. Motor breathes again.",
];

/// Radio when scrap-depot delivery forges a spike charge (combat reward).
const RADIO_SPIKE_READY: &[&str] = &[
    "Scrap-forged spike locked — Ash-Runner, press F when vandals close in.",
    "Depot forge complete. One spike charge. Heroes punch, Ash-Runner.",
];
/// Radio when F fires a scrap-forged spike into the chase proxy.
const RADIO_SPIKE_FIRE: &[&str] = &[
    "Spike out! Scrap bolt nails the chase — vandals reel!",
    "Ash-Runner looses the scrap-forged spike. Hostile mechos stunned hard!",
];
/// Radio when F is pressed with a charge but no chase in range.
const RADIO_SPIKE_MISS: &[&str] = &[
    "No target — spike stays hot. Wait for a chase in range.",
    "Ash-Runner, no vandal in spike range. Charge held.",
];

/// Max scrap-forged spike charges (clarity: one ready bolt).
const SPIKE_MAX_CHARGES: u8 = 1;
/// Fire range: chase proxy must be active and within this distance (m).
const SPIKE_RANGE: f32 = 20.0;
/// Stun duration from a scrap-forged spike (longer than ram stun).
const SPIKE_STUN_SECS: f32 = 5.0;
/// Knock chase proxy this far backward along the escape (player-forward) vector.
const SPIKE_KNOCKBACK: f32 = 16.0;
/// Seconds chipped off remaining threat time on a successful spike.
const SPIKE_TIMER_CHIP_SECS: f32 = 5.0;
/// If remaining threat time is at or below this after the chip, early-clear.
const SPIKE_EARLY_CLEAR_REMAINING: f32 = 5.0;

/// Multiplier applied to wheel target velocity while Left Shift is held.
const TURBO_FACTOR: f32 = 2.5;
/// Velocity for a tap-jump (Space pressed and immediately released). Sized
/// to clear a low obstacle without much drama.
const JUMP_MIN_VELOCITY: f32 = 4.0;
/// Velocity for a fully-charged jump (Space held for [`JUMP_MAX_CHARGE`]).
/// At max gravity (12 m/s²) this clears ~8 m; on lighter worlds proportionally
/// higher. Choose enough headroom that the player feels charge pays off.
const JUMP_MAX_VELOCITY: f32 = 14.0;
/// How long the player has to hold Space to reach `JUMP_MAX_VELOCITY`. After
/// this time the jump auto-fires so a held button doesn't lock the chassis.
const JUMP_MAX_CHARGE: time::Duration = time::Duration::from_millis(800);
/// How far the chase camera sits behind the car along its horizontal forward.
const FOLLOW_DIST: f32 = 5.0;
/// Radial-outward chase height. Lower than `FOLLOW_DIST` so the view sits
/// closer to the road instead of a high 45° look-down.
const FOLLOW_HEIGHT: f32 = 3.2;
/// Exponential rate at which the camera catches up to the computed follow
/// pose (per second). Higher = stiffer / more responsive; lower = floatier.
/// 8.0 closes ~99% of the gap in 0.5 s — visibly tracks the car without
/// snapping behind it on every sharp turn.
const CAMERA_FOLLOW_RATE: f32 = 8.0;

// --- Local lights (tune here) ---
/// Chassis-local headlight mount: slightly ahead of the front axle, above the
/// hubs, offset left/right. Forward is chassis -X.
const HEADLIGHT_LOCAL: [nalgebra::Vector3<f32>; 2] = [
    nalgebra::Vector3::new(-0.55, 0.05, 0.18),
    nalgebra::Vector3::new(-0.55, 0.05, -0.18),
];
const HEADLIGHT_COLOR: [f32; 3] = [1.0, 0.95, 0.85];
const HEADLIGHT_INTENSITY: f32 = 22.0;
const HEADLIGHT_RANGE: f32 = 28.0;
const HEADLIGHT_INNER: f32 = 0.22; // ~12.5°
const HEADLIGHT_OUTER: f32 = 0.55; // ~31.5°
const HEADLIGHT_FALLOFF: f32 = 1.4;
/// Soft world fills that travel with the car so the torus tube reads better.
const FILL_INTENSITY: f32 = 12.0;
const FILL_RANGE: f32 = 40.0;
/// Damping factor applied to wheel motors when no drive command is active. High
/// enough that the motor brakes any wheel rotation toward zero, so the static
/// wheel-ground friction holds the chassis still on slopes.
const IDLE_BRAKE_FACTOR: f32 = 50.0;
/// How far past the terrain radial range the chassis may travel before we
/// treat it as out-of-bounds. Sized above a charged jump (~a few metres) so
/// airborne play stays legal, but tight enough that falling through the torus
/// hole or flying into the skybox recovers quickly.
const OOB_RADIAL_OUTER_MARGIN: f32 = 20.0;
/// How far inside `radius.start` counts as fallen through the tube / into the
/// hollow before respawn.
const OOB_RADIAL_INNER_MARGIN: f32 = 5.0;
/// Extra axial slack past ±length/2 for cylinder worlds (torus wraps; sphere
/// has no axial ends).
const OOB_AXIAL_MARGIN: f32 = 30.0;

/// Maximum front-wheel steering angle in radians (~45°). Real cars top out
/// at 30–35° but this is a small buggy on tight cylindrical maps — the
/// extra range gives the chassis enough cross-track force to turn briskly
/// at modest speeds, which is what makes the controls feel responsive.
const MAX_STEER_ANGLE: f32 = std::f32::consts::FRAC_PI_4;
/// Steering motor stiffness. With wheel inertia ~0.003 kg·m² and the
/// damping below, the wheel reaches the target angle in about 100 ms.
const STEER_STIFFNESS: f32 = 200.0;
/// Steering motor damping. Already ~12× critical damping at the chosen
/// stiffness (critical ≈ 2·√(k·I) ≈ 1.6), so there's no wheel oscillation —
/// the straight-line wobble you saw came from elsewhere (suspension).
const STEER_DAMPING: f32 = 20.0;
/// Cap on the steering motor's force (N·m). Sized above the static-friction
/// torque the steered wheels see against terrain so the motor can actually
/// rotate them to the target.
const STEER_MAX_FORCE: f32 = 50.0;
/// Half-width of the procedural wheel mesh (so the visible cylinder is 2·
/// this wide along the axle). Sized to match the GLB-baked rear wheels:
/// inspecting body.glb's Wheel.001 primitive gives a per-wheel z half-
/// extent of 0.08 m (full width 0.16 m). Matching this here makes the
/// procedural front wheels the same thickness as the rear pair the
/// player sees on the model.
const WHEEL_HALF_WIDTH: f32 = 0.08;
/// Scale applied to the procedural wheel mesh radius (which otherwise
/// equals the physics collider radius from car.ron). 1.0 = render the
/// mesh at the same radius the physics uses; the GLB-baked rear wheels
/// are ~0.175 m across so a 0.15 m visible front wheel reads as roughly
/// the right size for the chassis.
const WHEEL_MESH_RADIUS_SCALE: f32 = 1.0;
/// Suspension spring stiffness (N/m). Higher → less body roll during cornering
/// and less bounce on terrain. Sized to give ~0.01 m static compression under
/// the chassis weight.
const SUSPENSION_STIFFNESS: f32 = 300.0;
/// Suspension damping coefficient (N·s/m). Critical for chassis mass ~1.67 kg
/// is `2·√(stiffness·m) ≈ 45`; the old 30 gave ζ ≈ 0.67 (under-damped → the
/// suspension oscillated → chassis pitched → straight-line wobble). At 50
/// the suspension is *slightly* over-damped so bumps absorb without bouncing.
const SUSPENSION_DAMPING: f32 = 50.0;
/// Cap on the suspension spring force per wheel (N). Limits force the spring can
/// transmit during hard impacts.
const SUSPENSION_MAX_FORCE: f32 = 500.0;
/// Chassis-local axis pointing toward the car's visible front. OxidizeMonk's
/// model has its rear wheels in the +X half (see data/cars/OxidizeMonk/car.ron),
/// so the front points along -X. The chase camera and motion convention assume
/// every car follows this same orientation.
fn car_forward_local() -> nalgebra::Vector3<f32> {
    -nalgebra::Vector3::x()
}

/// Build a closed cylinder mesh centred at the origin, with its axle along
/// local +Z, suitable for rendering a wheel attached to a rigid body whose
/// spin axis is local Z. Returns a single-material ModelDesc with a dark
/// tire-coloured material — uploadable through `Loader::load_model`.
fn create_wheel_mesh_desc(radius: f32, half_width: f32) -> ModelDesc {
    use nalgebra::{Point2, Point3, Vector3};
    const SEGMENTS: usize = 16;
    // 4 ring vertices per segment (side+top, side+bot, cap+top, cap+bot) plus
    // 2 cap centers. Caps need their own +Z/-Z normals — sharing the side
    // vertices' radial-outward normals across the cap triangles smooths the
    // edge into a sphere instead of a cylinder.
    let mut vertices: Vec<VertexDesc> = Vec::with_capacity(SEGMENTS * 4 + 2);
    let mut indices: Vec<[u32; 3]> = Vec::with_capacity(SEGMENTS * 4);

    for i in 0..SEGMENTS {
        let angle = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
        let (s, c) = angle.sin_cos();
        let outward = Vector3::new(c, s, 0.0);
        let u = i as f32 / SEGMENTS as f32;
        // Side-wall pair (radial-outward normal, used by the tread quads).
        vertices.push(VertexDesc {
            pos: Point3::new(radius * c, radius * s, half_width),
            tex_coords: Point2::new(u, 0.0),
            normal: outward,
        });
        vertices.push(VertexDesc {
            pos: Point3::new(radius * c, radius * s, -half_width),
            tex_coords: Point2::new(u, 1.0),
            normal: outward,
        });
        // Cap-edge pair (axis-aligned normal, used by the cap triangle fans).
        vertices.push(VertexDesc {
            pos: Point3::new(radius * c, radius * s, half_width),
            tex_coords: Point2::new(0.5 + 0.5 * c, 0.5 + 0.5 * s),
            normal: Vector3::new(0.0, 0.0, 1.0),
        });
        vertices.push(VertexDesc {
            pos: Point3::new(radius * c, radius * s, -half_width),
            tex_coords: Point2::new(0.5 + 0.5 * c, 0.5 + 0.5 * s),
            normal: Vector3::new(0.0, 0.0, -1.0),
        });
    }
    let top_center = vertices.len() as u32;
    vertices.push(VertexDesc {
        pos: Point3::new(0.0, 0.0, half_width),
        tex_coords: Point2::new(0.5, 0.5),
        normal: Vector3::new(0.0, 0.0, 1.0),
    });
    let bot_center = vertices.len() as u32;
    vertices.push(VertexDesc {
        pos: Point3::new(0.0, 0.0, -half_width),
        tex_coords: Point2::new(0.5, 0.5),
        normal: Vector3::new(0.0, 0.0, -1.0),
    });

    for i in 0..SEGMENTS {
        let next = (i + 1) % SEGMENTS;
        let s0 = (i * 4) as u32; // side top
        let s1 = (i * 4 + 1) as u32; // side bot
        let ct0 = (i * 4 + 2) as u32; // cap top
        let cb0 = (i * 4 + 3) as u32; // cap bot
        let s2 = (next * 4) as u32;
        let s3 = (next * 4 + 1) as u32;
        let ct1 = (next * 4 + 2) as u32;
        let cb1 = (next * 4 + 3) as u32;
        indices.push([s0, s1, s2]);
        indices.push([s1, s3, s2]);
        indices.push([top_center, ct1, ct0]);
        indices.push([bot_center, cb0, cb1]);
    }

    let materials = vec![
        // Default sentinel material at index 0 — Loader::read_gltf does the
        // same; load_model copies whatever's in slot 0 verbatim.
        MaterialDesc::default(),
        MaterialDesc {
            name: Some("tire".to_string()),
            base_color_factor: [0.4, 0.4, 0.4, 1.0],
            normal_scale: 0.0,
            transparent: false,
        },
    ];
    let geometry = GeometryDesc {
        name: "procedural_wheel".to_string(),
        vertices,
        indices,
        index_type: Some(gpu::IndexType::U32),
        transform: nalgebra::Matrix4::identity(),
        material_index: 1,
    };
    ModelDesc {
        materials,
        geometries: vec![geometry],
    }
}

pub struct Game {
    // engine stuff. The Choir + worker pool is retained for the next
    // parallel workload — the snow update was found to be smaller than the
    // task-spawn overhead at 2k particles, but the pool is cheap to keep
    // around and avoids re-spawning threads if we add a heavier parallel
    // stage later (e.g. soft-tire contact sampling).
    #[allow(dead_code)]
    choir: Arc<choir::Choir>,
    _choir_workers: Vec<choir::WorkerHandle>,
    render: Render,
    physics: Physics,
    recorder: Option<Recorder>,
    // windowing
    pub window: winit::window::Window,
    window_size: winit::dpi::PhysicalSize<u32>,
    // navigation
    camera: Camera,
    in_camera_drag: bool,
    last_mouse_pos: [i32; 2],
    // game
    mode: Mode,
    input: DriveInput,
    /// Last (throttle, steer, turbo) tuple actually pushed to the motors, used
    /// only to skip a log line when the values are unchanged.
    last_drive_cmd: (f32, f32, f32),
    /// Wall-clock time of the last redraw, used to drive the fixed-timestep
    /// physics accumulator.
    last_redraw_time: time::Instant,
    /// Unspent wall-clock time owed to the physics simulation, in fixed-timestep
    /// units. Accumulator pattern: each redraw adds elapsed real time; we then
    /// step physics 0..N times to drain it.
    physics_accumulator: time::Duration,
    /// When Space was first pressed while grounded. `None` when no jump is
    /// being charged. On release, the held duration scales the impulse
    /// velocity; at [`JUMP_MAX_CHARGE`] the jump auto-fires.
    jump_charge_start: Option<time::Instant>,
    /// False until the first `follow_camera` call snaps directly to the
    /// computed pose. After that the camera lerps each frame.
    camera_initialized: bool,
    terrain_body: TerrainBody,
    terrain: Terrain,
    car: Object,
    /// Initial upright spawn pose; fallback when no last-good pose exists yet.
    spawn_pose: nalgebra::Isometry3<f32>,
    /// Last chassis pose while grounded and in-bounds. Soft OOB respawn target.
    last_good_pose: nalgebra::Isometry3<f32>,
    /// Debug snow: tiny rapier balls falling from the outer shell. Their
    /// landing pattern shows where the *physics* surface sits, exposing any
    /// mismatch with the visual heightmap.
    snow: snow::Snow,
    /// Quiet / Threat / Cleared contact mission hook (vandals spotted).
    contact_phase: ContactPhase,
    /// Accumulated wall-clock time spent in Driving while Quiet (auto-contact).
    contact_quiet_elapsed: time::Duration,
    /// Wall-clock when the current Threat phase began.
    threat_started: Option<time::Instant>,
    /// Last scrap-pressure radio tick during Threat.
    last_scrap_tick: Option<time::Instant>,
    /// World-space hostile mechos marker (ahead of spawn); proximity tripwire.
    contact_marker_pos: nalgebra::Vector3<f32>,
    /// Idle → Active (beacon lit) → Complete scrap-run mission hook.
    scrap_run_phase: ScrapRunPhase,
    /// World-space scrap depot beacon (heroes stash); set when mission assigns.
    scrap_beacon_pos: nalgebra::Vector3<f32>,
    /// Brief motor boost after scrap delivery; `None` when inactive.
    scrap_boost_until: Option<time::Instant>,
    /// When the scrap-run beacon was assigned (pulse clock).
    scrap_run_started: Option<time::Instant>,
    /// True while Threat owns an active kinematic vandal chase proxy.
    vandal_chase_active: bool,
    /// Until this instant the chase proxy is stunned (slow pursue after ram).
    vandal_stun_until: Option<time::Instant>,
    /// Last ram or bump event (shared cooldown).
    last_vandal_hit: Option<time::Instant>,
    /// Brief window-title flash after a vandal bump (`None` = no flash).
    bump_title_until: Option<time::Instant>,
    /// Ash-Runner hull points (0..=HULL_MAX). Car-as-character under scrap pressure.
    hull: f32,
    /// True after we have radio'd the first critical warning this "life".
    hull_critical_warned: bool,
    /// Soft-fail limp window after hull ≤ 0; `None` when not breached.
    hull_breach_until: Option<time::Instant>,
    /// Scrap-forged spike charges (0..=SPIKE_MAX_CHARGES). Depot delivery grants; F spends.
    spike_charges: u8,
}

/// Fixed physics timestep, matching rapier's default `IntegrationParameters::dt`
/// (1/60 s). Don't change one without changing the other.
const PHYSICS_DT: time::Duration = time::Duration::from_nanos(16_666_667);
/// Hard cap on physics catch-up steps per redraw. Prevents the "spiral of death"
/// where a slow frame forces us to simulate longer than a frame, which then
/// makes the next frame even slower, etc. After the cap is hit we drop the
/// excess accumulated time (the world appears to briefly slow rather than freeze).
const MAX_PHYSICS_STEPS_PER_REDRAW: u32 = 6;

pub struct QuitEvent;

impl Game {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        log::info!("Initializing");
        log::info!("Player callsign: {PLAYER_CALLSIGN}");
        log::info!(
            "Hull integrity online: {:.0} pts (critical ≤{:.0}) — car-as-character scrap pressure",
            HULL_MAX,
            HULL_CRITICAL,
        );
        let radio_lines = pick_radio_chatter(2);
        for line in &radio_lines {
            log::info!("[wasteland radio] {line}");
        }
        let radio_subtitle = radio_lines.first().copied().unwrap_or("");

        let config: config::Config = ron::de::from_bytes(&assets::read(path::Path::new(
            "data/config.ron",
        )))
        .expect("Unable to parse the main config");

        let choir = choir::Choir::new();
        // Worker thread count: cap at 4. The dominant parallel consumer is
        // the snow update over a few thousand particles; more workers
        // increase contention without much throughput gain at this scale.
        let worker_count = std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(2);
        #[cfg(not(target_arch = "wasm32"))]
        let _choir_workers: Vec<choir::WorkerHandle> = (0..worker_count)
            .map(|i| choir.add_worker(&format!("choir-{i}")))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let _choir_workers: Vec<choir::WorkerHandle> = {
            // No threads on the web; the pool stays empty.
            let _ = worker_count;
            Vec::new()
        };
        let gpu_context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                ..Default::default()
            })
        }
        .expect("Unable to initialize GPU");

        log::info!("Creating the window");
        #[cfg(not(target_arch = "wasm32"))]
        let window_attributes = winit::window::Window::default_attributes()
            .with_title(format!("Vandals and Heroes — {PLAYER_CALLSIGN} · hull {:.0} · {radio_subtitle}", HULL_MAX))
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 800));
        // On the web, render into the page's existing canvas. Blade's WebGL2
        // backend looks the canvas up by id="blade", so winit must reuse that
        // same element rather than create its own. No fixed inner size: the
        // canvas fills the page via CSS and winit tracks that layout (times
        // the device pixel ratio) with a ResizeObserver, delivering Resized
        // events that reconfigure the surface — any window size works.
        #[cfg(target_arch = "wasm32")]
        let window_attributes = {
            let window_attributes = winit::window::Window::default_attributes()
                .with_title(format!("Vandals and Heroes — {PLAYER_CALLSIGN} · hull {:.0} · {radio_subtitle}", HULL_MAX));
            use wasm_bindgen::JsCast as _;
            use winit::platform::web::WindowAttributesExtWebSys as _;
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("blade"))
                .expect("the page must provide a <canvas id=\"blade\">")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("#blade must be a canvas");
            window_attributes.with_canvas(Some(canvas))
        };
        #[allow(deprecated)] //TODO
        let window = event_loop.create_window(window_attributes).unwrap();
        let window_size = window.inner_size();
        // On the web the canvas may not have been laid out yet and reports
        // zero; create the surface at a placeholder size and let the first
        // Resized event settle it.
        let extent = gpu::Extent {
            width: window_size.width.max(1),
            height: window_size.height.max(1),
            depth: 1,
        };

        let gpu_surface = gpu_context.create_surface(&window).unwrap();
        let mut render = Render::new(gpu_context, gpu_surface, extent);

        let mut loader = render.start_loading();

        let (terrain, terrain_mesh, map_extent, height_alpha) = {
            log::info!("Loading map: {}", config.map);
            let map_path = path::PathBuf::from("data/maps").join(config.map);
            let mut map_config: config::Map = ron::de::from_bytes(&assets::read(
                &map_path.join("map.ron"),
            ))
            .expect("Unable to parse the map config");

            // The map is far denser than the gameplay needs (~3 cm/texel on
            // Fostral). The web build shrinks it 4x: the single-threaded TIN
            // fit, the vertex buffers, and the shadow map all drop well
            // inside browser budgets, at ~12 cm/texel.
            let downsample = if cfg!(target_arch = "wasm32") { 4 } else { 1 };
            let (texture, map_extent, height_alpha) =
                loader.load_png_data(&assets::read(&map_path.join("map.png")), downsample);

            if map_config.length == 0.0 {
                let circumference = 2.0 * f32::consts::PI * map_config.radius.start;
                map_config.length =
                    circumference * (map_extent.height as f32) / (map_extent.width as f32);
                log::info!("Derived map length to be {}", map_config.length);
            }

            let env_texture = config.environment.as_ref().map(|name| {
                let env_path = path::PathBuf::from("data/envs").join(format!("{}.png", name));
                log::info!("Loading environment: {}", env_path.display());
                loader.load_environment_data(&assets::read(&env_path))
            });

            // Triangulate the height map once; the renderer draws these
            // chunks and the physics collides with the very same triangles.
            let mesh = tin::build(
                &height_alpha,
                map_extent.width,
                map_extent.height,
                &map_config,
                config.terrain_quality,
            );
            let chunks = loader.load_terrain_mesh(&mesh);

            (
                Terrain {
                    config: map_config,
                    texture,
                    env_texture,
                    chunks,
                },
                mesh,
                map_extent,
                height_alpha,
            )
        };
        // Cylinder/torus spawns keep the historical "just below the sky"
        // height; the sphere samples the heightmap at the spawn (θ, v) and
        // lands ~1 m above the actual surface so the chassis isn't dropped in
        // from radius_end (where it would fall ~half the world's radial range).
        let spawn_radius = if terrain.config.shape == WorldShape::Sphere {
            let sample_uv = |u: f32, v: f32| -> f32 {
                let ux = ((u * map_extent.width as f32) as u32).min(map_extent.width - 1);
                let vy = ((v * map_extent.height as f32) as u32).min(map_extent.height - 1);
                let idx = vy as usize * map_extent.width as usize + ux as usize;
                height_alpha[idx] as f32 / 255.0
            };
            // Spawn point in (u, v): u = 0.25 corresponds to longitude π/2
            // (the +Y axis), v = 0.5 is the equator (sin φ = 0).
            let spawn_alpha = sample_uv(0.25, 0.5);
            let dr_range = terrain.config.radius.end - terrain.config.radius.start;
            let ground_r = terrain.config.radius.start + spawn_alpha * dr_range;
            (ground_r + 1.0).min(terrain.config.radius.end - 0.1)
        } else {
            terrain.config.radius.end - 0.5
        };
        let mut physics = Physics::default();
        let terrain_body = physics.create_terrain_mesh(&terrain.config, &terrain_mesh);
        drop(terrain_mesh);
        drop(height_alpha);

        // Axial spawn offset: z on the cylinder, centreline arc length on the
        // torus (both 10% into the map so the seam isn't underfoot). The
        // sphere spawns on the equator.
        let spawn_axial = match terrain.config.shape {
            WorldShape::Sphere => 0.0,
            WorldShape::Cylinder | WorldShape::Torus => 0.1 * terrain.config.length,
        };
        let spawn_pose = Self::spawn_pose(&terrain.config, spawn_radius, spawn_axial);
        let car = Self::load_car(&mut loader, &mut physics, &config.car, spawn_pose);

        // Debug snow density: one particle per `config.snow_area_per_particle_m2`
        // m² of world surface. Same visual density across worlds with
        // different scales; tune in data/config.ron. Built before
        // `loader.finish` so the procedural mesh upload rides along the
        // same submission as the car + terrain textures.
        let snow = snow::Snow::new(
            &mut loader,
            &mut physics,
            config.snow_area_per_particle_m2,
            terrain.config.shape,
            terrain.config.radius.end,
            terrain_body.major_radius,
            spawn_axial,
        );

        let submission = loader.finish();
        render.accept_submission(submission);
        render.wait_for_gpu();
        render.configure_map(map_extent, &terrain.config);

        // Camera clip-far has to cover the far side of the world: the
        // cylinder is bounded by its length along Z, the sphere and the
        // torus by their outer diameter.
        let clip_far = match terrain.config.shape {
            WorldShape::Sphere => 4.0 * terrain.config.radius.end,
            WorldShape::Cylinder => terrain.config.length,
            WorldShape::Torus => 2.0 * (terrain_body.major_radius + terrain.config.radius.end),
        };
        let spawn_up = {
            let up = terrain_body.up(rapier3d::math::Vec3::new(
                spawn_pose.translation.vector.x,
                spawn_pose.translation.vector.y,
                spawn_pose.translation.vector.z,
            ));
            nalgebra::Vector3::new(up.x, up.y, up.z)
        };
        let camera = Camera {
            pos: spawn_pose.translation.vector + spawn_up * 0.5,
            rot: nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::x_axis(),
                0.3 * f32::consts::PI,
            ),
            clip: 1.0..clip_far,
            ..Default::default()
        };

        let recorder = config.record.as_ref().map(Recorder::new);

        // Hostile mechos contact marker: ahead of spawn along chassis forward,
        // lifted slightly along world-up so the threat light reads on the road.
        let contact_marker_pos = {
            let forward = spawn_pose.rotation * car_forward_local();
            spawn_pose.translation.vector + forward * CONTACT_MARKER_AHEAD + spawn_up * 1.5
        };
        log::info!(
            "Ready. Mode: Driving. Controls: WASD drive, Space jump, LShift turbo, V force vandal contact (then ram the red chase), F scrap-forged spike (after depot), ~ pause, Esc quit"
        );
        log::info!(
            "Wasteland contact marker at [{:.1}, {:.1}, {:.1}] (drive near or wait ~{:.0}s)",
            contact_marker_pos.x,
            contact_marker_pos.y,
            contact_marker_pos.z,
            CONTACT_AUTO_SECS,
        );

        Self {
            choir,
            _choir_workers,
            render,
            physics,
            recorder,
            window,
            window_size,
            camera,
            in_camera_drag: false,
            last_mouse_pos: [0; 2],
            mode: Mode::Driving,
            input: DriveInput::default(),
            last_drive_cmd: (f32::NAN, f32::NAN, f32::NAN),
            last_redraw_time: time::Instant::now(),
            physics_accumulator: time::Duration::ZERO,
            jump_charge_start: None,
            camera_initialized: false,
            terrain_body,
            terrain,
            car,
            spawn_pose,
            last_good_pose: spawn_pose,
            snow,
            contact_phase: ContactPhase::Quiet,
            contact_quiet_elapsed: time::Duration::ZERO,
            threat_started: None,
            last_scrap_tick: None,
            contact_marker_pos,
            scrap_run_phase: ScrapRunPhase::Idle,
            scrap_beacon_pos: nalgebra::Vector3::zeros(),
            scrap_boost_until: None,
            scrap_run_started: None,
            vandal_chase_active: false,
            vandal_stun_until: None,
            last_vandal_hit: None,
            bump_title_until: None,
            hull: HULL_MAX,
            hull_critical_warned: false,
            hull_breach_until: None,
            spike_charges: 0,
        }
    }

    /// Initial chassis pose: chassis +Y along the world "up" at the spawn
    /// point, chassis forward (-X) along the world's axial direction.
    fn spawn_pose(
        map: &config::Map,
        spawn_radius: f32,
        spawn_axial: f32,
    ) -> nalgebra::Isometry3<f32> {
        match map.shape {
            // Cylinder and sphere spawn on the +Y side: up = +Y, and rotating
            // the chassis 90° about Y points its forward (-X) along +Z.
            WorldShape::Cylinder | WorldShape::Sphere => nalgebra::Isometry3 {
                translation: nalgebra::Vector3::new(0.0, spawn_radius, spawn_axial).into(),
                rotation: nalgebra::UnitQuaternion::from_axis_angle(
                    &nalgebra::Vector3::y_axis(),
                    0.5 * f32::consts::PI,
                ),
            },
            WorldShape::Torus => {
                let major_radius = map.length / f32::consts::TAU;
                let phi = spawn_axial / major_radius;
                // Tube angle π/2: the +Z side of the tube, so up = +Z there.
                let translation = nalgebra::Vector3::new(
                    major_radius * phi.cos(),
                    major_radius * phi.sin(),
                    spawn_radius,
                );
                let forward = nalgebra::Vector3::new(-phi.sin(), phi.cos(), 0.0);
                let c_x = -forward; // chassis forward is -X
                let c_y = nalgebra::Vector3::z(); // up
                let c_z = c_x.cross(&c_y);
                let rotation = nalgebra::UnitQuaternion::from_rotation_matrix(
                    &nalgebra::Rotation3::from_matrix_unchecked(
                        nalgebra::Matrix3::from_columns(&[c_x, c_y, c_z]),
                    ),
                );
                nalgebra::Isometry3 {
                    translation: translation.into(),
                    rotation,
                }
            }
        }
    }

    /// World "up" (away from the gravity anchor) at a point, as nalgebra.
    fn world_up(&self, pos: nalgebra::Vector3<f32>) -> nalgebra::Vector3<f32> {
        let up = self
            .terrain_body
            .up(rapier3d::math::Vec3::new(pos.x, pos.y, pos.z));
        nalgebra::Vector3::new(up.x, up.y, up.z)
    }

    fn load_car(
        loader: &mut Loader,
        physics: &mut Physics,
        car_path: &str,
        transform: nalgebra::Isometry3<f32>,
    ) -> Object {
        log::info!("Loading car: {}", car_path);
        let car_path = path::PathBuf::from("data/cars").join(car_path);
        let car_config: config::Car = ron::de::from_bytes(&assets::read(
            &car_path.join("car.ron"),
        ))
        .expect("Unable to parse the car config");
        let model_desc = Loader::read_gltf_data(
            &assets::read(&car_path.join("body.glb")),
            &car_path.join("body.glb"),
            Matrix4::identity().scale(car_config.scale),
        );
        let mut model = loader.load_model(&model_desc);
        // Apply the car-wide body tint into each material's base color factor.
        // Skip materials whose name contains "wheel" so tires (typically dark
        // GLB materials) don't get multiplied down into invisibility by the
        // rust tint — wheels stay their authored colour.
        let body_color = car_config.body_color;
        for (material, desc) in model.materials.iter_mut().zip(model_desc.materials.iter()) {
            let is_wheel = desc
                .name
                .as_deref()
                .map(|n| n.to_lowercase().contains("wheel"))
                .unwrap_or(false);
            if is_wheel {
                continue;
            }
            for (factor, tint) in material.base_color_factor.iter_mut().zip(body_color.iter()) {
                *factor *= *tint;
            }
        }
        let chassis_colliders = Self::create_chassis_colliders(&model_desc);

        // The chassis collider has zero density (it's a stub — wheels own the
        // ground interaction), so set the chassis inertial mass AND moment of
        // inertia explicitly. additional_mass alone leaves I = 0, which makes
        // the chassis infinitely resistant to angular acceleration — i.e. it
        // can never yaw or roll under torque (steering becomes impossible).
        let aabb = Self::chassis_aabb(&model_desc);
        let lx = aabb.maxs.x - aabb.mins.x;
        let ly = aabb.maxs.y - aabb.mins.y;
        let lz = aabb.maxs.z - aabb.mins.z;
        let chassis_mass = lx * ly * lz * 0.1 * car_config.density;
        // Solid cuboid inertia about each principal axis: I = m/12 · (a² + b²)
        // where a, b are the two extents perpendicular to that axis.
        let inertia = rapier3d::math::Vec3::new(
            chassis_mass / 12.0 * (ly * ly + lz * lz),
            chassis_mass / 12.0 * (lx * lx + lz * lz),
            chassis_mass / 12.0 * (lx * lx + ly * ly),
        );
        // Shift the center of mass below the chassis geometric origin, toward the
        // wheel axle level. A high CoM relative to the wheel base makes the car
        // prone to flipping during turns; pulling the CoM down here gives us a
        // stable, low-slung buggy feel without changing the visual mass.
        let chassis_com = rapier3d::math::Vec3::new(0.0, -0.25, 0.0);
        log::info!(
            "chassis mass {chassis_mass:.2} kg, principal inertia ({:.3}, {:.3}, {:.3}), com_y={}",
            inertia.x,
            inertia.y,
            inertia.z,
            chassis_com.y,
        );
        let mass_props =
            rapier3d::dynamics::MassProperties::new(chassis_com, chassis_mass, inertia);

        let rigid_body = rapier3d::dynamics::RigidBodyBuilder::dynamic()
            .pose(transform.into())
            .additional_mass_properties(mass_props)
            .linear_damping(0.4)
            // rapier's angular_damping is a single scalar across all three
            // axes, which forces us to trade upright-stability for steering
            // response. We zero it here and instead apply per-axis damping
            // (see Physics::apply_local_angular_damping in update_physics)
            // with high roll/pitch and low yaw values.
            .angular_damping(0.0)
            .build();

        let PhysicsBodyHandle {
            rigid_body_handle: chassis,
            ..
        } = physics.add_rigid_body(rigid_body, chassis_colliders);

        let axis_local = rapier3d::math::Vec3::new(
            car_config.wheel_axis[0],
            car_config.wheel_axis[1],
            car_config.wheel_axis[2],
        );
        let chassis_pose: rapier3d::math::Pose = transform.into();
        let wheels: Vec<Wheel> = car_config
            .wheels
            .iter()
            .map(|w| {
                let anchor_local =
                    rapier3d::math::Vec3::new(w.position[0], w.position[1], w.position[2]);
                let wheel_world = chassis_pose * anchor_local;
                let wheel_body = rapier3d::dynamics::RigidBodyBuilder::dynamic()
                    .pose(rapier3d::math::Pose::from_parts(
                        wheel_world,
                        chassis_pose.rotation,
                    ))
                    .angular_damping(0.2)
                    .build();
                let wheel_collider = rapier3d::geometry::ColliderBuilder::ball(w.radius)
                    .density(car_config.density)
                    .friction(3.0)
                    .build();
                let PhysicsBodyHandle {
                    rigid_body_handle: wheel_rb,
                    ..
                } = physics.add_rigid_body(wheel_body, vec![wheel_collider]);

                // Two-joint chain for front (steered) wheels and a single
                // joint for rear wheels. Without the knuckle, a single
                // GenericJoint's AngZ motor rotates the wheel about chassis Z
                // — which is *not* the wheel's axle when steered, so the
                // wheel "wobbles" around the steered direction once it starts
                // spinning. With the knuckle: chassis ↔ knuckle owns AngY
                // (steering), knuckle ↔ wheel owns AngZ (spin) + LinY
                // (suspension). The knuckle-relative AngZ axis IS the steered
                // axle.
                use rapier3d::dynamics::{
                    GenericJointBuilder, JointAxesMask, JointAxis, MassProperties, MotorModel,
                };
                let is_steering = anchor_local.x < 0.0;
                let _ = axis_local; // OxidizeMonk uses chassis-Z; hardcoded below.

                let steering_joint = if is_steering {
                    let knuckle_body = rapier3d::dynamics::RigidBodyBuilder::dynamic()
                        .pose(rapier3d::math::Pose::from_parts(
                            wheel_world,
                            chassis_pose.rotation,
                        ))
                        .angular_damping(0.0)
                        .additional_mass_properties(MassProperties::new(
                            rapier3d::math::Vec3::ZERO,
                            0.01,
                            rapier3d::math::Vec3::new(1e-4, 1e-4, 1e-4),
                        ))
                        .build();
                    let PhysicsBodyHandle {
                        rigid_body_handle: knuckle_rb,
                        ..
                    } = physics.add_rigid_body(knuckle_body, vec![]);
                    // Chassis ↔ knuckle: lock everything except AngY.
                    let steer_locked = JointAxesMask::LIN_X
                        | JointAxesMask::LIN_Y
                        | JointAxesMask::LIN_Z
                        | JointAxesMask::ANG_X
                        | JointAxesMask::ANG_Z;
                    let steer_joint = GenericJointBuilder::new(steer_locked)
                        .local_anchor1(anchor_local)
                        .local_anchor2(rapier3d::math::Vec3::ZERO)
                        .contacts_enabled(false)
                        // ForceBased gives the steering motor a direct
                        // `stiffness × pos_err` torque (up to STEER_MAX_FORCE)
                        // independent of the knuckle's tiny inertia.
                        // AccelerationBased would multiply by mass and produce
                        // ~0.01 × accel = negligible torque, so even small
                        // gyroscopic precession from the spinning wheel would
                        // visibly wobble the steered direction.
                        .motor_model(JointAxis::AngY, MotorModel::ForceBased)
                        .motor_position(JointAxis::AngY, 0.0, STEER_STIFFNESS, STEER_DAMPING)
                        .motor_max_force(JointAxis::AngY, STEER_MAX_FORCE)
                        .limits(JointAxis::AngY, [-MAX_STEER_ANGLE, MAX_STEER_ANGLE])
                        .build();
                    Some((
                        knuckle_rb,
                        physics.add_generic_joint(chassis, knuckle_rb, steer_joint),
                    ))
                } else {
                    None
                };
                let knuckle_rb = steering_joint.as_ref().map(|(rb, _)| *rb);

                // wheel_joint: handles suspension (LinY) and spin (AngZ).
                // AngY is locked here: steering is owned by the chassis ↔
                // knuckle joint above (for front wheels) or doesn't exist
                // (for rear wheels).
                let wheel_locked = JointAxesMask::LIN_X
                    | JointAxesMask::LIN_Z
                    | JointAxesMask::ANG_X
                    | JointAxesMask::ANG_Y;
                let (parent_rb, parent_anchor) = match steering_joint {
                    Some((knuckle_rb, _)) => (knuckle_rb, rapier3d::math::Vec3::ZERO),
                    None => (chassis, anchor_local),
                };
                let wheel_joint = GenericJointBuilder::new(wheel_locked)
                    .local_anchor1(parent_anchor)
                    .local_anchor2(rapier3d::math::Vec3::ZERO)
                    .contacts_enabled(false)
                    .motor_model(JointAxis::LinY, MotorModel::ForceBased)
                    .motor_position(
                        JointAxis::LinY,
                        0.0,
                        SUSPENSION_STIFFNESS,
                        SUSPENSION_DAMPING,
                    )
                    .motor_max_force(JointAxis::LinY, SUSPENSION_MAX_FORCE)
                    .limits(JointAxis::LinY, [-0.3, 0.3])
                    .motor_model(JointAxis::AngZ, MotorModel::ForceBased)
                    .motor_velocity(JointAxis::AngZ, 0.0, IDLE_BRAKE_FACTOR)
                    .motor_max_force(JointAxis::AngZ, car_config.motor_max_force)
                    .build();
                let joint_handle = physics.add_generic_joint(parent_rb, wheel_rb, wheel_joint);
                Wheel {
                    rigid_body: wheel_rb,
                    joint: joint_handle,
                    steering_joint: steering_joint.map(|(_, j)| j),
                    knuckle: knuckle_rb,
                    anchor_local,
                    is_steering,
                }
            })
            .collect();

        let chassis_instance = ModelInstance {
            model: Arc::new(model),
            transform,
            geometry_filter: None,
            casts_shadow: true,
        };

        // Procedural wheel mesh, used for every physics wheel. The wheel
        // rigid body's spin axis is chassis-local Z (the joint axle), so
        // the cylinder's axis matches local Z too — rotation of the body
        // about its local Z visibly spins the mesh; rotation about local Y
        // (steering) visibly turns it. Matching the body convention is what
        // lets the player see the steering response.
        let wheel_radius = car_config.wheels.first().map(|w| w.radius).unwrap_or(0.15);
        let wheel_model_desc =
            create_wheel_mesh_desc(wheel_radius * WHEEL_MESH_RADIUS_SCALE, WHEEL_HALF_WIDTH);
        let wheel_model = Arc::new(loader.load_model(&wheel_model_desc));
        // Only render procedural meshes for the front (steered) wheels.
        // OxidizeMonk's GLB already includes baked-in rear wheels, so drawing
        // procedural ones on top would double them up.
        let wheel_instances: Vec<Option<ModelInstance>> = wheels
            .iter()
            .map(|w| {
                if !w.is_steering {
                    return None;
                }
                let pose = physics.get_transform(w.rigid_body);
                Some(ModelInstance {
                    model: wheel_model.clone(),
                    transform: pose,
                    geometry_filter: None,
                    casts_shadow: true,
                })
            })
            .collect();

        Object {
            chassis_instance,
            wheel_instances,
            wheel_template_anchor: nalgebra::Vector3::zeros(),
            rigid_body: chassis,
            wheels,
            motor_max_velocity: car_config.motor_max_velocity,
            chassis_bottom_y: aabb.mins.y,
            chassis_top_y: aabb.maxs.y,
        }
    }

    /// AABB of the non-wheel chassis vertices in chassis-local coords. Used as a
    /// coarse mass-volume estimate for the chassis (since the up-facing trimesh
    /// is an open surface that Rapier can't integrate over).
    fn chassis_aabb(model_desc: &ModelDesc) -> rapier3d::parry::bounding_volume::Aabb {
        use rapier3d::parry::bounding_volume::Aabb;
        let keep = |m: &vandals_and_heroes::MaterialDesc| {
            !m.name
                .as_deref()
                .map(|n| n.to_lowercase().contains("wheel"))
                .unwrap_or(false)
        };
        let positions = model_desc.positions_filtered(keep);
        if positions.is_empty() {
            return Aabb::new_invalid();
        }
        let mut mins = positions[0];
        let mut maxs = positions[0];
        for p in &positions[1..] {
            mins.x = mins.x.min(p.x);
            mins.y = mins.y.min(p.y);
            mins.z = mins.z.min(p.z);
            maxs.x = maxs.x.max(p.x);
            maxs.y = maxs.y.max(p.y);
            maxs.z = maxs.z.max(p.z);
        }
        Aabb::new(
            rapier3d::math::Vec3::new(mins.x, mins.y, mins.z),
            rapier3d::math::Vec3::new(maxs.x, maxs.y, maxs.z),
        )
    }

    /// Build the chassis's collision proxy as a set of small balls placed at the
    /// 8 corners of the (non-wheel) chassis AABB. The bilinear-surface dispatcher
    /// only generates contacts for Ball shapes, so using balls — rather than a
    /// single Cuboid/TriMesh — lets every chassis corner get a proper smooth
    /// contact with the terrain.
    ///
    /// Together the corners act as a coarse "do not sink through ground" cage:
    /// flipped over, the chassis-+Y corners (now pointing radially inward) catch
    /// on the surface before the body can fall through.
    ///
    /// Wheel-vs-chassis contacts are disabled at each wheel joint
    /// (`contacts_enabled(false)`), so the corner balls don't fight the wheel
    /// colliders even if they overlap geometrically.
    fn create_chassis_colliders(model_desc: &ModelDesc) -> Vec<rapier3d::geometry::Collider> {
        let aabb = Self::chassis_aabb(model_desc);
        // Only the TOP four corners (chassis-local +Y face). When the chassis is
        // upright, these sit above the wheel envelope and never touch terrain,
        // so they don't snag on ridges taller than the ground clearance. When
        // the chassis flips upside-down, they become the new bottom and support
        // the body from sinking through the heightfield (the original reason
        // these colliders exist). Bottom corners were dropped because they
        // caught on every Fostral ridge > ~0.5 m and wedged the car solid.
        const CORNER_RADIUS: f32 = 0.10;
        let corners = [
            rapier3d::math::Vec3::new(aabb.mins.x, aabb.maxs.y, aabb.mins.z),
            rapier3d::math::Vec3::new(aabb.maxs.x, aabb.maxs.y, aabb.mins.z),
            rapier3d::math::Vec3::new(aabb.mins.x, aabb.maxs.y, aabb.maxs.z),
            rapier3d::math::Vec3::new(aabb.maxs.x, aabb.maxs.y, aabb.maxs.z),
        ];
        log::info!(
            "chassis AABB: x=[{:.2}, {:.2}] y=[{:.2}, {:.2}] z=[{:.2}, {:.2}], {} top-corner balls (r={CORNER_RADIUS})",
            aabb.mins.x,
            aabb.maxs.x,
            aabb.mins.y,
            aabb.maxs.y,
            aabb.mins.z,
            aabb.maxs.z,
            corners.len(),
        );
        corners
            .iter()
            .map(|&p| {
                rapier3d::geometry::ColliderBuilder::ball(CORNER_RADIUS)
                    .translation(p)
                    // Zero density — chassis mass comes from additional_mass_properties.
                    .density(0.0)
                    // Frictionless: corner balls catch the chassis radially (normal
                    // force prevents sinking through terrain) but mustn't brake the
                    // chassis when it's driving past a bump that's tall enough for a
                    // corner to graze the surface. Wheel friction (3.0) still does
                    // all the driving traction work.
                    .friction(0.0)
                    .build()
            })
            .collect()
    }

    fn update_physics(&mut self) {
        profiling::scope!("Game::update_physics");
        if self.mode != Mode::Driving {
            return;
        }
        self.physics.update_gravity(&self.terrain_body);
        // Yaw / tumble damping split: low damping about the world radial-out
        // axis at the chassis position (steering stays responsive), high
        // damping for everything else (the chassis stays upright through
        // bumps). Replaces rapier's single-scalar angular_damping, which
        // forced us to trade upright-stability against steering response.
        self.physics
            // Light yaw damping so steering input integrates into a brisk
            // chassis turn rate; the over-damped suspension above stops the
            // straight-line wobble at its source.
            .apply_axial_angular_damping(self.car.rigid_body, &self.terrain_body, 0.15, 2.0);
        // apply_driving_input must run AFTER update_gravity because the latter
        // calls rb.reset_forces, which would wipe out any drive force we added.
        self.apply_driving_input();
        self.physics.step();
        self.car.chassis_instance.transform = self.physics.get_transform(self.car.rigid_body);
        // Per-physics-wheel transform sync so the procedural cylinder meshes
        // visibly spin (AngZ) and turn (AngY) with their rigid bodies.
        for (wi, w) in self.car.wheels.iter().enumerate() {
            if let Some(Some(inst)) = self.car.wheel_instances.get_mut(wi) {
                inst.transform = self.physics.get_transform(w.rigid_body);
            }
        }
        // Sync debug-snow render instances and recycle settled particles.
        self.snow.update(&mut self.physics);
        if let Some(recorder) = self.recorder.as_mut() {
            let mut bodies: Vec<(String, rapier3d::dynamics::RigidBodyHandle)> =
                vec![("car".to_string(), self.car.rigid_body)];
            for (i, w) in self.car.wheels.iter().enumerate() {
                bodies.push((format!("wheel{}", i), w.rigid_body));
            }
            recorder.record(
                self.physics.last_time(),
                &self.physics,
                bodies.iter().map(|(n, h)| (n.as_str(), *h)),
            );
        }
        self.check_out_of_bounds();
    }


    /// True when the chassis has left the playable radial shell (and, for
    /// cylinders, the axial slab). Uses distance from the terrain gravity
    /// anchor so torus / cylinder / sphere share one check.
    fn is_out_of_bounds(&self, pos: nalgebra::Vector3<f32>) -> bool {
        let p = rapier3d::math::Vec3::new(pos.x, pos.y, pos.z);
        let anchor = self.terrain_body.gravity_anchor(p);
        let radial = (p - anchor).length();
        let r = &self.terrain.config.radius;
        if radial > r.end + OOB_RADIAL_OUTER_MARGIN {
            return true;
        }
        if radial < (r.start - OOB_RADIAL_INNER_MARGIN).max(0.0) {
            return true;
        }
        // Cylinder has open ends along Z; torus wraps and sphere has none.
        if self.terrain.config.shape == WorldShape::Cylinder {
            let half = 0.5 * self.terrain.config.length + OOB_AXIAL_MARGIN;
            if pos.z.abs() > half {
                return true;
            }
        }
        false
    }

    /// Soft recovery: teleport the car assembly back to the last grounded
    /// in-bounds pose (else initial spawn), zero velocities, snap the chase
    /// camera so it does not linger in the skybox.
    fn respawn_car(&mut self, pose: nalgebra::Isometry3<f32>) {
        let chassis_pose: rapier3d::math::Pose = pose.into();
        self.physics
            .teleport_body_pose(self.car.rigid_body, pose);
        for w in &self.car.wheels {
            let wheel_world = chassis_pose * w.anchor_local;
            let part_pose = nalgebra::Isometry3 {
                translation: nalgebra::Vector3::new(
                    wheel_world.x,
                    wheel_world.y,
                    wheel_world.z,
                )
                .into(),
                rotation: pose.rotation,
            };
            if let Some(knuckle) = w.knuckle {
                self.physics.teleport_body_pose(knuckle, part_pose);
            }
            self.physics.teleport_body_pose(w.rigid_body, part_pose);
            self.physics
                .set_joint_motor_velocity(w.joint, 0.0, IDLE_BRAKE_FACTOR);
            if let Some(steering_joint) = w.steering_joint {
                self.physics.set_joint_motor_position(
                    steering_joint,
                    rapier3d::dynamics::JointAxis::AngY,
                    0.0,
                    STEER_STIFFNESS,
                    STEER_DAMPING,
                );
            }
        }
        self.car.chassis_instance.transform = pose;
        for (wi, w) in self.car.wheels.iter().enumerate() {
            if let Some(Some(inst)) = self.car.wheel_instances.get_mut(wi) {
                inst.transform = self.physics.get_transform(w.rigid_body);
            }
        }
        self.jump_charge_start = None;
        // Snap chase camera onto the recovered chassis next follow tick.
        self.camera_initialized = false;
        log::info!(
            "OOB respawn at [{:.1}, {:.1}, {:.1}]",
            pose.translation.vector.x,
            pose.translation.vector.y,
            pose.translation.vector.z,
        );
    }

    fn check_out_of_bounds(&mut self) {
        let xform = self.physics.get_transform(self.car.rigid_body);
        let pos = xform.translation.vector;
        if self.is_out_of_bounds(pos) {
            // Prefer last grounded pose; fall back to initial spawn if that
            // snapshot somehow drifted out of bounds too.
            let target = if self.is_out_of_bounds(self.last_good_pose.translation.vector) {
                self.spawn_pose
            } else {
                self.last_good_pose
            };
            self.respawn_car(target);
            return;
        }
        // Refresh last-good only while clearly on the surface so we do not
        // snapshot mid-air poses that would dump the player into free-fall.
        if self.chassis_grounded() {
            self.last_good_pose = xform;
        }
    }

    fn apply_driving_input(&mut self) {
        let throttle = match (self.input.forward, self.input.backward) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        };
        let steer = match (self.input.steer_right, self.input.steer_left) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        };
        let turbo = if self.input.turbo { TURBO_FACTOR } else { 1.0 };
        let cmd = (throttle, steer, turbo);
        if cmd != self.last_drive_cmd {
            log::info!("drive cmd: throttle={throttle:.1} steer={steer:.1} turbo={turbo:.1}");
            self.last_drive_cmd = cmd;
        }

        // All-wheel drive: every wheel gets the throttle. With the knuckle in
        // the chain the AngZ motor on each wheel pushes about the wheel's
        // actual axle (post-steer for front wheels, chassis Z for rear), so
        // applying drive to all four no longer fights the steering as it
        // would have on the single-joint setup.
        let max_v = self.car.motor_max_velocity;
        let threat_factor = if self.contact_phase == ContactPhase::Threat {
            THREAT_DRIVE_FACTOR
        } else {
            1.0
        };
        let hull_factor = if self.hull_breach_until.is_some() {
            HULL_BREACH_DRIVE_FACTOR
        } else if self.hull <= HULL_CRITICAL {
            HULL_CRITICAL_DRIVE_FACTOR
        } else {
            1.0
        };
        let boost_factor = match self.scrap_boost_until {
            Some(until) if time::Instant::now() < until => SCRAP_BOOST_FACTOR,
            Some(_) => {
                self.scrap_boost_until = None;
                1.0
            }
            None => 1.0,
        };
        let drive_v = throttle * max_v * turbo * threat_factor * hull_factor * boost_factor;
        let driving = drive_v != 0.0;
        let steer_angle = steer * MAX_STEER_ANGLE;
        for wheel in &self.car.wheels {
            let (target_v, factor) = if driving {
                (drive_v, 1.0)
            } else {
                (0.0, IDLE_BRAKE_FACTOR)
            };
            self.physics
                .set_joint_motor_velocity(wheel.joint, target_v, factor);
            if let Some(steering_joint) = wheel.steering_joint {
                self.physics.set_joint_motor_position(
                    steering_joint,
                    rapier3d::dynamics::JointAxis::AngY,
                    steer_angle,
                    STEER_STIFFNESS,
                    STEER_DAMPING,
                );
            }
        }
    }

    /// Apply a sharp angular impulse about the chassis-forward axis so the
    /// player can flip the car back upright after a roll-over. `direction`
    /// is +1 to roll right (clockwise viewed from behind), -1 to roll left.
    fn roll(&mut self, direction: f32) {
        // Roll only counts as a "rescue" while we're against the surface.
        // In the air the player has no leverage to flip the chassis — and
        // letting them spin it freely would feel arcadey rather than
        // physical.
        if !self.chassis_grounded() {
            log::info!("roll {:+.0}: airborne, ignored", direction);
            return;
        }
        let xform = self.physics.get_transform(self.car.rigid_body);
        // The chassis-local roll axis is the car's forward direction.
        let forward_world = xform.rotation * car_forward_local();
        let inertia = self
            .physics
            .body_kinematics(self.car.rigid_body)
            .map(|_| self.physics.body_mass(self.car.rigid_body))
            .unwrap_or(0.0);
        // ω target ~ 6 rad/s — enough to spin a typical chassis past 90°
        // before damping kicks in. Scale by mass so light/heavy vehicles
        // both flip in roughly the same time.
        let target_ang_speed = 6.0_f32;
        let angular_impulse_mag = inertia * target_ang_speed;
        let impulse_vec = forward_world * (direction * angular_impulse_mag);
        let torque = rapier3d::math::Vec3::new(impulse_vec.x, impulse_vec.y, impulse_vec.z);
        self.physics
            .apply_torque_impulse(self.car.rigid_body, torque);
        log::info!("roll {:+.0}", direction);
    }

    /// True if any wheel *or* the chassis itself is in contact with the
    /// terrain heightfield. The chassis fallback handles the upside-down
    /// case: when the car is on its roof, the wheels are airborne but the
    /// chassis's top-corner balls are pressed against the ground, and a
    /// jump from there should still launch the cabin off the surface.
    fn chassis_grounded(&self) -> bool {
        self.car.wheels.iter().any(|w| {
            self.physics
                .is_touching_terrain(w.rigid_body, &self.terrain_body)
        }) || self
            .physics
            .is_touching_terrain(self.car.rigid_body, &self.terrain_body)
    }

    /// Space-key state machine: on press, start charging (if grounded); on
    /// release, fire a jump scaled by the held duration. The redraw loop also
    /// calls [`Self::check_jump_max_charge`] to auto-fire when the player
    /// holds Space past [`JUMP_MAX_CHARGE`].
    fn handle_jump_key(&mut self, pressed: bool) {
        if pressed {
            if self.jump_charge_start.is_none() && self.chassis_grounded() {
                self.jump_charge_start = Some(time::Instant::now());
            }
        } else if let Some(start) = self.jump_charge_start.take() {
            let charge = time::Instant::now() - start;
            self.execute_jump(charge);
        }
    }

    /// Auto-fire the jump if the player has held Space past `JUMP_MAX_CHARGE`.
    /// Called from `redraw` so a held button doesn't leave the chassis
    /// permanently glued to the ground "charging".
    fn check_jump_max_charge(&mut self) {
        if let Some(start) = self.jump_charge_start
            && time::Instant::now() - start >= JUMP_MAX_CHARGE
        {
            self.execute_jump(JUMP_MAX_CHARGE);
            self.jump_charge_start = None;
        }
    }

    fn execute_jump(&mut self, charge: time::Duration) {
        // Grounded check at fire time too — the chassis may have rolled off a
        // cliff during the charge. Without this the player could "jump"
        // mid-air on release.
        if !self.chassis_grounded() {
            log::info!("jump: charge released mid-air, cancelled");
            return;
        }
        let charge_s = charge.as_secs_f32();
        let max_s = JUMP_MAX_CHARGE.as_secs_f32();
        let frac = (charge_s / max_s).clamp(0.0, 1.0);
        let velocity = JUMP_MIN_VELOCITY + (JUMP_MAX_VELOCITY - JUMP_MIN_VELOCITY) * frac;
        log::info!(
            "jump: charge {:.2}s/{:.2}s ({:.0}%) → v={velocity:.2} m/s",
            charge_s,
            max_s,
            frac * 100.0
        );

        // Detect upside-down. World "up" is radial-outward from the world's
        // gravity anchor (sphere origin, cylinder Z axis, torus centreline).
        // Compare it to the chassis +Y direction: if they're on opposite
        // sides we're upside-down and the impulse should originate from the
        // *cabin* (chassis +Y_max) pushing the body away from the ground
        // it's resting on, instead of from the wheels.
        let xform = self.physics.get_transform(self.car.rigid_body);
        let car_pos = xform.translation.vector;
        let world_up = self.world_up(car_pos);
        let chassis_y_world = xform.rotation * nalgebra::Vector3::y();
        let upright = chassis_y_world.dot(&world_up) >= 0.0;
        let (anchor_y, push_dir_local) = if upright {
            // Upright: bottom of chassis pushes off the ground in chassis +Y.
            (self.car.chassis_bottom_y, nalgebra::Vector3::y())
        } else {
            // Upside-down: top of chassis (the cabin, now resting against
            // the ground) pushes in chassis -Y, which is world +up.
            (self.car.chassis_top_y, -nalgebra::Vector3::y())
        };
        let push_local = nalgebra::Vector3::new(0.0, anchor_y, 0.0);
        let bottom_world = xform.translation.vector + (xform.rotation * push_local);
        let chassis_up_world = xform.rotation * push_dir_local;

        let mass = self.physics.body_mass(self.car.rigid_body);
        let impulse = chassis_up_world * (mass * velocity);
        self.physics.apply_impulse_at_point(
            self.car.rigid_body,
            rapier3d::math::Vec3::new(impulse.x, impulse.y, impulse.z),
            rapier3d::math::Vec3::new(bottom_world.x, bottom_world.y, bottom_world.z),
        );
    }

    /// Headlights (spots) locked to the chassis plus one or two soft fills
    /// so nearby terrain gets readable shading beyond the radial "sun".
    fn build_local_lights(&self) -> Vec<LocalLight> {
        let xform = &self.car.chassis_instance.transform;
        let rot = xform.rotation;
        let car_pos = xform.translation.vector;
        let forward = rot * car_forward_local();
        let forward_arr = [forward.x, forward.y, forward.z];

        let mut lights = Vec::with_capacity(6);
        for local in &HEADLIGHT_LOCAL {
            let world = car_pos + rot * *local;
            lights.push(LocalLight {
                position: [world.x, world.y, world.z],
                color: HEADLIGHT_COLOR,
                intensity: HEADLIGHT_INTENSITY,
                range: HEADLIGHT_RANGE,
                kind: LocalLightKind::Spot {
                    direction: forward_arr,
                    inner_cone: HEADLIGHT_INNER,
                    outer_cone: HEADLIGHT_OUTER,
                    falloff: HEADLIGHT_FALLOFF,
                },
            });
        }

        // Gentle omni fills: one slightly above/behind the car, one ahead and
        // a little to the side. Positions follow the chassis so the torus
        // silhouette stays lit as the player drives.
        let up = self.world_up(car_pos);
        let right = forward.cross(&up);
        let right = if right.norm_squared() < 1e-8 {
            nalgebra::Vector3::z()
        } else {
            right.normalize()
        };
        let fill_a = car_pos - forward * 4.0 + up * 6.0;
        let fill_b = car_pos + forward * 8.0 + up * 3.0 + right * 5.0;
        lights.push(LocalLight {
            position: [fill_a.x, fill_a.y, fill_a.z],
            color: [0.75, 0.85, 1.0],
            intensity: FILL_INTENSITY * 0.7,
            range: FILL_RANGE,
            kind: LocalLightKind::Omnidirectional,
        });
        lights.push(LocalLight {
            position: [fill_b.x, fill_b.y, fill_b.z],
            color: [1.0, 0.9, 0.75],
            intensity: FILL_INTENSITY,
            range: FILL_RANGE * 0.85,
            kind: LocalLightKind::Omnidirectional,
        });

        // Hostile mechos / chase proxy marker: dim ember Quiet, hot pulse Threat.
        let marker = self.contact_marker_pos;
        let (m_intensity, m_color) = match self.contact_phase {
            ContactPhase::Quiet => (6.0, [0.85, 0.25, 0.1]),
            ContactPhase::Threat => {
                let t = self
                    .threat_started
                    .map(|s| (time::Instant::now() - s).as_secs_f32())
                    .unwrap_or(0.0);
                let stunned = self
                    .vandal_stun_until
                    .is_some_and(|u| time::Instant::now() < u);
                // Stunned = slower amber flicker; chasing = hot red pulse.
                let (pulse_hz, base_i, color) = if stunned {
                    (3.0, 16.0, [1.0, 0.55, 0.12])
                } else {
                    (7.5, 32.0, THREAT_LIGHT_COLOR)
                };
                let pulse = 0.55 + 0.45 * (t * pulse_hz).sin();
                (base_i * pulse, color)
            }
            ContactPhase::Cleared => (2.5, [0.35, 0.4, 0.45]),
        };
        lights.push(LocalLight {
            position: [marker.x, marker.y, marker.z],
            color: m_color,
            intensity: m_intensity,
            range: THREAT_LIGHT_RANGE,
            kind: LocalLightKind::Omnidirectional,
        });
        // Scrap depot beacon: cyan/amber pulse while the heroes stash run is live.
        if self.scrap_run_phase == ScrapRunPhase::Active {
            let beacon = self.scrap_beacon_pos;
            let t = self
                .scrap_run_started
                .map(|s| (time::Instant::now() - s).as_secs_f32())
                .unwrap_or(0.0);
            let pulse = 0.55 + 0.45 * (t * 4.5).sin();
            let mix = 0.5 + 0.5 * (t * 2.2).sin();
            let color = [
                SCRAP_BEACON_COLOR[0] * (1.0 - mix) + SCRAP_BEACON_AMBER[0] * mix,
                SCRAP_BEACON_COLOR[1] * (1.0 - mix) + SCRAP_BEACON_AMBER[1] * mix,
                SCRAP_BEACON_COLOR[2] * (1.0 - mix) + SCRAP_BEACON_AMBER[2] * mix,
            ];
            lights.push(LocalLight {
                position: [beacon.x, beacon.y, beacon.z],
                color,
                intensity: 26.0 * pulse,
                range: SCRAP_BEACON_RANGE,
                kind: LocalLightKind::Omnidirectional,
            });
        }

        // Chassis alarm fill while chased — reads as scrap-fire pressure.
        if self.contact_phase == ContactPhase::Threat {
            let alarm = car_pos + up * 2.0 - forward * 1.5;
            let t = self
                .threat_started
                .map(|s| (time::Instant::now() - s).as_secs_f32())
                .unwrap_or(0.0);
            let pulse = 0.4 + 0.6 * ((t * 8.0).sin() * 0.5 + 0.5);
            lights.push(LocalLight {
                position: [alarm.x, alarm.y, alarm.z],
                color: THREAT_LIGHT_COLOR,
                intensity: 14.0 * pulse,
                range: 12.0,
                kind: LocalLightKind::Omnidirectional,
            });
        }
        lights
    }

    fn follow_camera(&mut self, dt: time::Duration) {
        let xform = &self.car.chassis_instance.transform;
        let car_pos = xform.translation.vector;
        // "Up" is radially outward from the world's gravity anchor. Gravity
        // points the opposite way (see Physics::update_gravity), so this
        // matches the player's intuition of "up away from the ground" in
        // every world shape.
        let up = self.world_up(car_pos);
        // Project the chassis-local forward direction onto the plane perpendicular
        // to up so the camera doesn't yaw with body roll.
        let forward_full = xform.rotation * car_forward_local();
        let mut forward = forward_full - up * forward_full.dot(&up);
        let fwd_len = forward.norm();
        forward = if fwd_len < 1e-6 {
            // Degenerate: car is pointing straight up. Fall back to any horizontal dir.
            nalgebra::Vector3::z()
        } else {
            forward / fwd_len
        };
        // Slightly lower than a 45° chase so more of the road fills the frame.
        let target_pos = car_pos - forward * FOLLOW_DIST + up * FOLLOW_HEIGHT;
        let look = (car_pos - target_pos).normalize();
        // Right-handed basis with camera local +X = right, +Y = down, +Z = forward
        // (matches the convention in shaders/terrain-draw.wgsl).
        let right = up.cross(&look).normalize();
        let down = look.cross(&right);
        let basis = nalgebra::Matrix3::from_columns(&[right, down, look]);
        let target_rot = nalgebra::UnitQuaternion::from_matrix(&basis);

        if !self.camera_initialized {
            // First frame: snap directly so we don't lerp from the
            // far-away initial pose set in Game::new.
            self.camera.pos = target_pos;
            self.camera.rot = target_rot;
            self.camera_initialized = true;
            return;
        }
        // Exponential follow: lerp position and slerp rotation toward target at
        // a rate that's framerate-independent. ~8 / sec means ~99% of the
        // remaining gap is closed every 0.5 s — fast enough that the camera
        // visibly tracks the car, slow enough that snap-pose changes (jumps,
        // collisions) don't teleport the view behind the chassis.
        let dt_secs = dt.as_secs_f32().min(0.1);
        let alpha = 1.0 - (-CAMERA_FOLLOW_RATE * dt_secs).exp();
        self.camera.pos += (target_pos - self.camera.pos) * alpha;
        self.camera.rot = self.camera.rot.slerp(&target_rot, alpha);
    }

    /// Refresh the window title with callsign + hull + contact / scrap-run status.
    fn refresh_window_title(&self) {
        let hull_n = self.hull.max(0.0).ceil() as i32;
        let hull_label = if self.hull_breach_until.is_some() {
            format!("hull 0 BREACHED")
        } else if self.hull <= HULL_CRITICAL {
            format!("hull {hull_n} CRITICAL")
        } else {
            format!("hull {hull_n}")
        };
        // Brief bump flash overrides status (camera-shake proxy) but keeps hull.
        if let Some(until) = self.bump_title_until {
            if time::Instant::now() < until {
                let title = format!(
                    "Vandals and Heroes — {PLAYER_CALLSIGN} · {hull_label} · ⚠ RAMMED — shake it off"
                );
                self.window.set_title(&title);
                return;
            }
        }
        let mut status = match self.scrap_run_phase {
            ScrapRunPhase::Active => "scrap run — find the depot".to_string(),
            ScrapRunPhase::Complete => "scrap delivered".to_string(),
            ScrapRunPhase::Idle => match self.contact_phase {
                ContactPhase::Quiet => "wasteland road — vandals quiet".to_string(),
                ContactPhase::Threat => {
                    if self.vandal_stun_until.is_some_and(|u| time::Instant::now() < u) {
                        "⚠ VANDAL STUNNED — press the scrap hit".to_string()
                    } else if self.vandal_chase_active {
                        "⚠ VANDAL CHASE — ram the red marker".to_string()
                    } else {
                        "⚠ VANDALS ON YOUR TRAIL".to_string()
                    }
                }
                ContactPhase::Cleared => "contact clear — heroes hold the road".to_string(),
            },
        };
        // Scrap-forged spike ready beat: title cue after depot delivery.
        if self.spike_charges > 0 {
            if self.scrap_run_phase == ScrapRunPhase::Complete
                && self.contact_phase != ContactPhase::Threat
            {
                status = "spike ready".to_string();
            } else if !status.contains("spike") {
                status = format!("{status} · spike ready");
            }
        }
        let title =
            format!("Vandals and Heroes — {PLAYER_CALLSIGN} · {hull_label} · {status}");
        self.window.set_title(&title);
    }

    /// Chip hull by `amount` (clamped ≥ 0). Fires critical radio once; may start soft fail.
    fn chip_hull(&mut self, amount: f32, reason: &str) {
        if amount <= 0.0 || self.hull_breach_until.is_some() {
            return;
        }
        let before = self.hull;
        self.hull = (self.hull - amount).max(0.0);
        log::info!(
            "Hull {before:.0} → {:.0} (−{amount:.0}) — {reason}",
            self.hull
        );
        if self.hull <= HULL_CRITICAL && !self.hull_critical_warned {
            self.hull_critical_warned = true;
            let tick = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as usize)
                .unwrap_or(0);
            let line = RADIO_HULL_CRITICAL[tick % RADIO_HULL_CRITICAL.len()];
            log::info!("[wasteland radio] {line}");
            log::info!(
                "Hull critical (≤{:.0}): extra motor derate {:.0}%",
                HULL_CRITICAL,
                HULL_CRITICAL_DRIVE_FACTOR * 100.0,
            );
        }
        if self.hull <= 0.0 {
            self.begin_hull_breach();
        } else {
            self.refresh_window_title();
        }
    }

    /// Repair hull by `amount` (capped at HULL_MAX). Clears critical warn if above threshold.
    fn repair_hull(&mut self, amount: f32, reason: &str) {
        if amount <= 0.0 {
            return;
        }
        let before = self.hull;
        self.hull = (self.hull + amount).min(HULL_MAX);
        if self.hull > HULL_CRITICAL {
            self.hull_critical_warned = false;
        }
        log::info!(
            "Hull {before:.0} → {:.0} (+{amount:.0}) — {reason}",
            self.hull
        );
        self.refresh_window_title();
    }

    /// Soft fail: radio breach, brief limp, then respawn at last-good with full hull.
    fn begin_hull_breach(&mut self) {
        if self.hull_breach_until.is_some() {
            return;
        }
        self.hull = 0.0;
        self.hull_breach_until =
            Some(time::Instant::now() + time::Duration::from_secs_f32(HULL_BREACH_LIMP_SECS));
        let line = RADIO_HULL_BREACH[0];
        log::info!("[wasteland radio] {line}");
        log::info!(
            "Hull breached — soft fail limp {:.1}s then respawn-at-last-good",
            HULL_BREACH_LIMP_SECS,
        );
        // Brief brake pulse so the breach reads as a hit.
        for wheel in &self.car.wheels {
            self.physics.set_joint_motor_velocity(
                wheel.joint,
                0.0,
                IDLE_BRAKE_FACTOR * 0.7,
            );
        }
        self.refresh_window_title();
    }

    /// Finish soft fail: teleport to last-good (or spawn), restore hull.
    fn finish_hull_breach(&mut self) {
        self.hull_breach_until = None;
        let target = if self.is_out_of_bounds(self.last_good_pose.translation.vector) {
            self.spawn_pose
        } else {
            self.last_good_pose
        };
        self.respawn_car(target);
        self.hull = HULL_MAX;
        self.hull_critical_warned = false;
        log::info!(
            "Hull soft-fail recovery — respawned, hull restored to {:.0}",
            HULL_MAX
        );
        self.refresh_window_title();
    }

    /// Passive regen + breach limp timer (wall-clock; Driving only).
    fn update_hull(&mut self, elapsed: time::Duration) {
        if let Some(until) = self.hull_breach_until {
            if time::Instant::now() >= until {
                self.finish_hull_breach();
            }
            return;
        }
        // Subtle regen while Quiet/Cleared, not turbo, not scrap-boosting.
        let calm = matches!(
            self.contact_phase,
            ContactPhase::Quiet | ContactPhase::Cleared
        );
        let boosting = self
            .scrap_boost_until
            .is_some_and(|u| time::Instant::now() < u);
        if calm && !self.input.turbo && !boosting && self.hull < HULL_MAX {
            let before = self.hull;
            self.hull = (self.hull + HULL_REGEN_PER_SEC * elapsed.as_secs_f32()).min(HULL_MAX);
            if self.hull > HULL_CRITICAL {
                self.hull_critical_warned = false;
            }
            // Refresh title when the displayed integer ticks.
            if before.ceil() as i32 != self.hull.ceil() as i32 {
                self.refresh_window_title();
            }
        }
    }

    /// Trip the wasteland contact beat: radio chatter, threat phase, chase proxy.
    fn begin_vandal_threat(&mut self, reason: &str) {
        if self.contact_phase != ContactPhase::Quiet {
            return;
        }
        // Prior scrap Complete yields to a fresh chase so the next clear can re-assign depot.
        if self.scrap_run_phase == ScrapRunPhase::Complete {
            self.scrap_run_phase = ScrapRunPhase::Idle;
            self.scrap_run_started = None;
        }
        self.contact_phase = ContactPhase::Threat;
        self.threat_started = Some(time::Instant::now());
        self.last_scrap_tick = Some(time::Instant::now());
        self.spawn_vandal_chase();
        log::info!("[wasteland radio] CONTACT ({reason})");
        for line in RADIO_CONTACT {
            log::info!("[wasteland radio] {line}");
        }
        log::info!(
            "Threat: drive derated to {:.0}% for {:.0}s — survive / ram the chase proxy",
            THREAT_DRIVE_FACTOR * 100.0,
            THREAT_DURATION_SECS,
        );
        log::info!(
            "Ram rules: closing ≥{:.0} m/s scrap-hit (chip {:.0}s); ≥{:.0} m/s early-clear",
            VANDAL_RAM_CLOSING_MIN,
            VANDAL_RAM_TIMER_CHIP_SECS,
            VANDAL_RAM_EARLY_CLEAR,
        );
        self.refresh_window_title();
        let hull_n = self.hull.max(0.0).ceil() as i32;
        log::info!(
            "Title beat: {PLAYER_CALLSIGN} · hull {hull_n} · ⚠ VANDAL CHASE (CONTACT)"
        );
    }

    /// Place / activate the kinematic vandal chase marker behind the chassis.
    fn spawn_vandal_chase(&mut self) {
        let xform = self.car.chassis_instance.transform;
        let forward = xform.rotation * car_forward_local();
        let p = xform.translation.vector;
        let up = {
            let u = self.terrain_body.up(rapier3d::math::Vec3::new(p.x, p.y, p.z));
            nalgebra::Vector3::new(u.x, u.y, u.z)
        };
        self.contact_marker_pos =
            p - forward * VANDAL_CHASE_SPAWN_BEHIND + up * 1.5;
        self.vandal_chase_active = true;
        self.vandal_stun_until = None;
        self.last_vandal_hit = None;
        log::info!(
            "Vandal chase proxy spawned at [{:.1}, {:.1}, {:.1}] — pursue + ram radius {:.0}m",
            self.contact_marker_pos.x,
            self.contact_marker_pos.y,
            self.contact_marker_pos.z,
            VANDAL_RAM_RADIUS,
        );
    }

    /// Stop / despawn the chase proxy (threat clear or scrap-run assign).
    fn despawn_vandal_chase(&mut self) {
        self.vandal_chase_active = false;
        self.vandal_stun_until = None;
        self.last_vandal_hit = None;
        self.bump_title_until = None;
    }

    fn clear_vandal_threat(&mut self) {
        if self.contact_phase != ContactPhase::Threat {
            return;
        }
        self.contact_phase = ContactPhase::Cleared;
        self.threat_started = None;
        self.last_scrap_tick = None;
        self.despawn_vandal_chase();
        let tick = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as usize)
            .unwrap_or(0);
        let line = RADIO_CLEAR[tick % RADIO_CLEAR.len()];
        log::info!("[wasteland radio] {line}");
        log::info!("Threat cleared — chase proxy despawned, full drive restored");
        self.refresh_window_title();
        // Heroes scrap-run beat kicks in once the chase pressure lifts.
        self.begin_scrap_run();
    }

    /// Assign the scrap-depot beacon mission (heroes stash) after contact clears.
    fn begin_scrap_run(&mut self) {
        if self.scrap_run_phase != ScrapRunPhase::Idle {
            return;
        }
        let xform = self.car.chassis_instance.transform;
        let forward = xform.rotation * car_forward_local();
        // Chassis right ≈ forward × up (local Z is roughly right for this model).
        let right = xform.rotation * nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let up = {
            let p = xform.translation.vector;
            let u = self.terrain_body.up(rapier3d::math::Vec3::new(p.x, p.y, p.z));
            nalgebra::Vector3::new(u.x, u.y, u.z)
        };
        self.scrap_beacon_pos =
            xform.translation.vector + forward * SCRAP_BEACON_AHEAD + right * SCRAP_BEACON_LATERAL
                + up * 1.5;
        self.scrap_run_phase = ScrapRunPhase::Active;
        self.scrap_run_started = Some(time::Instant::now());
        for line in RADIO_SCRAP_ASSIGN {
            log::info!("[wasteland radio] {line}");
        }
        log::info!(
            "Scrap depot beacon at [{:.1}, {:.1}, {:.1}] — reach within {:.0}m",
            self.scrap_beacon_pos.x,
            self.scrap_beacon_pos.y,
            self.scrap_beacon_pos.z,
            SCRAP_BEACON_RADIUS,
        );
        self.refresh_window_title();
    }

    fn complete_scrap_run(&mut self) {
        if self.scrap_run_phase != ScrapRunPhase::Active {
            return;
        }
        self.scrap_run_phase = ScrapRunPhase::Complete;
        let tick = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as usize)
            .unwrap_or(0);
        let line = RADIO_SCRAP_DONE[tick % RADIO_SCRAP_DONE.len()];
        log::info!("[wasteland radio] {line}");
        log::info!(
            "Scrap run complete — motor boost {:.0}% for {:.0}s",
            SCRAP_BOOST_FACTOR * 100.0,
            SCRAP_BOOST_SECS,
        );
        self.scrap_boost_until =
            Some(time::Instant::now() + time::Duration::from_secs_f32(SCRAP_BOOST_SECS));
        // Mission reward ties to car-as-character: depot welds patch the hull.
        self.repair_hull(HULL_SCRAP_REPAIR, "scrap depot delivery");
        let repair_line = RADIO_HULL_REPAIR[0];
        log::info!("[wasteland radio] {repair_line}");
        // Combat reward: scrap-forged spike charge (max SPIKE_MAX_CHARGES). Hull untouched by fire.
        let before = self.spike_charges;
        self.spike_charges = (self.spike_charges.saturating_add(1)).min(SPIKE_MAX_CHARGES);
        let spike_line = RADIO_SPIKE_READY[tick % RADIO_SPIKE_READY.len()];
        log::info!("[wasteland radio] {spike_line}");
        log::info!(
            "Scrap-forged spike charges {before} → {} (max {SPIKE_MAX_CHARGES}) — press F in chase ≤{SPIKE_RANGE:.0}m",
            self.spike_charges,
        );
        // Open the road for another chase so the spike can be spent in combat.
        self.contact_phase = ContactPhase::Quiet;
        self.contact_quiet_elapsed = time::Duration::ZERO;
        self.threat_started = None;
        self.last_scrap_tick = None;
        self.refresh_window_title();
    }

    /// Advance scrap-run proximity while the depot beacon is active.
    fn update_scrap_run(&mut self) {
        if self.scrap_run_phase != ScrapRunPhase::Active {
            return;
        }
        let car_pos = self.car.chassis_instance.transform.translation.vector;
        let dist = (car_pos - self.scrap_beacon_pos).norm();
        if dist <= SCRAP_BEACON_RADIUS {
            self.complete_scrap_run();
        }
    }

    /// Advance the contact / chase hook on wall-clock (called from redraw).
    fn update_vandal_contact(&mut self, elapsed: time::Duration) {
        // Clear expired bump title flash.
        if let Some(until) = self.bump_title_until {
            if time::Instant::now() >= until {
                self.bump_title_until = None;
                self.refresh_window_title();
            }
        }
        match self.contact_phase {
            ContactPhase::Quiet => {
                self.contact_quiet_elapsed += elapsed;
                let car_pos = self.car.chassis_instance.transform.translation.vector;
                let dist = (car_pos - self.contact_marker_pos).norm();
                if dist <= CONTACT_MARKER_RADIUS {
                    self.begin_vandal_threat("proximity to hostile marker");
                } else if self.contact_quiet_elapsed.as_secs_f32() >= CONTACT_AUTO_SECS {
                    self.begin_vandal_threat("auto wasteland contact timer");
                }
            }
            ContactPhase::Threat => {
                let Some(started) = self.threat_started else {
                    return;
                };
                let now = time::Instant::now();
                let threat_age = (now - started).as_secs_f32();
                if threat_age >= THREAT_DURATION_SECS {
                    self.clear_vandal_threat();
                    return;
                }
                // Pursue + ram/bump before scrap-tick so a solid ram can clear first.
                self.update_vandal_chase(elapsed);
                if self.contact_phase != ContactPhase::Threat {
                    return;
                }
                let due = self
                    .last_scrap_tick
                    .map(|t| (now - t).as_secs_f32() >= THREAT_SCRAP_TICK_SECS)
                    .unwrap_or(true);
                if due {
                    self.last_scrap_tick = Some(now);
                    let idx = (threat_age / THREAT_SCRAP_TICK_SECS) as usize;
                    let line = RADIO_THREAT_TICK[idx % RADIO_THREAT_TICK.len()];
                    log::info!("[wasteland radio] {line}");
                    // Brief scrap-pressure brake pulse so the player feels the hit.
                    for wheel in &self.car.wheels {
                        self.physics.set_joint_motor_velocity(
                            wheel.joint,
                            0.0,
                            IDLE_BRAKE_FACTOR * 0.35,
                        );
                    }
                }
            }
            ContactPhase::Cleared => {}
        }
    }

    /// Move the kinematic chase proxy toward the player; resolve ram / bump.
    fn update_vandal_chase(&mut self, elapsed: time::Duration) {
        if !self.vandal_chase_active {
            return;
        }
        let car_pos = self.car.chassis_instance.transform.translation.vector;
        let marker = self.contact_marker_pos;
        let to_player = car_pos - marker;
        let dist = to_player.norm();

        let now = time::Instant::now();
        let stunned = self.vandal_stun_until.is_some_and(|u| now < u);
        if self.vandal_stun_until.is_some_and(|u| now >= u) {
            self.vandal_stun_until = None;
            self.refresh_window_title();
        }
        let speed = if stunned {
            VANDAL_CHASE_SPEED * VANDAL_STUN_SPEED_FACTOR
        } else {
            VANDAL_CHASE_SPEED
        };
        // Hold a small stand-off so large frame dt (slow GPU) cannot teleport
        // onto the chassis and skip the overlap test.
        const STAND_OFF: f32 = 2.0;
        if dist > STAND_OFF {
            let step = (speed * elapsed.as_secs_f32()).min(dist - STAND_OFF);
            self.contact_marker_pos += to_player * (step / dist);
        }

        // Closing speed: player linvel projected onto player→vandal.
        let lv = self.physics.body_linvel(self.car.rigid_body);
        let car_vel = nalgebra::Vector3::new(lv.x, lv.y, lv.z);
        let to_vandal = self.contact_marker_pos - car_pos;
        let dist = to_vandal.norm();
        if dist > VANDAL_RAM_RADIUS {
            return;
        }
        let closing = if dist > 1e-3 {
            car_vel.dot(&(to_vandal / dist))
        } else {
            // Nested / overlapped — treat as contact with no player punch-through.
            0.0
        };

        let on_cooldown = self
            .last_vandal_hit
            .map(|t| (now - t).as_secs_f32() < VANDAL_HIT_COOLDOWN_SECS)
            .unwrap_or(false);
        if on_cooldown {
            return;
        }

        if closing >= VANDAL_RAM_CLOSING_MIN {
            self.on_player_ram_vandal(closing);
        } else if !stunned {
            // Vandal is on the bumper and player is not punching through.
            self.on_vandal_bump_player();
        }
    }

    /// Player rammed the chase proxy hard enough for scrap-hit feedback.
    fn on_player_ram_vandal(&mut self, closing: f32) {
        let now = time::Instant::now();
        self.last_vandal_hit = Some(now);
        self.vandal_stun_until =
            Some(now + time::Duration::from_secs_f32(VANDAL_STUN_SECS));

        // Knock the proxy slightly away so the overlap doesn't re-trigger instantly.
        let car_pos = self.car.chassis_instance.transform.translation.vector;
        let away = self.contact_marker_pos - car_pos;
        if away.norm() > 1e-3 {
            self.contact_marker_pos += away.normalize() * (VANDAL_RAM_RADIUS * 0.6);
        }

        if closing >= VANDAL_RAM_EARLY_CLEAR {
            // Solid player ram: aggressor — hull chip HULL_SOLID_RAM_CHIP (0).
            let tick = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as usize)
                .unwrap_or(0);
            let line = RADIO_RAM_CLEAR[tick % RADIO_RAM_CLEAR.len()];
            log::info!("[wasteland radio] {line}");
            log::info!(
                "Solid ram ({closing:.1} m/s ≥ {VANDAL_RAM_EARLY_CLEAR:.0}) — early-clear Threat (hull chip {:.0})",
                HULL_SOLID_RAM_CHIP,
            );
            self.clear_vandal_threat();
            return;
        }

        // Normal ram: tiny hull chip (you're the aggressor).
        self.chip_hull(HULL_RAM_CHIP, "player ram (aggressor, light chip)");
        // Chip remaining threat time by aging the phase start clock.
        if let Some(started) = self.threat_started.as_mut() {
            *started = *started
                - time::Duration::from_secs_f32(VANDAL_RAM_TIMER_CHIP_SECS);
        }
        let tick = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as usize)
            .unwrap_or(0);
        let line = RADIO_RAM_HIT[tick % RADIO_RAM_HIT.len()];
        log::info!("[wasteland radio] {line}");
        log::info!(
            "Scrap-hit ram ({closing:.1} m/s) — vandal stunned {:.1}s, threat −{:.0}s",
            VANDAL_STUN_SECS,
            VANDAL_RAM_TIMER_CHIP_SECS,
        );
        self.refresh_window_title();
    }

    /// Chasing vandal bumped the player: brake pulse + title flash.
    fn on_vandal_bump_player(&mut self) {
        let now = time::Instant::now();
        self.last_vandal_hit = Some(now);
        self.bump_title_until =
            Some(now + time::Duration::from_secs_f32(VANDAL_BUMP_TITLE_FLASH_SECS));
        for wheel in &self.car.wheels {
            self.physics.set_joint_motor_velocity(
                wheel.joint,
                0.0,
                IDLE_BRAKE_FACTOR * 0.55,
            );
        }
        let tick = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as usize)
            .unwrap_or(0);
        let line = RADIO_BUMPED[tick % RADIO_BUMPED.len()];
        log::info!("[wasteland radio] {line}");
        log::info!("Vandal bump — brake pulse + title flash + hull chip");
        self.chip_hull(HULL_BUMP_CHIP, "vandal bump");
        self.refresh_window_title();
    }

    /// Fire a scrap-forged spike if charged and a chase proxy is in range.
    /// Miss with charge held when no target; hull unchanged either way.
    fn try_fire_scrap_spike(&mut self) {
        if self.spike_charges == 0 {
            log::info!("Key F ignored — no scrap-forged spike charge (deliver scrap at depot)");
            return;
        }
        let chase_ok = self.vandal_chase_active && self.contact_phase == ContactPhase::Threat;
        let car_pos = self.car.chassis_instance.transform.translation.vector;
        let dist = (self.contact_marker_pos - car_pos).norm();
        let in_range = chase_ok && dist <= SPIKE_RANGE;
        if !in_range {
            let tick = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as usize)
                .unwrap_or(0);
            let line = RADIO_SPIKE_MISS[tick % RADIO_SPIKE_MISS.len()];
            log::info!("[wasteland radio] {line}");
            log::info!(
                "Spike miss — charge kept ({}); chase_active={} phase={:?} dist={dist:.1}m (need ≤{SPIKE_RANGE:.0}m)",
                self.spike_charges,
                self.vandal_chase_active,
                self.contact_phase,
            );
            return;
        }

        self.spike_charges -= 1;
        let now = time::Instant::now();
        self.last_vandal_hit = Some(now);
        self.vandal_stun_until =
            Some(now + time::Duration::from_secs_f32(SPIKE_STUN_SECS));

        // Knock chase backward along escape vector (player forward) — behind Ash-Runner.
        let escape = {
            let f = self.car.chassis_instance.transform.rotation * car_forward_local();
            let len = f.norm();
            if len > 1e-3 {
                f / len
            } else {
                nalgebra::Vector3::new(1.0, 0.0, 0.0)
            }
        };
        self.contact_marker_pos -= escape * SPIKE_KNOCKBACK;

        let tick = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as usize)
            .unwrap_or(0);
        let line = RADIO_SPIKE_FIRE[tick % RADIO_SPIKE_FIRE.len()];
        log::info!("[wasteland radio] {line}");
        log::info!(
            "Scrap-forged spike hit — stun {:.1}s, knockback {:.0}m, threat −{:.0}s (hull unchanged)",
            SPIKE_STUN_SECS,
            SPIKE_KNOCKBACK,
            SPIKE_TIMER_CHIP_SECS,
        );

        // Chip threat timer; early-clear if remaining is already low.
        if let Some(started) = self.threat_started.as_mut() {
            *started = *started - time::Duration::from_secs_f32(SPIKE_TIMER_CHIP_SECS);
        }
        let remaining = self
            .threat_started
            .map(|started| (THREAT_DURATION_SECS - (now - started).as_secs_f32()).max(0.0))
            .unwrap_or(0.0);
        if remaining <= SPIKE_EARLY_CLEAR_REMAINING {
            log::info!(
                "Spike early-clear — remaining threat {remaining:.1}s ≤ {SPIKE_EARLY_CLEAR_REMAINING:.0}s"
            );
            self.clear_vandal_threat();
            return;
        }
        self.refresh_window_title();
    }

    fn on_drive_key(&mut self, code: winit::keyboard::KeyCode, pressed: bool) {
        use winit::keyboard::KeyCode as Kc;
        match code {
            Kc::KeyW => self.input.forward = pressed,
            Kc::KeyS => self.input.backward = pressed,
            Kc::KeyA => self.input.steer_left = pressed,
            Kc::KeyD => self.input.steer_right = pressed,
            Kc::ShiftLeft => self.input.turbo = pressed,
            Kc::Space => self.handle_jump_key(pressed),
            // Demo / force-trigger: trip the vandal contact beat immediately.
            Kc::KeyV if pressed => {
                if self.contact_phase == ContactPhase::Quiet {
                    self.begin_vandal_threat("manual Key V");
                } else {
                    log::info!(
                        "Key V ignored — contact phase already {:?}",
                        self.contact_phase
                    );
                }
            }
            // Scrap-forged spike: spend a depot charge to hard-stun the chase proxy.
            Kc::KeyF if pressed => {
                self.try_fire_scrap_spike();
            }
            // `<` and `>` (Comma and Period — same physical keys as `<` and
            // `>` when Shift isn't held). Apply a sharp roll impulse about
            // the chassis-forward axis so the player can right an upside-
            // down or sideways-stuck vehicle.
            Kc::Comma if pressed => self.roll(-1.0),
            Kc::Period if pressed => self.roll(1.0),
            _ => return,
        }
        log::info!(
            "drive key {:?} -> input: fwd={} back={} L={} R={} turbo={}",
            code,
            self.input.forward,
            self.input.backward,
            self.input.steer_left,
            self.input.steer_right,
            self.input.turbo,
        );
    }

    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            Mode::Driving => Mode::Paused,
            Mode::Paused => Mode::Driving,
        };
        // Drop any held keys — Pressed events that arrived while the other mode was
        // active wouldn't have been recorded, so the state is unreliable either way.
        self.input = DriveInput::default();
        // Make sure wheel motors stop the moment we leave Driving; on the re-enter
        // they'll be re-set by apply_driving_input from the (now-zeroed) input.
        for wheel in &self.car.wheels {
            self.physics.set_joint_motor_velocity(wheel.joint, 0.0, 0.2);
        }
        log::info!("Mode: {:?}", self.mode);
    }

    /// Keep the canvas backing store at its CSS layout size times the
    /// device pixel ratio. winit reports the CSS size as the inner size but
    /// never resizes an app-provided canvas's backing attributes, so without
    /// this the page renders at the canvas default (300x150) stretched to
    /// fill the window. Setting the attributes also clears the drawing
    /// buffer, which is fine — a frame renders right after.
    #[cfg(target_arch = "wasm32")]
    fn sync_canvas_size(&mut self) {
        use winit::platform::web::WindowExtWebSys as _;
        let Some(canvas) = self.window.canvas() else {
            return;
        };
        let Some(web_window) = web_sys::window() else {
            return;
        };
        let dpr = web_window.device_pixel_ratio();
        let width = (canvas.client_width() as f64 * dpr) as u32;
        let height = (canvas.client_height() as f64 * dpr) as u32;
        if width == 0 || height == 0 {
            return;
        }
        if canvas.width() != width || canvas.height() != height {
            canvas.set_width(width);
            canvas.set_height(height);
        }
        if self.window_size.width != width || self.window_size.height != height {
            log::info!("Canvas resized to {width}x{height} (dpr {dpr})");
            self.window_size = winit::dpi::PhysicalSize::new(width, height);
            self.render.resize(gpu::Extent {
                width,
                height,
                depth: 1,
            });
        }
    }

    fn redraw(&mut self) -> time::Duration {
        profiling::scope!("Game::redraw");
        #[cfg(target_arch = "wasm32")]
        self.sync_canvas_size();
        // Fixed-timestep physics with an accumulator: physics simulation time
        // tracks wall-clock time independent of how often redraws fire. winit's
        // event loop calls redraw both on its 16 ms timer AND on incoming events
        // (key auto-repeats etc.), so we can't tie one physics step per redraw —
        // doing that lets held keys speed up the simulation.
        let now = time::Instant::now();
        let elapsed = now - self.last_redraw_time;
        self.last_redraw_time = now;
        if self.mode == Mode::Driving {
            // Auto-fire the jump if the player has been holding Space past
            // JUMP_MAX_CHARGE — keep this in the redraw path (rather than a
            // physics tick) since charge timing is wall-clock-based.
            self.check_jump_max_charge();
            // Contact / chase hook is wall-clock too (radio + threat duration).
            self.update_vandal_contact(elapsed);
            self.update_scrap_run();
            self.update_hull(elapsed);
            self.physics_accumulator += elapsed;
            let mut steps = 0;
            while self.physics_accumulator >= PHYSICS_DT && steps < MAX_PHYSICS_STEPS_PER_REDRAW {
                self.update_physics();
                self.physics_accumulator -= PHYSICS_DT;
                steps += 1;
            }
            // If we hit the cap, drop the leftover so we don't perpetually try
            // to catch up.
            if self.physics_accumulator >= PHYSICS_DT * MAX_PHYSICS_STEPS_PER_REDRAW {
                self.physics_accumulator = time::Duration::ZERO;
            }
            self.follow_camera(elapsed);
        } else {
            // No physics ticks while paused; also stop accumulating time.
            self.physics_accumulator = time::Duration::ZERO;
        }

        let mut model_instances: Vec<&ModelInstance> =
            Vec::with_capacity(1 + self.car.wheel_instances.len() + self.snow.instances.len());
        model_instances.push(&self.car.chassis_instance);
        model_instances.extend(self.car.wheel_instances.iter().filter_map(|o| o.as_ref()));
        model_instances.extend(self.snow.instances.iter());
        let lights = self.build_local_lights();
        self.render
            .draw(&self.camera, &self.terrain, &model_instances, &lights);

        time::Duration::from_millis(16)
    }

    pub fn on_event(
        &mut self,
        event: &winit::event::WindowEvent,
    ) -> Result<winit::event_loop::ControlFlow, QuitEvent> {
        match *event {
            winit::event::WindowEvent::Resized(size) => {
                if size != self.window_size {
                    log::info!("Resizing to {:?}", size);
                    self.window_size = size;
                    self.render.resize(gpu::Extent {
                        width: size.width,
                        height: size.height,
                        depth: 1,
                    });
                }
            }
            winit::event::WindowEvent::KeyboardInput { ref event, .. } => {
                // Log every keyboard event up front so we can see what the OS is
                // actually emitting — including events where physical_key is
                // Unidentified (which would otherwise silently fall through the
                // PhysicalKey::Code arm).
                log::info!(
                    "KeyboardInput: phys={:?} logical={:?} state={:?} repeat={}",
                    event.physical_key,
                    event.logical_key,
                    event.state,
                    event.repeat,
                );
                let pressed = matches!(event.state, winit::event::ElementState::Pressed);
                let winit::keyboard::PhysicalKey::Code(key_code) = event.physical_key else {
                    return Ok(winit::event_loop::ControlFlow::Poll);
                };
                use winit::keyboard::KeyCode as Kc;
                match key_code {
                    Kc::Escape if pressed => return Err(QuitEvent),
                    Kc::Backquote if pressed => self.toggle_mode(),
                    // F12 prints the current camera + window size as a
                    // ready-to-use `snapshot.ron` block, so the bin/snapshot
                    // tool can repro this exact view headlessly.
                    Kc::F12 if pressed => {
                        let pos = self.camera.pos;
                        let q = self.camera.rot.as_vector();
                        // Dump the full rotation quaternion (i, j, k, w),
                        // not just pos+forward — the snapshot tool can't
                        // reconstruct the camera roll about the forward axis
                        // from pos+forward alone, and the game's camera
                        // doesn't keep world-up = +Z (its up tracks the
                        // car/radial direction).
                        let block = format!(
                            "// Dumped via F12 from running game.\n(\n    pos: ({:.3}, {:.3}, {:.3}),\n    rot: ({:.5}, {:.5}, {:.5}, {:.5}),\n    fov_y: {:.3},\n    extent: ({}, {}),\n    output: \"snap.png\",\n)\n",
                            pos.x, pos.y, pos.z,
                            q.x, q.y, q.z, q.w,
                            self.camera.fov_y,
                            self.window_size.width, self.window_size.height,
                        );
                        println!("{block}");
                    }
                    _ => match self.mode {
                        Mode::Driving => self.on_drive_key(key_code, pressed),
                        Mode::Paused if pressed => {
                            // Fly-camera step per tap. Matches the prototype-era behavior.
                            let delta = 0.1;
                            self.camera.on_key(key_code, delta);
                        }
                        Mode::Paused => {}
                    },
                }
            }
            winit::event::WindowEvent::MouseWheel { delta, .. } if self.mode == Mode::Paused => {
                self.camera.on_wheel(delta);
            }
            winit::event::WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } if self.mode == Mode::Paused => {
                self.in_camera_drag = true;
            }
            winit::event::WindowEvent::MouseInput {
                state: winit::event::ElementState::Released,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                // Release the drag flag in any mode, so a press in pause + toggle to
                // drive doesn't leave a stuck drag.
                self.in_camera_drag = false;
            }
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                if self.in_camera_drag && self.mode == Mode::Paused {
                    self.camera.on_drag(
                        self.last_mouse_pos[0] as f32 - position.x as f32,
                        self.last_mouse_pos[1] as f32 - position.y as f32,
                    );
                }
                self.last_mouse_pos = [position.x as i32, position.y as i32];
            }
            winit::event::WindowEvent::CloseRequested => {
                return Err(QuitEvent);
            }
            winit::event::WindowEvent::RedrawRequested => {
                let wait = self.redraw();

                return Ok(
                    if let Some(repaint_after_instant) = time::Instant::now().checked_add(wait)
                    {
                        winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant)
                    } else {
                        winit::event_loop::ControlFlow::Wait
                    },
                );
            }
            _ => {}
        }

        Ok(winit::event_loop::ControlFlow::Poll)
    }
}

impl Drop for Game {
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }
        log::info!("Deinitializing");
        self.render.wait_for_gpu();
        self.terrain.free(self.render.context());
        self.car.chassis_instance.model.free(self.render.context());
        // Procedural wheel mesh is its own GPU buffer, separate from the
        // chassis model. All wheel_instances share one Arc<Model> built by
        // create_wheel_mesh_desc, so freeing through any of them releases
        // the buffer once for the whole set.
        if let Some(wheel_instance) = self.car.wheel_instances.iter().flatten().next() {
            wheel_instance.model.free(self.render.context());
        }
        self.snow.free(self.render.context());
        self.render.deinit();
    }
}

/// Pick `n` distinct canned radio lines, rotated by wall-clock seconds so
/// restarts hear a slightly different pair without any RNG dependency.
fn pick_radio_chatter(n: usize) -> Vec<&'static str> {
    let n = n.min(RADIO_CHATTER.len()).max(1);
    let tick = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as usize)
        .unwrap_or(0);
    let start = tick % RADIO_CHATTER.len();
    (0..n)
        .map(|i| RADIO_CHATTER[(start + i) % RADIO_CHATTER.len()])
        .collect()
}

fn main() {
    // env_logger honors RUST_LOG (default: off). Set RUST_LOG=info to see
    // startup, load, wasteland-radio chatter, mode-toggle, and drive lines.
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("console logger");
    }
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut game = Game::new(&event_loop);

    #[allow(deprecated)] //TODO
    event_loop
        .run(|event, target| match event {
            winit::event::Event::AboutToWait => {
                game.window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match game.on_event(&event) {
                Ok(control_flow) => {
                    target.set_control_flow(control_flow);
                }
                Err(QuitEvent) => {
                    target.exit();
                }
            },
            _ => {}
        })
        .unwrap();
}
