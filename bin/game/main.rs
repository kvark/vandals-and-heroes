use blade_graphics as gpu;
use vandals_and_heroes::{
    config, Camera, GeometryDesc, Loader, MaterialDesc, ModelDesc, ModelInstance, Physics,
    PhysicsBodyHandle, Recorder, Render, Terrain, TerrainBody, VertexDesc,
};

use nalgebra::Matrix4;
use std::{f32, fs, path, sync::Arc, thread, time};

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
    pub wheel_instances: Vec<ModelInstance>,
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

#[derive(Default)]
struct DriveInput {
    forward: bool,
    backward: bool,
    steer_left: bool,
    steer_right: bool,
    turbo: bool,
}

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
/// How far the chase camera sits behind/above the car along its horizontal
/// forward + radial-outward directions (equal → ~45° pitch).
const FOLLOW_DIST: f32 = 5.0;
/// Exponential rate at which the camera catches up to the computed follow
/// pose (per second). Higher = stiffer / more responsive; lower = floatier.
/// 8.0 closes ~99% of the gap in 0.5 s — visibly tracks the car without
/// snapping behind it on every sharp turn.
const CAMERA_FOLLOW_RATE: f32 = 8.0;
/// Damping factor applied to wheel motors when no drive command is active. High
/// enough that the motor brakes any wheel rotation toward zero, so the static
/// wheel-ground friction holds the chassis still on slopes.
const IDLE_BRAKE_FACTOR: f32 = 50.0;
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
/// this wide along the axle). Sized small enough that the wheel fits
/// inside the GLB chassis sockets (z = ±0.42 with a tire there).
const WHEEL_HALF_WIDTH: f32 = 0.04;
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
    // engine stuff
    #[allow(dead_code)] //TODO
    choir: choir::Choir,
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
    /// Debug snow: tiny rapier balls falling from the outer shell. Their
    /// landing pattern shows where the *physics* surface sits, exposing any
    /// mismatch with the visual heightmap.
    snow: snow::Snow,
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

        let config: config::Config = ron::de::from_bytes(
            &fs::read("data/config.ron").expect("Unable to open the main config"),
        )
        .expect("Unable to parse the main config");

        let choir = choir::Choir::default();
        let gpu_context = unsafe {
            gpu::Context::init(gpu::ContextDesc {
                presentation: true,
                validation: cfg!(debug_assertions),
                ..Default::default()
            })
        }
        .expect("Unable to initialize GPU");

        log::info!("Creating the window");
        let window_attributes = winit::window::Window::default_attributes()
            .with_title("Vandals and Heroes")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 800));
        #[allow(deprecated)] //TODO
        let window = event_loop.create_window(window_attributes).unwrap();
        let window_size = window.inner_size();
        let extent = gpu::Extent {
            width: window_size.width,
            height: window_size.height,
            depth: 1,
        };

        let gpu_surface = gpu_context.create_surface(&window).unwrap();
        let mut render = Render::new(gpu_context, gpu_surface, extent);
        render.set_ray_params(&config.ray);

        let mut loader = render.start_loading();

        let terrain = {
            log::info!("Loading map: {}", config.map);
            let map_path = path::PathBuf::from("data/maps").join(config.map);
            let mut map_config: config::Map = ron::de::from_bytes(
                &fs::read(map_path.join("map.ron")).expect("Unable to open the map config"),
            )
            .expect("Unable to parse the map config");

            let (texture, map_extent, height_alpha) = loader.load_png(&map_path.join("map.png"));

            if map_config.length == 0.0 {
                let circumference = 2.0 * f32::consts::PI * map_config.radius.start;
                map_config.length =
                    circumference * (map_extent.height as f32) / (map_extent.width as f32);
                log::info!("Derived map length to be {}", map_config.length);
            }

            let env_texture = config.environment.as_ref().map(|name| {
                let env_path = path::PathBuf::from("data/envs").join(format!("{}.png", name));
                log::info!("Loading environment: {}", env_path.display());
                loader.load_environment(&env_path)
            });

            (
                Terrain {
                    config: map_config,
                    texture,
                    env_texture,
                },
                map_extent,
                height_alpha,
            )
        };
        let (terrain, map_extent, height_alpha) = terrain;
        // Cylinder spawn keeps the historical "just below the outer cylinder"
        // height; the sphere samples the heightmap at the spawn (θ, v) and
        // lands ~1 m above the actual surface so the chassis isn't dropped in
        // from radius_end (where it would fall ~half the world's radial range).
        let spawn_radius = if terrain.config.is_sphere {
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
        let terrain_body = physics.create_terrain(
            &terrain.config,
            height_alpha,
            map_extent.width,
            map_extent.height,
        );

        let spawn_z = if terrain.config.is_sphere {
            // Sphere world: any axial offset puts the spawn off the equator
            // toward a pole, so just spawn on the +Y axis at the equator.
            0.0
        } else {
            0.1 * terrain.config.length
        };
        let car = Self::load_car(
            &mut loader,
            &mut physics,
            &config.car,
            nalgebra::Isometry3 {
                translation: nalgebra::Vector3::new(0.0, spawn_radius, spawn_z).into(),
                rotation: nalgebra::UnitQuaternion::from_axis_angle(
                    &nalgebra::Vector3::y_axis(),
                    0.5 * f32::consts::PI,
                ),
            },
        );

        // Debug snow particles: 200 little balls falling from the outer shell.
        // Built here so the procedural mesh upload joins the same loader
        // submission as the car GLB and terrain texture. For long cylinder
        // worlds we bias spawn to a band around the car's z so the camera
        // always sees some snow.
        let snow = snow::Snow::new(
            &mut loader,
            &mut physics,
            200,
            terrain.config.is_sphere,
            terrain.config.radius.end,
            spawn_z,
        );

        let submission = loader.finish();
        render.accept_submission(submission);
        render.wait_for_gpu();
        render.set_shadow_extent(map_extent);

        // Camera clip-far has to cover the far side of the world. The cylinder
        // is bounded by its length along Z; the sphere by its diameter.
        let clip_far = if terrain.config.is_sphere {
            4.0 * terrain.config.radius.end
        } else {
            terrain.config.length
        };
        let camera = Camera {
            pos: nalgebra::Vector3::new(0.0, spawn_radius + 0.5, spawn_z),
            rot: nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::x_axis(),
                0.3 * f32::consts::PI,
            ),
            clip: 1.0..clip_far,
            ..Default::default()
        };

        let recorder = config.record.as_ref().map(Recorder::new);

        log::info!(
            "Ready. Mode: Driving. Controls: WASD drive, Space jump, LShift turbo, ~ pause, Esc quit"
        );

        Self {
            choir,
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
            snow,
        }
    }

    fn load_car(
        loader: &mut Loader,
        physics: &mut Physics,
        car_path: &str,
        transform: nalgebra::Isometry3<f32>,
    ) -> Object {
        log::info!("Loading car: {}", car_path);
        let car_path = path::PathBuf::from("data/cars").join(car_path);
        let car_config: config::Car = ron::de::from_bytes(
            &fs::read(car_path.join("car.ron")).expect("Unable to open the car config"),
        )
        .expect("Unable to parse the car config");
        let model_desc = Loader::read_gltf(
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
                        .motor_model(JointAxis::AngY, MotorModel::AccelerationBased)
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
                    is_steering,
                }
            })
            .collect();

        let chassis_instance = ModelInstance {
            model: Arc::new(model),
            transform,
            geometry_filter: None,
        };

        // Procedural wheel mesh, used for every physics wheel. The wheel
        // rigid body's spin axis is chassis-local Z (the joint axle), so
        // the cylinder's axis matches local Z too — rotation of the body
        // about its local Z visibly spins the mesh; rotation about local Y
        // (steering) visibly turns it. Matching the body convention is what
        // lets the player see the steering response.
        let wheel_radius = car_config.wheels.first().map(|w| w.radius).unwrap_or(0.15);
        let wheel_model_desc = create_wheel_mesh_desc(wheel_radius, WHEEL_HALF_WIDTH);
        let wheel_model = Arc::new(loader.load_model(&wheel_model_desc));
        let wheel_instances: Vec<ModelInstance> = wheels
            .iter()
            .map(|w| {
                let pose = physics.get_transform(w.rigid_body);
                ModelInstance {
                    model: wheel_model.clone(),
                    transform: pose,
                    geometry_filter: None,
                }
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
            aabb.mins.x, aabb.maxs.x, aabb.mins.y, aabb.maxs.y, aabb.mins.z, aabb.maxs.z,
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
            .apply_axial_angular_damping(self.car.rigid_body, 0.15, 2.0);
        // apply_driving_input must run AFTER update_gravity because the latter
        // calls rb.reset_forces, which would wipe out any drive force we added.
        self.apply_driving_input();
        self.physics.step();
        self.car.chassis_instance.transform = self.physics.get_transform(self.car.rigid_body);
        // Per-physics-wheel transform sync so the procedural cylinder meshes
        // visibly spin (AngZ) and turn (AngY) with their rigid bodies.
        for (wi, w) in self.car.wheels.iter().enumerate() {
            if let Some(inst) = self.car.wheel_instances.get_mut(wi) {
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
        let drive_v = throttle * max_v * turbo;
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

    /// True if any wheel is currently in contact with the terrain heightfield.
    /// Used as the gate for starting a jump charge and for actually firing.
    fn chassis_grounded(&self) -> bool {
        self.car.wheels.iter().any(|w| {
            self.physics
                .is_touching_terrain(w.rigid_body, &self.terrain_body)
        })
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
        if let Some(start) = self.jump_charge_start {
            if time::Instant::now() - start >= JUMP_MAX_CHARGE {
                self.execute_jump(JUMP_MAX_CHARGE);
                self.jump_charge_start = None;
            }
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

        // Detect upside-down. World "up" is radial-outward from the
        // gravitational centre (sphere origin, or the cylinder's Z axis).
        // Compare it to the chassis +Y direction: if they're on opposite
        // sides we're upside-down and the impulse should originate from the
        // *cabin* (chassis +Y_max) pushing the body away from the ground
        // it's resting on, instead of from the wheels.
        let xform = self.physics.get_transform(self.car.rigid_body);
        let car_pos = xform.translation.vector;
        let world_up = if self.terrain_body.is_sphere {
            car_pos.normalize()
        } else {
            let xy = nalgebra::Vector3::new(car_pos.x, car_pos.y, 0.0);
            xy.normalize()
        };
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

    fn follow_camera(&mut self, dt: time::Duration) {
        let xform = &self.car.chassis_instance.transform;
        let car_pos = xform.translation.vector;
        // "Up" is radially outward from the world centre — the Z axis for the
        // cylinder, the origin for the sphere. Gravity points the opposite way
        // (see Physics::update_gravity), so this matches the player's intuition
        // of "up away from the planet" in both world types.
        let mut up = if self.terrain_body.is_sphere {
            car_pos
        } else {
            nalgebra::Vector3::new(car_pos.x, car_pos.y, 0.0)
        };
        let up_len = up.norm();
        up = if up_len < 1e-6 {
            nalgebra::Vector3::y()
        } else {
            up / up_len
        };
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
        // Equal back-offset and up-offset gives roughly 45° look-down.
        let target_pos = car_pos - forward * FOLLOW_DIST + up * FOLLOW_DIST;
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

    fn on_drive_key(&mut self, code: winit::keyboard::KeyCode, pressed: bool) {
        use winit::keyboard::KeyCode as Kc;
        match code {
            Kc::KeyW => self.input.forward = pressed,
            Kc::KeyS => self.input.backward = pressed,
            Kc::KeyA => self.input.steer_left = pressed,
            Kc::KeyD => self.input.steer_right = pressed,
            Kc::ShiftLeft => self.input.turbo = pressed,
            Kc::Space => self.handle_jump_key(pressed),
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

    fn redraw(&mut self) -> time::Duration {
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
        model_instances.extend(self.car.wheel_instances.iter());
        model_instances.extend(self.snow.instances.iter());
        self.render
            .draw(&self.camera, &self.terrain, &model_instances);

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
                    if let Some(repaint_after_instant) = std::time::Instant::now().checked_add(wait)
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
        self.terrain.texture.deinit(self.render.context());
        if let Some(env) = self.terrain.env_texture.as_ref() {
            env.deinit(self.render.context());
        }
        self.car.chassis_instance.model.free(self.render.context());
        // Procedural wheel mesh is its own GPU buffer, separate from the
        // chassis model. All wheel_instances share one Arc<Model> built by
        // create_wheel_mesh_desc, so freeing through any of them releases
        // the buffer once for the whole set.
        if let Some(wheel_instance) = self.car.wheel_instances.first() {
            wheel_instance.model.free(self.render.context());
        }
        self.snow.free(self.render.context());
        self.render.deinit();
    }
}

fn main() {
    // env_logger honors RUST_LOG (default: off). Set RUST_LOG=info to see
    // startup, load, mode-toggle, and drive-input/drive-cmd lines.
    env_logger::init();
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
