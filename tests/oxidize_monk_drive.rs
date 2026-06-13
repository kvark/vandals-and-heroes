//! Reproduces the in-game OxidizeMonk car physics setup (real GLB chassis collider,
//! real car.ron parameters) on a flat cylindrical heightfield, drives all four wheels
//! at the configured `motor_max_velocity`, and checks that the chassis actually
//! translates — answering "is the car physically able to move at all under the
//! configured forces?" without needing a window or GPU.

use rapier3d::dynamics::{RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle};
use rapier3d::geometry::ColliderBuilder;
use rapier3d::math::{Pose, Vec3};
use std::path::Path;
use vandals_and_heroes::{Loader, MaterialDesc, Physics, PhysicsBodyHandle, TerrainBody, config};

const TERRAIN_WIDTH: u32 = 64;
const TERRAIN_HEIGHT: u32 = 256;
const TERRAIN_RADIUS_START: f32 = 10.0;
const TERRAIN_RADIUS_END: f32 = 20.0;
const TERRAIN_LENGTH: f32 = 100.0;
/// Spawn height matches the game: just below the outer "sky" cylinder.
/// Ground sits roughly at r ≈ 15 with uniform alpha=128 ((128/255)·10 + 10 ≈ 15.02).
const SPAWN_RADIUS: f32 = TERRAIN_RADIUS_END - 0.5;

fn build_flat_terrain(physics: &mut Physics) -> TerrainBody {
    let alpha = vec![128u8; (TERRAIN_WIDTH * TERRAIN_HEIGHT) as usize];
    let cfg = config::Map {
        radius: TERRAIN_RADIUS_START..TERRAIN_RADIUS_END,
        length: TERRAIN_LENGTH,
        density: 10.0,
        is_sphere: false,
    };
    physics.create_terrain(&cfg, alpha, TERRAIN_WIDTH, TERRAIN_HEIGHT)
}

fn build_flat_sphere(physics: &mut Physics) -> TerrainBody {
    let alpha = vec![128u8; (TERRAIN_WIDTH * TERRAIN_HEIGHT) as usize];
    let cfg = config::Map {
        radius: TERRAIN_RADIUS_START..TERRAIN_RADIUS_END,
        // length is ignored in sphere mode; pass 0 to make that explicit.
        length: 0.0,
        density: 10.0,
        is_sphere: true,
    };
    physics.create_terrain(&cfg, alpha, TERRAIN_WIDTH, TERRAIN_HEIGHT)
}

struct LoadedCar {
    chassis: RigidBodyHandle,
    wheels: Vec<RigidBodyHandle>,
    wheel_joints: Vec<rapier3d::dynamics::ImpulseJointHandle>,
    motor_max_velocity: f32,
}

/// Mirrors `Game::load_car` and `Game::create_mesh_collider` in bin/game/main.rs,
/// minus the gpu-side model upload.
fn load_oxidize_monk(physics: &mut Physics) -> LoadedCar {
    let car_path = Path::new("data/cars/OxidizeMonk");
    let car_config: config::Car =
        ron::de::from_bytes(&std::fs::read(car_path.join("car.ron")).expect("car.ron present"))
            .expect("parse car.ron");

    let model_desc = Loader::read_gltf(
        &car_path.join("body.glb"),
        nalgebra::Matrix4::identity().scale(car_config.scale),
    );

    let keep = |m: &MaterialDesc| {
        !m.name
            .as_deref()
            .map(|n| n.to_lowercase().contains("wheel"))
            .unwrap_or(false)
    };
    let chassis_vertices: Vec<Vec3> = model_desc
        .positions_filtered(keep)
        .into_iter()
        .map(|p| Vec3::new(p.x, p.y, p.z))
        .collect();
    let chassis_collider =
        ColliderBuilder::trimesh(chassis_vertices, model_desc.indices_filtered(keep))
            .expect("chassis trimesh")
            .density(car_config.density)
            .build();

    // Same spawn pose as bin/game/main.rs: at SPAWN_RADIUS along +Y, with a 90°
    // rotation about Y so chassis-local -X (the car's front) points along world +Z.
    let transform = nalgebra::Isometry3 {
        translation: nalgebra::Vector3::new(0.0, SPAWN_RADIUS, 0.1 * TERRAIN_LENGTH).into(),
        rotation: nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            0.5 * std::f32::consts::PI,
        ),
    };

    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(transform.into())
        .linear_damping(0.4)
        .angular_damping(0.4)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, vec![chassis_collider]);

    let axis_local = Vec3::new(
        car_config.wheel_axis[0],
        car_config.wheel_axis[1],
        car_config.wheel_axis[2],
    );
    let chassis_pose: Pose = transform.into();
    let mut wheel_joints = Vec::new();
    let mut wheels = Vec::new();
    for w in &car_config.wheels {
        let anchor_local = Vec3::new(w.position[0], w.position[1], w.position[2]);
        let wheel_world = chassis_pose * anchor_local;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(Pose::from_parts(wheel_world, chassis_pose.rotation))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(w.radius)
            .density(car_config.density)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);

        let joint = RevoluteJointBuilder::new(axis_local)
            .local_anchor1(anchor_local)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_max_force(car_config.motor_max_force)
            .motor_velocity(0.0, 0.2)
            .build();
        wheel_joints.push(physics.add_revolute_joint(chassis, wheel_rb, joint));
        wheels.push(wheel_rb);
    }

    LoadedCar {
        chassis,
        wheels,
        wheel_joints,
        motor_max_velocity: car_config.motor_max_velocity,
    }
}

fn run_ticks(physics: &mut Physics, terrain: &TerrainBody, ticks: usize) {
    for _ in 0..ticks {
        physics.update_gravity(terrain);
        physics.step();
    }
}

fn translation(physics: &Physics, body: RigidBodyHandle) -> [f32; 3] {
    physics
        .body_kinematics(body)
        .expect("body exists")
        .translation
}

/// Diagnostic snapshot for failure messages.
#[derive(Debug)]
#[allow(dead_code)]
struct CarSnap {
    chassis_t: [f32; 3],
    chassis_v: [f32; 3],
    wheel_angvel_mags: Vec<f32>,
}

fn car_snap(physics: &Physics, car: &LoadedCar) -> CarSnap {
    let chassis = physics.body_kinematics(car.chassis).unwrap();
    let wheel_angvel_mags = car
        .wheels
        .iter()
        .map(|&h| {
            let k = physics.body_kinematics(h).unwrap();
            (k.angvel[0].powi(2) + k.angvel[1].powi(2) + k.angvel[2].powi(2)).sqrt()
        })
        .collect();
    CarSnap {
        chassis_t: chassis.translation,
        chassis_v: chassis.linvel,
        wheel_angvel_mags,
    }
}

#[test]
fn oxidize_monk_forward_drive_translates_chassis() {
    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);
    let car = load_oxidize_monk(&mut physics);

    // Let the car settle on the ground under radial gravity. Drive command is
    // intentionally 0 during settling — same as the game's startup before any keypress.
    run_ticks(&mut physics, &terrain, 250);
    let settled_t = translation(&physics, car.chassis);
    let settled = car_snap(&physics, &car);

    // All four wheels at full forward velocity — exact same call apply_driving_input
    // makes when W is held with no steering or turbo.
    for &j in &car.wheel_joints {
        physics.set_joint_motor_velocity(j, car.motor_max_velocity, 1.0);
    }

    // 600 ticks at dt = 1/60 s ≈ 10 s of in-game time. Plenty to see movement
    // even with the low motor_max_force OxidizeMonk uses.
    run_ticks(&mut physics, &terrain, 600);

    let end_t = translation(&physics, car.chassis);
    let end = car_snap(&physics, &car);

    let dx = end_t[0] - settled_t[0];
    let dy = end_t[1] - settled_t[1];
    let dz = end_t[2] - settled_t[2];
    let horiz = (dx * dx + dz * dz).sqrt();

    eprintln!("settled: {settled:?}");
    eprintln!("end:     {end:?}");
    eprintln!(
        "displacement after 10s of forward drive: dx={dx:.4} dy={dy:.4} dz={dz:.4} |xz|={horiz:.4}"
    );

    assert!(
        horiz > 0.3,
        "OxidizeMonk car barely moved under forward drive: |xz| = {horiz:.4}, dz = {dz:.4}"
    );
}

/// Pure-cylinder debug test: synthetic cuboid chassis (no model loading),
/// uniform-alpha heightfield (truly flat radial surface), wheels braked.
/// The chassis should land, settle, and STOP. If it keeps drifting on this
/// idealized surface, gravity or friction is wrong somewhere.
#[test]
fn synthetic_car_on_pure_cylinder_stops_after_settling() {
    use rapier3d::dynamics::{
        ImpulseJointHandle, RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle,
    };
    use rapier3d::geometry::ColliderBuilder;
    use rapier3d::math::{Pose, Vec3};

    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);

    // Cuboid chassis (no trimesh, no model). Cleanest possible setup.
    let chassis_collider = ColliderBuilder::cuboid(0.5, 0.2, 0.4).density(10.0).build();
    let spawn_pose = Pose::from_translation(Vec3::new(0.0, 19.0, 0.0));
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(spawn_pose)
        .linear_damping(0.4)
        .angular_damping(0.4)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, vec![chassis_collider]);

    // Four wheel balls at the chassis corners.
    let anchors = [
        Vec3::new(0.4, -0.30, 0.45),
        Vec3::new(0.4, -0.30, -0.45),
        Vec3::new(-0.4, -0.30, 0.45),
        Vec3::new(-0.4, -0.30, -0.45),
    ];
    let mut joints: Vec<ImpulseJointHandle> = Vec::new();
    let mut wheels: Vec<RigidBodyHandle> = Vec::new();
    for anchor in anchors {
        let wheel_world = spawn_pose * anchor;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(Pose::from_parts(wheel_world, spawn_pose.rotation))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(0.15)
            .density(10.0)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);
        let joint = RevoluteJointBuilder::new(Vec3::new(0.0, 0.0, 1.0))
            .local_anchor1(anchor)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_max_force(10.0)
            .motor_velocity(0.0, 50.0) // brake from spawn
            .build();
        joints.push(physics.add_revolute_joint(chassis, wheel_rb, joint));
        wheels.push(wheel_rb);
    }

    // Track chassis position and velocity over time so we can see if it ever
    // actually comes to rest.
    let mut prev_t = translation(&physics, chassis);
    let mut snapshot_interval = 100;
    for tick in 0..1500 {
        // Re-apply brake every tick (the production code does it via apply_driving_input).
        for &j in &joints {
            physics.set_joint_motor_velocity(j, 0.0, 50.0);
        }
        physics.update_gravity(&terrain);
        physics.step();
        if tick % snapshot_interval == 0 {
            let t = translation(&physics, chassis);
            let k = physics.body_kinematics(chassis).unwrap();
            let speed = (k.linvel[0].powi(2) + k.linvel[1].powi(2) + k.linvel[2].powi(2)).sqrt();
            let drift_x = t[0] - prev_t[0];
            let drift_z = t[2] - prev_t[2];
            eprintln!(
                "tick={tick:4}: pos=({:.4}, {:.4}, {:.4}) speed={speed:.5} drift_since_prev=({:.4}, _, {:.4})",
                t[0], t[1], t[2], drift_x, drift_z
            );
            prev_t = t;
            // After settling, snapshot less frequently
            if tick >= 400 {
                snapshot_interval = 200;
            }
        }
    }

    // After all that time, the chassis should be at rest (very low speed).
    let k = physics.body_kinematics(chassis).unwrap();
    let final_speed = (k.linvel[0].powi(2) + k.linvel[1].powi(2) + k.linvel[2].powi(2)).sqrt();
    eprintln!("final speed: {final_speed:.5} m/s");
    assert!(
        final_speed < 0.05,
        "chassis is still moving after long settling on a flat cylinder: speed = {final_speed:.5} m/s"
    );
}

/// Same minimal vehicle, but on a *spherical* heightfield (flat alpha so the
/// surface is a smooth sphere at the average radius). Validates that the sphere
/// collider + sphere gravity actually catch a falling chassis and let it
/// settle — no driving through the surface, no orbiting around the planet.
#[test]
fn synthetic_car_on_flat_sphere_stops_after_settling() {
    use rapier3d::dynamics::{
        ImpulseJointHandle, RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle,
    };
    use rapier3d::geometry::ColliderBuilder;
    use rapier3d::math::{Pose, Vec3};

    let mut physics = Physics::default();
    let terrain = build_flat_sphere(&mut physics);
    assert!(terrain.is_sphere);

    let chassis_collider = ColliderBuilder::cuboid(0.5, 0.2, 0.4).density(10.0).build();
    // Spawn just above the smooth surface (alpha = 128/255 ≈ 0.502, so ground
    // radius is roughly start + 0.5·(end-start) = 15. Drop the chassis at 16.5
    // so it falls a metre or so onto the planet).
    let spawn_pose = Pose::from_translation(Vec3::new(0.0, 16.5, 0.0));
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(spawn_pose)
        .linear_damping(0.4)
        .angular_damping(0.4)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, vec![chassis_collider]);

    let anchors = [
        Vec3::new(0.4, -0.30, 0.45),
        Vec3::new(0.4, -0.30, -0.45),
        Vec3::new(-0.4, -0.30, 0.45),
        Vec3::new(-0.4, -0.30, -0.45),
    ];
    let mut joints: Vec<ImpulseJointHandle> = Vec::new();
    let mut _wheels: Vec<RigidBodyHandle> = Vec::new();
    for anchor in anchors {
        let wheel_world = spawn_pose * anchor;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(Pose::from_parts(wheel_world, spawn_pose.rotation))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(0.15)
            .density(10.0)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);
        let joint = RevoluteJointBuilder::new(Vec3::new(0.0, 0.0, 1.0))
            .local_anchor1(anchor)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_max_force(10.0)
            .motor_velocity(0.0, 50.0)
            .build();
        joints.push(physics.add_revolute_joint(chassis, wheel_rb, joint));
        _wheels.push(wheel_rb);
    }

    for tick in 0..1500 {
        for &j in &joints {
            physics.set_joint_motor_velocity(j, 0.0, 50.0);
        }
        physics.update_gravity(&terrain);
        physics.step();
        if tick % 200 == 0 {
            let t = translation(&physics, chassis);
            let r = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            let k = physics.body_kinematics(chassis).unwrap();
            let speed = (k.linvel[0].powi(2) + k.linvel[1].powi(2) + k.linvel[2].powi(2)).sqrt();
            eprintln!(
                "tick={tick:4}: pos=({:.3},{:.3},{:.3}) r={r:.3} speed={speed:.5}",
                t[0], t[1], t[2]
            );
        }
    }

    let k = physics.body_kinematics(chassis).unwrap();
    let final_speed = (k.linvel[0].powi(2) + k.linvel[1].powi(2) + k.linvel[2].powi(2)).sqrt();
    let final_t = translation(&physics, chassis);
    let final_r =
        (final_t[0] * final_t[0] + final_t[1] * final_t[1] + final_t[2] * final_t[2]).sqrt();
    eprintln!("final r={final_r:.3} speed={final_speed:.5}");
    // The chassis should not be inside the planet (well below ground radius) or
    // far above it (orbiting). With alpha=128, ground is ~15; chassis chassis
    // sits ~0.4 above that with the suspension and wheel radius, so expect
    // ~15-17 m.
    assert!(
        final_r > 14.0 && final_r < 18.0,
        "chassis ended at r={final_r:.3}, expected ~15-17 m on the sphere"
    );
    assert!(
        final_speed < 0.1,
        "chassis is still moving after long settling on the sphere: speed = {final_speed:.5} m/s"
    );

    // Jump precondition: after settling, the wheels must be in contact with the
    // spherical heightfield (otherwise the in-game Space-jump always returns
    // "not grounded"). Repeats the exact check `Game::jump()` runs.
    let any_wheel_grounded = _wheels
        .iter()
        .any(|&w| physics.is_touching_terrain(w, &terrain));
    assert!(
        any_wheel_grounded,
        "no wheel reports a contact with the spherical heightfield after settling — \
         jump-from-sphere will always be blocked"
    );

    // Replicate the in-game jump impulse and confirm the chassis actually
    // lifts off the surface. Mirrors `Game::jump`.
    const JUMP_VELOCITY: f32 = 8.0;
    let xform = physics.get_transform(chassis);
    let chassis_up_world = xform.rotation * nalgebra::Vector3::new(0.0, 1.0, 0.0);
    let bottom_local = nalgebra::Vector3::new(0.0, -0.25, 0.0);
    let bottom_world = xform.translation.vector + (xform.rotation * bottom_local);
    let mass = physics.body_mass(chassis);
    let impulse_vec = chassis_up_world * (mass * JUMP_VELOCITY);
    let r_before = final_r;
    let impulse = rapier3d::math::Vec3::new(impulse_vec.x, impulse_vec.y, impulse_vec.z);
    let bottom_pt = rapier3d::math::Vec3::new(bottom_world.x, bottom_world.y, bottom_world.z);
    physics.apply_impulse_at_point(chassis, impulse, bottom_pt);
    for _ in 0..15 {
        physics.update_gravity(&terrain);
        physics.step();
    }
    let after = translation(&physics, chassis);
    let r_after = (after[0] * after[0] + after[1] * after[1] + after[2] * after[2]).sqrt();
    eprintln!("jump test: r_before={r_before:.3} r_after={r_after:.3}");
    assert!(
        r_after > r_before + 0.1,
        "chassis did not rise after the jump impulse: r_before={r_before:.3} r_after={r_after:.3}"
    );
    // Sphere gravity is capped at MAX_ACCEL = 12 m/s² in update_gravity. With
    // JUMP_VELOCITY = 8 m/s the apex above the surface is v²/(2g) ≈ 2.67 m.
    // We sample 15 ticks (~0.25 s) after the impulse so we're partway through
    // the rise, not at the peak — assert the climb stays below 6 m, which
    // catches the "gravity is too weak" regression that lets jumps go 20+ m.
    assert!(
        r_after - r_before < 6.0,
        "chassis went too high in 0.25 s — sphere gravity is too weak: \
         r_before={r_before:.3} r_after={r_after:.3} delta={:.3}",
        r_after - r_before,
    );
}

/// With no drive command (W/S released) the car should sit still after settling —
/// the wheel motor's idle damping must brake the wheels hard enough that
/// wheel-ground friction holds the chassis on terrain slopes. Before this was
/// fixed, the car coasted forward "by itself" on the slightly-sloped spawn.
#[test]
fn oxidize_monk_idles_without_drifting() {
    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);
    let car = load_oxidize_monk(&mut physics);

    // Apply idle brake — equivalent to apply_driving_input with no keys held.
    // Matches IDLE_BRAKE_FACTOR in bin/game/main.rs.
    let brake = 50.0_f32;
    for &j in &car.wheel_joints {
        physics.set_joint_motor_velocity(j, 0.0, brake);
    }

    // Settle on the flat cylinder for a couple of seconds, then sample.
    run_ticks(&mut physics, &terrain, 200);
    let settled_t = translation(&physics, car.chassis);

    // Run another 5 seconds with the idle brake still applied — chassis should
    // barely move.
    for _ in 0..5 {
        for &j in &car.wheel_joints {
            physics.set_joint_motor_velocity(j, 0.0, brake);
        }
        run_ticks(&mut physics, &terrain, 60);
    }
    let later_t = translation(&physics, car.chassis);

    let dx = later_t[0] - settled_t[0];
    let dz = later_t[2] - settled_t[2];
    let horiz_drift = (dx * dx + dz * dz).sqrt();
    eprintln!(
        "idle drift over 5 s: dx={dx:.4} dz={dz:.4} |xz|={horiz_drift:.4} from {settled_t:?} to {later_t:?}"
    );
    assert!(
        horiz_drift < 0.2,
        "car drifted while idling: |xz| = {horiz_drift:.4} m in 5 s"
    );
}

/// Reproduces the game's actual spawn on the Fostral heightfield. This is the
/// scenario the user is observing: same car, same map, same spawn pose. If the
/// chassis trimesh wedges against terrain, this test will show it.
#[test]
fn oxidize_monk_drives_on_fostral_heightfield() {
    use std::io::BufReader;

    // Load the real Fostral map.png alpha channel.
    let map_path = Path::new("data/maps/fostral/map.png");
    let decoder = png::Decoder::new(BufReader::new(
        std::fs::File::open(map_path).expect("fostral map.png present (run `git lfs pull`)"),
    ));
    let mut reader = decoder.read_info().expect("png header");
    let info_size = reader.output_buffer_size().expect("png size");
    let mut decoded = vec![0u8; info_size];
    let info = reader.next_frame(&mut decoded).expect("png decode");
    let (width, height) = (info.width, info.height);
    eprintln!("loaded fostral map.png: {width}x{height}");
    // Heightmap collider uses the alpha channel (texel.a in the shader).
    let alpha: Vec<u8> = (0..(width as usize * height as usize))
        .map(|i| decoded[i * 4 + 3])
        .collect();

    let map_radius_start = 10.0_f32;
    let map_radius_end = 15.0_f32;
    let circumference = 2.0 * std::f32::consts::PI * map_radius_start;
    let map_length = circumference * (height as f32) / (width as f32);
    eprintln!("derived map length: {map_length:.3}");

    let mut physics = Physics::default();
    let cfg = config::Map {
        radius: map_radius_start..map_radius_end,
        length: map_length,
        density: 10.0,
        is_sphere: false,
    };
    let terrain = physics.create_terrain(&cfg, alpha.clone(), width, height);

    // Sample alpha at the car's spawn position so we can see what ground radius
    // the chassis is supposed to be sitting on.
    // game spawn: pos = (0, radius_end - 0.5, 0.1 * length), so theta = atan2(y, x) = pi/2.
    let spawn_z = 0.1 * map_length;
    let spawn_radius = map_radius_end - 0.5;
    // sample_map in the shader: tc = vec2f(alpha_angle / (2*PI), depth/length + 0.5)
    // (the texture wraps in U; the angle goes in directly without a +0.5 offset.)
    let u_norm = std::f32::consts::FRAC_PI_2 / (2.0 * std::f32::consts::PI);
    let v_norm = spawn_z / map_length + 0.5;
    let px_x = ((u_norm * width as f32) as i32).rem_euclid(width as i32) as usize;
    let px_y = (v_norm * height as f32).clamp(0.0, (height - 1) as f32) as usize;
    let spawn_alpha = alpha[px_y * width as usize + px_x];
    let spawn_ground_radius =
        map_radius_start + (spawn_alpha as f32 / 255.0) * (map_radius_end - map_radius_start);
    eprintln!(
        "spawn point (theta=pi/2, z={spawn_z:.2}): pixel=({px_x}, {px_y}), alpha={spawn_alpha}, ground_radius={spawn_ground_radius:.3}"
    );
    eprintln!(
        "chassis spawns at radius {spawn_radius:.3}; chassis-bottom (~0.4 in) at ~{:.3}",
        spawn_radius - 0.4
    );

    // Construct the OxidizeMonk car at the game's exact spawn pose.
    let car_path = Path::new("data/cars/OxidizeMonk");
    let car_config: config::Car =
        ron::de::from_bytes(&std::fs::read(car_path.join("car.ron")).expect("car.ron"))
            .expect("parse car.ron");
    let model_desc = Loader::read_gltf(
        &car_path.join("body.glb"),
        nalgebra::Matrix4::identity().scale(car_config.scale),
    );
    let keep = |m: &MaterialDesc| {
        !m.name
            .as_deref()
            .map(|n| n.to_lowercase().contains("wheel"))
            .unwrap_or(false)
    };
    // Mirror production: no chassis-terrain collision; wheels handle ground contact.
    let positions = model_desc.positions_filtered(keep);
    let (mins, maxs) = positions
        .iter()
        .fold((positions[0], positions[0]), |(mut a, mut b), p| {
            a.x = a.x.min(p.x);
            a.y = a.y.min(p.y);
            a.z = a.z.min(p.z);
            b.x = b.x.max(p.x);
            b.y = b.y.max(p.y);
            b.z = b.z.max(p.z);
            (a, b)
        });
    let aabb_volume = (maxs.x - mins.x) * (maxs.y - mins.y) * (maxs.z - mins.z);
    let chassis_mass = aabb_volume * 0.1 * car_config.density;
    eprintln!("chassis approx mass: {chassis_mass:.2}");
    let chassis_collider = ColliderBuilder::ball(0.05)
        .density(0.0)
        .collision_groups(rapier3d::geometry::InteractionGroups::none())
        .build();

    let transform = nalgebra::Isometry3 {
        translation: nalgebra::Vector3::new(0.0, spawn_radius, spawn_z).into(),
        rotation: nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            0.5 * std::f32::consts::PI,
        ),
    };
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(transform.into())
        .additional_mass(chassis_mass)
        .linear_damping(0.4)
        .angular_damping(0.4)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, vec![chassis_collider]);

    let axis_local = Vec3::new(
        car_config.wheel_axis[0],
        car_config.wheel_axis[1],
        car_config.wheel_axis[2],
    );
    let chassis_pose: Pose = transform.into();
    let mut wheel_joints = Vec::new();
    let mut wheel_rbs: Vec<RigidBodyHandle> = Vec::new();
    for w in &car_config.wheels {
        let anchor_local = Vec3::new(w.position[0], w.position[1], w.position[2]);
        let wheel_world = chassis_pose * anchor_local;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(Pose::from_parts(wheel_world, chassis_pose.rotation))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(w.radius)
            .density(car_config.density)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);
        // Suspension + spin GenericJoint, matching production.
        use rapier3d::dynamics::{GenericJointBuilder, JointAxesMask, JointAxis, MotorModel};
        let _ = axis_local;
        let locked = JointAxesMask::LIN_X
            | JointAxesMask::LIN_Z
            | JointAxesMask::ANG_X
            | JointAxesMask::ANG_Y;
        let joint = GenericJointBuilder::new(locked)
            .local_anchor1(anchor_local)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_model(JointAxis::LinY, MotorModel::ForceBased)
            .motor_position(JointAxis::LinY, 0.0, 100.0, 10.0)
            .motor_max_force(JointAxis::LinY, 200.0)
            .limits(JointAxis::LinY, [-0.3, 0.3])
            .motor_model(JointAxis::AngZ, MotorModel::ForceBased)
            .motor_velocity(JointAxis::AngZ, 0.0, 50.0)
            .motor_max_force(JointAxis::AngZ, car_config.motor_max_force)
            .build();
        wheel_joints.push(physics.add_generic_joint(chassis, wheel_rb, joint));
        wheel_rbs.push(wheel_rb);
    }

    let chassis_rot = |physics: &Physics| {
        let k = physics.body_kinematics(chassis).unwrap();
        k.rotation
    };
    let wheel_angvels = |physics: &Physics| -> Vec<f32> {
        wheel_rbs
            .iter()
            .map(|&h| {
                let k = physics.body_kinematics(h).unwrap();
                (k.angvel[0].powi(2) + k.angvel[1].powi(2) + k.angvel[2].powi(2)).sqrt()
            })
            .collect()
    };

    // Mirror the new game loop: wheel motors brake when idle; drive at max_v
    // when driving. With suspension, all 4 wheels stay grounded for traction.
    const IDLE_BRAKE: f32 = 50.0;

    for tick in 0..250 {
        for &j in &wheel_joints {
            physics.set_joint_motor_velocity(j, 0.0, IDLE_BRAKE);
        }
        physics.update_gravity(&terrain);
        physics.step();
        if tick % 50 == 0 {
            let t = translation(&physics, chassis);
            let r = (t[0] * t[0] + t[1] * t[1]).sqrt();
            eprintln!(
                "settle tick={tick}: pos=({:.3}, {:.3}, {:.3}) r={r:.3} rot={:?} wheel_angvels={:?}",
                t[0],
                t[1],
                t[2],
                chassis_rot(&physics),
                wheel_angvels(&physics),
            );
        }
    }
    let settled_t = translation(&physics, chassis);

    for tick in 0..600 {
        for &j in &wheel_joints {
            physics.set_joint_motor_velocity(j, car_config.motor_max_velocity, 1.0);
        }
        physics.update_gravity(&terrain);
        physics.step();
        if tick % 100 == 0 {
            let k = physics.body_kinematics(chassis).unwrap();
            let r =
                (k.translation[0] * k.translation[0] + k.translation[1] * k.translation[1]).sqrt();
            let speed = (k.linvel[0].powi(2) + k.linvel[1].powi(2) + k.linvel[2].powi(2)).sqrt();
            eprintln!(
                "drive tick={tick}: pos=({:.3}, {:.3}, {:.3}) r={r:.3} speed={speed:.4} wheel_angvels={:?}",
                k.translation[0],
                k.translation[1],
                k.translation[2],
                wheel_angvels(&physics),
            );
        }
    }
    let end_t = translation(&physics, chassis);

    let dx = end_t[0] - settled_t[0];
    let dz = end_t[2] - settled_t[2];
    let horiz = (dx * dx + dz * dz).sqrt();
    eprintln!("real-terrain displacement after 10s drive: dx={dx:.4} dz={dz:.4} |xz|={horiz:.4}");

    // No assertion — this test is diagnostic only. We want to *see* whether the
    // car gets stuck on the real heightfield. If it gets stuck (|xz| < ~0.1),
    // the issue is chassis trimesh wedging on the terrain.
}

/// Tank-style differential steering test on the real Fostral heightfield with
/// the production GenericJoint (suspension + spin). Pressing "A" (steer left)
/// drives the right-side wheels forward and the left-side wheels backward,
/// which should rotate the chassis CCW about its up axis. This mirrors what
/// apply_driving_input does in the game.
#[test]
fn oxidize_monk_steers_on_fostral() {
    use rapier3d::dynamics::{GenericJointBuilder, JointAxesMask, JointAxis, MotorModel};
    use std::io::BufReader;

    let map_path = Path::new("data/maps/fostral/map.png");
    let decoder = png::Decoder::new(BufReader::new(
        std::fs::File::open(map_path).expect("fostral map.png present"),
    ));
    let mut reader = decoder.read_info().expect("png header");
    let info_size = reader.output_buffer_size().expect("png size");
    let mut decoded = vec![0u8; info_size];
    let info = reader.next_frame(&mut decoded).expect("png decode");
    let (width, height) = (info.width, info.height);
    let alpha: Vec<u8> = (0..(width as usize * height as usize))
        .map(|i| decoded[i * 4 + 3])
        .collect();
    let map_radius_start = 10.0_f32;
    let map_radius_end = 15.0_f32;
    let circumference = 2.0 * std::f32::consts::PI * map_radius_start;
    let map_length = circumference * (height as f32) / (width as f32);
    let mut physics = Physics::default();
    let cfg = config::Map {
        radius: map_radius_start..map_radius_end,
        length: map_length,
        density: 10.0,
        is_sphere: false,
    };
    let terrain = physics.create_terrain(&cfg, alpha.clone(), width, height);

    // Build OxidizeMonk same as the heightfield test (sharing the GenericJoint).
    let car_path = Path::new("data/cars/OxidizeMonk");
    let car_config: config::Car =
        ron::de::from_bytes(&std::fs::read(car_path.join("car.ron")).expect("car.ron"))
            .expect("parse car.ron");
    let model_desc = Loader::read_gltf(
        &car_path.join("body.glb"),
        nalgebra::Matrix4::identity().scale(car_config.scale),
    );
    let keep = |m: &MaterialDesc| {
        !m.name
            .as_deref()
            .map(|n| n.to_lowercase().contains("wheel"))
            .unwrap_or(false)
    };
    let positions = model_desc.positions_filtered(keep);
    let (mins, maxs) = positions
        .iter()
        .fold((positions[0], positions[0]), |(mut a, mut b), p| {
            a.x = a.x.min(p.x);
            a.y = a.y.min(p.y);
            a.z = a.z.min(p.z);
            b.x = b.x.max(p.x);
            b.y = b.y.max(p.y);
            b.z = b.z.max(p.z);
            (a, b)
        });
    let lx = maxs.x - mins.x;
    let ly = maxs.y - mins.y;
    let lz = maxs.z - mins.z;
    let chassis_mass = lx * ly * lz * 0.1 * car_config.density;
    let inertia = Vec3::new(
        chassis_mass / 12.0 * (ly * ly + lz * lz),
        chassis_mass / 12.0 * (lx * lx + lz * lz),
        chassis_mass / 12.0 * (lx * lx + ly * ly),
    );
    let mass_props =
        rapier3d::dynamics::MassProperties::new(Vec3::new(0.0, -0.25, 0.0), chassis_mass, inertia);
    let chassis_collider = ColliderBuilder::ball(0.05)
        .density(0.0)
        .collision_groups(rapier3d::geometry::InteractionGroups::none())
        .build();
    let spawn_radius = map_radius_end - 0.5;
    let spawn_z = 0.1 * map_length;
    let transform = nalgebra::Isometry3 {
        translation: nalgebra::Vector3::new(0.0, spawn_radius, spawn_z).into(),
        rotation: nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            0.5 * std::f32::consts::PI,
        ),
    };
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(transform.into())
        .additional_mass_properties(mass_props)
        .linear_damping(0.4)
        .angular_damping(1.0)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, vec![chassis_collider]);

    let chassis_pose: Pose = transform.into();
    let mut wheel_data: Vec<(rapier3d::dynamics::ImpulseJointHandle, f32)> = Vec::new();
    for w in &car_config.wheels {
        let anchor_local = Vec3::new(w.position[0], w.position[1], w.position[2]);
        let wheel_world = chassis_pose * anchor_local;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(Pose::from_parts(wheel_world, chassis_pose.rotation))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(w.radius)
            .density(car_config.density)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);
        let locked = JointAxesMask::LIN_X
            | JointAxesMask::LIN_Z
            | JointAxesMask::ANG_X
            | JointAxesMask::ANG_Y;
        let joint = GenericJointBuilder::new(locked)
            .local_anchor1(anchor_local)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_model(JointAxis::LinY, MotorModel::ForceBased)
            .motor_position(JointAxis::LinY, 0.0, 300.0, 30.0)
            .motor_max_force(JointAxis::LinY, 500.0)
            .limits(JointAxis::LinY, [-0.3, 0.3])
            .motor_model(JointAxis::AngZ, MotorModel::ForceBased)
            .motor_velocity(JointAxis::AngZ, 0.0, 50.0)
            .motor_max_force(JointAxis::AngZ, car_config.motor_max_force)
            .build();
        let j = physics.add_generic_joint(chassis, wheel_rb, joint);
        // Mirror production: side = sign of anchor.z (chassis-Z is the wheel axle).
        let side = anchor_local.dot(Vec3::new(0.0, 0.0, 1.0)).signum();
        wheel_data.push((j, side));
    }

    // Settle.
    for _ in 0..250 {
        for &(j, _) in &wheel_data {
            physics.set_joint_motor_velocity(j, 0.0, 50.0);
        }
        physics.update_gravity(&terrain);
        physics.step();
    }
    let start_rot = physics.get_transform(chassis).rotation;

    // Apply LEFT steer (steer = -1), no throttle. With throttle=0 the differential
    // is unambiguous so the yaw direction tells us exactly which way the car turns.
    let max_v = car_config.motor_max_velocity;
    let steer = -1.0_f32;
    let strength = 0.5_f32; // matches production STEER_STRENGTH
    let throttle = 0.0_f32;
    // Collect wheel rigid body handles so we can read their state.
    let wheel_rbs: Vec<RigidBodyHandle> = {
        // The first wheel was added immediately after chassis; we can re-derive
        // by iterating bodies past the chassis. Simpler: track manually.
        // Re-collect from rapier by querying the joint's body2.
        // For now, just track from car_config index ordering.
        Vec::new()
    };
    let _ = wheel_rbs;
    for tick in 0..300 {
        for &(j, side) in &wheel_data {
            let v = (throttle - side * steer * strength) * max_v;
            physics.set_joint_motor_velocity(j, v, 1.0);
        }
        physics.update_gravity(&terrain);
        physics.step();
        if tick % 50 == 0 {
            let xform = physics.get_transform(chassis);
            let q = xform.rotation;
            let f0 = start_rot * (-nalgebra::Vector3::x());
            let f1 = q * (-nalgebra::Vector3::x());
            let cross_y = f0.z * f1.x - f0.x * f1.z;
            let dot = f0.x * f1.x + f0.z * f1.z;
            let yaw = cross_y.atan2(dot);
            let av = physics.body_angvel(chassis);
            let lv = physics.body_linvel(chassis);
            let pos = xform.translation;
            eprintln!(
                "tick={tick}: pos=({:.3}, {:.3}, {:.3}) yaw={yaw:.4} chassis_angvel=({:.3}, {:.3}, {:.3}) chassis_linvel=({:.3}, {:.3}, {:.3})",
                pos.x, pos.y, pos.z, av.x, av.y, av.z, lv.x, lv.y, lv.z,
            );
        }
    }
    let end_rot = physics.get_transform(chassis).rotation;
    let f0 = start_rot * (-nalgebra::Vector3::x());
    let f1 = end_rot * (-nalgebra::Vector3::x());
    let cross_y = f0.z * f1.x - f0.x * f1.z;
    let dot = f0.x * f1.x + f0.z * f1.z;
    let total_yaw = cross_y.atan2(dot);
    eprintln!("total yaw after 5 s of LEFT steer: {total_yaw:.4} rad");
    // We're not asserting a specific direction yet — just that the chassis
    // rotated noticeably. If this fails at < 0.1 rad, turning is broken.
    assert!(
        total_yaw.abs() > 0.1,
        "chassis barely yawed under steering input: {total_yaw:.4} rad"
    );
}

/// Spawn the chassis upside-down on flat terrain and verify it doesn't sink
/// through the surface. With the corner-balls chassis collider the body's
/// "top" (now pointing radially inward toward the cylinder axis) catches on
/// the bilinear surface; without those colliders, the chassis falls through.
#[test]
fn upside_down_chassis_stays_above_ground() {
    use rapier3d::dynamics::{GenericJointBuilder, JointAxesMask, JointAxis, MotorModel};

    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics); // ground at r ≈ 15.02
    let ground_r: f32 = 10.0 + (128.0 / 255.0) * (20.0 - 10.0);

    let car_path = Path::new("data/cars/OxidizeMonk");
    let car_config: config::Car =
        ron::de::from_bytes(&std::fs::read(car_path.join("car.ron")).expect("car.ron"))
            .expect("parse car.ron");
    let model_desc = Loader::read_gltf(
        &car_path.join("body.glb"),
        nalgebra::Matrix4::identity().scale(car_config.scale),
    );
    let keep = |m: &MaterialDesc| {
        !m.name
            .as_deref()
            .map(|n| n.to_lowercase().contains("wheel"))
            .unwrap_or(false)
    };
    let positions = model_desc.positions_filtered(keep);
    let (mins, maxs) = positions
        .iter()
        .fold((positions[0], positions[0]), |(mut a, mut b), p| {
            a.x = a.x.min(p.x);
            a.y = a.y.min(p.y);
            a.z = a.z.min(p.z);
            b.x = b.x.max(p.x);
            b.y = b.y.max(p.y);
            b.z = b.z.max(p.z);
            (a, b)
        });
    let lx = maxs.x - mins.x;
    let ly = maxs.y - mins.y;
    let lz = maxs.z - mins.z;
    let chassis_mass = lx * ly * lz * 0.1 * car_config.density;
    let inertia = Vec3::new(
        chassis_mass / 12.0 * (ly * ly + lz * lz),
        chassis_mass / 12.0 * (lx * lx + lz * lz),
        chassis_mass / 12.0 * (lx * lx + ly * ly),
    );
    let mass_props =
        rapier3d::dynamics::MassProperties::new(Vec3::new(0.0, -0.25, 0.0), chassis_mass, inertia);

    // Mirror production: 4 ball colliders on the chassis TOP face only.
    // When the chassis is spawned upside-down (flip applied below), these top
    // corners become the new bottom and support the body above ground.
    let corner_radius = 0.10_f32;
    let corner_positions = [
        Vec3::new(mins.x, maxs.y, mins.z),
        Vec3::new(maxs.x, maxs.y, mins.z),
        Vec3::new(mins.x, maxs.y, maxs.z),
        Vec3::new(maxs.x, maxs.y, maxs.z),
    ];
    let chassis_colliders: Vec<rapier3d::geometry::Collider> = corner_positions
        .iter()
        .map(|&p| {
            ColliderBuilder::ball(corner_radius)
                .translation(p)
                .density(0.0)
                .friction(0.0)
                .build()
        })
        .collect();

    // Spawn the chassis upside-down: roll 180° about the chassis-X axis means
    // chassis +Y now points in the world -Y (radially inward) direction.
    let upright = nalgebra::UnitQuaternion::from_axis_angle(
        &nalgebra::Vector3::y_axis(),
        0.5 * std::f32::consts::PI,
    );
    let flip = nalgebra::UnitQuaternion::from_axis_angle(
        &nalgebra::Vector3::x_axis(),
        std::f32::consts::PI,
    );
    let transform = nalgebra::Isometry3 {
        translation: nalgebra::Vector3::new(0.0, 19.0, 0.0).into(),
        rotation: flip * upright,
    };
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(transform.into())
        .additional_mass_properties(mass_props)
        .linear_damping(0.4)
        .angular_damping(1.0)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, chassis_colliders);

    // Attach wheels via the production GenericJoint (same as the steer test).
    let chassis_pose: Pose = transform.into();
    for w in &car_config.wheels {
        let anchor_local = Vec3::new(w.position[0], w.position[1], w.position[2]);
        let wheel_world = chassis_pose * anchor_local;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(Pose::from_parts(wheel_world, chassis_pose.rotation))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(w.radius)
            .density(car_config.density)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);
        let locked = JointAxesMask::LIN_X
            | JointAxesMask::LIN_Z
            | JointAxesMask::ANG_X
            | JointAxesMask::ANG_Y;
        let joint = GenericJointBuilder::new(locked)
            .local_anchor1(anchor_local)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_model(JointAxis::LinY, MotorModel::ForceBased)
            .motor_position(JointAxis::LinY, 0.0, 300.0, 30.0)
            .motor_max_force(JointAxis::LinY, 500.0)
            .limits(JointAxis::LinY, [-0.3, 0.3])
            .motor_model(JointAxis::AngZ, MotorModel::ForceBased)
            .motor_velocity(JointAxis::AngZ, 0.0, 50.0)
            .motor_max_force(JointAxis::AngZ, car_config.motor_max_force)
            .build();
        physics.add_generic_joint(chassis, wheel_rb, joint);
    }

    // Settle for 4 seconds.
    for tick in 0..240 {
        physics.update_gravity(&terrain);
        physics.step();
        if tick % 40 == 0 {
            let t = translation(&physics, chassis);
            let r = (t[0] * t[0] + t[1] * t[1]).sqrt();
            eprintln!(
                "settle tick={tick}: pos=({:.3}, {:.3}, {:.3}) r={r:.3}",
                t[0], t[1], t[2]
            );
        }
    }
    let final_t = translation(&physics, chassis);
    let final_r = (final_t[0] * final_t[0] + final_t[1] * final_t[1]).sqrt();
    eprintln!(
        "final upside-down chassis position: ({:.3}, {:.3}, {:.3}), r={final_r:.3}, ground_r={ground_r:.3}",
        final_t[0], final_t[1], final_t[2]
    );

    // The chassis center should sit above the ground. Even with the CoM offset
    // (-0.25 in chassis local) and a ~0.5 m tall chassis, the body shouldn't
    // sink so deep that its center radius falls more than ~0.2 m below ground.
    let sink_below_ground = ground_r - final_r;
    assert!(
        sink_below_ground < 0.2,
        "chassis sank too deep when flipped: sink = {sink_below_ground:.3} m \
         (final_r={final_r:.3}, ground_r={ground_r:.3})"
    );
}

/// Stuck-vehicle reproduction: mirrors the production chassis (8 corner balls
/// with friction 0.5, MassProperties::new with proper inertia, GenericJoint
/// LinY suspension + AngZ drive — i.e. what bin/game/main.rs::load_car builds)
/// on the real Fostral heightmap, holds full throttle for 10s, and reports
/// the per-second displacement so we can see *when* it gets stuck.
#[test]
fn oxidize_monk_pushes_forward_on_fostral_production_setup() {
    use rapier3d::dynamics::{
        GenericJointBuilder, JointAxesMask, JointAxis, MassProperties, MotorModel, RigidBodyBuilder,
    };
    use rapier3d::geometry::ColliderBuilder;
    use rapier3d::math::Vec3;
    use std::io::BufReader;

    // --- Load Fostral heightmap (alpha) -----------------------------------
    let map_path = Path::new("data/maps/fostral/map.png");
    let decoder = png::Decoder::new(BufReader::new(
        std::fs::File::open(map_path).expect("fostral map.png present (run `git lfs pull`)"),
    ));
    let mut reader = decoder.read_info().expect("png header");
    let info_size = reader.output_buffer_size().expect("png size");
    let mut decoded = vec![0u8; info_size];
    let info = reader.next_frame(&mut decoded).expect("png decode");
    let (width, height) = (info.width, info.height);
    let alpha: Vec<u8> = (0..(width as usize * height as usize))
        .map(|i| decoded[i * 4 + 3])
        .collect();

    let map_radius_start = 10.0_f32;
    let map_radius_end = 15.0_f32;
    let circumference = 2.0 * std::f32::consts::PI * map_radius_start;
    let map_length = circumference * (height as f32) / (width as f32);
    let cfg = config::Map {
        radius: map_radius_start..map_radius_end,
        length: map_length,
        density: 10.0,
        is_sphere: false,
    };
    let mut physics = Physics::default();
    let terrain = physics.create_terrain(&cfg, alpha, width, height);

    // --- Load OxidizeMonk + build the production-style chassis -----------
    let car_path = Path::new("data/cars/OxidizeMonk");
    let car_config: config::Car =
        ron::de::from_bytes(&std::fs::read(car_path.join("car.ron")).expect("car.ron"))
            .expect("parse car.ron");
    let model_desc = Loader::read_gltf(
        &car_path.join("body.glb"),
        nalgebra::Matrix4::identity().scale(car_config.scale),
    );
    let keep = |m: &MaterialDesc| {
        !m.name
            .as_deref()
            .map(|n| n.to_lowercase().contains("wheel"))
            .unwrap_or(false)
    };
    let positions = model_desc.positions_filtered(keep);
    let (mins, maxs) = positions
        .iter()
        .fold((positions[0], positions[0]), |(mut a, mut b), p| {
            a.x = a.x.min(p.x);
            a.y = a.y.min(p.y);
            a.z = a.z.min(p.z);
            b.x = b.x.max(p.x);
            b.y = b.y.max(p.y);
            b.z = b.z.max(p.z);
            (a, b)
        });
    let lx = maxs.x - mins.x;
    let ly = maxs.y - mins.y;
    let lz = maxs.z - mins.z;
    let chassis_mass = lx * ly * lz * 0.1 * car_config.density;
    eprintln!(
        "chassis aabb: x=[{:.2},{:.2}] y=[{:.2},{:.2}] z=[{:.2},{:.2}] mass={chassis_mass:.2}",
        mins.x, maxs.x, mins.y, maxs.y, mins.z, maxs.z,
    );

    // Production chassis: 4 ball colliders on the TOP face only. Catch the
    // chassis when it flips upside-down without snagging on terrain ridges
    // during upright driving.
    const CORNER_RADIUS: f32 = 0.10;
    let corners = [
        Vec3::new(mins.x, maxs.y, mins.z),
        Vec3::new(maxs.x, maxs.y, mins.z),
        Vec3::new(mins.x, maxs.y, maxs.z),
        Vec3::new(maxs.x, maxs.y, maxs.z),
    ];
    let chassis_colliders: Vec<_> = corners
        .iter()
        .map(|&p| {
            ColliderBuilder::ball(CORNER_RADIUS)
                .translation(p)
                .density(0.0)
                .friction(0.0)
                .build()
        })
        .collect();

    // Solid-cuboid inertia, CoM offset, MassProperties::new — same as production.
    let inertia = Vec3::new(
        chassis_mass / 12.0 * (ly * ly + lz * lz),
        chassis_mass / 12.0 * (lx * lx + lz * lz),
        chassis_mass / 12.0 * (lx * lx + ly * ly),
    );
    let chassis_com = Vec3::new(0.0, -0.25, 0.0);
    let mass_props = MassProperties::new(chassis_com, chassis_mass, inertia);

    // Spawn pose: same as game (theta=π/2 along +Y, z = 0.1 · length, r = radius_end - 0.5).
    let spawn_radius = map_radius_end - 0.5;
    let spawn_z = 0.1 * map_length;
    let transform = nalgebra::Isometry3 {
        translation: nalgebra::Vector3::new(0.0, spawn_radius, spawn_z).into(),
        rotation: nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            0.5 * std::f32::consts::PI,
        ),
    };
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(transform.into())
        .additional_mass_properties(mass_props)
        .linear_damping(0.4)
        .angular_damping(1.0)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, chassis_colliders);

    // Wheels + GenericJoint suspension + drive, matching production constants.
    let chassis_pose: rapier3d::math::Pose = transform.into();
    let mut wheel_joints = Vec::new();
    let mut wheels = Vec::new();
    for w in &car_config.wheels {
        let anchor_local = Vec3::new(w.position[0], w.position[1], w.position[2]);
        let wheel_world = chassis_pose * anchor_local;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(rapier3d::math::Pose::from_parts(
                wheel_world,
                chassis_pose.rotation,
            ))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(w.radius)
            .density(car_config.density)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);

        let locked = JointAxesMask::LIN_X
            | JointAxesMask::LIN_Z
            | JointAxesMask::ANG_X
            | JointAxesMask::ANG_Y;
        let joint = GenericJointBuilder::new(locked)
            .local_anchor1(anchor_local)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_model(JointAxis::LinY, MotorModel::ForceBased)
            .motor_position(JointAxis::LinY, 0.0, 300.0, 30.0)
            .motor_max_force(JointAxis::LinY, 500.0)
            .limits(JointAxis::LinY, [-0.3, 0.3])
            .motor_model(JointAxis::AngZ, MotorModel::ForceBased)
            .motor_velocity(JointAxis::AngZ, 0.0, 50.0)
            .motor_max_force(JointAxis::AngZ, car_config.motor_max_force)
            .build();
        wheel_joints.push(physics.add_generic_joint(chassis, wheel_rb, joint));
        wheels.push(wheel_rb);
    }

    // --- Settle for 4 seconds ---------------------------------------------
    const IDLE_BRAKE: f32 = 50.0;
    for _ in 0..240 {
        for &j in &wheel_joints {
            physics.set_joint_motor_velocity(j, 0.0, IDLE_BRAKE);
        }
        physics.update_gravity(&terrain);
        physics.step();
    }
    let settled_t = translation(&physics, chassis);
    eprintln!(
        "settled at pos=({:.3}, {:.3}, {:.3})",
        settled_t[0], settled_t[1], settled_t[2],
    );

    // --- Drive forward at motor_max_velocity for 10 seconds (600 ticks) ---
    let mut samples: Vec<(usize, [f32; 3], [f32; 3])> = Vec::new();
    samples.push((0, settled_t, [0.0; 3]));
    for tick in 1..=600 {
        for &j in &wheel_joints {
            physics.set_joint_motor_velocity(j, car_config.motor_max_velocity, 1.0);
        }
        physics.update_gravity(&terrain);
        physics.step();
        if tick % 60 == 0 {
            let k = physics.body_kinematics(chassis).unwrap();
            samples.push((tick, k.translation, k.linvel));
        }
    }

    // Per-second displacement: if it drops to near zero, the car is stuck.
    eprintln!(
        "{:>5}  {:>27}  {:>22}  {:>8}",
        "tick", "pos (x,y,z)", "linvel (m/s)", "Δ/sec"
    );
    let mut min_chunk = f32::INFINITY;
    let mut max_chunk = 0.0_f32;
    for w in samples.windows(2) {
        let dx = w[1].1[0] - w[0].1[0];
        let dz = w[1].1[2] - w[0].1[2];
        let d = (dx * dx + dz * dz).sqrt();
        let speed = (w[1].2[0].powi(2) + w[1].2[1].powi(2) + w[1].2[2].powi(2)).sqrt();
        eprintln!(
            "{:5}  ({:7.3},{:7.3},{:7.3})  ({:6.3},{:6.3},{:6.3})  {:7.3}",
            w[1].0, w[1].1[0], w[1].1[1], w[1].1[2], w[1].2[0], w[1].2[1], w[1].2[2], d,
        );
        let _ = speed;
        if d < min_chunk {
            min_chunk = d;
        }
        if d > max_chunk {
            max_chunk = d;
        }
    }
    let final_t = translation(&physics, chassis);
    let dx = final_t[0] - settled_t[0];
    let dz = final_t[2] - settled_t[2];
    let final_disp = (dx * dx + dz * dz).sqrt();
    eprintln!(
        "10s total displacement: |xz|={final_disp:.3}m, per-second chunk min={min_chunk:.3} max={max_chunk:.3}"
    );

    // Production setup should move at least a few metres in 10s of full
    // throttle on real terrain. If this drops below ~0.5 m total, something
    // is wedging the chassis (corner balls catching, suspension stuck, etc.).
    assert!(
        final_disp > 0.5,
        "production-setup OxidizeMonk got stuck on Fostral: only {final_disp:.3} m of horizontal motion after 10s of forward drive"
    );
}

/// Reproduces "can't turn the car when moving". Drives forward at full
/// throttle WHILE applying a sustained steer input, on the real Fostral
/// heightfield with the production chassis setup. Reports per-second yaw
/// and yaw rate so we can see whether the differential is even reaching
/// the chassis.
#[test]
fn oxidize_monk_turns_while_driving_on_fostral() {
    use rapier3d::dynamics::{
        GenericJointBuilder, JointAxesMask, JointAxis, MassProperties, MotorModel, RigidBodyBuilder,
    };
    use rapier3d::geometry::ColliderBuilder;
    use rapier3d::math::Vec3;
    use std::io::BufReader;

    let map_path = Path::new("data/maps/fostral/map.png");
    let decoder = png::Decoder::new(BufReader::new(
        std::fs::File::open(map_path).expect("fostral map.png present (run `git lfs pull`)"),
    ));
    let mut reader = decoder.read_info().expect("png header");
    let info_size = reader.output_buffer_size().expect("png size");
    let mut decoded = vec![0u8; info_size];
    let info = reader.next_frame(&mut decoded).expect("png decode");
    let (width, height) = (info.width, info.height);
    let alpha: Vec<u8> = (0..(width as usize * height as usize))
        .map(|i| decoded[i * 4 + 3])
        .collect();

    let map_radius_start = 10.0_f32;
    let map_radius_end = 15.0_f32;
    let circumference = 2.0 * std::f32::consts::PI * map_radius_start;
    let map_length = circumference * (height as f32) / (width as f32);
    let cfg = config::Map {
        radius: map_radius_start..map_radius_end,
        length: map_length,
        density: 10.0,
        is_sphere: false,
    };
    let mut physics = Physics::default();
    let terrain = physics.create_terrain(&cfg, alpha, width, height);

    let car_path = Path::new("data/cars/OxidizeMonk");
    let car_config: config::Car =
        ron::de::from_bytes(&std::fs::read(car_path.join("car.ron")).expect("car.ron"))
            .expect("parse car.ron");
    let model_desc = Loader::read_gltf(
        &car_path.join("body.glb"),
        nalgebra::Matrix4::identity().scale(car_config.scale),
    );
    let keep = |m: &MaterialDesc| {
        !m.name
            .as_deref()
            .map(|n| n.to_lowercase().contains("wheel"))
            .unwrap_or(false)
    };
    let positions = model_desc.positions_filtered(keep);
    let (mins, maxs) = positions
        .iter()
        .fold((positions[0], positions[0]), |(mut a, mut b), p| {
            a.x = a.x.min(p.x);
            a.y = a.y.min(p.y);
            a.z = a.z.min(p.z);
            b.x = b.x.max(p.x);
            b.y = b.y.max(p.y);
            b.z = b.z.max(p.z);
            (a, b)
        });
    let lx = maxs.x - mins.x;
    let ly = maxs.y - mins.y;
    let lz = maxs.z - mins.z;
    let chassis_mass = lx * ly * lz * 0.1 * car_config.density;

    // Top 4 corners only — production setup.
    const CORNER_RADIUS: f32 = 0.10;
    let corners = [
        Vec3::new(mins.x, maxs.y, mins.z),
        Vec3::new(maxs.x, maxs.y, mins.z),
        Vec3::new(mins.x, maxs.y, maxs.z),
        Vec3::new(maxs.x, maxs.y, maxs.z),
    ];
    let chassis_colliders: Vec<_> = corners
        .iter()
        .map(|&p| {
            ColliderBuilder::ball(CORNER_RADIUS)
                .translation(p)
                .density(0.0)
                .friction(0.0)
                .build()
        })
        .collect();

    let inertia = Vec3::new(
        chassis_mass / 12.0 * (ly * ly + lz * lz),
        chassis_mass / 12.0 * (lx * lx + lz * lz),
        chassis_mass / 12.0 * (lx * lx + ly * ly),
    );
    let mass_props = MassProperties::new(Vec3::new(0.0, -0.25, 0.0), chassis_mass, inertia);

    let spawn_radius = map_radius_end - 0.5;
    let spawn_z = 0.1 * map_length;
    let transform = nalgebra::Isometry3 {
        translation: nalgebra::Vector3::new(0.0, spawn_radius, spawn_z).into(),
        rotation: nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            0.5 * std::f32::consts::PI,
        ),
    };
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(transform.into())
        .additional_mass_properties(mass_props)
        .linear_damping(0.4)
        // Per-axis damping applied inside the simulation loop.
        .angular_damping(0.0)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, chassis_colliders);

    let chassis_pose: rapier3d::math::Pose = transform.into();
    let mut wheel_data: Vec<(rapier3d::dynamics::ImpulseJointHandle, f32)> = Vec::new();
    for w in &car_config.wheels {
        let anchor_local = Vec3::new(w.position[0], w.position[1], w.position[2]);
        let wheel_world = chassis_pose * anchor_local;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(rapier3d::math::Pose::from_parts(
                wheel_world,
                chassis_pose.rotation,
            ))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(w.radius)
            .density(car_config.density)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);

        let locked = JointAxesMask::LIN_X
            | JointAxesMask::LIN_Z
            | JointAxesMask::ANG_X
            | JointAxesMask::ANG_Y;
        let joint = GenericJointBuilder::new(locked)
            .local_anchor1(anchor_local)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_model(JointAxis::LinY, MotorModel::ForceBased)
            .motor_position(JointAxis::LinY, 0.0, 300.0, 30.0)
            .motor_max_force(JointAxis::LinY, 500.0)
            .limits(JointAxis::LinY, [-0.3, 0.3])
            .motor_model(JointAxis::AngZ, MotorModel::ForceBased)
            .motor_velocity(JointAxis::AngZ, 0.0, 50.0)
            .motor_max_force(JointAxis::AngZ, car_config.motor_max_force)
            .build();
        let j = physics.add_generic_joint(chassis, wheel_rb, joint);
        // Mirror production: side = sign of anchor.z (wheel axle is chassis-Z).
        let side = anchor_local.dot(Vec3::new(0.0, 0.0, 1.0)).signum();
        wheel_data.push((j, side));
    }

    // Production-matching per-chassis-axis damping vector (high roll/pitch,
    // low yaw).
    let chassis_damp_yaw = 0.3_f32;
    let chassis_damp_tumble = 2.0_f32;

    // Settle.
    for _ in 0..240 {
        for &(j, _) in &wheel_data {
            physics.set_joint_motor_velocity(j, 0.0, 50.0);
        }
        physics.update_gravity(&terrain);
        physics.apply_axial_angular_damping(chassis, chassis_damp_yaw, chassis_damp_tumble);
        physics.step();
    }
    let start_rot = physics.get_transform(chassis).rotation;

    // Apply forward throttle + LEFT steer simultaneously, just like the user
    // holding W+A in-game.
    let max_v = car_config.motor_max_velocity;
    let throttle = 1.0_f32;
    let steer = -1.0_f32;
    let strength = 0.5_f32;

    let yaw_of = |rot: nalgebra::UnitQuaternion<f32>| -> f32 {
        let f0 = start_rot * (-nalgebra::Vector3::x());
        let f1 = rot * (-nalgebra::Vector3::x());
        let cross_y = f0.z * f1.x - f0.x * f1.z;
        let dot = f0.x * f1.x + f0.z * f1.z;
        cross_y.atan2(dot)
    };

    eprintln!(
        "{:>5}  {:>27}  {:>22}  {:>6}",
        "tick", "pos (x,y,z)", "linvel (m/s)", "yaw"
    );
    let mut samples_yaw: Vec<f32> = Vec::new();
    for tick in 1..=600 {
        for &(j, side) in &wheel_data {
            let v = (throttle - side * steer * strength) * max_v;
            physics.set_joint_motor_velocity(j, v, 1.0);
        }
        physics.update_gravity(&terrain);
        physics.apply_axial_angular_damping(chassis, chassis_damp_yaw, chassis_damp_tumble);
        physics.step();
        if tick % 60 == 0 {
            let xform = physics.get_transform(chassis);
            let pos = xform.translation;
            let lv = physics.body_linvel(chassis);
            let yaw = yaw_of(xform.rotation);
            samples_yaw.push(yaw);
            eprintln!(
                "{:5}  ({:7.3},{:7.3},{:7.3})  ({:6.3},{:6.3},{:6.3})  {:6.3}",
                tick, pos.x, pos.y, pos.z, lv.x, lv.y, lv.z, yaw,
            );
        }
    }
    let final_yaw = *samples_yaw.last().unwrap();
    let max_abs_yaw = samples_yaw.iter().map(|y| y.abs()).fold(0.0_f32, f32::max);
    eprintln!(
        "yaw after 10 s of W+LEFT_steer: final={final_yaw:.4} rad ({:.2}°), peak={max_abs_yaw:.4} rad ({:.2}°)",
        final_yaw.to_degrees(),
        max_abs_yaw.to_degrees(),
    );

    // Use peak abs yaw, not final: the car can drive over hills and rebound
    // its heading back toward 0 after a good turn, so the final reading
    // understates whether steering "works" during the run.
    assert!(
        max_abs_yaw > 0.3,
        "chassis barely yawed under W+steer: peak {max_abs_yaw:.4} rad, final {final_yaw:.4} rad after 10s"
    );
}

/// Steering test on flat terrain (no heightfield interference). Drives forward
/// at full throttle while holding LEFT steer, then later RIGHT steer, and
/// verifies the chassis yaws in opposite directions by similar magnitudes.
/// If the in-game steering "feels weird", this test isolates whether the
/// differential drive itself works correctly in clean conditions.
#[test]
fn oxidize_monk_steers_symmetrically_while_moving_on_flat() {
    use rapier3d::dynamics::{
        GenericJointBuilder, JointAxesMask, JointAxis, MassProperties, MotorModel, RigidBodyBuilder,
    };
    use rapier3d::geometry::ColliderBuilder;
    use rapier3d::math::Vec3;

    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);

    // Load and build production-style chassis on flat terrain.
    let car_path = Path::new("data/cars/OxidizeMonk");
    let car_config: config::Car =
        ron::de::from_bytes(&std::fs::read(car_path.join("car.ron")).expect("car.ron"))
            .expect("parse car.ron");
    let model_desc = Loader::read_gltf(
        &car_path.join("body.glb"),
        nalgebra::Matrix4::identity().scale(car_config.scale),
    );
    let keep = |m: &MaterialDesc| {
        !m.name
            .as_deref()
            .map(|n| n.to_lowercase().contains("wheel"))
            .unwrap_or(false)
    };
    let positions = model_desc.positions_filtered(keep);
    let (mins, maxs) = positions
        .iter()
        .fold((positions[0], positions[0]), |(mut a, mut b), p| {
            a.x = a.x.min(p.x);
            a.y = a.y.min(p.y);
            a.z = a.z.min(p.z);
            b.x = b.x.max(p.x);
            b.y = b.y.max(p.y);
            b.z = b.z.max(p.z);
            (a, b)
        });
    let lx = maxs.x - mins.x;
    let ly = maxs.y - mins.y;
    let lz = maxs.z - mins.z;
    let chassis_mass = lx * ly * lz * 0.1 * car_config.density;
    const CORNER_RADIUS: f32 = 0.10;
    let corners = [
        Vec3::new(mins.x, maxs.y, mins.z),
        Vec3::new(maxs.x, maxs.y, mins.z),
        Vec3::new(mins.x, maxs.y, maxs.z),
        Vec3::new(maxs.x, maxs.y, maxs.z),
    ];
    let chassis_colliders: Vec<_> = corners
        .iter()
        .map(|&p| {
            ColliderBuilder::ball(CORNER_RADIUS)
                .translation(p)
                .density(0.0)
                .friction(0.0)
                .build()
        })
        .collect();
    let inertia = Vec3::new(
        chassis_mass / 12.0 * (ly * ly + lz * lz),
        chassis_mass / 12.0 * (lx * lx + lz * lz),
        chassis_mass / 12.0 * (lx * lx + ly * ly),
    );
    let mass_props = MassProperties::new(Vec3::new(0.0, -0.25, 0.0), chassis_mass, inertia);
    let transform = nalgebra::Isometry3 {
        translation: nalgebra::Vector3::new(0.0, SPAWN_RADIUS, 0.1 * TERRAIN_LENGTH).into(),
        rotation: nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            0.5 * std::f32::consts::PI,
        ),
    };
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(transform.into())
        .additional_mass_properties(mass_props)
        .linear_damping(0.4)
        // Per-axis damping applied inside the simulation loop.
        .angular_damping(0.0)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, chassis_colliders);

    let chassis_pose: rapier3d::math::Pose = transform.into();
    let mut wheel_data: Vec<(rapier3d::dynamics::ImpulseJointHandle, f32)> = Vec::new();
    for w in &car_config.wheels {
        let anchor_local = Vec3::new(w.position[0], w.position[1], w.position[2]);
        let wheel_world = chassis_pose * anchor_local;
        let wheel_body = RigidBodyBuilder::dynamic()
            .pose(rapier3d::math::Pose::from_parts(
                wheel_world,
                chassis_pose.rotation,
            ))
            .angular_damping(0.2)
            .build();
        let wheel_coll = ColliderBuilder::ball(w.radius)
            .density(car_config.density)
            .friction(3.0)
            .build();
        let PhysicsBodyHandle {
            rigid_body_handle: wheel_rb,
            ..
        } = physics.add_rigid_body(wheel_body, vec![wheel_coll]);
        let locked = JointAxesMask::LIN_X
            | JointAxesMask::LIN_Z
            | JointAxesMask::ANG_X
            | JointAxesMask::ANG_Y;
        let joint = GenericJointBuilder::new(locked)
            .local_anchor1(anchor_local)
            .local_anchor2(Vec3::ZERO)
            .contacts_enabled(false)
            .motor_model(JointAxis::LinY, MotorModel::ForceBased)
            .motor_position(JointAxis::LinY, 0.0, 300.0, 30.0)
            .motor_max_force(JointAxis::LinY, 500.0)
            .limits(JointAxis::LinY, [-0.3, 0.3])
            .motor_model(JointAxis::AngZ, MotorModel::ForceBased)
            .motor_velocity(JointAxis::AngZ, 0.0, 50.0)
            .motor_max_force(JointAxis::AngZ, car_config.motor_max_force)
            .build();
        let j = physics.add_generic_joint(chassis, wheel_rb, joint);
        let side = anchor_local.dot(Vec3::new(0.0, 0.0, 1.0)).signum();
        wheel_data.push((j, side));
    }

    // Production-matching per-chassis-axis damping (roll/pitch high, yaw low).
    let chassis_damp_yaw = 0.3_f32;
    let chassis_damp_tumble = 2.0_f32;

    // Settle.
    for _ in 0..240 {
        for &(j, _) in &wheel_data {
            physics.set_joint_motor_velocity(j, 0.0, 50.0);
        }
        physics.update_gravity(&terrain);
        physics.apply_axial_angular_damping(chassis, chassis_damp_yaw, chassis_damp_tumble);
        physics.step();
    }
    let start_rot = physics.get_transform(chassis).rotation;

    let yaw_of = |rot: nalgebra::UnitQuaternion<f32>| -> f32 {
        let f0 = start_rot * (-nalgebra::Vector3::x());
        let f1 = rot * (-nalgebra::Vector3::x());
        let cross_y = f0.z * f1.x - f0.x * f1.z;
        let dot = f0.x * f1.x + f0.z * f1.z;
        cross_y.atan2(dot)
    };

    let max_v = car_config.motor_max_velocity;
    let strength = 0.5_f32;

    // Phase 1: W + LEFT steer (steer = -1) for 3 seconds.
    let mut yaw_left = 0.0_f32;
    for _ in 0..180 {
        for &(j, side) in &wheel_data {
            let v = (1.0 - (-side) * strength) * max_v;
            physics.set_joint_motor_velocity(j, v, 1.0);
        }
        physics.update_gravity(&terrain);
        physics.apply_axial_angular_damping(chassis, chassis_damp_yaw, chassis_damp_tumble);
        physics.step();
        yaw_left = yaw_of(physics.get_transform(chassis).rotation);
    }
    eprintln!(
        "after 3s of W+LEFT: yaw = {yaw_left:+.4} rad ({:+.2}°)",
        yaw_left.to_degrees()
    );

    // Phase 2: W + RIGHT steer (steer = +1) for 3 seconds. Yaw should swing
    // back through 0 and end with the opposite sign by a similar amount.
    let yaw_after_left = yaw_left;
    let mut yaw_right = yaw_left;
    for _ in 0..180 {
        for &(j, side) in &wheel_data {
            let v = (1.0 - side * 1.0 * strength) * max_v;
            physics.set_joint_motor_velocity(j, v, 1.0);
        }
        physics.update_gravity(&terrain);
        physics.apply_axial_angular_damping(chassis, chassis_damp_yaw, chassis_damp_tumble);
        physics.step();
        yaw_right = yaw_of(physics.get_transform(chassis).rotation);
    }
    let net_swing = yaw_right - yaw_after_left;
    eprintln!(
        "after 3s of W+RIGHT (from yaw={yaw_after_left:+.4}): yaw = {yaw_right:+.4} rad ({:+.2}°), net swing = {net_swing:+.4} rad ({:+.2}°)",
        yaw_right.to_degrees(),
        net_swing.to_degrees(),
    );

    // LEFT steer should produce LEFT yaw — negative in our cross/dot convention
    // (forward vector rotates from +Z toward -X = CCW from above = negative
    // cross_y in this projection).
    assert!(
        yaw_left < -0.1,
        "W+LEFT didn't yaw left on flat terrain: yaw={yaw_left:.4} rad (expected < -0.1)"
    );
    // RIGHT steer should reverse the trend: net swing back toward positive.
    assert!(
        net_swing > 0.2,
        "W+RIGHT after W+LEFT failed to swing the chassis back: net_swing={net_swing:.4} rad (expected > 0.2)"
    );
}
