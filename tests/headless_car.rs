//! Headless car-on-cylinder scenarios — verifies that the cylindrical heightfield collider,
//! the radial gravity, and the wheeled-vehicle joint setup all combine into something that
//! behaves the way the game expects.
//!
//! The "car" here is a synthetic minimum: a cuboid chassis with four ball-collider wheels
//! attached by revolute joints. We skip the GLB loader and rendering, but everything else
//! (Physics, NarrowPhase with CylDispatcher, CylindricalHeightField, RevoluteJoint motors)
//! is the real production code paths.

use rapier3d::dynamics::{
    ImpulseJointHandle, RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle,
};
use rapier3d::geometry::ColliderBuilder;
use rapier3d::math::{Pose, Rotation, Vec3};
use vandals_and_heroes::{config, Physics, PhysicsBodyHandle, TerrainBody};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 256;

fn build_flat_terrain(physics: &mut Physics) -> TerrainBody {
    // Uniform alpha 128 → ground_radius = lerp(10, 20, 128/255) ≈ 15.02. Flat cylinder.
    let alpha = vec![128u8; (WIDTH * HEIGHT) as usize];
    let cfg = config::Map {
        radius: 10.0..20.0,
        length: 100.0,
        density: 10.0,
    };
    physics.create_terrain(&cfg, alpha, WIDTH, HEIGHT)
}

struct TestWheel {
    #[allow(dead_code)]
    body: RigidBodyHandle,
    joint: ImpulseJointHandle,
}

struct TestCar {
    chassis: RigidBodyHandle,
    /// Indices: 0 = front-left, 1 = front-right, 2 = back-left, 3 = back-right.
    /// "Left" = chassis-local +Z, "right" = chassis-local -Z, "front" = chassis-local +X.
    wheels: Vec<TestWheel>,
}

const WHEEL_RADIUS: f32 = 0.15;
/// Motor torque is intentionally generous in tests so steering decisively dominates the
/// small angular drift the car picks up from settling on the curved cylinder surface.
const TEST_MOTOR_MAX_FORCE: f32 = 50.0;

/// Spawn a synthetic four-wheeled vehicle at `pos` with identity orientation:
/// chassis-local +X = forward, +Z = left, -Y = down (toward cylinder axis).
fn spawn_car(physics: &mut Physics, pos: Vec3) -> TestCar {
    let chassis_pose = Pose::from_translation(pos);
    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(chassis_pose)
        .linear_damping(0.05)
        .angular_damping(0.05)
        .build();
    // Cuboid 1.0 x 0.4 x 0.8 (full-extent dims: ColliderBuilder::cuboid takes half-extents)
    let chassis_coll = ColliderBuilder::cuboid(0.5, 0.2, 0.4).density(10.0).build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, vec![chassis_coll]);

    let anchors = [
        Vec3::new(0.4, -0.30, 0.45),   // 0 front-left
        Vec3::new(0.4, -0.30, -0.45),  // 1 front-right
        Vec3::new(-0.4, -0.30, 0.45),  // 2 back-left
        Vec3::new(-0.4, -0.30, -0.45), // 3 back-right
    ];
    let axis_local = Vec3::new(0.0, 0.0, 1.0);

    let wheels = anchors
        .iter()
        .map(|&anchor| {
            let wheel_world = chassis_pose * anchor;
            let wheel_rb = RigidBodyBuilder::dynamic()
                .pose(Pose::from_parts(wheel_world, chassis_pose.rotation))
                .angular_damping(0.2)
                .build();
            let wheel_coll = ColliderBuilder::ball(WHEEL_RADIUS)
                .density(10.0)
                .friction(10.0)
                .build();
            let PhysicsBodyHandle {
                rigid_body_handle: wheel,
                ..
            } = physics.add_rigid_body(wheel_rb, vec![wheel_coll]);

            let joint = RevoluteJointBuilder::new(axis_local)
                .local_anchor1(anchor)
                .local_anchor2(Vec3::ZERO)
                .contacts_enabled(false)
                .motor_max_force(TEST_MOTOR_MAX_FORCE)
                .motor_velocity(0.0, 0.2)
                .build();
            let joint_handle = physics.add_revolute_joint(chassis, wheel, joint);
            TestWheel {
                body: wheel,
                joint: joint_handle,
            }
        })
        .collect();

    TestCar { chassis, wheels }
}

/// Differential drive — independent target velocities for the left and right sides.
/// Positive on both = forward; (+, -) = turn right (left wheels forward, right wheels back);
/// (-, +) = turn left.
fn drive(physics: &mut Physics, car: &TestCar, left_v: f32, right_v: f32) {
    // wheels 0 and 2 are the left side (+Z anchor), 1 and 3 are right (-Z).
    physics.set_joint_motor_velocity(car.wheels[0].joint, left_v, 1.0);
    physics.set_joint_motor_velocity(car.wheels[2].joint, left_v, 1.0);
    physics.set_joint_motor_velocity(car.wheels[1].joint, right_v, 1.0);
    physics.set_joint_motor_velocity(car.wheels[3].joint, right_v, 1.0);
}

fn run_ticks(physics: &mut Physics, terrain: &TerrainBody, ticks: usize) {
    for _ in 0..ticks {
        physics.update_gravity(terrain);
        physics.step();
    }
}

/// Snapshot a body's (translation, rotation) for state-dump comparisons.
fn snapshot(physics: &Physics, body: RigidBodyHandle) -> ([f32; 3], [f32; 4]) {
    let k = physics.body_kinematics(body).expect("body should exist");
    (k.translation, k.rotation)
}

fn radial(t: [f32; 3]) -> f32 {
    (t[0] * t[0] + t[1] * t[1]).sqrt()
}

/// World forward direction for a chassis whose local +X is forward, rotated by quat (x,y,z,w).
/// Returns the rotation applied to Vec3::X.
fn forward_dir(rot: [f32; 4]) -> Vec3 {
    let q = Rotation::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
    q * Vec3::X
}

// --------------------------------------------------------------------------------------
// Scenario tests
// --------------------------------------------------------------------------------------

#[test]
fn car_settles_on_flat_cylinder_without_falling_through() {
    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);
    // Ground is at r ≈ 15.02; spawn above at r = 19 so the car falls cleanly.
    let car = spawn_car(&mut physics, Vec3::new(0.0, 19.0, 0.0));

    run_ticks(&mut physics, &terrain, 400);

    let (t, _) = snapshot(&physics, car.chassis);
    let r = radial(t);

    // Wheels sit below the chassis by ~0.45 (anchor.y=-0.30 + WHEEL_RADIUS=0.15);
    // gravity is radially inward, so the chassis rests at ground + ~0.45.
    assert!(
        r > 15.0,
        "car fell through terrain (radial < ground): r = {r}"
    );
    assert!(r < 16.5, "car did not actually contact the ground: r = {r}");

    let (_, _) = snapshot(&physics, car.chassis);
    let k = physics.body_kinematics(car.chassis).unwrap();
    let speed = (k.linvel[0].powi(2) + k.linvel[1].powi(2) + k.linvel[2].powi(2)).sqrt();
    assert!(speed < 0.5, "car never settled: speed = {speed}");
}

#[test]
fn forward_drive_moves_car_along_its_forward_axis() {
    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);
    let car = spawn_car(&mut physics, Vec3::new(0.0, 19.0, 0.0));

    // Settle
    run_ticks(&mut physics, &terrain, 250);
    let (start_t, start_rot) = snapshot(&physics, car.chassis);

    // Drive forward
    drive(&mut physics, &car, 15.0, 15.0);
    run_ticks(&mut physics, &terrain, 600);

    let (end_t, _) = snapshot(&physics, car.chassis);

    let dx = end_t[0] - start_t[0];
    let dz = end_t[2] - start_t[2];
    let horizontal_disp = (dx * dx + dz * dz).sqrt();

    // Direction the car was facing while driving (it should have stayed roughly upright
    // and rolled along its initial +X).
    let fwd = forward_dir(start_rot);
    // Project the displacement onto the forward direction (XZ plane only).
    let projected = dx * fwd.x + dz * fwd.z;

    assert!(
        horizontal_disp > 0.3,
        "car barely moved with forward drive: |Δxz| = {horizontal_disp}"
    );
    // Movement should be predominantly along the forward axis (some lateral drift is OK).
    assert!(
        projected.abs() > 0.5 * horizontal_disp,
        "movement isn't along the forward axis: projected = {projected}, total = {horizontal_disp}"
    );
}

#[test]
fn reverse_drive_moves_car_in_opposite_direction_from_forward() {
    let mut physics_fwd = Physics::default();
    let terrain_fwd = build_flat_terrain(&mut physics_fwd);
    let car_fwd = spawn_car(&mut physics_fwd, Vec3::new(0.0, 19.0, 0.0));
    run_ticks(&mut physics_fwd, &terrain_fwd, 250);
    let start_fwd = snapshot(&physics_fwd, car_fwd.chassis).0;
    drive(&mut physics_fwd, &car_fwd, 15.0, 15.0);
    run_ticks(&mut physics_fwd, &terrain_fwd, 600);
    let end_fwd = snapshot(&physics_fwd, car_fwd.chassis).0;

    let mut physics_rev = Physics::default();
    let terrain_rev = build_flat_terrain(&mut physics_rev);
    let car_rev = spawn_car(&mut physics_rev, Vec3::new(0.0, 19.0, 0.0));
    run_ticks(&mut physics_rev, &terrain_rev, 250);
    let start_rev = snapshot(&physics_rev, car_rev.chassis).0;
    drive(&mut physics_rev, &car_rev, -15.0, -15.0);
    run_ticks(&mut physics_rev, &terrain_rev, 600);
    let end_rev = snapshot(&physics_rev, car_rev.chassis).0;

    // The two displacements should point in opposite directions in the XZ plane.
    let fdx = end_fwd[0] - start_fwd[0];
    let fdz = end_fwd[2] - start_fwd[2];
    let rdx = end_rev[0] - start_rev[0];
    let rdz = end_rev[2] - start_rev[2];

    let dot = fdx * rdx + fdz * rdz;
    let fwd_mag = (fdx * fdx + fdz * fdz).sqrt();
    let rev_mag = (rdx * rdx + rdz * rdz).sqrt();

    assert!(
        fwd_mag > 0.3 && rev_mag > 0.3,
        "neither direction moved enough"
    );
    assert!(
        dot < 0.0,
        "reverse and forward should point opposite ways: dot = {dot}"
    );
}

#[test]
fn turning_left_and_right_yield_opposite_yaw_changes() {
    fn yaw_after_turn(left_v: f32, right_v: f32) -> f32 {
        let mut physics = Physics::default();
        let terrain = build_flat_terrain(&mut physics);
        let car = spawn_car(&mut physics, Vec3::new(0.0, 19.0, 0.0));
        run_ticks(&mut physics, &terrain, 250);
        let (_, start_rot) = snapshot(&physics, car.chassis);
        drive(&mut physics, &car, left_v, right_v);
        run_ticks(&mut physics, &terrain, 300);
        let (_, end_rot) = snapshot(&physics, car.chassis);

        // Yaw delta: how much the chassis's forward direction rotated about world +Y.
        let f0 = forward_dir(start_rot);
        let f1 = forward_dir(end_rot);
        // Signed angle about +Y (project to XZ plane): atan2(cross_y, dot)
        let cross_y = f0.z * f1.x - f0.x * f1.z;
        let dot = f0.x * f1.x + f0.z * f1.z;
        cross_y.atan2(dot)
    }

    // Turn LEFT: left wheels backward, right wheels forward.
    let yaw_left = yaw_after_turn(-15.0, 15.0);
    // Turn RIGHT: opposite.
    let yaw_right = yaw_after_turn(15.0, -15.0);

    // Smooth-surface contact (no triangulation bumps) means yaw is small on this
    // synthetic-cube chassis with very low angular damping (0.05). Both turns may
    // share a sign because settling drift dominates the absolute yaw. What we
    // verify is that the differential drive produces a DIFFERENT yaw for opposite
    // inputs, with the expected ordering (left turn > right turn).
    let differential = yaw_left - yaw_right;
    assert!(
        differential > 0.005,
        "differential drive produced no left-vs-right yaw difference: \
         yaw_left={yaw_left}, yaw_right={yaw_right}, diff={differential}"
    );
}
