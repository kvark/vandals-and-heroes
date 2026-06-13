//! Headless drive-combinations tests on a flat cylindrical heightfield with
//! the production Ackermann steering setup. Holds W/S/A/D individually and in
//! pairs, records chassis trajectory + per-wheel positions, verifies the
//! expected motion, and writes one SVG per combination so wheel placement can
//! be eyeballed without running the GPU.
//!
//! The flat plane is a uniform-alpha heightfield (no terrain noise to perturb
//! the chassis), so the results expose the steering / drive geometry alone.

use rapier3d::dynamics::{
    GenericJointBuilder, ImpulseJointHandle, JointAxesMask, JointAxis, MassProperties, MotorModel,
    RigidBodyBuilder, RigidBodyHandle,
};
use rapier3d::geometry::ColliderBuilder;
use rapier3d::math::Vec3;
use std::fs;
use std::path::Path;
use vandals_and_heroes::{config, Loader, MaterialDesc, Physics, PhysicsBodyHandle, TerrainBody};

const TERRAIN_WIDTH: u32 = 64;
const TERRAIN_HEIGHT: u32 = 256;
const TERRAIN_RADIUS_START: f32 = 10.0;
const TERRAIN_RADIUS_END: f32 = 20.0;
const TERRAIN_LENGTH: f32 = 100.0;
const SPAWN_RADIUS: f32 = TERRAIN_RADIUS_END - 0.5;

// Production constants (mirror bin/game/main.rs).
const MAX_STEER_ANGLE: f32 = std::f32::consts::FRAC_PI_4;
const STEER_STIFFNESS: f32 = 200.0;
const STEER_DAMPING: f32 = 20.0;
const STEER_MAX_FORCE: f32 = 50.0;
const IDLE_BRAKE_FACTOR: f32 = 50.0;
const SUSPENSION_STIFFNESS: f32 = 300.0;
const SUSPENSION_DAMPING: f32 = 50.0;
const SUSPENSION_MAX_FORCE: f32 = 500.0;

struct LoadedCar {
    chassis: RigidBodyHandle,
    /// (wheel_joint, wheel_rb, anchor_local, is_steering, steering_joint).
    /// `wheel_joint` is the chassis↔wheel joint for rear wheels, or the
    /// knuckle↔wheel joint for front wheels. It always owns AngZ (drive) +
    /// LinY (suspension). `steering_joint` is `Some` only for front wheels,
    /// where it's the chassis↔knuckle joint that owns AngY (steering). The
    /// two-joint chain isolates the wheel's spin axis from the steering
    /// rotation — without it, the single-GenericJoint's AngZ motor rotates
    /// about chassis Z and the wheel "wobbles" around the steered direction
    /// once the wheel starts spinning.
    wheels: Vec<(
        ImpulseJointHandle,
        RigidBodyHandle,
        Vec3,
        bool,
        Option<ImpulseJointHandle>,
    )>,
    motor_max_velocity: f32,
    chassis_damp_yaw: f32,
    chassis_damp_tumble: f32,
}

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

fn load_car_ackermann(physics: &mut Physics) -> LoadedCar {
    load_car_ackermann_at(
        physics,
        nalgebra::Isometry3 {
            translation: nalgebra::Vector3::new(0.0, SPAWN_RADIUS, 0.0).into(),
            rotation: nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::y_axis(),
                0.5 * std::f32::consts::PI,
            ),
        },
    )
}

fn load_car_ackermann_at(physics: &mut Physics, transform: nalgebra::Isometry3<f32>) -> LoadedCar {
    let car_path = Path::new("data/cars/OxidizeMonk");
    let car_config: config::Car =
        ron::de::from_bytes(&fs::read(car_path.join("car.ron")).expect("car.ron"))
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

    // Top 4 corner balls.
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

    let chassis_rb = RigidBodyBuilder::dynamic()
        .pose(transform.into())
        .additional_mass_properties(mass_props)
        .linear_damping(0.4)
        .angular_damping(0.0)
        .build();
    let PhysicsBodyHandle {
        rigid_body_handle: chassis,
        ..
    } = physics.add_rigid_body(chassis_rb, chassis_colliders);

    let chassis_pose: rapier3d::math::Pose = transform.into();
    let mut wheels = Vec::new();
    for w in &car_config.wheels {
        let anchor_local = Vec3::new(w.position[0], w.position[1], w.position[2]);
        let is_steering = anchor_local.x < 0.0;
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

        // For steered wheels we need a knuckle between chassis and wheel.
        // Without the knuckle, a single GenericJoint's AngZ motor rotates the
        // wheel about chassis Z — which is wrong once the steer angle is non-
        // zero, because the wheel's actual axle (post-AngY) points elsewhere.
        // With the knuckle the AngZ motor lives on knuckle↔wheel, so it
        // rotates about the *knuckle*'s Z, which IS the steered axle.
        let steering_joint = if is_steering {
            let knuckle_body = RigidBodyBuilder::dynamic()
                .pose(rapier3d::math::Pose::from_parts(
                    wheel_world,
                    chassis_pose.rotation,
                ))
                .angular_damping(0.0)
                .additional_mass_properties(MassProperties::new(
                    Vec3::ZERO,
                    0.01,
                    Vec3::new(1e-4, 1e-4, 1e-4),
                ))
                .build();
            let PhysicsBodyHandle {
                rigid_body_handle: knuckle_rb,
                ..
            } = physics.add_rigid_body(knuckle_body, vec![]);

            // chassis ↔ knuckle: only AngY free (steering).
            let steer_locked = JointAxesMask::LIN_X
                | JointAxesMask::LIN_Y
                | JointAxesMask::LIN_Z
                | JointAxesMask::ANG_X
                | JointAxesMask::ANG_Z;
            let steer_joint = GenericJointBuilder::new(steer_locked)
                .local_anchor1(anchor_local)
                .local_anchor2(Vec3::ZERO)
                .contacts_enabled(false)
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

        // wheel_joint: handles suspension (LinY) and spin (AngZ). For steered
        // wheels this connects knuckle↔wheel; for rear wheels chassis↔wheel.
        // The wheel's AngY is always locked here — steering is upstream.
        let wheel_locked = JointAxesMask::LIN_X
            | JointAxesMask::LIN_Z
            | JointAxesMask::ANG_X
            | JointAxesMask::ANG_Y;
        let (parent_rb, parent_anchor) = match steering_joint {
            Some((knuckle_rb, _)) => (knuckle_rb, Vec3::ZERO),
            None => (chassis, anchor_local),
        };
        let builder = GenericJointBuilder::new(wheel_locked)
            .local_anchor1(parent_anchor)
            .local_anchor2(Vec3::ZERO)
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
            .motor_max_force(JointAxis::AngZ, car_config.motor_max_force);
        let joint_handle = physics.add_generic_joint(parent_rb, wheel_rb, builder.build());
        wheels.push((
            joint_handle,
            wheel_rb,
            anchor_local,
            is_steering,
            steering_joint.map(|(_, j)| j),
        ));
    }

    LoadedCar {
        chassis,
        wheels,
        motor_max_velocity: car_config.motor_max_velocity,
        chassis_damp_yaw: 0.15,
        chassis_damp_tumble: 2.0,
    }
}

fn apply_inputs(physics: &mut Physics, car: &LoadedCar, throttle: f32, steer: f32) {
    // Front-wheel drive (mirrors bin/game/main.rs::apply_driving_input):
    // only the steered wheels get the throttle motor; rear wheels roll freely
    // while driving and brake when idle. Steering is applied on the chassis ↔
    // knuckle joint (`steering_joint`), drive on the wheel joint.
    let drive_v = throttle * car.motor_max_velocity;
    let driving = drive_v != 0.0;
    let steer_angle = steer * MAX_STEER_ANGLE;
    for &(j, _, _, is_steering, steering_joint) in &car.wheels {
        let (target_v, factor) = if is_steering {
            if driving {
                (drive_v, 1.0)
            } else {
                (0.0, IDLE_BRAKE_FACTOR)
            }
        } else if driving {
            (drive_v, 0.01)
        } else {
            (0.0, IDLE_BRAKE_FACTOR)
        };
        physics.set_joint_motor_velocity(j, target_v, factor);
        if let Some(sj) = steering_joint {
            physics.set_joint_motor_position(
                sj,
                JointAxis::AngY,
                steer_angle,
                STEER_STIFFNESS,
                STEER_DAMPING,
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    tick: usize,
    chassis_pos: [f32; 3],
    chassis_yaw: f32,
    wheels: [[f32; 3]; 4],
}

fn settle_then_drive(
    name: &str,
    throttle: f32,
    steer: f32,
    settle_ticks: usize,
    drive_ticks: usize,
) -> Vec<Sample> {
    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);
    let car = load_car_ackermann(&mut physics);
    let start_rot = physics.get_transform(car.chassis).rotation;
    let yaw_of = |rot: nalgebra::UnitQuaternion<f32>| -> f32 {
        let f0 = start_rot * (-nalgebra::Vector3::x());
        let f1 = rot * (-nalgebra::Vector3::x());
        let cross_y = f0.z * f1.x - f0.x * f1.z;
        let dot = f0.x * f1.x + f0.z * f1.z;
        cross_y.atan2(dot)
    };

    // Settle.
    for _ in 0..settle_ticks {
        apply_inputs(&mut physics, &car, 0.0, 0.0);
        physics.update_gravity(&terrain);
        physics.apply_axial_angular_damping(
            car.chassis,
            car.chassis_damp_yaw,
            car.chassis_damp_tumble,
        );
        physics.step();
    }

    // Drive.
    let mut samples = Vec::new();
    for tick in 0..drive_ticks {
        apply_inputs(&mut physics, &car, throttle, steer);
        physics.update_gravity(&terrain);
        physics.apply_axial_angular_damping(
            car.chassis,
            car.chassis_damp_yaw,
            car.chassis_damp_tumble,
        );
        physics.step();
        if tick % 30 == 0 || tick + 1 == drive_ticks {
            let xform = physics.get_transform(car.chassis);
            let mut wheel_positions = [[0.0_f32; 3]; 4];
            for (i, &(_, rb, _, _, _)) in car.wheels.iter().enumerate() {
                let wp = physics.get_transform(rb).translation;
                wheel_positions[i] = [wp.x, wp.y, wp.z];
            }
            samples.push(Sample {
                tick,
                chassis_pos: [
                    xform.translation.x,
                    xform.translation.y,
                    xform.translation.z,
                ],
                chassis_yaw: yaw_of(xform.rotation),
                wheels: wheel_positions,
            });
        }
    }

    eprintln!("=== {name} (throttle={throttle:+.1} steer={steer:+.1}) ===");
    for s in &samples {
        eprintln!(
            "  tick={:3}  pos=({:6.2},{:6.2},{:6.2})  yaw={:+7.3} ({:+6.1}°)",
            s.tick,
            s.chassis_pos[0],
            s.chassis_pos[1],
            s.chassis_pos[2],
            s.chassis_yaw,
            s.chassis_yaw.to_degrees(),
        );
    }
    samples
}

/// Top-down (X-Z plane) SVG with the chassis trajectory and per-wheel position
/// markers. Y axis is "up" radially — we drop it because the flat plane keeps
/// it nearly constant.
fn write_svg(name: &str, samples: &[Sample], settled_pos: [f32; 3]) {
    use std::fmt::Write;
    let dir = Path::new("target/drive_svgs");
    fs::create_dir_all(dir).ok();
    let safe = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c
            } else if c == '+' {
                'p'
            } else {
                '_'
            }
        })
        .collect::<String>();
    let path = dir.join(format!("{safe}.svg"));

    // World-space bbox of all sampled positions plus a margin.
    let (mut min_x, mut max_x) = (settled_pos[0], settled_pos[0]);
    let (mut min_z, mut max_z) = (settled_pos[2], settled_pos[2]);
    for s in samples {
        min_x = min_x.min(s.chassis_pos[0]);
        max_x = max_x.max(s.chassis_pos[0]);
        min_z = min_z.min(s.chassis_pos[2]);
        max_z = max_z.max(s.chassis_pos[2]);
        for w in &s.wheels {
            min_x = min_x.min(w[0]);
            max_x = max_x.max(w[0]);
            min_z = min_z.min(w[2]);
            max_z = max_z.max(w[2]);
        }
    }
    let margin = ((max_x - min_x).max(max_z - min_z) * 0.1).max(0.5);
    min_x -= margin;
    max_x += margin;
    min_z -= margin;
    max_z += margin;
    let w = (max_x - min_x).max(0.01);
    let h = (max_z - min_z).max(0.01);
    let scale = 800.0_f32 / w.max(h);
    let canvas_w = (w * scale).max(100.0);
    let canvas_h = (h * scale).max(100.0);
    // World (X, Z) → SVG (px, py). SVG y grows downward; we flip Z so increasing
    // world Z (the car's forward direction at spawn) renders upward.
    let to_svg = |x: f32, z: f32| -> (f32, f32) {
        let px = (x - min_x) * scale;
        let py = canvas_h - (z - min_z) * scale;
        (px, py)
    };

    let mut svg = String::new();
    writeln!(
        &mut svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"##,
        canvas_w as i32, canvas_h as i32, canvas_w as i32, canvas_h as i32,
    )
    .unwrap();
    writeln!(
        &mut svg,
        r##"<rect width="100%" height="100%" fill="#1a1d20"/>"##
    )
    .unwrap();
    // Grid every 1 m.
    let grid_color = "#2c3035";
    let mut gx = min_x.ceil();
    while gx <= max_x {
        let (px, _) = to_svg(gx, 0.0);
        writeln!(
            &mut svg,
            r##"<line x1="{px}" y1="0" x2="{px}" y2="{canvas_h}" stroke="{grid_color}" stroke-width="1"/>"##,
        )
        .unwrap();
        gx += 1.0;
    }
    let mut gz = min_z.ceil();
    while gz <= max_z {
        let (_, py) = to_svg(0.0, gz);
        writeln!(
            &mut svg,
            r##"<line x1="0" y1="{py}" x2="{canvas_w}" y2="{py}" stroke="{grid_color}" stroke-width="1"/>"##,
        )
        .unwrap();
        gz += 1.0;
    }

    // Trajectory polyline.
    let mut poly = String::new();
    let (sx, sz) = to_svg(settled_pos[0], settled_pos[2]);
    poly.push_str(&format!("{sx},{sz} "));
    for s in samples {
        let (px, py) = to_svg(s.chassis_pos[0], s.chassis_pos[2]);
        poly.push_str(&format!("{px},{py} "));
    }
    writeln!(
        &mut svg,
        r##"<polyline points="{poly}" stroke="#e2c067" stroke-width="2" fill="none"/>"##
    )
    .unwrap();

    // Per-sample wheel markers and a chassis dot.
    let n = samples.len() as f32;
    for (idx, s) in samples.iter().enumerate() {
        // Fade from cool to warm across the timeline.
        let t = if n > 1.0 { idx as f32 / (n - 1.0) } else { 0.0 };
        let r = (60.0 + 195.0 * t) as i32;
        let g = (140.0 + 50.0 * (1.0 - t)) as i32;
        let b = (255.0 * (1.0 - t)) as i32;
        let color = format!("rgb({r},{g},{b})");
        let (px, py) = to_svg(s.chassis_pos[0], s.chassis_pos[2]);
        writeln!(
            &mut svg,
            r##"<circle cx="{px}" cy="{py}" r="4" fill="{color}" stroke="white" stroke-width="0.5"/>"##,
        )
        .unwrap();
        for (wi, w) in s.wheels.iter().enumerate() {
            let (wx, wy) = to_svg(w[0], w[2]);
            // First two wheels are at chassis-local +X (rear); last two are -X
            // (front, steered). Mark front wheels with a different colour.
            let wheel_color = if wi < 2 { "#80d080" } else { "#ff6b6b" };
            writeln!(
                &mut svg,
                r##"<circle cx="{wx}" cy="{wy}" r="2" fill="{wheel_color}"/>"##,
            )
            .unwrap();
        }
    }

    // Legend.
    writeln!(
        &mut svg,
        r##"<text x="8" y="16" font-family="monospace" font-size="14" fill="#eee">{name}</text>"##,
    )
    .unwrap();
    writeln!(
        &mut svg,
        r##"<text x="8" y="32" font-family="monospace" font-size="10" fill="#80d080">rear wheels</text>"##,
    )
    .unwrap();
    writeln!(
        &mut svg,
        r##"<text x="8" y="46" font-family="monospace" font-size="10" fill="#ff6b6b">front wheels (steered)</text>"##,
    )
    .unwrap();
    writeln!(
        &mut svg,
        r##"<text x="8" y="60" font-family="monospace" font-size="10" fill="#e2c067">chassis trajectory</text>"##,
    )
    .unwrap();
    writeln!(&mut svg, "</svg>").unwrap();
    fs::write(&path, svg).unwrap();
    eprintln!("  → {}", path.display());
}

fn run_combo(name: &str, throttle: f32, steer: f32) -> Vec<Sample> {
    // 4 s settle (matches the upside-down/stuck tests) so the suspension is
    // fully relaxed before we start driving. 3 s of driving captures a
    // useful trajectory without running the chassis off the cylinder.
    let samples = settle_then_drive(name, throttle, steer, 240, 180);
    // Use the first sample as the "settled" anchor for the SVG.
    let anchor = samples.first().map(|s| s.chassis_pos).unwrap_or([0.0; 3]);
    write_svg(name, &samples, anchor);
    samples
}

fn horiz_displacement(start: [f32; 3], end: [f32; 3]) -> f32 {
    let dx = end[0] - start[0];
    let dz = end[2] - start[2];
    (dx * dx + dz * dz).sqrt()
}

#[test]
fn w_drives_forward() {
    let samples = run_combo("W", 1.0, 0.0);
    let start = samples.first().unwrap().chassis_pos;
    let end = samples.last().unwrap().chassis_pos;
    // World +Z is "forward" given the spawn rotation; W should make Z increase.
    let dz = end[2] - start[2];
    assert!(dz > 0.5, "W: expected forward (+Z) motion, got dz={dz:.3}");
    let yaw = samples.last().unwrap().chassis_yaw.abs();
    assert!(yaw < 0.2, "W: expected ~no yaw, got {yaw:.3} rad");
}

#[test]
fn s_drives_backward() {
    let samples = run_combo("S", -1.0, 0.0);
    let start = samples.first().unwrap().chassis_pos;
    let end = samples.last().unwrap().chassis_pos;
    let dz = end[2] - start[2];
    assert!(
        dz < -0.5,
        "S: expected backward (-Z) motion, got dz={dz:.3}"
    );
}

#[test]
fn a_at_standstill_does_not_drive() {
    let samples = run_combo("A", 0.0, -1.0);
    let start = samples.first().unwrap().chassis_pos;
    let end = samples.last().unwrap().chassis_pos;
    let dist = horiz_displacement(start, end);
    // The Ackermann steering wheels should turn but the chassis must not yaw
    // or drift far while standing still.
    assert!(
        dist < 0.4,
        "A at standstill should not move the chassis; got |xz|={dist:.3} m"
    );
    let yaw = samples.last().unwrap().chassis_yaw.abs();
    assert!(
        yaw < 0.2,
        "A at standstill should not yaw the chassis; got {yaw:.3} rad"
    );
}

#[test]
fn d_at_standstill_does_not_drive() {
    let samples = run_combo("D", 0.0, 1.0);
    let start = samples.first().unwrap().chassis_pos;
    let end = samples.last().unwrap().chassis_pos;
    let dist = horiz_displacement(start, end);
    assert!(
        dist < 0.4,
        "D at standstill should not move the chassis; got |xz|={dist:.3} m"
    );
    let yaw = samples.last().unwrap().chassis_yaw.abs();
    assert!(
        yaw < 0.2,
        "D at standstill should not yaw the chassis; got {yaw:.3} rad"
    );
}

// W+A, W+D, S+A, S+D use *partial* steer (±0.5) rather than full-lock,
// because with the stronger ForceBased steering motor a full-lock turn at
// full throttle spins the chassis in place rather than driving forward —
// realistic, but the test wants to verify "turn while moving", not "spin in
// place".
#[test]
fn wa_turns_left_while_moving_forward() {
    let samples = run_combo("WpA", 1.0, -0.5);
    let start = samples.first().unwrap().chassis_pos;
    let end = samples.last().unwrap().chassis_pos;
    let dz = end[2] - start[2];
    assert!(
        dz > 0.3,
        "W+A: expected forward motion alongside the turn, got dz={dz:.3}"
    );
    let final_yaw = samples.last().unwrap().chassis_yaw;
    let peak_yaw = samples
        .iter()
        .map(|s| s.chassis_yaw)
        .fold(0.0_f32, |acc, y| if y.abs() > acc.abs() { y } else { acc });
    // The chassis spawns with rotation 90° about world +Y. With this spawn the
    // yaw-of helper returns *negative* values when the chassis yaws LEFT.
    assert!(
        peak_yaw < -0.15,
        "W+A: expected left (negative) yaw, peak was {peak_yaw:.3} rad, final {final_yaw:.3} rad"
    );
}

#[test]
fn wd_turns_right_while_moving_forward() {
    let samples = run_combo("WpD", 1.0, 0.5);
    let start = samples.first().unwrap().chassis_pos;
    let end = samples.last().unwrap().chassis_pos;
    let dz = end[2] - start[2];
    assert!(dz > 0.3, "W+D: expected forward motion, got dz={dz:.3}");
    let peak_yaw = samples
        .iter()
        .map(|s| s.chassis_yaw)
        .fold(0.0_f32, |acc, y| if y.abs() > acc.abs() { y } else { acc });
    assert!(
        peak_yaw > 0.15,
        "W+D: expected right (positive) yaw, peak was {peak_yaw:.3} rad"
    );
}

#[test]
fn sa_reverses_and_yaws_opposite_to_wa() {
    let samples = run_combo("SpA", -1.0, -0.5);
    let start = samples.first().unwrap().chassis_pos;
    let end = samples.last().unwrap().chassis_pos;
    let dz = end[2] - start[2];
    assert!(dz < -0.3, "S+A: expected backward motion, got dz={dz:.3}");
    // Reversing with the same steer flips the yaw direction relative to W+A.
    let peak_yaw = samples
        .iter()
        .map(|s| s.chassis_yaw)
        .fold(0.0_f32, |acc, y| if y.abs() > acc.abs() { y } else { acc });
    assert!(
        peak_yaw > 0.15,
        "S+A: expected right (positive) yaw under reverse, peak was {peak_yaw:.3} rad"
    );
}

#[test]
fn sd_reverses_and_yaws_opposite_to_wd() {
    let samples = run_combo("SpD", -1.0, 0.5);
    let start = samples.first().unwrap().chassis_pos;
    let end = samples.last().unwrap().chassis_pos;
    let dz = end[2] - start[2];
    assert!(dz < -0.3, "S+D: expected backward motion, got dz={dz:.3}");
    let peak_yaw = samples
        .iter()
        .map(|s| s.chassis_yaw)
        .fold(0.0_f32, |acc, y| if y.abs() > acc.abs() { y } else { acc });
    assert!(
        peak_yaw < -0.15,
        "S+D: expected left (negative) yaw under reverse, peak was {peak_yaw:.3} rad"
    );
}

/// Sanity check: in the settled pose, each wheel rigid body must sit at its
/// declared anchor position transformed by the chassis pose. If this drifts,
/// the joint anchor is wrong or the suspension is settling outside its limits.
#[test]
fn wheel_rigid_bodies_settle_at_anchor_positions() {
    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);
    let car = load_car_ackermann(&mut physics);

    // Settle.
    for _ in 0..240 {
        apply_inputs(&mut physics, &car, 0.0, 0.0);
        physics.update_gravity(&terrain);
        physics.apply_axial_angular_damping(
            car.chassis,
            car.chassis_damp_yaw,
            car.chassis_damp_tumble,
        );
        physics.step();
    }

    let chassis_xform = physics.get_transform(car.chassis);
    eprintln!(
        "chassis settled at pos=({:.3},{:.3},{:.3})",
        chassis_xform.translation.x, chassis_xform.translation.y, chassis_xform.translation.z,
    );

    for (i, &(_, rb, anchor_local, is_steering, _)) in car.wheels.iter().enumerate() {
        let wp = physics.get_transform(rb).translation;
        // Expected: chassis_pose * anchor_local, modulo suspension travel (which
        // moves the wheel only along chassis-Y).
        let anchor_local_nl =
            nalgebra::Vector3::new(anchor_local.x, anchor_local.y, anchor_local.z);
        let anchor_world =
            chassis_xform.translation.vector + chassis_xform.rotation * anchor_local_nl;
        let dx = wp.x - anchor_world.x;
        let dy = wp.y - anchor_world.y;
        let dz = wp.z - anchor_world.z;
        eprintln!(
            "  wheel {i} ({}{}): pos=({:.3},{:.3},{:.3}) anchor_world=({:.3},{:.3},{:.3}) Δ=({:.3},{:.3},{:.3})",
            if is_steering { "front" } else { "rear" },
            if anchor_local.z > 0.0 { "-L" } else { "-R" },
            wp.x, wp.y, wp.z,
            anchor_world.x, anchor_world.y, anchor_world.z,
            dx, dy, dz,
        );
        // The suspension allows ±0.3 m of chassis-Y travel; the wheel position
        // can drift that far along the radial direction but should match in
        // the tangential plane to within a few centimetres.
        let radial_xform = chassis_xform.rotation * nalgebra::Vector3::new(0.0, 1.0, 0.0);
        let drift = nalgebra::Vector3::new(dx, dy, dz);
        let radial_component = drift.dot(&radial_xform);
        let tangential = drift - radial_xform * radial_component;
        let tang_mag = tangential.norm();
        assert!(
            tang_mag < 0.05,
            "wheel {i} drifted tangentially by {tang_mag:.3} m (anchor wrong?)"
        );
        assert!(
            radial_component.abs() < 0.5,
            "wheel {i} radial drift = {radial_component:.3} m (suspension limit exceeded?)"
        );
    }
}

/// Spawn the chassis high above the terrain so no wheel ever touches the
/// ground, then apply forward + right and verify that
///   (a) the front wheels actually reach their steering target in chassis
///       frame and stay there (currently they wobble),
///   (b) all four wheels accumulate a sustained spin (AngZ in chassis frame).
///
/// Without ground contact the joint motors are the only force acting on the
/// wheels, so any wobble or drive failure is purely a joint-motor problem.
#[test]
fn wheels_hold_steer_and_spin_with_no_ground_contact() {
    use rapier3d::dynamics::JointAxis;
    use rapier3d::math::{Pose, Vec3};

    let mut physics = Physics::default();
    let terrain = build_flat_terrain(&mut physics);

    // Spawn far above the outer cylinder (radius_end = 20 m). Gravity will
    // pull the chassis down a bit during the test but it won't reach ground.
    let spawn_radius = 50.0_f32;
    let spawn_pose = nalgebra::Isometry3 {
        translation: nalgebra::Vector3::new(0.0, spawn_radius, 0.0).into(),
        rotation: nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            0.5 * std::f32::consts::PI,
        ),
    };
    let car = load_car_ackermann_at(&mut physics, spawn_pose);

    // Forward + right.
    let throttle = 1.0_f32;
    let steer = 1.0_f32;

    // Run for 1 second of physics. By the end we expect:
    //   - all four wheels' AngZ chassis-relative angular velocity > 0
    //   - the two front wheels' AngY chassis-relative position ≈ +MAX_STEER_ANGLE
    let mut steer_samples: Vec<(f32, f32)> = Vec::new(); // (front-left AngY, front-right AngY)
    let mut spin_samples: Vec<[f32; 4]> = Vec::new();
    for tick in 0..120 {
        apply_inputs(&mut physics, &car, throttle, steer);
        physics.update_gravity(&terrain);
        physics.step();

        if tick % 10 == 9 {
            let chassis_pose: Pose = physics.get_transform(car.chassis).into();
            let chassis_inv = chassis_pose.inverse();
            let chassis_angvel_world = physics.body_angvel(car.chassis);
            let chassis_angvel_local = chassis_pose.rotation.inverse() * chassis_angvel_world;
            eprintln!(
                "tick={tick:3} chassis ω_local=({:+.2},{:+.2},{:+.2}) ω_world=({:+.2},{:+.2},{:+.2})",
                chassis_angvel_local.x, chassis_angvel_local.y, chassis_angvel_local.z,
                chassis_angvel_world.x, chassis_angvel_world.y, chassis_angvel_world.z,
            );
            let mut steer_pair = (f32::NAN, f32::NAN);
            let mut spin_quad = [0.0_f32; 4];
            for (i, &(_, rb, anchor_local, is_steering, _)) in car.wheels.iter().enumerate() {
                let wheel_pose: Pose = physics.get_transform(rb).into();
                let rel = chassis_inv * wheel_pose;
                // The wheel's Z axis (spin axle) is invariant under spin
                // (AngZ rotation is *about* this axis) and is the only axis
                // affected by steering (AngY rotation about chassis +Y). After
                // AngY = θ the axle reads (sin θ, 0, cos θ) in chassis frame.
                let z_in_chassis = rel.rotation * Vec3::new(0.0, 0.0, 1.0);
                let y_in_chassis = rel.rotation * Vec3::new(0.0, 1.0, 0.0);
                let ang_y_local = z_in_chassis.x.atan2(z_in_chassis.z);
                // Diagnostic: if ANG_X were truly locked, the wheel's Y axis
                // would always be exactly chassis Y. Any deviation is the
                // joint failing to hold its locked axis.
                let y_tilt_x = y_in_chassis.x;
                let y_tilt_z = y_in_chassis.z;
                // Spin: project wheel angular velocity onto wheel-local Z, then
                // compare to chassis-local Z to read the AngZ joint axis rate.
                let wheel_angvel_world = physics.body_angvel(rb);
                let chassis_z_world = chassis_pose.rotation * Vec3::new(0.0, 0.0, 1.0);
                let spin = chassis_z_world.dot(wheel_angvel_world);
                spin_quad[i] = spin;
                if is_steering {
                    // anchor_local.z > 0 → "front-left" in chassis frame (anchor_local.x < 0
                    // is the steering side, z>0 is +Z).
                    if anchor_local.z > 0.0 {
                        steer_pair.0 = ang_y_local;
                    } else {
                        steer_pair.1 = ang_y_local;
                    }
                }
                eprintln!(
                    "tick={tick:3} wheel {i} {} AngY={ang_y_local:+.3} ({:.1}°) spin={spin:+.3} y_tilt=({:+.3},{:+.3})",
                    if is_steering { "front" } else { "rear " },
                    ang_y_local.to_degrees(),
                    y_tilt_x, y_tilt_z,
                );
            }
            steer_samples.push(steer_pair);
            spin_samples.push(spin_quad);
            eprintln!();
        }
    }

    let last_steer = steer_samples.last().copied().unwrap();
    let last_spin = spin_samples.last().copied().unwrap();
    let target = MAX_STEER_ANGLE;
    eprintln!(
        "final: front_l_steer={:.3} ({:.1}°) front_r_steer={:.3} ({:.1}°) target={:.3} ({:.1}°)",
        last_steer.0,
        last_steer.0.to_degrees(),
        last_steer.1,
        last_steer.1.to_degrees(),
        target,
        target.to_degrees(),
    );
    eprintln!("final spin: {:?}", last_spin);

    // Front wheels should be steered right (positive AngY) within a small
    // tolerance of the target. Tight tolerance — without ground contact and
    // with the high-torque steering motor, the front wheels should reach the
    // target near-instantly.
    let tol = 0.05_f32; // ~3°
    assert!(
        (last_steer.0 - target).abs() < tol,
        "front-left wheel did not hold steer target: {:+.3} rad vs {:+.3}",
        last_steer.0,
        target
    );
    assert!(
        (last_steer.1 - target).abs() < tol,
        "front-right wheel did not hold steer target: {:+.3} rad vs {:+.3}",
        last_steer.1,
        target
    );

    // All four wheels should be spinning forward (chassis +Z dot wheel
    // angular velocity > 0). Drive velocity target is motor_max_velocity but
    // without ground we expect some appreciable fraction — at least 1 rad/s.
    for (i, s) in last_spin.iter().enumerate() {
        assert!(
            *s > 1.0,
            "wheel {i} not spinning forward in chassis frame: {s:+.3} rad/s"
        );
    }
    let _ = JointAxis::AngY; // (currently unused, kept for future variants)
}
