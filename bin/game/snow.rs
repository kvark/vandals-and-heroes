//! Debug snow particles.
//!
//! Each particle is a small rapier `Ball` collider that spawns near the world's
//! outer shell, falls under the same radial gravity the chassis sees, and lands
//! on the same heightfield collider. Where snow accumulates *is* where the
//! physics surface is — making any mismatch between the visual heightmap and
//! the physics heightfield (or any inconsistency between cylinder and sphere
//! worlds) immediately obvious.
//!
//! Each particle is rendered as a tiny white sphere (procedural mesh; no GLB
//! assets needed). Particles that have been slow-moving for a while get
//! respawned at the top.

use nalgebra::{Point3, Vector3};
use std::sync::Arc;
use vandals_and_heroes::{
    GeometryDesc, Loader, MaterialDesc, Model, ModelDesc, ModelInstance, Physics,
    PhysicsBodyHandle, VertexDesc,
};

/// Particle radius (m). Sized to be unambiguously a sphere in the camera
/// view but small enough that they read as "snow" rather than rocks.
const PARTICLE_RADIUS: f32 = 0.12;
/// Particle density. Snow is *light* — we only want gravity to pull it inward,
/// not let it dig into the terrain when it lands.
const PARTICLE_DENSITY: f32 = 0.2;
/// Linear damping applied to every particle. Without this they slide
/// indefinitely on flat ground (rapier doesn't have its own friction for
/// shape-shape pairs that go through our custom dispatcher).
const PARTICLE_LINEAR_DAMPING: f32 = 0.6;
/// Speed below which we consider a particle "settled". If it stays under this
/// threshold for [`SETTLE_TICKS`] physics ticks we recycle it.
const SETTLE_SPEED: f32 = 0.05;
/// Number of consecutive ticks under [`SETTLE_SPEED`] before recycling.
/// Long enough (20 s) for the accumulation pattern to be visible during
/// analysis; short enough that a stream of new particles is always falling.
const SETTLE_TICKS: u32 = 1200;
/// How far above the outer radius we spawn fresh particles. Just outside the
/// shell so they drop in from a slight height.
const SPAWN_RADIUS_OFFSET: f32 = 0.05;

pub struct Snow {
    pub model: Arc<Model>,
    pub instances: Vec<ModelInstance>,
    bodies: Vec<rapier3d::dynamics::RigidBodyHandle>,
    settled_ticks: Vec<u32>,
    is_sphere: bool,
    radius_end: f32,
    /// Cylinder z-band: ±[`CYLINDER_Z_HALF_BAND`] m centred here. Cylinders are
    /// long enough that uniformly-distributed snow vanishes off-camera; biasing
    /// to a fixed band keeps the debug view dense. Ignored in sphere mode.
    cylinder_z_center: f32,
    rng_state: u64,
    debug_tick: u32,
}

/// How far (in metres) cylinder-mode snow spawns either side of the car's
/// initial z. 30 m corresponds to roughly twice the car's clip-near + chase
/// camera distance, so spawning here keeps a comfortable density right where
/// the player is looking.
const CYLINDER_Z_HALF_BAND: f32 = 30.0;

impl Snow {
    pub fn new(
        loader: &mut Loader,
        physics: &mut Physics,
        count: usize,
        is_sphere: bool,
        radius_end: f32,
        cylinder_z_center: f32,
    ) -> Self {
        let model = Arc::new(loader.load_model(&snowflake_mesh_desc(PARTICLE_RADIUS)));
        let mut snow = Self {
            model,
            instances: Vec::with_capacity(count),
            bodies: Vec::with_capacity(count),
            settled_ticks: vec![0; count],
            is_sphere,
            radius_end,
            cylinder_z_center,
            // Arbitrary seed; the LCG below shuffles enough across the count.
            rng_state: 0x1234_5678_9abc_def0,
            debug_tick: 0,
        };
        for _ in 0..count {
            let (pos, rot) = snow.sample_spawn();
            let body = rapier3d::dynamics::RigidBodyBuilder::dynamic()
                .pose(rapier3d::math::Pose::from_parts(pos, rot))
                .linear_damping(PARTICLE_LINEAR_DAMPING)
                .angular_damping(0.2)
                .build();
            let collider = rapier3d::geometry::ColliderBuilder::ball(PARTICLE_RADIUS)
                .density(PARTICLE_DENSITY)
                .friction(0.5)
                .build();
            let PhysicsBodyHandle {
                rigid_body_handle, ..
            } = physics.add_rigid_body(body, vec![collider]);
            snow.bodies.push(rigid_body_handle);
            snow.instances.push(ModelInstance {
                model: snow.model.clone(),
                transform: nalgebra::Isometry3 {
                    translation: nalgebra::Vector3::new(pos.x, pos.y, pos.z).into(),
                    rotation: nalgebra::UnitQuaternion::identity(),
                },
                geometry_filter: None,
            });
        }
        snow
    }

    /// Advance one physics tick worth of bookkeeping: sync the render
    /// instances to the rigid bodies' poses, and recycle particles that have
    /// been still long enough.
    pub fn update(&mut self, physics: &mut Physics) {
        for i in 0..self.bodies.len() {
            let pose = physics.get_transform(self.bodies[i]);
            self.instances[i].transform = pose;
            let speed = physics
                .body_kinematics(self.bodies[i])
                .map(|k| (k.linvel[0].powi(2) + k.linvel[1].powi(2) + k.linvel[2].powi(2)).sqrt())
                .unwrap_or(0.0);
            if speed < SETTLE_SPEED {
                self.settled_ticks[i] += 1;
            } else {
                self.settled_ticks[i] = 0;
            }
            if self.settled_ticks[i] >= SETTLE_TICKS {
                let (pos, _rot) = self.sample_spawn();
                physics.teleport_body(self.bodies[i], pos);
                self.settled_ticks[i] = 0;
            }
        }
        // Debug: every 5 s of physics ticks, dump a radial histogram so the
        // log shows where particles settle vs. the heightmap's expected
        // distribution. Disable by setting `debug_tick` past u32::MAX/2.
        self.debug_tick += 1;
        if self.debug_tick % 300 == 0 {
            let mut bins = [0u32; 12];
            let mut moving = 0u32;
            for &b in &self.bodies {
                let p = physics.get_transform(b).translation.vector;
                // Cylinder gravity sees only XY radius; sphere sees full 3D.
                // Match the heightfield's parameterisation so a histogram bin
                // means the same thing as the heightmap's `mix(start, end, α)`.
                let r = if self.is_sphere {
                    (p.x * p.x + p.y * p.y + p.z * p.z).sqrt()
                } else {
                    (p.x * p.x + p.y * p.y).sqrt()
                };
                let k = physics.body_kinematics(b).unwrap();
                let sp = (k.linvel[0].powi(2) + k.linvel[1].powi(2) + k.linvel[2].powi(2)).sqrt();
                if sp >= SETTLE_SPEED {
                    moving += 1;
                }
                // Bin into 1-m buckets from 9 to 21.
                let bin = ((r - 9.0).clamp(0.0, 11.999) as usize).min(11);
                bins[bin] += 1;
            }
            log::info!("snow r-histogram (moving={}): {:?}", moving, bins);
        }
    }

    pub fn free(&self, ctx: &blade_graphics::Context) {
        self.model.free(ctx);
    }

    /// Pick a random spawn point on the outer shell.
    fn sample_spawn(&mut self) -> (rapier3d::math::Vec3, rapier3d::math::Rotation) {
        let theta = self.rand_f32() * std::f32::consts::TAU;
        let pos = if self.is_sphere {
            // Uniform on a sphere: sin φ ∈ [-1, 1] uniform gives equal area.
            let sin_phi = self.rand_f32() * 2.0 - 1.0;
            let cos_phi = (1.0 - sin_phi * sin_phi).max(0.0).sqrt();
            let r = self.radius_end + SPAWN_RADIUS_OFFSET;
            rapier3d::math::Vec3::new(
                r * cos_phi * theta.cos(),
                r * cos_phi * theta.sin(),
                r * sin_phi,
            )
        } else {
            // Cylinder: random theta around the axis, random z in the band
            // around the car's spawn z so cylindrical-long worlds stay covered
            // in the camera view without us needing thousands of particles.
            let z = self.cylinder_z_center + (self.rand_f32() - 0.5) * (2.0 * CYLINDER_Z_HALF_BAND);
            let r = self.radius_end + SPAWN_RADIUS_OFFSET;
            rapier3d::math::Vec3::new(r * theta.cos(), r * theta.sin(), z)
        };
        (pos, rapier3d::math::Rotation::IDENTITY)
    }

    /// 64-bit LCG → f32 ∈ [0, 1). Numerical-recipes constants; quality is
    /// fine for debug particle scattering.
    fn rand_f32(&mut self) -> f32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 32) as u32 as f32) / (u32::MAX as f32 + 1.0)
    }
}

fn snowflake_mesh_desc(radius: f32) -> ModelDesc {
    // Low-poly icosphere-ish: octahedron with each face split once. 18 vertices,
    // 32 triangles. Plenty for a 0.12 m particle.
    use nalgebra::{Point2, Vector3 as V3};
    let mut vertices: Vec<VertexDesc> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();

    let octahedron_vertices = [
        V3::new(1.0, 0.0, 0.0),
        V3::new(-1.0, 0.0, 0.0),
        V3::new(0.0, 1.0, 0.0),
        V3::new(0.0, -1.0, 0.0),
        V3::new(0.0, 0.0, 1.0),
        V3::new(0.0, 0.0, -1.0),
    ];
    let octahedron_faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];

    // For each octahedron face, split into 4 triangles by adding midpoints.
    for face in octahedron_faces {
        let a = octahedron_vertices[face[0]];
        let b = octahedron_vertices[face[1]];
        let c = octahedron_vertices[face[2]];
        let ab = ((a + b) * 0.5).normalize();
        let bc = ((b + c) * 0.5).normalize();
        let ca = ((c + a) * 0.5).normalize();
        let base = vertices.len() as u32;
        for v in [a, b, c, ab, bc, ca] {
            vertices.push(VertexDesc {
                pos: Point3::from(v * radius),
                tex_coords: Point2::new(0.5, 0.5),
                normal: v,
            });
        }
        // a(0) - ab(3) - ca(5)
        indices.push([base, base + 3, base + 5]);
        // ab(3) - b(1) - bc(4)
        indices.push([base + 3, base + 1, base + 4]);
        // ca(5) - bc(4) - c(2)
        indices.push([base + 5, base + 4, base + 2]);
        // ab(3) - bc(4) - ca(5)
        indices.push([base + 3, base + 4, base + 5]);
    }

    let _ = (Point3::<f32>::origin(), Vector3::<f32>::zeros());
    let materials = vec![
        MaterialDesc::default(),
        MaterialDesc {
            name: Some("snow".to_string()),
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            normal_scale: 0.0,
            transparent: false,
        },
    ];
    let geometry = GeometryDesc {
        name: "snow_particle".to_string(),
        vertices,
        indices,
        index_type: Some(blade_graphics::IndexType::U32),
        transform: nalgebra::Matrix4::identity(),
        material_index: 1,
    };
    ModelDesc {
        materials,
        geometries: vec![geometry],
    }
}
