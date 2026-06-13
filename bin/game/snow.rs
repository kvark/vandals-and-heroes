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

/// Particle radius (m). Small enough to read as "snow" at the density used
/// (≈10× the original count); large enough not to tunnel through terrain at
/// terminal velocity (~1.2 m/s with the linear damping below; per-tick step
/// is ~0.02 m ≪ radius).
const PARTICLE_RADIUS: f32 = 0.05;
/// Particle density. Snow is *light* — we only want gravity to pull it inward,
/// not let it dig into the terrain when it lands.
const PARTICLE_DENSITY: f32 = 0.2;
/// Linear damping applied to every particle. Without this they slide
/// indefinitely on flat ground (rapier doesn't have its own friction for
/// shape-shape pairs that go through our custom dispatcher).
const PARTICLE_LINEAR_DAMPING: f32 = 0.6;
/// Lifetime range, in physics ticks (60 Hz): each particle is assigned a
/// random total lifetime in [MIN, MAX] when it spawns. When `age_ticks`
/// passes its lifetime the particle teleports back to the outer shell and
/// gets a new random lifetime. Range chosen so the in-flight + settled
/// portion both stay visible — 6 s minimum, 40 s maximum.
const LIFETIME_MIN_TICKS: u32 = 360;
const LIFETIME_MAX_TICKS: u32 = 2400;
/// How far above the outer radius we spawn fresh particles. Just outside the
/// shell so they drop in from a slight height.
const SPAWN_RADIUS_OFFSET: f32 = 0.05;

/// `*mut T` newtype that promises (via unsafe) it's OK to send & share across
/// threads. Used inside [`Snow::update`] to give each Choir worker a raw
/// pointer to disjoint index ranges of the snow buffers; safety relies on
/// the caller never letting two workers touch the same slot.
///
/// Use the helper methods rather than touching `.0` directly — accessing the
/// field inside a closure triggers Rust 2021 fine-grained captures, which
/// would re-capture the bare `*mut T` (not Send) instead of the whole
/// `SyncSlice` (Send via the unsafe impls below).
#[derive(Copy, Clone)]
struct SyncSlice<T>(*mut T);
unsafe impl<T> Send for SyncSlice<T> {}
unsafe impl<T> Sync for SyncSlice<T> {}

impl<T> SyncSlice<T> {
    /// SAFETY: `i` must be a valid index in the underlying buffer and the
    /// caller must guarantee no concurrent writer touches the same slot.
    /// Returns a `'static` lifetime to satisfy the closure's `'static` bound;
    /// the caller (a Choir worker scoped to `Snow::update`) must not let the
    /// reference escape past the `task.join()` call.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self, i: usize) -> &'static mut T {
        unsafe { &mut *self.0.add(i) }
    }
    /// SAFETY: `i` must be a valid index in the underlying buffer.
    #[inline]
    unsafe fn read(&self, i: usize) -> T
    where
        T: Copy,
    {
        unsafe { *self.0.add(i) }
    }
    /// SAFETY: `i` must be a valid index in the underlying buffer and the
    /// caller must guarantee no concurrent writer touches the same slot.
    #[inline]
    unsafe fn write(&self, i: usize, value: T) {
        unsafe { *self.0.add(i) = value }
    }
}

pub struct Snow {
    pub model: Arc<Model>,
    pub instances: Vec<ModelInstance>,
    bodies: Vec<rapier3d::dynamics::RigidBodyHandle>,
    /// Tick counter per particle; reset to 0 on respawn.
    age_ticks: Vec<u32>,
    /// Lifetime in ticks per particle; redrawn from `[MIN, MAX]` on respawn so
    /// each particle's recycle moment is uncorrelated with the others.
    lifetime_ticks: Vec<u32>,
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
            age_ticks: Vec::with_capacity(count),
            lifetime_ticks: Vec::with_capacity(count),
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
            // Stagger initial ages over [0, lifetime) so respawn moments are
            // uncorrelated from the very first tick — otherwise the first
            // generation of particles would all expire together.
            let lifetime = snow.rand_lifetime();
            let initial_age = snow.rand_uniform_u32(lifetime);
            snow.lifetime_ticks.push(lifetime);
            snow.age_ticks.push(initial_age);
        }
        snow
    }

    fn rand_lifetime(&mut self) -> u32 {
        LIFETIME_MIN_TICKS + self.rand_uniform_u32(LIFETIME_MAX_TICKS - LIFETIME_MIN_TICKS)
    }

    fn rand_uniform_u32(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            return 0;
        }
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 32) as u32) % upper
    }

    /// Advance one physics tick worth of bookkeeping: sync the render
    /// instances to the rigid bodies' poses, age each particle, and recycle
    /// particles that have passed their (random) lifetime.
    ///
    /// The per-particle bookkeeping (transform sync + age increment +
    /// lifetime check) is fanned out to a Choir `init_multi` indexed task so
    /// each worker handles a disjoint slice. The actual `teleport_body` calls
    /// mutate rapier's RigidBodySet and have to stay serial.
    pub fn update(&mut self, physics: &mut Physics, choir: &Arc<choir::Choir>) {
        let n = self.bodies.len();
        if n == 0 {
            return;
        }

        // Phase 1 (serial): fetch every body's current pose. Reading from
        // rapier's RigidBodySet across multiple threads isn't supported, so
        // we snapshot once here.
        let poses: Vec<nalgebra::Isometry3<f32>> = self
            .bodies
            .iter()
            .map(|&b| physics.get_transform(b))
            .collect();

        // Phase 2 (parallel): write transforms, age, and flag particles
        // whose lifetime has expired. Workers operate on disjoint
        // [start..end) index ranges so the raw-pointer writes don't race.
        let mut respawn_flags: Vec<u8> = vec![0; n];
        // SAFETY: the slices these point into (`self.instances`,
        // `self.age_ticks`, `respawn_flags`, `self.lifetime_ticks`) live
        // for the duration of this function and aren't reallocated. Each
        // worker is restricted to a unique [start..end) range computed
        // from its SubIndex below, so no two workers ever write to the
        // same slot.
        let instances_p = SyncSlice(self.instances.as_mut_ptr());
        let ages_p = SyncSlice(self.age_ticks.as_mut_ptr());
        let lifetimes_p = SyncSlice(self.lifetime_ticks.as_ptr() as *mut u32);
        let respawn_p = SyncSlice(respawn_flags.as_mut_ptr());

        let workers: u32 = 4;
        let chunk = n.div_ceil(workers as usize);
        let poses_arc = Arc::new(poses);

        let task = choir
            .spawn("snow_update")
            .init_multi(workers, move |_ctx, worker_idx| {
                let start = (worker_idx as usize) * chunk;
                let end = ((worker_idx as usize + 1) * chunk).min(n);
                for i in start..end {
                    // SAFETY: each worker handles its own [start..end) range
                    // (no overlap with peers), and the underlying buffers
                    // outlive this task because `update` calls `join` below
                    // before its `&mut self` borrow ends.
                    unsafe {
                        let inst = instances_p.get_mut(i);
                        inst.transform = poses_arc[i];
                        let age = ages_p.get_mut(i);
                        *age = age.saturating_add(1);
                        let lifetime = lifetimes_p.read(i);
                        if *age >= lifetime {
                            respawn_p.write(i, 1);
                        }
                    }
                }
            })
            .run();
        let _ = task.join();

        // Phase 3 (serial): respawn the flagged particles. Single-threaded
        // because `physics.teleport_body` mutates rapier state and
        // `sample_spawn` advances `self.rng_state`.
        for i in 0..n {
            if respawn_flags[i] != 0 {
                let (pos, _rot) = self.sample_spawn();
                physics.teleport_body(self.bodies[i], pos);
                self.age_ticks[i] = 0;
                self.lifetime_ticks[i] = self.rand_lifetime();
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
                // 0.05 m/s = roughly the speed at which a particle reads as
                // "resting" rather than "falling/sliding" in the camera view.
                if sp >= 0.05 {
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
    // Tetrahedron: 12 vertices (4 per face × 4 face-unique normal slots so
    // each face is flat-shaded), 4 triangles. The mesh is too small at the
    // current particle scale for any sphere-ish silhouette to matter; this
    // is ~10× cheaper to vertex-process than the icosphere it replaces.
    use nalgebra::{Point2, Vector3 as V3};
    let mut vertices: Vec<VertexDesc> = Vec::with_capacity(12);
    let mut indices: Vec<[u32; 3]> = Vec::with_capacity(4);

    // Regular tetrahedron vertices (unit-sphere-inscribed); coords lifted
    // from the canonical (+,+,+) / (+,-,-) / (-,+,-) / (-,-,+) embedding,
    // scaled to lie on the unit sphere so the radial normal at each corner
    // equals the position direction.
    let inv_sqrt3 = 1.0 / 3.0_f32.sqrt();
    let corners = [
        V3::new(inv_sqrt3, inv_sqrt3, inv_sqrt3),
        V3::new(inv_sqrt3, -inv_sqrt3, -inv_sqrt3),
        V3::new(-inv_sqrt3, inv_sqrt3, -inv_sqrt3),
        V3::new(-inv_sqrt3, -inv_sqrt3, inv_sqrt3),
    ];
    let faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    for face in faces {
        let a = corners[face[0]];
        let b = corners[face[1]];
        let c = corners[face[2]];
        // Flat shading: each face has its own outward normal so the
        // tetrahedron reads as 4 distinct facets.
        let n = (b - a).cross(&(c - a)).normalize();
        let base = vertices.len() as u32;
        for v in [a, b, c] {
            vertices.push(VertexDesc {
                pos: Point3::from(v * radius),
                tex_coords: Point2::new(0.5, 0.5),
                normal: n,
            });
        }
        indices.push([base, base + 1, base + 2]);
    }

    let _ = Vector3::<f32>::zeros();
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
