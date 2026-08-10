use crate::config::WorldShape;
use rapier3d::math::{Vec3, Vector};
use std::default::Default;

pub struct TerrainBody {
    pub(crate) body: rapier3d::dynamics::RigidBodyHandle,
    pub shape: WorldShape,
    /// Torus centreline radius (`length / 2π`); unused for other shapes.
    pub major_radius: f32,
    /// Effective attracting mass for the Newtonian gravity formula. Computed
    /// analytically from the map config — the terrain colliders are open
    /// triangle meshes, which have no meaningful volume of their own.
    gravity_mass: f32,
}

impl TerrainBody {
    /// The point gravity pulls toward from `pos`: the nearest point on the
    /// world's "core" — the Z axis for the cylinder, the origin for the
    /// sphere, the centreline circle for the torus.
    pub fn gravity_anchor(&self, pos: Vec3) -> Vec3 {
        match self.shape {
            WorldShape::Cylinder => Vec3::new(0.0, 0.0, pos.z),
            WorldShape::Sphere => Vec3::ZERO,
            WorldShape::Torus => {
                let rxy = (pos.x * pos.x + pos.y * pos.y).sqrt();
                if rxy < 1e-6 {
                    // On the torus axis every centreline point is equally
                    // near; pick one so the force stays finite.
                    Vec3::new(self.major_radius, 0.0, 0.0)
                } else {
                    let scale = self.major_radius / rxy;
                    Vec3::new(pos.x * scale, pos.y * scale, 0.0)
                }
            }
        }
    }

    /// Unit "up" (radially away from the gravity anchor) at `pos`. Falls
    /// back to +Y when `pos` is degenerate (on the anchor itself).
    pub fn up(&self, pos: Vec3) -> Vec3 {
        let d = pos - self.gravity_anchor(pos);
        let len = d.length();
        if len < 1e-6 {
            Vec3::Y
        } else {
            d / len
        }
    }
}

pub struct PhysicsBodyHandle {
    pub rigid_body_handle: rapier3d::dynamics::RigidBodyHandle,
    pub collider_handles: Vec<rapier3d::geometry::ColliderHandle>,
}

#[derive(Clone, Copy, Debug)]
pub struct Kinematics {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub linvel: [f32; 3],
    pub angvel: [f32; 3],
}

#[derive(Default)]
pub struct Physics {
    rigid_bodies: rapier3d::dynamics::RigidBodySet,
    integration_params: rapier3d::dynamics::IntegrationParameters,
    island_manager: rapier3d::dynamics::IslandManager,
    impulse_joints: rapier3d::dynamics::ImpulseJointSet,
    multibody_joints: rapier3d::dynamics::MultibodyJointSet,
    solver: rapier3d::dynamics::CCDSolver,
    colliders: rapier3d::geometry::ColliderSet,
    broad_phase: rapier3d::geometry::DefaultBroadPhase,
    narrow_phase: rapier3d::geometry::NarrowPhase,
    pipeline: rapier3d::pipeline::PhysicsPipeline,
    last_time: f32,
}

impl Physics {
    /// Attach the terrain TIN as one fixed body with a trimesh collider per
    /// chunk (finest LOD) — the *same* mesh the renderer draws, so the
    /// physics surface and the visual surface cannot disagree.
    pub fn create_terrain_mesh(
        &mut self,
        config: &super::MapConfig,
        mesh: &super::tin::TerrainMesh,
    ) -> TerrainBody {
        use rapier3d::geometry::TriMeshFlags;
        use std::f32::consts::PI;

        let body =
            rapier3d::dynamics::RigidBodyBuilder::new(rapier3d::dynamics::RigidBodyType::Fixed)
                .build();
        let body_handle = self.rigid_bodies.insert(body);

        let mut triangles = 0usize;
        for chunk in &mesh.chunks {
            let (vertices, indices) = chunk.lod0();
            if indices.is_empty() {
                continue;
            }
            triangles += indices.len() / 3;
            let vertices: Vec<Vec3> = vertices
                .iter()
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect();
            let indices: Vec<[u32; 3]> = indices
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]])
                .collect();
            // FIX_INTERNAL_EDGES keeps wheels from snagging on the shared
            // edges between coplanar-ish triangles as they roll across;
            // DELETE_DEGENERATE_TRIANGLES drops the zero-area slivers the
            // sphere's pole rows produce.
            let collider = rapier3d::geometry::ColliderBuilder::trimesh_with_flags(
                vertices,
                indices,
                TriMeshFlags::MERGE_DUPLICATE_VERTICES
                    | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES
                    | TriMeshFlags::FIX_INTERNAL_EDGES,
            )
            .expect("degenerate terrain chunk trimesh")
            .friction(1.0)
            .build();
            self.colliders
                .insert_with_parent(collider, body_handle, &mut self.rigid_bodies);
        }

        // The Newtonian gravity formula (see `update_gravity`) wants a mass
        // for the terrain. The meshes are open surfaces, so derive it from
        // an equivalent solid instead. The sphere keeps its deliberately
        // inflated virtual ball (see the git history of sphere gravity
        // tuning): near the surface it saturates the MAX_ACCEL cap, which is
        // what makes driving feel rooted.
        let r_mid = 0.5 * (config.radius.start + config.radius.end);
        let major_radius = config.length / std::f32::consts::TAU;
        let volume = match config.shape {
            WorldShape::Cylinder => PI * r_mid * r_mid * config.length,
            WorldShape::Sphere => {
                let r = 3.0 * config.radius.end;
                4.0 / 3.0 * PI * r * r * r
            }
            WorldShape::Torus => 2.0 * PI * PI * major_radius * r_mid * r_mid,
        };
        log::info!(
            "Terrain body: {:?}, {} trimesh chunks, {} triangles, gravity mass {:.3e}",
            config.shape,
            mesh.chunks.len(),
            triangles,
            volume * config.density,
        );

        TerrainBody {
            body: body_handle,
            shape: config.shape,
            major_radius,
            gravity_mass: volume * config.density,
        }
    }

    /// Convenience for tests and tools: build the TIN from a raw height map
    /// at full quality, then attach it.
    pub fn create_terrain(
        &mut self,
        config: &super::MapConfig,
        alpha: Vec<u8>,
        width: u32,
        height: u32,
    ) -> TerrainBody {
        let mesh = super::tin::build(&alpha, width, height, config, 1.0);
        self.create_terrain_mesh(config, &mesh)
    }

    pub fn add_rigid_body(
        &mut self,
        rigid_body: rapier3d::dynamics::RigidBody,
        colliders: Vec<rapier3d::geometry::Collider>,
    ) -> PhysicsBodyHandle {
        let rigid_body_handle = self.rigid_bodies.insert(rigid_body);
        let collider_handles = colliders
            .into_iter()
            .map(|collider| {
                self.colliders.insert_with_parent(
                    collider,
                    rigid_body_handle,
                    &mut self.rigid_bodies,
                )
            })
            .collect();
        PhysicsBodyHandle {
            rigid_body_handle,
            collider_handles,
        }
    }

    pub fn add_revolute_joint(
        &mut self,
        body1: rapier3d::dynamics::RigidBodyHandle,
        body2: rapier3d::dynamics::RigidBodyHandle,
        joint: rapier3d::dynamics::RevoluteJoint,
    ) -> rapier3d::dynamics::ImpulseJointHandle {
        self.impulse_joints.insert(body1, body2, joint, true)
    }

    pub fn add_generic_joint(
        &mut self,
        body1: rapier3d::dynamics::RigidBodyHandle,
        body2: rapier3d::dynamics::RigidBodyHandle,
        joint: rapier3d::dynamics::GenericJoint,
    ) -> rapier3d::dynamics::ImpulseJointHandle {
        self.impulse_joints.insert(body1, body2, joint, true)
    }

    /// Sets the velocity-target motor on the wheel's spin axis. Works with both
    /// the synthetic-test RevoluteJoint setup and the production GenericJoint
    /// (suspension + spin) setup — the latter spins around joint AngZ.
    pub fn set_joint_motor_velocity(
        &mut self,
        handle: rapier3d::dynamics::ImpulseJointHandle,
        velocity: f32,
        factor: f32,
    ) {
        if let Some(joint) = self.impulse_joints.get_mut(handle, true) {
            if let Some(rev) = joint.data.as_revolute_mut() {
                rev.set_motor_velocity(velocity, factor);
            } else {
                joint.data.set_motor_velocity(
                    rapier3d::dynamics::JointAxis::AngZ,
                    velocity,
                    factor,
                );
            }
        }
    }

    /// Sets a position-target spring motor on the given joint axis. Used by
    /// front-wheel steering: the wheel's AngY joint axis is free, and a motor
    /// pulls it toward the steer-input angle with the given spring constants.
    pub fn set_joint_motor_position(
        &mut self,
        handle: rapier3d::dynamics::ImpulseJointHandle,
        axis: rapier3d::dynamics::JointAxis,
        target_pos: f32,
        stiffness: f32,
        damping: f32,
    ) {
        if let Some(joint) = self.impulse_joints.get_mut(handle, true) {
            joint
                .data
                .set_motor_position(axis, target_pos, stiffness, damping);
        }
    }

    /// Split the chassis's angular velocity into a "yaw" component (about the
    /// world up axis at its current position — i.e. the direction gravity
    /// points away from) and a "tumble" component (everything else), then
    /// decay each at its own rate. Lets us suppress roll and pitch while
    /// leaving yaw responsive, regardless of how the chassis is currently
    /// tilted. Call once per physics step, BEFORE `step()`, with rapier's own
    /// `angular_damping` set to 0 for this body.
    ///
    /// `damping_yaw` and `damping_tumble` are per-second rates (matching
    /// rapier's `angular_damping` convention: ω *= exp(-rate · dt) per step).
    /// Implemented as a direct angvel scaling rather than a torque so the
    /// damping rate is independent of the body's inertia tensor.
    pub fn apply_axial_angular_damping(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        terrain: &TerrainBody,
        damping_yaw: f32,
        damping_tumble: f32,
    ) {
        let Some(rb) = self.rigid_bodies.get_mut(rb_handle) else {
            return;
        };
        let yaw_axis = terrain.up(rb.position().translation);

        let dt = self.integration_params.dt;
        let f_yaw = (-damping_yaw * dt).exp();
        let f_tumble = (-damping_tumble * dt).exp();

        let angvel = rb.angvel();
        let omega_yaw_scalar = angvel.dot(yaw_axis);
        let omega_yaw = yaw_axis * omega_yaw_scalar;
        let omega_tumble = angvel - omega_yaw;
        rb.set_angvel(omega_yaw * f_yaw + omega_tumble * f_tumble, true);
    }

    /// Apply radial gravity (toward the terrain's gravity anchor) to every
    /// dynamic body.
    pub fn update_gravity(&mut self, terrain: &TerrainBody) {
        profiling::scope!("Physics::update_gravity");
        //Note: real world power is -11, but our scales are different
        const GRAVITY: f32 = 1e-3;
        /// Cap on the effective radial acceleration (m/s²). Without it the Newtonian
        /// G·M_terrain/r² spikes well past the wheel motor's friction cap on larger
        /// maps and pins the vehicle in place. Picked above the effective gravity
        /// the legacy synthetic tests see (~10 m/s² near the axis) so their
        /// settling dynamics are preserved.
        const MAX_ACCEL: f32 = 12.0;
        let terrain_mass = terrain.gravity_mass;
        for (_handle, rb) in self.rigid_bodies.iter_mut() {
            if !rb.is_dynamic() {
                continue;
            }
            let pos = rb.position().translation;
            let to_body = pos - terrain.gravity_anchor(pos);
            let radial_sq = to_body.length_squared();
            if radial_sq < 1e-6 {
                rb.reset_forces(false);
                continue;
            }
            let mass = rb.mass();
            let gravity_uncapped = GRAVITY * mass * terrain_mass / radial_sq;
            let gravity = gravity_uncapped.min(MAX_ACCEL * mass);
            rb.reset_forces(false);
            rb.add_force(-to_body.normalize() * gravity, true);
        }
    }

    pub fn get_transform(
        &self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
    ) -> nalgebra::Isometry3<f32> {
        (*self.rigid_bodies.get(rb_handle).unwrap().position()).into()
    }

    pub fn body_mass(&self, rb_handle: rapier3d::dynamics::RigidBodyHandle) -> f32 {
        self.rigid_bodies.get(rb_handle).map_or(0.0, |rb| rb.mass())
    }

    /// Reset a body's translation and zero its velocities. Used by the debug
    /// snow system to recycle settled particles back to the outer shell
    /// without having to delete and re-create their colliders.
    pub fn teleport_body(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        translation: rapier3d::math::Vec3,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            let mut pose = *rb.position();
            pose.translation = translation;
            rb.set_position(pose, true);
            rb.set_linvel(rapier3d::math::Vec3::ZERO, true);
            rb.set_angvel(rapier3d::math::Vec3::ZERO, true);
        }
    }

    pub fn apply_impulse(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        impulse: rapier3d::math::Vec3,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            rb.apply_impulse(impulse, true);
        }
    }

    /// Apply an impulse at a world-space point on the body. Generates both a
    /// linear and angular component if the point is offset from the CoM —
    /// used by the jump button to push off from the bottom of the chassis.
    pub fn apply_impulse_at_point(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        impulse: rapier3d::math::Vec3,
        point_world: rapier3d::math::Vec3,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            rb.apply_impulse_at_point(impulse, point_world, true);
        }
    }

    /// True if any collider attached to `rb_handle` is currently touching any
    /// of the terrain's chunk colliders. Cheaper than tracking contact-pair
    /// events because we only call it on the rare frames where the player
    /// presses jump.
    pub fn is_touching_terrain(
        &self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        terrain: &TerrainBody,
    ) -> bool {
        let Some(rb) = self.rigid_bodies.get(rb_handle) else {
            return false;
        };
        for &c in rb.colliders() {
            for pair in self.narrow_phase.contact_pairs_with(c) {
                if !pair.has_any_active_contact() {
                    continue;
                }
                let other = if pair.collider1 == c {
                    pair.collider2
                } else {
                    pair.collider1
                };
                if self
                    .colliders
                    .get(other)
                    .and_then(|col| col.parent())
                    == Some(terrain.body)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Adds a continuous force to a body (applied for the duration of one physics
    /// step, then cleared on the next `reset_forces`). Must be called AFTER
    /// `update_gravity` since `update_gravity` resets forces.
    pub fn add_force(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        force: rapier3d::math::Vec3,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            rb.add_force(force, true);
        }
    }

    pub fn add_torque(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        torque: rapier3d::math::Vec3,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            rb.add_torque(torque, true);
        }
    }

    /// Applies an instantaneous angular impulse (units: N·m·s). Used by the
    /// player's `<` / `>` roll keys to flip the chassis back upright.
    pub fn apply_torque_impulse(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        torque_impulse: rapier3d::math::Vec3,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            rb.apply_torque_impulse(torque_impulse, true);
        }
    }

    pub fn body_linvel(
        &self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
    ) -> rapier3d::math::Vec3 {
        self.rigid_bodies
            .get(rb_handle)
            .map_or(rapier3d::math::Vec3::ZERO, |rb| rb.linvel())
    }

    pub fn body_angvel(
        &self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
    ) -> rapier3d::math::Vec3 {
        self.rigid_bodies
            .get(rb_handle)
            .map_or(rapier3d::math::Vec3::ZERO, |rb| rb.angvel())
    }

    pub fn set_linvel(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        linvel: rapier3d::math::Vec3,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            rb.set_linvel(linvel, true);
        }
    }

    pub fn set_angvel(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        angvel: rapier3d::math::Vec3,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            rb.set_angvel(angvel, true);
        }
    }

    pub fn body_kinematics(
        &self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
    ) -> Option<Kinematics> {
        let rb = self.rigid_bodies.get(rb_handle)?;
        let p = rb.position();
        let lv = rb.linvel();
        let av = rb.angvel();
        Some(Kinematics {
            translation: [p.translation.x, p.translation.y, p.translation.z],
            rotation: [p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w],
            linvel: [lv.x, lv.y, lv.z],
            angvel: [av.x, av.y, av.z],
        })
    }

    pub fn step(&mut self) {
        profiling::scope!("Physics::step");
        let physics_hooks = ();
        let event_handler = ();
        self.pipeline.step(
            Vector::ZERO, // we apply our own radial gravity each tick
            &self.integration_params,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.solver,
            &physics_hooks,
            &event_handler,
        );
        self.last_time += self.integration_params.dt;
    }

    pub fn last_time(&self) -> f32 {
        self.last_time
    }
}
