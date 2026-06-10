use rapier3d::math::Vector;
use std::default::Default;
use std::sync::Arc;

pub struct TerrainBody {
    pub(crate) collider: rapier3d::geometry::ColliderHandle,
    _body: rapier3d::dynamics::RigidBodyHandle,
    /// `true` when the world is a sphere. Gravity then points to the origin
    /// in 3D instead of toward the Z axis, and the spawn / camera / wheel
    /// collider use sphere geometry. `false` keeps the cylinder world.
    pub is_sphere: bool,
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

impl Default for Physics {
    fn default() -> Self {
        // Custom NarrowPhase dispatcher so the CylindricalHeightField gets cell-grid
        // contact generation instead of falling into the default Custom-shape path
        // (which would treat it as convex and return nothing).
        let narrow_phase =
            rapier3d::geometry::NarrowPhase::with_query_dispatcher(super::CylDispatcher::new());
        Self {
            rigid_bodies: Default::default(),
            integration_params: Default::default(),
            island_manager: Default::default(),
            impulse_joints: Default::default(),
            multibody_joints: Default::default(),
            solver: Default::default(),
            colliders: Default::default(),
            broad_phase: Default::default(),
            narrow_phase,
            pipeline: Default::default(),
            last_time: 0.0,
        }
    }
}

impl Physics {
    /// Build a `CylindricalHeightField` collider directly from the raw alpha channel.
    /// No downsampling — every pixel becomes a sample; triangles are generated lazily
    /// per contact query by [`super::CylindricalHeightField::map_elements_in_local_aabb`].
    pub fn create_terrain(
        &mut self,
        config: &super::MapConfig,
        alpha: Vec<u8>,
        width: u32,
        height: u32,
    ) -> TerrainBody {
        let collider = if config.is_sphere {
            log::info!(
                "Spherical world: {}x{} heightmap, smooth-sphere collider (radius_end={:.2}). \
                 Heightmap drives the renderer; physics drives on a smooth sphere for now.",
                width,
                height,
                config.radius.end,
            );
            // First-cut sphere physics: treat the world as a smooth sphere at
            // the *outer* radius — this is the upper bound of the heightmap.
            // The vehicle will drive on a frictionful sphere with no terrain
            // detail; replacing this with a SphericalHeightField gives back the
            // heightmap relief. Average mass-properties come from the same
            // density × volume the cylinder used at construction.
            rapier3d::geometry::ColliderBuilder::ball(config.radius.end)
                .density(config.density)
                .friction(1.0)
                .build()
        } else {
            log::info!(
                "Terrain heightfield: {}x{} samples (full resolution, on-the-fly triangulation)",
                width,
                height,
            );
            let hf = super::CylindricalHeightField::new(
                alpha,
                width,
                height,
                config.radius.start,
                config.radius.end,
                config.length,
            );
            rapier3d::geometry::ColliderBuilder::new(rapier3d::geometry::SharedShape(Arc::new(hf)))
                .density(config.density)
                .friction(1.0)
                .build()
        };

        let body =
            rapier3d::dynamics::RigidBodyBuilder::new(rapier3d::dynamics::RigidBodyType::Fixed)
                .build();
        let body_handle = self.rigid_bodies.insert(body);

        TerrainBody {
            collider: self.colliders.insert_with_parent(
                collider,
                body_handle,
                &mut self.rigid_bodies,
            ),
            _body: body_handle,
            is_sphere: config.is_sphere,
        }
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
    /// world radial-outward axis at its current position — i.e. the direction
    /// gravity points away from) and a "tumble" component (everything else),
    /// then decay each at its own rate. Lets us suppress roll and pitch while
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
        damping_yaw: f32,
        damping_tumble: f32,
    ) {
        let Some(rb) = self.rigid_bodies.get_mut(rb_handle) else {
            return;
        };
        let pos = rb.position().translation;
        let radial_sq = pos.x * pos.x + pos.y * pos.y;
        if radial_sq < 1e-6 {
            return;
        }
        let inv_r = radial_sq.sqrt().recip();
        let yaw_axis = rapier3d::math::Vec3::new(pos.x * inv_r, pos.y * inv_r, 0.0);

        let dt = self.integration_params.dt;
        let f_yaw = (-damping_yaw * dt).exp();
        let f_tumble = (-damping_tumble * dt).exp();

        let angvel = rb.angvel();
        let omega_yaw_scalar = angvel.dot(yaw_axis);
        let omega_yaw = yaw_axis * omega_yaw_scalar;
        let omega_tumble = angvel - omega_yaw;
        rb.set_angvel(omega_yaw * f_yaw + omega_tumble * f_tumble, true);
    }

    /// Apply radial gravity (toward Z axis) to every dynamic body.
    pub fn update_gravity(&mut self, terrain: &TerrainBody) {
        //Note: real world power is -11, but our scales are different
        const GRAVITY: f32 = 1e-3;
        /// Cap on the effective radial acceleration (m/s²). Without it the Newtonian
        /// G·M_terrain/r² spikes well past the wheel motor's friction cap on larger
        /// maps and pins the vehicle in place. Picked above the effective gravity
        /// the legacy synthetic tests see (~10 m/s² near the axis) so their
        /// settling dynamics are preserved.
        const MAX_ACCEL: f32 = 12.0;
        let terrain_mass = self.rigid_bodies.get(terrain._body).unwrap().mass();
        for (_handle, rb) in self.rigid_bodies.iter_mut() {
            if !rb.is_dynamic() {
                continue;
            }
            let mut pos = rb.position().translation;
            if !terrain.is_sphere {
                // Cylinder world: gravity points to the Z axis, so flatten
                // the position to its XY component first.
                pos.z = 0.0;
            }
            let radial_sq = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
            if radial_sq < 1e-6 {
                rb.reset_forces(false);
                continue;
            }
            let mass = rb.mass();
            let gravity_uncapped = GRAVITY * mass * terrain_mass / radial_sq;
            let gravity = gravity_uncapped.min(MAX_ACCEL * mass);
            rb.reset_forces(false);
            rb.add_force(-pos.normalize() * gravity, true);
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

    /// True if any collider attached to `rb_handle` is currently touching the
    /// terrain collider. Cheaper than tracking contact-pair events because we
    /// only call it on the rare frames where the player presses jump.
    pub fn is_touching_terrain(
        &self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        terrain: &TerrainBody,
    ) -> bool {
        let Some(rb) = self.rigid_bodies.get(rb_handle) else {
            return false;
        };
        for &c in rb.colliders() {
            if let Some(pair) = self.narrow_phase.contact_pair(c, terrain.collider) {
                if pair.has_any_active_contact() {
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
