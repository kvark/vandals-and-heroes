use rapier3d::math::Vector;
use std::default::Default;
use std::sync::Arc;

pub struct TerrainBody {
    _collider: rapier3d::geometry::ColliderHandle,
    body: rapier3d::dynamics::RigidBodyHandle,
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
        let narrow_phase = rapier3d::geometry::NarrowPhase::with_query_dispatcher(
            super::CylDispatcher::new(),
        );
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
        log::info!(
            "Terrain heightfield: {}x{} samples (full resolution, on-the-fly triangulation)",
            width,
            height
        );
        let hf = super::CylindricalHeightField::new(
            alpha,
            width,
            height,
            config.radius.start,
            config.radius.end,
            config.length,
        );
        let collider = rapier3d::geometry::ColliderBuilder::new(
            rapier3d::geometry::SharedShape(Arc::new(hf)),
        )
        .density(config.density)
        .friction(1.0)
        .build();

        let body =
            rapier3d::dynamics::RigidBodyBuilder::new(rapier3d::dynamics::RigidBodyType::Fixed)
                .build();
        let body_handle = self.rigid_bodies.insert(body);

        TerrainBody {
            _collider: self.colliders.insert_with_parent(
                collider,
                body_handle,
                &mut self.rigid_bodies,
            ),
            body: body_handle,
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

    pub fn set_joint_motor_velocity(
        &mut self,
        handle: rapier3d::dynamics::ImpulseJointHandle,
        velocity: f32,
        factor: f32,
    ) {
        if let Some(joint) = self.impulse_joints.get_mut(handle, true) {
            if let Some(rev) = joint.data.as_revolute_mut() {
                rev.set_motor_velocity(velocity, factor);
            }
        }
    }

    /// Apply radial gravity (toward Z axis) to every dynamic body.
    pub fn update_gravity(&mut self, terrain: &TerrainBody) {
        //Note: real world power is -11, but our scales are different
        const GRAVITY: f32 = 1e-3;
        let terrain_mass = self.rigid_bodies.get(terrain.body).unwrap().mass();
        for (_handle, rb) in self.rigid_bodies.iter_mut() {
            if !rb.is_dynamic() {
                continue;
            }
            let mut pos = rb.position().translation;
            pos.z = 0.0;
            let radial_sq = pos.x * pos.x + pos.y * pos.y;
            if radial_sq < 1e-6 {
                rb.reset_forces(false);
                continue;
            }
            let gravity = GRAVITY * rb.mass() * terrain_mass / radial_sq;
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
