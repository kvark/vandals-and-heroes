use rapier3d::math::Vector;
use std::default::Default;

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
    /// Build a cylindrical heightmap mesh from an alpha-channel buffer.
    ///
    /// Layout: `alpha` is `width * height` bytes in row-major order; width spans the
    /// circumference (theta = 0..2π) and height spans the cylinder axis (z = -length/2..+length/2).
    /// `alpha=0` → surface at `radius.start`; `alpha=255` → surface at `radius.end`. This matches
    /// the GPU sampling in shaders/terrain-draw.wgsl.
    ///
    /// `step_u`/`step_v` downsample the source map for the physics mesh (e.g. step=16 turns a
    /// 2048×16384 texture into a 128×1024 collision mesh ≈ 260k triangles).
    pub fn create_terrain(
        &mut self,
        config: &super::MapConfig,
        alpha: &[u8],
        width: u32,
        height: u32,
        step_u: u32,
        step_v: u32,
    ) -> TerrainBody {
        let body =
            rapier3d::dynamics::RigidBodyBuilder::new(rapier3d::dynamics::RigidBodyType::Fixed)
                .build();
        let body_handle = self.rigid_bodies.insert(body);

        let n_u = width / step_u;
        let n_v = height / step_v;
        let r_start = config.radius.start;
        let r_range = config.radius.end - config.radius.start;
        let half_len = 0.5 * config.length;

        let sample = |u: u32, v: u32| -> f32 {
            // Clamp v (we don't wrap along axis); u is wrapped via modulo.
            let su = (u * step_u) % width;
            let sv = (v * step_v).min(height - 1);
            let idx = (sv * width + su) as usize;
            alpha[idx] as f32 / 255.0
        };

        let mut vertices: Vec<rapier3d::math::Vec3> =
            Vec::with_capacity((n_u as usize) * (n_v as usize));
        for v in 0..n_v {
            let z = -half_len + (v as f32 / (n_v - 1) as f32) * config.length;
            for u in 0..n_u {
                let theta = (u as f32 / n_u as f32) * std::f32::consts::TAU;
                let r = r_start + sample(u, v) * r_range;
                vertices.push(rapier3d::math::Vec3::new(
                    r * theta.cos(),
                    r * theta.sin(),
                    z,
                ));
            }
        }

        let mut indices: Vec<[u32; 3]> =
            Vec::with_capacity(2 * (n_u as usize) * ((n_v - 1) as usize));
        for v in 0..n_v - 1 {
            for u in 0..n_u {
                let u1 = (u + 1) % n_u;
                let v1 = v + 1;
                let i00 = v * n_u + u;
                let i10 = v * n_u + u1;
                let i01 = v1 * n_u + u;
                let i11 = v1 * n_u + u1;
                indices.push([i00, i01, i11]);
                indices.push([i00, i11, i10]);
            }
        }

        log::info!(
            "Terrain collider: {} verts, {} tris (sampled {}x{} from {}x{})",
            vertices.len(),
            indices.len(),
            n_u,
            n_v,
            width,
            height,
        );

        let collider = rapier3d::geometry::ColliderBuilder::trimesh(vertices, indices)
            .expect("Building terrain trimesh")
            .density(config.density)
            .friction(1.0)
            .build();

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
