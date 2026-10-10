use crate::config::WorldShape;
use rapier3d::math::{Vec3, Vector};
use std::default::Default;

pub struct TerrainBody {
    pub(crate) body: rapier3d::dynamics::RigidBodyHandle,
    pub shape: WorldShape,
    /// Torus centreline radius (`length / 2π`); unused for other shapes.
    pub major_radius: f32,
    /// Bounds used by radial terrain queries (not the car/snow colliders).
    outer_radius: f32,
    axial_bounds: [f32; 2],
    /// Static chunk BVH for camera queries, including before the first step.
    /// Terrain colliders never move; dynamic car/snow bodies are not included.
    query_bvh: rapier3d::parry::partitioning::Bvh,
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
        let mut axial_bounds = [f32::INFINITY, f32::NEG_INFINITY];
        for chunk in &mesh.chunks {
            let (vertices, indices) = chunk.lod0();
            if indices.is_empty() {
                continue;
            }
            triangles += indices.len() / 3;
            let vertices: Vec<Vec3> = vertices
                .iter()
                .map(|v| {
                    axial_bounds[0] = axial_bounds[0].min(v[2]);
                    axial_bounds[1] = axial_bounds[1].max(v[2]);
                    Vec3::new(v[0], v[1], v[2])
                })
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

        let query_bvh = rapier3d::parry::partitioning::Bvh::from_iter(
            rapier3d::parry::partitioning::BvhBuildStrategy::default(),
            self.rigid_bodies[body_handle]
                .colliders()
                .iter()
                .map(|&handle| {
                    (
                        handle.into_raw_parts().0 as usize,
                        self.colliders[handle].compute_aabb(),
                    )
                }),
        );
        TerrainBody {
            body: body_handle,
            shape: config.shape,
            major_radius,
            outer_radius: config.radius.end,
            axial_bounds,
            query_bvh,
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

    /// Project onto the actual terrain triangles along local gravity. This
    /// deliberately ignores dynamic bodies and works before the first step.
    /// Open cylinder ends are inset by up to one metre so a target cannot
    /// be placed beyond the driveable mesh. At a sphere's unmeshed pole cap,
    /// use the closest terrain point instead of inventing a ground height.
    pub fn terrain_surface_point(&self, terrain: &TerrainBody, mut pos: Vec3) -> Option<Vec3> {
        let body = self.rigid_bodies.get(terrain.body)?;
        if body.colliders().is_empty() {
            return None;
        }
        if terrain.shape == WorldShape::Cylinder {
            let [min, max] = terrain.axial_bounds;
            let inset = 1.0_f32.min((max - min) * 0.5);
            pos.z = pos.z.clamp(min + inset, max - inset);
        }
        let up = terrain.up(pos);
        let ray_length = terrain.outer_radius + 1.0;
        let ray = rapier3d::geometry::Ray::new(terrain.gravity_anchor(pos) + up * ray_length, -up);
        let hit = body
            .colliders()
            .iter()
            .filter_map(|&handle| {
                let collider = &self.colliders[handle];
                collider
                    .shape()
                    .cast_ray(collider.position(), &ray, ray_length, false)
            })
            .min_by(f32::total_cmp);
        if let Some(distance) = hit {
            return Some(ray.point_at(distance));
        }
        body.colliders()
            .iter()
            .map(|&handle| {
                let collider = &self.colliders[handle];
                collider
                    .shape()
                    .project_point(collider.position(), pos, false)
                    .point
            })
            .min_by(|a, b| a.distance_squared(pos).total_cmp(&b.distance_squared(pos)))
    }

    /// Shorten a camera boom before its sphere touches the actual terrain.
    /// A static chunk BVH avoids scanning all 2,048 chunks every frame, and
    /// each candidate trimesh uses its own triangle BVH. Ignore car and snow.
    /// Start near the car. Resolve a sphere already touching a steep ridge
    /// before sweeping, otherwise the zero-time hit would pin the camera.
    pub fn terrain_camera_position(
        &self,
        terrain: &TerrainBody,
        mut origin: Vec3,
        desired: Vec3,
        radius: f32,
    ) -> Vec3 {
        let query = rapier3d::pipeline::QueryPipeline {
            dispatcher: self.narrow_phase.query_dispatcher(),
            bvh: &terrain.query_bvh,
            bodies: &self.rigid_bodies,
            colliders: &self.colliders,
            filter: rapier3d::pipeline::QueryFilter::default(),
        };
        let sphere = rapier3d::parry::shape::Ball::new(radius);
        // Most frames do only the cheap local overlap query. Projection is
        // needed only when the focus grazes terrain; cap corner iterations.
        for _ in 0..4 {
            if query
                .intersect_shape(rapier3d::math::Pose::from_translation(origin), &sphere)
                .next()
                .is_none()
            {
                break;
            }
            let Some((_, projection)) = query.project_point(origin, radius, false) else {
                break;
            };
            let separation = origin - projection.point;
            let normal = separation
                .try_normalize()
                .unwrap_or_else(|| terrain.up(origin));
            origin = projection.point + normal * (radius + 0.02);
        }
        let delta = desired - origin;
        let distance = delta.length();
        if !distance.is_finite() || distance < 1e-5 {
            return origin;
        }
        let direction = delta / distance;
        let hit = query.cast_shape(
            &rapier3d::math::Pose::from_translation(origin),
            direction,
            &sphere,
            rapier3d::parry::query::ShapeCastOptions::with_max_time_of_impact(distance),
        );
        // Leave a small skin beyond the swept sphere to avoid roundoff flicker.
        let travel = hit.map_or(distance, |(_, hit)| (hit.time_of_impact - 0.02).max(0.0));
        origin + direction * travel
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

    /// Snap a body to a full pose (translation + rotation) and zero velocities.
    /// Used for soft out-of-bounds player respawn.
    pub fn teleport_body_pose(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        pose: nalgebra::Isometry3<f32>,
    ) {
        if let Some(rb) = self.rigid_bodies.get_mut(rb_handle) {
            rb.set_position(pose.into(), true);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tin::{ChunkBuffers, Mapping, Stats, TerrainMesh};

    fn camera_terrain(chunks: usize) -> (Physics, TerrainBody) {
        let config = crate::config::Map {
            radius: 1.0..15.0,
            length: 100.0,
            density: 10.0,
            shape: WorldShape::Cylinder,
        };
        let chunks = (0..chunks)
            .map(|i| {
                let x = 3.0 + i as f32 * 10.0;
                ChunkBuffers {
                    vertices: vec![
                        [x, -1.0, -1.0],
                        [x, 1.0, -1.0],
                        [x, 1.0, 1.0],
                        [x, -1.0, 1.0],
                    ],
                    indices: vec![0, 1, 2, 0, 2, 3],
                    lods: vec![(0, 6)],
                    lod0_vertex_count: 4,
                    min: [x, -1.0, -1.0],
                    max: [x, 1.0, 1.0],
                }
            })
            .collect();
        let mesh = TerrainMesh {
            mapping: Mapping::new(&config, 2, 2),
            chunks,
            stats: Stats::default(),
        };
        let mut physics = Physics::default();
        let terrain = physics.create_terrain_mesh(&config, &mesh);
        (physics, terrain)
    }

    #[test]
    fn camera_sweep_hits_terrain_before_first_step_and_ignores_dynamic_bodies() {
        let (mut physics, terrain) = camera_terrain(1);
        physics.add_rigid_body(
            rapier3d::dynamics::RigidBodyBuilder::dynamic()
                .translation(Vec3::X)
                .build(),
            vec![rapier3d::geometry::ColliderBuilder::ball(0.5).build()],
        );
        let camera = physics.terrain_camera_position(&terrain, Vec3::ZERO, Vec3::X * 6.0, 0.3);
        assert!((camera.x - 2.68).abs() < 0.001, "{camera:?}");
        // Both sides of the terrain triangle must block the camera.
        let reverse = physics.terrain_camera_position(&terrain, Vec3::X * 6.0, Vec3::ZERO, 0.3);
        assert!((reverse.x - 3.32).abs() < 0.001, "{reverse:?}");
        assert_eq!(
            physics.terrain_camera_position(&terrain, Vec3::ZERO, Vec3::ZERO, 0.3),
            Vec3::ZERO
        );
    }

    #[test]
    fn camera_escapes_initial_wall_overlap_without_crossing_the_wall() {
        let (physics, terrain) = camera_terrain(1);
        let origin = Vec3::new(2.8, 0.0, 0.0);
        let away = physics.terrain_camera_position(&terrain, origin, Vec3::ZERO, 0.3);
        assert!(away.distance(Vec3::ZERO) < 1e-5, "{away:?}");
        let into = physics.terrain_camera_position(&terrain, origin, Vec3::X * 6.0, 0.3);
        assert!(into.x <= 2.7, "{into:?}");
        // A heavily retracted/smoothed target can itself be inside the overlap.
        let near = physics.terrain_camera_position(&terrain, origin, origin, 0.3);
        assert!(near.x <= 2.7, "{near:?}");
    }

    #[test]
    fn camera_sphere_catches_edges_missed_by_center_ray() {
        let (physics, terrain) = camera_terrain(1);
        let start = Vec3::new(0.0, 1.1, 0.0);
        let end = Vec3::new(6.0, 1.1, 0.0);
        let camera = physics.terrain_camera_position(&terrain, start, end, 0.3);
        assert!(camera.x > 2.6 && camera.x < 3.0, "{camera:?}");
        let clear = Vec3::new(6.0, 2.0, 0.0);
        assert_eq!(
            physics.terrain_camera_position(&terrain, Vec3::new(0.0, 2.0, 0.0), clear, 0.3),
            clear
        );
    }

    #[test]
    fn camera_queries_nearest_chunk_in_large_static_bvh() {
        let (physics, terrain) = camera_terrain(2048);
        let camera = physics.terrain_camera_position(&terrain, Vec3::ZERO, Vec3::X * 20_000.0, 0.3);
        assert!((camera.x - 2.68).abs() < 0.001, "{camera:?}");
    }

    #[test]
    fn surface_query_uses_terrain_triangles_not_other_colliders_or_max_radius() {
        let config = crate::config::Map {
            radius: 10.0..15.0,
            length: 100.0,
            density: 10.0,
            shape: WorldShape::Cylinder,
        };
        // A sloped terrain patch: its height at y=0 is exactly x=12.
        let mesh = TerrainMesh {
            mapping: Mapping::new(&config, 2, 2),
            chunks: vec![ChunkBuffers {
                vertices: vec![
                    [11.0, -2.0, -2.0],
                    [13.0, 2.0, -2.0],
                    [13.0, 2.0, 2.0],
                    [11.0, -2.0, 2.0],
                ],
                indices: vec![0, 1, 2, 0, 2, 3],
                lods: vec![(0, 6)],
                lod0_vertex_count: 4,
                min: [11.0, -2.0, -2.0],
                max: [13.0, 2.0, 2.0],
            }],
            stats: Stats::default(),
        };
        let mut physics = Physics::default();
        let terrain = physics.create_terrain_mesh(&config, &mesh);
        physics.add_rigid_body(
            rapier3d::dynamics::RigidBodyBuilder::dynamic()
                .translation(Vec3::new(14.0, 0.0, 0.0))
                .build(),
            vec![rapier3d::geometry::ColliderBuilder::ball(0.5).build()],
        );
        let hit = physics
            .terrain_surface_point(&terrain, Vec3::new(20.0, 0.0, 0.0))
            .unwrap();
        assert!(hit.distance(Vec3::new(12.0, 0.0, 0.0)) < 1e-5);
        // Clamp to the actual mesh boundary, not the config's ±50 m ends.
        let end = physics
            .terrain_surface_point(&terrain, Vec3::new(20.0, 0.0, 70.0))
            .unwrap();
        assert!(end.distance(Vec3::new(12.0, 0.0, 1.0)) < 1e-5);
    }
}
