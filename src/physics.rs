use std::{f32, ops, sync::Arc};

/*
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum JointAxis {
    LinearX = 0,
    LinearY = 1,
    LinearZ = 2,
    AngularX = 3,
    AngularY = 4,
    AngularZ = 5,
}
impl JointAxis {
    fn into_rapier(self) -> rapier3d::dynamics::JointAxis {
        use rapier3d::dynamics::JointAxis as Ja;
        match self {
            Self::LinearX => Ja::LinX,
            Self::LinearY => Ja::LinY,
            Self::LinearZ => Ja::LinZ,
            Self::AngularX => Ja::AngX,
            Self::AngularY => Ja::AngY,
            Self::AngularZ => Ja::AngZ,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum JointHandle {
    Soft(#[doc(hidden)] rapier3d::dynamics::ImpulseJointHandle),
    Hard(#[doc(hidden)] rapier3d::dynamics::MultibodyJointHandle),
}*/

pub struct TerrainBody {
    pub config: super::config::Map,
    extent: [u32; 2],
    data: Vec<u8>,
}

struct RadialCoordinates {
    alpha: f32,
    radius: f32,
    depth: f32,
}

impl RadialCoordinates {
    fn new(p: nalgebra::Point3<f32>) -> Self {
        Self {
            alpha: f32::atan2(p.y, p.x),
            radius: p.xy().coords.magnitude(),
            depth: p.z,
        }
    }
}

impl TerrainBody {
    pub fn new(config: &super::config::Map, extent: [u32; 2], data: Vec<u8>) -> Self {
        Self {
            config: config.clone(),
            extent,
            data,
        }
    }

    pub fn mass(&self) -> f32 {
        let r = 0.5 * (self.config.radius.start + self.config.radius.end);
        let volume = f32::consts::PI * r * r * self.config.length;
        volume * self.config.density
    }

    pub fn compute_resistance(&self, point: nalgebra::Point3<f32>) -> nalgebra::Vector3<f32> {
        let rc = RadialCoordinates::new(point);
        let mut alpha_rad = rc.alpha / (2.0 * f32::consts::PI);
        if alpha_rad < 0.0 {
            alpha_rad = 1.0 + alpha_rad;
        }
        let x_coord_float = alpha_rad * (self.extent[0] as f32);
        let x_coord = ((x_coord_float + 0.5) as i32).clamp(0, self.extent[0] as i32);
        let y_coord = ((rc.depth + 0.5) as i32).clamp(0, self.extent[1] as i32);
        let r_raw = self.data[y_coord as usize * self.extent[0] as usize + x_coord as usize];
        let r_ratio = (r_raw as f32) / 255.0;
        let radius = self.config.radius.start * (1.0 - r_ratio) + self.config.radius.end * r_ratio;
        let impulse = (rc.radius - radius).min(0.0) * 5.0;
        nalgebra::Vector3::new(point.x, point.y, 0.0).normalize() * impulse
    }
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
    //debug_pipeline: rapier3d::pipeline::DebugRenderPipeline,
    last_time: f32,
}

impl Physics {
    fn _create_terrain_from_collider(
        &mut self,
        collider: rapier3d::geometry::Collider,
    ) -> TerrainBody {
        let body =
            rapier3d::dynamics::RigidBodyBuilder::new(rapier3d::dynamics::RigidBodyType::Fixed)
                .build();
        let body_handle = self.rigid_bodies.insert(body);
        let _collider =
            self.colliders
                .insert_with_parent(collider, body_handle, &mut self.rigid_bodies);
        /*TerrainBody {
            _collider: ,
            body: body_handle,
        }*/
        unimplemented!()
    }

    pub fn create_object(
        &mut self,
        model: Arc<super::Model>,
        transform: nalgebra::Isometry3<f32>,
    ) -> super::Object {
        let rb_inner =
            rapier3d::dynamics::RigidBodyBuilder::new(rapier3d::dynamics::RigidBodyType::Dynamic)
                .position(transform)
                .build();
        let rigid_body = self.rigid_bodies.insert(rb_inner);
        let _collider_handle = self.colliders.insert_with_parent(
            model.collider.clone(),
            rigid_body,
            &mut self.rigid_bodies,
        );
        super::Object {
            rigid_body,
            model,
            transform,
        }
    }

    pub fn update_gravity(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        terrain: &TerrainBody,
    ) {
        //Note: real world power is -11, but our scales are different
        const GRAVITY: f32 = 6.6743e-6;
        let rb = self.rigid_bodies.get_mut(rb_handle).unwrap();
        let mut pos = rb.position().translation.vector;
        pos.z = 0.0; // attracted to the cylinder axis
        let gravity = GRAVITY * rb.mass() * terrain.mass() / pos.xy().norm_squared();
        rb.reset_forces(false);
        rb.add_force(-pos.normalize() * gravity, true);
    }

    pub fn apply_impulse(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
        impulse: nalgebra::Vector3<f32>,
    ) {
        let rb = self.rigid_bodies.get_mut(rb_handle).unwrap();
        rb.apply_impulse(impulse, true);
    }

    pub fn get_transform(
        &mut self,
        rb_handle: rapier3d::dynamics::RigidBodyHandle,
    ) -> nalgebra::Isometry3<f32> {
        *self.rigid_bodies.get(rb_handle).unwrap().position()
    }

    pub fn step(&mut self) {
        let query_pipeline = None;
        let physics_hooks = ();
        let event_handler = ();
        self.pipeline.step(
            &Default::default(), // not using built-in gravity
            &self.integration_params,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.solver,
            query_pipeline,
            &physics_hooks,
            &event_handler,
        );
        self.last_time += self.integration_params.dt;
    }
}

/*
impl ops::Index<JointHandle> for Physics {
    type Output = rapier3d::dynamics::GenericJoint;
    fn index(&self, handle: JointHandle) -> &Self::Output {
        match handle {
            JointHandle::Soft(h) => &self.impulse_joints.get(h).unwrap().data,
            JointHandle::Hard(h) => {
                let (multibody, link_index) = self.multibody_joints.get(h).unwrap();
                &multibody.link(link_index).unwrap().joint.data
            }
        }
    }
}
impl ops::IndexMut<JointHandle> for Physics {
    fn index_mut(&mut self, handle: JointHandle) -> &mut Self::Output {
        match handle {
            JointHandle::Soft(h) => &mut self.impulse_joints.get_mut(h).unwrap().data,
            JointHandle::Hard(h) => {
                let (multibody, link_index) = self.multibody_joints.get_mut(h).unwrap();
                &mut multibody.link_mut(link_index).unwrap().joint.data
            }
        }
    }
}
 */

#[derive(Clone)]
#[allow(unused)]
struct TerrainShape {
    radius: ops::Range<f32>,
    length: f32,
}

impl TerrainShape {
    fn cylinder(&self, ratio: f32) -> rapier3d::geometry::Cylinder {
        rapier3d::geometry::Cylinder {
            half_height: 0.5 * self.length,
            radius: self.radius.start * (1.0 - ratio) + self.radius.end * ratio,
        }
    }
}

impl rapier3d::geometry::PointQuery for TerrainShape {
    fn project_local_point(
        &self,
        _pt: &nalgebra::Point3<f32>,
        _solid: bool,
    ) -> rapier3d::parry::query::point::PointProjection {
        todo!()
    }
    fn project_local_point_and_get_feature(
        &self,
        _pt: &nalgebra::Point3<f32>,
    ) -> (
        rapier3d::parry::query::point::PointProjection,
        rapier3d::geometry::FeatureId,
    ) {
        todo!()
    }
}

impl rapier3d::geometry::RayCast for TerrainShape {
    fn cast_local_ray_and_get_normal(
        &self,
        _ray: &rapier3d::parry::query::details::Ray,
        _max_time_of_impact: f32,
        _solid: bool,
    ) -> Option<rapier3d::parry::query::details::RayIntersection> {
        None
    }
}

impl rapier3d::geometry::SupportMap for TerrainShape {
    fn local_support_point(&self, dir: &nalgebra::Vector3<f32>) -> nalgebra::Point3<f32> {
        let r = 0.5 * (self.radius.start + self.radius.end);
        let length = r / (dir.xy().norm());
        (length * dir).into()
    }
}

impl rapier3d::geometry::Shape for TerrainShape {
    fn compute_local_aabb(&self) -> rapier3d::parry::bounding_volume::Aabb {
        let r = self.radius.end;
        rapier3d::parry::bounding_volume::Aabb {
            mins: nalgebra::Point3::new(-r, -r, -0.5 * self.length),
            maxs: nalgebra::Point3::new(r, r, 0.5 * self.length),
        }
    }
    fn compute_local_bounding_sphere(&self) -> rapier3d::parry::bounding_volume::BoundingSphere {
        rapier3d::parry::bounding_volume::BoundingSphere {
            center: nalgebra::Point3::default(),
            radius: nalgebra::Vector2::new(self.radius.end, 0.5 * self.length).norm(),
        }
    }
    fn clone_dyn(&self) -> Box<dyn rapier3d::geometry::Shape> {
        Box::new(self.clone())
    }
    fn scale_dyn(
        &self,
        _scale: &nalgebra::Vector3<f32>,
        _num_subdivisions: u32,
    ) -> Option<Box<dyn rapier3d::geometry::Shape>> {
        todo!()
    }
    fn mass_properties(&self, density: f32) -> rapier3d::dynamics::MassProperties {
        self.cylinder(0.2).mass_properties(density)
    }
    fn shape_type(&self) -> rapier3d::geometry::ShapeType {
        rapier3d::geometry::ShapeType::Custom
    }
    fn as_typed_shape(&self) -> rapier3d::geometry::TypedShape {
        rapier3d::geometry::TypedShape::Custom(self)
    }
    fn ccd_thickness(&self) -> f32 {
        self.cylinder(0.2).ccd_thickness()
    }
    fn ccd_angular_thickness(&self) -> f32 {
        self.cylinder(0.2).ccd_angular_thickness()
    }
    fn as_support_map(&self) -> Option<&dyn rapier3d::geometry::SupportMap> {
        Some(self)
    }
}
