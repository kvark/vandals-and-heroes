use crate::content_pack::ContentPack;
use crate::definitions::{ColliderDesc, ObjectDesc, PhysicsBodyDesc, ShapeDesc};
use crate::instances::Object;
use blade_graphics as gpu;
use std::sync::Arc;
use vandals_and_heroes::Physics;
use vandals_and_heroes::{Loader, Model, ModelInstance};

pub struct ObjectTemplate {
    pub model: Option<Arc<Model>>,
    pub desc: ObjectDesc,
}

fn create_collider(
    content_pack: &ContentPack,
    desc: &ColliderDesc,
) -> rapier3d::geometry::Collider {
    let builder = match &desc.shape {
        ShapeDesc::Box { size: (hx, hy, hz) } => {
            rapier3d::geometry::ColliderBuilder::cuboid(*hx, *hy, *hz)
        }
        ShapeDesc::Sphere { radius } => rapier3d::geometry::ColliderBuilder::ball(*radius),
        ShapeDesc::Mesh { path } => {
            let full_path = content_pack.get_resource_path(path);
            let model_desc = Loader::read_gltf(&full_path, nalgebra::Matrix4::identity());
            rapier3d::geometry::ColliderBuilder::trimesh(
                model_desc.positions(),
                model_desc.indices(),
            )
        }
    };
    builder
        .position(desc.transform.clone().into())
        // TODO: density
        .build()
}

impl ObjectTemplate {
    pub fn instantiate(
        &self,
        content_pack: &ContentPack,
        physics: &mut Physics,
        transform: nalgebra::Isometry3<f32>,
    ) -> Object {
        let model_instance = self.model.as_ref().map(|m| ModelInstance {
            model: m.clone(),
            transform,
        });
        let body = self.desc.physics.as_ref().map(|p| {
            let colliders = p
                .colliders
                .iter()
                .map(|c| create_collider(content_pack, c))
                .collect();
            let body_type = match p.body {
                PhysicsBodyDesc::RigidBody { .. } => rapier3d::dynamics::RigidBodyType::Dynamic,
                PhysicsBodyDesc::StaticBody => rapier3d::dynamics::RigidBodyType::Fixed,
            };
            let rigid_body = rapier3d::dynamics::RigidBodyBuilder::new(body_type)
                .position(transform)
                .gravity_scale(1.0f32)
                .build();
            physics.add_rigid_body(rigid_body, colliders)
        });

        Object {
            model_instance,
            body,
            transform,
        }
    }

    pub fn deinit(&mut self, context: &gpu::Context) {
        if let Some(model) = &mut self.model {
            Arc::get_mut(model).unwrap().free(context);
        }
    }
}
