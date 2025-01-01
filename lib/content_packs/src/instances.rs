use vandals_and_heroes::{ModelInstance, Physics, PhysicsBodyHandle};

pub struct Object {
    pub model_instance: Option<ModelInstance>,
    pub body: Option<PhysicsBodyHandle>,
    pub transform: nalgebra::Isometry3<f32>,
    // TODO: script instance
}

impl Object {
    pub fn update(&mut self, physics: &Physics) {
        if let Some(body) = &self.body {
            let transform = physics.get_transform(body.rigid_body_handle);
            self.transform = transform;
        }

        if let Some(model_instance) = &mut self.model_instance {
            model_instance.transform = self.transform;
        }
    }

    pub fn model_instance(&self) -> Option<&ModelInstance> {
        self.model_instance.as_ref()
    }
}