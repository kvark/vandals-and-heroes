use vandals_and_heroes::{ModelInstance, Physics, PhysicsBodyHandle, Terrain, TerrainBody};

pub struct Object {
    pub model_instance: Option<ModelInstance>,
    pub body: Option<PhysicsBodyHandle>,
    pub transform: nalgebra::Isometry3<f32>,
    // TODO: script instance
}

pub struct TerrainObject {
    pub terrain: Terrain,
    pub body: TerrainBody,
}