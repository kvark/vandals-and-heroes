#![allow(irrefutable_let_patterns)]

mod camera;
pub mod config;
mod loader;
mod model;
mod physics;
mod render;
mod submission;
mod texture;

pub use camera::Camera;
use config::{Map as MapConfig, Ray as RayConfig};
pub use loader::Loader;
pub use model::{Geometry, Material, Model};
pub use physics::{Physics, TerrainBody};
pub use render::{Render, Vertex};
use submission::Submission;
pub use texture::Texture;

use std::sync::Arc;

pub struct Object {
    pub model: Arc<Model>,
    pub rigid_body: rapier3d::dynamics::RigidBodyHandle,
    pub transform: nalgebra::Isometry3<f32>,
}
