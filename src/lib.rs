#![allow(irrefutable_let_patterns)]

mod camera;
pub mod config;
mod loader;
mod model;
mod physics;
mod render;
mod submission;
mod terrain;
mod texture;

pub use camera::Camera;
use config::{Map as MapConfig, Ray as RayConfig};
pub use loader::Loader;
pub use model::{Geometry, GeometryDesc, Material, MaterialDesc, Model, ModelDesc, ModelInstance};
pub use physics::{Physics, PhysicsBodyHandle, TerrainBody};
pub use render::{Render, Vertex};
use submission::Submission;
pub use terrain::Terrain;
pub use texture::Texture;
