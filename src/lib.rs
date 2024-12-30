#![allow(irrefutable_let_patterns)]

mod camera;
pub mod config;
mod loader;
mod model;
mod physics;
mod render;
mod submission;
mod texture;
mod terrain;

pub use camera::Camera;
use config::{Map as MapConfig, Ray as RayConfig};
pub use loader::Loader;
pub use model::{Geometry, Material, Model, ModelInstance, GeometryDesc, MaterialDesc, ModelDesc};
pub use physics::{Physics, TerrainBody};
pub use render::{Render, Vertex};
use submission::Submission;
pub use texture::Texture;
pub use terrain::Terrain;
