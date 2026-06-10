#![allow(
    irrefutable_let_patterns,
    clippy::match_like_matches_macro,
    clippy::redundant_pattern_matching,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::single_match,
    clippy::too_many_arguments,
    clippy::collapsible_if
)]
#![warn(
    trivial_numeric_casts,
    unused_extern_crates,
    clippy::pattern_type_mismatch
)]

mod camera;
pub mod config;
mod cyl_heightfield;
mod loader;
mod model;
mod physics;
mod recorder;
mod render;
mod sphere_heightfield;
mod submission;
mod terrain;
mod texture;

pub use camera::Camera;
use config::{Map as MapConfig, Ray as RayConfig};
pub use cyl_heightfield::{CylDispatcher, CylindricalHeightField};
pub use loader::Loader;
pub use model::{
    Geometry, GeometryDesc, Material, MaterialDesc, Model, ModelDesc, ModelInstance, VertexDesc,
};
pub use physics::{Kinematics, Physics, PhysicsBodyHandle, TerrainBody};
pub use recorder::{ObjectSnapshot, Recorder, Snapshot};
pub use render::{Render, Vertex};
pub use sphere_heightfield::SphericalHeightField;
use submission::Submission;
pub use terrain::Terrain;
pub use texture::Texture;
