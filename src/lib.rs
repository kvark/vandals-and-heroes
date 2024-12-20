#![allow(irrefutable_let_patterns)]

pub mod camera;
pub mod config;
pub mod loader;
pub mod model;
pub mod render;
pub mod texture;
pub mod submission;

use texture::Texture;
use submission::Submission;
use model::{Model, Geometry, Material, ModelInstance};
use render::Vertex;
use loader::Loader;
use config::Ray as RayConfig;
use camera::Camera;