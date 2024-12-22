#![allow(irrefutable_let_patterns)]

pub mod camera;
pub mod config;
pub mod loader;
pub mod model;
pub mod render;
pub mod submission;
pub mod texture;

use camera::Camera;
use config::Ray as RayConfig;
use loader::Loader;
use model::{Geometry, Material, Model, ModelInstance};
use render::Vertex;
use submission::Submission;
use texture::Texture;
