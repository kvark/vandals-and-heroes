use std::ops::Range;

#[derive(serde::Deserialize)]
pub struct Ray {
    pub march_count: u32,
    pub march_closest_power: f32,
    pub bisect_count: u32,
}

#[derive(serde::Deserialize)]
pub struct Config {
    pub map: String,
    pub car: String,
    pub ray: Ray,
}

#[derive(serde::Deserialize)]
pub struct Map {
    pub radius: Range<f32>,
    #[serde(default)]
    pub length: f32,
    pub density: f32,
}

#[derive(serde::Deserialize)]
pub enum Shape {
    Mesh,
    Cylinder { depth: f32, radius: f32 },
}

#[derive(serde::Deserialize)]
pub struct Model {
    pub model: String,
    pub scale: f32,
    pub density: f32,
    #[serde(default)]
    pub friction: f32,
    pub shape: Shape,
}

#[derive(serde::Deserialize)]
pub struct Axle {
    pub wheel: Model,
    pub xs: Vec<f32>,
    pub y: f32,
    pub z: f32,
}

#[derive(serde::Deserialize)]
pub struct Car {
    pub body: Model,
    pub drive_factor: f32,
    pub axles: Vec<Axle>,
}
