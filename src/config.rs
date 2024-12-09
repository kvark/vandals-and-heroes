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
pub struct Car {
    pub scale: f32,
    pub density: f32,
}
