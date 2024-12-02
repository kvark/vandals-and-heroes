use std::ops::Range;

#[derive(serde::Deserialize)]
pub struct Ray {
    pub march_count: u32,
    pub march_closest_power: f32,
    pub bisect_count: u32,
}

#[derive(serde::Deserialize)]
pub struct Main {
    pub map: String,
    pub map_radius: Range<f32>,
    pub ray: Ray,
}
