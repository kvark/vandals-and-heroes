use std::{ops::Range, path::PathBuf};

#[derive(serde::Deserialize)]
pub struct Ray {
    pub march_count: u32,
    pub march_closest_power: f32,
    pub bisect_count: u32,
}

#[derive(serde::Deserialize, Clone, Copy, Debug)]
pub enum RecorderFormat {
    Ron,
    Bincode,
}

#[derive(serde::Deserialize)]
pub struct Recorder {
    pub path: PathBuf,
    pub format: RecorderFormat,
}

#[derive(serde::Deserialize)]
pub struct Config {
    pub map: String,
    pub car: String,
    pub ray: Ray,
    #[serde(default)]
    pub environment: Option<String>,
    #[serde(default)]
    pub record: Option<Recorder>,
}

#[derive(serde::Deserialize)]
pub struct Map {
    pub radius: Range<f32>,
    #[serde(default)]
    pub length: f32,
    pub density: f32,
}

#[derive(serde::Deserialize)]
pub struct Wheel {
    /// Wheel center in chassis-local coordinates.
    pub position: [f32; 3],
    pub radius: f32,
}

fn default_wheel_axis() -> [f32; 3] {
    [0.0, 0.0, 1.0]
}

fn default_motor_max_velocity() -> f32 {
    20.0
}

fn default_motor_max_force() -> f32 {
    50.0
}

#[derive(serde::Deserialize)]
pub struct Car {
    pub scale: f32,
    pub density: f32,
    #[serde(default)]
    pub wheels: Vec<Wheel>,
    /// Wheel rotation axis in chassis-local coordinates (the axle direction).
    #[serde(default = "default_wheel_axis")]
    pub wheel_axis: [f32; 3],
    #[serde(default = "default_motor_max_velocity")]
    pub motor_max_velocity: f32,
    #[serde(default = "default_motor_max_force")]
    pub motor_max_force: f32,
}
