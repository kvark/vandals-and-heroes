use std::{ops::Range, path::PathBuf};

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

fn default_terrain_quality() -> f32 {
    // Matches the sweet spot from the vange-rs meshing write-up: visually
    // indistinguishable from the full-resolution surface while cutting the
    // triangle count by an order of magnitude.
    0.75
}

#[derive(serde::Deserialize)]
pub struct Config {
    pub map: String,
    pub car: String,
    #[serde(default)]
    pub environment: Option<String>,
    #[serde(default)]
    pub record: Option<Recorder>,
    /// Debug-snow density: one particle per `snow_area_per_particle_m2` m² of
    /// world surface. Smaller = denser snow = slower frame. `0` (the default)
    /// disables snow entirely — set a positive value in `data/config.ron`
    /// to opt in.
    #[serde(default)]
    pub snow_area_per_particle_m2: f32,
    /// Terrain mesh fit quality in `0..=1`. `1.0` fits the mesh to within
    /// one height-map quantisation step; every 0.25 below doubles the
    /// tolerance (and roughly halves the triangles).
    #[serde(default = "default_terrain_quality")]
    pub terrain_quality: f32,
}

/// The topology the height map is wrapped onto.
#[derive(serde::Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorldShape {
    /// A cylinder around the Z axis: `u` wraps around it (θ), `v` runs along
    /// it. The world ends at `z = ±length/2`.
    #[default]
    Cylinder,
    /// A sphere at the origin; the height map wraps it via Lambert equal-area
    /// cylindrical projection (u = θ/2π, v = (sin φ + 1)/2). `length` is
    /// ignored.
    Sphere,
    /// The cylinder's axis bent into a circle of radius `length / 2π` in the
    /// XY plane, so the axial direction wraps too — a world with no ends,
    /// meant for long maps. Requires `length / 2π > radius.end` or the tube
    /// self-intersects.
    Torus,
}

#[derive(serde::Deserialize)]
pub struct Map {
    pub radius: Range<f32>,
    #[serde(default)]
    pub length: f32,
    pub density: f32,
    #[serde(default)]
    pub shape: WorldShape,
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

fn default_body_color() -> [f32; 4] {
    // Rusty brown, applied as a multiplicative tint over the GLB material's
    // base_color_factor. Vangers vehicles are rust-and-dust; this is the
    // baseline before the player's palette / livery would be applied.
    [0.55, 0.35, 0.20, 1.0]
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
    /// Multiplicative tint applied to every loaded material's base_color_factor.
    /// Lets a `car.ron` override the GLB's default white materials without
    /// re-authoring the model.
    #[serde(default = "default_body_color")]
    pub body_color: [f32; 4],
}
