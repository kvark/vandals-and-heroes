use std::ops::Range;

#[derive(serde::Deserialize)]
pub struct Main {
    pub map: String,
    pub map_radius: Range<f32>,
}
