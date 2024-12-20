use blade_graphics as gpu;

pub struct Submission {
    pub sync_point: gpu::SyncPoint,
    pub temp_buffers: Vec<gpu::Buffer>,
}