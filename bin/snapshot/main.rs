//! Headless renderer: loads a map, sets the camera from a config file (or
//! defaults), renders ONE frame to disk, exits. Used for iterating on the
//! terrain renderer without depending on a window or external screenshot
//! tools.
//!
//!     cargo run --release --bin snapshot -- snapshot.ron
//!
//! `snapshot.ron` overrides the on-disk `data/config.ron` for map + render
//! mode and adds the camera position and output path. See `SnapshotConfig`
//! below for the field set.

use blade_graphics as gpu;
use std::{fs, path::PathBuf};
use vandals_and_heroes::{Camera, Render, Terrain, config, config::WorldShape, tin};

#[derive(serde::Deserialize)]
struct SnapshotConfig {
    /// Map name under `data/maps/` (overrides `data/config.ron`).
    #[serde(default)]
    map: Option<String>,
    /// Optional environment override.
    #[serde(default)]
    environment: Option<String>,
    /// Camera world position.
    pos: [f32; 3],
    /// Camera rotation quaternion as (i, j, k, w). When present, fully
    /// specifies the camera orientation (preferred — matches the F12 dump
    /// from the game so the snapshot reproduces the live view exactly).
    #[serde(default)]
    rot: Option<[f32; 4]>,
    /// Legacy: look-at target. Only used when `rot` is not set.
    #[serde(default)]
    look_at: Option<[f32; 3]>,
    /// Legacy: world-up vector — only used with `look_at`. Defaults to
    /// (0, 0, 1) (the cylinder axis). For curved worlds this is rarely
    /// the right choice; prefer `rot`.
    #[serde(default = "default_up")]
    up: [f32; 3],
    /// Vertical field of view, radians.
    #[serde(default = "default_fov")]
    fov_y: f32,
    /// Render output dimensions.
    extent: [u32; 2],
    /// PNG output path.
    output: PathBuf,
}

fn default_up() -> [f32; 3] {
    [0.0, 0.0, 1.0]
}
fn default_fov() -> f32 {
    1.0
}

fn parse_args() -> SnapshotConfig {
    let arg_path = std::env::args().nth(1).unwrap_or_else(|| "snapshot.ron".into());
    let bytes = fs::read(&arg_path)
        .unwrap_or_else(|e| panic!("Failed to read snapshot config {arg_path}: {e}"));
    ron::de::from_bytes(&bytes).expect("Failed to parse snapshot config")
}

/// Build a camera from `rot` (preferred) or fall back to `look_at`/`up`.
/// The look-at fallback is approximate: it can't recover the roll about the
/// forward axis the game keeps to track the car's radial up.
fn make_camera(snap: &SnapshotConfig, clip_far: f32) -> Camera {
    use nalgebra::{Quaternion, UnitQuaternion, Vector3};
    let pos = Vector3::new(snap.pos[0], snap.pos[1], snap.pos[2]);
    let rot = if let Some(q) = snap.rot {
        UnitQuaternion::from_quaternion(Quaternion::new(q[3], q[0], q[1], q[2]))
    } else {
        let target_arr = snap.look_at.expect(
            "snapshot config must provide either `rot` (preferred — dump via F12) \
             or legacy `look_at`",
        );
        let target = Vector3::new(target_arr[0], target_arr[1], target_arr[2]);
        let up = Vector3::new(snap.up[0], snap.up[1], snap.up[2]);
        let forward = (target - pos).normalize();
        UnitQuaternion::look_at_lh(&forward, &up).inverse()
    };
    Camera {
        pos,
        rot,
        fov_y: snap.fov_y,
        clip: 0.5..clip_far,
        fly_speed: 0.0,
        rotate_speed: 0.0,
        drag_speed: 0.0,
    }
}

fn save_png(path: &PathBuf, extent: gpu::Extent, bgra: &[u8]) {
    use png::Encoder;
    // PNG wants RGBA, swap B↔R per pixel.
    let mut rgba = vec![0u8; bgra.len()];
    for chunk in 0..(bgra.len() / 4) {
        let o = chunk * 4;
        rgba[o] = bgra[o + 2];
        rgba[o + 1] = bgra[o + 1];
        rgba[o + 2] = bgra[o];
        rgba[o + 3] = bgra[o + 3];
    }
    let file = fs::File::create(path).expect("create snapshot output");
    let buf = std::io::BufWriter::new(file);
    let mut enc = Encoder::new(buf, extent.width, extent.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(&rgba).expect("png write");
}

fn main() {
    env_logger::init();

    // Pull the snapshot config + the base data/config.ron so we share the
    // game's map/car defaults when the snapshot config doesn't override.
    let snap = parse_args();
    let base_config: config::Config = ron::de::from_bytes(
        &fs::read("data/config.ron").expect("read data/config.ron"),
    )
    .expect("parse data/config.ron");
    let map_name = snap.map.clone().unwrap_or(base_config.map.clone());
    let env_name = snap
        .environment
        .clone()
        .or(base_config.environment.clone());

    log::info!("Snapshot: map={map_name} extent={}×{}", snap.extent[0], snap.extent[1]);

    // ---- winit: tiny invisible window (Wayland needs the surface) ----
    use winit::event_loop::EventLoop;
    let event_loop = EventLoop::new().expect("event loop");
    let window_attrs = winit::window::Window::default_attributes()
        .with_title("Vandals snapshot")
        .with_inner_size(winit::dpi::PhysicalSize::new(snap.extent[0], snap.extent[1]))
        .with_visible(false);
    #[allow(deprecated)]
    let window = event_loop.create_window(window_attrs).expect("window");

    let gpu_context = unsafe {
        gpu::Context::init(gpu::ContextDesc {
            presentation: true,
            validation: cfg!(debug_assertions),
            ..Default::default()
        })
    }
    .expect("gpu init");
    let extent = gpu::Extent {
        width: snap.extent[0],
        height: snap.extent[1],
        depth: 1,
    };
    let gpu_surface = gpu_context.create_surface(&window).expect("surface");
    let mut render = Render::new(gpu_context, gpu_surface, extent);

    // ---- Load terrain ----
    let mut loader = render.start_loading();
    let map_path = PathBuf::from("data/maps").join(&map_name);
    let mut map_config: config::Map = ron::de::from_bytes(
        &fs::read(map_path.join("map.ron")).expect("read map.ron"),
    )
    .expect("parse map.ron");
    let (texture, map_extent, height_alpha) = loader.load_png(&map_path.join("map.png"));
    if map_config.length == 0.0 {
        let circumference = 2.0 * std::f32::consts::PI * map_config.radius.start;
        map_config.length = circumference * (map_extent.height as f32) / (map_extent.width as f32);
    }
    let env_texture = env_name.as_ref().map(|name| {
        let env_path = PathBuf::from("data/envs").join(format!("{name}.png"));
        loader.load_environment(&env_path)
    });
    let mesh = tin::build(
        &height_alpha,
        map_extent.width,
        map_extent.height,
        &map_config,
        base_config.terrain_quality,
    );
    let chunks = loader.load_terrain_mesh(&mesh);
    let terrain = Terrain {
        config: map_config,
        texture,
        env_texture,
        chunks,
    };
    let submission = loader.finish();
    render.accept_submission(submission);
    render.wait_for_gpu();
    render.set_shadow_extent(map_extent);

    // ---- Render one frame ----
    let clip_far = match terrain.config.shape {
        WorldShape::Sphere => 4.0 * terrain.config.radius.end,
        WorldShape::Cylinder => terrain.config.length,
        WorldShape::Torus => {
            terrain.config.length / std::f32::consts::PI + 2.0 * terrain.config.radius.end
        }
    };
    let camera = make_camera(&snap, clip_far);
    let bgra = render.render_to_buffer(&camera, &terrain, &Vec::new(), extent);

    save_png(&snap.output, extent, &bgra);
    log::info!("Wrote {}", snap.output.display());

    render.wait_for_gpu();
    terrain.free(render.context());
    render.deinit();
    drop(window);
}
