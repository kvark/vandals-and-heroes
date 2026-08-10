//! End-to-end voxel-cast orbit tests.
//!
//! Two synthetic scenes, both rendered from a ring of camera angles around
//! a feature placed at the heightmap centre, both run through the real
//! voxel bake + DDA + K-walk pipeline.
//!
//! 1. `voxel_cast_handles_rays_from_all_directions`: a tall hard-edged
//!    BLOCK in a flat plain. The block is the maximally-hostile case for
//!    bake/K-walk direction handling (sharp vertical sides, below→above
//!    transitions for bottom-up rays). This test caught the original
//!    bottom-up bug.
//!
//! 2. `voxel_cast_hill_is_contiguous`: a smooth Gaussian HILL — the
//!    "should-be-easy" case. A continuous bilinear surface across the
//!    whole hill; any pixel showing sky surrounded by terrain pixels is a
//!    fundamental DDA/K-walk failure (not a sub-cell aliasing trade-off,
//!    not a heightmap pathology). Catches the silhouette/face stippling
//!    artefact.
//!
//! Heavy: needs a windowing system + GPU. Run with
//!     cargo test --release --test voxel_orbit -- --nocapture
//! `--nocapture` so the per-camera coverage / stippling counts land on
//! stdout, and the failing PNGs in `/tmp/voxel_orbit/` are easy to inspect.

use blade_graphics as gpu;
use vandals_and_heroes::{Camera, Render, Terrain, Voxels, config};

const HEIGHTMAP_W: u32 = 256;
const HEIGHTMAP_H: u32 = 256;
const BLOCK_HALF_W: u32 = 16;
const BG_ALPHA: u8 = 51; // 0.2 → ground_r = 11.0
const BLOCK_ALPHA: u8 = 204; // 0.8 → ground_r = 14.0
const FRAME_W: u32 = 256;
const FRAME_H: u32 = 256;

fn synth_heightmap() -> Vec<u8> {
    let mut buf = vec![0u8; (HEIGHTMAP_W * HEIGHTMAP_H * 4) as usize];
    let cx = HEIGHTMAP_W / 2;
    let cy = HEIGHTMAP_H / 2;
    for y in 0..HEIGHTMAP_H {
        for x in 0..HEIGHTMAP_W {
            let i = ((y * HEIGHTMAP_W + x) * 4) as usize;
            // Plain yellow albedo so block vs. background look identical — we
            // care about the voxel cast hitting *some* surface, not what
            // colour it picks.
            buf[i] = 200;
            buf[i + 1] = 180;
            buf[i + 2] = 80;
            let in_block = x >= cx - BLOCK_HALF_W
                && x < cx + BLOCK_HALF_W
                && y >= cy - BLOCK_HALF_W
                && y < cy + BLOCK_HALF_W;
            buf[i + 3] = if in_block { BLOCK_ALPHA } else { BG_ALPHA };
        }
    }
    buf
}

/// Smooth Gaussian hill at the heightmap centre. Base alpha 0.2 (ground_r =
/// 11) everywhere; the hill rises to peak alpha 0.85 (ground_r ≈ 14.25)
/// with σ chosen so the hill is wide enough to span ~25 voxel cells in
/// both θ and z (it's a *hill*, not a needle — every cell on the hill has
/// neighbours of similar height, so there's no isolated-tall-texel
/// pathology to blame the bake on).
fn synth_hill_heightmap() -> Vec<u8> {
    let mut buf = vec![0u8; (HEIGHTMAP_W * HEIGHTMAP_H * 4) as usize];
    let cx = HEIGHTMAP_W as f32 * 0.5;
    let cy = HEIGHTMAP_H as f32 * 0.5;
    // Width in texels: ~50 (half spread at 1σ ≈ 25 texels). With the
    // halved voxel grid that's ~25 cells across the hill — wide enough
    // that the silhouette covers many pixels at all camera angles.
    let sigma = 25.0;
    for y in 0..HEIGHTMAP_H {
        for x in 0..HEIGHTMAP_W {
            let i = ((y * HEIGHTMAP_W + x) * 4) as usize;
            let dx = (x as f32) - cx;
            let dy = (y as f32) - cy;
            let g = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            let alpha = 0.2 + 0.65 * g;
            buf[i] = 200;
            buf[i + 1] = 180;
            buf[i + 2] = 80;
            buf[i + 3] = (alpha * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    buf
}

fn classify_pixel(bgra: &[u8], x: u32, y: u32, w: u32) -> bool {
    // Returns true if the pixel is "sky" (env-white fallback). Same rule
    // as terrain_coverage_pct so the two metrics agree.
    let i = ((y * w + x) * 4) as usize;
    let r = bgra[i + 2];
    let g = bgra[i + 1];
    let b = bgra[i];
    r.max(g).max(b) > 240
}

/// Count "stippling" pixels — pixels classified as sky whose 3×3
/// neighbourhood is *predominantly* terrain. This catches single dots and
/// 2-3 pixel sky clusters embedded in a contiguous terrain region (the
/// "looks broken, gaps are more than voxel size" artefact). True silhouette
/// edges have many sky neighbours, so they don't trigger this; only sky
/// pixels surrounded by terrain do.
fn isolated_sky_count(bgra: &[u8], w: u32, h: u32) -> u32 {
    let mut iso = 0u32;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if !classify_pixel(bgra, x, y, w) {
                continue;
            }
            let mut terrain_neighbours = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;
                    if !classify_pixel(bgra, nx, ny, w) {
                        terrain_neighbours += 1;
                    }
                }
            }
            // ≥ 6/8 neighbours are terrain → this sky pixel is an
            // isolated hole, not part of a contiguous sky region.
            if terrain_neighbours >= 6 {
                iso += 1;
            }
        }
    }
    iso
}

fn save_frame(bgra: &[u8], w: u32, h: u32, path: &std::path::Path) {
    let mut rgba = vec![0u8; bgra.len()];
    for chunk in 0..(bgra.len() / 4) {
        let o = chunk * 4;
        rgba[o] = bgra[o + 2];
        rgba[o + 1] = bgra[o + 1];
        rgba[o + 2] = bgra[o];
        rgba[o + 3] = bgra[o + 3];
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = std::fs::File::create(path).expect("create png");
    let buf = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(buf, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("png header");
    writer.write_image_data(&rgba).expect("png write");
}

fn make_camera(pos: [f32; 3], look_at: [f32; 3], clip_far: f32) -> Camera {
    use nalgebra::{UnitQuaternion, Vector3};
    let pos = Vector3::new(pos[0], pos[1], pos[2]);
    let target = Vector3::new(look_at[0], look_at[1], look_at[2]);
    let up = Vector3::new(0.0, 0.0, 1.0);
    let forward = (target - pos).normalize();
    let rot = UnitQuaternion::look_at_lh(&forward, &up).inverse();
    Camera {
        pos,
        rot,
        fov_y: 1.0,
        clip: 0.1..clip_far,
        fly_speed: 0.0,
        rotate_speed: 0.0,
        drag_speed: 0.0,
    }
}

/// Bucket a rendered frame into "terrain" vs "sky" pixels. With no env
/// texture configured the shader falls back to the white dummy, so "sky"
/// pixels come out as near-white; the synthetic terrain's 200/180/80
/// albedo modulated by shading is never close to white.
fn terrain_coverage_pct(bgra: &[u8]) -> f32 {
    let mut sky = 0u32;
    let mut hit = 0u32;
    for chunk in 0..(bgra.len() / 4) {
        let o = chunk * 4;
        let r = bgra[o + 2];
        let g = bgra[o + 1];
        let b = bgra[o];
        if r.max(g).max(b) > 240 {
            sky += 1;
        } else {
            hit += 1;
        }
    }
    100.0 * hit as f32 / (hit + sky) as f32
}

/// Bootstrap a headless GPU + Render with no terrain. Returns None if the
/// host has no windowing system. winit forbids more than one EventLoop
/// per process, so the test that uses this calls it ONCE and re-bakes
/// terrain in place between scenes.
fn try_bootstrap() -> Option<(
    winit::event_loop::EventLoop<()>,
    winit::window::Window,
    Render,
)> {
    use winit::event_loop::EventLoop;
    #[cfg(all(unix, not(target_os = "macos")))]
    let event_loop = {
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        use winit::platform::x11::EventLoopBuilderExtX11;
        let mut builder = EventLoop::builder();
        EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
        EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
        match builder.build() {
            Ok(el) => el,
            Err(e) => {
                eprintln!("voxel_orbit: no windowing system available ({e}); skipping");
                return None;
            }
        }
    };
    #[cfg(not(all(unix, not(target_os = "macos"))))]
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("voxel_orbit: no windowing system available ({e}); skipping");
            return None;
        }
    };
    let window_attrs = winit::window::Window::default_attributes()
        .with_title("voxel_orbit test")
        .with_inner_size(winit::dpi::PhysicalSize::new(FRAME_W, FRAME_H))
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
        width: FRAME_W,
        height: FRAME_H,
        depth: 1,
    };
    let gpu_surface = gpu_context.create_surface(&window).expect("surface");
    let mut render = Render::new(gpu_context, gpu_surface, extent);

    let base_config: config::Config = ron::de::from_bytes(
        &std::fs::read("data/config.ron").expect("read data/config.ron"),
    )
    .expect("parse data/config.ron");
    render.set_ray_params(&base_config.ray);
    render.set_voxel_render_mode(config::RenderMode::VoxelHiZ.to_mode_u32());

    Some((event_loop, window, render))
}

/// Bake a terrain scene into the existing Render. Returns the Terrain so
/// the caller can hand it back to render_to_buffer.
fn bake_scene(render: &mut Render, heightmap: Vec<u8>) -> Terrain {
    let mut loader = render.start_loading();
    let map_extent = gpu::Extent {
        width: HEIGHTMAP_W,
        height: HEIGHTMAP_H,
        depth: 1,
    };
    let texture = loader.load_terrain(map_extent, &heightmap);
    let map_config = vandals_and_heroes::config::Map {
        radius: 10.0..15.0,
        length: 50.0,
        density: 1.0,
        is_sphere: false,
    };
    let voxel_dim =
        vandals_and_heroes::pick_voxel_dim(map_extent.width, map_extent.height, 128);
    let voxels = Voxels::new(loader.context(), voxel_dim);
    loader.upload_voxel_metadata(&voxels);
    let terrain = Terrain {
        config: map_config,
        texture,
        env_texture: None,
        voxels,
    };
    let submission = loader.finish();
    render.accept_submission(submission);
    render.wait_for_gpu();
    render.set_shadow_extent(map_extent);
    render.bake_terrain_voxels(&terrain);
    terrain
}

/// Combined orbit test: one EventLoop+Render (winit refuses more than one
/// per process), two synthetic scenes baked back-to-back. The block sub-
/// test catches the bottom-up/below→above bugs; the hill sub-test catches
/// silhouette stippling — pixels that come back as env-sky despite being
/// surrounded by terrain pixels.
#[test]
fn voxel_cast_orbit() {
    let Some((_event_loop, window, mut render)) = try_bootstrap() else {
        return;
    };
    let extent = gpu::Extent {
        width: FRAME_W,
        height: FRAME_H,
        depth: 1,
    };

    let mut all_failures: Vec<String> = Vec::new();

    // ====================== SCENE 1: BLOCK ======================
    // Block at heightmap centre (u=0.5 → θ=π, v=0.5 → z=0). Cartesian
    // block centre is ≈ (-r_block, 0, 0); aim everyone at (-12.5, 0, 0).
    {
        let terrain = bake_scene(&mut render, synth_heightmap());
        let target = [-12.5, 0.0, 0.0];
        let cameras: &[(&str, [f32; 3])] = &[
            // ABOVE (rays go inward from outer cylinder): canonical above→below.
            ("01_above_centre", [-14.9, 0.0, 0.0]),
            ("02_above_east", [-14.5, 4.0, 0.0]),
            ("03_above_north", [-14.5, 0.0, 4.0]),
            ("04_above_diag", [-14.5, 3.0, 3.0]),
            // SIDE: tangential rays from the same r as the block.
            ("05_side_east", [-12.0, 7.0, 0.0]),
            ("06_side_west", [-12.0, -7.0, 0.0]),
            ("07_side_north_z", [-12.0, 0.0, 7.0]),
            ("08_side_south_z", [-12.0, 0.0, -7.0]),
            // BOTTOM-UP: camera just inside r_start (below background ground),
            // ray points UP through the block. These are the below→above case.
            ("09_lowR_centre", [-10.3, 0.0, 0.0]),
            ("10_lowR_east", [-10.3, 3.0, 0.0]),
            ("11_lowR_north_z", [-10.3, 0.0, 3.0]),
            ("12_lowR_diag", [-10.3, 2.0, 2.0]),
        ];
        let clip_far = 4.0 * terrain.config.radius.end;
        // The block fills a large slice of FOV at this distance. 30% is
        // well above the >0% noise floor and well below the 80-100% we
        // see when the cast is working — a comfortable margin so this
        // sub-test isn't flaky to small lighting / shading tweaks.
        let min_pct = 30.0;
        println!("voxel_orbit BLOCK coverage report:");
        for (label, pos) in cameras.iter() {
            let camera = make_camera(*pos, target, clip_far);
            let bgra = render.render_to_buffer(&camera, &terrain, &Vec::new(), extent);
            let pct = terrain_coverage_pct(&bgra);
            println!("  {label}: {pct:5.1}% terrain");
            if pct < min_pct {
                all_failures.push(format!("block:{label} only saw {pct:.1}% terrain"));
            }
        }
        render.wait_for_gpu();
    }

    // ====================== SCENE 2: HILL ======================
    // A smooth Gaussian hill is the easy case for a voxel ray-caster:
    // the bilinear surface is continuous, every cell on the hill has
    // neighbours at similar height (no isolated-tall-texel pathology),
    // every cell along any reasonable ray's path through the hill IS
    // marked occupied by the bake. So any pixel that comes back as sky
    // despite being surrounded by terrain pixels (an "isolated sky
    // pixel") is a fundamental DDA/K-walk bug — not a cell-resolution
    // trade-off, not a sub-cell aliasing artefact, not heightmap
    // pathology. This is the controlled-scene counterpart to the
    // in-game silhouette stippling artefact.
    {
        let terrain = bake_scene(&mut render, synth_hill_heightmap());
        let target = [-13.0, 0.0, 0.0];
        let cameras: &[(&str, [f32; 3])] = &[
            // Overhead: rays mostly radial inward, easiest case.
            ("01_above_centre", [-14.9, 0.0, 0.0]),
            // Side at peak height — the silhouette is the hill ridge.
            ("02_side_east_peak", [-13.5, 5.0, 0.0]),
            ("03_side_west_peak", [-13.5, -5.0, 0.0]),
            ("04_side_north_peak", [-13.5, 0.0, 5.0]),
            ("05_side_south_peak", [-13.5, 0.0, -5.0]),
            // Tangential at base level — silhouette covers a wide arc.
            ("06_base_east", [-11.5, 7.0, 0.0]),
            ("07_base_west", [-11.5, -7.0, 0.0]),
            ("08_base_north", [-11.5, 0.0, 7.0]),
            ("09_base_south", [-11.5, 0.0, -7.0]),
            // Diagonal angles where the hill's surface curves in 2 axes
            // simultaneously, exercising the cell-exit math in θ + z at once.
            ("10_diag_ne_peak", [-13.5, 4.0, 4.0]),
            ("11_diag_sw_peak", [-13.5, -4.0, -4.0]),
            ("12_diag_se_base", [-11.5, 5.0, -5.0]),
        ];
        let clip_far = 4.0 * terrain.config.radius.end;
        let outdir = std::path::Path::new("/tmp/voxel_orbit/hill");
        let _ = std::fs::remove_dir_all(outdir);
        // Allowable isolated-sky budget per frame. Zero would catch
        // genuine single-pixel stippling but be brittle to bake/shading
        // tweaks at the silhouette edge; anything well above this means
        // the algorithm is dropping pixels in the interior of a
        // contiguous terrain region.
        let max_isolated = 32u32;
        println!("voxel_orbit HILL stippling report (target = {target:?}):");
        for (label, pos) in cameras.iter() {
            let camera = make_camera(*pos, target, clip_far);
            let bgra = render.render_to_buffer(&camera, &terrain, &Vec::new(), extent);
            let pct = terrain_coverage_pct(&bgra);
            let iso = isolated_sky_count(&bgra, FRAME_W, FRAME_H);
            let out = outdir.join(format!("{label}.png"));
            save_frame(&bgra, FRAME_W, FRAME_H, &out);
            let marker = if iso > max_isolated { " STIPPLED" } else { "" };
            println!(
                "  {label}: {pct:5.1}% terrain, {iso:4} isolated sky px{marker} → {}",
                out.display()
            );
            if iso > max_isolated {
                all_failures.push(format!(
                    "hill:{label} stippled {iso} isolated sky pixels"
                ));
                // Also save VoxelDebug + RayMarch for the failing case so
                // we can see what the DDA was doing and what march finds.
                render.set_voxel_render_mode(
                    config::RenderMode::VoxelDebug.to_mode_u32(),
                );
                let dbg = render.render_to_buffer(&camera, &terrain, &Vec::new(), extent);
                save_frame(
                    &dbg,
                    FRAME_W,
                    FRAME_H,
                    &outdir.join(format!("{label}_debug.png")),
                );
                render.set_voxel_render_mode(
                    config::RenderMode::RayMarch.to_mode_u32(),
                );
                let march = render.render_to_buffer(&camera, &terrain, &Vec::new(), extent);
                save_frame(
                    &march,
                    FRAME_W,
                    FRAME_H,
                    &outdir.join(format!("{label}_march.png")),
                );
                render.set_voxel_render_mode(
                    config::RenderMode::VoxelHiZ.to_mode_u32(),
                );
            }
        }
        render.wait_for_gpu();
    }

    render.deinit();
    drop(window);

    assert!(
        all_failures.is_empty(),
        "voxel_orbit failures:\n  {}",
        all_failures.join("\n  ")
    );
}
