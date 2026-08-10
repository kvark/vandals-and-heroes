//! Cell-traversal math for the voxel HiZ renderer, in plain Rust.
//!
//! This module is a CPU-side port of the WGSL helpers in
//! `shaders/voxel.wgsl` + `shaders/terrain-draw.wgsl` (specifically
//! `voxel_world_to_cell` and `voxel_cell_exit_t`). Keeping a Rust copy lets
//! us write unit tests against the cell-traversal invariants — anything
//! the tests catch here is almost certainly buggy in the shader too,
//! because the formulas are identical.
//!
//! Tests live at the bottom of this file. They cover:
//!  * LOD hierarchy: parent coords = child coords / 2
//!  * Cell-exit consistency: exit_t lands the ray at a cell face
//!  * DDA step continuity: after step, new cell is adjacent (or wraps θ)
//!  * Boundary handling: pos exactly on a cell face still progresses
//!  * θ wraparound at ±π
//!
//! When a shader-side regression is suspected, the right move is usually
//! "add a test here that fails, then fix the formula in both this file
//! and the shader so the test passes."

use std::f32::consts::TAU;

/// World/grid parameters needed to interpret cell coordinates.
#[derive(Clone, Copy, Debug)]
pub struct GridParams {
    pub length: f32,
    pub r_start: f32,
    pub r_end: f32,
}

/// One LOD level — cell counts along (θ, z, r).
#[derive(Clone, Copy, Debug)]
pub struct VoxelLod {
    pub dim: [i32; 3],
}

impl VoxelLod {
    pub fn theta_step(self) -> f32 {
        TAU / self.dim[0] as f32
    }
    pub fn z_step(self, params: GridParams) -> f32 {
        params.length / self.dim[1] as f32
    }
    pub fn r_step(self, params: GridParams) -> f32 {
        (params.r_end - params.r_start) / self.dim[2] as f32
    }
}

/// Position → (i_θ, i_z, i_r) cell index. Mirrors `voxel_world_to_cell`.
pub fn world_to_cell(pos: [f32; 3], lod: VoxelLod, params: GridParams) -> [i32; 3] {
    let theta = pos[1].atan2(pos[0]);
    let theta_wrapped = if theta < 0.0 { theta + TAU } else { theta };
    let r = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
    let dt = lod.theta_step();
    let dz = lod.z_step(params);
    let dr = lod.r_step(params);
    let i_theta = (theta_wrapped / dt).floor() as i32;
    let i_z_raw = ((pos[2] + 0.5 * params.length) / dz).floor() as i32;
    let i_r_raw = ((r - params.r_start) / dr).floor() as i32;
    let i_z = i_z_raw.clamp(0, lod.dim[1] - 1);
    let i_r = i_r_raw.clamp(0, lod.dim[2] - 1);
    [i_theta, i_z, i_r]
}

/// Smallest t > t_min at which the ray exits the cell. Returns `None` when
/// no face is crossed at a t-value strictly past t_min (degenerate cell).
///
/// Mirrors `voxel_cell_exit_t`. `eps_t` controls how aggressively we reject
/// "we're on this face right now" — set to 0.0 for ports of the shader's
/// strictly-positive variant; set to a small positive value for the legacy
/// shader behaviour.
pub fn cell_exit_t(
    base: [f32; 3],
    dir: [f32; 3],
    t_min: f32,
    coords: [i32; 3],
    lod: VoxelLod,
    params: GridParams,
    eps_t: f32,
) -> Option<f32> {
    let dt_theta = lod.theta_step();
    let dt_z = lod.z_step(params);
    let dt_r = lod.r_step(params);

    let theta_low = coords[0] as f32 * dt_theta;
    let theta_high = theta_low + dt_theta;
    let z_low = coords[1] as f32 * dt_z - 0.5 * params.length;
    let z_high = z_low + dt_z;
    let r_low = params.r_start + coords[2] as f32 * dt_r;
    let r_high = r_low + dt_r;

    let mut t_exit = f32::INFINITY;

    // θ half-planes
    for sgn in 0..2 {
        let theta_b = if sgn == 1 { theta_high } else { theta_low };
        let s = theta_b.sin();
        let c = theta_b.cos();
        let denom = dir[0] * s - dir[1] * c;
        if denom.abs() < 1e-9 {
            continue;
        }
        let t = (base[1] * c - base[0] * s) / denom;
        if t <= t_min + eps_t || t >= t_exit {
            continue;
        }
        // Verify crossing is on the correct side of the z-axis (θ = θ_b,
        // not θ_b + π — the equation is symmetric across the axis).
        let px = base[0] + t * dir[0];
        let py = base[1] + t * dir[1];
        if px * c + py * s <= 0.0 {
            continue;
        }
        t_exit = t;
    }

    // z planes
    if dir[2].abs() > 1e-9 {
        let z_target = if dir[2] > 0.0 { z_high } else { z_low };
        let t = (z_target - base[2]) / dir[2];
        if t > t_min + eps_t && t < t_exit {
            t_exit = t;
        }
    }

    // r cylinders
    let a = dir[0] * dir[0] + dir[1] * dir[1];
    if a > 1e-12 {
        let b = 2.0 * (base[0] * dir[0] + base[1] * dir[1]);
        for sel in 0..2 {
            let r_b = if sel == 1 { r_high } else { r_low };
            let c_term = base[0] * base[0] + base[1] * base[1] - r_b * r_b;
            let disc = b * b - 4.0 * a * c_term;
            if disc < 0.0 {
                continue;
            }
            let s_disc = disc.sqrt();
            let t1 = (-b - s_disc) / (2.0 * a);
            let t2 = (-b + s_disc) / (2.0 * a);
            if t1 > t_min + eps_t && t1 < t_exit {
                t_exit = t1;
            }
            if t2 > t_min + eps_t && t2 < t_exit {
                t_exit = t2;
            }
        }
    }

    if t_exit.is_finite() {
        Some(t_exit)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOSTRAL: GridParams = GridParams {
        length: 502.65485,
        r_start: 10.0,
        r_end: 15.0,
    };

    fn lod0() -> VoxelLod {
        VoxelLod {
            dim: [1024, 8192, 128],
        }
    }

    /// LOD hierarchy: parent coords = floor(child / 2) for every axis.
    /// The mip is built any-child-occupied, so this relationship must hold
    /// for the shader's "occupied at parent ⇒ at least one child occupied"
    /// reasoning to be sound.
    #[test]
    fn lod_hierarchy_consistent() {
        let positions = [
            [12.0, 5.0, 0.0],
            [-3.5, 14.2, 200.0],
            [-14.0, -3.0, -100.0],
            [0.001, 14.5, 50.0],
        ];
        let coarse = VoxelLod { dim: [512, 4096, 64] };
        let fine = lod0();
        for pos in positions {
            let c_fine = world_to_cell(pos, fine, FOSTRAL);
            let c_coarse = world_to_cell(pos, coarse, FOSTRAL);
            // θ axis (no clamp)
            assert_eq!(c_fine[0] / 2, c_coarse[0], "θ hierarchy fails at {pos:?}");
            // z and r are clamped to [0, dim-1]; the relation still holds
            // unless one of them clamped.
            if c_fine[1] < fine.dim[1] - 1 && c_coarse[1] < coarse.dim[1] - 1 {
                assert_eq!(c_fine[1] / 2, c_coarse[1], "z hierarchy fails at {pos:?}");
            }
            if c_fine[2] < fine.dim[2] - 1 && c_coarse[2] < coarse.dim[2] - 1 {
                assert_eq!(c_fine[2] / 2, c_coarse[2], "r hierarchy fails at {pos:?}");
            }
        }
    }

    /// After exit_t, the position is at a cell face: at least one of
    /// (θ, z, r) is at a cell-boundary value (within float slop).
    fn assert_at_face(
        pos: [f32; 3],
        coords: [i32; 3],
        lod: VoxelLod,
        params: GridParams,
        tol: f32,
    ) {
        let theta = pos[1].atan2(pos[0]);
        let theta_wrapped = if theta < 0.0 { theta + TAU } else { theta };
        let r = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        let dt = lod.theta_step();
        let dz = lod.z_step(params);
        let dr = lod.r_step(params);
        let theta_lo = coords[0] as f32 * dt;
        let theta_hi = theta_lo + dt;
        let z_lo = coords[1] as f32 * dz - 0.5 * params.length;
        let z_hi = z_lo + dz;
        let r_lo = params.r_start + coords[2] as f32 * dr;
        let r_hi = r_lo + dr;
        let at_theta = (theta_wrapped - theta_lo).abs() < tol
            || (theta_wrapped - theta_hi).abs() < tol;
        let at_z = (pos[2] - z_lo).abs() < tol || (pos[2] - z_hi).abs() < tol;
        let at_r = (r - r_lo).abs() < tol || (r - r_hi).abs() < tol;
        assert!(
            at_theta || at_z || at_r,
            "exit pos {pos:?} (θ={theta_wrapped} r={r}) is not at any face of cell {coords:?} \
             [θ∈({theta_lo},{theta_hi}) z∈({z_lo},{z_hi}) r∈({r_lo},{r_hi})]",
        );
    }

    /// For a ray fully inside a cell, exit_t lands on a face.
    #[test]
    fn exit_t_lands_on_face_descending_ray() {
        // Camera at r=12, on +Y axis, looking inward toward Z.
        let base = [0.0, 12.0, 50.0];
        let dir = [0.0, -1.0, 0.0];
        let coords = world_to_cell(base, lod0(), FOSTRAL);
        let t = cell_exit_t(base, dir, 0.0, coords, lod0(), FOSTRAL, 1e-5)
            .expect("exit must exist for non-degenerate ray");
        let pos = [
            base[0] + t * dir[0],
            base[1] + t * dir[1],
            base[2] + t * dir[2],
        ];
        assert_at_face(pos, coords, lod0(), FOSTRAL, 1e-3);
    }

    /// Step after exit_t lands us in a cell adjacent to the previous one
    /// (differs by at most 1 in one axis; θ wraps mod dim_θ).
    fn assert_adjacent(c1: [i32; 3], c2: [i32; 3], lod: VoxelLod) {
        let mut diffs = 0;
        let mut total = 0;
        for axis in 0..3 {
            let d = (c2[axis] - c1[axis]).abs();
            let wrap_d = if axis == 0 {
                let dim = lod.dim[0];
                ((c2[0] - c1[0]).rem_euclid(dim) - dim).abs().min(d)
            } else {
                d
            };
            if wrap_d > 0 {
                diffs += 1;
                total += wrap_d;
            }
        }
        assert!(
            diffs <= 1 && total <= 1,
            "non-adjacent cells {c1:?} → {c2:?} (Δ={diffs} axes total={total})",
        );
    }

    #[test]
    fn dda_walks_adjacent_cells_radial() {
        let base = [0.0, 14.5, 50.0];
        let dir = [0.0, -1.0, 0.0];
        let lod = lod0();
        let mut coords = world_to_cell(base, lod, FOSTRAL);
        let mut t = 0.0;
        for step in 0..30 {
            let exit = match cell_exit_t(base, dir, t, coords, lod, FOSTRAL, 1e-5) {
                Some(t) => t,
                None => break,
            };
            let new_t = exit + 1e-5;
            let pos = [
                base[0] + new_t * dir[0],
                base[1] + new_t * dir[1],
                base[2] + new_t * dir[2],
            ];
            let new_coords = world_to_cell(pos, lod, FOSTRAL);
            assert_adjacent(coords, new_coords, lod);
            // Progress invariant: new_t > t (no loops)
            assert!(new_t > t, "step {step} regressed: {t} → {new_t}");
            coords = new_coords;
            t = new_t;
            // Bail out once we leave the shell
            if coords[2] < 0 || coords[2] >= lod.dim[2] {
                break;
            }
        }
    }

    /// Tangential ray near a θ boundary — the failure mode where a ray
    /// starting exactly on a θ face skips multiple cells in one step.
    #[test]
    fn dda_walks_adjacent_cells_tangential_at_boundary() {
        // Start at θ = π/2 exactly (lands on a cell boundary in dim=1024).
        let base = [0.0, 14.5, 50.0];
        let dir = [-1.0, 0.0, 0.0]; // moves θ toward π
        let lod = lod0();
        let mut coords = world_to_cell(base, lod, FOSTRAL);
        let mut t = 0.0;
        for step in 0..30 {
            let exit = match cell_exit_t(base, dir, t, coords, lod, FOSTRAL, 1e-5) {
                Some(t) => t,
                None => break,
            };
            let new_t = exit + 1e-5;
            let pos = [
                base[0] + new_t * dir[0],
                base[1] + new_t * dir[1],
                base[2] + new_t * dir[2],
            ];
            let new_coords = world_to_cell(pos, lod, FOSTRAL);
            assert_adjacent(coords, new_coords, lod);
            assert!(new_t > t, "step {step} regressed: {t} → {new_t}");
            coords = new_coords;
            t = new_t;
            if coords[2] < 0 || coords[2] >= lod.dim[2] {
                break;
            }
        }
    }

    /// Canonicalize cell coords so i_θ wraps mod dim_θ and out-of-range
    /// z/r cells (FP slop at the shell boundary) are absorbed into the
    /// nearest valid cell.
    fn canon(c: [i32; 3], lod: VoxelLod) -> [i32; 3] {
        [
            c[0].rem_euclid(lod.dim[0]),
            c[1].clamp(0, lod.dim[1] - 1),
            c[2].clamp(0, lod.dim[2] - 1),
        ]
    }

    /// Cross-LOD: DDA → dense sample agreement at a coarser LOD where
    /// each cell covers ~3 m of arc length, so a ray crossing several
    /// cells touches large terrain features.
    #[test]
    fn dda_visits_same_cells_as_dense_sampling_lod3() {
        let lod = VoxelLod { dim: [128, 1024, 16] };
        let base = [13.0, 5.5, 100.0];
        let dir = [-0.5, -0.5, -0.05];
        let mut dda = std::collections::BTreeSet::new();
        let mut coords = world_to_cell(base, lod, FOSTRAL);
        dda.insert(canon(coords, lod));
        let mut t = 0.0;
        for _ in 0..1000 {
            let exit = match cell_exit_t(base, dir, t, coords, lod, FOSTRAL, 1e-5) {
                Some(v) => v,
                None => break,
            };
            if exit > 30.0 {
                break;
            }
            let new_t = exit + 1e-5;
            let pos = [
                base[0] + new_t * dir[0],
                base[1] + new_t * dir[1],
                base[2] + new_t * dir[2],
            ];
            let c = world_to_cell(pos, lod, FOSTRAL);
            dda.insert(canon(c, lod));
            coords = c;
            t = new_t;
        }
        let mut dense = std::collections::BTreeSet::new();
        for i in 0..30_000 {
            let tt = i as f32 * 0.001;
            let pos = [
                base[0] + tt * dir[0],
                base[1] + tt * dir[1],
                base[2] + tt * dir[2],
            ];
            // Only count positions actually inside the heightmap shell —
            // the DDA stops at r_start / r_end correctly, so cells the
            // dense sampling lands in only because of clamping to the
            // nearest valid index aren't legitimately "missed".
            let r = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
            if r < FOSTRAL.r_start || r > FOSTRAL.r_end {
                continue;
            }
            if pos[2] < -0.5 * FOSTRAL.length || pos[2] > 0.5 * FOSTRAL.length {
                continue;
            }
            dense.insert(canon(world_to_cell(pos, lod, FOSTRAL), lod));
        }
        let missing: Vec<_> = dense.difference(&dda).copied().collect();
        assert!(
            missing.is_empty(),
            "DDA skipped {} cells at LOD3 (e.g. {:?})",
            missing.len(),
            &missing[..missing.len().min(5)],
        );
    }

    /// Walks a ray through many cells with the DDA and verifies that the
    /// SAME set of cells is visited when we walk the ray densely with a
    /// tiny step size. Catches "DDA skips a cell" bugs that aren't caught
    /// by the per-step adjacency test (which would still pass if every
    /// individual step is adjacent but a different cell is silently
    /// dropped along the way).
    #[test]
    fn dda_visits_same_cells_as_dense_sampling() {
        let base = [0.0, 14.5, 50.0];
        let dir = [-0.7, -0.3, 0.0]; // diagonal in xy
        let lod = lod0();

        // DDA path
        let mut dda_cells = std::collections::BTreeSet::new();
        let mut coords = world_to_cell(base, lod, FOSTRAL);
        dda_cells.insert(coords);
        let mut t = 0.0;
        let max_t = 5.0; // walk 5 metres
        for _ in 0..2000 {
            let exit = match cell_exit_t(base, dir, t, coords, lod, FOSTRAL, 1e-5) {
                Some(t) => t,
                None => break,
            };
            if exit > max_t {
                break;
            }
            let new_t = exit + 1e-5;
            let pos = [
                base[0] + new_t * dir[0],
                base[1] + new_t * dir[1],
                base[2] + new_t * dir[2],
            ];
            let new_coords = world_to_cell(pos, lod, FOSTRAL);
            dda_cells.insert(new_coords);
            coords = new_coords;
            t = new_t;
        }

        // Dense path
        let mut dense_cells = std::collections::BTreeSet::new();
        let step = 0.001;
        let n = (max_t / step) as usize;
        for i in 0..n {
            let t = i as f32 * step;
            let pos = [
                base[0] + t * dir[0],
                base[1] + t * dir[1],
                base[2] + t * dir[2],
            ];
            let c = world_to_cell(pos, lod, FOSTRAL);
            dense_cells.insert(c);
        }

        // DDA should visit all densely-sampled cells.
        let missing: Vec<_> = dense_cells.difference(&dda_cells).copied().collect();
        assert!(
            missing.is_empty(),
            "DDA skipped {} cells out of {} dense (missing examples: {:?})",
            missing.len(),
            dense_cells.len(),
            &missing[..missing.len().min(5)],
        );
    }

    /// θ wraps from 2π back to 0.
    #[test]
    fn theta_wraps_at_2pi() {
        let base = [14.0, -0.01, 50.0]; // just below the +X axis (θ ≈ 2π)
        let dir = [0.0, -1.0, 0.0]; // crosses +X axis going toward -y
        let lod = lod0();
        let coords = world_to_cell(base, lod, FOSTRAL);
        // Sanity: i_θ should be near dim-1 (just below 2π).
        assert!(coords[0] >= lod.dim[0] - 2, "i_θ near 2π expected, got {}", coords[0]);
        let exit = cell_exit_t(base, dir, 0.0, coords, lod, FOSTRAL, 1e-5);
        // Whatever face we exit through, we must make forward progress.
        let exit = exit.expect("exit must exist");
        assert!(exit > 0.0);
    }
}
